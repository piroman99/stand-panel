#!/usr/bin/env python3
"""Микро-UI для картинок на стенде: браузер -> ComfyUI, без графов.

Запуск: systemd-юнит stand-ui (порт 8288) или вручную:
    python3 stand-ui.py --port 8288

Слева поле текста (можно по-русски), выбор модели и размера, справа картинка
и лента последних генераций. Переводит локальная qwen3:8b на ollama, считает
ComfyUI. Прогресс по шагам отдаёт его же websocket, если браузер его видит.
"""
import argparse
import base64
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import psutil
except ImportError:  # на всякий случай: без psutil панель работает, вкладка «железо» — нет
    psutil = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from stand_guard import check_prompt as guard_check
except Exception as _e:  # нет модуля — отказываем, а не пропускаем молча
    _GUARD_ERR = str(_e)

    def guard_check(text, extra=''):
        return False, 'фильтр промптов не загружен (%s) — рисовать не буду' % _GUARD_ERR

# Все пути ниже — только умолчания. На своём стенде их переопределяют переменными
# окружения (см. panel.env.example), лезть в код не надо.
PANEL_HOME = os.environ.get('PANEL_HOME') or os.path.expanduser('~')
COMFY_HOME = os.environ.get('COMFY_HOME') or os.path.expanduser('~/ComfyUI')
API = os.environ.get('COMFY_API', 'http://127.0.0.1:8188')
OLLAMA = os.environ.get('OLLAMA_API', 'http://127.0.0.1:11434')
OUT_DIR = os.environ.get('COMFY_OUT', os.path.join(COMFY_HOME, 'output'))
INPUT_DIR = os.environ.get('COMFY_INPUT', os.path.join(COMFY_HOME, 'input'))
# Стилевые LoRA лежат в каталоге моделей ComfyUI; список читается с диска
LORAS_DIR = os.environ.get('COMFY_LORAS_DIR', os.path.join(COMFY_HOME, 'models', 'loras'))
# Подписи и подсказки к своим LoRA (остальные — просто именем файла)
LORA_LABELS = {}
HIST_FILE = os.environ.get('UI_HISTORY', os.path.join(PANEL_HOME, '.ui-history.json'))
MAX_HISTORY = 60

# Обучение LoRA: обёртка на стенде и рабочие каталоги
TRAINER_DIR = os.environ.get('TRAINER_DIR', os.path.join(PANEL_HOME, 'trainer'))
TRAIN_PHOTOS = os.path.join(TRAINER_DIR, 'photos')
TRAIN_RUNS = os.path.join(TRAINER_DIR, 'runs')
TRAIN_SCRIPT = os.path.join(TRAINER_DIR, 'lora-train.sh')
TRAIN_STATE = os.path.join(TRAINER_DIR, '.ui-train.json')
TRAIN_IMAGE = 'comfyui-trainer:latest'
PHOTO_EXT = ('.png', '.jpg', '.jpeg', '.webp')

# Нейросетевая обработка «плохого» видео (SCUNet + Real-ESRGAN + RIFE на CUDA)
VIDEO_DIR = os.environ.get('VIDEO_DIR', os.path.join(PANEL_HOME, 'video-enh'))
VIDEO_IN = os.path.join(VIDEO_DIR, 'in')
VIDEO_OUT = os.path.join(VIDEO_DIR, 'out')
VIDEO_LOG_DIR = os.path.join(VIDEO_DIR, 'logs')
VIDEO_IMAGE = os.environ.get('VIDEO_IMAGE', 'video-enh:latest')
VIDEO_STATE = os.environ.get('VIDEO_STATE', os.path.join(PANEL_HOME, '.ui-video.json'))
VIDEO_UNIT = os.environ.get('VIDEO_UNIT', 'standui-video')
# Имя контейнера: без него остановка не работает — systemctl убивает только
# клиент docker, а сам контейнер продолжает считать на видеокарте.
VIDEO_CONTAINER = os.environ.get('VIDEO_CONTAINER', 'standui-video-job')
VIDEO_EXT = ('.mp4', '.mkv', '.mov', '.avi', '.webm', '.m4v')
# разбивка ролика на куски по качеству картинки
VIDEO_SPLIT_STATE = os.environ.get('VIDEO_SPLIT_STATE', os.path.join(PANEL_HOME, '.ui-split.json'))
VIDEO_SPLIT_OUT = os.environ.get('VIDEO_SPLIT_OUT', os.path.join(PANEL_HOME, '.ui-split-out.json'))
VIDEO_SPLIT_LOG = os.environ.get('VIDEO_SPLIT_LOG', os.path.join(PANEL_HOME, '.ui-split.log'))
VIDEO_SPLIT_PLAN = os.environ.get('VIDEO_SPLIT_PLAN', os.path.join(PANEL_HOME, '.ui-split-plan.json'))
VIDEO_SPLIT_UNIT = 'standui-split'
VIDEO_SPLIT_SCRIPT = os.environ.get('VIDEO_SPLIT_SCRIPT', os.path.join(PANEL_HOME, 'vid-split.py'))
VIDEO_SPLIT_RUN = os.environ.get('VIDEO_SPLIT_RUN', os.path.join(PANEL_HOME, 'vsplit-run.py'))
PHOTO_IN_EXT = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')
VIDEO_LOCK = threading.Lock()
_VDUR_CACHE = {}
TRANSLATOR = os.environ.get('UI_TRANSLATOR', 'qwen3:8b')
CHAT_TIMEOUT = int(os.environ.get('UI_CHAT_TIMEOUT', '600'))
# Окно контекста чата. По умолчанию ollama даёт всего 4k, и длинный ответ
# упирается в него — отсюда «зависает». 16384 — рабочий компромисс для 9B:
# KV-кэш растёт линейно, каждый шаг вдвое удваивает его размер.
CHAT_CTX = int(os.environ.get('UI_CHAT_CTX', '16384'))
CHAT_CTX_MAX = int(os.environ.get('UI_CHAT_CTX_MAX', '65536'))
CHAT_CTX_CHOICES = [4096, 8192, 16384, 32768, 65536]
# Потолок на длину ответа. Без него залипшая модель молотит до конца окна:
# 30 тыс. токенов одного и того же абзаца и минуты ожидания.
CHAT_MAX_TOKENS = int(os.environ.get('UI_CHAT_MAX_TOKENS', '3000'))
# Сэмплеры против вырожденных повторов. У ollama умолчания слабые (repeat_penalty
# 1.1, repeat_last_n 64), и на длинном тексте модель уходит в цикл.
CHAT_TEMP = float(os.environ.get('UI_CHAT_TEMP', '0.7'))
CHAT_REPEAT_PENALTY = float(os.environ.get('UI_CHAT_REPEAT_PENALTY', '1.12'))
CHAT_REPEAT_LAST_N = int(os.environ.get('UI_CHAT_REPEAT_LAST_N', '512'))
CHAT_PRESENCE_PENALTY = float(os.environ.get('UI_CHAT_PRESENCE_PENALTY', '0.2'))
CHAT_FREQUENCY_PENALTY = float(os.environ.get('UI_CHAT_FREQUENCY_PENALTY', '0'))
CHAT_TOP_P = float(os.environ.get('UI_CHAT_TOP_P', '0.92'))
CHAT_MIN_P = float(os.environ.get('UI_CHAT_MIN_P', '0'))
# Сторож залипания: один и тот же шингл из N слов, встретившийся LIMIT раз, — цикл.
CHAT_LOOP_N = int(os.environ.get('UI_CHAT_LOOP_N', '10'))
CHAT_LOOP_LIMIT = int(os.environ.get('UI_CHAT_LOOP_LIMIT', '4'))

# Рисование прямо из чата: если в вопросе просят картинку, чат не зовёт модель
# словами, а ставит задачу в ту же очередь ComfyUI и рисует её.
CHAT_DRAW_MODEL = os.environ.get('UI_CHAT_DRAW_MODEL', 'sdxl')
CHAT_DRAW_SIZE = os.environ.get('UI_CHAT_DRAW_SIZE', '1024x1024')
DRAW_VERB = re.compile(r'\b(?:нарисуй(?:те)?|нарисуешь|рисуй(?:те)?|draw|paint|sketch|'
                       r'сгенерируй(?:те)?|сгенерируешь|generate)\b', re.I)
DRAW_FILLER = re.compile(r'^\s*(?:мне|нам|пожалуйста|плиз|картинку|картинка|изображение|'
                         r'рисунок|фото|image|picture|me|of|a|an|the)\b[\s,:;\-—–]*', re.I)

NEG_DEFAULT = ('blurry, low quality, worst quality, watermark, text, signature, '
               'deformed, extra limbs')
NEG_VIDEO = ('blurry, low quality, worst quality, watermark, text, subtitles, '
             'still image, deformed, extra limbs')

SIZES = [
    ('512x512', '512x512 (мало, быстро)'),
    ('640x480', '640x480 (черновик)'),
    ('768x768', '768x768 (квадрат)'),
    ('896x672', '896x672 (альбом)'),
    ('1024x1024', '1024x1024 (квадрат, обычный)'),
    ('832x1216', '832x1216 (портрет)'),
    ('1216x832', '1216x832 (альбом)'),
    ('1024x1536', '1024x1536 (макс. портрет)'),
]

# Наши модели: подпись -> файл в ComfyUI + настройки по умолчанию
# Файлы, которые этот граф (SDXL) не умеет вести: Flux и Z-Image грузятся другими нодами.
UNSUPPORTED = ('flux', 'z_image', 'z-image')

MODELS = [
    ('sdxl', 'SDXL base 1.0 (акварель через LoRA)', 'sd_xl_base_1.0.safetensors', 28, 7.0),
]

# Приставки к промпту и негативу по модели (пусто = модель без приставок)
PROMPT_PREFIX = {}
NEG_PREFIX = {}


def model_key(ckpt):
    """Ключ модели по имени файла (для приставок и настроек)."""
    for key, _label, fname, _s, _c in MODELS:
        if fname == ckpt:
            return key
    return None



def with_prefix(text, ckpt, negative=False):
    """Добавить приставку модели, если её ещё нет в тексте."""
    key = model_key(ckpt)
    table = NEG_PREFIX if negative else PROMPT_PREFIX
    head = table.get(key or '', '')
    if head and head.split(',')[0] not in (text or ''):
        return head + (text or '')
    return text

# Видео: LTXV 2B distilled (быстро) и Wan 2.2 ti2v-5B (лучше движение)
VIDEO_MODELS = [
    ('ltxv', 'LTXV 2B (видео, быстро)', 'ltxv-2b-0.9.8-distilled-fp8.safetensors', 8, 1.0,
     'ltxv'),
    ('wan22', 'Wan 2.2 5B (видео, лучше движение)', 'wan2.2_ti2v_5B_fp16.safetensors', 30, 5.0,
     'wan'),
    ('wan22flf', 'Wan 2.2 5B (первый и последний кадр)',
     'wan2.2_ti2v_5B_fp16.safetensors', 30, 5.0, 'wanflf'),
]
VIDEO_T5 = 't5xxl_fp8_e4m3fn.safetensors'
WAN_CLIP = 'umt5_xxl_fp8_e4m3fn_scaled.safetensors'
WAN_VAE = 'wan2.2_vae.safetensors'

# Настройки по семейству: у каждой модели свой шаг, cfg и набор размеров
VIDEO_FAMILY = {
    'ltxv': {'size': '768x512', 'length': 49, 'fps': 24, 'steps': 8, 'cfg': 1.0,
             'max_area': 1024 * 576, 'max_side': 1024, 'max_frames': 161, 'step': 8},
    'wan': {'size': '704x480', 'length': 49, 'fps': 24, 'steps': 30, 'cfg': 5.0,
            'max_area': 832 * 480, 'max_side': 1024, 'max_frames': 121, 'step': 4},
    'wanflf': {'size': '832x480', 'length': 49, 'fps': 24, 'steps': 30, 'cfg': 5.0,
               'max_area': 832 * 480, 'max_side': 1024, 'max_frames': 121, 'step': 4},
}
VIDEO_DEFAULTS = VIDEO_FAMILY['ltxv']

SIZES_VIDEO = {
    'ltxv': [
        ('512x320', '512x320 (черновик, быстро)'),
        ('768x512', '768x512 (по умолчанию)'),
        ('1024x576', '1024x576 (максимум, альбом)'),
        ('512x768', '512x768 (портрет)'),
        ('576x1024', '576x1024 (портрет, максимум)'),
    ],
    'wan': [
        ('640x448', '640x448 (быстро)'),
        ('704x480', '704x480 (по умолчанию)'),
        ('832x480', '832x480 (максимум, альбом)'),
        ('480x704', '480x704 (портрет)'),
    ],
    'wanflf': [
        ('704x480', '704x480 (быстро)'),
        ('832x480', '832x480 (по умолчанию)'),
        ('480x704', '480x704 (портрет)'),
    ],
}

IMG_HINT = ('1024x1024 считается ~20 с. Перевод русского текста занимает 5-80 с '
            '(дольше, если картинка уже заняла видеопамять), поэтому с русским '
            'текстом бывает ~1,5 минуты; черновик 640x480 ~10 с.')
VIDEO_HINT = {
    'ltxv': ('LTXV: 49 кадров = 2 с при 24 fps. Черновик 512x320/33 кадра — 4 с, '
             '768x512/49 кадров — 7 с, 768x512/73 кадра — 20 с с перезагрузкой модели. '
             'Кадров 8k+1 (33, 49, 65, 97, 121), стороны кратны 32.'),
    'wan': ('Wan 2.2 5B: 49 кадров = 2 с при 24 fps, по умолчанию 30 шагов. Замер: '
            '704x480/49 кадров/30 шагов — 144 с. Движение живее, чем у LTXV, плата — '
            'время. Кадров 4k+1 (49, 97, 121), стороны кратны 32.'),
    'wanflf': ('Первый и последний кадр: два кадра задают начало и конец, Wan сшивает '
               'переход. Замер 832x480/49 кадров/30 шагов — 186 с. Самая чистая схема: '
               'картинка не мылится к концу. Кадры можно загрузить или нарисовать '
               'здесь же — пустое поле = нарисовать по тексту. Чем сильнее различаются '
               'кадры, тем резче движение. Кадров 4k+1 (49, 97, 121), стороны кратны 32.'),
}
VIDEO_HINT_DEFAULT = VIDEO_HINT['ltxv']

TRANSLATE_PROMPT = (
    'You are a prompt writer for the SDXL image model. The user writes in Russian.\n'
    'Turn it into one English prompt of 20-40 words: keep every subject and detail, '
    'add a natural photographic or artistic style only if the user did not ask for one.\n'
    'Answer with the prompt only, no quotes, no comments, no explanations.\n\n'
    'User: '
)

# Для видео просим описать именно движение: иначе модель рисует почти статичный кадр
TRANSLATE_PROMPT_VIDEO = (
    'You are a prompt writer for a text-to-video model. The user writes in Russian.\n'
    'Turn it into one English prompt of 25-45 words describing ONE continuous shot: '
    'who or what moves, in which direction and how (walking, turning, waving, camera '
    'push-in), plus setting and light. Keep every subject and detail, make the motion '
    'explicit and continuous, no cuts, no text overlays.\n'
    'Answer with the prompt only, no quotes, no comments, no explanations.\n\n'
    'User: '
)


def has_cyr(text):
    return bool(re.search('[а-яА-ЯёЁ]', text))


def get_json(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def post_json(url, payload, timeout=120):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, body, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class LoopGuard:
    """Ловит вырожденное залипание по ходу генерации.

    Считаем, сколько раз встретился каждый шингл из n слов. Четвёртое
    повторение — это уже не стиль, а цикл: дальше модель будет молотить
    один и тот же абзац до конца окна контекста.
    """

    def __init__(self, n=10, limit=4):
        self.n = max(4, int(n))
        self.limit = max(2, int(limit))
        self.words = []
        self.count = {}
        self.at = 0
        self.hit = ''

    def feed(self, chunk):
        found = re.findall(r'\S+', (chunk or '').lower())
        if not found:
            return False
        self.words.extend(found)
        n = self.n
        while self.at <= len(self.words) - n:
            key = ' '.join(self.words[self.at:self.at + n])
            c = self.count.get(key, 0) + 1
            self.count[key] = c
            self.at += 1
            if c >= self.limit:
                self.hit = key[:120]
                return True
        return False


def ollama_chat_stream(model, messages, options, think, timeout, guard=None):
    """Чат потоком: нужен, чтобы оборвать залипание, не дожидаясь конца.

    Возвращает (текст, размышление, токенов, причина остановки). При
    причине 'loop' соединение закрывается, и ollama прекращает генерацию.
    """
    payload = {'model': model, 'stream': True, 'messages': messages,
               'think': bool(think), 'options': options}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url=OLLAMA + '/api/chat', data=body,
                                headers={'Content-Type': 'application/json'})
    parts, thoughts, count, reason = [], [], 0, ''
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            msg = d.get('message') or {}
            piece = msg.get('content') or ''
            if piece:
                parts.append(piece)
                if guard is not None and guard.feed(piece):
                    reason = 'loop'
                    break
            thought = msg.get('thinking') or ''
            if thought:
                thoughts.append(thought)
            if d.get('done'):
                count = int(d.get('eval_count') or 0)
                reason = d.get('done_reason') or reason
                break
    return ''.join(parts), ''.join(thoughts), count, reason


def all_checkpoints():
    """Весь список чекпойнтов ComfyUI, без фильтров."""
    try:
        info = get_json(API + '/object_info/CheckpointLoaderSimple', timeout=20)
        return list(info['CheckpointLoaderSimple']['input']['required']['ckpt_name'][0])
    except Exception:
        return [m[2] for m in MODELS] + [m[2] for m in VIDEO_MODELS]


def checkpoints():
    """Список файлов моделей из ComfyUI; видео-модели отбрасываем."""
    keep = []
    for n in all_checkpoints():
        if re.search(r'ltxv|video|sovits', n, re.I):
            continue
        keep.append(n)
    return keep or [m[2] for m in MODELS]


def model_list():
    """Наши модели + всё незнакомое, что реально лежит в ComfyUI."""
    have = checkpoints()
    out = []
    known = set()
    for key, label, fname, steps, cfg in MODELS:
        if fname in have:
            out.append({'key': key, 'label': label, 'file': fname, 'kind': 'image',
                        'steps': steps, 'cfg': cfg})
            known.add(fname)
    for fname in have:
        if fname in known:
            continue
        low = fname.lower()
        if any(x in low for x in UNSUPPORTED):   # другой загрузчик, не SDXL
            continue
        out.append({'key': fname, 'label': fname.replace('.safetensors', ''),
                    'file': fname, 'kind': 'image', 'steps': 28, 'cfg': 7.0})
    return out


def all_unets():
    """Список UNET-файлов ComfyUI. Wan лежит в models/diffusion_models, а не в checkpoints."""
    try:
        info = get_json(API + '/object_info/UNETLoader', timeout=20)
        return list(info['UNETLoader']['input']['required']['unet_name'][0])
    except Exception:
        return [m[2] for m in VIDEO_MODELS if m[5] == 'wan']


def video_model_list():
    """Видео-модели: только те, что реально есть в ComfyUI (Wan требует ещё VAE и энкодер)."""
    have = all_checkpoints()
    unets = None
    extra = None
    out = []
    for key, label, fname, steps, cfg, family in VIDEO_MODELS:
        if family in ('wan', 'wanflf'):
            # Wan грузится через UNETLoader: файл ищем в diffusion_models, не в checkpoints
            if unets is None:
                unets = all_unets()
            if fname not in unets:
                continue
            if extra is None:
                try:
                    vae = get_json(API + '/object_info/VAELoader', timeout=20)
                    clip = get_json(API + '/object_info/CLIPLoader', timeout=20)
                    extra = (vae['VAELoader']['input']['required']['vae_name'][0],
                             clip['CLIPLoader']['input']['required']['clip_name'][0])
                except Exception:
                    extra = ([], [])
            if WAN_VAE not in extra[0] or WAN_CLIP not in extra[1]:
                continue
        elif fname not in have:
            continue
        out.append({'key': key, 'label': label, 'file': fname, 'kind': 'video',
                    'family': family, 'steps': steps, 'cfg': cfg})
    return out


def all_models():
    return model_list() + video_model_list()


def free_comfy():
    try:
        post_json(API + '/free', {'unload_models': True, 'free_memory': True}, timeout=30)
    except Exception:
        pass


# Контейнеры стенда раз в несколько минут теряют видеокарту: новые процессы внутри
# уже не видят GPU (NVML: Unknown Error), а сам ComfyUI падает с «OutOfMemoryError»
# при пустой видеопамяти. Надёжное лечение — пересоздать контейнер. Чтобы не ждать
# таймера, панель перед генерацией сама это проверяет и лечит.
COMPOSE_DIR = os.environ.get('UI_COMPOSE_DIR') or COMFY_HOME

def gpu_ok(container):
    """Видит ли контейнер видеокарту: проба новым процессом внутри него."""
    try:
        r = subprocess.run(['docker', 'exec', container, 'nvidia-smi', '-L'],
                           capture_output=True, text=True, timeout=60)
    except Exception:
        return False
    return r.returncode == 0 and 'GPU 0:' in (r.stdout or '')


def heal_gpu(container, job=None):
    """Пересоздать контейнер, потерявший видеокарту. -> None или текст ошибки."""
    if job is not None:
        job['phase'] = 'пересоздаю %s (потерял видеокарту)' % container
    cmd = ['docker', 'compose', 'up', '-d', '--force-recreate', container]
    try:
        if container == 'comfyui':
            kill_comfy()
        r = subprocess.run(cmd, cwd=COMPOSE_DIR, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            return ('не смог пересоздать %s: %s'
                    % (container, (r.stderr or r.stdout or '').strip()[:300]))
    except Exception as e:
        return 'не смог пересоздать %s: %s' % (container, e)
    deadline = time.time() + (120 if container == 'comfyui' else 60)
    while time.time() < deadline:
        time.sleep(3)
        if gpu_ok(container):
            if container == 'comfyui':
                wait_comfy_ready()
            return None
    return '%s так и не увидел видеокарту после пересоздания' % container


def kill_comfy():
    """Гасим старый ComfyUI перед пересозданием (жёстко: он держит мёртвый контекст)."""
    try:
        subprocess.run(['docker', 'kill', 'comfyui'], capture_output=True, text=True,
                       timeout=60)
    except Exception:
        pass


def wait_comfy_ready(wait=120):
    deadline = time.time() + wait
    while time.time() < deadline:
        try:
            get_json(API + '/system_stats', timeout=10)
            time.sleep(2)
            return True
        except Exception:
            time.sleep(3)
    return False


COMFY_IN_CONTAINER = os.environ.get('UI_COMFY_IN_CONTAINER', '1') != '0'


def ensure_comfy(job=None):
    """Есть ли живой ComfyUI. В контейнере — через docker, у нас — просто процесс."""
    if not COMFY_IN_CONTAINER:
        try:
            get_json(API + '/system_stats', timeout=10)
            return None
        except Exception:
            return ('ComfyUI не отвечает на %s — запусти его '
                    '(~/opt/ComfyUI/main.py)' % API)
    if gpu_ok('comfyui'):
        return None
    return heal_gpu('comfyui', job)


def ensure_ollama(job=None):
    """ollama теряет карту и молча считает на CPU — перезапуск дешевле пересоздания."""
    if not COMFY_IN_CONTAINER:      # у нас ollama — обычный юнит, docker нет
        try:
            get_json(OLLAMA + '/api/tags', timeout=10)
            return None
        except Exception:
            return 'ollama не отвечает на %s' % OLLAMA
    if gpu_ok('ollama'):
        return None
    if job is not None:
        job['phase'] = 'перезапускаю ollama (потеряла видеокарту)'
    try:
        subprocess.run(['docker', 'restart', 'ollama'], capture_output=True, text=True,
                       timeout=180)
    except Exception:
        pass
    for _ in range(20):
        time.sleep(3)
        if gpu_ok('ollama'):
            return None
    return heal_gpu('ollama', job)


def draw_prompt(text):
    """Текст для картинки, если в вопросе просят «нарисуй …», иначе None."""
    t = (text or '').strip()
    if not t or len(t) > 600:
        return None
    m = DRAW_VERB.search(t)
    if not m:
        return None
    rest = t[m.end():]
    while True:
        nxt = DRAW_FILLER.sub('', rest, count=1)
        if nxt == rest:
            break
        rest = nxt
    rest = rest.strip(' \t:;,-—–"«»')
    if len(rest) < 3:
        return None
    return rest[:600]


def queue_image(prompt, model_key=None, size=None, steps=None, translate=True,
                keep_ollama=False, negative=None):
    """Поставить картинку в очередь ComfyUI. -> (id задачи, текст ошибки)."""
    models = [m for m in model_list() if m.get('kind') == 'image']
    if not models:
        return None, 'ComfyUI не отдал ни одной модели картинок'
    model = next((m for m in models if m['key'] == (model_key or CHAT_DRAW_MODEL)), models[0])
    size = size or CHAT_DRAW_SIZE
    m = re.fullmatch(r'(\d+)x(\d+)', size)
    if not m:
        return None, 'размер писать как 1024x1024'
    w, h = int(m.group(1)), int(m.group(2))
    if w % 8 or h % 8 or w * h > 1024 * 1536 or max(w, h) > 1536:
        return None, 'размер вне 8-кратных пределов 1024x1536'
    ok, why = guard_check(prompt)
    if not ok:
        return None, why
    steps = max(1, min(80, int(steps or model['steps'])))
    jid = uuid.uuid4().hex[:12]
    job = {'id': jid, 'state': 'queued', 'phase': 'в очереди', 'created': time.time(),
           'req': {'prompt': prompt, 'w': w, 'h': h, 'steps': steps,
                   'cfg': model['cfg'], 'file': model['file'], 'label': model['label'],
                   'kind': 'image', 'family': None, 'length': None, 'fps': None,
                   'seed': None, 'negative': negative or NEG_DEFAULT,
                   'start_image': None, 'end_image': None, 'end_prompt': None,
                   'frame_model': None, 'translate': bool(translate),
                   'translator': TRANSLATOR, 'keep_ollama': bool(keep_ollama)}}
    with JOBS_LOCK:
        JOBS[jid] = job
        if len(JOBS) > 200:
            for k in [k for k, v in JOBS.items() if v.get('state') in ('done', 'error')][:100]:
                JOBS.pop(k, None)
    JOBQ.put(job)
    return jid, None


def safe_input(name):
    """Имя загруженного кадра, если файл действительно лежит в input ComfyUI."""
    if not name:
        return None
    base = os.path.basename(str(name))
    if not re.fullmatch(r'[\w.\-]{1,120}', base):
        return None
    if not os.path.isfile(os.path.join(INPUT_DIR, base)):
        return None
    return base


def translate(text, model=None, video=False):
    model = model or TRANSLATOR
    head = TRANSLATE_PROMPT_VIDEO if video else TRANSLATE_PROMPT
    try:
        r = post_json(OLLAMA + '/api/generate',
                      {'model': model, 'prompt': head + text,
                       'stream': False, 'think': False,
                       'options': {'temperature': 0.4, 'num_ctx': 4096}},
                      timeout=300)
        out = (r.get('response') or '').strip().strip('"').strip()
    except Exception as e:
        return None, 'перевод не удался (%s)' % e
    out = re.sub(r'^(prompt|Prompt)\s*:\s*', '', out)
    if not out or has_cyr(out):
        return None, 'перевод пустой или с кириллицей'
    return out, None


def ollama_models():
    """Модели ollama, годные для перевода (без эмбеддингов)."""
    try:
        tags = get_json(OLLAMA + '/api/tags', timeout=20).get('models', [])
    except Exception:
        return [{'name': TRANSLATOR, 'label': TRANSLATOR}]
    out = []
    for m in tags:
        name = m.get('name') or ''
        if not name or re.search(r'embed|bge|nomic|e5-|clip', name, re.I):
            continue
        label = re.sub(r':latest$', '', name)
        # пометка «без цензуры» — по любому из слов в имени
        if re.search(r'abliterated|uncensored|decensored|heretic', name, re.I):
            label += ' (без цензуры)'
        out.append({'name': name, 'label': label})
    return out or [{'name': TRANSLATOR, 'label': TRANSLATOR}]


# Распознавание картинки: только модели с возможностью vision (список спрашиваем у ollama)
VISION_DEFAULT = os.environ.get('UI_VISION_MODEL', 'qwen3-vl:8b')
VISION_TIMEOUT = int(os.environ.get('UI_VISION_TIMEOUT', '600'))
VISION_MAXSIDE = int(os.environ.get('UI_VISION_MAXSIDE', '1024'))
_VISION_CACHE = {'t': 0.0, 'models': []}

VISION_PROMPTS = {
    'ru': ('Опиши, что на картинке, по-русски — так, чтобы по описанию можно было нарисовать '
           'похожий кадр. Одним абзацем, через запятую: кто или что в кадре, внешность, одежда, '
           'поза, обстановка, время суток и свет, ракурс, стиль. Без вступлений и без выводов.'),
    'en': ('Describe this image as a prompt for a Stable Diffusion model. Answer with English '
           'comma-separated keywords only: subject, appearance, clothing, pose, setting, '
           'time of day, lighting, camera angle, style. No preamble, no explanations.'),
}


def vision_models():
    """Модели ollama, которые умеют смотреть картинки (capabilities: vision)."""
    now = time.time()
    if _VISION_CACHE['models'] and now - _VISION_CACHE['t'] < 120:
        return _VISION_CACHE['models']
    try:
        tags = get_json(OLLAMA + '/api/tags', timeout=20).get('models', [])
    except Exception:
        return _VISION_CACHE['models']
    out = []
    for m in tags:
        name = m.get('name') or ''
        if not name:
            continue
        try:
            d = post_json(OLLAMA + '/api/show', {'model': name}, timeout=30)
        except Exception:
            continue
        caps = d.get('capabilities') or []
        if 'vision' not in caps:
            continue
        out.append({'name': name, 'label': re.sub(r':latest$', '', name),
                    'thinking': 'thinking' in caps})
    out.sort(key=lambda x: (x['name'] != VISION_DEFAULT, x['name']))
    _VISION_CACHE.update(t=now, models=out)
    return out


def vision_prepare(raw, ext):
    """Сохранить картинку и уменьшить до VISION_MAXSIDE по длинной стороне."""
    src = os.path.join('/tmp', 'vision-%s%s' % (uuid.uuid4().hex[:8], ext))
    with open(src, 'wb') as f:
        f.write(raw)
    dst = src.rsplit('.', 1)[0] + '-small.png'
    try:
        subprocess.run([GRADE_FFMPEG, '-nostdin', '-v', 'error', '-y', '-i', src,
                        '-vf', 'scale=%d:%d:force_original_aspect_ratio=decrease'
                        % (VISION_MAXSIDE, VISION_MAXSIDE), dst], timeout=90)
    except Exception:
        pass
    if os.path.isfile(dst) and os.path.getsize(dst) > 2048:
        try:
            os.remove(src)
        except OSError:
            pass
        return dst
    return src


def vision_describe(model, lang, raw, ext, hint=''):
    """Один запрос к модели со зрением: (текст, секунды). Временные файлы убирает сам."""
    path = vision_prepare(raw, ext)
    try:
        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        prompt = VISION_PROMPTS.get(lang) or VISION_PROMPTS['en']
        hint = (hint or '').strip()[:300]
        if hint:
            prompt += (' Особое внимание: %s.' % hint if lang == 'ru'
                       else ' Pay special attention to: %s.' % hint)
        caps = {}
        try:
            caps = post_json(OLLAMA + '/api/show', {'model': model}, timeout=30)
        except Exception:
            pass
        payload = {'model': model, 'prompt': prompt, 'images': [b64], 'stream': False,
                   'options': {'num_ctx': 4096, 'temperature': 0.2}, 'keep_alive': '10m'}
        if 'thinking' in (caps.get('capabilities') or []):
            payload['think'] = False          # иначе весь ответ уйдёт в размышление
        t0 = time.time()
        r = None
        last = None
        for attempt in (1, 2):
            try:
                r = post_json(OLLAMA + '/api/generate', payload, timeout=VISION_TIMEOUT)
                break
            except urllib.error.HTTPError as e:
                last = e
                try:
                    body = e.read().decode('utf-8', 'replace')
                except Exception:
                    body = ''
                # 500 от ollama почти всегда «не хватило видеопамяти»: второй раз без соседей
                if attempt == 1 and 'memory' in body.lower():
                    try:
                        post_json(OLLAMA + '/api/generate', {'model': model, 'keep_alive': 0},
                                  timeout=30)
                    except Exception:
                        pass
                    time.sleep(6)
                    continue
                raise
            except Exception as e:
                last = e
                if attempt == 1:
                    time.sleep(5)
                    continue
                raise
        if r is None:
            raise last or RuntimeError('модель не ответила')
        return (r.get('response') or '').strip(), round(time.time() - t0, 1)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def unload_ollama():
    names = [m['name'] for m in ollama_models()] or [TRANSLATOR]
    for model in names:
        try:
            post_json(OLLAMA + '/api/generate', {'model': model, 'keep_alive': 0}, timeout=30)
        except Exception:
            pass
    time.sleep(4)


def drop_file(name):
    """Удалить файл картинки из каталога ComfyUI; вне его ничего не трогаем."""
    if not name or '..' in name or name.startswith('/'):
        return 0
    base = os.path.realpath(OUT_DIR)
    full = os.path.realpath(os.path.join(base, name))
    if not full.startswith(base + os.sep):
        return 0
    try:
        os.remove(full)
        return 1
    except Exception:
        return 0


def lora_list():
    """LoRA, которые лежат в каталоге ComfyUI: [{'file', 'label', 'note'}]"""
    out = []
    try:
        names = sorted(os.listdir(LORAS_DIR))
    except Exception:
        return out
    for n in names:
        if not n.lower().endswith(('.safetensors', '.ckpt', '.pt')):
            continue
        label, note = LORA_LABELS.get(n, (re.sub(r'\.(safetensors|ckpt|pt)$', '', n), ''))
        out.append({'file': n, 'label': label, 'note': note})
    return out


def lora_ok(name):
    """Имя LoRA строго из списка (никаких путей в запросе)"""
    if not name:
        return None
    if not isinstance(name, str) or not re.fullmatch(r'[\w.\- ]{1,120}', name):
        return None
    for x in lora_list():
        if x['file'] == name:
            return name
    return None


def lora_from_req(req):
    """(файл LoRA, вес) из запроса: чужое имя превращается в (None, None)."""
    lora = lora_ok(req.get('lora'))
    try:
        w = float(req.get('lora_weight') or 0.8)
    except (TypeError, ValueError):
        w = 0.8
    return (lora, max(0.0, min(1.5, w))) if lora else (None, None)


def apply_lora(wf, lora, weight=0.8):
    """Вставить LoraLoader между чекпойнтом и теми, кто берёт у него модель/CLIP.

    Чекпойнт в наших графах — узел "1" (MODEL 0, CLIP 1, VAE 2); VAE остаётся
    прямым. Для видео не применяется (там UNETLoader).
    """
    if not lora:
        return wf
    w = max(0.0, min(1.5, float(weight)))
    nid = str(max(int(k) for k in wf) + 1)
    wf[nid] = {'class_type': 'LoraLoader',
               'inputs': {'lora_name': lora, 'strength_model': round(w, 2),
                          'strength_clip': round(w, 2),
                          'model': ['1', 0], 'clip': ['1', 1]}}
    for node in wf.values():
        if node.get('class_type') == 'LoraLoader':
            continue            # саму LoRA не перекручиваем (иначе она сошлётся на себя)
        ins = node.get('inputs') or {}
        for key, val in list(ins.items()):
            if (isinstance(val, list) and len(val) == 2 and val[0] == '1'
                    and val[1] in (0, 1)):
                ins[key] = [nid, val[1]]
    return wf


def build_workflow(prompt, neg, w, h, steps, cfg, seed, ckpt):
    prompt = with_prefix(prompt, ckpt)
    neg = with_prefix(neg, ckpt, negative=True)
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": w, "height": h, "batch_size": 1}},
        "5": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
                         "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                         "latent_image": ["4", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "ui", "images": ["6", 0]}},
    }


def build_video_workflow(prompt, neg, w, h, length, fps, steps, cfg, seed, ckpt):
    """LTXV 2B distilled: t5 -> conditioning -> латент видео -> сэмплер -> mp4."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": VIDEO_T5, "type": "ltxv", "device": "default"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["2", 0]}},
        "5": {"class_type": "EmptyLTXVLatentVideo",
              "inputs": {"width": w, "height": h, "length": length, "batch_size": 1}},
        "6": {"class_type": "LTXVConditioning",
              "inputs": {"positive": ["3", 0], "negative": ["4", 0], "frame_rate": float(fps)}},
        "7": {"class_type": "LTXVScheduler",
              "inputs": {"steps": steps, "max_shift": 2.05, "base_shift": 0.95,
                         "stretch": True, "terminal": 0.1, "latent": ["5", 0]}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "9": {"class_type": "SamplerCustom",
              "inputs": {"model": ["1", 0], "add_noise": True, "noise_seed": seed, "cfg": cfg,
                         "positive": ["6", 0], "negative": ["6", 1], "sampler": ["8", 0],
                         "sigmas": ["7", 0], "latent_image": ["5", 0]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "CreateVideo", "inputs": {"images": ["10", 0], "fps": float(fps)}},
        "12": {"class_type": "SaveVideo",
               "inputs": {"video": ["11", 0], "filename_prefix": "ui-vid",
                          "format": "mp4", "codec": "h264"}},
    }


def build_wan_workflow(prompt, neg, w, h, length, fps, steps, cfg, seed, ckpt):
    """Wan 2.2 ti2v-5B: unet + umt5 + wan-vae, ModelSamplingSD3 (shift 8)."""
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": ckpt, "weight_dtype": "default"}},
        "2": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 8.0}},
        "3": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": WAN_CLIP, "type": "wan", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": WAN_VAE}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["3", 0]}},
        "7": {"class_type": "Wan22ImageToVideoLatent",
              "inputs": {"vae": ["4", 0], "width": w, "height": h,
                         "length": length, "batch_size": 1}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["2", 0], "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                         "positive": ["5", 0], "negative": ["6", 0],
                         "latent_image": ["7", 0]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["4", 0]}},
        "10": {"class_type": "CreateVideo", "inputs": {"images": ["9", 0], "fps": float(fps)}},
        "11": {"class_type": "SaveVideo",
               "inputs": {"video": ["10", 0], "filename_prefix": "ui-vid",
                          "format": "mp4", "codec": "h264"}},
    }


def build_wan_flf_workflow(prompt, neg, w, h, length, fps, steps, cfg, seed, ckpt,
                           start_image, end_image):
    """Wan 2.2 first+last: два кадра задают начало и конец, модель ведёт переход.

    Картинка не уплывает в мыло к концу, потому что оба конца закреплены кадрами.
    """
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": ckpt, "weight_dtype": "default"}},
        "2": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 8.0}},
        "3": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": WAN_CLIP, "type": "wan", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": WAN_VAE}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["3", 0]}},
        "7": {"class_type": "LoadImage", "inputs": {"image": start_image}},
        "13": {"class_type": "LoadImage", "inputs": {"image": end_image}},
        "8": {"class_type": "WanFirstLastFrameToVideo",
              "inputs": {"positive": ["5", 0], "negative": ["6", 0], "vae": ["4", 0],
                         "width": w, "height": h, "length": length, "batch_size": 1,
                         "start_image": ["7", 0], "end_image": ["13", 0]}},
        "9": {"class_type": "KSampler",
              "inputs": {"model": ["2", 0], "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                         "positive": ["8", 0], "negative": ["8", 1],
                         "latent_image": ["8", 2]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["4", 0]}},
        "11": {"class_type": "CreateVideo", "inputs": {"images": ["10", 0], "fps": float(fps)}},
        "12": {"class_type": "SaveVideo",
               "inputs": {"video": ["11", 0], "filename_prefix": "ui-vid",
                          "format": "mp4", "codec": "h264"}},
    }


def png_size(raw):
    """Ширина и высота из заголовка PNG (IHDR) — без внешних библиотек."""
    if len(raw) < 24 or raw[:8] != b'\x89PNG\r\n\x1a\n':
        return None
    w = int.from_bytes(raw[16:20], 'big')
    h = int.from_bytes(raw[20:24], 'big')
    return w, h


def build_inpaint_workflow(prompt, neg, ckpt, src_image, mask_image, steps, cfg, seed,
                           denoise, grow):
    """Перерисовать выделенную область: маска белым по чёрному -> SetLatentNoiseMask.

    Работает с обычными (не inpaint) чекпойнтами: вне маски латент не трогается,
    внутри модель рисует по промпту с нуля наполовину (denoise).
    """
    prompt = with_prefix(prompt, ckpt)
    neg = with_prefix(neg, ckpt, negative=True)
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": src_image}},
        "5": {"class_type": "LoadImage", "inputs": {"image": mask_image}},
        "6": {"class_type": "ImageToMask",
              "inputs": {"image": ["5", 0], "channel": "red"}},
        "7": {"class_type": "GrowMask",
              "inputs": {"mask": ["6", 0], "expand": grow, "tapered_corners": True}},
        "8": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["1", 2]}},
        "9": {"class_type": "SetLatentNoiseMask",
              "inputs": {"samples": ["8", 0], "mask": ["7", 0]}},
        "10": {"class_type": "KSampler",
               "inputs": {"model": ["1", 0], "seed": seed, "steps": steps, "cfg": cfg,
                          "sampler_name": "dpmpp_2m", "scheduler": "karras",
                          "denoise": denoise, "positive": ["2", 0], "negative": ["3", 0],
                          "latent_image": ["9", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "13": {"class_type": "ImageCompositeMasked",
               "inputs": {"destination": ["4", 0], "source": ["11", 0], "x": 0, "y": 0,
                          "resize_source": False, "mask": ["7", 0]}},
        "12": {"class_type": "SaveImage",
               "inputs": {"images": ["13", 0], "filename_prefix": "ui-inpaint"}},
    }


DETECTORS = {
    'hands': ('bbox/hand_yolov8s.pt', 'кисти', 'ui-hands'),
    'face': ('bbox/face_yolov8m.pt', 'лицо', 'ui-face'),
}

# Умолчания ремонта по типу (замеры 2026-11-15, один кадр, seed 4242).
# Кисти: на силе 0.45 диффузер картину только «дорисовывает» — лишние пальцы
# остаются; выбор Ивана — 0.60 с рамкой 2.0 (на 0.70 и рамке 1.8 он сказал
# «уже карикатура»). Порог 0.15 — чтобы ловились и сомнительные кисти.
# Лицо оставлено как было (0.50, 2.6) — оно и так починилось.
DETAIL_DEFAULTS = {
    'hands': {'denoise': 0.60, 'crop_factor': 2.0, 'threshold': 0.15, 'dilation': 8},
    'face': {'denoise': 0.50, 'crop_factor': 2.6, 'threshold': 0.30, 'dilation': 8},
}

# Без этого SDXL тянет лишние пальцы даже при полной перерисовке куска
HAND_NEG_EXTRA = ('extra fingers, six fingers, seven fingers, fused fingers, '
                  'malformed hand, deformed hand')
UPSCALE_MODEL = '4x-UltraSharp.pth'
CONTROLNET_MODEL = 'controlnet-union-sdxl-promax.safetensors'
# Скелет (openpose) — отдельная модель xinsir: она обучена именно на кистях.
CONTROLNET_POSE = 'controlnet-openpose-sdxl-1.0.safetensors'
SDPOSE_CKPT = 'sdpose_wholebody_fp16.safetensors'


# Чем готовим подсказку под каждый тип Union ControlNet.
# Контур снимает встроенный узел ComfyUI, остальное — препроцессоры
# из comfyui_controlnet_aux (пакет поставлен 02.11).
# 'openpose' здесь НЕТ: скелет считает SDPose (ядро ComfyUI), см. add_pose_nodes.
PREPROCESSORS = {
    'depth': ('DepthAnythingV2Preprocessor',
              {'ckpt_name': 'depth_anything_v2_vits.pth', 'resolution': 1024}),
    'hed': ('HEDPreprocessor', {'safe': 'enable', 'resolution': 1024}),
    'normal': ('BAE-NormalMapPreprocessor', {'resolution': 1024}),
}


def add_pose_nodes(wf, draw_body=False, score_threshold=0.3):
    """Положить в граф добычу скелета и вернуть ссылку на нарисованную карту позы.

    SDPose — модель ядра ComfyUI (heatmap_head внутри диффузионной модели):
    грузится обычным CheckpointLoaderSimple, выдаёт 133 ключевых точки, включая
    по 21 на каждую кисть. SDPoseDrawKeypoints рисует из них картинку в стиле
    openpose — её и ждёт ControlNet. Считает и на рисованных кадрах
    (для этого она и сделана — out-of-distribution).
    """
    wf["20"] = {"class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": SDPOSE_CKPT}}
    wf["21"] = {"class_type": "SDPoseKeypointExtractor",
                "inputs": {"model": ["20", 0], "vae": ["20", 2],
                           "image": ["4", 0], "batch_size": 1}}
    wf["22"] = {"class_type": "SDPoseDrawKeypoints",
                "inputs": {"keypoints": ["21", 0], "draw_body": draw_body,
                           "draw_hands": True, "draw_face": False,
                           "draw_feet": False, "stick_width": 4,
                           "face_point_size": 3,
                           "score_threshold": score_threshold, "draw_head": False}}
    return ["22", 0]


def build_detail_workflow(prompt, neg, ckpt, src_image, steps, cfg, seed, denoise,
                          detector='hands', upscale=False, size=None, threshold=0.3,
                          dilation=8, crop_factor=3.0, guide_size=1024,
                          pose=False, pose_strength=0.9):
    """Починить кисти или лицо: детектор -> перерисовка каждого куска -> (апскейл).

    Ноды из Impact Pack и Impact Subpack: UltralyticsDetectorProvider находит объект,
    BboxDetectorSEGS превращает рамки в SEGS, DetailerForEach вырезает кусок,
    увеличивает его до guide_size, перерисовывает с denoise и вставляет обратно.
    Если ничего не нашлось, картинка возвращается без изменений.

    pose=True — сверху добавляется скелет из SDPose и ControlNet (openpose):
    диффузер тогда ставит пальцы по 21 точке кисти, а не угадывает.
    """
    model_file, what, prefix = DETECTORS.get(detector, DETECTORS['hands'])
    prompt = with_prefix(prompt, ckpt)
    neg = with_prefix(neg, ckpt, negative=True)
    wf = {        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": src_image}},
        "5": {"class_type": "UltralyticsDetectorProvider",
              "inputs": {"model_name": model_file}},
        "6": {"class_type": "BboxDetectorSEGS",
              "inputs": {"bbox_detector": ["5", 0], "image": ["4", 0],
                         "threshold": threshold, "dilation": dilation,
                         "crop_factor": crop_factor, "drop_size": 10, "labels": "all"}},
        "7": {"class_type": "DetailerForEach",
              "inputs": {"image": ["4", 0], "segs": ["6", 0], "model": ["1", 0],
                         "clip": ["1", 1], "vae": ["1", 2],
                         "guide_size": float(guide_size), "guide_size_for": True,
                         "max_size": 1024.0, "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "dpmpp_2m", "scheduler": "karras",
                         "positive": ["2", 0], "negative": ["3", 0],
                         "denoise": denoise, "feather": 5, "noise_mask": True,
                         "force_inpaint": True, "wildcard": "", "cycle": 1,
                         "noise_mask_feather": 12}},
    }
    if pose:
        pose_img = add_pose_nodes(wf, draw_body=False)
        wf["23"] = {"class_type": "ControlNetLoader",
                    "inputs": {"control_net_name": CONTROLNET_POSE}}
        wf["24"] = {"class_type": "ImpactControlNetApplyAdvancedSEGS",
                    "inputs": {"segs": ["6", 0], "control_net": ["23", 0],
                               "strength": float(pose_strength),
                               "start_percent": 0.0, "end_percent": 1.0,
                               "control_image": pose_img, "vae": ["1", 2]}}
        wf["7"]["inputs"]["segs"] = ["24", 0]
    last = ["7", 0]
    if upscale and size:
        wf["8"] = {"class_type": "UpscaleModelLoader",
                   "inputs": {"model_name": UPSCALE_MODEL}}
        wf["9"] = {"class_type": "ImageUpscaleWithModel",
                   "inputs": {"upscale_model": ["8", 0], "image": last}}
        wf["10"] = {"class_type": "ImageScale",
                    "inputs": {"image": ["9", 0], "upscale_method": "lanczos",
                               "width": int(size[0] * 1.5), "height": int(size[1] * 1.5),
                               "crop": "disabled"}}
        last = ["10", 0]
    wf["11"] = {"class_type": "SaveImage",
                "inputs": {"images": last, "filename_prefix": prefix}}
    return wf


def build_redraw_workflow(prompt, neg, ckpt, src_image, steps, cfg, seed, denoise,
                          strength=0.8, ctrl_type='canny/lineart/anime_lineart/mlsd'):
    """Перерисовать кадр целиком, держа подсказку исходника через ControlNet Union.

    Подсказка зависит от типа: контур (canny) снимает встроенный узел, глубина —
    DepthAnythingV2, поза — Openpose, линии — HED, нормали — BAE; плитка берёт
    сам кадр. KSampler с denoise 0.5-0.8 меняет анатомию и детали, не сдвигая
    композицию настолько, насколько её держит выбранная подсказка.
    """
    prompt = with_prefix(prompt, ckpt)
    neg = with_prefix(neg, ckpt, negative=True)
    hint = (ctrl_type or '').split('/')[0]
    pose_hint = hint == 'openpose'
    if pose_hint:
        hint_node = None          # подсказку даст SDPose, добавим ниже
    elif hint in PREPROCESSORS:
        cls, extra = PREPROCESSORS[hint]
        hint_node = {"class_type": cls, "inputs": dict(extra, image=["4", 0])}
    elif hint == 'tile':
        hint_node = None          # подсказка — сам кадр, без препроцессора
    else:
        hint_node = {"class_type": "Canny",
                     "inputs": {"image": ["4", 0], "low_threshold": 0.4,
                                "high_threshold": 0.8}}
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": src_image}},
        "6": {"class_type": "ControlNetLoader",
              "inputs": {"control_net_name": CONTROLNET_POSE if pose_hint
                         else CONTROLNET_MODEL}},
        "7": {"class_type": "SetUnionControlNetType",
              "inputs": {"control_net": ["6", 0], "type": ctrl_type}},
        "8": {"class_type": "ControlNetApplyAdvanced",
              "inputs": {"positive": ["2", 0], "negative": ["3", 0],
                         "control_net": ["7", 0],
                         "image": ["5", 0] if hint_node else ["4", 0],
                         "strength": strength, "start_percent": 0.0, "end_percent": 0.9}},
        "9": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["1", 2]}},
        "10": {"class_type": "KSampler",
               "inputs": {"model": ["1", 0], "seed": seed, "steps": steps, "cfg": cfg,
                          "sampler_name": "dpmpp_2m", "scheduler": "karras",
                          "denoise": denoise, "positive": ["8", 0], "negative": ["8", 1],
                          "latent_image": ["9", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage",
               "inputs": {"images": ["11", 0], "filename_prefix": "ui-redraw"}},
    }
    if hint_node:
        wf["5"] = hint_node
    if pose_hint:
        # отдельная модель openpose не union: тип ей задавать нечем и не нужно
        wf.pop("7", None)
        pose_img = add_pose_nodes(wf, draw_body=True)
        wf["8"]["inputs"]["control_net"] = ["6", 0]
        wf["8"]["inputs"]["image"] = pose_img
    return wf


def render_frame(prompt, w, h, seed, ckpt, prefix='ui-frame'):
    """Нарисовать кадр SDXL и положить копию в input ComfyUI; вернуть имя файла."""
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": NEG_DEFAULT, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": w, "height": h, "batch_size": 1}},
        "5": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "seed": seed, "steps": 28, "cfg": 5.0,
                         "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
                         "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage",
              "inputs": {"images": ["6", 0], "filename_prefix": prefix}},
    }
    pid = post_json(API + '/prompt', {'prompt': wf, 'client_id': uuid.uuid4().hex},
                    timeout=60)['prompt_id']
    t0 = time.time()
    while True:
        time.sleep(1)
        hst = get_json(API + '/history/' + pid, timeout=30)
        if pid in hst:
            if not hst[pid].get('status', {}).get('completed'):
                return None, 'ComfyUI не смог нарисовать кадр'
            files = []
            for node in hst[pid].get('outputs', {}).values():
                files += node.get('images', [])
            if not files:
                return None, 'кадр не сохранился'
            name = files[0]['filename']
            src = os.path.join(OUT_DIR, files[0].get('subfolder', ''), name)
            shutil.copyfile(src, os.path.join(INPUT_DIR, name))
            os.chmod(os.path.join(INPUT_DIR, name), 0o644)
            return name, None
        if time.time() - t0 > 900:
            return None, 'таймаут при отрисовке кадра'


# ---------------------------------------------------------------- очередь задач
JOBS = {}
JOBQ = queue.Queue()
JOBS_LOCK = threading.Lock()
HIST_LOCK = threading.Lock()


def job_estimate(items, kind, w, h):
    """Медиана времени из ленты для похожей задачи (тот же вид и размер)."""
    same = [x['sec'] for x in items
            if x.get('kind') == kind and x.get('w') == w and x.get('h') == h and x.get('sec')]
    if not same:
        same = [x['sec'] for x in items if x.get('kind') == kind and x.get('sec')]
    if not same:
        return None
    same.sort()
    return round(same[len(same) // 2], 1)


def queue_snapshot():
    """Что сейчас делается и что ждёт: наши задачи + состояние ComfyUI."""
    now = time.time()
    with JOBS_LOCK:
        jobs = list(JOBS.values())
    hist = load_history()
    active = [j for j in jobs if j.get('state') in ('queued', 'running')]
    active.sort(key=lambda j: (j.get('state') != 'running', j.get('created') or 0))
    rows = []
    for j in active:
        req = j.get('req') or {}
        ru = (req.get('prompt') or '').strip()
        row = {'id': j.get('id'), 'state': j.get('state'),
               'phase': j.get('phase') or '',
               'what': ru[:90] + ('…' if len(ru) > 90 else ''),
               'label': req.get('label') or '',
               'kind': req.get('kind') or 'image',
               'size': '%dx%d' % (req.get('w') or 0, req.get('h') or 0),
               'steps': req.get('steps'),
               'waited': round(now - (j.get('created') or now), 1),
               'elapsed': round(now - (j.get('started') or now), 1)
               if j.get('state') == 'running' else None,
               'est': job_estimate(hist, req.get('kind') or 'image',
                                   req.get('w'), req.get('h')),
               'prompt_id': j.get('prompt_id')}
        rows.append(row)
    let_in_comfy = None
    running = 0
    pending = 0
    try:
        q = get_json(API + '/queue', timeout=8)
        running = len(q.get('queue_running') or [])
        pending = len(q.get('queue_pending') or [])
        ours = {j.get('prompt_id') for j in jobs if j.get('prompt_id')}
        ids = [x[1] for x in (q.get('queue_running') or []) if len(x) > 1]
        let_in_comfy = [i for i in ids if i not in ours]
    except Exception:
        pass
    return {'now': rows[0] if rows else None,
            'pending': rows[1:] if len(rows) > 1 else [],
            'count': len(rows),
            'comfy': {'running': running, 'pending': pending,
                      'foreign': bool(let_in_comfy) if let_in_comfy is not None else False,
                      'reachable': let_in_comfy is not None},
            'server_time': int(now)}


def load_history():
    try:
        with open(HIST_FILE, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def save_history(items):
    tmp = HIST_FILE + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    os.replace(tmp, HIST_FILE)


def add_history(entry):
    with HIST_LOCK:
        items = load_history()
        items.insert(0, entry)
        save_history(items[:MAX_HISTORY])


def photos_list():
    """Фотографии, лежащие в каталоге обучения."""
    out = []
    try:
        for n in sorted(os.listdir(TRAIN_PHOTOS)):
            if n.lower().endswith(PHOTO_EXT):
                p = os.path.join(TRAIN_PHOTOS, n)
                out.append({'name': n, 'kb': max(1, os.path.getsize(p) // 1024)})
    except Exception:
        pass
    return out


def caption_path(photo):
    """Файл подписи рядом с фотографией: имя то же, расширение .txt."""
    return os.path.join(TRAIN_PHOTOS, os.path.splitext(photo)[0] + '.txt')


def caption_read():
    """Общая подпись набора: отдаём, только если у всех фотографий она одна."""
    vals = set()
    for x in photos_list():
        try:
            with open(caption_path(x['name']), encoding='utf-8') as f:
                vals.add(f.read().strip())
        except Exception:
            vals.add('')
    if len(vals) == 1:
        return vals.pop()
    return ''


def caption_write(text):
    """Записать одну подпись на все фотографии; пусто — удалить файлы подписей."""
    text = (text or '').strip()
    n = 0
    for x in photos_list():
        p = caption_path(x['name'])
        try:
            if text:
                with open(p, 'w', encoding='utf-8') as f:
                    f.write(text + '\n')
                n += 1
            elif os.path.isfile(p):
                os.remove(p)
        except Exception:
            pass
    return n


def load_train_job():
    try:
        with open(TRAIN_STATE, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def save_train_job(job):
    tmp = TRAIN_STATE + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(job, f, ensure_ascii=False)
    os.replace(tmp, TRAIN_STATE)


STEP_RE = re.compile(r'(\d+)/(\d+)\s*\[[^\]]*?([\d.]+)s/it')


def trainer_busy():
    """Идёт ли контейнер обучения (карта занята)."""
    try:
        out = subprocess.run(['docker', 'ps', '--filter', 'ancestor=' + TRAIN_IMAGE,
                              '--format', '{{.Names}}'], capture_output=True,
                             text=True, timeout=20).stdout.strip()
        return bool(out)
    except Exception:
        return False


def tail_text(path, limit=40000):
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - limit))
            return f.read().decode('utf-8', 'replace')
    except Exception:
        return ''


def train_status():
    """Состояние обучения для панели: проценты, шаг, готовый файл."""
    job = load_train_job()
    out = {'state': 'idle', 'photos': photos_list(), 'loras': lora_list(),
           'busy': trainer_busy(), 'steps': 1500, 'dim': 16, 'name': '',
           'caption': caption_read()}
    if not job:
        return out
    out['name'] = job.get('name') or ''
    out['steps'] = job.get('steps') or 1500
    out['dim'] = job.get('dim') or 16
    out['started'] = job.get('started')
    log = job.get('log') or ''
    txt = tail_text(log)
    # sd-scripts печатает несколько полос прогресса (подготовка датасета, кэш
    # латентов и сама учёба): берём ту, где всего шагов столько же, сколько заказали
    matches = [(int(m.group(1)), int(m.group(2)), float(m.group(3)))
               for m in STEP_RE.finditer(txt)]
    want = job.get('steps') or 0
    pick = None
    for s, t, per in matches:
        if t == want:
            pick = (s, t, per)
    if pick is None and matches and not want:
        pick = matches[-1]
    step = total = per = None
    if pick:
        step, total, per = pick
    if total:
        out['step'], out['total'], out['sec_per_it'] = step, total, per
        out['pct'] = max(0, min(100, round(100.0 * (step or 0) / total)))
        if per and step is not None and total:
            out['left_sec'] = int((total - step) * per)
    # публикация идёт отдельным сторожевым процессом: publish.log — самый
    # надёжный признак «стиль готов» / «файл не появился»
    pub = os.path.join(TRAIN_RUNS, job.get('name') or '', 'publish.log')
    pubtxt = tail_text(pub, 4000)
    pub_yes = 'опубликована' in pubtxt
    pub_no = 'не опубликован' in pubtxt
    if pub_yes:
        out['published'] = True
    elif pub_no:
        out['published'] = False
    # «model saved» — файл только появился на диске; в список стилей его кладёт
    # сторожевой процесс (до 30 с), поэтому готовым считаем только по publish.log
    saved = 'model saved' in txt
    if job.get('cancelled') and not trainer_busy() and not (pub_yes or pub_no):
        out['state'] = 'stopped'
    elif pub_yes or pub_no:
        out['state'] = 'done'
        out['lora'] = job.get('name')
    elif saved or trainer_busy():
        out['state'] = 'publishing' if saved else 'running'
    elif 'без файла' in pubtxt:
        out['state'] = 'error'
        out['error'] = 'обучение закончилось без файла, смотри журнал'
    elif 'Traceback' in txt or 'Error' in txt.splitlines()[-1:]:
        out['state'] = 'error'
        out['error'] = 'обучение упало, смотри журнал'
    else:
        out['state'] = 'stalled'
        out['error'] = 'обучение остановилось без готового файла'
    return out


def train_start(name, steps, dim, base):
    """Запустить обучение. Возвращает (ok, текст ошибки)."""
    if trainer_busy():
        return False, 'уже идёт обучение, подождите'
    if _unit_active(VIDEO_UNIT) in ('active', 'activating'):
        return False, 'сейчас идёт обработка видео, видеокарта занята'
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{1,40}', name or ''):
        return False, 'название: латиница, цифры, _ и -, от 2 символов'
    if not photos_list():
        return False, 'сначала добавьте фотографии'
    if not os.path.isfile(TRAIN_SCRIPT):
        return False, 'на стенде нет %s' % TRAIN_SCRIPT
    steps = max(20, min(6000, int(steps or 1500)))
    dim = max(4, min(64, int(dim or 16)))
    base = base if base in ('juggernaut', 'realvis') else 'juggernaut'
    cmd = ['bash', TRAIN_SCRIPT, '--data', TRAIN_PHOTOS, '--name', name,
           '--steps', str(steps), '--dim', str(dim), '--base', base]
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True, cwd=TRAINER_DIR)
    except Exception as e:
        return False, 'не смог запустить: %s' % e
    save_train_job({'name': name, 'steps': steps, 'dim': dim, 'base': base,
                    'started': time.time(),
                    'log': os.path.join(TRAIN_RUNS, name, 'train.log')})
    return True, None


def train_stop():
    """Остановить обучение и пометить его как отменённое."""
    try:
        ids = subprocess.run(['docker', 'ps', '-q', '--filter', 'ancestor=' + TRAIN_IMAGE],
                             capture_output=True, text=True, timeout=20).stdout.split()
        for cid in ids:
            subprocess.run(['docker', 'kill', cid], capture_output=True, timeout=30)
    except Exception as e:
        return False, str(e)
    job = load_train_job() or {}
    job['cancelled'] = True
    save_train_job(job)
    return True, None

# -------------------------------------------------------------- обработка видео

def _vdirs():
    for d in (VIDEO_IN, VIDEO_OUT, VIDEO_LOG_DIR):
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass


def _vtail(path, limit=8000):
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - limit))
            return f.read().decode('utf-8', 'replace')
    except Exception:
        return ''


def _unit_active(unit):
    try:
        r = subprocess.run(['systemctl', 'is-active', unit + '.service'],
                           capture_output=True, text=True, timeout=10)
        return (r.stdout or '').strip() or 'inactive'
    except Exception:
        return 'unknown'


def video_state():
    try:
        with open(VIDEO_STATE) as f:
            st = json.load(f)
        return st if isinstance(st, dict) else {}
    except Exception:
        return {}


def video_save(st):
    try:
        tmp = VIDEO_STATE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(st, f, ensure_ascii=False)
        os.replace(tmp, VIDEO_STATE)
    except Exception:
        pass


def split_state():
    try:
        with open(VIDEO_SPLIT_STATE) as f:
            st = json.load(f)
        return st if isinstance(st, dict) else {}
    except Exception:
        return {}


def split_save(st):
    try:
        tmp = VIDEO_SPLIT_STATE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(st, f, ensure_ascii=False)
        os.replace(tmp, VIDEO_SPLIT_STATE)
    except Exception:
        pass


# --- «цвет и яркость»: грейд готового видео под палитру исходника ---
GRADE_DIR = os.environ.get('GRADE_DIR', os.path.join(OUT_DIR, 'vace'))
GRADE_OUT = os.environ.get('GRADE_OUT', os.path.join(PANEL_HOME, 'graded'))
# результат «улучшения видео» (вкладка нейросети) — тоже годится для грейда
GRADE_ENH = os.environ.get('GRADE_ENH', VIDEO_OUT)
GRADE_SRC_OUT = (('перерисовка', GRADE_DIR), ('улучшение', GRADE_ENH), ('грейд', GRADE_OUT))
GRADE_SRC_IN = (('вход', INPUT_DIR),)
GRADE_FFMPEG = os.environ.get('GRADE_FFMPEG', '/usr/bin/ffmpeg')
GRADE_FFPROBE = os.environ.get('GRADE_FFPROBE', '/usr/bin/ffprobe')
GRADE_EXT = ('.mp4', '.mkv', '.mov', '.webm')
GRADE_MAXSIDE = 960
GRADE_DEFAULTS = {'gamma': 90, 'contrast': 115, 'black': 12, 'sat': 72,
                  'shadow': 90, 'cb': 1.6, 'cr': -2.2, 'sharp': 0}
GRADE_LOCK = threading.Lock()
_GRADE_FRAMES = {}
_GRADE_STATS = {}


def _grade_files():
    """видео для выбора: перерисовка (VACE), улучшение (нейросеть) и грейд"""
    res = {'out': [], 'in': []}
    for key, dirs in (('out', GRADE_SRC_OUT), ('in', GRADE_SRC_IN)):
        have = set()
        for tag, d in dirs:
            try:
                names = sorted(os.listdir(d))
            except OSError:
                names = []
            for n in names:
                full = os.path.join(d, n)
                if (n.lower().endswith(GRADE_EXT) and n not in have
                        and os.path.isfile(full)):
                    have.add(n)
                    res[key].append({'name': n, 'hint': tag,
                                     'size': os.path.getsize(full),
                                     'mtime': int(os.path.getmtime(full))})
    for key in res:
        res[key].sort(key=lambda x: -x['mtime'])
    return res


def _grade_find(name):
    name = os.path.basename(name or '')
    if not name:
        return None
    for d in (GRADE_OUT, GRADE_DIR, GRADE_ENH, INPUT_DIR):
        full = os.path.join(d, name)
        if os.path.isfile(full):
            return full
    return None


def _grade_frames(name):
    """все кадры файла как uint8 (N,h,w,3); кэш по имени, mtime и размеру"""
    import numpy as np
    full = _grade_find(name)
    if not full:
        raise RuntimeError('нет файла %s' % name)
    key = (name, int(os.path.getmtime(full)), os.path.getsize(full))
    hit = _GRADE_FRAMES.get(key)
    if hit is not None:
        return hit
    pr = subprocess.run([GRADE_FFPROBE, '-v', 'error', '-select_streams', 'v:0',
                         '-show_entries', 'stream=width,height', '-of', 'csv=p=0', full],
                        capture_output=True, text=True)
    try:
        w, h = [int(float(x)) for x in pr.stdout.strip().split(',')[:2]]
    except Exception:
        raise RuntimeError('не смог прочитать размер кадра в %s' % name)
    if w > GRADE_MAXSIDE or h > GRADE_MAXSIDE:
        k = min(GRADE_MAXSIDE / float(w), GRADE_MAXSIDE / float(h))
        w, h = int(w * k), int(h * k)
    w -= w % 2
    h -= h % 2
    buf = subprocess.run([GRADE_FFMPEG, '-nostdin', '-v', 'error', '-i', full,
                          '-vf', 'scale=%d:%d:flags=bicubic' % (w, h),
                          '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                         capture_output=True).stdout
    step = w * h * 3
    n = len(buf) // step
    if not n:
        raise RuntimeError('кадров не нашёл в %s' % name)
    arr = np.frombuffer(buf[:n * step], dtype=np.uint8).reshape(n, h, w, 3)
    if len(_GRADE_FRAMES) > 4:
        _GRADE_FRAMES.clear()
    _GRADE_FRAMES[key] = arr
    return arr


def _grade_stats(name):
    """средние и СКО по Y, Cb, Cr — по всем кадрам файла"""
    import numpy as np
    full = _grade_find(name)
    if not full:
        raise RuntimeError('нет файла %s' % name)
    key = (name, int(os.path.getmtime(full)), os.path.getsize(full))
    hit = _GRADE_STATS.get(key)
    if hit is not None:
        return hit
    a = _grade_frames(name).astype(np.float32)
    y, cb, cr = _grade_ycc(a)
    st = (float(y.mean()), float(y.std()), float(cb.mean()), float(cb.std()),
          float(cr.mean()), float(cr.std()))
    del a, y, cb, cr
    if len(_GRADE_STATS) > 8:
        _GRADE_STATS.clear()
    _GRADE_STATS[key] = st
    return st


def _grade_ycc(f):
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    return y, cb, cr


def _grade_params(req):
    p = dict(GRADE_DEFAULTS)
    for k in GRADE_DEFAULTS:
        v = req.get(k)
        if v is None:
            continue
        try:
            p[k] = float(v)
        except (TypeError, ValueError):
            pass
    return p


def _grade_apply_np(arr, st_out, st_src, p):
    """линейная подгонка под палитру исходника + ручки, на массиве кадров (N,h,w,3)"""
    import numpy as np
    f = arr.astype(np.float32)
    y, cb, cr = _grade_ycc(f)
    del f
    oy, oys, ocb, ocbs, ocr, ocrs = st_out
    sy, sys_, scb, scbs, scr, scrs = st_src
    y = (y - oy) * (sys_ / (oys or 1.0)) + sy
    y = (y - 60.0) * (p['contrast'] / 100.0) + 60.0
    # гашение цвета в тенях: base — доля цвета, которая остаётся в самых тёмных местах
    base = max(0.0, (100.0 - p['shadow']) / 100.0)
    shade = np.clip(y / 100.0, 0.0, 1.0)
    k = (p['sat'] / 100.0) * (base + (1.0 - base) * shade)
    cb = 128.0 + (cb - ocb) * (scbs / (ocbs or 1.0)) * k + p['cb'] * shade
    cr = 128.0 + (cr - ocr) * (scrs / (ocrs or 1.0)) * k + p['cr'] * shade
    y = 255.0 * np.clip(y / 255.0, 0.0, 1.0) ** (p['gamma'] / 100.0)
    if p['black'] > 0:
        y = (y - p['black']) * 255.0 / max(255.0 - p['black'], 1.0)
    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)
    out = np.clip(np.stack([r, g, b], axis=-1), 0.0, 255.0)
    if p['sharp'] > 0:
        w = min(p['sharp'] / 100.0, 0.9)
        p4 = np.pad(out, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='edge')
        lap = (5.0 * p4[:, 1:-1, 1:-1] - p4[:, :-2, 1:-1] - p4[:, 2:, 1:-1]
               - p4[:, 1:-1, :-2] - p4[:, 1:-1, 2:])
        out = np.clip(out * (1.0 - w) + lap * w, 0.0, 255.0)
    return out.astype(np.uint8)


def _grade_numbers(arr):
    """что показать рядом с примером: яркость, медиана, цвет в тенях"""
    import numpy as np
    f = arr.astype(np.float32)
    y, cb, cr = _grade_ycc(f)
    sat = np.hypot(cb - 128.0, cr - 128.0)
    dark = y <= np.percentile(y, 20)
    return {'y': round(float(y.mean()), 1), 'med': int(np.median(y)),
            'shadow': round(float(sat[dark].mean()), 2)}


def grade_preview(req):
    name = (req.get('file') or '').strip()
    src = (req.get('src') or '').strip()
    if not name or not src:
        raise RuntimeError('нужен файл и откуда брать палитру')
    p = _grade_params(req)
    a_out = _grade_frames(name)
    n = int(a_out.shape[0])
    idx = int(req.get('frame') or n // 3)
    idx = max(0, min(n - 1, idx))
    st_out = _grade_stats(name)
    st_src = _grade_stats(src)
    out = _grade_apply_np(a_out[idx:idx + 1], st_out, st_src, p)[0]
    from PIL import Image
    import io
    b1 = io.BytesIO()
    Image.fromarray(out).save(b1, 'JPEG', quality=88)
    b2 = io.BytesIO()
    Image.fromarray(a_out[idx]).save(b2, 'JPEG', quality=88)
    return {'jpeg': 'data:image/jpeg;base64,' + base64.b64encode(b1.getvalue()).decode(),
            'jpeg_src': 'data:image/jpeg;base64,' + base64.b64encode(b2.getvalue()).decode(),
            'frame': idx, 'frames': n,
            'stats': _grade_numbers(out), 'stats0': _grade_numbers(a_out[idx:idx + 1])}


def grade_apply(req):
    name = (req.get('file') or '').strip()
    src = (req.get('src') or '').strip()
    if not name or not src:
        raise RuntimeError('нужен файл и откуда брать палитру')
    p = _grade_params(req)
    import numpy as np
    t0 = time.time()
    with GRADE_LOCK:
        a = _grade_frames(name)
        st_out = _grade_stats(name)
        st_src = _grade_stats(src)
        h, w = a.shape[1], a.shape[2]
        pieces = []
        step = 12
        for i in range(0, a.shape[0], step):
            pieces.append(_grade_apply_np(a[i:i + step], st_out, st_src, p))
        res = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        pr = subprocess.run([GRADE_FFPROBE, '-v', 'error', '-select_streams', 'v:0',
                             '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0',
                             _grade_find(name)], capture_output=True, text=True)
        fps = 25.0
        try:
            num, _, den = pr.stdout.strip().partition('/')
            fps = float(num) / float(den or 1)
        except Exception:
            pass
        tag = time.strftime('%Y%m%d-%H%M%S')
        try:
            os.makedirs(GRADE_OUT, exist_ok=True)
        except OSError:
            pass
        dst = os.path.join(GRADE_OUT, 'grade_%s.mp4' % tag)
        proc = subprocess.run([GRADE_FFMPEG, '-nostdin', '-v', 'error', '-y',
                               '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                               '-s', '%dx%d' % (w, h), '-r', '%.3f' % fps, '-i', '-',
                               '-c:v', 'libx264', '-crf', '6', '-pix_fmt', 'yuv420p', dst],
                              input=res.tobytes(), capture_output=True)
        if proc.returncode:
            raise RuntimeError('ffmpeg не записал файл: ' + proc.stderr.decode()[:300])
    return {'name': os.path.basename(dst), 'url': '/vg/' + os.path.basename(dst),
            'sec': round(time.time() - t0, 1), 'frames': int(res.shape[0]),
            'stats': _grade_numbers(res[:1])}


def grade_ready():
    try:
        names = sorted(os.listdir(GRADE_OUT), reverse=True)
    except OSError:
        return []
    return [n for n in names if n.startswith('grade_') and n.lower().endswith('.mp4')][:12]


def video_split_status():
    """Что известно о разбивке выбранного файла: идёт / готово / ошибка."""
    st = split_state()
    name = st.get('name') or ''
    unit = _unit_active(VIDEO_SPLIT_UNIT)
    res = None
    try:
        with open(VIDEO_SPLIT_OUT) as f:
            res = json.load(f)
    except Exception:
        res = None
    out = {'name': name, 'state': 'idle', 'at': st.get('at') or 0,
           'segments': [], 'thresholds': {}, 'err': '', 'dur': 0, 'fps': 0}
    if isinstance(res, dict) and res.get('name') == name and res.get('segments'):
        out.update({'segments': res.get('segments') or [],
                    'thresholds': res.get('thresholds') or {},
                    'dur': res.get('dur') or 0, 'fps': res.get('fps') or 0,
                    'state': 'ready'})
    if unit in ('active', 'activating', 'deactivating'):
        out['state'] = 'running'
        out['segments'] = []
    elif st.get('error') and out['state'] != 'ready':
        out['state'] = 'error'
        out['err'] = st.get('error') or ''
    elif not out['segments'] and st.get('name') and st.get('finished'):
        tail = [ln for ln in _vtail(VIDEO_SPLIT_LOG, 4000).strip().splitlines() if ln.strip()]
        out['state'] = 'error'
        out['err'] = '\n'.join(tail[-3:])[-400:] or 'разбор не дал кусков'
    if st.get('started'):
        out['sec'] = max(0, int(time.time() - float(st['started'])))
    return out


def video_split_start(name):
    """Разбор на куски — отдельным юнитом: длинный файл считается минутами."""
    name = os.path.basename((name or '').strip())
    if not name or not name.lower().endswith(VIDEO_EXT):
        return False, 'разбивка работает только с видеофайлами'
    full = os.path.join(VIDEO_IN, name)
    if not os.path.isfile(full):
        return False, 'нет такого файла на стенде'
    if _unit_active(VIDEO_SPLIT_UNIT) in ('active', 'activating'):
        return False, 'разбор уже идёт'
    for p in (VIDEO_SPLIT_OUT, VIDEO_SPLIT_LOG):
        try:
            os.remove(p)
        except OSError:
            pass
    cmd = ['sudo', '-n', 'systemd-run', '--unit=' + VIDEO_SPLIT_UNIT, '--collect',
           '--no-block', '--property=Type=oneshot',
           '--property=StandardOutput=append:' + VIDEO_SPLIT_OUT,
           '--property=StandardError=append:' + VIDEO_SPLIT_LOG,
           '/usr/bin/env', 'FFMPEG=/usr/bin/ffmpeg', 'FFPROBE=/usr/bin/ffprobe',
           '/usr/bin/python3', VIDEO_SPLIT_SCRIPT, full, '--json']
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception as e:
        return False, 'не смог запустить разбор: %s' % e
    if r.returncode != 0:
        return False, 'разбор не запустился: %s' % ((r.stderr or r.stdout or '').strip()[-300:])
    split_save({'name': name, 'started': time.time(), 'at': time.time()})
    return True, None


def _vmeta(full, mtime):
    """Формат, разрешение, кадры, длительность и частота (ffprobe, кэш по mtime)."""
    key = (full, mtime)
    if key in _VDUR_CACHE:
        return _VDUR_CACHE[key]
    info = {'dur': 0, 'w': 0, 'h': 0, 'frames': 0, 'fps': 0.0,
            'codec': '', 'container': '', 'photo': False}
    low = full.lower()
    if low.endswith(PHOTO_IN_EXT):
        info['photo'] = True
        ext = os.path.splitext(low)[1]
        info['container'] = {'.png': 'PNG', '.jpg': 'JPEG', '.jpeg': 'JPEG',
                             '.webp': 'WEBP', '.bmp': 'BMP',
                             '.tif': 'TIFF', '.tiff': 'TIFF'}.get(ext, ext.lstrip('.').upper())
        try:
            import struct
            with open(full, 'rb') as f:
                head = f.read(32)
            if head[:8] == b'\x89PNG\r\n\x1a\n':
                w, h = struct.unpack('>II', head[16:24])
                info['w'], info['h'] = int(w), int(h)
            elif head[:2] == b'\xff\xd8':
                import subprocess as _sp
                r = _sp.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                             '-show_entries', 'stream=width,height', '-of', 'csv=p=0', full],
                            capture_output=True, text=True, timeout=15)
                parts = (r.stdout or '').strip().split(',')
                if len(parts) >= 2:
                    info['w'], info['h'] = int(parts[0]), int(parts[1])
        except Exception:
            pass
        info['frames'] = 1
        info['codec'] = ''
        _VDUR_CACHE[key] = info
        if len(_VDUR_CACHE) > 300:
            _VDUR_CACHE.clear()
        return info
    try:
        r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries',
                            'stream=codec_name,width,height,nb_frames,avg_frame_rate',
                            '-show_entries', 'format=duration,format_name',
                            '-of', 'json', full],
                           capture_output=True, text=True, timeout=20)
        d = json.loads(r.stdout or '{}')
        st = (d.get('streams') or [{}])[0]
        fmt = d.get('format') or {}
        info['w'] = int(st.get('width') or 0)
        info['h'] = int(st.get('height') or 0)
        info['frames'] = int(st.get('nb_frames') or 0)
        info['codec'] = (st.get('codec_name') or '').upper()
        num, _, den = (st.get('avg_frame_rate') or '0/0').partition('/')
        try:
            info['fps'] = round(float(num) / float(den), 2) if float(den or 0) else 0.0
        except (ValueError, ZeroDivisionError):
            info['fps'] = 0.0
        try:
            info['dur'] = int(round(float(fmt.get('duration') or 0)))
        except (TypeError, ValueError):
            info['dur'] = 0
        fmt_names = (fmt.get('format_name') or '').split(',')
        if 'mp4' in fmt_names:
            info['container'] = 'MP4'
        elif 'matroska' in fmt_names or 'webm' in fmt_names:
            info['container'] = 'MKV' if 'matroska' in fmt_names else 'WEBM'
        elif 'mov' in fmt_names:
            info['container'] = 'MOV'
        elif 'avi' in fmt_names:
            info['container'] = 'AVI'
        else:
            info['container'] = (fmt_names[0] or '').upper()
        if not info['frames'] and info['fps'] and info['dur']:
            info['frames'] = int(round(info['fps'] * info['dur']))
    except Exception:
        pass
    _VDUR_CACHE[key] = info
    if len(_VDUR_CACHE) > 300:
        _VDUR_CACHE.clear()
    return info


def video_sources():
    """Файлы (видео и картинки), уже лежащие во входном каталоге на стенде."""
    _vdirs()
    out = []
    try:
        for nm in sorted(os.listdir(VIDEO_IN)):
            if not nm.lower().endswith(VIDEO_EXT + PHOTO_IN_EXT):
                continue
            full = os.path.join(VIDEO_IN, nm)
            if not os.path.isfile(full):
                continue
            try:
                stt = os.stat(full)
            except OSError:
                continue
            out.append({'name': nm, 'bytes': stt.st_size, 'mtime': stt.st_mtime,
                        **_vmeta(full, stt.st_mtime)})
    except Exception:
        pass
    return out


def video_results(limit=12):
    """Готовые файлы (самые свежие сверху) и сколько их всего."""
    _vdirs()
    items = []
    try:
        for nm in os.listdir(VIDEO_OUT):
            if not nm.lower().endswith(('.mp4', '.png', '.jpg', '.jpeg', '.webp')):
                continue
            full = os.path.join(VIDEO_OUT, nm)
            if not os.path.isfile(full):
                continue
            try:
                stt = os.stat(full)
            except OSError:
                continue
            items.append({'name': nm, 'bytes': stt.st_size, 'mtime': stt.st_mtime,
                          **_vmeta(full, stt.st_mtime)})
    except Exception:
        pass
    items.sort(key=lambda x: x['mtime'], reverse=True)
    return items[:limit], len(items)


def _vframes(full):
    """Сколько кадров в файле (для прогресса). 0 = не знаю."""
    try:
        r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream=nb_frames,duration,avg_frame_rate',
                            '-show_entries', 'format=duration', '-of', 'json', full],
                           capture_output=True, text=True, timeout=20)
        d = json.loads(r.stdout or '{}')
        st = (d.get('streams') or [{}])[0]
        n = int(st.get('nb_frames') or 0)
        if n:
            return n
        num, _, den = (st.get('avg_frame_rate') or '0/0').partition('/')
        fps = float(num) / float(den) if float(den or 0) else 0.0
        dur = float(st.get('duration') or (d.get('format') or {}).get('duration') or 0)
        if fps and dur:
            return int(round(fps * dur))
    except Exception:
        pass
    return 0


def video_status():
    st = video_state()
    results, results_total = video_results()
    log = _vtail(st.get('log') or '', 20000)
    active = _unit_active(VIDEO_UNIT)
    done = 0
    total = int(st.get('frames_total') or 0)
    for m in re.finditer(r'кадров:\s*(\d+)', log):
        done = max(done, int(m.group(1)))
    if not total:
        # в начале лога есть строка «кадров в файле: N», но хвост лога её обрезает
        m = re.search(r'кадров в файле:\s*(\d+)', log)
        if m:
            total = int(m.group(1))
    limit = int(st.get('limit') or 0)
    if limit:
        total = min(total or limit, limit)
    if not total and done:
        total = done
    pct = int(round(done * 100.0 / total)) if total else 0
    name = st.get('name') or ''
    out = st.get('out') or ''
    state = 'idle'
    error = ''
    outfull = os.path.join(VIDEO_OUT, out) if out else ''
    fresh = False
    if outfull and os.path.isfile(outfull):
        # файл считается результатом этой задачи, только если он новее запуска
        try:
            fresh = os.path.getmtime(outfull) >= float(st.get('started') or 0) - 1
        except (OSError, TypeError, ValueError):
            fresh = False
    if active in ('active', 'activating', 'deactivating', 'start'):
        state = 'running'
    elif st.get('cancel') and 'записано' not in log:
        # остановили на середине: недописанный файл результатом не считаем
        state = 'stopped'
        error = 'обработка остановлена'
    elif fresh:
        state = 'done'
    elif not st:
        state = 'idle'
    elif st.get('cancel'):
        state = 'stopped'
        error = 'обработка остановлена'
    elif 'записано' in log or 'готово кадров' in log:
        # задача отработала, а файла нет — значит его убрали с диска
        state = 'idle'
    elif time.time() - float(st.get('started') or 0) < 120:
        # юнит ещё стартует: systemd-run отдаёт управление сразу
        state = 'running'
    else:
        state = 'error'
        tail = [ln for ln in log.strip().splitlines() if ln.strip()]
        error = '\n'.join(tail[-4:])[-600:] or 'задача закрылась без файла — смотри лог'
    sec = 0
    if st.get('started'):
        end = st.get('finished') or 0
        if not end and fresh and outfull:
            # задачу никто не закрывал — берём время файла, в state не пишем
            try:
                end = os.path.getmtime(outfull)
            except OSError:
                end = time.time()
        end = end or time.time()
        sec = max(0, int(end - st['started']))
    left = 0
    if state == 'running' and done and sec:
        left = int(max(0, sec / done * (total - done)))
    return {'state': state, 'name': name, 'out': out, 'done': done, 'total': total,
            'pct': pct if state == 'running' else (100 if state == 'done' else 0),
            'sec': sec, 'left': left, 'error': error, 'unit': active,
            'opts': st.get('opts') or {},
            'log': '\n'.join([ln for ln in log.strip().splitlines() if ln.strip()][-6:])[-800:],
            'sources': video_sources(), 'results': results,
            'split': video_split_status(),
            'results_total': results_total}


def video_start(req):
    """Запустить обработку на стенде (systemd-юнит, чтобы пережить рестарт панели)."""
    with VIDEO_LOCK:
        if _unit_active(VIDEO_UNIT) in ('active', 'activating', 'deactivating'):
            return False, 'уже идёт обработка, дождитесь конца'
        # от прошлой остановки мог остаться живой контейнер — он держит видеокарту
        _docker_kill_job(wait=15)
        if trainer_busy():
            return False, 'сейчас учится LoRA и видеокарта занята — дождитесь конца обучения'
        _vdirs()
        name = os.path.basename((req.get('name') or '').strip())
        is_photo = name.lower().endswith(PHOTO_IN_EXT)
        if not name or not name.lower().endswith(VIDEO_EXT + PHOTO_IN_EXT):
            return False, 'нужен видеофайл (mp4, mkv, mov, webm) или картинка (png, jpg)'
        if not os.path.isfile(os.path.join(VIDEO_IN, name)):
            return False, 'нет такого файла на стенде'
        scunet = bool(req.get('scunet', True))
        esrgan = bool(req.get('esrgan', True))
        rife = bool(req.get('rife', True))
        color = bool(req.get('color', False))
        try:
            tstab = float(req.get('tstab') or 0)
        except (TypeError, ValueError):
            tstab = 0.0
        tstab = max(0.0, min(0.9, tstab))
        try:
            csmooth = float(req.get('color_smooth') or 0.55)
            csat = float(req.get('color_sat') or 1.0)
        except (TypeError, ValueError):
            csmooth, csat = 0.55, 1.0
        csmooth = max(0.0, min(0.95, csmooth))
        csat = max(0.2, min(2.0, csat))
        if not (scunet or esrgan):
            return False, 'выбери хотя бы чистку или апскейл'
        try:
            scale = float(req.get('scale') or 2)
            crf = int(req.get('crf') or 18)
            limit = int(req.get('limit') or 0)
        except (TypeError, ValueError):
            return False, 'не разобрал числовые поля'
        scale = max(1.0, min(4.0, scale))
        crf = max(12, min(30, crf))
        limit = max(0, min(100000, limit))

        base = os.path.splitext(name)[0]
        tag = time.strftime('%Y%m%d-%H%M%S')
        log_path = os.path.join(VIDEO_LOG_DIR, 'ui-%s.log' % tag)

        segs_req = req.get('segments')
        if isinstance(segs_req, list) and segs_req:
            # сборка из кусков: отмеченные идут через нейросеть, остальные копией
            sta = os.stat(os.path.join(VIDEO_IN, name))
            meta = _vmeta(os.path.join(VIDEO_IN, name), sta.st_mtime)
            dur_src = float(meta.get('dur') or 0)
            fps_src = float(meta.get('fps') or 0) or 25.0
            segs = []
            for item in segs_req:
                try:
                    a = max(0.0, float(item.get('t0')))
                    b = float(item.get('t1'))
                except (TypeError, ValueError):
                    return False, 'не разобрал таймкоды кусков'
                if dur_src:
                    b = min(b, dur_src)
                if b - a < 0.1:
                    continue
                segs.append({'t0': round(a, 3), 't1': round(b, 3),
                             'enhance': bool(item.get('enhance'))})
            if not segs:
                return False, 'куски пустые — поправь таймкоды'
            color_fix = bool(req.get('color_fix'))
            if not any(s['enhance'] for s in segs) and not color_fix:
                return False, 'не отмечено ни одного куска и цвет не включён'
            out = '%s-split.mp4' % base
            if os.path.exists(os.path.join(VIDEO_OUT, out)):
                out = '%s-split-%s.mp4' % (base, time.strftime('%H%M%S'))
            plan = {'src': os.path.join(VIDEO_IN, name),
                    'out': os.path.join(VIDEO_OUT, out), 'segments': segs,
                    'opts': {'scunet': scunet, 'tstab': tstab, 'crf': crf, 'scale': 1,
                             'color': bool(req.get('color_fix'))}}
            try:
                with open(VIDEO_SPLIT_PLAN, 'w') as f:
                    json.dump(plan, f, ensure_ascii=False)
            except Exception as e:
                return False, 'не смог записать план кусков: %s' % e
            nframes = sum(int(round((s['t1'] - s['t0']) * fps_src))
                          for s in segs if s['enhance']) or 1
            log_path = os.path.join(VIDEO_LOG_DIR, 'ui-split-%s.log' % tag)
            cmd = ['sudo', '-n', 'systemd-run', '--unit=' + VIDEO_UNIT, '--collect',
                   '--no-block', '--property=Type=oneshot',
                   '--property=StandardOutput=append:' + log_path,
                   '--property=StandardError=append:' + log_path,
                   '/usr/bin/python3', VIDEO_SPLIT_RUN, VIDEO_SPLIT_PLAN]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            except Exception as e:
                return False, 'не смог запустить: %s' % e
            if r.returncode != 0:
                return False, ('запуск не удался: %s'
                               % ((r.stderr or r.stdout or '').strip()[-300:]))
            video_save({'name': name, 'out': out, 'started': time.time(), 'log': log_path,
                        'limit': 0, 'frames_total': nframes, 'mode': 'split',
                        'opts': {'split': True, 'segments': segs, 'crf': crf}})
            return True, None

        if is_photo:
            # Картинка: только раскраска (чистка/апскейл для одного кадра не нужны)
            if not color:
                return False, 'для картинки поставь галочку «раскрасить ч/б»'
            out = '%s-color.png' % base
            if os.path.exists(os.path.join(VIDEO_OUT, out)):
                out = '%s-color-%s.png' % (base, time.strftime('%H%M%S'))
            args = ['/work/code/color.py', '--in', '/work/in/' + name,
                    '--out', '/work/out/' + out, '--half',
                    '--sat', '%.2f' % csat]
            cmd = ['sudo', '-n', 'systemd-run', '--unit=' + VIDEO_UNIT, '--collect', '--no-block',
                   '--property=Type=oneshot',
                   '--property=StandardOutput=append:' + log_path,
                   '--property=StandardError=append:' + log_path,
                   'docker', 'run', '--rm', '--name', VIDEO_CONTAINER, '--gpus', 'all',
                   '-v', VIDEO_DIR + ':/work', '-w', '/work', VIDEO_IMAGE, 'python'] + args
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            except Exception as e:
                return False, 'не смог запустить: %s' % e
            if r.returncode != 0:
                return False, ('запуск не удался: %s' % ((r.stderr or r.stdout or '').strip()[-300:]))
            video_save({'name': name, 'out': out, 'started': time.time(), 'log': log_path,
                        'limit': 0, 'frames_total': 1,
                        'opts': {'photo': True, 'color': True, 'color_sat': csat}})
            return True, None

        out = '%s-enh.mp4' % base
        # не затираем прошлый результат: второй прогон получает метку времени
        if os.path.exists(os.path.join(VIDEO_OUT, out)):
            out = '%s-enh-%s.mp4' % (base, time.strftime('%H%M%S'))
        # сколько кадров будет обработано — сразу, иначе прогресс врёт до конца
        nframes = _vframes(os.path.join(VIDEO_IN, name))
        if limit and nframes:
            nframes = min(nframes, limit)

        args = ['/work/code/enh.py', '--in', '/work/in/' + name, '--out', '/work/out/' + out,
                '--scale', str(scale), '--crf', str(crf), '--half']
        if limit:
            args += ['--limit', str(limit)]
        if scunet:
            args += ['--scunet', '/work/weights/scunet_color_real_psnr.pth']
        if esrgan:
            args += ['--esrgan', '/work/weights/RealESRGAN_x4plus.pth']
        if rife:
            args += ['--rife', '/work/weights/rife49.pth', '--fps-mult', '3']
        if tstab > 0:
            args += ['--tstab', '%.2f' % tstab]
        if color:
            args += ['--color', '/work/weights/ddcolor_modelscope.pth',
                     '--color-smooth', '%.2f' % csmooth, '--color-sat', '%.2f' % csat]

        cmd = ['sudo', '-n', 'systemd-run', '--unit=' + VIDEO_UNIT, '--collect', '--no-block',
               '--property=Type=oneshot',
               '--property=StandardOutput=append:' + log_path,
               '--property=StandardError=append:' + log_path,
               'docker', 'run', '--rm', '--name', VIDEO_CONTAINER, '--gpus', 'all',
               '-v', VIDEO_DIR + ':/work', '-w', '/work', VIDEO_IMAGE, 'python'] + args
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except Exception as e:
            return False, 'не смог запустить: %s' % e
        if r.returncode != 0:
            return False, ('запуск не удался: %s' % ((r.stderr or r.stdout or '').strip()[-300:]))

        video_save({'name': name, 'out': out, 'started': time.time(), 'log': log_path,
                    'limit': limit, 'frames_total': nframes,
                    'opts': {'scunet': scunet, 'esrgan': esrgan, 'rife': rife,
                             'tstab': tstab, 'color': color,
                             'scale': scale, 'crf': crf, 'limit': limit}})
        return True, None


def _docker_kill_job(wait=0):
    """Убить контейнер задачи (если он есть). Возвращает True, если он был."""
    try:
        r = subprocess.run(['docker', 'rm', '-f', VIDEO_CONTAINER],
                           capture_output=True, text=True, timeout=60)
        had = r.returncode == 0 and VIDEO_CONTAINER in (r.stdout or '')
    except Exception:
        had = False
    if wait:
        deadline = time.time() + wait
        while time.time() < deadline:
            try:
                q = subprocess.run(['docker', 'ps', '-q', '--filter',
                                    'name=^%s$' % VIDEO_CONTAINER],
                                   capture_output=True, text=True, timeout=15)
                if not (q.stdout or '').strip():
                    break
            except Exception:
                break
            time.sleep(0.5)
    return had


def video_stop():
    """Остановить обработку.

    Порядок важен: сначала убиваем контейнер. Если начать с `systemctl stop`,
    он ждёт закрытия юнита (до 90 с), а контейнер тем временем продолжает
    считать на видеокарте — задача успевает дойти до конца.
    """
    killed = _docker_kill_job(wait=15)
    try:
        subprocess.run(['sudo', '-n', 'systemctl', 'stop', '--no-block',
                        VIDEO_UNIT + '.service'], capture_output=True, timeout=30)
    except Exception as e:
        if not killed:
            return False, 'не смог остановить: %s' % e
    st = video_state()
    # ffmpeg успевает создать файл и писать его по ходу: после остановки
    # это огрызок, а не результат — убираем, иначе он попадёт в ленту как готовый
    out = st.get('out') or ''
    if out and 'записано' not in _vtail(st.get('log') or '', 20000):
        full = os.path.join(VIDEO_OUT, out)
        if os.path.isfile(full):
            try:
                os.remove(full)
            except OSError:
                pass
    st['finished'] = time.time()
    st['cancel'] = True
    video_save(st)
    # если контейнер всё ещё жив — это уже настоящая ошибка, молчать нельзя
    try:
        q = subprocess.run(['docker', 'ps', '-q', '--filter',
                            'name=^%s$' % VIDEO_CONTAINER],
                           capture_output=True, text=True, timeout=15)
        if (q.stdout or '').strip():
            return False, 'контейнер ещё работает — проверь docker ps'
    except Exception:
        pass
    return True, None


def run_job(job):
    req = job['req']
    # Карта одна: во время обучения LoRA рисование гарантированно упрётся в память
    if trainer_busy():
        job['state'] = 'error'
        job['error'] = ('Сейчас учится LoRA и видеокарта занята. '
                        'Дождитесь окончания обучения на вкладке «обучение».')
        return
    job['state'] = 'running'
    job['started'] = time.time()
    prompt = req['prompt']

    if req.get('translate') and has_cyr(prompt):
        job['phase'] = 'перевод на английский'
        free_comfy()
        err0 = ensure_ollama(job)
        if err0:
            job['note'] = err0
        en, err = translate(prompt, req.get('translator'), video=(req.get('kind') == 'video'))
        if en:
            prompt = en
            job['prompt_en'] = en
        else:
            job['note'] = err
        job['phase'] = ''

    ok, why = guard_check(prompt)
    if not ok:
        job['state'] = 'error'
        job['error'] = why
        return

    if not req.get('keep_ollama'):
        job['phase'] = 'освобождаю видеопамять'
        unload_ollama()

    err0 = ensure_comfy(job)
    if err0:
        job['state'] = 'error'
        job['error'] = err0
        return

    job['phase'] = 'отправка в ComfyUI'
    seed = req['seed'] if req.get('seed') else int(time.time()) % 10 ** 12
    job['seed'] = seed
    if req.get('kind') == 'detail':
        ddet = req.get('detector') or 'hands'
        ddef = DETAIL_DEFAULTS.get(ddet, DETAIL_DEFAULTS['hands'])
        dneg = req.get('negative') or NEG_DEFAULT
        if ddet == 'hands' and 'extra fingers' not in dneg:
            dneg = dneg + ', ' + HAND_NEG_EXTRA
        wf = build_detail_workflow(prompt, dneg, req['file'], req['src'],
                                   req['steps'], req['cfg'], seed, req['denoise'],
                                   detector=ddet,
                                   upscale=bool(req.get('upscale')),
                                   size=(req['w'], req['h']),
                                   threshold=ddef['threshold'],
                                   dilation=ddef['dilation'],
                                   crop_factor=ddef['crop_factor'],
                                   pose=bool(req.get('pose')) and ddet == 'hands',
                                   pose_strength=float(req.get('pose_strength') or 0.9))
    elif req.get('kind') == 'redraw':
        wf = build_redraw_workflow(prompt, req.get('negative') or NEG_DEFAULT,
                                   req['file'], req['src'], req['steps'], req['cfg'],
                                   seed, req['denoise'],
                                   strength=req.get('strength') or 0.8,
                                   ctrl_type=req.get('ctrl_type') or
                                   'canny/lineart/anime_lineart/mlsd')
    elif req.get('kind') == 'inpaint':
        wf = build_inpaint_workflow(prompt, req.get('negative') or NEG_DEFAULT,
                                    req['file'], req['src'], req['mask_image'],
                                    req['steps'], req['cfg'], seed,
                                    req['denoise'], req['grow'])
    elif req.get('kind') == 'video':
        family = req.get('family')
        if family == 'wanflf':
            # Два кадра: загруженные или дорисованные здесь же, затем Wan сшивает переход
            start = req.get('start_image')
            end = req.get('end_image')
            ckpt = req.get('frame_model') or MODELS[0][2]
            if not start:
                job['phase'] = 'рисую первый кадр'
                start, err = render_frame(prompt, req['w'], req['h'], seed, ckpt, 'ui-start')
                if not start:
                    job['state'] = 'error'
                    job['error'] = err
                    return
            if not end:
                job['phase'] = 'рисую последний кадр'
                end, err = render_frame(req.get('end_prompt') or prompt,
                                        req['w'], req['h'], seed + 1, ckpt, 'ui-end')
                if not end:
                    job['state'] = 'error'
                    job['error'] = err
                    return
            job['phase'] = 'отправка в ComfyUI'
            job['start_image'] = start
            job['end_image'] = end
            wf = build_wan_flf_workflow(prompt, req.get('negative') or NEG_VIDEO,
                                        req['w'], req['h'], req['length'], req['fps'],
                                        req['steps'], req['cfg'], seed, req['file'],
                                        start, end)
        elif family == 'wan':
            wf = build_wan_workflow(prompt, req.get('negative') or NEG_VIDEO,
                                    req['w'], req['h'], req['length'], req['fps'],
                                    req['steps'], req['cfg'], seed, req['file'])
        else:
            wf = build_video_workflow(prompt, req.get('negative') or NEG_VIDEO,
                                      req['w'], req['h'], req['length'], req['fps'],
                                      req['steps'], req['cfg'], seed, req['file'])
    else:
        wf = build_workflow(prompt, req.get('negative') or NEG_DEFAULT,
                            req['w'], req['h'], req['steps'], req['cfg'], seed, req['file'])
    if req.get('lora') and req.get('kind') != 'video':
        if not lora_ok(req['lora']):
            job['state'] = 'error'
            job['error'] = 'нет такой LoRA: %s' % req['lora']
            return
        wf = apply_lora(wf, req['lora'], req.get('lora_weight') or 0.8)
        job['lora'] = req['lora']

    try:
        resp = post_json(API + '/prompt',
                         {'prompt': wf, 'client_id': uuid.uuid4().hex}, timeout=60)
    except urllib.error.HTTPError as e:
        detail = e.read().decode('utf-8', 'replace')[:1200]
        job['state'] = 'error'
        job['error'] = 'ComfyUI не принял задачу: %s %s' % (e.code, detail)
        return
    except Exception as e:
        job['state'] = 'error'
        job['error'] = 'ComfyUI не принял задачу (%s). Часто это нехватка видеопамяти.' % e
        return
    if 'prompt_id' not in resp:
        job['state'] = 'error'
        job['error'] = 'ОТКАЗ: ' + json.dumps(resp, ensure_ascii=False)[:1200]
        return

    pid = resp['prompt_id']
    job['prompt_id'] = pid
    job['phase'] = 'рисую' if req.get('kind') != 'video' else 'снимаю видео'
    if req.get('kind') == 'detail':
        job['phase'] = ('ищу %s и перерисовываю'
                        % ('лицо' if req.get('detector') == 'face' else 'кисти'))
    elif req.get('kind') == 'redraw':
        job['phase'] = 'перерисовываю по контуру'
    t0 = time.time()
    limit = 1800 if req.get('kind') == 'video' else 900
    lost = 0          # подряд идёт «задачи нет ни в очереди, ни в истории»
    silent = 0        # подряд идёт «ComfyUI не отвечает»
    while True:
        time.sleep(0.7)
        if job.get('cancelled'):
            job['state'] = 'error'
            job['error'] = 'остановлено'
            return
        try:
            h = get_json(API + '/history/' + pid, timeout=30)
            silent = 0
        except Exception:
            silent += 1
            if silent >= 30:
                job['state'] = 'error'
                job['error'] = 'ComfyUI не отвечает — проверь, что контейнер comfyui жив'
                return
            continue
        if pid not in h:
            # ComfyUI мог перезапуститься и потерять задачу — тогда ждать 15 минут незачем
            try:
                q = get_json(API + '/queue', timeout=10)
                ids = [x[1] for x in (q.get('queue_running') or []) + (q.get('queue_pending') or [])]
            except Exception:
                ids = None
            if ids is not None and pid not in ids:
                lost += 1
                if lost >= 6:
                    job['state'] = 'error'
                    job['error'] = ('задача потерялась: ComfyUI перезапустился и забыл очередь. '
                                    'Нажми ещё раз')
                    return
            else:
                lost = 0
        if pid in h:
            st = h[pid].get('status', {})
            if not st.get('completed'):
                msgs = [m[1] for m in st.get('messages', []) if m[0] == 'execution_error']
                txt = json.dumps(msgs, ensure_ascii=False)
                job['state'] = 'error'
                if 'OutOfMemoryError' in txt or 'ran out of memory' in txt:
                    job['error'] = ('ComfyUI потерял видеокарту (это не нехватка памяти). Нажми ещё раз: '
                                    'стенд сам пересоздаёт контейнер раз в 15 минут, или вручную '
                                    'docker compose up -d --force-recreate comfyui в каталоге стенда (UI_COMPOSE_DIR).')
                else:
                    job['error'] = 'ошибка ComfyUI: ' + txt[:1200]
                return
            files = []
            for node in h[pid].get('outputs', {}).values():
                files += node.get('images', [])
            if not files:
                job['state'] = 'error'
                job['error'] = 'ComfyUI не вернул файл'
                return
            f = files[0]
            if f.get('subfolder'):
                job['image'] = f['subfolder'] + '/' + f['filename']
            else:
                job['image'] = f['filename']
            job['sec'] = round(time.time() - t0, 1)
            job['state'] = 'done'
            job['phase'] = ''
            add_history({'ts': int(time.time()), 'ru': req['prompt'],
                         'en': job.get('prompt_en') or prompt,
                         'kind': req.get('kind', 'image'),
                         'family': req.get('family'),
                         'model': req['label'], 'w': req['w'], 'h': req['h'],
                         'steps': req['steps'], 'seed': seed,
                         'length': req.get('length'), 'fps': req.get('fps'),
                         'start': job.get('start_image'), 'end': job.get('end_image'),
                          'source': req.get('source_file'),
                          'denoise': req.get('denoise'), 'grow': req.get('grow'),
                          'detector': req.get('detector'), 'strength': req.get('strength'),
                          'upscale': req.get('upscale'),
                          'ctrl_type': req.get('ctrl_type') if req.get('kind') == 'redraw' else None,
                          'file': job['image'], 'sec': job['sec']})
            return
        if time.time() - t0 > limit:
            job['state'] = 'error'
            job['error'] = 'таймаут %d минут' % (limit // 60)
            return


def worker():
    while True:
        job = JOBQ.get()
        try:
            run_job(job)
        except Exception as e:
            job['state'] = 'error'
            job['error'] = 'внутренняя ошибка: %s' % e
        finally:
            job['finished'] = time.time()
            JOBQ.task_done()


# --------------------------------------- замороженная база для документов
# Отдельный стек на стенде: внутри контейнера ollama с qwen3:8b и поиском по
# документам, сверху Open WebUI. Сеть замкнутая, наружу ничего не публикуется.
# Управление — тот же скрипт, что и в терминале (stand-doc.sh).
DOC_SCRIPT = os.environ.get('DOC_SCRIPT', os.path.join(PANEL_HOME, 'stand-doc.sh'))
DOC_IMAGE = os.environ.get('DOC_IMAGE', 'doc:2')
DOC_LOG = os.environ.get('DOC_LOG', os.path.join(PANEL_HOME, '.ui-doc.log'))
DOC_UI_IP = os.environ.get('DOC_UI_IP', '172.23.0.10')
# Готовая команда туннеля для браузера: подставь свой хост и пользователя.
DOC_TUNNEL = os.environ.get('DOC_TUNNEL') or 'ssh -N -L 3100:%s:8080 <user>@<host>' % DOC_UI_IP
DOC_ACTIONS = ('ui-on', 'ui-off', 'wipe', 'build')

# Счётчик «загружено / проиндексировано». Файлы берём из webui.db (таблица file),
# число кусков — из chroma.sqlite3 (collections → segments → embeddings).
# Внутри контейнера, только имена и числа: содержимое документов не читается.
DOC_FILES_PY = r'''
import json, os, sqlite3, sys
base = os.environ.get('DOC_BASE') or '/app/backend/data'
res = {'ok': True, 'files': [], 'chunks': {}, 'error': ''}
try:
    db = sqlite3.connect('file:%s?mode=ro' % os.path.join(base, 'webui.db'), uri=True)
    cur = db.cursor()
    cur.execute('select id, filename, meta, data, created_at from file order by created_at')
    for fid, name, meta, data, created in cur.fetchall():
        coll = ''
        for src in (meta, data):
            try:
                coll = (json.loads(src or '{}') or {}).get('collection_name') or coll
            except Exception:
                pass
        if not coll:
            coll = 'file-%s' % fid   # так его называет сам Open WebUI при загрузке
        res['files'].append({'id': fid, 'name': name, 'collection': coll,
                             'created': created or 0})
    db.close()
except Exception as e:
    res['ok'] = False
    res['error'] = 'webui.db: %s' % e
try:
    p = os.path.join(base, 'vector_db', 'chroma.sqlite3')
    if os.path.exists(p):
        ch = sqlite3.connect('file:%s?mode=ro' % p, uri=True)
        cc = ch.cursor()
        # В chroma 1.x записи лежат в metadata-сегменте, в старых — в обоих,
        # поэтому считаем различные embedding_id по всем сегментам коллекции.
        cc.execute('select c.name, count(distinct e.embedding_id) from collections c '
                   'join segments s on s.collection = c.id '
                   'left join embeddings e on e.segment_id = s.id group by c.name')
        for name, n in cc.fetchall():
            res['chunks'][name] = int(n or 0)
        ch.close()
    else:
        res['error'] = (res['error'] + ' нет vector_db/chroma.sqlite3').strip()
except Exception as e:
    res['error'] = (res['error'] + ' chroma: %s' % e).strip()
for f in res['files']:
    n = res['chunks'].get(f['collection'], 0)
    f['chunks'] = n
    f['indexed'] = n > 0
used = set(f['collection'] for f in res['files'])
res['chunks'] = dict((k, v) for k, v in res['chunks'].items() if k in used)
res['total'] = len(res['files'])
res['indexed_total'] = sum(1 for f in res['files'] if f['indexed'])
sys.stdout.write(json.dumps(res, ensure_ascii=False))
'''

_DOC_FILES_CACHE = {'t': 0.0, 'v': None}
_DOC_FILES_TTL = 15.0

_DOC_LOCK = threading.Lock()
_DOC_JOB = {'proc': None, 'action': '', 'started': 0, 'finished': 0, 'rc': None}


def _doc_tail(limit=6000):
    try:
        with open(DOC_LOG, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - limit))
            return f.read().decode('utf-8', 'replace')
    except Exception:
        return ''


def _doc_tick():
    """Отметить, что фоновая задача закончилась (без этого статус вечно «идёт»)."""
    with _DOC_LOCK:
        p = _DOC_JOB['proc']
        if p is None:
            return
        rc = p.poll()
        if rc is not None and not _DOC_JOB['finished']:
            _DOC_JOB['finished'] = time.time()
            _DOC_JOB['rc'] = rc


def _doc_busy():
    _doc_tick()
    with _DOC_LOCK:
        return bool(_DOC_JOB['proc'] is not None and _DOC_JOB['finished'] == 0)


def _doc_run(action):
    """Запустить действие в фоне: скрипт долгий, панель показывает его вывод."""
    if action not in DOC_ACTIONS:
        return False, 'неизвестное действие'
    if not os.path.isfile(DOC_SCRIPT):
        return False, 'нет скрипта %s' % DOC_SCRIPT
    if _doc_busy():
        with _DOC_LOCK:
            return False, 'уже идёт: %s' % _DOC_JOB['action']
    try:
        log = open(DOC_LOG, 'w')
    except Exception as e:
        return False, 'не открыть лог: %s' % e
    try:
        proc = subprocess.Popen(['bash', DOC_SCRIPT, action],
                                stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
    except Exception as e:
        log.close()
        return False, 'не запустить скрипт: %s' % e
    log.close()
    with _DOC_LOCK:
        _DOC_JOB.update({'proc': proc, 'action': action, 'started': time.time(),
                         'finished': 0, 'rc': None})
    return True, None


def _docker_out(args, timeout=20):
    try:
        r = subprocess.run(['docker'] + args, capture_output=True, text=True,
                           timeout=timeout)
        return (r.stdout or '').strip()
    except Exception:
        return ''


def _doc_containers():
    out = _docker_out(['ps', '-a', '--filter', 'name=doc-',
                       '--format', '{{.Names}}|{{.State}}|{{.Status}}'])
    res = {}
    for line in out.splitlines():
        parts = line.split('|')
        if len(parts) >= 3 and parts[0]:
            res[parts[0]] = {'state': parts[1], 'status': parts[2]}
    return res


def _doc_health():
    try:
        with urllib.request.urlopen('http://%s:8080/health' % DOC_UI_IP,
                                    timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _doc_files():
    """Сколько документов загружено и у скольких есть индекс (кэш 15 с)."""
    empty = {'ok': False, 'error': '', 'files': [], 'total': 0, 'indexed_total': 0}
    now = time.time()
    if _DOC_FILES_CACHE['v'] is not None and now - _DOC_FILES_CACHE['t'] < _DOC_FILES_TTL:
        return _DOC_FILES_CACHE['v']
    try:
        r = subprocess.run(['docker', 'exec', '-i', 'doc-webui', 'python3', '-'],
                           input=DOC_FILES_PY, capture_output=True, text=True,
                           timeout=25)
        out = (r.stdout or '').strip()
        if not out:
            lines = [l for l in (r.stderr or '').strip().splitlines() if l.strip()]
            empty['error'] = (lines[-1] if lines else 'пусто')[:200]
            val = empty
        else:
            val = json.loads(out)
    except Exception as e:
        empty['error'] = str(e)[:200]
        val = empty
    _DOC_FILES_CACHE['t'] = now
    _DOC_FILES_CACHE['v'] = val
    return val


def doc_status():
    img = _docker_out(['image', 'inspect', '-f', '{{.Id}}|{{.Size}}', DOC_IMAGE])
    image = {'present': False, 'id': '', 'bytes': 0}
    if img:
        iid, _, size = img.partition('|')
        try:
            nbytes = int(size)
        except ValueError:
            nbytes = 0
        image = {'present': True, 'id': iid.replace('sha256:', '')[:12], 'bytes': nbytes}
    net = _docker_out(['network', 'inspect', '-f', '{{.Internal}}', 'docnet'])
    conts = _doc_containers()
    files = ({'ok': False, 'error': 'панель не поднята', 'files': [],
              'total': 0, 'indexed_total': 0}
             if conts.get('doc-webui', {}).get('state') != 'running'
             else _doc_files())
    _doc_tick()
    with _DOC_LOCK:
        action = _DOC_JOB['action']
        started = _DOC_JOB['started']
        finished = _DOC_JOB['finished']
        rc = _DOC_JOB['rc']
    busy = bool(started and not finished)
    secs = 0
    if started:
        secs = int((finished or time.time()) - started)
    return {'image': image,
            'image_name': DOC_IMAGE,
            'container_names': ['doc-ollama', 'doc-webui'],
            'net_name': 'docnet',
            'ui_ip': DOC_UI_IP,
            'containers': conts,
            'files': files,
            'net': {'present': bool(net), 'internal': net.lower() == 'true'},
            'health': _doc_health(),
            'busy': busy,
            'action': action,
            'secs': secs,
            'rc': rc,
            'log': _doc_tail(),
            'tunnel': DOC_TUNNEL,
            'url': 'http://127.0.0.1:3100/'}


# ------------------------------------------- железо: загрузка и температуры
# Всё, что здесь нужно, читается без прав администратора: psutil даёт процессор,
# память и датчики через /sys/class/hwmon (k10temp — процессор, nvme — диск,
# amdgpu — встроенная графика), видеокарту спрашиваем у nvidia-smi.
_SYS_LOCK = threading.Lock()
_SYS_CACHE = {'t': 0.0, 'v': None}
_SYS_TTL = 3.0
_SYS_CPU = {'total': 0.0, 'busy': 0.0}   # прошлый снимок /proc/stat
_SYS_IO = {'name': '', 'read': 0, 'write': 0, 'io': 0, 't': 0.0}   # прошлый снимок диска


def _sys_cpu_percent():
    """Загрузка процессора по /proc/stat.

    psutil.cpu_percent(interval=None) здесь не годится: сервер — ThreadingHTTPServer,
    каждый запрос идёт в новом потоке, а psutil держит прошлый снимок в потоке →
    каждый раз «первый вызов» и честный 0.0. Поэтому считаем дельту сами.
    """
    try:
        with open('/proc/stat', 'r') as f:
            parts = f.readline().split()
    except OSError:
        return None
    if len(parts) < 8 or parts[0] != 'cpu':
        return None
    vals = []
    for x in parts[1:9]:
        try:
            vals.append(float(x))
        except ValueError:
            vals.append(0.0)
    idle = vals[3] + vals[4]          # idle + iowait
    total = sum(vals)
    with _SYS_LOCK:
        d_total = total - _SYS_CPU['total']
        d_busy = (total - idle) - _SYS_CPU['busy']
        first = _SYS_CPU['total'] == 0.0
        _SYS_CPU['total'] = total
        _SYS_CPU['busy'] = total - idle
    if first or d_total <= 0:
        return None
    return round(max(0.0, min(100.0, 100.0 * d_busy / d_total)), 1)


def _sys_out(cmd, timeout=5):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           timeout=timeout)
        return p.stdout.decode('utf-8', 'replace').strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def _sys_read(path):
    try:
        with open(path, 'r') as f:
            return f.read().strip()
    except OSError:
        return ''


def _sys_cpu_name():
    for line in _sys_read('/proc/cpuinfo').splitlines():
        if line.startswith('model name'):
            return line.split(':', 1)[1].strip()
    return ''


def _sys_temps():
    """датчики по чипам: {'k10temp': [('Tctl', 41.1)], 'nvme': [('Composite', 53.8)]}"""
    out = {}
    if psutil is None:
        return out
    try:
        for chip, lst in psutil.sensors_temperatures().items():
            out[chip] = [[x.label or '', round(float(x.current), 1)]
                         for x in lst if x.current is not None]
    except Exception:  # noqa: BLE001
        pass
    return out


def _sys_temp(temps, chip, label=None):
    for name, cur in temps.get(chip, []):
        if label is None or name == label:
            return cur
    return None


# --- питание процессора (счётчик RAPL) и температура SATA-диска (модуль drivetemp)
#
# У материнской платы и памяти датчиков тока нет вообще: nct6779 умеет только
# напряжения, температуры и обороты вентиляторов. Поэтому «сколько ест вся
# машина» система не знает — измеряются ровно два потребителя: процессор
# (накопительный счётчик энергии в /sys/class/powercap) и видеокарта (NVML).

RAPL_DIR = '/sys/class/powercap/intel-rapl:0'
RAPL_ENERGY = RAPL_DIR + '/energy_uj'
RAPL_MAX = RAPL_DIR + '/max_energy_range_uj'
SATA_DISK = os.environ.get('UI_SATA_DEVICE', '/sys/block/sda')
# Отдельный диск под модели (необязателен): пусто — строки про него не будет
SATA_MOUNT = os.environ.get('UI_SATA_MOUNT', '')
# Какие ещё диски показывать на вкладке «железо»: «метка=точка,метка=точка»
DISK_CANDIDATES = [x for x in os.environ.get('UI_DISKS', '/home').split(',') if x.strip()]

_RAPL = {'t': None, 'e': None, 'w': None}
_RAPL_LOCK = threading.Lock()


def _rapl_ok():
    """Открыт ли счётчик энергии процессора: по умолчанию он только для root."""
    try:
        with open(RAPL_ENERGY, 'rb') as f:
            f.read(1)
        return True
    except OSError:
        return False


def _cpu_watts(now):
    """Сколько ватт ест процессор: приращение счётчика RAPL делить на время.

    Счётчик копит микро-джоули, поэтому ватты — это разница энергии между
    двумя запросами, поделённая на прошедшее время. Запросы приходят раз в
    ~3 с (кэш ответа), этого хватает и для простоя, и для нагрузки; короткие
    интервалы сглаживаются, иначе цифра дёргается на единицы ватт.
    """
    raw = _sys_read(RAPL_ENERGY)
    if not raw.isdigit():
        return None
    e = int(raw)
    with _RAPL_LOCK:
        prev_t, prev_e, prev_w = _RAPL['t'], _RAPL['e'], _RAPL['w']
        _RAPL['t'], _RAPL['e'] = now, e
        if prev_e is None or prev_t is None:
            return None
        dt = now - prev_t
        if dt <= 0.2:
            return prev_w
        de = e - prev_e
        if de < 0:                       # счётчик перешёл через максимум
            mx = _sys_read(RAPL_MAX)
            if not mx.isdigit():
                return prev_w
            de += int(mx)
        w = de / 1e6 / dt
        _RAPL['w'] = w if prev_w is None else 0.6 * prev_w + 0.4 * w
        return round(_RAPL['w'], 1)


def _drivetemp_loaded(temps):
    return any(chip.lower().startswith('drivetemp') for chip in temps)


def _disk_temp(temps, chip_prefix='drivetemp'):
    """Температура диска из отдельного чипа hwmon: drivetemp (SATA)."""
    for chip, lst in temps.items():
        if not chip.lower().startswith(chip_prefix):
            continue
        vals = [cur for _, cur in lst if cur is not None]
        if vals:
            return round(max(vals), 1)
    return None


def _sys_tops():
    """Кто грузит машину прямо сейчас: топ по процессору и по памяти.

    Через ps, а не через psutil: у psutil проценты считаются от момента запуска
    процесса, и после systemd-старта панель показывала бы старые средние.
    """
    out = _sys_out(['ps', '-eo', 'pcpu,pmem,comm', '--no-headers', '--sort=-pcpu'], timeout=8)
    rows = []
    for line in out.splitlines()[:200]:
        f = line.split(None, 2)
        if len(f) < 3:
            continue
        try:
            rows.append((float(f[0]), float(f[1]), f[2].strip()))
        except ValueError:
            continue
    rows = [r for r in rows if r[2] not in ('ps', 'sh', 'ssh', 'bash', 'grep')]
    cpu = [{'name': r[2], 'cpu': round(r[0], 1), 'mem': round(r[1], 1)}
           for r in sorted(rows, key=lambda x: -x[0])[:5] if r[0] >= 1.0]
    mem = [{'name': r[2], 'cpu': round(r[0], 1), 'mem': round(r[1], 1)}
           for r in sorted(rows, key=lambda x: -x[1])[:5] if r[1] >= 0.5]
    return cpu, mem


def _sys_gpu_apps():
    """Кто занимает видеопамять: имя процесса и сколько МБ."""
    out = _sys_out(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                    '--format=csv,noheader,nounits'])
    apps = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) < 3:
            continue
        name = os.path.basename(parts[1].split()[0]) if parts[1] else '?'
        try:
            mem = float(parts[2])
        except ValueError:
            mem = 0.0
        apps.append({'pid': parts[0], 'name': name, 'mem': mem})
    apps.sort(key=lambda a: -a['mem'])
    return apps


def _sys_gpu():
    out = _sys_out(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,'
                    'memory.total,temperature.gpu,power.draw',
                    '--format=csv,noheader,nounits'])
    if not out:
        return {'present': False}
    parts = [p.strip() for p in out.splitlines()[0].split(',')]
    if len(parts) < 6:
        return {'present': False}

    def num(x):
        try:
            return float(x)
        except ValueError:
            return None

    return {'present': True, 'name': parts[0], 'util': num(parts[1]),
            'mem_used': num(parts[2]), 'mem_total': num(parts[3]),
            'temp': num(parts[4]), 'power': num(parts[5])}


def _sys_mounts():
    """Точки монтирования из /proc/self/mountinfo (set)."""
    out = set()
    for line in _sys_read('/proc/self/mountinfo').splitlines():
        f = line.split()
        if len(f) > 4:
            out.add(f[4].replace('\\040', ' '))
    return out


def _sys_disks():
    """Место на дисках: список берётся из UI_DISKS.

    Формат — «метка=точка,метка=точка» (по умолчанию система и домашний).
    Точка монтирования проверяется по mountinfo: если отдельного диска нет,
    psutil вернул бы цифры корня и второй «диск» был бы фантомом, поэтому
    такой кандидат отбрасывается.
    """
    out = []
    seen = set()
    mounts = _sys_mounts()
    cands = [('система', '/')]
    for pair in DISK_CANDIDATES:
        if '=' in pair:
            label, mount = pair.split('=', 1)
            cands.append((label.strip(), mount.strip()))
        else:
            cands.append((os.path.basename(pair.rstrip('/')) or pair, pair.strip()))
    for label, mount in cands:
        if mount != '/' and mount not in mounts:
            continue
        try:
            u = psutil.disk_usage(mount) if psutil else None
        except OSError:
            u = None
        if u is None:
            continue
        key = (u.total, u.used)
        if key in seen:  # /home и /home/docker — одна и та же файловая система
            continue
        seen.add(key)
        out.append({'label': label, 'mount': mount, 'total': u.total,
                    'used': u.used, 'percent': round(u.percent, 1),
                    'free': u.free})
    return out


def _sys_diskstats():
    """Строки /proc/diskstats по имени устройства."""
    out = {}
    for ln in _sys_read('/proc/diskstats').splitlines():
        f = ln.split()
        if len(f) >= 14:
            out[f[2]] = f
    return out


def _sys_root_disk(stats):
    """Имя диска (не раздела), на котором лежит корень: nvme0n1, sda…

    Брать номер устройства из mountinfo для корня НЕЛЬЗЯ: btrfs подставляет
    свой анонимный номер (0:35), поэтому вторым путём идём через источник
    тома (/dev/nvme0n1p2) и /sys/class/block.
    """
    majmin = ''
    src = ''
    for line in _sys_read('/proc/self/mountinfo').splitlines():
        f = line.split()
        if len(f) > 4 and f[4] == '/':
            majmin = f[2]
            if '-' in f:
                i = f.index('-')
                if len(f) > i + 2:
                    src = f[i + 2]
            break
    links = []
    if majmin and majmin != '0:0':
        links.append(os.path.realpath('/sys/dev/block/' + majmin))
    if src.startswith('/dev/'):
        links.append(os.path.realpath('/sys/class/block/' + os.path.basename(src)))
    for link in links:
        if not os.path.exists(link):
            continue
        name = os.path.basename(link)
        parent = os.path.basename(os.path.dirname(link))
        # /sys/devices/.../block/nvme0n1/nvme0n1p2 → диск nvme0n1
        for cand in (parent if parent and parent != 'block' else name, name):
            if cand in stats:
                return cand
    # последний шанс: самый «читаемый» настоящий диск
    best, best_rd = '', 0
    for name, f in stats.items():
        if name.startswith(('loop', 'ram', 'dm-', 'zram')) or name[-1].isdigit():
            continue
        try:
            rd = int(f[5])
        except (IndexError, ValueError):
            continue
        if rd > best_rd:
            best, best_rd = name, rd
    return best


def _sys_disk_io(now):
    """Загрузка диска: сколько читает/пишет и какую долю времени занят.

    Считается по /proc/diskstats разницей с прошлым снимком: поле sectors —
    всегда по 512 байт, io_ticks — миллисекунды работы устройства.
    """
    stats = _sys_diskstats()
    name = _sys_root_disk(stats)
    line = stats.get(name)
    if not name or not line:
        return None
    try:
        rd, wr, io = int(line[5]), int(line[9]), int(line[12])
    except (IndexError, ValueError):
        return None
    with _SYS_LOCK:
        prev = dict(_SYS_IO)
        dt = now - prev['t'] if prev['t'] else 0.0
        busy = None
        read_b = write_b = 0.0
        if prev['name'] == name and dt >= 0.2:
            read_b = max(0, rd - prev['read']) * 512.0 / dt
            write_b = max(0, wr - prev['write']) * 512.0 / dt
            busy = round(min(100.0, 100.0 * max(0, io - prev['io']) / (dt * 1000.0)), 1)
        _SYS_IO.update({'name': name, 'read': rd, 'write': wr, 'io': io, 't': now})
    return {'device': name, 'busy': busy,
            'read': round(read_b / 1048576.0, 1),
            'write': round(write_b / 1048576.0, 1)}


def _cpu_temp(temps):
    """Температура процессора: Intel (coretemp Package) или AMD (k10temp Tctl)."""
    for chip, label in (('coretemp', 'Package id 0'), ('k10temp', 'Tctl')):
        v = _sys_temp(temps, chip, label)
        if v is not None:
            return v
    for chip in ('coretemp', 'k10temp'):
        vals = [c for _, c in temps.get(chip, [])]
        if vals:
            return round(max(vals), 1)
    return None


_SYS_HIST = []          # [(cpu%, gpu%, mem%)] для полосок истории
_SYS_HIST_MAX = 60      # 60 снимков по 3 с = 3 минуты


def system_status():
    now = time.time()
    with _SYS_LOCK:
        if _SYS_CACHE['v'] is not None and now - _SYS_CACHE['t'] < _SYS_TTL:
            return _SYS_CACHE['v']
    temps = _sys_temps()
    if psutil is None:
        mem = swap = {'total': 0, 'used': 0, 'available': 0, 'percent': 0.0}
        cpu_pct = None
        threads = cores = None
        boot = 0
    else:
        m = psutil.virtual_memory()
        s = psutil.swap_memory()
        mem = {'total': m.total, 'used': m.used, 'available': m.available,
               'percent': round(m.percent, 1)}
        swap = {'total': s.total, 'used': s.used, 'available': s.free,
                'percent': round(s.percent, 1)}
        cpu_pct = _sys_cpu_percent()
        threads = psutil.cpu_count()
        cores = psutil.cpu_count(logical=False)
        boot = psutil.boot_time()
    try:
        load = [round(x, 2) for x in os.getloadavg()]
    except OSError:
        load = [0.0, 0.0, 0.0]
    gpu = _sys_gpu()
    if gpu.get('present'):
        gpu['apps'] = _sys_gpu_apps()
    top_cpu, top_mem = _sys_tops()
    watts = _cpu_watts(now)
    data = {'cpu': {'name': _sys_cpu_name(), 'percent': cpu_pct,
                    'threads': threads, 'cores': cores, 'load': load,
                    'temp': _cpu_temp(temps),
                    'watts': watts, 'watts_ok': watts is not None or _rapl_ok()},
            'mem': mem,
            'swap': swap,
            'gpu': gpu,
            'io': _sys_disk_io(now),
            'disks': _sys_disks(),
            'sata': {'mount': SATA_MOUNT,
                     'mounted': bool(SATA_MOUNT) and SATA_MOUNT in _sys_mounts(),
                     'model': _sys_read(SATA_DISK + '/device/model') if SATA_MOUNT else None,
                     'temp': _disk_temp(temps) if SATA_MOUNT else None,
                     'sensor': _drivetemp_loaded(temps) if SATA_MOUNT else None},
            'ssd': {'model': _sys_read('/sys/class/nvme/nvme0/model'),
                    'temp': _sys_temp(temps, 'nvme', 'Composite')},
            'igpu': _sys_temp(temps, 'amdgpu', 'edge'),
            'uptime': int(now - boot) if boot else 0,
            'top_cpu': top_cpu,
            'top_mem': top_mem,
            'psutil': psutil is not None,
            'ts': int(now)}
    with _SYS_LOCK:
        _SYS_CACHE['t'] = now
        _SYS_CACHE['v'] = data
        _SYS_HIST.append((round(cpu_pct or 0.0, 1),
                          round((gpu.get('util') or 0.0), 1),
                          round(mem.get('percent') or 0.0, 1)))
        del _SYS_HIST[:-_SYS_HIST_MAX]
    return data


if psutil is not None:  # готовим первый снимок /proc/stat, иначе первый ответ без цифры
    _sys_cpu_percent()
_sys_disk_io(time.time())   # то же для диска: первый ответ без дельты


# ------------------------------------------------------------------- html/js

HOME = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>картинки и видео на стенде</title>
<style>
:root{--bg:#16181d;--pan:#1e2128;--line:#2c3038;--tx:#e6e8ec;--dim:#9aa1ad;--acc:#5b9dd9}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{display:grid;grid-template-columns:380px 1fr;gap:18px;padding:18px;max-width:1600px;margin:0 auto}
@media(max-width:900px){.wrap{grid-template-columns:1fr}}
h1{font-size:16px;font-weight:600;margin:0 0 14px;color:var(--dim);letter-spacing:.02em}
.card{background:var(--pan);border:1px solid var(--line);border-radius:10px;padding:14px}
label{display:block;font-size:12px;color:var(--dim);margin:12px 0 4px}
label:first-of-type{margin-top:0}
textarea,select,input[type=number],input[type=text]{width:100%;background:#14161a;color:var(--tx);
border:1px solid var(--line);border-radius:7px;padding:8px 9px;font:inherit;resize:vertical}
input[type=file]{width:100%;font-size:12px;color:var(--dim)}
textarea{min-height:86px}
.row{display:flex;gap:10px}.row>*{flex:1}
button{font:inherit;cursor:pointer;border-radius:7px;border:1px solid var(--line);
background:#262a33;color:var(--tx);padding:9px 14px}
button:hover{background:#2f343e}
button.go{background:var(--acc);border-color:var(--acc);color:#0d1117;font-weight:600;width:100%;margin-top:14px;padding:11px}
button.go:disabled{opacity:.5;cursor:default}
button.mini{padding:5px 9px;font-size:12px}
.hint{color:var(--dim);font-size:12px;margin-top:8px}
details{margin-top:12px}summary{cursor:pointer;color:var(--dim);font-size:12px}
.stage{display:flex;flex-direction:column;gap:12px}
.frame{background:var(--pan);border:1px solid var(--line);border-radius:10px;
min-height:320px;display:flex;align-items:center;justify-content:center;padding:12px;position:relative}
.frame img,.frame video{max-width:100%;max-height:70vh;border-radius:6px;display:block}
.frame .empty{color:var(--dim)}
.bar{height:4px;background:#14161a;border-radius:3px;overflow:hidden;margin-top:10px}
.bar i{display:block;height:100%;width:0;background:var(--acc);transition:width .3s}
.status{font-size:13px;color:var(--dim);display:flex;justify-content:space-between;align-items:center;gap:10px}
.status b{color:var(--tx);font-weight:600}
.warn{color:#e0b072}
.err{color:#e08a80;white-space:pre-wrap;font-size:13px}
.gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(112px,1fr));gap:8px;margin-top:4px}
.gal figure{margin:0;background:#14161a;border:1px solid var(--line);border-radius:8px;
overflow:hidden;cursor:pointer;position:relative}
.gal img,.gal video{width:100%;height:96px;object-fit:cover;display:block}
.gal figcaption{font-size:11px;color:var(--dim);padding:4px 5px;white-space:nowrap;
overflow:hidden;text-overflow:ellipsis}
.gal .del{position:absolute;top:3px;right:3px;padding:1px 6px;font-size:13px;line-height:1.2;
background:rgba(10,12,16,.72);color:#e6e8ec;border:1px solid rgba(255,255,255,.18);
border-radius:5px;opacity:.45;transition:opacity .15s}
.gal figure:hover .del{opacity:1}
.meta{font-size:12px;color:var(--dim);margin-top:6px;word-break:break-word}
.q{display:flex;flex-direction:column;gap:6px;margin-top:8px}
.qrow{display:flex;align-items:center;gap:10px;background:#14161a;border:1px solid var(--line);
border-radius:7px;padding:7px 9px;font-size:12.5px}
.qrow.run{border-color:#3c5f80}
.qrow .n{color:var(--dim);min-width:18px;text-align:right}
.qrow .w{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.qrow .t{color:var(--dim);white-space:nowrap;font-variant-numeric:tabular-nums}
.qrow .mini2{height:4px;background:#242830;border-radius:2px;overflow:hidden;width:90px}
.qrow .mini2 i{display:block;height:100%;width:0;background:var(--acc);transition:width .4s}
.qfoot{font-size:12px;color:var(--dim);margin-top:7px}
.vitem{background:#14161a;border:1px solid var(--line);border-radius:8px;padding:8px}
.vitem video{width:100%;border-radius:6px;display:block}
.vitem .vn{font-size:12.5px;margin-top:6px;word-break:break-all}
.vitem .vs{font-size:12px;color:var(--dim)}
.vitem .vact{display:flex;gap:6px;margin-top:8px;align-items:stretch}
.vitem .vact a{flex:1;text-align:center;padding:6px 9px;font-size:12px;border-radius:7px;
border:1px solid var(--line);background:#262a33;color:var(--tx);text-decoration:none}
.vitem .vact a:hover{background:#2f343e}
.vitem .vact button{flex:0 0 auto;padding:5px 11px}
.howto{background:#14161a;border:1px solid var(--line);border-radius:8px;padding:9px 11px;
font-size:12.5px;color:var(--dim);line-height:1.65;margin-bottom:12px}
.howto b{color:var(--tx);font-weight:600}
.vinfo{font-size:12px;color:var(--dim);margin-top:7px;min-height:16px;line-height:1.5}
.vinfo b{color:var(--tx);font-weight:600}
.head{grid-column:1/-1}
.tabs{display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap}
.tab{background:#20242c;border:1px solid var(--line);color:var(--dim);padding:7px 12px;font-size:13px;white-space:nowrap}
.tab.on{background:var(--acc);border-color:var(--acc);color:#0d1117;font-weight:600}
.hw{display:grid;gap:9px;max-width:920px}
#hwcard{grid-column:1/-1}
.hwrow{display:grid;grid-template-columns:150px 1fr 340px;gap:11px;align-items:center;font-size:13px}
.hwrow .lbl{color:var(--dim)}
.hwrow .val{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums;font-size:12px}
.hwbar{height:9px;background:#242830;border-radius:5px;overflow:hidden}
.hwbar i{display:block;height:100%;width:0;background:var(--acc);transition:width .5s}
.hwrow.warn .hwbar i{background:#e5a50a}
.hwsub{margin-top:10px;padding-top:9px;border-top:1px solid var(--line);font-size:12.5px;color:var(--dim);line-height:1.7}
.hwsub b{color:var(--tx);font-weight:600}
.hwkick{margin-top:12px;padding-top:10px;border-top:1px solid var(--line);font-size:12px;color:var(--dim)}
.hwspark{display:flex;gap:16px;margin-top:6px;max-width:920px}
.hwspark>div{flex:1;min-width:0}
.hwspark .cap{display:flex;justify-content:space-between;font-size:11.5px;color:var(--dim);margin-bottom:4px}
.hwspark canvas{width:100%;height:44px;display:block;border-radius:6px;background:#14161a}
.log{background:#14161a;border:1px solid var(--line);border-radius:8px;padding:10px;margin-top:10px;
height:52vh;overflow:auto;font-size:13.5px;white-space:pre-wrap;word-break:break-word}
.log .empty{color:var(--dim)}
.log .u{color:#9ecbff;margin:0 0 10px}
.log .a{margin:0 0 10px}
.log b{font-weight:600;color:var(--dim);margin-right:4px}
.log details{margin-top:6px;color:var(--dim);font-size:12px}
.log img{max-width:100%;border-radius:8px;margin-top:8px;display:block;cursor:zoom-in}
.log .ph{color:var(--dim);font-size:12px;margin-top:6px}
</style></head><body>
<div class="wrap">
 <div class="head">
  <h1>картинки, видео и чат на стенде</h1>
  <div class="tabs">
   <button class="tab on" id="tabimg" type="button">картинки и видео</button>
   <button class="tab" id="tabvis" type="button">что на картинке</button>
   <button class="tab" id="tabchat" type="button">чат</button>
   <button class="tab" id="tabtrn" type="button">обучение стиля</button>
   <button class="tab" id="tabvid" type="button">улучшение видео</button>
   <button class="tab" id="tabgrd" type="button">цвет и яркость</button>
   <button class="tab" id="tabdoc" type="button">документы</button>
   <button class="tab" id="tabhw" type="button">железо</button>
  </div>
 </div>
 <div>
  <div class="card" id="gradecard" style="display:none">
   <div class="howto">
    <b>цвет и яркость готового видео</b><br>
    сначала картинка подгоняется под палитру исходника, потом работают ручки.<br>
    гамма — светлее/темнее в средних тонах; контраст — разница между светлым и тёмным;
    чёрный — глубина теней; насыщенность; чистка теней — сколько цвета остаётся в самых тёмных местах
    (100 — цвет гасится почти в ноль); синее/коричневое — сдвиг оттенка; резкость — лёгкая подточка.<br>
    пример кадра пересчитывается сам после движения ползунка, «применить ко всему видео» считает файл целиком.
   </div>
   <div class="row">
    <div><label for="gf">файл (после нейросети)</label><select id="gf"></select></div>
    <div><label for="gs">палитра от</label><select id="gs"></select></div>
   </div>
   <div class="row">
    <div><label for="g_gamma">гамма <span id="g_gamma_v">90</span></label><input id="g_gamma" type="range" min="60" max="120" value="90" style="width:100%"></div>
    <div><label for="g_contrast">контраст <span id="g_contrast_v">115</span></label><input id="g_contrast" type="range" min="80" max="160" value="115" style="width:100%"></div>
   </div>
   <div class="row">
    <div><label for="g_black">чёрный <span id="g_black_v">12</span></label><input id="g_black" type="range" min="0" max="60" value="12" style="width:100%"></div>
    <div><label for="g_sat">насыщенность <span id="g_sat_v">72</span></label><input id="g_sat" type="range" min="20" max="130" value="72" style="width:100%"></div>
   </div>
   <div class="row">
    <div><label for="g_shadow">чистка теней <span id="g_shadow_v">90</span></label><input id="g_shadow" type="range" min="0" max="100" value="90" style="width:100%"></div>
    <div><label for="g_sharp">резкость <span id="g_sharp_v">0</span></label><input id="g_sharp" type="range" min="0" max="60" value="0" style="width:100%"></div>
   </div>
   <div class="row">
    <div><label for="g_cb">синее <span id="g_cb_v">+1.6</span></label><input id="g_cb" type="range" min="-5" max="5" step="0.2" value="1.6" style="width:100%"></div>
    <div><label for="g_cr">коричневое <span id="g_cr_v">-2.2</span></label><input id="g_cr" type="range" min="-5" max="5" step="0.2" value="-2.2" style="width:100%"></div>
   </div>
   <div class="hint">кадр для примера: <input id="g_frame" type="number" min="0" value="20" style="width:80px"> <button class="mini" id="g_prev">показать кадр</button> <span id="g_info"></span></div>
   <div class="frame" id="gfr" style="min-height:100px"><span class="empty">здесь будет пример кадра</span></div>
   <button class="go" id="g_apply">применить ко всему видео</button>
   <div class="status"><span id="g_state">готов</span><span id="g_time"></span></div>
   <div class="err" id="g_err"></div>
   <div class="status" style="margin-top:14px"><b>готовые файлы</b></div>
   <div class="q" id="g_list"></div>
  </div>
   <div class="card" id="imgcard">
   <label for="p">что нарисовать (можно по-русски)</label>
   <textarea id="p" placeholder="рыжий кот на подоконнике, утренний свет"></textarea>
   <div class="row">
    <div><label for="m">модель</label><select id="m"></select></div>
    <div><label for="s">размер</label><select id="s"></select></div>
   </div>
   <div class="row">
    <div><label for="lo">стиль (LoRA)</label><select id="lo"></select></div>
    <div><label for="lw">сила стиля <span id="lwv">80 %</span></label><input id="lw" type="range" min="0" max="130" value="80" style="width:100%"></div>
   </div>
   <div class="hint" id="lohint" style="display:none"></div>
   <label for="trm">перевод делает</label><select id="trm"></select>
   <div class="row">
    <div><label for="st">шагов</label><input id="st" type="number" min="1" max="80" value="28"></div>
    <div><label for="sd">seed (пусто = любой)</label><input id="sd" type="text" placeholder=""></div>
   </div>
   <div class="row" id="vrow" style="display:none">
    <div><label for="ln" id="lnl">кадров (8k+1)</label><input id="ln" type="number" min="9" max="161" step="8" value="49"></div>
    <div><label for="fp">кадров в секунду</label><input id="fp" type="number" min="8" max="30" value="24"></div>
   </div>
   <div id="flfrow" style="display:none">
    <div class="row">
     <div><label for="fs">первый кадр</label><input id="fs" type="file" accept="image/png,image/jpeg"></div>
     <div><label for="fe">последний кадр</label><input id="fe" type="file" accept="image/png,image/jpeg"></div>
    </div>
    <label for="ep">последний кадр: что на нём (если файл не загружен)</label>
    <textarea id="ep" style="min-height:52px" placeholder="кот ближе к камере, тот же подоконник"></textarea>
    <div class="hint" id="flfhint">пустые поля = нарисую кадры сам по тексту</div>
   </div>
   <details><summary>дополнительно</summary>
    <label for="n">чего избегать (негативный промпт)</label>
    <textarea id="n" style="min-height:52px"></textarea>
    <label style="display:flex;gap:8px;align-items:center;margin-top:12px">
     <input id="tr" type="checkbox" checked style="width:auto"> переводить русский текст на английский</label>
    <label style="display:flex;gap:8px;align-items:center;margin-top:6px">
     <input id="ko" type="checkbox" style="width:auto"> не выгружать ollama (быстрее, но риск нехватки памяти)</label>
    <label style="display:flex;gap:8px;align-items:center;margin-top:6px">
     <input id="hup" type="checkbox" style="width:auto"> починка рук и лица: увеличить картинку в 1,5 раза</label>
     <label style="display:flex;gap:8px;align-items:center;margin-top:6px">
      <input id="pz" type="checkbox" checked style="width:auto"> починка рук: ставить пальцы по скелету (ControlNet)</label>
    <div class="row" style="align-items:flex-end;margin-top:6px">
     <div><label for="rs">по контуру: сила перерисовки</label><input id="rs" type="range" min="20" max="95" value="60" style="width:100%"></div>
     <div><label for="rst">по контуру: держать контур</label><input id="rst" type="range" min="20" max="120" value="80" style="width:100%"></div>
     <div><label for="rt">что держим</label><select id="rt">
       <option value="canny/lineart/anime_lineart/mlsd">контур (canny)</option>
       <option value="depth">глубина</option>
       <option value="openpose">поза</option>
       <option value="hed/pidi/scribble/ted">мягкие линии (hed)</option>
       <option value="normal">нормали</option>
       <option value="tile">плитка (детализация)</option>
     </select></div>
    </div>
   </details>
   <button class="go" id="go">нарисовать</button>
   <div class="hint" id="hint"></div>
   <div class="hint" id="bgbar" style="display:none"></div>
   <div class="hint"><button class="mini" id="inp">перерисовать область</button> <button class="mini" id="hands" title="найти кисти детектором и перерисовать их">починить руки</button> <button class="mini" id="face" title="найти лица детектором и перерисовать их">починить лицо</button> <button class="mini" id="redraw" title="перерисовать кадр целиком, держа контур исходника">перерисовать по контуру</button> <button class="mini" id="free">освободить видеопамять</button> <button class="mini" id="stop" style="display:none">остановить</button></div>
  </div>
  <div class="card" id="chatctl" style="display:none">
    <label for="cm">модель</label>
    <select id="cm"></select>
    <label for="cctx">окно контекста</label>
    <select id="cctx">
     <option value="4096">4 096 — самый быстрый, короткие ответы</option>
     <option value="8192">8 192</option>
     <option value="16384" selected>16 384 — обычная работа</option>
     <option value="32768">32 768 — длинные тексты, больше видеопамяти</option>
     <option value="65536">65 536 — предел модели, может не влезть в карту</option>
    </select>
    <label style="display:flex;gap:8px;align-items:center;margin-top:14px;color:var(--tx)"><input id="cnoth" type="checkbox" checked style="width:auto"> не размышлять (быстрее)</label>
   <label style="display:flex;gap:8px;align-items:center;margin-top:8px;color:var(--tx)"><input id="cpic" type="checkbox" checked style="width:auto"> рисовать, если попрошу («нарисуй …»)</label>
    <div class="hint">Ctrl+Enter — отправить. Модель живёт на той же видеокарте, что рисует: чат и генерация одновременно мешают друг другу. Напиши «нарисуй …» — панель сама нарисует картинку и покажет её в разговоре. Разговор сохраняется в этом браузере (кнопка «очистить разговор» его стирает).</div>
   <div class="hint"><button class="mini" id="cclr">очистить разговор</button> <button class="mini" id="cfree">выгрузить модель</button></div>
  </div>
  <div class="card" id="traincard" style="display:none">
   <label for="tph">фотографии (15–30 штук, крупно, с разных сторон)</label>
   <input id="tph" type="file" accept="image/png,image/jpeg,image/webp" multiple>
   <div class="hint" id="tphhint">пока ничего не добавлено</div>
   <div class="gal" id="tphlist" style="margin-top:8px"></div>
   <label for="tcap">что на фотографиях (подпись для обучения, одна на весь набор)</label>
   <textarea id="tcap" style="min-height:56px" maxlength="400" placeholder="sovpioner, red pioneer neckerchief, white shirt, photo"></textarea>
   <div class="hint" id="tcaphint">пусто — подпись панель напишет сама; своими словами можно описать одежду и обстановку</div>
   <div class="hint"><button class="mini" id="tcapgo">сохранить подпись</button></div>
   <label for="tname">как назвать стиль</label>
   <input id="tname" type="text" placeholder="pioner" maxlength="40">
   <details><summary>дополнительно</summary>
    <div class="row">
     <div><label for="tsteps">шагов обучения</label><input id="tsteps" type="number" min="100" max="6000" step="50" value="1500"></div>
     <div><label for="tdim">подробность</label><select id="tdim"><option value="8">8 — легче, ~30 МБ</option><option value="16" selected>16 — лучше, ~114 МБ</option></select></div>
    </div>
    <label for="tbase">на какой модели учить</label>
    <select id="tbase"><option value="juggernaut" selected>Juggernaut XI (реализм)</option><option value="realvis">RealVis XL v5</option></select>
    <div class="hint">1500 шагов на этой видеокарте — примерно 45 минут. Рисовать в это время нельзя: карта занята.</div>
   </details>
   <button class="go" id="tgo">учить стиль</button>
   <div class="status"><span id="tstate">готов</span><span id="ttime"></span></div>
   <div class="bar"><i id="tpb"></i></div>
   <div class="err" id="terr"></div>
   <div class="hint"><button class="mini" id="tstop" style="display:none">остановить обучение</button> <button class="mini" id="tclear">убрать все фотографии</button></div>
   <div class="status" style="margin-top:14px"><b>готовые стили</b><span id="tlcnt"></span></div>
   <div class="q" id="tloras"></div>
   <div class="hint">Готовый стиль сам появится в списке «стиль (LoRA)» на вкладке рисования.</div>
  </div>
  <div class="card" id="vidcard" style="display:none">
   <div class="howto">
    <b>как это делается</b><br>
    1. выбери файл в списке ниже — или загрузи свой через «выбрать файл»<br>
    2. оставь галочки как есть и нажми «улучшить видео»<br>
    3. готовый файл появится внизу: ▶ — посмотреть здесь, <b>скачать</b> — забрать себе, <b>×</b> — удалить со стенда<br>
    оригинал со стенда удаляется кнопкой «удалить оригинал» под списком файлов<br>
    чёрно-белое (видео или картинка) красится галочкой «раскрасить ч/б»
   </div>
   <label for="vfile">загрузить свой файл — видео или картинка (до 400 МБ; большие лучше сразу положить на стенд)</label>
   <input id="vfile" type="file" accept="video/mp4,video/quicktime,video/x-matroska,video/webm,image/png,image/jpeg,image/webp">
   <div class="hint" id="vfhint">…или выбери файл, который уже лежит на стенде:</div>
   <div style="display:flex;gap:8px;align-items:stretch">
    <select id="vsrc" style="flex:1"><option value="">— нет файлов —</option></select>
    <button class="mini" id="vdel" type="button" title="удалить выбранный оригинал со стенда">удалить оригинал</button>
   </div>
   <div class="vinfo" id="vinfo"></div>
   <div class="hint" style="margin-top:10px"><button class="mini" id="vsplit">разбить на куски по качеству</button> <span id="vspnote"></span></div>
   <label style="display:flex;gap:8px;align-items:center;margin-top:6px;color:var(--tx)">
    <input id="vcfix" type="checkbox" checked style="width:auto"> поправить цвет и яркость (только там, где кусок испорчен)</label>
   <div id="vsegs" style="display:none">
    <div class="status"><b>куски</b><span id="vsegcnt"></span></div>
    <div id="vseglist"></div>
    <div class="hint">таймкоды можно править руками (мм:сс.д). галочка — кусок пойдёт через нейросеть; неотмеченные копируются без перекодирования.</div>
    <div class="hint"><button class="mini" id="vsegmarks">отметить плохие</button> <button class="mini" id="vsegnone">снять все</button></div>
    <button class="go" id="vsegrun">улучшить отмеченные куски</button>
   </div>
   <label style="display:flex;gap:8px;align-items:center;margin-top:14px;color:var(--tx)">
    <input id="vsc" type="checkbox" checked style="width:auto"> чистить блоки и шум (SCUNet)</label>
   <label style="display:flex;gap:8px;align-items:center;margin-top:6px;color:var(--tx)">
    <input id="vrife" type="checkbox" checked style="width:auto"> дорисовать промежуточные кадры (плавнее, втрое больше кадров)</label>
   <label style="display:flex;gap:8px;align-items:center;margin-top:6px;color:var(--tx)">
    <input id="vstab" type="checkbox" checked style="width:auto"> убрать дрожание кадров (сглаживание в статике)</label>
   <label style="display:flex;gap:8px;align-items:center;margin-top:6px;color:var(--tx)">
    <input id="vcolor" type="checkbox" style="width:auto"> раскрасить ч/б (чёрно-белое видео или картинка)</label>
   <div class="row" style="margin-top:12px">
    <div><label for="vscale">растянуть</label><select id="vscale">
     <option value="1">как есть</option><option value="2" selected>в 2 раза</option><option value="4">в 4 раза</option></select></div>
    <div><label for="vlimit">только первых кадров (0 = все)</label><input id="vlimit" type="number" min="0" step="100" value="0"></div>
   </div>
   <button class="go" id="vgo">улучшить видео</button>
   <div class="status"><span id="vstate">готов</span><span id="vtime"></span></div>
   <div class="bar"><i id="vpb"></i></div>
   <div class="err" id="verr"></div>
   <div class="hint"><button class="mini" id="vstop" style="display:none">остановить</button></div>
   <div class="hint" id="vlog"></div>
   <div class="status" style="margin-top:14px"><b>готовые файлы</b><span id="vrcnt"></span></div>
   <div class="q" id="vrlist"></div>
   <div class="hint">Считает видеокарта: 640x360 — около 0,55 кадра в секунду (минута такого видео ≈ 15 минут), 1080p — 3–13 с на кадр. Во время обработки рисовать и обучать нельзя: карта одна.</div>
  </div>
  <div class="card" id="doccard" style="display:none">
   <div class="howto">
    <b>замороженная база для документов</b><br>
    отдельный стек: внутри ollama с qwen3:8b для чата и bge-m3 для поиска по загруженным документам,
    сверху Open WebUI. Сеть замкнутая, в интернет из контейнеров хода нет, наружу порты не публикуются.<br>
    1. нажми «поднять» и дождись надписи «панель поднята» (первый раз до двух минут)<br>
    2. на ноутбуке выполни строку ниже, потом открой <b>http://127.0.0.1:3100/</b> — пароля нет<br>
    3. документы и история живут в томе: «погасить» их не трогает, «снести всё» удаляет навсегда; сам образ при этом остаётся — заново ничего не скачивается<br>
    4. схемы Visio (.vsdx) читаются: подписи фигур, вложенность и связи «кто → куда»; старый .vsd не читается — сохрани его в Visio как .vsdx
   </div>
   <div class="status"><b id="dstate">не проверено</b><span id="dtime"></span></div>
   <div class="vinfo" id="dinfo"></div>
   <div class="status" style="margin-top:12px"><b>документы в базе</b><span id="dfiles"></span></div>
   <div class="q" id="dflist"></div>
   <label for="dtun">туннель с ноутбука</label>
   <div style="display:flex;gap:8px;align-items:stretch">
    <input id="dtun" readonly style="flex:1;min-width:0;font-family:monospace;font-size:12px">
    <button class="mini" id="dcopy" type="button">копировать</button>
   </div>
   <div class="row" style="margin-top:12px">
    <button class="go" id="dstart" type="button">поднять</button>
    <button class="mini" id="dstop" type="button">погасить</button>
   </div>
   <div class="hint" id="dbuildrow" style="display:none">
    образа на стенде нет: <button class="mini" id="dbuild" type="button">собрать заново</button>
    <span id="dbuildnote">6 минут, модель берётся с диска стенда</span>
   </div>
   <div class="hint" style="margin-top:10px">
    <button class="mini" id="dwipe" type="button">снести всё</button>
    <span id="dwnote">контейнеры, документы и история удалятся навсегда; образ остаётся, качать заново не надо</span>
   </div>
   <div class="err" id="derr"></div>
   <div class="hint" id="dlog" style="white-space:pre-wrap;font-family:monospace;font-size:11.5px"></div>
  </div>
  <div class="card" id="viscard" style="display:none">
   <div class="howto">
    <b>что на картинке</b><br>
    загрузи фото — модель со зрением опишет кадр так, чтобы по описанию можно было нарисовать похожий.<br>
    по-русски — описание словами; по-английски — готовый промпт для SDXL (рисует лучше по-английски).<br>
    кнопка «в поле промпта» переносит ответ на вкладку рисования.
   </div>
   <label for="qphoto">фотография (png, jpeg, webp; до 25 МБ)</label>
   <input id="qphoto" type="file" accept="image/png,image/jpeg,image/webp">
   <div class="hint" id="qhint">файл не выбран</div>
   <div class="frame" id="qfr" style="min-height:80px"><span class="empty">здесь будет превью</span></div>
   <div class="row">
    <div><label for="qm">модель</label><select id="qm"></select></div>
    <div><label for="qlang">язык ответа</label><select id="qlang"><option value="en" selected>английский</option><option value="ru">русский</option></select></div>
   </div>
   <label for="qwhat">что важно описать (не обязательно)</label>
   <input id="qwhat" type="text" maxlength="300" placeholder="например: одежда, свет, ракурс">
   <button class="go" id="qgo">описать</button>
   <div class="status"><span id="qstate">готов</span><span id="qtime"></span></div>
   <div class="err" id="qerr"></div>
   <label for="qout">описание</label>
   <textarea id="qout" style="min-height:120px" placeholder="здесь будет ответ модели"></textarea>
   <div class="hint"><button class="mini" id="qcopy" type="button">копировать</button> <button class="mini" id="qprompt" type="button">в поле промпта</button></div>
  </div>
 </div>
 <div class="stage">
  <div class="card" id="chatlog" style="display:none">
   <div class="status"><b>разговор</b><span id="cstate"></span></div>
   <div class="log" id="clog"><span class="empty">здесь будет ответ</span></div>
   <label for="cq">вопрос</label>
   <textarea id="cq" style="min-height:70px" placeholder="спроси у модели что-нибудь"></textarea>
   <button class="go" id="csend">спросить</button>
  </div>
  <div class="card" id="qcard" style="display:none">
   <div class="status"><b id="qh">очередь</b><span id="qs"></span></div>
   <div class="q" id="qlist"></div>
  </div>
  <div class="card" id="framecard">
   <div class="status"><span id="ph">готов</span><span id="tm"></span></div>
   <div class="bar"><i id="pb"></i></div>
   <div class="frame" id="fr"><span class="empty">здесь появится картинка или видео</span></div>
   <div id="mbar" style="display:none">
    <div class="row" style="align-items:flex-end">
     <div><label for="br">кисть</label><input id="br" type="range" min="10" max="300" value="60" style="width:100%"></div>
     <div><label for="dn">сила перерисовки</label><input id="dn" type="range" min="10" max="100" value="60" style="width:100%"></div>
     <div><label for="gw">расширение</label><input id="gw" type="number" min="0" max="64" value="8" style="width:100%"></div>
    </div>
    <div class="row" style="margin-top:8px">
     <button class="mini" id="mkgo">перерисовать</button>
     <button class="mini" id="mker">ластик</button>
     <button class="mini" id="mkclr">сбросить маску</button>
     <button class="mini" id="mkx">отмена</button>
    </div>
    <div class="hint" id="mkhint">закрась то, что надо перерисовать</div>
   </div>
   <div class="meta" id="mt"></div>
   <div class="err" id="er"></div>
  </div>
  <div class="card" id="histcard">
   <div class="status"><b>последние</b><span><span id="cnt"></span> <button class="mini" id="clr">очистить ленту</button></span></div>
   <div class="gal" id="gal"></div>
  </div>
 </div>
  <div class="card" id="hwcard" style="display:none">
   <div class="howto">
    <b>что сейчас с машиной</b><br>
    загрузка процессора, памяти, видеокарты и дисков и их температуры.
    Обновляется само, раз в три секунды.
   </div>
   <div class="hw" id="hwbody"><span class="empty">считаю…</span></div>
   <div class="hwkick">последние 3 минуты</div>
   <div class="hwspark">
    <div><div class="cap"><span>процессор</span><span id="hscpu">—</span></div><canvas id="hccpu"></canvas></div>
    <div><div class="cap"><span>видеокарта</span><span id="hsgpu">—</span></div><canvas id="hcgpu"></canvas></div>
    <div><div class="cap"><span>память</span><span id="hsmem">—</span></div><canvas id="hcmem"></canvas></div>
   </div>
   <div class="status"><span id="hwstate">—</span><span id="hwtime"></span></div>
   <div class="err" id="hwerr"></div>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
let CFG=null, CUR=null, T0=0, WS=null, PID=null, timer=null;
let UP={start:null, end:null};

function fmt(x){return x.toLocaleString('ru-RU')}
function esc(s){return (s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}

async function boot(){
  CFG=await (await fetch('/api/config')).json();
  const im=CFG.models||[], vm=CFG.video_models||[];
  let html='';
  if(im.length) html+='<optgroup label="картинки">'+im.map(m=>
    `<option value="${esc(m.key)}">${esc(m.label)}</option>`).join('')+'</optgroup>';
  if(vm.length) html+='<optgroup label="видео">'+vm.map(m=>
    `<option value="${esc(m.key)}">${esc(m.label)}</option>`).join('')+'</optgroup>';
  $('m').innerHTML=html;
  $('trm').innerHTML=(CFG.translators||[]).map(t=>`<option value="${esc(t.name)}"${
    t.name===CFG.default_translator?' selected':''}>${esc(t.label)}</option>`).join('');
  $('cm').innerHTML=(CFG.translators||[]).map(t=>`<option value="${esc(t.name)}"${
    t.name===CFG.default_translator?' selected':''}>${esc(t.label)}</option>`).join('');
  /* окно контекста: список из сервера, выбор помнит браузер */
  if(CFG.ctx_choices&&$('cctx')){
    $('cctx').innerHTML=CFG.ctx_choices.map(v=>`<option value="${v}"${
      v===(CFG.ctx_default||16384)?' selected':''}>${v.toLocaleString('ru-RU')}${v===(CFG.ctx_default||16384)?' — обычная работа':''}</option>`).join('');
  }
  try{ const c=localStorage.getItem(CTX_KEY);
    if(c&&$('cctx')&&[...$('cctx').options].some(o=>o.value===c)) $('cctx').value=c }catch(e){}
  if($('cctx')) $('cctx').onchange=()=>{ try{ localStorage.setItem(CTX_KEY,$('cctx').value) }catch(e){} };
  $('m').onchange=applyModel;
  /* стиль (LoRA): список берётся с диска стенда, выбор помнит браузер */
  const lo=CFG.loras||[];
  $('lo').innerHTML='<option value="">без стиля (обычная модель)</option>'+
    lo.map(x=>`<option value="${esc(x.file)}">${esc(x.label)}</option>`).join('');
  try{ const sv=localStorage.getItem(LORA_KEY); if(sv&&lo.some(x=>x.file===sv)) $('lo').value=sv }catch(e){}
  $('lo').onchange=()=>{ loraHint(); try{ localStorage.setItem(LORA_KEY,$('lo').value) }catch(e){} };
  $('lw').oninput=()=>{ $('lwv').textContent=(Number($('lw').value)||0)+' %' };
  loraHint();
  applyModel();
  watchWs();
  loadHistory();
  startQueuePoll();
  loadChat(); renderChat();
  /* если страницу закрыли посреди рисования — досматриваем задачу */
  CHAT.filter(m=>m.job && !m.image).forEach(m=>watchPic(m.job,m));
  try{
    const t=localStorage.getItem(TAB_KEY);
    if(t==='chat') setMode('chat'); else if(t==='train') setMode('train');
    else if(t==='video') setMode('video'); else if(t==='grade') setMode('grade');
    else if(t==='doc') setMode('doc'); else if(t==='hw') setMode('hw'); else setMode('img');
  }catch(e){ setMode('img') }
  try{ const nm=localStorage.getItem(TRAIN_KEY); if(nm&&!$('tname').value) $('tname').value=nm }catch(e){}
}
function allM(){ return (CFG.models||[]).concat(CFG.video_models||[]) }
function curM(){ return allM().find(x=>x.key===$('m').value) }
function isVideo(m){ return !!m && m.kind==='video' }
function applyModel(){
  const m=curM(); if(!m) return;
  const vid=isVideo(m);
  if($('lo')){ $('lo').disabled=vid; }
  if(typeof loraHint==='function') loraHint();
  const fam=vid?(m.family||'ltxv'):null;
  const d=vid?((CFG.video_defaults||{})[fam]||{}):{};
  const sizes=vid?(((CFG.video_sizes||{})[fam])||[]):(CFG.sizes||[]);
  const def=vid?(d.size||'768x512'):'1024x1024';
  $('s').innerHTML=sizes.map(s=>`<option value="${s[0]}"${
    s[0]===def?' selected':''}>${esc(s[1])}</option>`).join('');
  if(m.steps) $('st').value=m.steps;
  if(vid){
    $('ln').value=d.length||49; $('fp').value=d.fps||24;
    if(!$('n').value.trim()) $('n').value=CFG.negative_video||'';
  } else if(!$('n').value.trim()){
    $('n').value=CFG.negative||'';
  }
  $('vrow').style.display=vid?'':'none';
  $('flfrow').style.display=(vid && fam==='wanflf')?'':'none';
  const lnl=$('lnl'); if(lnl) lnl.textContent = fam==='ltxv'?'кадров (8k+1)':'кадров (4k+1)';
  if(!(vid && fam==='wanflf') && $('ep')) $('ep').value='';
  $('hint').textContent=vid?(((CFG.video_hint||{})[fam])||''):(CFG.image_hint||'');
  $('go').textContent=vid?'снять видео':'нарисовать';
}
function loraHint(){
  const f=$('lo').value, x=(CFG.loras||[]).find(y=>y.file===f), el=$('lohint');
  if(!f || !x){ el.style.display='none'; el.textContent=''; return }
  el.style.display='';
  el.textContent='LoRA: '+x.file+(x.note?' — '+x.note:'')+
    (isVideo(curM())?' · к видео не применяется':'');
}
function comfyWsUrl(){
  /* адрес ComfyUI берём из /api/config; если он локальный для сервера,
     в браузере подставляем имя хоста, по которому открыта панель */
  let host=location.hostname, port=8188;
  try{
    const u=new URL((CFG&&CFG.comfy)||'');
    if(u.hostname&&u.hostname!=='127.0.0.1'&&u.hostname!=='localhost') host=u.hostname;
    if(u.port) port=u.port;
  }catch(e){}
  return `ws://${host}:${port}/ws?clientId=ui-${Date.now()}`;
}
function watchWs(){
  try{
    WS=new WebSocket(comfyWsUrl());
    WS.onmessage=ev=>{
      let m; try{m=JSON.parse(ev.data)}catch(e){return}
      if(m.type==='progress'&&m.data){
        const pc=Math.round(100*m.data.value/m.data.max);
        QPID=m.data.prompt_id; QPCT=pc;
        if(!PID) PID=QPID;
        if(m.data.prompt_id===PID){
          $('pb').style.width=pc+'%';
          $('tm').textContent=`${m.data.value}/${m.data.max} шагов`;
        }
      }
    };
    WS.onclose=()=>{WS=null; setTimeout(watchWs,4000)};
  }catch(e){WS=null}
}

async function upload(which, inp){
  const f=inp.files && inp.files[0];
  if(!f) return;
  const b=new Uint8Array(await f.arrayBuffer());
  let s=''; for(let i=0;i<b.length;i+=0x8000) s+=String.fromCharCode.apply(null,b.subarray(i,i+0x8000));
  let r;
  try{
    r=await fetch('/api/upload',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name:f.name,data:btoa(s)})});
  }catch(e){ $('er').textContent='загрузка не удалась: '+e; return }
  const j=await r.json();
  if(!r.ok){ $('er').textContent=j.error||('HTTP '+r.status); return }
  UP[which]=j.name;
  $('flfhint').textContent='первый: '+(UP.start||'нарисую по тексту')+
    ' · последний: '+(UP.end||'нарисую по описанию');
  $('er').textContent='';
}

async function generate(){
  const txt=$('p').value.trim();
  if(!txt){$('p').focus(); return}
  const [w,h]=$('s').value.split('x').map(Number);
  const model=curM();
  const vid=isVideo(model);
  const seedRaw=$('sd').value.trim();
  $('go').disabled=true; $('stop').style.display=''; $('er').textContent=''; $('mt').innerHTML='';
  $('ph').textContent='отправляю…'; $('pb').style.width='0';
  T0=Date.now(); PID=null;
  clearInterval(timer);
  timer=setInterval(()=>{$('tm').textContent=((Date.now()-T0)/1000).toFixed(1)+' с'},100);
  let r;
  try{
    r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt:txt,model:$('m').value,size:$('s').value,
        steps:Number($('st').value)||28,seed:seedRaw?Number(seedRaw):null,
        length:vid?Number($('ln').value):null, fps:vid?Number($('fp').value):null,
        negative:$('n').value,translate:$('tr').checked,keep_ollama:$('ko').checked,
        start_image:UP.start, end_image:UP.end,
        end_prompt:($('ep')?$('ep').value:''),
        translator:$('trm').value, ...loraPayload()})});
  }catch(e){fail('сервер не отвечает: '+e); return}
  const j=await r.json();
  if(!r.ok){fail(j.error||('HTTP '+r.status)); return}
  CUR=j.id;
  const t=setInterval(async()=>{
    let s;
    try{ s=await (await fetch('/api/job/'+CUR)).json() }catch(e){ return }
    if(s.prompt_id) PID=s.prompt_id;
    if(s.phase) $('ph').textContent=s.phase+'…';
    if(s.state==='done'){
      clearInterval(t); clearInterval(timer);
      $('ph').textContent='готово'; $('tm').textContent=(s.sec||0)+' с';
      $('go').disabled=false; $('stop').style.display='none'; $('pb').style.width='100%';
      CURFILE=s.image;
      $('fr').innerHTML=mediaTag(s.image);
      $('mt').innerHTML=mediaMeta(s);
      loadHistory();
    } else if(s.state==='error'){
      clearInterval(t); clearInterval(timer); fail(s.error);
    } else if(!s.phase){ $('ph').textContent=vid?'снимаю…':'рисую…' }
  },700);
}
function isVid(f){ return /\.mp4$/i.test(f||'') }
function loraPayload(){ return {lora:($('lo')&&!$('lo').disabled)?($('lo').value||null):null,
                              lora_weight:(Number($('lw').value)||0)/100} }
let CURFILE=null, MASK=null, ERASE=false, BGTEXT='';
const PH_MAIN='рыжий кот на подоконнике, утренний свет';
let Q={now:null, pending:[], comfy:{}}, QPCT=0, QPID=null;
let qtimer=null, playingId=null;

function secs(x){ return x==null?'':Math.round(x)+' с' }
async function pollQueue(){
  let s;
  try{ s=await (await fetch('/api/queue')).json() }catch(e){ setQHidden(); return }
  Q=s;
  if(!s.count && !(s.comfy&&s.comfy.foreign)){ setQHidden(); return }
  $('qcard').style.display='';
  let html='';
  const rows=[s.now].filter(Boolean).concat(s.pending||[]);
  rows.forEach((r,i)=>{
    const run=r.state==='running';
    const pct=run&&r.prompt_id&&r.prompt_id===QPID?QPCT:null;
    const time=run
      ? (pct!=null&&pct>3
          ? `${secs(r.elapsed)} · ${pct}%`
          : secs(r.elapsed))
      : (r.est?`≈${r.est} с`:'зап: '+secs(r.waited));
    html+=`<div class="qrow${run?' run':''}">
      <span class="n">${i+1}</span>
      <span class="w" title="${esc(r.what)}">${esc(r.what||'—')}</span>
      <span class="t" style="opacity:.7">${esc(r.size)} · ${r.steps}ш</span>
      ${run?`<span class="mini2"><i style="width:${pct||0}%"></i></span>`:''}
      <span class="t">${time}</span></div>`;
  });
  if(s.comfy && s.comfy.foreign){
    html+=`<div class="qrow"><span class="n">!</span><span class="w">в ComfyUI идёт задача не из панели</span>
      <span class="t">жду его окончания</span></div>`;
  }
  $('qlist').innerHTML=html;
  const total=(s.count||0);
  const est=(s.pending||[]).reduce((a,r)=>a+(r.est||0),0)+(s.now&&s.now.est&&s.now.state!=='running'?s.now.est:0);
  $('qh').textContent=total>1?`очередь: ${total}`:'сейчас делается';
  $('qs').textContent=est>0?`ждут ≈ ${Math.round(est)} с`:(s.comfy&&s.comfy.running?'модель занята':'');
}
function setQHidden(){ $('qcard').style.display='none'; $('qlist').innerHTML=''; }
function startQueuePoll(){
  clearInterval(qtimer);
  pollQueue();
  qtimer=setInterval(pollQueue, 2000);
}

function bgBar(){
  const el=$('bgbar');
  if(!BGTEXT){ el.style.display='none'; el.innerHTML=''; return }
  const cut=BGTEXT.length>90?BGTEXT.slice(0,90)+'…':BGTEXT;
  el.style.display='';
  el.innerHTML='описание фона в памяти: «'+esc(cut)+'» '+
    '<button class="mini" id="bgback">вернуть фон в поле</button>';
  $('bgback').onclick=()=>{
    $('p').value=BGTEXT; BGTEXT=''; $('p').placeholder=PH_MAIN; bgBar();
  };
}
function maskCoverage(){
  if(!MASK) return 0;
  const d=MASK.ctx.getImageData(0,0,MASK.nw,MASK.nh).data;
  let n=0;
  for(let i=3;i<d.length;i+=4) if(d[i]>127) n++;
  return n/(MASK.nw*MASK.nh);
}
function maskHint(){
  if(!MASK) return;
  const frac=maskCoverage(), pct=frac*100;
  let t;
  if(frac<=0) t='закрась место, которое надо перерисовать';
  else t='закрашено '+(pct<1?pct.toFixed(1):Math.round(pct))+'% кадра';
  if(pct>50) t+=' — это почти весь кадр, перерисуется всё';
  $('mkhint').textContent=t;
}
function startMask(){
  const fr=$('fr'), img=fr.querySelector('img');
  if(!img||!CURFILE){ $('er').textContent='сначала нужна картинка: сгенерируй или выбери из ленты'; return }
  stopMask(false);
  const cur=$('p').value.trim();
  if(!BGTEXT && cur){ BGTEXT=cur; $('p').value=''; }
  if(BGTEXT){ $('p').placeholder='что вставить в закрашенное место'; bgBar(); }
  const nw=img.naturalWidth, nh=img.naturalHeight;
  const c=document.createElement('canvas');
  c.id='maskc'; c.width=nw; c.height=nh;
  const r=img.getBoundingClientRect(), f=fr.getBoundingClientRect();
  c.style.cssText='position:absolute;border-radius:6px;cursor:crosshair;mix-blend-mode:screen;'+
    `left:${r.left-f.left}px;top:${r.top-f.top}px;width:${r.width}px;height:${r.height}px;opacity:.55`;
  fr.appendChild(c);
  const ctx=c.getContext('2d');
  ctx.lineCap='round'; ctx.lineJoin='round';
  MASK={c, ctx, nw, nh, drawing:false, last:null};
  const pos=e=>{ const b=c.getBoundingClientRect();
    return {x:(e.clientX-b.left)*nw/b.width, y:(e.clientY-b.top)*nh/b.height} };
  c.addEventListener('pointerdown',e=>{
    c.setPointerCapture(e.pointerId);
    MASK.drawing=true; MASK.last=pos(e); paint(MASK.last, MASK.last);
  });
  c.addEventListener('pointermove',e=>{
    if(!MASK||!MASK.drawing) return;
    const p=pos(e); paint(MASK.last, p); MASK.last=p;
  });
  const end=()=>{ if(MASK){ MASK.drawing=false; maskHint() } };
  c.addEventListener('pointerup',end); c.addEventListener('pointercancel',end);
  $('mbar').style.display='';
  $('go').textContent='перерисовать выделенное';
  $('mkhint').textContent=BGTEXT
    ? 'закрась место, а в поле слева напиши, что там должно быть (фон я убрал и верну после)'
    : 'закрась то, что надо перерисовать';
  if(BGTEXT) $('p').focus();
}
function paint(a,b){
  if(!MASK) return;
  const ctx=MASK.ctx, w=Math.max(1,Number($('br').value)||60);
  ctx.globalCompositeOperation=ERASE?'destination-out':'source-over';
  ctx.strokeStyle='#fff'; ctx.lineWidth=w; ctx.lineCap='round'; ctx.lineJoin='round';
  ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
  ctx.beginPath(); ctx.arc(b.x,b.y,w/2,0,Math.PI*2); ctx.fillStyle='#fff'; ctx.fill();
}
function stopMask(restore){
  const c=document.getElementById('maskc');
  if(c) c.remove();
  MASK=null; $('mbar').style.display='none';
  $('go').textContent='нарисовать';
  $('p').placeholder=PH_MAIN;
  if(restore!==false && BGTEXT){ $('p').value=BGTEXT; BGTEXT='' }
  bgBar();
}
function maskPng(){
  if(!MASK) return null;
  const out=document.createElement('canvas');
  out.width=MASK.nw; out.height=MASK.nh;
  const o=out.getContext('2d');
  o.fillStyle='#000'; o.fillRect(0,0,out.width,out.height);
  o.drawImage(MASK.c,0,0);
  return out.toDataURL('image/png');
}
async function startJob(url, body){
  $('go').disabled=true; $('stop').style.display=''; $('er').textContent='';
  $('ph').textContent='отправляю…'; $('pb').style.width='0';
  T0=Date.now(); clearInterval(timer);
  timer=setInterval(()=>{$('tm').textContent=((Date.now()-T0)/1000).toFixed(1)+' с'},100);
  let r, j;
  try{
    r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    j=await r.json();
  }catch(e){ fail('сервер не отвечает: '+e); return null }
  if(!r.ok){ fail(j.error||('HTTP '+r.status)); return null }
  CUR=j.id; PID=null;
  follow(j.id);
  return j.id;
}
function follow(id){
  const t=setInterval(async()=>{
    let s; try{ s=await (await fetch('/api/job/'+id)).json() }catch(e){ return }
    if(s.prompt_id) PID=s.prompt_id;
    if(s.phase) $('ph').textContent=s.phase+'…';
    if(s.state==='done'){
      clearInterval(t); clearInterval(timer);
      $('ph').textContent='готово'; $('tm').textContent=(s.sec||0)+' с';
      $('go').disabled=false; $('stop').style.display='none'; $('pb').style.width='100%';
      CURFILE=s.image;
      $('fr').innerHTML=mediaTag(s.image);
      $('mt').innerHTML=mediaMeta(s);
      loadHistory();
    } else if(s.state==='error'){
      clearInterval(t); clearInterval(timer); fail(s.error);
    }
  },800);
}
async function inpaintGo(){
  const m=maskPng();
  if(!m){ $('er').textContent='нарисуй маску на картинке'; return }
  const id=await startJob('/api/inpaint',{image:CURFILE,mask:m,prompt:$('p').value.trim(),
    model:$('m').value, steps:Number($('st').value)||28,
    denoise:(Number($('dn').value)||60)/100, grow:Number($('gw').value)||0,
    negative:$('n').value, translate:$('tr').checked,
     keep_ollama:$('ko').checked, translator:$('trm').value, ...loraPayload()});
   if(id) stopMask();
}
async function detailGo(detector){
  if(!CURFILE){ $('er').textContent='сначала нарисуй картинку или выбери её в ленте'; return }
  if(isVid(CURFILE)){ $('er').textContent='это видео, починка только для картинок'; return }
  $('er').textContent='';
  const what=detector==='face'?'лицо':'кисти';
  // кисти начинают перерисовываться только с силы 70, лицо живёт и на 50
  const dmin=detector==='face'?0.5:0.6;
  const dn=Math.max((Number($('dn').value)||0)/100, dmin);
  await startJob('/api/detail',{image:CURFILE,prompt:$('p').value.trim(),
    detector:detector, model:$('m').value, steps:Number($('st').value)||28,
    denoise:dn, negative:$('n').value,
     upscale:$('hup').checked, translate:$('tr').checked,
     pose:$('pz').checked, keep_ollama:$('ko').checked, translator:$('trm').value, ...loraPayload()});
 }
 async function handsGo(){ return detailGo('hands') }
async function faceGo(){ return detailGo('face') }
async function redrawGo(){
  if(!CURFILE){ $('er').textContent='сначала нарисуй картинку или выбери её в ленте'; return }
  if(isVid(CURFILE)){ $('er').textContent='это видео, перерисовка по контуру только для картинок'; return }
  if(!$('p').value.trim()){ $('er').textContent='опиши, что должно получиться'; return }
  $('er').textContent='';
  await startJob('/api/redraw',{image:CURFILE,prompt:$('p').value.trim(),
    model:$('m').value, steps:Number($('st').value)||28,
    denoise:(Number($('rs').value)||60)/100,
    strength:(Number($('rst').value)||80)/100,
    ctrl_type:$('rt').value, negative:$('n').value,
     translate:$('tr').checked, keep_ollama:$('ko').checked,
     translator:$('trm').value, ...loraPayload()});
}
function mediaTag(f){
  const u='/img/'+encodeURI(f||'');
  return isVid(f)
    ? `<video src="${u}" controls autoplay loop muted playsinline></video>`
    : `<img src="${u}" alt="">`;
}
const CTRL_NAMES={'canny/lineart/anime_lineart/mlsd':'контур','depth':'глубина',
  'openpose':'поза','hed/pidi/scribble/ted':'мягкие линии','normal':'нормали','tile':'плитка'};
function mediaMeta(s){
  const parts=[`${s.w}x${s.h}`, `${s.steps} шагов`];
  if(s.kind==='inpaint'){
    parts.push('перерисовка области: сила '+(Math.round((s.denoise||0.6)*100))+'%, расширение '+(s.grow||0));
  }
  if(s.kind==='detail'){
    parts.push('починка '+(s.detector==='face'?'лица':'кистей')+': сила '+
               (Math.round((s.denoise||0.5)*100))+'%'+(s.pose?', по скелету':'')+
               (s.upscale?', с апскейлом 1,5x':''));
  }
  if(s.kind==='redraw'){
    parts.push('по опоре '+(CTRL_NAMES[s.ctrl_type]||'контур')+': сила перерисовки '+
               (Math.round((s.denoise||0.6)*100))+'%, прочность '+
               (Math.round((s.strength||0.8)*100))+'%');
  }
  if(s.kind==='video'&&s.length) parts.push(`${s.length} кадров, ${s.fps} fps`);
  parts.push(`seed ${s.seed}`);
  let h=`<div><b>фраза:</b> ${esc(s.prompt_ru||'')}</div>`;
  h+=`<div><b>перевод:</b> ${esc(s.prompt_en||'—')}`+(s.note?` (${esc(s.note)})`:'')+'</div>';
  if(s.start||s.end) h+=`<div><b>кадры:</b> первый ${esc(s.start||'нарисован')}, последний ${esc(s.end||'нарисован')}</div>`;
  return h+`<div>${esc(parts.join(', '))}</div>`;
}
function fail(msg){
  $('go').disabled=false; $('stop').style.display='none'; $('ph').textContent='не вышло';
  $('er').textContent=msg||'неизвестная ошибка'; clearInterval(timer);
}
let HIST=[];
async function loadHistory(){
  HIST=await (await fetch('/api/history')).json();
  $('cnt').textContent=HIST.length?`${HIST.length}`:'пусто';
  $('gal').innerHTML=HIST.map(x=>{
    const ttl=esc((x.ru||'')+(x.en?'\n\nперевод: '+x.en:''));
    const media=isVid(x.file)
      ? `<video src="/img/${encodeURI(x.file)}" muted preload="metadata" onclick="show('${esc(x.file)}')"></video>`
      : `<img src="/img/${encodeURI(x.file)}" alt="" onclick="show('${esc(x.file)}')">`;
    const cap=(isVid(x.file)?'▶ ':'')+esc((x.model||'').split(' ')[0])+' '+x.w+'x'+x.h;
    return `<figure title="${ttl}">${media}
      <button class="del" title="удалить" onclick="drop('${esc(x.file)}',event)">×</button>
      <figcaption>${cap}</figcaption></figure>`;
  }).join('');
}
async function drop(file, ev){
  if(ev) ev.stopPropagation();
  if(!confirm('Удалить файл с диска и запись в ленте?')) return;
  await fetch('/api/delete',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({file:file,files:true})});
  const im=document.querySelector('#fr img, #fr video');
  if(im && im.getAttribute('src')==='/img/'+encodeURI(file)) blank();
  loadHistory();
}
function blank(){
  $('fr').innerHTML='<span class="empty">здесь появится картинка или видео</span>';
  $('ph').textContent='готов'; $('tm').textContent=''; $('mt').textContent=''; $('er').textContent=''; $('pb').style.width='0';
}
async function clearAll(){
  const n=document.querySelectorAll('.gal figure').length;
  if(!n){ return }
  if(!confirm('Очистить ленту: '+n+' записей?')) return;
  const files=confirm('Удалить и сами файлы с диска? ОК - удалить, Отмена - только записи в ленте.');
  await fetch('/api/clear',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({files:files})});
  blank(); loadHistory();
}
async function show(file){
  const x=HIST.find(i=>i.file===file)||{};
  $('fr').innerHTML=mediaTag(file);
  $('ph').textContent='из ленты'; $('tm').textContent=''; $('er').textContent='';
  $('mt').innerHTML=mediaMeta({w:x.w, h:x.h, steps:x.steps, seed:x.seed, kind:x.kind,
                               length:x.length, fps:x.fps, prompt_ru:x.ru, prompt_en:x.en,
                                denoise:x.denoise, grow:x.grow, start:x.start, end:x.end,
                                detector:x.detector, strength:x.strength,
                                upscale:x.upscale});
  $('p').value=x.ru||'';
  CURFILE=file;
}
$('go').onclick=()=>{ if(MASK) return inpaintGo(); return generate() };
$('inp').onclick=startMask;
$('mkgo').onclick=inpaintGo;
$('hands').onclick=handsGo;
$('face').onclick=faceGo;
$('redraw').onclick=redrawGo;
$('mkx').onclick=()=>stopMask(false);
$('mkclr').onclick=()=>{ if(MASK){ MASK.ctx.clearRect(0,0,MASK.nw,MASK.nh) } };
$('mker').onclick=()=>{ ERASE=!ERASE; $('mker').textContent=ERASE?'ластик: вкл':'ластик' };
$('fs').onchange=()=>upload('start',$('fs'));
$('fe').onchange=()=>upload('end',$('fe'));
$('clr').onclick=clearAll;
$('stop').onclick=async()=>{
  $('stop').disabled=true;
  try{await fetch('/api/interrupt',{method:'POST'})}catch(e){}
  $('stop').disabled=false; $('stop').style.display='none';
};
$('free').onclick=async()=>{ $('free').disabled=true;
  try{await fetch('/api/unload',{method:'POST'})}catch(e){}
  $('free').disabled=false; $('free').textContent='освободил'; setTimeout(()=>$('free').textContent='освободить видеопамять',2500);
};
$('p').addEventListener('keydown',e=>{ if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)) generate() });

/* --- чат с моделью: одно поле, одна кнопка --- */
/* разговор держим в браузере: закрыл/обновил страницу — вернулся на то же место */
const CHAT_KEY='standui-chat-v1', TAB_KEY='standui-tab-v1', LORA_KEY='standui-lora-v1',
      CTX_KEY='standui-ctx-v1',
      TRAIN_KEY='standui-train-name-v1';
let CHAT=[];
function saveChat(){
  try{
    const data=CHAT.slice(-40).map(m=>({role:m.role,content:m.content,thinking:m.thinking||'',
                                      model:m.model||'',image:m.image||null,job:m.job||null}));
    if(data.length) localStorage.setItem(CHAT_KEY,JSON.stringify(data));
    else localStorage.removeItem(CHAT_KEY);
  }catch(e){}
}
function loadChat(){
  try{
    const raw=localStorage.getItem(CHAT_KEY);
    CHAT=raw?JSON.parse(raw):[];
    if(!Array.isArray(CHAT)) CHAT=[];
  }catch(e){ CHAT=[] }
  CHAT=CHAT.filter(m=>m&&(m.role==='user'||m.role==='assistant')&&typeof m.content==='string');
}
function setMode(mode){
  const m=mode===true?'chat':(mode===false?'img':mode);
  const chat=m==='chat', trn=m==='train', vid=m==='video', grd=m==='grade', doc=m==='doc', vis=m==='vision', hw=m==='hw';
  $('imgcard').style.display=(chat||trn||vid||grd||doc||vis||hw)?'none':'';
  $('chatctl').style.display=chat?'':'none';
  $('chatlog').style.display=chat?'':'none';
  $('traincard').style.display=trn?'':'none';
  $('vidcard').style.display=vid?'':'none';
  $('gradecard').style.display=grd?'':'none';
  $('doccard').style.display=doc?'':'none';
  $('viscard').style.display=vis?'':'none';
  $('hwcard').style.display=hw?'':'none';
  $('framecard').style.display=(chat||vid||grd||doc||vis||hw)?'none':'';
  $('histcard').style.display=(chat||vid||grd||doc||vis||hw)?'none':'';
  $('tabimg').classList.toggle('on',m==='img');
  $('tabvis').classList.toggle('on',vis);
  $('tabchat').classList.toggle('on',chat);
  $('tabtrn').classList.toggle('on',trn);
  $('tabvid').classList.toggle('on',vid);
  $('tabgrd').classList.toggle('on',grd);
  $('tabdoc').classList.toggle('on',doc);
  $('tabhw').classList.toggle('on',hw);
  try{ localStorage.setItem(TAB_KEY, m) }catch(e){}
  if(chat) $('cq').focus();
  if(vis) loadVision();
  if(trn){ loadTrain(); startTrainPoll() } else stopTrainPoll();
  if(vid){ loadVideo(); startVideoPoll() } else stopVideoPoll();
  if(grd){ gradeLoad() }
  if(doc){ loadDoc(); startDocPoll() } else stopDocPoll();
  if(hw){ loadSys(); startSysPoll() } else stopSysPoll();
}
function renderChat(){
  const box=$('clog');
  if(!CHAT.length){ box.innerHTML='<span class="empty">здесь будет ответ</span>'; return }
  box.innerHTML=CHAT.map(m=>m.role==='user'
    ? `<div class="u"><b>вы:</b>${esc(m.content)}</div>`
    : `<div class="a"><b>${esc(m.model||'модель')}:</b>${esc(m.content)}${m.image?`<img src="/img/${encodeURI(m.image)}" alt="" onclick="window.open(this.src,'_blank')">`:''}${m.pending?`<div class="ph">${esc(m.pending)}</div>`:''}${m.thinking?`<details><summary>ход мысли</summary>${esc(m.thinking)}</details>`:''}</div>`
  ).join('');
  box.scrollTop=box.scrollHeight;
}
async function sendChat(){
  const txt=$('cq').value.trim();
  if(!txt){ $('cq').focus(); return }
  const model=$('cm').value;
  CHAT.push({role:'user',content:txt});
  $('cq').value=''; renderChat(); saveChat();
  $('csend').disabled=true;
  const t0=Date.now();
  const tick=setInterval(()=>{$('cstate').textContent='думаю… '+((Date.now()-t0)/1000).toFixed(0)+' с'},200);
  let j=null, err='';
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model,think:!$('cnoth').checked,draw:$('cpic').checked,
        ctx:Number($('cctx').value)||16384,
        messages:CHAT.slice(-12).map(m=>({role:m.role,content:m.content}))})});
    j=await r.json();
    if(!r.ok) err=j.error||('HTTP '+r.status);
  }catch(e){ err='сервер не отвечает: '+e }
  clearInterval(tick); $('csend').disabled=false;
  if(err){ CHAT.pop(); renderChat(); $('cstate').textContent=err; return }
  CHAT.push({role:'assistant',content:j.text||'(пусто)',thinking:j.thinking||'',
             model:j.model||model,image:j.image||null});
  $('cstate').textContent=(j.model||model)+', '+j.sec+' с';
  renderChat();
  if(j.image_job) watchPic(j.image_job,CHAT[CHAT.length-1]);
  else saveChat();
}
/* картинка по просьбе из чата: ждём задачу и вставляем её в разговор */
async function watchPic(id,msg){
  const t0=Date.now();
  msg.job=id; msg.pending='рисую…'; renderChat(); saveChat();
  while(true){
    await new Promise(r=>setTimeout(r,1500));
    let j=null;
    try{ const r=await fetch('/api/job/'+id); j=await r.json(); }catch(e){}
    if(!j) continue;
    const sec=((Date.now()-t0)/1000).toFixed(0);
    if(j.state==='queued'||j.state==='running'){
      msg.pending='рисую: '+(j.phase||'…')+', '+sec+' с';
      $('cstate').textContent=msg.pending; renderChat(); continue;
    }
    msg.pending=''; msg.job=null;
    if(j.state==='done'){
      msg.image=j.image;
      msg.content=(j.prompt_ru||msg.content)+' — '+j.w+'x'+j.h+', '+j.sec+' с';
      $('cstate').textContent='картинка готова, '+j.sec+' с';
    }else if(j.state){
      msg.content=(msg.content?msg.content+', ':'')+'не вышло: '+(j.error||'ошибка ComfyUI');
      $('cstate').textContent='картинка не вышла';
    }else{
      msg.content=(msg.content?msg.content+', ':'')+'рисование не досмотрено (стенд перезапускался)';
      $('cstate').textContent=''; 
    }
    renderChat(); saveChat(); return;
  }
}
/* --- обучение стиля (LoRA): фотографии, название, одна кнопка --- */
let TPOL=null, TLAST=null, TPUB=null, TCLIENT='', TCAPDIRTY=false;
function stopTrainPoll(){ if(TPOL){ clearInterval(TPOL); TPOL=null } }
function startTrainPoll(){ stopTrainPoll(); TPOL=setInterval(loadTrain,3000) }
function humanSec(s){
  s=Math.max(0,Math.round(s||0));
  if(s<60) return s+' с';
  const m=Math.round(s/60);
  if(m<60) return m+' мин';
  return Math.floor(m/60)+' ч '+(m%60)+' мин';
}
function renderPhotos(ps){
  const box=$('tphlist');
  const kb=ps.reduce((a,x)=>a+(x.kb||0),0);
  $('tphhint').textContent=ps.length?ps.length+' шт., '+fmt(kb)+' КБ на стенде':'пока ничего не добавлено';
  box.innerHTML=ps.map(x=>`<figure><img src="/api/train/photo/${encodeURI(x.name)}" alt=""
    onclick="window.open(this.src,'_blank')"><button class="del" title="убрать" data-n="${esc(x.name)}">×</button></figure>`).join('');
  box.querySelectorAll('.del').forEach(b=>b.onclick=async ev=>{
    ev.stopPropagation();
    TCLIENT='';
    try{ await fetch('/api/train/photo/delete',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name:b.dataset.n})}) }catch(e){}
    loadTrain();
  });
}
function renderStyles(ls){
  $('tlcnt').textContent=ls.length?ls.length+' шт.':'';
  const box=$('tloras');
  box.innerHTML=ls.length?ls.map(x=>`<div class="qrow"><span style="flex:1">${esc(x.label)}
    <span style="color:var(--dim)">${esc(x.file)}</span></span>
    <button class="mini" data-n="${esc(x.file)}">удалить</button></div>`).join('')
    :'<div class="hint">стилей пока нет</div>';
  box.querySelectorAll('button[data-n]').forEach(b=>b.onclick=async()=>{
    if(!confirm('Удалить стиль '+b.dataset.n+'?')) return;
    try{ await fetch('/api/lora/delete',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name:b.dataset.n})}) }catch(e){}
    loadTrain(); refreshLoras();
  });
}
async function refreshLoras(){
  try{
    const c=await (await fetch('/api/config')).json();
    CFG.loras=c.loras||[];
    const cur=$('lo').value;
    $('lo').innerHTML='<option value="">без стиля (обычная модель)</option>'+
      CFG.loras.map(x=>`<option value="${esc(x.file)}">${esc(x.label)}</option>`).join('');
    if(cur&&CFG.loras.some(x=>x.file===cur)) $('lo').value=cur;
    loraHint();
  }catch(e){}
}
async function loadTrain(){
  let j=null;
  try{ j=await (await fetch('/api/train')).json() }catch(e){ return }
  TLAST=j;
  renderPhotos(j.photos||[]);
  renderStyles(j.loras||[]);
  if(!TCAPDIRTY && document.activeElement!==$('tcap')){
    const cap=j.caption||'';
    if($('tcap').value!==cap) $('tcap').value=cap;
  }
  const st=j.state||'idle', busy=!!j.busy;
  $('tstop').style.display=busy?'':'none';
  $('tgo').disabled=busy;
  const names={idle:'готов',running:'учу…',publishing:'сохраняю стиль…',done:'стиль готов',
               error:'не вышло',stalled:'остановилось',stopped:'остановлено'};
  let line=names[st]||st;
  if((st==='running'||st==='publishing')&&j.step&&j.total) line+=' — шаг '+j.step+' из '+j.total;
  $('tstate').textContent=line;
  $('tpb').style.width=(st==='done'?100:(j.pct||0))+'%';
  let right='';
  if(busy&&j.left_sec) right='осталось ~'+humanSec(j.left_sec);
  else if(st==='done') right=j.published?'файл в стилях':'в стили не попал';
  $('ttime').textContent=right;
  $('terr').textContent=TCLIENT||j.error||'';
  if(st==='done'&&j.published&&TPUB!==j.name){ TPUB=j.name; refreshLoras() }
}
$('tph').onchange=async ev=>{
  const files=[...ev.target.files];
  if(!files.length) return;
  TCLIENT=''; $('terr').textContent='';
  $('tph').disabled=true;
  let ok=0, bad=0;
  for(let i=0;i<files.length;i++){
    $('tphhint').textContent='загружаю '+(i+1)+' из '+files.length+'…';
    try{
      const b64=await new Promise((res,rej)=>{const r=new FileReader();
        r.onload=()=>res(String(r.result).split(',')[1]||''); r.onerror=rej; r.readAsDataURL(files[i])});
      const r=await fetch('/api/train/photo',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({data:b64})});
      if(r.ok) ok++; else bad++;
    }catch(e){ bad++ }
  }
  $('tph').value=''; $('tph').disabled=false;
  await loadTrain();
  if(bad) $('tphhint').textContent='добавлено '+ok+', не принято '+bad;
  else if(ok&&(TLAST&&TLAST.caption)) $('tcaphint').textContent='фотографии получили подпись набора';
}
$('tgo').onclick=async()=>{
  TCLIENT='';
  const name=($('tname').value||'').trim().toLowerCase();
  if(!name){ $('tname').focus(); TCLIENT='напишите название стиля'; $('terr').textContent=TCLIENT; return }
  if(!(TLAST&&(TLAST.photos||[]).length)){ TCLIENT='сначала добавьте фотографии'; $('terr').textContent=TCLIENT; return }
  $('tgo').disabled=true; TCLIENT=''; $('terr').textContent='';
  let err='';
  try{
    const r=await fetch('/api/train/start',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name,steps:Number($('tsteps').value)||1500,
        dim:Number($('tdim').value)||16,base:$('tbase').value})});
    const j=await r.json();
    if(!r.ok) err=j.error||('HTTP '+r.status);
  }catch(e){ err='сервер не отвечает: '+e }
  if(err){ $('tgo').disabled=false; $('terr').textContent=err; return }
  try{ localStorage.setItem(TRAIN_KEY,name) }catch(e){}
  startTrainPoll(); setTimeout(loadTrain,800);
}
$('tstop').onclick=async()=>{
  if(!confirm('Остановить обучение? Недоделанный стиль не появится.')) return;
  $('tstop').disabled=true;
  try{ await fetch('/api/train/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}) }catch(e){}
  $('tstop').disabled=false;
  setTimeout(loadTrain,2000);
}
$('tclear').onclick=async()=>{
  if(!confirm('Убрать все фотографии со стенда?')) return;
  TCLIENT='';
  try{ await fetch('/api/train/clear',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}) }catch(e){}
  loadTrain();
}
$('tcapgo').onclick=async()=>{
  const text=($('tcap').value||'').trim();
  const n=(TLAST&&(TLAST.photos||[]).length)||0;
  $('tcapgo').disabled=true;
  try{
    const r=await fetch('/api/train/caption',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text})});
    const j=await r.json();
    if(!r.ok){ $('tcaphint').textContent=j.error||('HTTP '+r.status) }
    else{
      TCAPDIRTY=false;
      $('tcaphint').textContent=text?('подпись сохранена для '+j.count+' фото'):('подпись убрана, панель напишет сама');
    }
  }catch(e){ $('tcaphint').textContent='сервер не отвечает: '+e }
  $('tcapgo').disabled=false;
}
$('tcap').oninput=()=>{ TCAPDIRTY=true };
/* --- документы: замороженная база (ollama + Open WebUI в замкнутой сети) --- */
let DPOLL=null, DARM=0, DBUSY=false, DON=false;
function dHsize(n){
  n=Number(n)||0; if(!n) return '0';
  const u=['Б','КБ','МБ','ГБ']; let i=0;
  while(n>=1024 && i<u.length-1){ n/=1024; i++ }
  return (i<2?n.toFixed(0):n.toFixed(1))+' '+u[i];
}
function dAge(s){
  s=Math.max(0, Math.round(Number(s)||0));
  if(s<60) return s+' с';
  if(s<3600) return Math.floor(s/60)+' мин '+(s%60)+' с';
  return Math.floor(s/3600)+' ч '+Math.floor((s%3600)/60)+' мин';
}
async function loadDoc(){
  let j;
  try{ j=await (await fetch('/api/doc')).json() }
  catch(e){ $('dstate').textContent='сервер не отвечает'; return }
  DBUSY=!!j.busy;
  const c=j.containers||{}, up=Object.keys(c).filter(k=>c[k].state==='running').length;
  let st='', err='';
  if(j.busy){
    const w={ 'ui-on':'поднимаю', 'ui-off':'гашу', 'wipe':'сношу', 'build':'собираю образ' }[j.action]||'работаю';
    st=w+'… '+dAge(j.secs);
  }else if(j.action && j.rc!==null && j.rc!==undefined){
    if(j.rc!==0) st='ошибка (код '+j.rc+') · '+dAge(j.secs);
    else if(j.health) st='панель поднята · '+dAge(j.secs);
    else st='готово · '+dAge(j.secs);
    if(j.rc!==0) err='последнее действие не доделалось — вывод скрипта ниже';
  }else{
    st=j.health?'панель поднята':(up?'поднято не всё':'погашено');
  }
  $('dstate').textContent=st;
  $('dtime').textContent=(!j.busy && j.action)?dAge(j.secs):'';
  $('derr').textContent=err;
  const img=j.image||{};
  const rows=[];
  rows.push('<b>образ</b> '+esc(j.image_name||'doc')+' — '+
            (img.present?('есть, '+dHsize(img.bytes)):'НЕТ (собери заново)'));
  (j.container_names||[]).forEach(n=>{
    const x=c[n];
    rows.push('<b>'+esc(n)+'</b> — '+(x?(x.state==='running'?'работает, '+esc(x.status):esc(x.status)):'нет'));
  });
  rows.push('<b>сеть</b> '+esc(j.net_name||'docnet')+' — '+
            (j.net&&j.net.present?(j.net.internal?'замкнутая, интернета нет':'ВНИМАНИЕ: с выходом в сеть'):'нет'));
  rows.push('<b>панель</b> '+esc(j.ui_ip||'')+':8080 — '+(j.health?'отвечает':'не отвечает'));
  $('dinfo').innerHTML=rows.join('<br>');
  const df=j.files||{};
  const dlist=df.files||[], dbad=dlist.filter(f=>!f.indexed);
  if(!df.ok){
    $('dfiles').textContent='не посмотреть'+(df.error?(' ('+df.error+')'):'');
    $('dflist').innerHTML='';
  }else{
    let s='загружено '+dlist.length+', проиндексировано '+(df.indexed_total||0);
    if(dbad.length) s+=' · без индекса '+dbad.length;
    $('dfiles').textContent=s;
    $('dflist').innerHTML=dlist.map(f=>{
      const ok=!!f.indexed;
      return '<div class="qrow"><span class="n">'+(ok?'✓':'—')+'</span>'+
             '<span class="w" title="'+esc(f.name||'')+'">'+esc(f.name||'?')+'</span>'+
             '<span class="t">'+(ok?(f.chunks+' кусков'):'не проиндексирован')+'</span></div>';
    }).join('');
  }
  $('dtun').value=j.tunnel||'';
  $('dbuildrow').style.display=img.present?'none':'block';
  $('dstart').disabled=DBUSY;
  $('dstop').disabled=DBUSY||!up;
  $('dwipe').disabled=DBUSY;
  $('dbuild').disabled=DBUSY;
  const tail=(j.log||'').trim();
  $('dlog').textContent=tail?tail.split('\n').slice(-12).join('\n'):'';
}
function startDocPoll(){
  if(DON) return;
  DON=true;
  const tick=async()=>{ if(!DON) return; await loadDoc(); if(DON) DPOLL=setTimeout(tick, DBUSY?2000:6000) };
  DPOLL=setTimeout(tick, DBUSY?2000:6000);
}
function stopDocPoll(){ DON=false; if(DPOLL){ clearTimeout(DPOLL); DPOLL=null } }
async function docAct(what, body){
  $('derr').textContent='';
  ['dstart','dstop','dwipe','dbuild'].forEach(b=>{ if($(b)) $(b).disabled=true });
  try{
    const r=await fetch('/api/doc/'+what,{method:'POST',headers:{'Content-Type':'application/json'},
                                          body:JSON.stringify(body||{})});
    const j=await r.json();
    if(!r.ok) $('derr').textContent=j.error||('HTTP '+r.status);
  }catch(e){ $('derr').textContent='сервер не отвечает: '+e }
  await loadDoc();
  startDocPoll();
}
$('dstart').onclick=()=>docAct('start');
$('dstop').onclick=()=>docAct('stop');
$('dbuild').onclick=()=>docAct('build');
$('dwipe').onclick=async()=>{
  const now=Date.now();
  if(now-DARM>6000){
    DARM=now;
    $('dwipe').textContent='снести всё!';
    $('dwnote').textContent='точно? нажми ещё раз в течение 6 секунд — вернуть будет нельзя';
    setTimeout(()=>{ if(DARM && Date.now()-DARM>=6000){
      DARM=0; $('dwipe').textContent='снести всё';
      $('dwnote').textContent='контейнеры, документы и история удалятся навсегда'; } },6200);
    return;
  }
  DARM=0; $('dwipe').textContent='снести всё'; $('dwnote').textContent='удаляю…';
  await docAct('wipe',{confirm:1});
  $('dwnote').textContent='контейнеры, документы и история удалятся навсегда';
};
$('dcopy').onclick=async()=>{
  const v=$('dtun').value;
  let ok=false;
  try{ await navigator.clipboard.writeText(v); ok=true }catch(e){
    try{ $('dtun').select(); $('dtun').setSelectionRange(0,v.length); ok=document.execCommand('copy') }catch(_){ ok=false }
  }
  $('dcopy').textContent=ok?'скопировано':'выделено — Ctrl+C';
  setTimeout(()=>{ $('dcopy').textContent='копировать' },2000);
};
/* --- железо: загрузка и температуры, обновляется само --- */
let SPOLL=null;
function stopSysPoll(){ if(SPOLL){ clearInterval(SPOLL); SPOLL=null } }
function startSysPoll(){ stopSysPoll(); SPOLL=setInterval(loadSys,3000) }
function hwTemp(t){ return (t===null||t===undefined)?'—':Number(t).toFixed(1)+' °C' }
function hwRow(label,pct,val){
  const p=(pct===null||pct===undefined)?null:Math.max(0,Math.min(100,Number(pct)||0));
  const cls=(p===null)?'':(p>=85?'warn':'');
  return `<div class="hwrow ${cls}"><div class="lbl">${esc(label)}</div>`+
    `<div class="hwbar"${p===null?' style="visibility:hidden"':''}><i style="width:${p===null?0:Math.round(p)}%"></i></div>`+
    `<div class="val">${esc(val)}</div></div>`;
}
function hwUp(s){ s=Math.round(Number(s)||0);
  const d=Math.floor(s/86400), h=Math.floor(s%86400/3600), mi=Math.floor(s%3600/60);
  if(d) return d+' сут '+h+' ч'; if(h) return h+' ч '+mi+' мин'; return mi+' мин' }
function hwRate(x){
  if(x===null||x===undefined) return '—';
  if(x>=100) return Math.round(x)+' МБ/с';
  if(x>=0.05) return x.toFixed(1)+' МБ/с';
  return '0 МБ/с';
}
function hwDraw(id, vals, capId, color){
  const c=$(id); if(!c) return;
  const ctx=c.getContext('2d'), w=c.width=c.clientWidth*2, h=c.height=44*2;
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle='#242830'; ctx.lineWidth=2;
  for(let i=1;i<3;i++){ ctx.beginPath(); ctx.moveTo(0,h*i/3); ctx.lineTo(w,h*i/3); ctx.stroke() }
  if(!vals||vals.length<2){ $(capId).textContent='—'; return }
  let v=vals.slice(-60); while(v.length<60) v=[v[0]].concat(v);
  ctx.beginPath();
  v.forEach((x,i)=>{ const px=w*i/(v.length-1), py=h-(h-4)*(Math.max(0,Math.min(100,Number(x)||0))/100)-2;
    i?ctx.lineTo(px,py):ctx.moveTo(px,py) });
  ctx.strokeStyle=color; ctx.lineWidth=3; ctx.stroke();
  $(capId).textContent=Number(v[v.length-1]).toFixed(1)+' %';
}
function hwHist(hist){
  const c=[],g=[],m=[];
  (hist||[]).forEach(x=>{ c.push(x[0]); g.push(x[1]); m.push(x[2]) });
  hwDraw('hccpu',c,'hscpu','#5b9dd9');
  hwDraw('hcgpu',g,'hsgpu','#67c08a');
  hwDraw('hcmem',m,'hsmem','#c0a25b');
}
function renderSys(j){
  const c=j.cpu||{}, m=j.mem||{}, s=j.swap||{}, g=j.gpu||{}, ssd=j.ssd||{}, io=j.io||{};
  const sz=n=>dHsize(n);
  let h='';
  h+=hwRow('процессор', c.percent, ((c.percent===null||c.percent===undefined)
      ? '—' : Math.round(c.percent)+' %, '+hwTemp(c.temp))+
      ((c.watts===null||c.watts===undefined)?'':', '+Math.round(c.watts)+' Вт'));
  h+=hwRow('память', m.percent, sz(m.used)+' из '+sz(m.total)+' свободно '+sz(m.available));
  if(s.total) h+=hwRow('подкачка', s.percent, sz(s.used)+' из '+sz(s.total));
  if(g.present){
    h+=hwRow('видеокарта', g.util, Math.round(g.util||0)+' %'+(g.power!==null?', '+Math.round(g.power)+' Вт':'')+', '+hwTemp(g.temp));
    h+=hwRow('память карты', g.mem_total?100*g.mem_used/g.mem_total:null,
      sz(g.mem_used*1048576)+' из '+sz(g.mem_total*1048576));
  }else{
    h+=hwRow('видеокарта', null, 'nvidia-smi не ответил');
  }
  if(io.device){
    h+=hwRow('диск: загрузка', io.busy,
      (io.busy===null||io.busy===undefined?'—':Math.round(io.busy)+' %')+
      ', чтение '+hwRate(io.read)+' / запись '+hwRate(io.write));
  }
  (j.disks||[]).forEach(d=>{ h+=hwRow('диск: место '+(d.label||(d.mount==='/'?'(система)':(d.mount==='/home'?'(домашний)':d.mount))),
    d.percent, sz(d.used)+' из '+sz(d.total)+(d.free!==undefined?', свободно '+sz(d.free):'')) });
  if(j.sata&&j.sata.mount&&!j.sata.mounted){
    h+=hwRow('диск: модели (отдельный диск)', null, 'не смонтирован — проверь шифрованный диск');
  }
  if(j.sata&&j.sata.temp!==null&&j.sata.temp!==undefined){
    h+=hwRow('диск: температура SATA', null,
      hwTemp(j.sata.temp)+(j.sata.model?', '+esc(j.sata.model.trim()):''));
  }
  const sub=[];
  if(c.name) sub.push('<b>процессор:</b> '+esc(c.name)+(c.cores?' ('+c.cores+' ядер, '+c.threads+' потоков)':''));
  if(c.load&&c.load.length) sub.push('<b>очередь задач:</b> '+c.load.map(x=>Number(x).toFixed(2)).join(' / ')+' (1 мин / 5 мин / 15 мин)');
  if(g.present&&g.name) sub.push('<b>видеокарта:</b> '+esc(g.name)+', '+sz(g.mem_total*1048576)+' видеопамяти');
  if(ssd.temp!==null&&ssd.temp!==undefined) sub.push('<b>диск:</b> '+hwTemp(ssd.temp)+(ssd.model?', '+esc(ssd.model.trim()):''));
  if(j.igpu!==null&&j.igpu!==undefined) sub.push('<b>встроенная графика:</b> '+hwTemp(j.igpu));
  if(g.present&&g.power!==null&&g.power!==undefined&&c.watts!==null&&c.watts!==undefined){
    sub.push('<b>питание:</b> процессор и карта вместе '+Math.round(c.watts+g.power)+' Вт; '+
      'память и материнская плата ватт не отдают — датчиков тока нет');
  }
  if((c.watts===null||c.watts===undefined)&&c.watts_ok===false){
    sub.push('<b>питание процессора:</b> счётчик энергии закрыт от панели (разово нужен root-хелпер)');
  }
  if(j.sata&&j.sata.mount&&(j.sata.temp===null||j.sata.temp===undefined)&&j.sata.sensor===false){
    sub.push('<b>диск SATA:</b> датчика температуры не видно — модуль drivetemp не загружен');
  }
  if(g.present&&g.apps&&g.apps.length){
    sub.push('<b>держит видеопамять:</b> '+g.apps.slice(0,4).map(a=>
      esc(a.name)+' '+Math.round(a.mem)+' МБ').join(', '));
  }
  if(j.top_cpu&&j.top_cpu.length){
    sub.push('<b>грузит процессор:</b> '+j.top_cpu.slice(0,4).map(x=>
      esc(x.name)+' '+Math.round(x.cpu)+'%').join(', '));
  }
  if(j.top_mem&&j.top_mem.length){
    sub.push('<b>занимает память:</b> '+j.top_mem.slice(0,4).map(x=>
      esc(x.name)+' '+Number(x.mem).toFixed(1)+'%').join(', '));
  }
  if(j.uptime) sub.push('<b>машина работает:</b> '+hwUp(j.uptime));
  if(!j.psutil) sub.push('<b>psutil не стоит:</b> часть цифр недоступна');
  h+='<div class="hwsub">'+sub.join('<br>')+'</div>';
  $('hwbody').innerHTML=h;
  hwHist(j.history);
  $('hwstate').textContent='данные с машины стенда';
  $('hwtime').textContent='обновлено '+new Date().toLocaleTimeString('ru-RU');
  $('hwerr').textContent='';
}
async function loadSys(){
  try{
    const r=await fetch('/api/sys');
    const j=await r.json();
    if(!r.ok){ $('hwerr').textContent=j.error||('HTTP '+r.status); return }
    renderSys(j);
  }catch(e){ $('hwerr').textContent='сервер не отвечает: '+e }
}
$('tabimg').onclick=()=>setMode('img');
$('tabvis').onclick=()=>setMode('vision');
$('tabchat').onclick=()=>setMode('chat');
$('tabtrn').onclick=()=>setMode('train');
$('tabvid').onclick=()=>setMode('video');
$('tabgrd').onclick=()=>setMode('grade');
$('tabdoc').onclick=()=>setMode('doc');
$('tabhw').onclick=()=>setMode('hw');

/* --- что на картинке: модель со зрением описывает кадр --- */
let VFILE=null, VBUSY=false, VMODELS=[], VDEFAULT='qwen3-vl:8b';
async function loadVision(){
  if(VMODELS.length) return;
  try{
    const j=await (await fetch('/api/config')).json();
    VMODELS=j.vision_models||[]; VDEFAULT=j.vision_default||VDEFAULT;
  }catch(e){ VMODELS=[] }
  const sel=$('qm');
  sel.innerHTML=VMODELS.length
    ? VMODELS.map(m=>`<option value="${esc(m.name)}">${esc(m.label)}</option>`).join('')
    : '<option value="">— нет моделей со зрением —</option>';
  const want=VMODELS.find(m=>m.name===VDEFAULT)||VMODELS[0];
  if(want) sel.value=want.name;
}
$('qphoto').onchange=async ev=>{
  const f=(ev.target.files||[])[0];
  if(!f) return;
  $('qerr').textContent=''; $('qout').value=''; $('qtime').textContent='';
  if(f.size>25*1024*1024){ $('qerr').textContent='файл больше 25 МБ'; VFILE=null; return }
  try{
    const b64=await new Promise((res,rej)=>{const r=new FileReader();
      r.onload=()=>res(String(r.result).split(',')[1]||''); r.onerror=rej; r.readAsDataURL(f)});
    VFILE={data:b64,name:f.name};
    $('qhint').textContent=f.name+' · '+(f.size/1048576).toFixed(1)+' МБ';
    $('qfr').innerHTML=`<img src="data:${f.type||'image/png'};base64,${b64}" alt="" style="max-width:100%;border-radius:8px;display:block">`;
  }catch(e){ $('qerr').textContent='не смог прочитать файл'; VFILE=null }
};
$('qgo').onclick=async()=>{
  if(VBUSY) return;
  if(!VFILE){ $('qerr').textContent='сначала выбери фотографию'; return }
  const model=$('qm').value;
  if(!model){ $('qerr').textContent='нет модели со зрением'; return }
  VBUSY=true; $('qgo').disabled=true; $('qerr').textContent=''; $('qout').value='';
  const t0=Date.now();
  const tick=setInterval(()=>{ $('qstate').textContent='смотрю… '+((Date.now()-t0)/1000).toFixed(0)+' с' },200);
  let j=null, err='';
  try{
    const r=await fetch('/api/vision',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model,lang:$('qlang').value,hint:($('qwhat').value||'').trim(),data:VFILE.data})});
    j=await r.json();
    if(!r.ok) err=j.error||('HTTP '+r.status);
  }catch(e){ err='сервер не отвечает: '+e }
  clearInterval(tick); VBUSY=false; $('qgo').disabled=false;
  if(err){ $('qstate').textContent='не вышло'; $('qerr').textContent=err; $('qtime').textContent=''; return }
  $('qout').value=j.text||'';
  $('qstate').textContent='готово'; $('qtime').textContent=(j.sec||0)+' с · '+(j.model||model);
};
$('qcopy').onclick=async()=>{
  const v=$('qout').value;
  if(!v){ $('qerr').textContent='сначала получи описание'; return }
  let ok=false;
  try{ await navigator.clipboard.writeText(v); ok=true }catch(e){
    try{ $('qout').select(); $('qout').setSelectionRange(0,v.length); ok=document.execCommand('copy') }catch(_){ ok=false }
  }
  $('qcopy').textContent=ok?'скопировано':'выделено — Ctrl+C';
  setTimeout(()=>{ $('qcopy').textContent='копировать' },2000);
};
$('qprompt').onclick=()=>{
  const t=($('qout').value||'').trim();
  if(!t){ $('qerr').textContent='сначала получи описание'; return }
  $('p').value=t; setMode('img'); $('p').focus();
};

/* --- цвет и яркость готового видео: ползунки --- */
const G_KEY='standui-grade-v1';
const G_P=['gamma','contrast','black','sat','shadow','sharp','cb','cr'];
let GTMR=null, GBUSY=false, GFILES={out:[],in:[]};
function gRead(){
  const p={};
  G_P.forEach(k=>{ p[k]=parseFloat(($('g_'+k).value||'0')) });
  return p;
}
function gShow(){
  const p=gRead();
  G_P.forEach(k=>{
    const v=(k==='cb'||k==='cr')?((p[k]>0?'+':'')+p[k].toFixed(1)):String(Math.round(p[k]));
    const el=$('g_'+k+'_v'); if(el) el.textContent=v;
  });
  try{ localStorage.setItem(G_KEY, JSON.stringify(p)) }catch(e){}
}
function gRestore(){
  let raw=null; try{ raw=localStorage.getItem(G_KEY) }catch(e){}
  if(!raw) return;
  try{
    const p=JSON.parse(raw)||{};
    G_P.forEach(k=>{ if(p[k]!==undefined&&$('g_'+k)) $('g_'+k).value=p[k] });
  }catch(e){}
}
function gFill(el,items,cur){
  el.innerHTML=items.length
    ? '<option value="">— выбери файл —</option>'+items.map(x=>`<option value="${esc(x.name)}">${esc(x.name)}${x.hint?' · '+esc(x.hint):''}</option>`).join('')
    : '<option value="">— нет файлов —</option>';
  if(cur&&items.some(x=>x.name===cur)) el.value=cur;
}
function renderReady(items){
  $('g_list').innerHTML=(items&&items.length)
    ? items.map(n=>`<div class="vitem"><video src="/vg/${encodeURIComponent(n)}" preload="metadata" controls></video>`+
        `<div class="vn">${esc(n)}</div><div class="vact">`+
        `<a href="/vg/${encodeURIComponent(n)}?dl=1">скачать</a>`+
        `<a href="/vg/${encodeURIComponent(n)}" target="_blank">открыть</a></div></div>`).join('')
    : '<span class="empty">пока пусто</span>';
}
async function gradeLoad(){
  gRestore(); gShow();
  let j=null;
  try{ j=await (await fetch('/api/grade')).json() }catch(e){ $('g_info').textContent='сервер не отвечает: '+e; return }
  GFILES=j.files||{out:[],in:[]};
  const outs=(GFILES.out||[]).filter(x=>!/^grade_/.test(x.name));
  const curF=$('gf').value, curS=$('gs').value;
  const pick=(items,re)=>((items.find(x=>re.test(x.name))||items[0]||{}).name||'');
  gFill($('gf'),outs,curF||pick(outs,/vace_gray_full|vace_full_mp4|vace_canny/));
  gFill($('gs'),GFILES.in||[],curS||pick(GFILES.in||[],/vace_lossless|vace_src832|vace_src/));
  renderReady(j.ready||[]);
}
async function gPreview(){
  const file=$('gf').value, src=$('gs').value;
  if(!file||!src){ $('g_info').textContent='выбери файл и откуда брать палитру'; return }
  if(GBUSY) return;
  GBUSY=true; $('g_info').textContent='считаю пример…';
  try{
    const body={file,src,frame:parseInt($('g_frame').value||'0',10)};
    Object.assign(body,gRead());
    const r=await fetch('/api/grade/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json();
    if(!r.ok){ $('g_info').textContent=j.error||('HTTP '+r.status); GBUSY=false; return }
    $('gfr').innerHTML=`<img src="${j.jpeg}" style="width:100%;border-radius:8px;display:block" alt="">`+
      `<img src="${j.jpeg_src}" style="width:100%;border-radius:8px;display:block;margin-top:6px;opacity:.6" alt="">`;
    $('g_info').textContent=`кадр ${j.frame+1} из ${j.frames}: яркость ${j.stats.y} (было ${j.stats0.y}), медиана ${j.stats.med} (${j.stats0.med}), цвет в тенях ${j.stats.shadow} (${j.stats0.shadow})`;
  }catch(e){ $('g_info').textContent='сервер не отвечает: '+e }
  GBUSY=false;
}
async function gApply(){
  const file=$('gf').value, src=$('gs').value;
  if(!file||!src){ $('g_err').textContent='выбери файл и откуда брать палитру'; return }
  $('g_apply').disabled=true; $('g_err').textContent=''; $('g_state').textContent='считаю видео…';
  try{
    const body={file,src}; Object.assign(body,gRead());
    const r=await fetch('/api/grade/apply',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json();
    if(!r.ok){ $('g_err').textContent=j.error||('HTTP '+r.status); $('g_state').textContent='ошибка' }
    else { $('g_state').textContent='готово за '+Math.round(j.sec)+' с, кадров '+j.frames; await gradeLoad() }
  }catch(e){ $('g_err').textContent='сервер не отвечает: '+e; $('g_state').textContent='ошибка' }
  $('g_apply').disabled=false;
}
G_P.forEach(k=>{
  const el=$('g_'+k); if(!el) return;
  el.oninput=()=>{ gShow(); clearTimeout(GTMR); GTMR=setTimeout(gPreview,400) };
});
$('g_prev').onclick=gPreview;
$('g_apply').onclick=gApply;
$('gf').onchange=()=>{ gPreview() };
$('gs').onchange=()=>{ gPreview() };

/* --- улучшение видео на видеокарте (SCUNet, Real-ESRGAN, RIFE) --- */
let VPOL=null, VCLIENT='', VSRC='', VLIST=[], VDONE='', VSIG='', VSRCLIST=[];
function stopVideoPoll(){ if(VPOL){ clearInterval(VPOL); VPOL=null } }
function startVideoPoll(){ stopVideoPoll(); VPOL=setInterval(loadVideo,3000) }
function fmtDur(s){ s=Math.round(s||0); if(!s) return '';
  return Math.floor(s/60)+':'+String(s%60).padStart(2,'0') }
function isPic(n){ return /\.(png|jpe?g|webp)$/i.test(n||'') }
function fmtMeta(x){
  // «MP4 · H264 · 640x360 · 489 кадров · 7.8 к/с · 1:02» — в подсказку
  if(!x) return '';
  const p=[];
  if(x.container) p.push(x.container);
  if(x.codec) p.push(x.codec);
  if(x.w&&x.h) p.push(x.w+'x'+x.h);
  if(x.frames) p.push(x.frames+' кадров');
  if(x.fps) p.push(x.fps+' к/с');
  if(x.dur) p.push(fmtDur(x.dur));
  return p.join(' · ');
}
function fmtShort(x){
  // «MP4 · 7.8 к/с» — в списке, чтобы не растягивать
  if(!x) return '';
  const p=[];
  const c=[x.container,x.codec].filter(Boolean).join(' ');
  if(c) p.push(c);
  if(x.fps) p.push(x.fps+' к/с');
  return p.join(' · ');
}
function renderVInfo(){
  const name=($('vsrc').value||'').trim();
  const box=$('vinfo');
  if(!box) return;
  const src=(VSRCLIST||[]).find(s=>s.name===name);
  if(!src){ box.textContent=''; return }
  const bits=[];
  if(src.container||src.codec) bits.push([src.container,src.codec].filter(Boolean).join(' / '));
  if(src.w&&src.h) bits.push(src.w+'x'+src.h);
  if(src.frames) bits.push(src.frames+' кадров');
  if(src.fps) bits.push(src.fps+' к/с');
  if(src.dur) bits.push(fmtDur(src.dur));
  bits.push((src.bytes/1048576).toFixed(1)+' МБ');
  box.innerHTML='<b>'+esc(name)+'</b> — '+esc(bits.join(' · '));
}
function vSourceName(){
  const sel=($('vsrc').value||'').trim();
  return VSRC||sel;
}
/* --- куски ролика по качеству: разбор, правка таймкодов, обработка только плохих --- */
let VSEGSIG='';
function tcFmt(t){ t=Math.max(0,t||0); const m=Math.floor(t/60), s=t-m*60;
  return String(m).padStart(2,'0')+':'+(s<10?'0':'')+s.toFixed(1) }
function tcParse(v){ v=String(v||'').trim().replace(',','.');
  if(/^\d+(\.\d+)?$/.test(v)) return Number(v);
  const p=v.split(':').map(Number); if(p.some(isNaN)) return NaN;
  if(p.length===2) return p[0]*60+p[1];
  if(p.length===3) return p[0]*3600+p[1]*60+p[2];
  return NaN }
function segNote(sp){
  const st=(sp&&sp.state)||'idle';
  if(st==='running') return 'разбираю по кадрам… '+((sp&&sp.sec)||0)+' с';
  if(st==='error') return 'разбор не вышел: '+((sp&&sp.err)||'').split('\n').pop();
  return '';
}
function renderSplit(sp){
  const box=$('vsegs'); if(!box) return;
  const note=$('vspnote');
  const name=vSourceName();
  note.textContent=segNote(sp);
  const ready=sp&&sp.state==='ready'&&sp.name===name&&(sp.segments||[]).length;
  if(!ready){ box.style.display='none'; VSEGSIG=''; return }
  const segs=sp.segments||[];
  const sig=sp.name+'|'+segs.map(s=>[s.t0,s.t1,s.bad,s.reason].join(',')).join(';');
  box.style.display='';
  $('vsegcnt').textContent=segs.length+' шт — плохих '+
    segs.filter(s=>s.bad).length+' ('+Math.round(segs.filter(s=>s.bad)
      .reduce((a,s)=>a+(s.t1-s.t0),0))+' с)';
  if(sig===VSEGSIG) return;
  VSEGSIG=sig;
  $('vseglist').innerHTML=segs.map((s,i)=>{
    const m=[];
    m.push('размытие '+s.blur.toFixed(1));
    m.push('блоки '+s.block.toFixed(2));
    if(s.kbps) m.push(Math.round(s.kbps)+' кбит/с');
    return '<div class="vseg" data-i="'+i+'" data-bad="'+(s.bad?1:0)+'" '+
      'style="display:flex;gap:8px;align-items:center;margin:4px 0;flex-wrap:wrap">'+
      '<label style="display:flex;gap:6px;align-items:center;color:var(--tx)">'+
      '<input type="checkbox" class="vsegchk" style="width:auto"'+(s.bad?' checked':'')+'>'+
      '<input class="vsegfrom" value="'+tcFmt(s.t0)+'" size="7" '+
      'style="width:72px;padding:2px 4px"> – '+
      '<input class="vsegto" value="'+tcFmt(s.t1)+'" size="7" '+
      'style="width:72px;padding:2px 4px"></label>'+
      '<span class="vs">'+esc(m.join(' · '))+'</span>'+
      (s.bad?'<span class="vs" style="color:#e5a50a">плохой'+(s.reason?': '+esc(s.reason):'')+'</span>'
      :'<span class="vs">годный</span>')+'</div>';
  }).join('');
}
function renderVideo(j){
  if(!j) return;
  renderSplit(j.split);
  const list=(j.results||[]);
  const tot=Number(j.results_total||list.length);
  $('vrcnt').textContent=list.length?(tot>list.length?('последние '+list.length+' из '+tot):(tot+' шт')):'пока пусто';
  // ленту перерисовываем ТОЛЬКО когда набор файлов изменился,
  // иначе браузер каждые 3 секунды начинает просмотр заново
  const sig=list.map(x=>x.name+'|'+x.mtime+'|'+x.bytes).join(';');
  if(sig!==VSIG){
    VSIG=sig;
    $('vrlist').innerHTML=list.length?list.map(x=>`<div class="vitem">
      ${isPic(x.name)
        ? `<img src="/vid/${encodeURI(x.name)}" alt="" style="width:100%;border-radius:6px;display:block">`
        : `<video src="/vid/${encodeURI(x.name)}" controls preload="metadata"></video>`}
      <div class="vn">${esc(x.name)}</div>
      <div class="vs" title="${esc(fmtMeta(x))}">${esc(fmtShort(x))||((x.bytes/1048576).toFixed(1)+' МБ')}</div>
      <div class="vact">
        <a href="/vid/${encodeURI(x.name)}?dl=1" download="${esc(x.name)}">скачать</a>
        <button class="mini" data-vdel="${esc(x.name)}" title="удалить файл со стенда">×</button>
      </div></div>`).join('')
      :'<span class="empty">ещё ничего не улучшено — выбери файл выше и нажми «улучшить видео»</span>';
  }
  const srcs=(j.sources||[]);
  VSRCLIST=srcs;
  const names=srcs.map(s=>s.name);
  const cur=$('vsrc').value;
  if(JSON.stringify(names)!==JSON.stringify(VLIST)){
    VLIST=names;
    $('vsrc').innerHTML=srcs.length
      ? srcs.map(s=>{
          const short=[];
          if(s.container||s.codec) short.push([s.container,s.codec].filter(Boolean).join(' '));
          if(s.fps) short.push(s.fps+' к/с');
          return `<option value="${esc(s.name)}" title="${esc(fmtMeta(s))}">${esc(s.name)} — ${esc(short.join(' · ')||'—')}</option>`;
        }).join('')
      : '<option value="">— нет файлов —</option>';
    if(names.includes(cur)) $('vsrc').value=cur;
  }
  $('vdel').disabled=!(($('vsrc').value||'').trim());
  renderVInfo();
  const st=j.state||'idle';
  const names2={idle:'готов',running:'обрабатываю',done:'готово',error:'ошибка',stopped:'остановлено'};
  let line=names2[st]||st;
  if(st==='running'&&j.total) line+=' — кадр '+j.done+' из '+j.total;
  if(st==='done'&&j.out) line+=': '+j.out;
  if(VCLIENT) line=VCLIENT;
  $('vstate').textContent=line;
  $('vpb').style.width=(st==='done'?100:(j.pct||0))+'%';
  $('vtime').textContent=(st==='running'&&j.left)?('осталось ~'+humanSec(j.left)):(j.sec?humanSec(j.sec):'');
  $('verr').textContent=j.error||'';
  $('vlog').textContent=(st==='running'||st==='error')?(j.log||''):'';
  $('vstop').style.display=(st==='running')?'':'none';
  $('vgo').disabled=(st==='running');
  if($('vsegrun')) $('vsegrun').disabled=(st==='running');
  if($('vsplit')) $('vsplit').disabled=(st==='running');
  if(st==='done'&&j.out&&VDONE!==j.out){
    VDONE=j.out;
    setTimeout(loadVideo,500);
  }
}
async function loadVideo(){
  let j=null;
  try{ j=await (await fetch('/api/video')).json() }catch(e){ return }
  renderVideo(j);
}
async function videoDelete(kind,name){
  const what=kind==='out'?'готовый файл':'оригинал';
  if(!confirm('Удалить '+what+' «'+name+'» со стенда? Вернуть будет нельзя.')) return;
  try{
    const r=await fetch('/api/video/delete',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({kind,name})});
    const j=await r.json();
    if(!r.ok){ $('verr').textContent=j.error||('HTTP '+r.status); return }
    $('verr').textContent='';
    if(kind==='in'){ VSRC=''; $('vfhint').textContent='оригинал «'+name+'» удалён со стенда' }
  }catch(e){ $('verr').textContent='сервер не отвечает: '+e; return }
  await loadVideo();
}
$('vrlist').addEventListener('click',ev=>{
  const b=ev.target.closest('[data-vdel]');
  if(b) videoDelete('out', b.getAttribute('data-vdel'));
});
$('vdel').onclick=()=>{
  const name=(($('vsrc').value||'')||'').trim();
  if(!name){ $('verr').textContent='сначала выбери оригинал в списке'; return }
  videoDelete('in', name);
};
$('vfile').onchange=async ev=>{
  const f=ev.target.files&&ev.target.files[0];
  if(!f) return;
  if(f.size>400*1024*1024){ VCLIENT='файл больше 400 МБ — положи его на стенд в папку in';
    $('verr').textContent=VCLIENT; $('vfile').value=''; return }
  VCLIENT=''; $('verr').textContent='';
  $('vfhint').textContent='загружаю '+f.name+' ('+(f.size/1048576).toFixed(1)+' МБ)…';
  $('vfile').disabled=true;
  try{
    const b64=await new Promise((res,rej)=>{const r=new FileReader();
      r.onload=()=>res(String(r.result).split(',')[1]||''); r.onerror=rej; r.readAsDataURL(f)});
    const r=await fetch('/api/video/upload',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name:f.name,data:b64})});
    const j=await r.json();
    if(!r.ok){ $('vfhint').textContent='не принял: '+(j.error||r.status) }
    else{ VSRC=j.name; $('vfhint').textContent='на стенде: '+j.name; await loadVideo();
      if([...$('vsrc').options].some(o=>o.value===j.name)) $('vsrc').value=j.name; }
  }catch(e){ $('vfhint').textContent='загрузка не удалась: '+e }
  $('vfile').value='';
  $('vfile').disabled=false;
};
$('vsplit').onclick=async()=>{
  const name=(($('vsrc').value||'')||VSRC).trim();
  if(!name){ $('verr').textContent='сначала выбери файл'; return }
  if(isPic(name)){ $('verr').textContent='разбивка работает с видео, а не с картинкой'; return }
  $('vsplit').disabled=true; $('verr').textContent='';
  try{
    const r=await fetch('/api/video/split',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name})});
    const j=await r.json();
    if(!r.ok){ $('verr').textContent=j.error||('HTTP '+r.status) }
    else renderVideo(j);
  }catch(e){ $('verr').textContent='сервер не отвечает: '+e }
  $('vsplit').disabled=false;
  startVideoPoll();
};
$('vsegmarks').onclick=()=>{
  document.querySelectorAll('#vseglist .vseg').forEach(d=>{
    const c=d.querySelector('.vsegchk'); if(c) c.checked=d.getAttribute('data-bad')==='1' });
};
$('vsegnone').onclick=()=>{
  document.querySelectorAll('#vseglist .vsegchk').forEach(c=>{ c.checked=false });
};
function readSegs(){
  const out=[];
  document.querySelectorAll('#vseglist .vseg').forEach(d=>{
    const a=tcParse(d.querySelector('.vsegfrom').value);
    const b=tcParse(d.querySelector('.vsegto').value);
    const on=d.querySelector('.vsegchk').checked;
    if(isNaN(a)||isNaN(b)||b<=a) out.push(null);
    else out.push({t0:a,t1:b,enhance:on});
  });
  return out;
}
$('vsegrun').onclick=async()=>{
  const name=(($('vsrc').value||'')||VSRC).trim();
  if(!name){ $('verr').textContent='сначала выбери файл'; return }
  const segs=readSegs();
  if(segs.some(s=>s===null)){ $('verr').textContent='в таймкодах ошибка: пиши мм:сс.д, «до» больше «от»'; return }
  const bad=segs.filter(s=>s.enhance);
  if(!bad.length){ $('verr').textContent='не отмечено ни одного куска'; return }
  VCLIENT=''; $('verr').textContent=''; VDONE='';
  $('vsegrun').disabled=true;
  let err='';
  try{
    const r=await fetch('/api/video/start',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name,segments:segs,scunet:$('vsc').checked,
        tstab:$('vstab').checked?0.9:0,color_fix:$('vcfix').checked,crf:18})});
    const j=await r.json();
    if(!r.ok) err=j.error||('HTTP '+r.status); else renderVideo(j);
  }catch(e){ err='сервер не отвечает: '+e }
  if(err){ $('vsegrun').disabled=false; $('verr').textContent=err; return }
  startVideoPoll(); setTimeout(loadVideo,800);
};
$('vgo').onclick=async()=>{
  const name=(($('vsrc').value||'')||VSRC).trim();
  if(!name){ VCLIENT='сначала выбери файл'; $('verr').textContent=VCLIENT; return }
  VCLIENT=''; $('verr').textContent=''; VDONE='';
  $('vgo').disabled=true;
  let err='';
  try{
    const r=await fetch('/api/video/start',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name,scunet:$('vsc').checked,rife:$('vrife').checked,
        tstab:$('vstab').checked?0.9:0, color:$('vcolor').checked,
        color_smooth:0.55, color_sat:1.0,
        esrgan:true,scale:Number($('vscale').value)||2,limit:Number($('vlimit').value)||0})});
    const j=await r.json();
    if(!r.ok) err=j.error||('HTTP '+r.status); else renderVideo(j);
  }catch(e){ err='сервер не отвечает: '+e }
  if(err){ $('vgo').disabled=false; $('verr').textContent=err; return }
  startVideoPoll(); setTimeout(loadVideo,800);
};
$('vsrc').onchange=()=>{ VSRC=''; renderVInfo(); VSEGSIG=''; loadVideo() };
$('vstop').onclick=async()=>{
  if(!confirm('Остановить обработку?')) return;
  $('vstop').disabled=true;
  $('vstate').textContent='останавливаю…';
  try{
    const r=await fetch('/api/video/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    const j=await r.json();
    if(!r.ok) $('verr').textContent=j.error||('HTTP '+r.status);
    else $('verr').textContent='';
  }catch(e){ $('verr').textContent='сервер не отвечает: '+e }
  $('vstop').disabled=false;
  await loadVideo();
  setTimeout(loadVideo,2000);
};
$('csend').onclick=sendChat;
$('cq').addEventListener('keydown',e=>{ if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)) sendChat() });
$('cclr').onclick=()=>{ CHAT=[]; renderChat(); saveChat(); $('cstate').textContent='' };
$('cfree').onclick=async()=>{ $('cfree').disabled=true;
  try{await fetch('/api/unload',{method:'POST'})}catch(e){}
  $('cfree').disabled=false; $('cstate').textContent='модель выгружена';
};
boot();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = 'stand-ui/1.0'

    def log_message(self, fmt, *args):
        print('%s %s' % (self.address_string(), fmt % args), flush=True)

    def _send(self, code, body, ctype='application/json'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split('?')[0]
        if path in ('/', '/index.html'):
            return self._send(200, HOME, 'text/html; charset=utf-8')
        if path == '/api/config':
            videos = video_model_list()
            fams = [m['family'] for m in videos] or ['ltxv']
            return self._send(200, {'models': model_list(),
                                    'video_models': videos,
                                    'sizes': SIZES,
                                    'video_sizes': {f: SIZES_VIDEO[f] for f in fams},
                                    'video_defaults': {f: VIDEO_FAMILY[f] for f in fams},
                                    'video_hint': {f: VIDEO_HINT[f] for f in fams},
                                    'negative': NEG_DEFAULT,
                                    'negative_video': NEG_VIDEO,
                                    'image_hint': IMG_HINT,
                                     'loras': lora_list(),
                                    'translators': ollama_models(),
                                    'vision_models': vision_models(),
                                    'vision_default': VISION_DEFAULT,
                                     'default_translator': TRANSLATOR,
                                     'ctx_choices': CHAT_CTX_CHOICES,
                                     'ctx_default': CHAT_CTX,
                                     'ctx_max': CHAT_CTX_MAX,
                                     'comfy': API})
        if path == '/api/train':
            return self._send(200, train_status())
        if path == '/api/video':
            return self._send(200, video_status())
        if path == '/api/grade':
            return self._send(200, {'files': _grade_files(), 'defaults': GRADE_DEFAULTS,
                                    'ready': grade_ready(), 'limit': GRADE_MAXSIDE})
        if path == '/api/doc':
            return self._send(200, doc_status())
        if path == '/api/sys':
            st = system_status()
            with _SYS_LOCK:
                st['history'] = list(_SYS_HIST)
            return self._send(200, st)
        if path.startswith('/vg/'):
            name = urllib.parse.unquote(path[4:])
            full = os.path.join(GRADE_OUT, name)
            if (not name or name != os.path.basename(name) or '..' in name
                    or not os.path.isfile(full)):
                return self._send(404, {'error': 'нет файла'})
            dl = 'dl=1' in (self.path.split('?', 1)[1] if '?' in self.path else '')
            size = os.path.getsize(full)
            rng = '' if dl else (self.headers.get('Range') or '')
            start, end = 0, size - 1
            if rng.startswith('bytes='):
                a, _, b = rng[6:].partition('-')
                try:
                    start = int(a) if a else 0
                    end = int(b) if b else size - 1
                except ValueError:
                    start, end = 0, size - 1
            end = min(end, size - 1)
            self.send_response(206 if rng else 200)
            self.send_header('Content-Type', 'video/mp4')
            if dl:
                ascii_name = re.sub(r'[^\w.\-]+', '_', name) or 'video.mp4'
                self.send_header('Content-Disposition',
                                 'attachment; filename="%s"; filename*=UTF-8\'\'%s'
                                 % (ascii_name, urllib.parse.quote(name)))
            self.send_header('Accept-Ranges', 'bytes')
            if rng:
                self.send_header('Content-Range', 'bytes %d-%d/%d' % (start, end, size))
            self.send_header('Content-Length', str(end - start + 1))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            with open(full, 'rb') as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(262144, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            return
        if path.startswith('/vid/'):
            name = urllib.parse.unquote(path[5:])
            full = os.path.join(VIDEO_OUT, name)
            if (not name or name != os.path.basename(name) or '..' in name
                    or not os.path.isfile(full)):
                return self._send(404, {'error': 'нет файла'})
            dl = 'dl=1' in (self.path.split('?', 1)[1] if '?' in self.path else '')
            size = os.path.getsize(full)
            low = name.lower()
            ctype = 'video/mp4'
            if low.endswith('.png'):
                ctype = 'image/png'
            elif low.endswith(('.jpg', '.jpeg')):
                ctype = 'image/jpeg'
            elif low.endswith('.webp'):
                ctype = 'image/webp'
            rng = '' if dl else (self.headers.get('Range') or '')
            self.send_response(206 if rng else 200)
            self.send_header('Content-Type', ctype)
            if dl:
                ascii_name = re.sub(r'[^\w.\-]+', '_', name) or 'video.mp4'
                self.send_header('Content-Disposition',
                                 'attachment; filename="%s"; filename*=UTF-8\'\'%s'
                                 % (ascii_name, urllib.parse.quote(name)))
            self.send_header('Accept-Ranges', 'bytes')
            start, end = 0, size - 1
            if rng.startswith('bytes='):
                try:
                    a, _, b = rng[6:].partition('-')
                    if a:
                        start = int(a)
                    if b:
                        end = int(b)
                except ValueError:
                    start, end = 0, size - 1
            end = min(end, size - 1)
            if rng:
                self.send_header('Content-Range', 'bytes %d-%d/%d' % (start, end, size))
            self.send_header('Content-Length', str(end - start + 1))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            with open(full, 'rb') as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(262144, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            return
        if path.startswith('/api/train/photo/'):
            nm = urllib.parse.unquote(path[len('/api/train/photo/'):])
            if (not nm or nm != os.path.basename(nm)
                    or not nm.lower().endswith(PHOTO_EXT)):
                return self._send(404, {'error': 'нет такой фотографии'})
            full = os.path.join(TRAIN_PHOTOS, nm)
            if not os.path.isfile(full):
                return self._send(404, {'error': 'нет такой фотографии'})
            with open(full, 'rb') as f:
                data = f.read()
            ctype = 'image/png' if nm.lower().endswith('.png') else 'image/jpeg'
            return self._send(200, data, ctype)
        if path == '/api/history':
            return self._send(200, load_history())
        if path == '/api/queue':
            return self._send(200, queue_snapshot())
        if path.startswith('/api/job/'):
            with JOBS_LOCK:
                job = JOBS.get(path.rsplit('/', 1)[-1])
                if not job:
                    return self._send(404, {'error': 'нет такой задачи'})
                out = {k: job.get(k) for k in
                       ('state', 'phase', 'error', 'image', 'prompt_en', 'note',
                        'seed', 'sec', 'prompt_id', 'start_image', 'end_image')}
                out.update(w=job['req']['w'], h=job['req']['h'], steps=job['req']['steps'],
                           prompt_ru=job['req']['prompt'],
                            kind=job['req'].get('kind', 'image'),
                            denoise=job['req'].get('denoise'),
                            grow=job['req'].get('grow'),
                            detector=job['req'].get('detector'),
                            strength=job['req'].get('strength'),
                            ctrl_type=job['req'].get('ctrl_type'),
                            upscale=job['req'].get('upscale'),
                             lora=job['req'].get('lora'),
                             lora_weight=job['req'].get('lora_weight'),
                             length=job['req'].get('length'), fps=job['req'].get('fps'))
            return self._send(200, out)
        if path.startswith('/img/'):
            name = urllib.parse.unquote(path[5:])
            full = os.path.join(OUT_DIR, name)
            if (not name or name.startswith('/') or '..' in name
                    or not os.path.isfile(full)):
                return self._send(404, {'error': 'нет файла'})
            with open(full, 'rb') as f:
                data = f.read()
            ctype = 'video/mp4' if full.lower().endswith('.mp4') else 'image/png'
            return self._send(200, data, ctype)
        return self._send(404, {'error': 'нет такого адреса'})

    def do_POST(self):
        path = self.path.split('?')[0]
        length = int(self.headers.get('Content-Length') or 0)
        raw = self.rfile.read(length) if length else b'{}'
        try:
            req = json.loads(raw or b'{}')
        except Exception:
            return self._send(400, {'error': 'не разобрал json'})

        if path == '/api/train/photo':
            data = (req.get('data') or '').strip()
            if data.startswith('data:') and ',' in data[:128]:
                data = data.split(',', 1)[1]
            if not data:
                return self._send(400, {'error': 'пустой файл'})
            try:
                raw = base64.b64decode(data + '=' * (-len(data) % 4))
            except Exception as e:
                return self._send(400, {'error': 'не разобрал файл: %s' % e})
            if len(raw) > 25 * 1024 * 1024:
                return self._send(400, {'error': 'файл больше 25 МБ'})
            if raw[:3] == b'\xff\xd8\xff':
                ext = '.jpg'
            elif raw[:8] == b'\x89PNG\r\n\x1a\n':
                ext = '.png'
            elif raw[:4] == b'RIFF' and raw[8:12] == b'WEBP':
                ext = '.webp'
            else:
                return self._send(400, {'error': 'это не png, jpeg или webp'})
            os.makedirs(TRAIN_PHOTOS, exist_ok=True)
            nm = 'ph-%s%s' % (uuid.uuid4().hex[:12], ext)
            dest = os.path.join(TRAIN_PHOTOS, nm)
            with open(dest, 'wb') as f:
                f.write(raw)
            os.chmod(dest, 0o644)
            # новая фотография получает ту же подпись, что и остальной набор
            cap = caption_read()
            if cap:
                caption_write(cap)
            return self._send(200, {'name': nm, 'count': len(photos_list())})

        if path == '/api/train/caption':
            text = (req.get('text') or '').strip()
            if len(text) > 400:
                return self._send(400, {'error': 'подпись длинновата, до 400 символов'})
            n = caption_write(text)
            return self._send(200, {'ok': True, 'count': n, 'caption': caption_read()})

        if path == '/api/train/photo/delete':
            nm = (req.get('name') or '').strip()
            if (not nm or nm != os.path.basename(nm)
                    or not nm.lower().endswith(PHOTO_EXT)):
                return self._send(404, {'error': 'нет такой фотографии'})
            try:
                os.remove(os.path.join(TRAIN_PHOTOS, nm))
            except Exception:
                return self._send(404, {'error': 'нет такой фотографии'})
            cap = caption_path(nm)
            if os.path.isfile(cap):
                try:
                    os.remove(cap)
                except Exception:
                    pass
            return self._send(200, {'ok': True, 'count': len(photos_list())})

        if path == '/api/train/clear':
            removed = 0
            for x in photos_list():
                try:
                    os.remove(os.path.join(TRAIN_PHOTOS, x['name']))
                    removed += 1
                except Exception:
                    pass
                cap = caption_path(x['name'])
                if os.path.isfile(cap):
                    try:
                        os.remove(cap)
                    except Exception:
                        pass
            return self._send(200, {'ok': True, 'removed': removed})

        if path == '/api/train/start':
            ok, why = train_start((req.get('name') or '').strip().lower(),
                                  req.get('steps'), req.get('dim'), req.get('base'))
            if not ok:
                return self._send(400, {'error': why})
            return self._send(200, {'ok': True})

        if path == '/api/train/stop':
            ok, why = train_stop()
            if not ok:
                return self._send(500, {'error': why})
            return self._send(200, {'ok': True})

        if path == '/api/lora/delete':
            nm = (req.get('name') or '').strip()
            if not lora_ok(nm):
                return self._send(404, {'error': 'нет такого стиля'})
            try:
                subprocess.run(['sudo', '-n', 'rm', '-f',
                                os.path.join(LORAS_DIR, nm)], capture_output=True, timeout=30)
            except Exception as e:
                return self._send(500, {'error': str(e)})
            return self._send(200, {'ok': True})

        if path == '/api/delete':
            name = (req.get('file') or '').strip()
            with HIST_LOCK:
                items = load_history()
                keep = [x for x in items if x.get('file') != name]
            if len(keep) == len(items):
                return self._send(404, {'error': 'нет такой записи'})
            removed = drop_file(name) if req.get('files', True) else 0
            with HIST_LOCK:
                save_history(keep)
            return self._send(200, {'ok': True, 'files': removed})

        if path == '/api/clear':
            with HIST_LOCK:
                items = load_history()
                removed = 0
                if req.get('files'):
                    for x in items:
                        removed += drop_file(x.get('file'))
                save_history([])
            return self._send(200, {'ok': True, 'count': len(items), 'files': removed})

        if path == '/api/unload':
            threading.Thread(target=unload_ollama, daemon=True).start()
            return self._send(200, {'ok': True})

        if path.startswith('/api/doc/'):
            what = path[len('/api/doc/'):]
            if what == 'start':
                ok, why = _doc_run('ui-on')
            elif what == 'stop':
                ok, why = _doc_run('ui-off')
            elif what == 'build':
                ok, why = _doc_run('build')
            elif what == 'wipe':
                if not req.get('confirm'):
                    return self._send(400, {'error': 'без подтверждения не сношу'})
                ok, why = _doc_run('wipe')
            else:
                return self._send(404, {'error': 'неизвестное действие'})
            if not ok:
                return self._send(409, {'error': why})
            return self._send(200, {'ok': True})

        if path == '/api/interrupt':
            for job in list(JOBS.values()):
                if job.get('state') == 'running':
                    job['cancelled'] = True
            try:
                rq = urllib.request.Request(API + '/interrupt', b'{}',
                                            {'Content-Type': 'application/json'})
                urllib.request.urlopen(rq, timeout=20).read()
            except Exception as e:
                return self._send(200, {'ok': False, 'error': 'ComfyUI не ответил: %s' % e})
            return self._send(200, {'ok': True})

        if path == '/api/upload':
            data = (req.get('data') or '').strip()
            if data.startswith('data:') and ',' in data[:128]:
                data = data.split(',', 1)[1]
            if not data:
                return self._send(400, {'error': 'пустой файл'})
            try:
                raw = base64.b64decode(data + '=' * (-len(data) % 4))
            except Exception as e:
                return self._send(400, {'error': 'не разобрал картинку: %s' % e})
            if len(raw) > 12 * 1024 * 1024:
                return self._send(400, {'error': 'картинка больше 12 МБ, уменьши'})
            if raw[:3] == b'\xff\xd8\xff':
                ext = '.jpg'
            elif raw[:8] == b'\x89PNG\r\n\x1a\n':
                ext = '.png'
            else:
                return self._send(400, {'error': 'это не png и не jpeg'})
            name = 'up-%s%s' % (uuid.uuid4().hex[:12], ext)
            dest = os.path.join(INPUT_DIR, name)
            try:
                with open(dest, 'wb') as f:
                    f.write(raw)
                os.chmod(dest, 0o644)
            except Exception as e:
                return self._send(500, {'error': 'не смог сохранить: %s' % e})
            return self._send(200, {'name': name, 'bytes': len(raw)})

        if path == '/api/video/upload':
            data = (req.get('data') or '').strip()
            if data.startswith('data:') and ',' in data[:128]:
                data = data.split(',', 1)[1]
            if not data:
                return self._send(400, {'error': 'пустой файл'})
            try:
                raw = base64.b64decode(data + '=' * (-len(data) % 4))
            except Exception as e:
                return self._send(400, {'error': 'не разобрал файл: %s' % e})
            if len(raw) > 400 * 1024 * 1024:
                return self._send(400, {'error': 'файл больше 400 МБ — положи его на стенд в папку in'})
            name = os.path.basename((req.get('name') or '').strip().replace('\\', '/'))
            if not name.lower().endswith(VIDEO_EXT + PHOTO_IN_EXT):
                return self._send(400, {'error': 'нужен видеофайл (mp4, mkv, mov, webm) или картинка (png, jpg)'})
            name = re.sub(r'[^\w.\- ]+', '_', name)[:120]
            if not name.lower().endswith(VIDEO_EXT + PHOTO_IN_EXT):
                name += '.mp4'
            _vdirs()
            try:
                with open(os.path.join(VIDEO_IN, name), 'wb') as f:
                    f.write(raw)
            except Exception as e:
                return self._send(500, {'error': 'не смог сохранить: %s' % e})
            return self._send(200, {'name': name, 'bytes': len(raw)})

        if path == '/api/video/split':
            ok, why = video_split_start(req.get('name'))
            if not ok:
                return self._send(400, {'error': why})
            return self._send(200, video_status())

        if path == '/api/video/start':
            ok, why = video_start(req)
            if not ok:
                return self._send(400, {'error': why})
            return self._send(200, video_status())

        if path == '/api/video/stop':
            ok, why = video_stop()
            if not ok:
                return self._send(500, {'error': why})
            return self._send(200, video_status())

        if path == '/api/video/delete':
            kind = (req.get('kind') or 'out').strip()
            name = os.path.basename((req.get('name') or '').strip())
            if kind not in ('in', 'out'):
                return self._send(400, {'error': 'kind бывает in или out'})
            if not name or name != (req.get('name') or '').strip():
                return self._send(400, {'error': 'плохое имя файла'})
            base = VIDEO_IN if kind == 'in' else VIDEO_OUT
            full = os.path.join(base, name)
            if not os.path.isfile(full):
                return self._send(404, {'error': 'такого файла нет на стенде'})
            if kind == 'in' and _unit_active(VIDEO_UNIT) in ('active', 'activating'):
                st = video_state()
                if st.get('name') == name:
                    return self._send(400, {'error': 'этот файл сейчас обрабатывается — сначала останови'})
            try:
                os.remove(full)
            except OSError as e:
                return self._send(500, {'error': 'не смог удалить: %s' % e})
            return self._send(200, {'ok': True, 'kind': kind, 'name': name})

        if path in ('/api/detail', '/api/redraw', '/api/hands'):
            if path == '/api/hands':
                req['detector'] = 'hands'
            kind = 'redraw' if path == '/api/redraw' else 'detail'
            ok, why = guard_check((req.get('prompt') or '').strip())
            if not ok:
                return self._send(403, {'error': why})
            detector = (req.get('detector') or 'hands').strip()
            if detector not in DETECTORS:
                return self._send(400, {'error': 'детектор бывает hands или face'})
            src = os.path.basename((req.get('image') or '').strip())
            if not src or not re.fullmatch(r'[\w.\-]{1,120}', src):
                return self._send(400, {'error': 'нужна картинка из ленты'})
            src_path = os.path.join(OUT_DIR, src)
            if not os.path.isfile(src_path):
                return self._send(404, {'error': 'такой картинки нет на диске'})
            model = next((m for m in all_models() if m['key'] == req.get('model')), None)
            if model and model.get('kind') != 'video':
                ckpt, label, cfg = model['file'], model['label'], model['cfg']
            else:
                fb = next(m for m in MODELS if m[0] == 'juggernaut')
                ckpt, label, cfg = fb[2], fb[1], fb[4]
            with open(src_path, 'rb') as f:
                src_raw = f.read()
            size = png_size(src_raw) or (1024, 1024)
            tag = uuid.uuid4().hex[:12]
            src_name = '%s-%s.png' % ('redraw' if kind == 'redraw' else detector, tag)
            try:
                with open(os.path.join(INPUT_DIR, src_name), 'wb') as f:
                    f.write(src_raw)
                os.chmod(os.path.join(INPUT_DIR, src_name), 0o644)
            except Exception as e:
                return self._send(500, {'error': 'не смог скопировать картинку: %s' % e})
            try:
                steps = int(req.get('steps') or (model['steps'] if model else 28))
                dflt = 0.6 if kind == 'redraw' else DETAIL_DEFAULTS.get(
                    detector, DETAIL_DEFAULTS['hands'])['denoise']
                denoise = float(req.get('denoise') or dflt)
                strength = float(req.get('strength') or 0.8)
            except (TypeError, ValueError):
                return self._send(400, {'error': 'шаги, сила перерисовки и прочность '
                                                 'контура — числа'})
            steps = max(1, min(80, steps))
            denoise = max(0.05, min(1.0, denoise))
            strength = max(0.0, min(2.0, strength))
            jid = uuid.uuid4().hex[:12]
            dlora, dlw = lora_from_req(req)
            job = {'id': jid, 'state': 'queued', 'phase': 'в очереди',
                   'created': time.time(),
                   'req': {'prompt': (req.get('prompt') or '').strip(),
                           'w': size[0], 'h': size[1], 'steps': steps, 'cfg': cfg,
                           'file': ckpt, 'label': label, 'kind': kind,
                           'detector': detector if kind == 'detail' else None,
                           'family': None, 'length': None, 'fps': None,
                           'seed': req.get('seed') or None,
                           'lora': dlora, 'lora_weight': dlw,
                           'negative': (req.get('negative') or '').strip() or NEG_DEFAULT,
                           'translate': bool(req.get('translate')),
                           'translator': (req.get('translator') or TRANSLATOR),
                           'keep_ollama': bool(req.get('keep_ollama')),
                           'src': src_name, 'source_file': src,
                           'denoise': denoise, 'strength': strength,
                           'ctrl_type': (req.get('ctrl_type') or
                                         'canny/lineart/anime_lineart/mlsd'),
                           'pose': bool(req.get('pose')),
                           'pose_strength': float(req.get('pose_strength') or 0.9),
                           'upscale': bool(req.get('upscale'))}}
            with JOBS_LOCK:
                JOBS[jid] = job
            JOBQ.put(job)
            return self._send(200, {'id': jid})

        if path == '/api/inpaint':
            src = os.path.basename((req.get('image') or '').strip())
            if not src or not re.fullmatch(r'[\w.\-]{1,120}', src):
                return self._send(400, {'error': 'нужна картинка из ленты'})
            src_path = os.path.join(OUT_DIR, src)
            if not os.path.isfile(src_path):
                return self._send(404, {'error': 'такой картинки нет на диске'})
            model = next((m for m in all_models() if m['key'] == req.get('model')), None)
            if model and model.get('kind') != 'video':
                ckpt = model['file']
                label = model['label']
            else:
                fallback = next(m for m in MODELS if m[0] == 'juggernaut')
                ckpt = fallback[2]
                label = fallback[1]
            data = (req.get('mask') or '').strip()
            if data.startswith('data:') and ',' in data[:128]:
                data = data.split(',', 1)[1]
            if not data:
                return self._send(400, {'error': 'пустая маска: закрась область'})
            try:
                mask_raw = base64.b64decode(data + '=' * (-len(data) % 4))
            except Exception as e:
                return self._send(400, {'error': 'не разобрал маску: %s' % e})
            size = png_size(mask_raw)
            if not size:
                return self._send(400, {'error': 'маска должна быть png'})
            with open(src_path, 'rb') as f:
                src_raw = f.read()
            src_size = png_size(src_raw)
            if src_size and src_size != size:
                return self._send(400, {'error': 'маска %dx%d не совпадает с картинкой %dx%d'
                                                 % (size[0], size[1], src_size[0], src_size[1])})
            tag = uuid.uuid4().hex[:12]
            src_name = 'src-%s.png' % tag
            mask_name = 'mask-%s.png' % tag
            try:
                with open(os.path.join(INPUT_DIR, src_name), 'wb') as f:
                    f.write(src_raw)
                with open(os.path.join(INPUT_DIR, mask_name), 'wb') as f:
                    f.write(mask_raw)
                os.chmod(os.path.join(INPUT_DIR, src_name), 0o644)
                os.chmod(os.path.join(INPUT_DIR, mask_name), 0o644)
            except Exception as e:
                return self._send(500, {'error': 'не смог сохранить маску: %s' % e})
            text = (req.get('prompt') or '').strip()
            ok, why = guard_check(text)
            if not ok:
                return self._send(403, {'error': why})
            try:
                steps = int(req.get('steps') or 28)
                denoise = float(req.get('denoise') or 0.6)
                grow = int(req.get('grow') or 8)
            except (TypeError, ValueError):
                return self._send(400, {'error': 'шаги, denoise и расширение — числа'})
            steps = max(1, min(80, steps))
            denoise = max(0.05, min(1.0, denoise))
            grow = max(0, min(64, grow))
            jid = uuid.uuid4().hex[:12]
            ilora, ilw = lora_from_req(req)
            job = {'id': jid, 'state': 'queued', 'phase': 'в очереди',
                   'created': time.time(),
                   'req': {'prompt': text, 'w': size[0], 'h': size[1],
                           'steps': steps, 'cfg': 5.0, 'file': ckpt,
                           'label': label, 'kind': 'inpaint', 'family': None,
                           'length': None, 'fps': None, 'seed': req.get('seed') or None,
                           'lora': ilora, 'lora_weight': ilw,
                           'negative': (req.get('negative') or '').strip() or NEG_DEFAULT,
                           'translate': bool(req.get('translate')),
                           'translator': (req.get('translator') or TRANSLATOR),
                           'keep_ollama': bool(req.get('keep_ollama')),
                           'src': src_name, 'mask_image': mask_name,
                           'source_file': src, 'denoise': denoise, 'grow': grow}}
            with JOBS_LOCK:
                JOBS[jid] = job
            JOBQ.put(job)
            return self._send(200, {'id': jid})

        if path == '/api/grade/preview':
            try:
                return self._send(200, grade_preview(req))
            except Exception as e:
                return self._send(400, {'error': str(e)[:300]})

        if path == '/api/grade/apply':
            try:
                return self._send(200, grade_apply(req))
            except Exception as e:
                return self._send(400, {'error': str(e)[:300]})

        if path == '/api/vision':
            model = (req.get('model') or VISION_DEFAULT).strip()
            langs = [m['name'] for m in vision_models()]
            if model not in langs:
                return self._send(400, {'error': 'модель не видит картинки: %s' % model})
            lang = 'ru' if (req.get('lang') or '').lower().startswith('ru') else 'en'
            data = (req.get('data') or '').strip()
            if data.startswith('data:') and ',' in data[:128]:
                data = data.split(',', 1)[1]
            if not data:
                return self._send(400, {'error': 'нет картинки'})
            try:
                raw = base64.b64decode(data + '=' * (-len(data) % 4))
            except Exception as e:
                return self._send(400, {'error': 'не разобрал картинку: %s' % e})
            if len(raw) > 25 * 1024 * 1024:
                return self._send(400, {'error': 'файл больше 25 МБ'})
            if raw[:3] == b'\xff\xd8\xff':
                ext = '.jpg'
            elif raw[:8] == b'\x89PNG\r\n\x1a\n':
                ext = '.png'
            elif raw[:4] == b'RIFF' and raw[8:12] == b'WEBP':
                ext = '.webp'
            else:
                return self._send(400, {'error': 'это не png, jpeg или webp'})
            try:
                text, sec = vision_describe(model, lang, raw, ext, req.get('hint') or '')
            except urllib.error.HTTPError as e:
                try:
                    body = e.read().decode('utf-8', 'replace')[:300]
                except Exception:
                    body = ''
                return self._send(502, {'error': 'ollama отказал: %s %s' % (e.code, body)})
            except Exception as e:
                return self._send(502, {'error': 'модель не ответила: %s' % e})
            if not text:
                return self._send(502, {'error': 'модель вернула пустой ответ'})
            return self._send(200, {'text': text, 'model': model, 'lang': lang,
                                    'sec': sec})

        if path == '/api/chat':
            model = (req.get('model') or TRANSLATOR).strip()
            msgs = []
            raw = req.get('messages')
            if isinstance(raw, list):
                for m in raw[-16:]:
                    if not isinstance(m, dict):
                        continue
                    if m.get('role') not in ('user', 'assistant') or not m.get('content'):
                        continue
                    msgs.append({'role': m['role'], 'content': str(m['content'])[:8000]})
            text = (req.get('text') or '').strip()
            if text:
                msgs.append({'role': 'user', 'content': text[:8000]})
            if not msgs:
                return self._send(400, {'error': 'нужен текст'})
            last_user = ''
            for mm in reversed(msgs):
                if mm['role'] == 'user':
                    last_user = mm['content']
                    break
            if req.get('draw') is not False:
                pic = draw_prompt(last_user)
                if pic:
                    jid, why = queue_image(pic, model_key=req.get('draw_model'),
                                           size=req.get('draw_size'))
                    if not jid:
                        return self._send(200, {'text': 'картинку поставить не смог: ' + why,
                                                'thinking': '', 'model': 'рисование',
                                                'sec': 0})
                    return self._send(200, {'text': 'рисую: ' + pic, 'thinking': '',
                                            'model': 'рисование', 'sec': 0,
                                            'image_job': jid, 'drawn': pic})
            payload = {'model': model, 'stream': False, 'messages': msgs,
                       'think': bool(req.get('think'))}
            # окно контекста: без него ollama берёт 4k из Modelfile
            ctx = req.get('ctx') or CHAT_CTX
            try:
                ctx = int(ctx)
            except (TypeError, ValueError):
                ctx = CHAT_CTX
            ctx = max(2048, min(ctx, CHAT_CTX_MAX))
            options = {'num_ctx': ctx,
                       'num_predict': CHAT_MAX_TOKENS,
                       'temperature': CHAT_TEMP,
                       'repeat_penalty': CHAT_REPEAT_PENALTY,
                       'repeat_last_n': CHAT_REPEAT_LAST_N,
                       'presence_penalty': CHAT_PRESENCE_PENALTY,
                       'frequency_penalty': CHAT_FREQUENCY_PENALTY,
                       'top_p': CHAT_TOP_P,
                       'min_p': CHAT_MIN_P}
            t0 = time.time()
            guard = LoopGuard(CHAT_LOOP_N, CHAT_LOOP_LIMIT)
            try:
                text, thinking, tokens, stopped = ollama_chat_stream(
                    model, msgs, options, payload.get('think'), CHAT_TIMEOUT, guard)
            except urllib.error.HTTPError as e:
                try:
                    body = e.read().decode('utf-8', 'replace')[:300]
                except Exception:
                    body = ''
                return self._send(502, {'error': 'ollama отказал: %s %s' % (e.code, body)})
            except Exception as e:
                return self._send(502, {'error': 'ollama не ответил: %s' % e})
            note = ''
            if stopped == 'loop':
                tail = ' '.join(guard.hit.split(' ')[:6])
                note = ('\n\n_обрезал: модель начала повторять одно и то же '
                        '(«%s…»). Скажите «короче» или смените тему._' % tail)
            elif stopped == 'length':
                note = ('\n\n_обрезал по потолку %d токенов — попросите продолжение, '
                        'если нужно длиннее._' % CHAT_MAX_TOKENS)
            return self._send(200, {'text': (text + note), 'thinking': thinking,
                                    'model': model, 'ctx': ctx, 'tokens': tokens,
                                    'stopped': stopped or 'stop',
                                    'sec': round(time.time() - t0, 1)})

        if path == '/api/generate':
            text = (req.get('prompt') or '').strip()
            if not text:
                return self._send(400, {'error': 'нужен текст'})
            ok, why = guard_check(text, req.get('end_prompt') or '')
            if not ok:
                return self._send(403, {'error': why})
            model = next((m for m in all_models() if m['key'] == req.get('model')), None)
            if not model:
                return self._send(400, {'error': 'такой модели нет'})
            kind = model['kind']
            family = model.get('family')
            if kind == 'video' and family in VIDEO_FAMILY:
                size = req.get('size') or VIDEO_FAMILY[family]['size']
            else:
                size = req.get('size') or ('768x512' if kind == 'video' else '1024x1024')
            m = re.fullmatch(r'(\d+)x(\d+)', size)
            if not m:
                return self._send(400, {'error': 'размер писать как 1024x1024'})
            w, h = int(m.group(1)), int(m.group(2))
            steps = int(req.get('steps') or model['steps'])
            steps = max(1, min(80, steps))
            length = fps = None
            if kind == 'video':
                step = 32
                d = VIDEO_FAMILY[family]
                if (w % step or h % step or w * h > d['max_area']
                        or max(w, h) > d['max_side']):
                    return self._send(400, {'error': 'для этой модели стороны кратны 32, '
                                                     'не больше %dx' % (d['max_side'],
                                                                       d['max_side'])})
                try:
                    length = int(req.get('length') or d['length'])
                    fps = int(req.get('fps') or d['fps'])
                except (TypeError, ValueError):
                    return self._send(400, {'error': 'кадры и fps — числа'})
                base = 8 if family == 'ltxv' else 4
                if (length < base + 1 or length > d['max_frames']
                        or (length - 1) % base):
                    return self._send(400, {'error': 'кадров должно быть %dk+1, не больше %d'
                                                     % (base, d['max_frames'])})
                if fps < 8 or fps > 30:
                    return self._send(400, {'error': 'fps от 8 до 30'})
                if family == 'wanflf' and not (req.get('end_image') or
                                               (req.get('end_prompt') or '').strip()):
                    return self._send(400, {'error': 'для схемы «первый и последний кадр» '
                                                     'нужен последний кадр или его описание'})
            elif w % 8 or h % 8 or w * h > 1024 * 1536 or max(w, h) > 1536:
                return self._send(400, {'error': 'размер вне 8-кратных пределов 1024x1536'})
            neg = (req.get('negative') or '').strip()
            if not neg:
                neg = NEG_VIDEO if kind == 'video' else NEG_DEFAULT
            lora = lora_ok(req.get('lora'))
            if req.get('lora') and not lora:
                return self._send(400, {'error': 'нет такой LoRA: %s' % req.get('lora')})
            try:
                lw = float(req.get('lora_weight') or 0.8)
            except (TypeError, ValueError):
                return self._send(400, {'error': 'сила стиля — число от 0 до 1.5'})
            lw = max(0.0, min(1.5, lw))
            if kind == 'video':      # LoRA ставится на чекпойнт, у видео UNETLoader
                lora, lw = None, None
            jid = uuid.uuid4().hex[:12]
            job = {'id': jid, 'state': 'queued', 'phase': 'в очереди',
                   'created': time.time(),
                   'req': {'prompt': text, 'w': w, 'h': h, 'steps': steps,
                           'cfg': model['cfg'], 'file': model['file'],
                           'label': model['label'], 'kind': kind,
                           'family': family,
                           'length': length, 'fps': fps,
                           'seed': req.get('seed') or None,
                           'negative': neg,
                           'start_image': safe_input(req.get('start_image')),
                           'end_image': safe_input(req.get('end_image')),
                           'end_prompt': (req.get('end_prompt') or '').strip() or None,
                           'frame_model': req.get('frame_model') or None,
                            'translate': bool(req.get('translate')),
                            'translator': (req.get('translator') or TRANSLATOR),
                            'keep_ollama': bool(req.get('keep_ollama')),
                            'lora': lora, 'lora_weight': lw}}
            with JOBS_LOCK:
                JOBS[jid] = job
                if len(JOBS) > 200:
                    for k in [k for k, v in JOBS.items() if v.get('state') in ('done', 'error')][:100]:
                        JOBS.pop(k, None)
            JOBQ.put(job)
            return self._send(200, {'id': jid})

        return self._send(404, {'error': 'нет такого адреса'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8288)
    ap.add_argument('--host', default='0.0.0.0')
    a = ap.parse_args()
    threading.Thread(target=worker, daemon=True).start()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print('stand-ui на http://%s:%d/, ComfyUI %s' % (a.host, a.port, API), flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
