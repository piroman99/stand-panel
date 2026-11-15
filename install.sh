#!/bin/bash
# Установка панели стенда БЕЗ sudo — под своим пользователем.
#
#   bash install.sh              # поставить и запустить
#   bash install.sh status       # состояние юнита, порта и ответа API
#   bash install.sh restart      # перезапустить (задачи в очереди не теряются)
#   bash install.sh logs         # последние строки журнала
#   bash install.sh stop         # остановить
#   bash install.sh remove       # снять юнит (файлы панели остаются)
#
# Ничего не пишет вне домашнего каталога: код — $PREFIX, состояние — panel.env,
# автозапуск — пользовательский юнит ~/.config/systemd/user/stand-ui.service.
# Root не нужен. Панель слушает только 127.0.0.1, наружу не выходит.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
ACTION="${1:-install}"
PREFIX="${PREFIX:-$HOME/panels/stand-panel}"
PORT="${PORT:-8288}"
HOST="${HOST:-127.0.0.1}"
PY="${PY:-$(command -v python3 || true)}"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="$UNIT_DIR/stand-ui.service"

say() { printf '%s\n' "$*"; }
die() { printf 'ошибка: %s\n' "$*" >&2; exit 1; }

need_python() {
    [ -n "$PY" ] && [ -x "$PY" ] || die "не нашёл python3 в PATH"
    "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
        || die "нужен python3 3.10 или новее (сейчас: $("$PY" -V 2>&1))"
}

have_systemd_user() {
    [ -d /run/systemd/system ] && systemctl --user show-environment >/dev/null 2>&1
}

probe() {
    "$PY" - "$PORT" <<'PYEOF'
import json
import sys
import urllib.request

port = sys.argv[1]
base = 'http://127.0.0.1:%s' % port
ok = True
try:
    print('страница: HTTP', urllib.request.urlopen(base + '/', timeout=10).status)
except Exception as exc:
    print('страница: НЕТ ОТВЕТА (%s)' % exc)
    sys.exit(1)
for path, keys in (('/api/config', ('models', 'sizes', 'video_models')),
                   ('/api/sys', ('cpu', 'mem', 'gpu'))):
    try:
        j = json.load(urllib.request.urlopen(base + path, timeout=25))
    except Exception as exc:
        print('%s: ошибка (%s)' % (path, exc))
        ok = False
        continue
    bits = []
    for k in keys:
        v = j.get(k)
        if isinstance(v, list):
            bits.append('%s: %d' % (k, len(v)))
        elif isinstance(v, dict):
            bits.append('%s: да' % k)
    print('%s: %s' % (path, '; '.join(bits) or 'пусто'))
cpu = None
try:
    cpu = json.load(urllib.request.urlopen(base + '/api/sys', timeout=25)).get('cpu') or {}
except Exception:
    pass
if cpu is not None:
    print('процессор: %s %%, ватты %s' % (cpu.get('percent'), cpu.get('watts')))
    if not cpu.get('watts_ok'):
        print('  подсказка: ватты не читаются — нужен разовый root-хелпер (см. README)')
sys.exit(0 if ok else 1)
PYEOF
}

case "$ACTION" in
  install|restart)
    need_python
    [ -f "$HERE/stand-ui.py" ] || die "рядом нет stand-ui.py"
    [ -f "$HERE/stand_guard.py" ] || die "рядом нет stand_guard.py"

    say "== каталог $PREFIX =="
    mkdir -p "$PREFIX" || die "не смог создать $PREFIX"

    # прежнюю копию — в резерв, держим последние пять
    if [ -f "$PREFIX/stand-ui.py" ]; then
        ts=$(date +%Y%m%d-%H%M%S)
        cp -p "$PREFIX/stand-ui.py" "$PREFIX/stand-ui.py.bak-$ts" 2>/dev/null || true
        ls -1t "$PREFIX"/stand-ui.py.bak-* 2>/dev/null | tail -n +6 | xargs -r rm -f
    fi
    install -m 644 "$HERE/stand-ui.py" "$PREFIX/stand-ui.py" || die "копирование stand-ui.py"
    install -m 644 "$HERE/stand_guard.py" "$PREFIX/stand_guard.py" || die "копирование stand_guard.py"
    say "код на месте: $(ls -l "$PREFIX/stand-ui.py" | awk '{print $5}') байт"

    say "== синтаксис =="
    "$PY" -m py_compile "$PREFIX/stand-ui.py" "$PREFIX/stand_guard.py" \
        || die "код не компилируется"
    say "ok"

    # конфиг: создаём из примера, существующий не трогаем
    if [ ! -f "$PREFIX/panel.env" ]; then
        if [ -f "$HERE/panel.env.example" ]; then
            cp "$HERE/panel.env.example" "$PREFIX/panel.env"
            say "== panel.env создан из примера — проверь адреса и пути =="
        else
            : > "$PREFIX/panel.env"
            say "== panel.env пустой: всё на умолчаниях =="
        fi
    else
        say "== panel.env уже есть, не трогаю =="
    fi

    say "== psutil (вкладка «железо») =="
    if "$PY" -c 'import psutil' >/dev/null 2>&1; then
        say "есть"
    else
        say "нет — вкладка «железо» будет пустой. Поставить: $PY -m pip install --user psutil"
    fi

    if ! have_systemd_user; then
        say "== systemd пользователя недоступен: запускаю как обычный процесс =="
        pkill -f "$PREFIX/stand-ui.py" 2>/dev/null || true
        sleep 1
        ( cd "$PREFIX" && set -a && . ./panel.env && set +a && \
          setsid nohup "$PY" "$PREFIX/stand-ui.py" --port "$PORT" --host "$HOST" \
          > "$PREFIX/stand-ui.log" 2>&1 & )
        sleep 3
        probe || { say "смотри $PREFIX/stand-ui.log"; exit 1; }
        say "открывать: http://127.0.0.1:$PORT/  (с другой машины — ssh-туннель)"
        exit 0
    fi

    say "== юнит =="
    mkdir -p "$UNIT_DIR"
    cat > "$UNIT" <<UNITEOF
[Unit]
Description=stand-ui: панель стенда (картинки, видео, чат, железо)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$PREFIX
EnvironmentFile=$PREFIX/panel.env
ExecStart=$PY $PREFIX/stand-ui.py --port $PORT --host $HOST
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
UNITEOF
    systemctl --user daemon-reload
    systemctl --user enable stand-ui.service >/dev/null 2>&1 || true
    systemctl --user restart stand-ui.service || die "юнит не запустился: systemctl --user status stand-ui"
    sleep 2
    say "юнит: $(systemctl --user is-active stand-ui.service)"

    # без linger панель не поднимется после перезагрузки, пока не войдёшь в систему
    if loginctl show-user "$USER" --property=Linger 2>/dev/null | grep -q 'Linger=yes'; then
        say "автозапуск после перезагрузки: включён (linger)"
    else
        say "автозапуск после перезагрузки: выключен"
        say "  включить: loginctl enable-linger $USER"
    fi

    say "== проверка =="
    probe || { say "журнал: systemctl --user status stand-ui"; exit 1; }
    say "открывать: http://127.0.0.1:$PORT/  (с другой машины — ssh-туннель)"
    ;;
  status)
    need_python
    if have_systemd_user; then
        systemctl --user --no-pager --lines=0 status stand-ui.service | head -6 || true
    else
        pgrep -af "$PREFIX/stand-ui.py" || say "процесс не найден"
    fi
    ss -lntp 2>/dev/null | grep -E ":$PORT\b" || say "порт $PORT не слушается"
    probe || true
    ;;
  logs)
    if have_systemd_user; then
        journalctl --user -u stand-ui.service -n 40 --no-pager
    else
        tail -n 40 "$PREFIX/stand-ui.log" 2>/dev/null || say "лога нет"
    fi
    ;;
  stop)
    if have_systemd_user; then
        systemctl --user stop stand-ui.service && say остановлено
    else
        pkill -f "$PREFIX/stand-ui.py" && say остановлено || say "процесс не найден"
    fi
    ;;
  remove)
    if have_systemd_user; then
        systemctl --user disable --now stand-ui.service 2>/dev/null || true
        "$PY" -c "import os; os.remove('$UNIT')" 2>/dev/null || true
        systemctl --user daemon-reload
    else
        pkill -f "$PREFIX/stand-ui.py" 2>/dev/null || true
    fi
    say "снято (код и panel.env в $PREFIX оставлены)"
    ;;
  *)
    say "usage: bash $0 [install|restart|status|logs|stop|remove]" >&2
    exit 2
    ;;
esac
