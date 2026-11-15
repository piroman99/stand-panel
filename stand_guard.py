"""Фильтр промптов на стенде: не даёт нарисовать несовершеннолетних в сексуальном контексте.

Два списка маркеров — «ребёнок/подросток» и «сексуальное/обнажённое». Если в тексте
есть и то и другое, задача отклоняется. Отдельный список жёстких терминов блокируется
всегда, без второго условия. Проверять надо и русский текст, и английский перевод:
переводчик может добавить слово, которого в исходной фразе не было.

    from stand_guard import check_prompt
    ok, why = check_prompt('...')   # ok=False -> why = человеческая причина отказа
"""

import re

# Ребёнок или подросток: возраст, слова-маркеры, школьная лексика
MINOR = [
    r'\b(?:1[0-7]|[1-9])\s*(?:лет|года|год|y\.?\s?o\.?|yo|yrs?|years?[- ]old)\b',
    r'\b(?:лет|возраст|age)\s*(?:1[0-7]|[1-9])\b',
    r'\b1[0-7][- ]?(?:year|yr)s?[- ]?old\b',
    r'\b(?:child|children|kid|kids|minor|underage|preteen|preteenager|tween|teen|teens|'
    r'teenager|teenage|adolescent|schoolgirl|schoolboy|schoolkid|young\s+girl|young\s+boy|'
    r'little\s+girl|little\s+boy|baby|babies|toddler|infant|loli|lolicon|shota|shotacon)\b',
    r'(?:девочк|девчонк|девченк|мальчик|мальчуган|ребен|ребён|подрост|школьниц|школьник|'
    r'малолет|несовершеннолет|дитят|малыш|кроха|ясельн|первокласс)',
]

# Сексуальный контекст, обнажённость, эротика
SEXUAL = [
    r'\b(?:nude|naked|nsfw|explicit|erotic|erotica|porn|porno|pornography|sexual|sex|'
    r'sexy|topless|bottomless|lingerie|underwear|undies|panties|knickers|bra|bras|'
    r'breasts?|boobs?|nipples?|genitals?|genitalia|penis|vagina|vulva|pussy|cock|dick|'
    r'cum|semen|sperm|orgasm|masturbat\w*|blowjob|handjob|anal|incest|hentai|ecchi|'
    r'suggestive|seductive|fetish|bdsm|bondage|straddl\w*|spread\s+legs|bare\s+skin|'
    r'bare\s+chest|shirtless|bottomless)\b',
    r'(?:гол(?:ая|ые|ый|ым|ой|еньк|ыш)|наг(?:ая|ие|ой|им)|обнаж|раздет|без\s+одежд|'
    r'эротич|эротик|секс|порн|интим|нижн(?:ее|ем|его)\s+бель|бель[её]|трусик|'
    r'топлес|купальник|плавки|лифчик|бюстгальтер|груд[ьи]|сосок|соск[иа]|пенис|вагин|'
    r'попк|ягодиц|прелюди|минет|оргазм|изнасил|растл|соблазн|стриптиз|нудист)',
]

# Всегда блокируем: термины, за которыми стоит только одно
HARD = [
    r'\b(?:loli|lolicon|shota|shotacon|child\s?porn|cp|csex|pedo|pedophile|pedophilia|'
    r'underage\s+sex|child\s+abuse)\b',
    r'(?:педофил|детск(?:ое|ая|ого)\s+порн|секс\s+с\s+ребен|секс\s+с\s+ребён|'
    r'растлен\w*\s+малолет|малолетк\w*\s+секс)',
]

MINOR_RE = [re.compile(p, re.I) for p in MINOR]
SEXUAL_RE = [re.compile(p, re.I) for p in SEXUAL]
HARD_RE = [re.compile(p, re.I) for p in HARD]


def normalize(text):
    """Привести текст к виду, по которому регулярки ловят уловки: регистр, ё, разделители."""
    t = (text or '').lower().replace('ё', 'е')
    t = re.sub(r'[_\-*·•]+', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    return t


def _hit(patterns, text):
    for rx in patterns:
        m = rx.search(text)
        if m:
            return m.group(0)
    return None


def check_prompt(text, extra=''):
    """Проверить промпт. Возвращает (True, None) или (False, причина отказа)."""
    blob = normalize('%s %s' % (text or '', extra or ''))
    hard = _hit(HARD_RE, blob)
    if hard:
        return False, ('в промпте есть «%s» — такое не рисуем' % hard)
    minor = _hit(MINOR_RE, blob)
    if not minor:
        return True, None
    sexual = _hit(SEXUAL_RE, blob)
    if sexual:
        return False, ('в промпте рядом стоят «%s» и «%s»: несовершеннолетние в сексуальном '
                       'контексте — не рисуем' % (minor, sexual))
    return True, None


def main(argv=None):
    """Проверка из командной строки: python3 stand-guard.py "текст"."""
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print('usage: stand-guard.py "текст промпта"')
        return 2
    ok, why = check_prompt(' '.join(argv))
    print('OK' if ok else 'ОТКАЗ: ' + why)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
