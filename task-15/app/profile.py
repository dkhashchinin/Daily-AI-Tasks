"""User profile: personalization on top of the memory model.

A profile is a named set of preferences about HOW the assistant should answer
(style, format, tone, standing constraints) — not what it knows (that is LTM)
and not the current task (that is WM). Exactly one profile is active at a
time; its rules are injected into every request as the first system message.

Two ways to fill a profile:

- **вручную** — the profile editor in the UI (:func:`save_profile`);
- **из диалога** — after each assistant reply
  :func:`update_from_dialog` asks the LLM whether the user expressed a stable
  preference ("отвечай короче", "не надо кода", "будь попроще") and merges it
  into the profile as a ``source="dialog"`` rule. Manual rules are never
  touched by this updater.
"""
import json

from . import store
from .llm import chat, parse_json_object

# Preference kinds, in prompt order. 'constraint' is the free-form bucket.
KINDS = ("style", "format", "tone", "constraint")
KIND_LABELS = {
    "style": "Стиль",
    "format": "Формат",
    "tone": "Тон",
    "constraint": "Ограничения",
}

# Dimensions offered as ready-made opposite pairs in the profile editor.
# Each option maps to a concrete rule text.
STYLE_OPTIONS = {
    "brief": "Отвечай кратко: 2–4 предложения, только суть, без вступлений и воды",
    "detailed": "Отвечай подробно и развёрнуто: с объяснениями, примерами и разбивкой по пунктам",
}
CODE_OPTIONS = {
    "no_code": "Не приводи код и примеры кода, объясняй словами",
    "code": "Всегда приводи примеры кода, где это уместно",
}
TONE_OPTIONS = {
    "formal": "Общайся формально и по-деловому, на «Вы», без разговорных выражений и эмодзи",
    "informal": "Общайся неформально и дружелюбно, на «ты», можно с лёгким юмором",
}

PROMPT_HEADER = (
    "ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ. Это устойчивые предпочтения о том, КАК ты должен "
    "отвечать. Строго соблюдай их в каждом ответе; они важнее твоих "
    "стандартных привычек оформления."
)

PROFILE_SYSTEM = (
    "Ты ведёшь блок «предпочтения пользователя» в формате JSON — постоянные "
    "правила о том, КАК ассистент должен отвечать этому пользователю: стиль "
    "(кратко/подробно), формат (с кодом/без кода, списки, таблицы), тон "
    "(формально/неформально), постоянные ограничения (язык, без эмодзи и т.п.). "
    "Предпочтением считается только устойчивое пожелание о манере ответов; "
    "тема, факты и данные задачи предпочтениями НЕ являются. "
    "Учитывай сообщение пользователя (и его реакцию на ответ ассистента): если "
    "он явно попросил изменить манеру ответов («отвечай короче», «не надо "
    "кода», «будь попроще», «можно неформально») — обнови блок. Мелкие "
    "разовые уточнения задачи игнорируй. Не выдумывай предпочтений. "
    'Формат: {"rules": [{"kind": "style|format|tone|constraint", "text": "..."}]}. '
    "В ответе — только JSON без пояснений."
)

UPDATE_LIMIT = 8  # максимум правил, выученных из диалога, чтобы блок не разрастался


def render(profile_id: int | None = None) -> str:
    """Render the profile as a system message; empty string when it has no rules.

    ``profile_id=None`` renders the active profile.
    """
    prof = (store.get_active_profile() if profile_id is None
            else store.get_profile(profile_id))
    if not prof:
        return ""
    rules = store.list_profile_rules(prof["id"])
    if not rules:
        return ""
    lines = [f"{PROMPT_HEADER} Активный профиль: «{prof['name']}»."]
    if prof["description"]:
        lines.append(f"О пользователе: {prof['description']}")
    for kind in KINDS:
        items = [r for r in rules if r["kind"] == kind]
        if not items:
            continue
        lines.append(f"{KIND_LABELS[kind]}:")
        lines.extend(f"- {r['text']}" for r in items)
    return "\n".join(lines)


def _extract_rules(raw: str | None) -> list[dict] | None:
    """Parse the updater reply into a rules list (tolerates ``` fences)."""
    parsed = parse_json_object(raw)
    if parsed is None:
        return None
    rules = parsed.get("rules")
    if not isinstance(rules, list):
        return None
    out: list[dict] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        kind = str(rule.get("kind") or "").strip()
        text = str(rule.get("text") or "").strip()
        if kind in KINDS and text:
            out.append({"kind": kind, "text": text})
    return out


def update_from_dialog(user_text: str, assistant_text: str,
                       profile_id: int | None = None) -> dict | None:
    """Learn standing preferences from the dialog and merge them into the profile.

    Called after each assistant reply. Only the ``source="dialog"`` rules are
    rewritten (manual rules stay untouched). Returns
    ``{"added": [...], "removed": [...], "usage": {...}}`` or ``None`` when
    there is no active profile / the call failed.
    """
    prof = (store.get_active_profile() if profile_id is None
            else store.get_profile(profile_id))
    if not prof:
        return None
    pid = prof["id"]
    current = [{"kind": r["kind"], "text": r["text"]}
               for r in store.list_profile_rules(pid, source="dialog")]
    current_keys = {(r["kind"], r["text"].lower()) for r in current}

    parts = [
        "Текущие выученные предпочтения (JSON; может быть пустым):",
        json.dumps({"rules": current}, ensure_ascii=False),
        "Сообщение пользователя:", user_text or "(нет)",
        "Ответ ассистента:", (assistant_text or "(нет)")[:1500],
        "Верни обновлённый блок предпочтений в формате JSON.",
    ]
    try:
        response = chat(
            [
                {"role": "system", "content": PROFILE_SYSTEM},
                {"role": "user", "content": "\n\n".join(parts)},
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[profile] не удалось обновить предпочтения: {exc}")
        return None

    rules = _extract_rules(raw)
    if rules is None:
        print("[profile] updater вернул мусор вместо JSON, блок не изменён")
        return None
    rules = rules[-UPDATE_LIMIT:]  # держим блок компактным

    added = [r for r in rules if (r["kind"], r["text"].lower()) not in current_keys]
    removed = [r for r in current
               if (r["kind"], r["text"].lower())
               not in {(x["kind"], x["text"].lower()) for x in rules}]
    store.replace_dialog_rules(pid, rules)
    return {
        "profile": prof["name"],
        "added": added,
        "removed": removed,
        "usage": {
            "prompt": getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
            "completion": getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
        },
    }


def save_profile(name: str, description: str,
                 style: str | None, code: str | None, tone: str | None,
                 constraints: list[str] | None = None,
                 profile_id: int | None = None) -> int:
    """Create or update a profile from editor fields.

    ``style``/``code``/``tone`` are keys of the ready-made opposite-pair
    options (or None = не задано); ``constraints`` are free-form rule lines.
    Returns the profile id.
    """
    rules: list[dict] = []
    if style in STYLE_OPTIONS:
        rules.append({"kind": "style", "text": STYLE_OPTIONS[style]})
    if code in CODE_OPTIONS:
        rules.append({"kind": "format", "text": CODE_OPTIONS[code]})
    if tone in TONE_OPTIONS:
        rules.append({"kind": "tone", "text": TONE_OPTIONS[tone]})
    for line in constraints or []:
        text = str(line).strip()
        if text:
            rules.append({"kind": "constraint", "text": text})
    name = str(name).strip()
    if not name:
        raise ValueError("Имя профиля не может быть пустым.")
    if profile_id is None:
        return store.create_profile(name, description, rules)
    store.update_profile(profile_id, name, description, rules)
    return profile_id


def summary(profile_id: int | None = None) -> dict | None:
    """Profile record with its rules split by kind — for the UI."""
    prof = (store.get_active_profile() if profile_id is None
            else store.get_profile(profile_id))
    if not prof:
        return None
    rules = store.list_profile_rules(prof["id"])
    return {
        **prof,
        "rules": rules,
        "manual": [r for r in rules if r["source"] == "manual"],
        "dialog": [r for r in rules if r["source"] == "dialog"],
    }
