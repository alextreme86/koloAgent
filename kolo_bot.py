#!/usr/bin/env python3
"""
KoloAgent Telegram Bot — hybrid reasoning (MES-69)

Architecture:
  Telegram message
        ↓
  Ollama health check → agentmex reachable?
        ├── YES → {OLLAMA_MODEL} (primary reasoning)
        └── NO  → Gemini 2.5 Flash (full fallback)

  Camera vision: Gemini converts image → text → injected into qwen context

Slash commands (bypass LLM):
  /snap     — 4 camera snapshots
  /weather  — current weather in Hedehusene
  /status   — mower + schedule status
  /analyse  — snapshots + grass analysis
  /mow      — analyse + start mower if OK
  /start    — start mower now
  /pause    — pause mower
  /dock     — return to dock
  /log      — last 20 log lines
  /help     — list commands
"""

import time, sys, os, tempfile, datetime, json, math
sys.path.insert(0, os.path.dirname(__file__))

import requests
from kolo_agent import (
    TELEGRAM_TOKEN, TELEGRAM_CHAT, GEMINI_API_KEY,
    capture_all_snapshots, capture_garden_snapshots, get_weather, get_indego_state,
    analyse_with_gemini, send_telegram_message, send_telegram_media_group,
    start_mowing, pause_mowing, dock_mower,
    load_state, should_mow_today, record_mow_session, days_since_last_mow,
    load_moisture_commentary,
)
import base64
from PIL import Image
import io

LOG_FILE      = os.path.join(os.path.dirname(__file__), "kolo_agent.log")
PENDING_FILE  = os.path.join(os.path.dirname(__file__), "kolo_pending.json")
ALLOWED_CHAT  = int(TELEGRAM_CHAT)
_last_text    = ""   # set in main loop so command handlers can read full message text
_notified_tok = None # last pending token we already sent a TG message for

# ── Moisture notification state ────────────────────────────────────────────────
_MOISTURE_CHECK_HOURS = {9, 18}        # fire at 09:00 and 18:00
_moisture_last_check_hour: int = -1    # which hour we last ran the check
_moisture_alerted_today: set = set()   # sensor keys already notified today
_moisture_alert_date: str = ""         # resets per-sensor tracking each new day

OLLAMA_BASE  = "http://100.67.199.79:11434"
OLLAMA_MODEL = "qwen3.5:4b"

CAM_USER     = "admin"
CAM_PASS     = os.environ.get("CAM_PASS", "")

SYSTEM_PROMPT = """You are KoloAgent, an assistant for a kolonihave (Danish allotment garden) in Hedehusene, Copenhagen.

Your scope is limited to: garden conditions, grass/mowing, weather, irrigation, plants, scheduling, and the Bosch Indego mower.

Rules:
- Never execute physical actions (mowing, watering) without explicit user confirmation
- Never mow if ground is wet or temperature below 10°C
- Always check live conditions before recommending actions
- On greetings, proactively share garden status (weather, mow schedule, any alerts)
- Always reply in English regardless of the language the user writes in
- Keep replies concise — this is a chat interface

If asked about something outside the garden scope, politely redirect:
"I'm only set up to help with the kolonihave!"
"""

# ── Ollama ────────────────────────────────────────────────────────────────────

def ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def ask_qwen(messages: list) -> str:
    """Call {OLLAMA_MODEL} via Ollama with full conversation history."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.3, "num_predict": 300},
    }
    r = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=120)
    if r.status_code == 200:
        return r.json()["message"]["content"].strip()
    raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:200]}")


def ask_gemini_text(prompt: str) -> str:
    """Call Gemini 2.5 Flash for text-only reasoning (fallback)."""
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}")
    payload = {
        "contents": [{"parts": [{"text": SYSTEM_PROMPT + "\n\n" + prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 512},
    }
    r = requests.post(url, json=payload, timeout=30)
    if r.status_code == 200:
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    raise RuntimeError(f"Gemini error {r.status_code}: {r.text[:200]}")

# ── Vision: Gemini image → text ───────────────────────────────────────────────

def vision_describe(snap_path: str, cam_name: str) -> str:
    """Use Gemini vision to convert a camera frame to a text description."""
    try:
        img = Image.open(snap_path)
        img.thumbnail((768, 768), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        data = base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        return f"[vision error: {e}]"

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-3-flash-preview:generateContent?key={GEMINI_API_KEY}")
    payload = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": data}},
            {"text": f"Camera: {cam_name}. Describe in 1-2 sentences: grass height, color, condition, any visible issues. Be concise and factual."}
        ]}],
        "generationConfig": {
            "temperature": 0.1, "maxOutputTokens": 100,
            "thinkingConfig": {"thinkingBudget": 0}
        },
    }
    try:
        r = requests.post(url, json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"[vision timeout: {e}]"
    return "[vision unavailable]"

# ── Context builder ───────────────────────────────────────────────────────────

_context_cache: dict = {"text": None, "ts": 0.0}
_CONTEXT_TTL = 300  # seconds — reuse context for 5 minutes between chat messages

def build_context(include_vision: bool = False) -> str:
    """Gather live garden context — kept concise to leave room for LLM response."""
    import time
    # For non-vision calls, return cached context if fresh enough
    if not include_vision:
        age = time.time() - _context_cache["ts"]
        if _context_cache["text"] and age < _CONTEXT_TTL:
            return _context_cache["text"]

    w = get_weather()
    indego = get_indego_state()
    state = load_state()
    state_map = {258: "Docked", 257: "Charging", 1: "Mowing",
                 513: "Leaving dock", 512: "Paused", 514: "Mowing", 64513: "Offline"}
    mower_state = state_map.get(indego.get("state"), "Unknown")
    mowed_pct = indego.get("mowed", "?")
    days_ago = days_since_last_mow(state)
    will_mow, reason = should_mow_today(state, True, w)
    yesterday_rain = state.get("yesterday_rain_mm", 0)
    w_str = (f"{w['temp_c']}°C, {w['desc']}, humidity {w['humidity']}%, "
             f"rain today {w['rain_mm_today']}mm, yesterday {yesterday_rain}mm"
             ) if w else "unavailable"
    # Aqara sensors (3 HT sensors: indoor, outdoor, greenhouse)
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from kolo_aqara import get_sensor_data
        aqara = get_sensor_data()
        parts = []
        for key, lbl in [("kolonihavehus","Indoor"),("outdoor","Outdoor"),("greenhouse","Greenhouse")]:
            s = aqara.get(key, {})
            if s:
                parts.append(f"{lbl} {s.get('temperature','?')}°C {s.get('humidity','?')}%RH")
        aqara_str = " | ".join(parts) if parts else "unavailable"
    except Exception:
        aqara_str = "unavailable"

    # Soil sensors
    try:
        from kolo_soil import get_soil_data, SENSORS as SOIL_SENSORS
        soil = get_soil_data()
        soil_parts = []
        for key in SOIL_SENSORS:
            s = soil.get(key, {})
            if not s or not s.get("active", True):
                continue
            m = s.get("moisture")
            if m is not None:
                soil_parts.append(f"{s.get('label', key)} {m}%")
        soil_str = " | ".join(soil_parts) if soil_parts else "unavailable"
    except Exception:
        soil_str = "unavailable"

    # Valve / irrigation status
    try:
        from kolo_irrigation import get_all_valve_status
        valves = get_all_valve_status()
        v_parts = []
        for zone, v in valves.items():
            st = "OPEN" if v.get("open") else "closed"
            cd = f" {v.get('countdown')}s" if v.get("open") and v.get("countdown") else ""
            v_parts.append(f"{zone} {st}{cd}")
        valve_str = " | ".join(v_parts) if v_parts else "unavailable"
    except Exception:
        valve_str = "unavailable"

    # Last irrigation events per zone (from DB)
    try:
        import sqlite3 as _sq
        _DB = os.path.join(os.path.dirname(__file__), "kolo_data.db")
        irr_parts = []
        if os.path.exists(_DB):
            _con = _sq.connect(_DB)
            _con.row_factory = _sq.Row
            for zone in ("greenhouse", "outdoor"):
                row = _con.execute(
                    """SELECT date(ts_open) as day, round(duration_actual_s/60.0,1) as dur,
                              moisture_before, moisture_after FROM irrigation_events
                       WHERE zone=? AND status='completed' ORDER BY id DESC LIMIT 1""",
                    (zone,)
                ).fetchone()
                if row:
                    irr_parts.append(f"{zone}: {row['day']} {row['dur']}min "
                                     f"{row['moisture_before']}%→{row['moisture_after']}%")
            _con.close()
        irr_str = " | ".join(irr_parts) if irr_parts else "none recorded"
    except Exception:
        irr_str = "unavailable"

    # Moisture commentary (latest Ollama insight)
    try:
        mc = load_moisture_commentary()
        moisture_comment = mc.get("comment", "")[:300] if mc else ""
        moisture_comment_ts = mc.get("ts", "")[:16] if mc else ""
    except Exception:
        moisture_comment = ""
        moisture_comment_ts = ""

    # Last grass analysis from state
    grass_analysis = state.get("last_analysis", "")
    grass_analysis_ts = state.get("last_analysis_ts", "")

    lines = [
        f"[{datetime.datetime.now().strftime('%d/%m %H:%M')}]",
        f"Weather: {w_str}",
        f"Sensors: {aqara_str}",
        f"Soil moisture: {soil_str}",
        f"Valves: {valve_str}",
        f"Last irrigation: {irr_str}",
        f"Mower: {mower_state}, {mowed_pct}% mowed, last mow {days_ago}d ago",
        f"Mow OK today: {'yes' if will_mow else 'no — ' + reason}",
    ]
    if grass_analysis:
        lines.append(f"Grass analysis ({grass_analysis_ts}): {grass_analysis}")
    if moisture_comment:
        lines.append(f"Moisture insight ({moisture_comment_ts}): {moisture_comment}")

    # Vision (if requested) — uses Greenhouse camera (.102) for PTZ
    if include_vision:
        lines.append("\n=== Camera Analysis ===")
        with tempfile.TemporaryDirectory() as tmpdir:
            for preset, label in [("1", "Garden angle 1"), ("2", "Garden angle 2")]:
                ptz("192.168.1.102", f"preset{preset}")
                time.sleep(5)
                snaps = capture_all_snapshots(tmpdir)
                for s in snaps:
                    if s["ok"]:
                        desc = vision_describe(s["path"], label)
                        lines.append(f"{label}: {desc}")
                    else:
                        lines.append(f"{label}: [capture failed]")
            ptz("192.168.1.102", "preset1")

    result = "\n".join(lines)
    if not include_vision:
        _context_cache["text"] = result
        _context_cache["ts"] = time.time()
    return result

# ── Conversation history (per session, resets on restart) ─────────────────────

_history: list = []   # list of {"role": "user"|"assistant", "content": str}

def chat(user_text: str, include_vision: bool = False) -> str:
    """Send a message through qwen (or Gemini fallback) with context."""
    context = build_context(include_vision=include_vision)
    message = f"{context}\n\nUser: {user_text}"

    _history.append({"role": "user", "content": message})

    if ollama_available():
        print(f"  [LLM] {OLLAMA_MODEL}")
        try:
            reply = ask_qwen(_history)
        except Exception as e:
            print(f"  [LLM] qwen failed: {e}, falling back to Gemini")
            reply = ask_gemini_text(message)
    else:
        print(f"  [LLM] Gemini fallback (Ollama unreachable)")
        try:
            reply = ask_gemini_text(message)
        except Exception as e:
            reply = f"Both AI backends unavailable: {e}"

    _history.append({"role": "assistant", "content": reply})

    # Keep history bounded
    if len(_history) > 20:
        _history.pop(0)
        _history.pop(0)

    return reply

# ── Telegram polling ──────────────────────────────────────────────────────────

def get_updates(offset: int) -> list:
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
            params={"timeout": 30, "offset": offset},
            timeout=35)
        if r.status_code == 200:
            return r.json().get("result", [])
    except Exception as e:
        print(f"[poll] {e}")
    return []


def reply(text: str):
    send_telegram_message(text)


def send_typing():
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendChatAction",
            json={"chat_id": TELEGRAM_CHAT, "action": "typing"},
            timeout=5)
    except Exception:
        pass

# ── Slash command handlers ────────────────────────────────────────────────────

def ptz(ip: str, action: str, speed: int = 5) -> bool:
    """Send PTZ command to a camera by IP. action: left/right/up/down/home/preset1/preset2"""
    try:
        if action.startswith("preset"):
            num = action.replace("preset", "")
            params = {"-act": "call", "-preset": num}
            url = f"http://{ip}/cgi-bin/hi3510/preset.cgi"
        else:
            params = {"-step": "0", "-act": action, "-speed": str(speed)}
            url = f"http://{ip}/cgi-bin/hi3510/ptzctrl.cgi"
        r = requests.get(url, params=params, auth=(CAM_USER, CAM_PASS), timeout=5)
        return "Succeed" in r.text
    except Exception as e:
        print(f"[ptz] {ip} error: {e}")
        return False



def _read_pending() -> dict | None:
    try:
        if not os.path.exists(PENDING_FILE):
            return None
        with open(PENDING_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def _update_pending(token: str, status: str, result: str = None,
                    tg_msg_id: int = None, snap_data: list = None):
    try:
        data = _read_pending()
        if not data or data.get("token") != token:
            return
        data["status"] = status
        if result    is not None: data["result"]    = result
        if tg_msg_id is not None: data["tg_msg_id"] = tg_msg_id
        if snap_data is not None: data["snap_data"] = snap_data
        with open(PENDING_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[pending] update error: {e}")


def _send_confirm_message(text: str, token: str) -> int | None:
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT,
                "text": text,
                "parse_mode": "Markdown",
                "reply_markup": {"inline_keyboard": [[
                    {"text": "✅ Yes, proceed", "callback_data": f"confirm_{token}"},
                    {"text": "❌ No, cancel",   "callback_data": f"deny_{token}"},
                ]]},
            }, timeout=10)
        if r.status_code == 200:
            return r.json()["result"]["message_id"]
    except Exception as e:
        print(f"[pending] send confirm error: {e}")
    return None


def _edit_message(msg_id: int, text: str):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText",
            json={"chat_id": TELEGRAM_CHAT, "message_id": msg_id,
                  "text": text, "parse_mode": "Markdown"},
            timeout=10)
    except Exception:
        pass


def _answer_callback(cq_id: str, text: str = ""):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery",
            json={"callback_query_id": cq_id, "text": text}, timeout=5)
    except Exception:
        pass


def _snaps_to_b64(snaps: list) -> list:
    import base64 as b64mod
    result = []
    for s in snaps:
        if s.get("ok") and os.path.exists(s.get("path", "")):
            with open(s["path"], "rb") as f:
                result.append({"name": s["name"], "ok": True,
                                "data": b64mod.b64encode(f.read()).decode()})
        else:
            result.append({"name": s["name"], "ok": False, "data": None})
    return result


def _execute_pending(pending: dict) -> tuple:
    """Returns (result_str, snap_data_or_None)."""
    action = pending.get("action", "")
    params = pending.get("params", {})
    if action == "mower_start":
        ok, reason = start_mowing()
        return ("Mower started ✅" if ok else f"Failed to start mower ❌\n_{reason}_", None)
    if action == "mower_pause":
        ok, reason = pause_mowing()
        return ("Mower paused ✅" if ok else f"Failed to pause ❌\n_{reason}_", None)
    if action == "mower_dock":
        ok, reason = dock_mower()
        return ("Mower docking ✅" if ok else f"Failed to dock ❌\n_{reason}_", None)
    if action == "valve_open":
        from kolo_irrigation import open_valve
        v, m = params.get("valve", "outdoor"), params.get("minutes", 10)
        ok = open_valve(v, m)
        return (f"{v.title()} valve open for {m} min ✅" if ok else f"Failed to open {v} ❌", None)
    if action == "valve_close":
        from kolo_irrigation import close_valve
        v = params.get("valve", "outdoor")
        ok = close_valve(v)
        return (f"{v.title()} valve closed ✅" if ok else f"Failed to close {v} ❌", None)
    if action == "snap":
        with tempfile.TemporaryDirectory() as tmpdir:
            snaps = capture_all_snapshots(tmpdir)
            snap_data = _snaps_to_b64(snaps)
            ok_count = sum(1 for s in snap_data if s["ok"])
            return (f"{ok_count} snapshots ready ✅", snap_data)
    if action == "analyse":
        w = get_weather()
        with tempfile.TemporaryDirectory() as tmpdir:
            snaps = capture_garden_snapshots(tmpdir)
            snap_data = _snaps_to_b64(snaps)
            analysis = analyse_with_gemini(snaps, w)
            state = load_state()
            will_mow, reason = should_mow_today(state, True, w)
            result = f"{analysis}\n\nMow: {'YES ✅' if will_mow else 'NO ❌'} — {reason}"
            return (result, snap_data)
    return (f"Unknown action: {action} ❌", None)


def check_pending_web_actions():
    global _notified_tok
    pending = _read_pending()
    if not pending or pending.get("status") != "pending":
        return
    token = pending["token"]
    if time.time() > pending.get("expires", 0):
        _update_pending(token, "expired")
        msg_id = pending.get("tg_msg_id")
        if msg_id:
            _edit_message(msg_id, f"🌐 *Web action expired*\n_{pending['label']}_")
        return
    if token == _notified_tok:
        return  # already sent successfully for this token
    ts  = datetime.datetime.fromtimestamp(pending["ts"]).strftime("%H:%M:%S")
    txt = f"🌐 *Web action requested* _{ts}_\n\n*{pending['label']}*\n\nConfirm to proceed?"
    msg_id = _send_confirm_message(txt, token)
    if msg_id:
        _notified_tok = token  # only mark as notified if send succeeded
        _update_pending(token, "pending", tg_msg_id=msg_id)
    else:
        print(f"[pending] send failed for {token}, will retry next loop")


def handle_callback_query(cq: dict):
    cq_id = cq["id"]
    data  = cq.get("data", "")
    _answer_callback(cq_id)

    if data.startswith("confirm_"):
        token = data[len("confirm_"):]
        pending = _read_pending()
        if not pending or pending.get("token") != token or pending.get("status") != "pending":
            return
        _update_pending(token, "running")
        try:
            result, snap_data = _execute_pending(pending)
            _update_pending(token, "done", result=result, snap_data=snap_data)
        except Exception as e:
            result = f"Error: {e}"
            _update_pending(token, "error", result=result)
        msg_id = pending.get("tg_msg_id")
        if msg_id:
            _edit_message(msg_id, f"🌐 *{pending['label']}*\n\n✅ Confirmed\n{result}")
        else:
            reply(result)

    elif data.startswith("deny_"):
        token = data[len("deny_"):]
        pending = _read_pending()
        if not pending or pending.get("token") != token:
            return
        _update_pending(token, "denied", result="Cancelled")
        msg_id = pending.get("tg_msg_id")
        if msg_id:
            _edit_message(msg_id, f"🌐 *{pending['label']}*\n\n❌ Cancelled")


def check_and_notify_moisture():
    """Called from main loop at 09:00 and 18:00. Sends Telegram alert if any active
    sensor is dry. Each sensor alerts at most once per day per threshold level."""
    global _moisture_last_check_hour, _moisture_alerted_today, _moisture_alert_date

    now = datetime.datetime.now()
    today = now.strftime("%Y-%m-%d")
    hour  = now.hour

    # Reset per-sensor tracking on new day
    if today != _moisture_alert_date:
        _moisture_alerted_today = set()
        _moisture_alert_date = today

    # Only fire at the designated hours, once per hour
    if hour not in _MOISTURE_CHECK_HOURS or hour == _moisture_last_check_hour:
        return
    _moisture_last_check_hour = hour

    try:
        from kolo_soil import get_soil_data, SENSORS
        from kolo_irrigation import MOISTURE_LOW, MOISTURE_CRITICAL
        soil = get_soil_data(use_cache=False)
    except Exception as e:
        print(f"[moisture check] soil read error: {e}")
        return

    # Which valve serves each sensor
    SENSOR_TO_VALVE = {
        "greenhouse":        "greenhouse",
        "greenhouse_basil":  "greenhouse",
        "cassa_bassa":       "outdoor",
        "cassa_bassa_serra": "outdoor",
        "cassa_alta":        "outdoor",
        "fragole":           "outdoor",
    }

    critical, low = [], []
    valves_needed: set = set()
    for key in SENSORS:
        s = soil.get(key, {})
        if not s.get("active", True):
            continue
        m = s.get("moisture")
        if m is None:
            continue
        label = s.get("label", key)
        alert_key = f"{key}:{today}"
        if m <= MOISTURE_CRITICAL and alert_key + ":critical" not in _moisture_alerted_today:
            critical.append(f"🚨 *{label}*: {m}% — water immediately")
            _moisture_alerted_today.add(alert_key + ":critical")
            valves_needed.add(SENSOR_TO_VALVE.get(key, "outdoor"))
        elif m <= MOISTURE_LOW and alert_key + ":low" not in _moisture_alerted_today:
            low.append(f"⚠️ *{label}*: {m}% — needs water")
            _moisture_alerted_today.add(alert_key + ":low")
            valves_needed.add(SENSOR_TO_VALVE.get(key, "outdoor"))

    if not critical and not low:
        return

    lines = ["💧 *Watering reminder*"]
    lines.extend(critical)
    lines.extend(low)
    if valves_needed:
        valve_hints = {"greenhouse": "greenhouse valve", "outdoor": "outdoor valve"}
        hint = " and ".join(valve_hints[v] for v in sorted(valves_needed))
        lines.append(f"\n_Open the {hint} — use /water [greenhouse|outdoor] [minutes]_")
    else:
        lines.append("\n_Use /water to check status and open valves._")
    try:
        send_telegram_message("\n".join(lines))
        print(f"[moisture check] alert sent: {len(critical)} critical, {len(low)} low, valves: {valves_needed}")
    except Exception as e:
        print(f"[moisture check] send error: {e}")


def cmd_water():
    """Show soil moisture + valve status. Usage: /water [outdoor|greenhouse] [minutes]"""
    from kolo_soil import get_soil_data, SENSORS
    from kolo_irrigation import get_all_valve_status, open_valve, VALVES

    # Parse optional args from the message text (e.g. "/water outdoor 15")
    parts = _last_text.lower().split()
    valve_arg = None
    duration = 10  # default minutes
    for p in parts[1:]:
        if p in VALVES:
            valve_arg = p
        elif p.isdigit():
            duration = max(1, min(60, int(p)))

    if valve_arg:
        # Open requested valve
        reply(f"Opening {valve_arg} valve for {duration} min...")
        ok = open_valve(valve_arg, duration)
        if ok:
            reply(f"✅ {VALVES[valve_arg][1]} valve open — auto-closes in {duration} min.")
        else:
            reply(f"❌ Failed to open {valve_arg} valve.")
        return

    # No valve specified → show moisture + valve status
    soil = get_soil_data(use_cache=False)
    valves = get_all_valve_status()

    lines = ["💧 *Soil moisture*"]
    for key in SENSORS:
        s = soil.get(key, {})
        if not s.get("moisture"):
            continue
        label  = s.get("label", key)
        active = s.get("active", True)
        m      = s.get("moisture")
        suffix = " _(inactive)_" if not active else ""
        flag   = " ⚠️" if active and m <= 30 else ("🚨" if active and m <= 15 else "")
        lines.append(f"  {label}: *{m}%*{flag}{suffix}")

    lines.append("\n🔧 *Valves*")
    for v, status in valves.items():
        label = status.get("label", v)
        state = "OPEN" if status.get("open") else "closed"
        bat   = status.get("battery", "?")
        cd    = status.get("countdown", 0)
        cd_str = f", closes in {cd//60}min" if cd and status.get("open") else ""
        lines.append(f"  {label}: {state}, {bat}% battery{cd_str}")

    lines.append("\n_/water outdoor [min] or /water greenhouse [min] to irrigate_")
    reply("\n".join(lines))


def cmd_help():
    reply(
        "KoloAgent commands:\n\n"
        "/snap     - snapshots (Greenhouse ×2, Beds, Overview ×2)\n"
        "/weather  - weather + all sensors\n"
        "/status   - mower + schedule status\n"
        "/water    - soil moisture + valve control\n"
        "/analyse  - grass analysis + mow recommendation\n"
        "/mow      - same as /analyse (suggestion only)\n"
        "/start    - start mower now (manual)\n"
        "/pause    - pause mower\n"
        "/dock     - return to dock\n"
        "/log      - last log lines\n"
        "/help     - this list\n\n"
        "Or just write anything — I'll answer!"
    )


def cmd_snap():
    """Take snapshots from preset 1 and preset 2, then return to preset 1."""
    reply("Taking snapshots (7 pictures: Garden ×2, Greenhouse ×2, Beds, Overview ×2)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        snaps = capture_all_snapshots(tmpdir)  # handles PTZ internally
        ok = [s for s in snaps if s["ok"]]
        if ok:
            send_telegram_media_group(ok, [s["name"] for s in ok])
        else:
            reply("No snapshots captured.")


def cmd_weather():
    w = get_weather()
    if not w:
        reply("Could not fetch weather.")
        return

    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from kolo_aqara import get_sensor_data
        aqara = get_sensor_data()
        sensor_lines = []
        for key, lbl in [("kolonihavehus","Indoor"),("outdoor","Outdoor"),("greenhouse","Greenhouse")]:
            s = aqara.get(key, {})
            if s:
                sensor_lines.append(f"{lbl}: {s.get('temperature','?')}C, {s.get('humidity','?')}% RH")
        indoor_str = ("\n\nAqara sensors:\n" + "\n".join(sensor_lines)) if sensor_lines else ""
    except Exception:
        indoor_str = ""

    try:
        from kolo_soil import get_soil_data, SENSORS as SOIL_SENSORS
        soil = get_soil_data()
        soil_lines = []
        for key in SOIL_SENSORS:
            s = soil.get(key, {})
            if not s:
                continue
            lbl = s.get("label", key)
            m = s.get("moisture", "?")
            t = s.get("temperature", "?")
            active = s.get("active", True)
            suffix = " (inactive)" if not active else ""
            soil_lines.append(f"{lbl}: {m}%, {t}C{suffix}")
        soil_str = ("\n\nSoil moisture:\n" + "\n".join(soil_lines)) if soil_lines else ""
    except Exception:
        soil_str = ""

    reply(
        f"Weather - Hedehusene\n"
        f"Temp: {w['temp_c']}C (feels {w['feels_like']}C)\n"
        f"Conditions: {w['desc']}\n"
        f"Wind: {w['wind_kmh']} km/h\n"
        f"Humidity: {w['humidity']}%\n"
        f"Rain today: {w['rain_mm_today']}mm\n"
        f"Tomorrow: {w['max_c_tomorrow']}C, rain {w['rain_tomorrow']}mm"
        f"{indoor_str}"
        f"{soil_str}"
    )


def cmd_status():
    state = load_state()
    indego = get_indego_state()
    state_map = {258: "Docked", 257: "Charging", 1: "Mowing",
                 64512: "Offline", 64513: "Offline"}
    mower_state = state_map.get(indego.get("state"), f"State {indego.get('state','?')}")
    mowed_pct   = indego.get("mowed", "?")
    days_ago    = days_since_last_mow(state)
    consec      = state.get("consecutive_100pct", 0)
    last_mow    = state.get("last_mow_date") or "never"
    history     = state.get("mow_history", [])
    last_pct    = history[-1]["mowed_pct"] if history else "?"
    w = get_weather()
    will_mow, reason = should_mow_today(state, True, w)
    ollama_ok = ollama_available()
    reply(
        f"Mower: {mower_state} | Mowed: {mowed_pct}%\n"
        f"Last session coverage: {last_pct}%\n"
        f"Last mow: {last_mow} ({days_ago}d ago)\n"
        f"Consecutive 100%: {consec}/2\n\n"
        f"Next mow: {'YES - ' + reason if will_mow else 'NO - ' + reason}\n\n"
        f"AI: {OLLAMA_MODEL + ' (local)' if ollama_ok else 'Gemini (fallback)'}"
    )


def cmd_analyse():
    reply("Capturing + analysing grass...")
    w = get_weather()
    with tempfile.TemporaryDirectory() as tmpdir:
        snaps = capture_all_snapshots(tmpdir)
        send_telegram_media_group(snaps, [s["name"] for s in snaps])
        analysis = analyse_with_gemini(snaps, w)
        state = load_state()
        will_mow, reason = should_mow_today(state, True, w)
        w_line = f"{w['temp_c']}C, {w['desc']}, wind {w['wind_kmh']} km/h" if w else "N/A"
        reply(
            f"Grass analysis - {datetime.date.today()}\n"
            f"Weather: {w_line}\n\n"
            f"{analysis}\n\n"
            f"Mow decision: {'YES - ' + reason if will_mow else 'NO - ' + reason}"
        )


def cmd_mow():
    reply("Analysing grass conditions...")
    w = get_weather()
    with tempfile.TemporaryDirectory() as tmpdir:
        snaps = capture_all_snapshots(tmpdir)
        send_telegram_media_group(snaps, [s["name"] for s in snaps])
        analysis = analyse_with_gemini(snaps, w)
        state = load_state()
        needs = "yes" in analysis.lower().split("needs_mowing:")[-1][:10] or \
                "soon" in analysis.lower().split("needs_mowing:")[-1][:10]
        will_mow, reason = should_mow_today(state, needs, w)
        if will_mow:
            reply(f"Recommendation: MOW TODAY\nReason: {reason}\n\n{analysis}\n\nUse /start to start the mower.")
        else:
            reply(f"Recommendation: NO MOW\nReason: {reason}\n\n{analysis}")


def cmd_start():
    reply("Starting mower...")
    if start_mowing():
        reply("Mower started!")
    else:
        reply("Failed to start mower.")


def cmd_pause():
    reply("Pausing mower...")
    if pause_mowing():
        reply("Mower paused.")
    else:
        reply("Failed to pause.")


def cmd_dock():
    reply("Returning mower to dock...")
    if dock_mower():
        reply("Mower returning to dock.")
    else:
        reply("Failed to send dock command.")


def cmd_log():
    if not os.path.exists(LOG_FILE):
        reply("No log file yet.")
        return
    with open(LOG_FILE) as f:
        lines = f.readlines()
    last = "".join(lines[-20:]).strip()
    reply(f"Last log:\n{last}" if last else "Log is empty.")


COMMANDS = {
    "/help":    cmd_help,
    "/snap":    cmd_snap,
    "/weather": cmd_weather,
    "/status":  cmd_status,
    "/water":   cmd_water,
    "/analyse": cmd_analyse,
    "/mow":     cmd_mow,
    "/start":   cmd_start,
    "/pause":   cmd_pause,
    "/dock":    cmd_dock,
    "/log":     cmd_log,
}

# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print(f"KoloBot started — polling Telegram...")
    ollama_ok = ollama_available()
    backend = f"{OLLAMA_MODEL} via Ollama" if ollama_ok else "Gemini fallback"
    reply(f"KoloAgent online. AI: {backend}\nSend /help for commands or just chat!")
    offset = 0
    while True:
        check_and_notify_moisture()
        check_pending_web_actions()
        updates = get_updates(offset)
        for upd in updates:
            offset = upd["update_id"] + 1

            # Inline keyboard button press
            if "callback_query" in upd:
                cq = upd["callback_query"]
                if cq.get("message", {}).get("chat", {}).get("id") == ALLOWED_CHAT:
                    handle_callback_query(cq)
                continue

            msg = upd.get("message", {})
            chat_id = msg.get("chat", {}).get("id")
            text = msg.get("text", "").strip()

            if chat_id != ALLOWED_CHAT:
                continue

            print(f"[{datetime.datetime.now():%H:%M}] {text}")
            global _last_text
            _last_text = text

            # Slash commands bypass LLM
            cmd_key = text.lower().split()[0].split("@")[0] if text else ""
            handler = COMMANDS.get(cmd_key)
            if handler:
                try:
                    handler()
                except Exception as e:
                    reply(f"Error: {e}")
            elif text:
                # Keyword intent detection — reliable pre-LLM filter
                tl = text.lower()
                intent = None
                if any(w in tl for w in ["foto", "photo", "picture", "snap", "kamera",
                                          "camera", "fotografi", "billede", "fammi vedere",
                                          "show me", "fai una foto", "tag et billede"]):
                    intent = cmd_snap
                elif any(w in tl for w in ["analisi", "analyse", "analyze", "erba", "grass",
                                            "prato", "græs", "lawn", "mow condition",
                                            "kondition", "how is the grass", "com'è il prato"]):
                    intent = cmd_analyse
                elif any(w in tl for w in ["weather", "tempo", "vejr", "rain", "pioggia",
                                            "regn", "temperature", "temperatura", "temperatur",
                                            "forecast", "previsioni"]):
                    intent = cmd_weather
                elif any(w in tl for w in ["status", "mower", "tagliaerba", "raider",
                                            "indego", "mow", "taglio", "klip", "tosaerba",
                                            "schedule", "pianificazione", "planlægning"]):
                    intent = cmd_status

                if intent:
                    try:
                        intent()
                    except Exception as e:
                        reply(f"Error: {e}")
                else:
                    # General chat → LLM
                    try:
                        send_typing()
                        answer = chat(text)
                        reply(answer)
                    except Exception as e:
                        reply(f"Error: {e}")


if __name__ == "__main__":
    main()
