#!/usr/bin/env python3
"""
Kolonihave Grass-Cutting Agent
- Captures snapshots from 4 cameras via Pi Zero
- Analyses grass with Gemini Vision
- Checks weather via wttr.in
- Sends Telegram notification with photos + recommendation
- Optionally triggers Bosch Indego mower
"""

import os, json, time, datetime, subprocess, tempfile, requests, base64, math

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT   = os.environ.get("TELEGRAM_CHAT", "")

CAM_USER        = "admin"
CAM_PASS        = os.environ.get("CAM_PASS", "")

PI_HOST         = "100.92.207.62"
PI_USER         = "kolo"
PI_SSH_KEY      = os.path.expanduser("~/.ssh/claude_remote")

TOKENS_FILE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indego_tokens.json")
VERIFIER_FILE   = "/tmp/indego_verifier.json"
INDEGO_CLIENT   = "65bb8c9d-1070-4fb4-aa95-853618acc876"
INDEGO_BASE     = "https://api.indego-cloud.iot.bosch-si.com/api/v1"
INDEGO_ALM_SN   = "326602963"

# ── Scheduling config ─────────────────────────────────────────────────────────
MIN_DAYS_BETWEEN_MOWS = 3      # don't mow more often than this
STATE_FILE = os.path.join(os.path.dirname(__file__), "kolo_state.json")

CAMERAS = [
    # Garden (192.168.1.115) removed — not on LAN, re-add when reconnected
    {"name": "Garden",     "url": f"rtsp://{CAM_USER}:{CAM_PASS}@192.168.1.102/11",
     "ptz": True, "ptz_ip": "192.168.1.102", "ptz_onvif_port": 8080},
    {"name": "GreenInt",   "url": f"rtsp://{CAM_USER}:{CAM_PASS}@192.168.1.107/12",
     "ptz": True, "ptz_ip": "192.168.1.107", "ptz_onvif_port": 8080},
    {"name": "Greenhouse", "url": f"rtsp://{CAM_USER}:{CAM_PASS}@192.168.1.114/12",
     "ptz": True, "ptz_ip": "192.168.1.114", "ptz_onvif_port": 8080},
]

PTZ_PRESETS = ["Preset001", "Preset002"]  # same for all cameras

WEATHER_URL = "https://wttr.in/Hedehusene?format=j1"

# ── SSH / snapshot ────────────────────────────────────────────────────────────

def capture_snapshot(cam: dict, local_path: str) -> bool:
    """Capture one RTSP frame via local ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-rtsp_transport", "udp",
        "-i", cam["url"],
        "-frames:v", "1", "-update", "1", "-q:v", "2", local_path
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        print(f"  [WARN] ffmpeg failed for {cam['name']}: {result.stderr[-200:]}")
        return False
    return True


CAM_PTZ_USER  = CAM_USER
CAM_PTZ_PASS  = CAM_PASS
CAM_PAN_SPEED = 5


def onvif_goto_preset(ip: str, preset_token: str, port: int = 8080) -> bool:
    """Move camera to a named ONVIF preset."""
    body = f"""<?xml version="1.0" encoding="utf-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
  <s:Body>
    <GotoPreset xmlns="http://www.onvif.org/ver20/ptz/wsdl">
      <ProfileToken>MainStreamProfileToken</ProfileToken>
      <PresetToken>{preset_token}</PresetToken>
    </GotoPreset>
  </s:Body>
</s:Envelope>"""
    try:
        r = requests.post(
            f"http://{ip}:{port}/onvif/ptz_service",
            data=body,
            headers={"Content-Type": "application/soap+xml"},
            auth=(CAM_PTZ_USER, CAM_PTZ_PASS),
            timeout=8,
        )
        return "GotoPresetResponse" in r.text
    except Exception as e:
        print(f"  [onvif_goto_preset] {ip} {preset_token} error: {e}")
        return False


def ptz_preset(ip: str, n: int) -> bool:
    """Call a saved preset by number on a specific camera."""
    try:
        r = requests.get(
            f"http://{ip}/cgi-bin/hi3510/preset.cgi",
            params={"-act": "call", "-preset": str(n)},
            auth=(CAM_PTZ_USER, CAM_PTZ_PASS), timeout=5)
        return "Succeed" in r.text
    except Exception as e:
        print(f"  [ptz_preset] {ip} error: {e}")
        return False


def ptz_move(ip: str, direction: str, duration: float, speed: int = CAM_PAN_SPEED) -> bool:
    """Pan/tilt a specific camera for a fixed duration then stop."""
    base = f"http://{ip}/cgi-bin/hi3510/ptzctrl.cgi"
    params = {"-step": "0", "-act": direction, "-speed": str(speed)}
    try:
        requests.get(base, params=params, auth=(CAM_PTZ_USER, CAM_PTZ_PASS), timeout=5)
        time.sleep(duration)
        requests.get(base, params={"-step": "0", "-act": "stop", "-speed": str(speed)},
                     auth=(CAM_PTZ_USER, CAM_PTZ_PASS), timeout=5)
        return True
    except Exception as e:
        print(f"  [ptz_move] {ip} error: {e}")
        return False


def capture_all_snapshots(tmpdir: str) -> list:
    """Capture from all cameras. PTZ cameras shoot 2 angles; fixed cameras shoot once."""
    import threading

    snaps = []

    # Fixed cameras — capture in background while PTZ cameras are moving
    fixed_cams = [c for c in CAMERAS if not c.get("ptz")]
    fixed_results = {}

    def snap_fixed(cam):
        path = os.path.join(tmpdir, f"snap_{cam['name'].lower()}.jpg")
        ok = capture_snapshot(cam, path)
        fixed_results[cam["name"]] = {"name": cam["name"], "path": path, "ok": ok}

    threads = [threading.Thread(target=snap_fixed, args=(c,)) for c in fixed_cams]
    for t in threads:
        t.start()

    # PTZ cameras — ONVIF preset 1 then 2, return to 1
    for cam in [c for c in CAMERAS if c.get("ptz")]:
        ip   = cam["ptz_ip"]
        port = cam.get("ptz_onvif_port", 8080)
        name = cam["name"]
        for i, token in enumerate(PTZ_PRESETS, 1):
            onvif_goto_preset(ip, token, port)
            time.sleep(4)
            path = os.path.join(tmpdir, f"snap_{name.lower()}_{i}.jpg")
            print(f"  Capturing {name} preset {i} ...")
            ok = capture_snapshot(cam, path)
            snaps.append({"name": f"{name} preset {i}", "path": path, "ok": ok})
        onvif_goto_preset(ip, PTZ_PRESETS[0], port)  # return home

    # Collect fixed camera results
    for t in threads:
        t.join()
    for cam in fixed_cams:
        snaps.append(fixed_results[cam["name"]])

    return snaps


def capture_garden_snapshots(tmpdir: str) -> list:
    """Capture garden camera (192.168.1.102) at both PTZ presets — used for grass analysis."""
    garden = next((c for c in CAMERAS if c.get("ptz_ip") == "192.168.1.102"), None)
    if not garden:
        return []
    ip   = garden["ptz_ip"]
    port = garden.get("ptz_onvif_port", 8080)
    snaps = []
    for i, token in enumerate(PTZ_PRESETS, 1):
        onvif_goto_preset(ip, token, port)
        time.sleep(4)
        path = os.path.join(tmpdir, f"snap_garden_{i}.jpg")
        print(f"  Capturing Garden preset {i} ...")
        ok = capture_snapshot(garden, path)
        snaps.append({"name": f"Garden preset {i}", "path": path, "ok": ok})
    onvif_goto_preset(ip, PTZ_PRESETS[0], port)
    return snaps


# ── Weather ───────────────────────────────────────────────────────────────────

def get_weather() -> dict:
    try:
        r = requests.get(WEATHER_URL, timeout=10)
        d = r.json()
        cur = d["current_condition"][0]
        today = d["weather"][0]
        tomorrow = d["weather"][1] if len(d["weather"]) > 1 else {}

        # Hourly forecast — wttr.in uses time values 0,300,600,...2100 (HHMM without colon)
        now_hour = datetime.datetime.now().hour
        hourly = today.get("hourly", [])
        # Find hours in the next 3h window
        upcoming = [
            h for h in hourly
            if int(h.get("time", 0)) / 100 > now_hour
            and int(h.get("time", 0)) / 100 <= now_hour + 3
        ]
        rain_next_3h = sum(float(h.get("precipMM", 0)) for h in upcoming)
        rain_chance_next_3h = max((int(h.get("chanceofrain", 0)) for h in upcoming), default=0)

        # Closest hourly entry for solar/UV data
        current_h = min(hourly, key=lambda h: abs(int(h.get("time", 0)) / 100 - now_hour)) if hourly else {}

        min_c_tomorrow = int(tomorrow.get("mintempC", 99))
        # Overnight = min of tonight's late hours + tomorrow's early hours
        tonight_hours  = [h for h in hourly if int(h.get("time", 0)) / 100 >= 18]
        tomorrow_hours = tomorrow.get("hourly", [])
        early_tomorrow = [h for h in tomorrow_hours if int(h.get("time", 0)) / 100 <= 6]
        overnight_temps = [int(h.get("tempC", 99)) for h in tonight_hours + early_tomorrow]
        min_c_overnight = min(overnight_temps) if overnight_temps else min_c_tomorrow

        return {
            "temp_c":              int(cur["temp_C"]),
            "feels_like":          int(cur["FeelsLikeC"]),
            "desc":                cur["weatherDesc"][0]["value"],
            "humidity":            int(cur["humidity"]),
            "wind_kmh":            int(cur["windspeedKmph"]),
            "rain_mm_today":       float(today.get("hourly", [{}])[-1].get("precipMM", 0)),
            "rain_next_3h":        round(rain_next_3h, 1),
            "rain_chance_next_3h": rain_chance_next_3h,
            "max_c_today":         int(today["maxtempC"]),
            "min_c_today":         int(today["mintempC"]),
            "max_c_tomorrow":      int(tomorrow.get("maxtempC", 0)),
            "min_c_tomorrow":      min_c_tomorrow,
            "min_c_overnight":     min_c_overnight,
            "rain_tomorrow":       float(tomorrow.get("hourly", [{}])[-1].get("precipMM", 0)),
            "uv_index":            int(cur.get("uvIndex", 0)),
            "cloud_cover":         int(cur.get("cloudcover", 0)),
            "solar_rad_wm2":       float(current_h.get("shortRad", 0)),
            "dew_point_c":         int(current_h.get("DewPointC", 0)),
        }
    except Exception as e:
        print(f"  [WARN] Weather fetch failed: {e}")
        return {}

# ── Gemini Vision ─────────────────────────────────────────────────────────────

def analyse_with_gemini(snaps: list, weather: dict) -> str:
    """Send the main grass snapshot to Gemini with a focused prompt."""
    parts = []

    # Use only the main grass camera (index 0 = Garden/104/12) to save quota
    main_snap = next((s for s in snaps if s["ok"] and os.path.exists(s["path"])), None)
    if not main_snap:
        return "Could not analyse — no snapshots available."

    # Resize to 768×768, JPEG quality 85
    try:
        from PIL import Image as _PIL
        import io as _io
        img = _PIL.open(main_snap["path"])
        img.thumbnail((768, 768), _PIL.LANCZOS)
        buf = _io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        data = base64.b64encode(buf.getvalue()).decode()
    except Exception:
        with open(main_snap["path"], "rb") as f:
            data = base64.b64encode(f.read()).decode()
    parts.append({"inline_data": {"mime_type": "image/jpeg", "data": data}})

    if not parts:
        return "Could not analyse — no snapshots available."

    parts.append({"text": """Analyse this kolonihave garden image. Respond ONLY in this exact format, no extra text:
GRASS_LENGTH: <very short|short|medium|long|very long>
NEEDS_MOWING: <yes|no|soon>
GRASS_COLOR: <yellow|pale|green|lush green>
OBSERVATIONS: <one sentence about grass condition only>"""})

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-3-flash-preview:generateContent?key={GEMINI_API_KEY}")
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 200,
            "thinkingConfig": {"thinkingBudget": 0},
        }
    }
    for attempt in range(3):
        r = requests.post(url, json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        if r.status_code == 429 and attempt < 2:
            wait = 30 * (attempt + 1)
            print(f"  [Gemini] Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        return f"Gemini error {r.status_code}: {r.text[:200]}"
    return "Gemini error: max retries exceeded"


_IRRIGATION_SYSTEM_PROMPT = """You are an irrigation decision assistant for a smart garden system \
located in Hedehusene, Taastrup (Denmark, ~25km west of Copenhagen).
Marine west coast climate (Cfb). Current month context matters for \
evaporation and rain frequency.

## SYSTEM SETUP
- 2 irrigation zones, each with combined drip + nebulizer on a single valve
- Greenhouse zone: fully enclosed, rain never enters, temperature controlled
- Outdoor zone: exposed to weather, urban/suburban setting

## SENSORS AVAILABLE
Greenhouse zone:
- Greenhouse soil moisture %
- Greenhouse Basil soil moisture %
- Greenhouse air temp (°C)
- Greenhouse RH %

Outdoor zone:
- Bed 80cm soil moisture %
- Strawberries soil moisture %
- Germination Box soil moisture %
- Outdoor air temp (°C)
- Outdoor RH %
- Outdoor pressure (hPa)

## DECISION RULES

### Greenhouse valve
Trigger ON if:
- ANY soil sensor drops to <= 65%
- Run for 10 min (adjustable)
- Target moisture: 75-80%
- Ignore weather completely (rain irrelevant)

### Outdoor valve
Trigger ON only if BOTH:
1. ANY soil sensor drops to <= 65%
2. No rain expected in next 24h
- Run for 10 min (adjustable)
- Target moisture: 70-78%
- Skip if rain occurred in last 6h

## TIMING RULES
- Best watering windows: early morning (06:00-10:00) or late afternoon (17:00-20:00)
- Avoid midday (11:00-16:00): high evaporation, water loss before absorption
- If moisture is marginal and current time is midday, set water_now=false and explain \
  to water in evening instead
- Timestamp is provided in the input — use it to determine time of day

## YOUR TASK
When called, you receive current sensor readings and a weather \
forecast summary. You must respond ONLY with a JSON object, \
no explanation, no markdown, exactly this structure:

{
  "greenhouse": {
    "water_now": true or false,
    "reason": "brief reason",
    "duration_min": 10
  },
  "outdoor": {
    "water_now": true or false,
    "reason": "brief reason",
    "duration_min": 10
  }
}"""


def analyse_moisture_with_gemini(soil: dict, weather: dict,
                                  soil_rows=None, weather_rows=None, irr_rows=None,
                                  aqara: dict = None) -> dict | str:
    """Build structured irrigation decision via Gemini. Returns dict on success, str on error."""
    import re as _re

    def _soil(key):
        if soil_rows:
            for row in soil_rows:
                z = row["zone"] if hasattr(row, "keys") else row[0]
                if z == key:
                    return row["moisture"] if hasattr(row, "keys") else row[1]
        s = soil.get(key, {})
        return s.get("moisture") if isinstance(s, dict) else None

    def _aq(zone, field):
        a = (aqara or {}).get(zone, {})
        return a.get(field) if isinstance(a, dict) else None

    if weather_rows:
        w = weather_rows[0]
        _wf = lambda k: w[k] if hasattr(w, "keys") else None
    else:
        _wf = lambda k: weather.get(k)

    rain_3h    = float(_wf("rain_next_3h")  or 0)
    rain_today = float(_wf("rain_mm_today") or 0)
    rain_tmrw  = float(_wf("rain_tomorrow") or 0)

    input_data = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "greenhouse": {
            "temp_c":                  _aq("greenhouse", "temperature"),
            "rh_percent":              _aq("greenhouse", "humidity"),
            "pressure_hpa":            None,
            "soil_greenhouse_percent": _soil("greenhouse"),
            "soil_basil_percent":      _soil("greenhouse_basil"),
        },
        "outdoor": {
            "temp_c":                    _aq("outdoor", "temperature") or _wf("temp_c"),
            "rh_percent":                _aq("outdoor", "humidity")    or _wf("humidity"),
            "pressure_hpa":              _aq("outdoor", "pressure"),
            "soil_bed80_percent":        _soil("cassa_alta"),
            "soil_strawberries_percent": _soil("fragole"),
            "soil_germination_percent":  _soil("cassa_bassa_serra") or _soil("cassa_bassa"),
            "rain_expected_24h":         rain_3h > 0.5 or rain_tmrw > 2.0,
            "rain_last_6h":              rain_today > 0.5,
        },
    }

    full_prompt = _IRRIGATION_SYSTEM_PROMPT + "\n\n" + json.dumps(input_data, indent=2)
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}")
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 300},
    }
    for attempt in range(3):
        r = requests.post(url, json=payload, timeout=30)
        if r.status_code == 200:
            raw = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                m = _re.search(r"\{.*\}", raw, _re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group())
                    except Exception:
                        pass
            return raw
        if r.status_code == 429 and attempt < 2:
            time.sleep(20)
            continue
        return f"Gemini error {r.status_code}: {r.text[:200]}"
    return "Gemini error: max retries exceeded"

# ── Indego ────────────────────────────────────────────────────────────────────

def _indego_headers() -> dict:
    if not os.path.exists(TOKENS_FILE):
        return None
    with open(TOKENS_FILE) as f:
        tokens = json.load(f)
    # Always refresh proactively so refresh token 30-day clock resets daily
    access_token = _refresh_indego_token(tokens)
    if not access_token:
        # Fallback: use existing token if not yet expired
        access_token = tokens.get("access_token", "")
        exp = tokens.get("expires_on", 0)
        if time.time() > exp:
            return None
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Indego/3.8.0 (iPhone; iOS 16.0; Scale/3.00)",
    }


def _refresh_indego_token(tokens: dict) -> str:
    try:
        r = requests.post(
            f"https://prodindego.b2clogin.com/prodindego.onmicrosoft.com"
            f"/b2c_1a_signup_signin/oauth2/v2.0/token",
            data={
                "client_id":     INDEGO_CLIENT,
                "grant_type":    "refresh_token",
                "refresh_token": tokens["refresh_token"],
            }, timeout=60)
        if r.status_code == 200:
            new_tokens = r.json()
            with open(TOKENS_FILE, "w") as f:
                json.dump(new_tokens, f)
            print("  [Indego] Token refreshed.")
            # Warn if new refresh token expires within 7 days
            rt_exp = int(new_tokens.get("refresh_token_expires_in", 0))
            if 0 < rt_exp < 7 * 86400:
                days_left = rt_exp // 86400
                send_telegram_message(
                    f"WARNING: Bosch Indego refresh token expires in {days_left} day(s).\n"
                    f"Re-authentication required soon — contact agent admin."
                )
            return new_tokens["access_token"]
        elif r.status_code == 400:
            send_telegram_message(
                "WARNING: Bosch Indego refresh token has expired.\n"
                "Mower control is unavailable until re-authentication."
            )
    except Exception as e:
        print(f"  [Indego] Refresh failed: {e}")
    return ""


def get_indego_state() -> dict:
    headers = _indego_headers()
    if not headers:
        return {"error": "No token"}
    try:
        r = requests.get(f"{INDEGO_BASE}/alms/{INDEGO_ALM_SN}/state",
                         headers=headers, timeout=60)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        return {"error": str(e)}
    return {"error": f"HTTP {r.status_code}"}


def _indego_command(state: int, label: str) -> bool:
    headers = _indego_headers()
    if not headers:
        return False
    try:
        r = requests.put(f"{INDEGO_BASE}/alms/{INDEGO_ALM_SN}/state",
                         headers=headers, json={"state": state}, timeout=60)
        ok = r.status_code in (200, 204)
        print(f"  [Indego] {label}: {'OK' if ok else f'HTTP {r.status_code}'}")
        return ok
    except Exception as e:
        print(f"  [Indego] {label} failed: {e}")
        return False

def start_mowing() -> bool:  return _indego_command(1,   "Start mowing")
def pause_mowing() -> bool:  return _indego_command(2,   "Pause")
def dock_mower()   -> bool:  return _indego_command(257, "Return to dock")

# ── State / scheduling ───────────────────────────────────────────────────────

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_mow_date": None, "consecutive_100pct": 0, "mow_history": []}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def days_since_last_mow(state: dict) -> int:
    if not state.get("last_mow_date"):
        return 999
    last = datetime.date.fromisoformat(state["last_mow_date"])
    return (datetime.date.today() - last).days


def should_mow_today(state: dict, grass_needs_mow: bool, weather: dict) -> tuple[bool, str]:
    """Returns (should_mow, reason)."""
    days = days_since_last_mow(state)

    # Weather gate
    if weather:
        if weather.get("temp_c", 0) < 8:
            return False, f"Too cold ({weather['temp_c']}°C, min 8°C)"
        if weather.get("rain_mm_today", 0) >= 1.0:
            return False, f"Rain today ({weather['rain_mm_today']}mm)"
        if weather.get("wind_kmh", 0) >= 40:
            return False, f"Too windy ({weather['wind_kmh']} km/h)"
        if weather.get("humidity", 0) >= 75:
            return False, f"Too humid ({weather['humidity']}%, grass likely wet)"
        if weather.get("rain_next_3h", 0) >= 1.0:
            return False, f"Rain forecast in next 3h ({weather['rain_next_3h']}mm, {weather['rain_chance_next_3h']}% chance)"
        # Soil wetness: if it rained heavily yesterday, ground is still wet
        yesterday_rain = state.get("yesterday_rain_mm", 0)
        if yesterday_rain >= 5.0:
            return False, f"Soil likely wet (rained {yesterday_rain}mm yesterday)"

    # Minimum interval
    if days < MIN_DAYS_BETWEEN_MOWS:
        return False, f"Mowed {days}d ago (min {MIN_DAYS_BETWEEN_MOWS}d)"

    if not grass_needs_mow:
        return False, "Grass doesn't need mowing yet"

    # Second cut needed if two consecutive 100% sessions
    double_cut = state.get("consecutive_100pct", 0) >= 2
    reason = "Grass ready" + (" — double cut needed (2x 100%)" if double_cut else "")
    return True, reason


def record_daily_rain(state: dict, weather: dict):
    """Shift today's rain into yesterday_rain for next day's soil check."""
    today_rain = weather.get("rain_mm_today", 0) if weather else 0
    state["yesterday_rain_mm"] = today_rain
    save_state(state)


def record_mow_session(state: dict, mowed_pct):
    today = datetime.date.today().isoformat()
    pct = int(mowed_pct) if mowed_pct not in ("?", None) else 0
    state["last_mow_date"] = today
    state["mow_history"].append({"date": today, "mowed_pct": pct})
    # Track consecutive 100% completions
    if pct >= 100:
        state["consecutive_100pct"] = state.get("consecutive_100pct", 0) + 1
    else:
        state["consecutive_100pct"] = 0
    save_state(state)


# ── Telegram ──────────────────────────────────────────────────────────────────

def send_telegram_message(text: str):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        json={"chat_id": TELEGRAM_CHAT, "text": text},
        timeout=15)


def send_telegram_photo(path: str, caption: str):
    with open(path, "rb") as f:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
            data={"chat_id": TELEGRAM_CHAT, "caption": caption},
            files={"photo": f},
            timeout=30)


def send_telegram_media_group(snaps: list, captions: list):
    """Send up to 10 photos as a media group."""
    media = []
    files = {}
    for i, (s, cap) in enumerate(zip(snaps, captions)):
        if not s["ok"] or not os.path.exists(s["path"]):
            continue
        key = f"photo{i}"
        media.append({
            "type": "photo",
            "media": f"attach://{key}",
            "caption": cap[:1024],
        })
        files[key] = open(s["path"], "rb")

    if not media:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMediaGroup",
            data={"chat_id": TELEGRAM_CHAT, "media": json.dumps(media)},
            files=files,
            timeout=60)
    finally:
        for f in files.values():
            f.close()

# ── Main ──────────────────────────────────────────────────────────────────────

def check_frost_risk(weather: dict) -> str | None:
    """
    Return a Telegram alert string if there is frost/freeze risk for pipes,
    otherwise return None.

    Levels:
      CRITICAL : temp already <= 0 deg OR overnight min <= -2 deg
      WARNING  : overnight min 1-3 deg (near-freezing risk)
    """
    if not weather:
        return None

    temp_now      = weather.get("temp_c", 99)
    min_overnight = weather.get("min_c_overnight", 99)
    min_tomorrow  = weather.get("min_c_tomorrow", 99)

    if temp_now <= 0 or min_overnight <= -2:
        level = "CRITICAL"
        icon  = "\U0001f9ca"
        actions = (
            "- Open the WOOX water valve slightly (trickling prevents freezing)\n"
            "- Drain outdoor hoses and disconnect them\n"
            "- Insulate exposed pipe sections with foam or rags\n"
            "- Check the greenhouse — keep door closed to retain heat"
        )
    elif min_overnight <= 3 or min_tomorrow <= 0:
        level = "WARNING"
        icon  = "\u26a0\ufe0f"
        actions = (
            "- Monitor overnight — may need to drip water through valve\n"
            "- Disconnect hoses from outdoor taps\n"
            "- Close greenhouse door before sunset"
        )
    else:
        return None

    return (
        f"{icon} *FROST RISK — {level}*\n"
        f"Current: {temp_now}°C | Overnight min: {min_overnight}°C | "
        f"Tomorrow min: {min_tomorrow}°C\n\n"
        f"Recommended actions:\n{actions}"
    )


def main(trigger_mow: bool = False):
    print(f"\n{'='*60}")
    print(f"Kolonihave Agent — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory() as tmpdir:

        # 1. Capture snapshots
        print("\n[1/4] Capturing camera snapshots...")
        snaps = capture_all_snapshots(tmpdir)
        ok_count = sum(1 for s in snaps if s["ok"])
        print(f"  {ok_count}/{len(snaps)} snapshots captured.")

        # 2. Weather
        print("\n[2/4] Fetching weather...")
        weather = get_weather()
        if weather:
            print(f"  {weather['temp_c']}°C, {weather['desc']}, "
                  f"wind {weather['wind_kmh']} km/h")

        # 2b. Frost/freeze pipe alert
        frost_alert = check_frost_risk(weather)
        if frost_alert:
            print(f"\n  [FROST ALERT] {frost_alert[:80]}...")
            send_telegram_message(frost_alert)

        # 3. Indego state
        print("\n[3/4] Checking mower state...")
        indego = get_indego_state()
        state_map = {258: "Docked", 257: "Charging", 1: "Mowing",
                     0: "Reading status", 64512: "Offline"}
        mower_state = state_map.get(indego.get("state"), f"State {indego.get('state', '?')}")
        mowed_pct   = indego.get("mowed", "?")
        print(f"  Mower: {mower_state}, mowed: {mowed_pct}%")

        # 4. Gemini analysis
        print("\n[4/4] Analysing with Gemini Vision...")
        analysis = analyse_with_gemini(snaps, weather)
        print(f"\n{analysis}")

        # Load scheduling state
        state = load_state()

        # Auto-detect mow session: if mower completed (>=95%) and not yet recorded today
        today_iso = datetime.date.today().isoformat()
        if (mowed_pct not in ("?", None) and int(mowed_pct) >= 95
                and state.get("last_mow_date") != today_iso
                and indego.get("state") in (257, 258)):  # charging or docked after mow
            print(f"  [auto-detect] Mow session detected ({mowed_pct}%) — recording.")
            record_mow_session(state, mowed_pct)

        days_ago = days_since_last_mow(state)
        double_cut = state.get("consecutive_100pct", 0) >= 2

        # Parse grass assessment from Gemini
        needs_mowing = "yes" in analysis.lower().split("needs_mowing:")[-1][:10] or \
                       "soon" in analysis.lower().split("needs_mowing:")[-1][:10]
        state["grass_needs_mow"] = needs_mowing
        save_state(state)

        # Smart scheduling decision
        will_mow, mow_reason = should_mow_today(state, needs_mowing, weather)

        # Build Telegram message
        date_str = datetime.date.today().strftime("%d %b %Y")
        w_line = ""
        if weather:
            yesterday_rain = state.get("yesterday_rain_mm", 0)
            w_line = (f"Weather: {weather['temp_c']}C | {weather['desc']} | "
                      f"Humidity {weather['humidity']}% | "
                      f"Wind {weather['wind_kmh']} km/h | "
                      f"Rain yesterday {yesterday_rain}mm | "
                      f"Rain today {weather['rain_mm_today']}mm | "
                      f"Next 3h {weather['rain_next_3h']}mm ({weather['rain_chance_next_3h']}%) | "
                      f"Tomorrow {weather['rain_tomorrow']}mm")

        last_mow_str = f"{days_ago}d ago" if days_ago < 999 else "never"
        double_str   = f" [2x 100% -> double cut!]" if double_cut else ""

        if will_mow:
            decision = f"MOW TODAY - {mow_reason}{double_str}"
        else:
            decision = f"NO MOW - {mow_reason}"

        msg = (f"Kolonihave - {date_str}\n\n"
               f"{w_line}\n"
               f"Mower: {mower_state} ({mowed_pct}% done) | Last mow: {last_mow_str}\n\n"
               f"{analysis}\n"
               f"Decision: {decision}")

        # Save today's rain for tomorrow's soil check
        record_daily_rain(state, weather)

        # Send photos + message
        captions = [s["name"] for s in snaps]
        print("\nSending to Telegram...")
        send_telegram_media_group(snaps, captions)
        send_telegram_message(msg)
        print("  Sent.")

        # Manual trigger only — never auto-start from daily run
        if trigger_mow:
            print(f"\nStarting mower (manual --mow flag)...")
            if start_mowing():
                send_telegram_message(f"Mower started! {mow_reason}")
                record_mow_session(state, mowed_pct)
                print("  Mower started.")
                if double_cut:
                    send_telegram_message("Double cut scheduled: mower will run again in 2h.")
            else:
                print("  Failed to start mower.")

    print("\nDone.")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]

    # Direct mower commands (no analysis)
    if "--start" in args:
        start_mowing(); sys.exit()
    if "--pause" in args:
        pause_mowing(); sys.exit()
    if "--dock" in args:
        dock_mower(); sys.exit()

    main(trigger_mow="--mow" in args)
