#!/usr/bin/env python3
"""
KoloAgent — Autonomous Garden Agent Loop

Architecture from the slide:

    env = Environment()       # Pi Zero kolo + sensor readings = garden state
    tools = Tools(env)        # mower control, irrigation valves, moisture sensors,
                              # weather forecast, temperatures, camera visual
    system_prompt = "..."     # goals, constraints, how to act

    while True:
        action = llm.run(system_prompt + env.state)
        env.state = tools.run(action)

The LLM (Claude on AgentMex) observes the current garden state and decides
which tools to call. Tools execute on Pi Zero kolo (camera, sensors) and via
cloud APIs (mower, valves, weather). The loop continues until Claude has
finished its assessment and taken all necessary actions.

Usage:
    python3 kolo_agent_loop.py              # full autonomous run
    python3 kolo_agent_loop.py --dry-run    # assess only, no actions
"""

import os, json, datetime, logging, sys
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────────────────
_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_ENV):
    with open(_ENV) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

import anthropic

log = logging.getLogger(__name__)

# ── System Prompt ─────────────────────────────────────────────────────────────
# "Goals, constraints, and how to act"

SYSTEM_PROMPT = """You are KoloAgent, the autonomous manager of a Danish kolonihave (allotment garden) in Hedehusene.

## Goals
1. Keep the grass trimmed — mow when the grass needs it and conditions allow
2. Keep plants watered — irrigate when soil moisture is low
3. Monitor plant health — track temperatures, moisture trends, frost risk
4. Keep the owner informed — send Telegram notifications for significant actions and alerts

## How to act
- Gather data first (sensors, weather, camera) before making any decisions
- After collecting all relevant data, decide what actions (if any) to take
- Always notify the owner when you start/stop mowing or irrigation
- Check the camera to visually confirm grass state before sending the mower
- If in doubt, gather more information before taking irreversible actions

## Mowing constraints
- Minimum 3 days between mowing sessions
- Only mow if ALL of these are true:
  - Temperature ≥ 8°C
  - No rain expected in next 3 hours (< 1mm)
  - Wind speed < 40 km/h
  - Humidity < 75%
  - No heavy rain yesterday (< 5mm) — ground would still be wet
- Camera must confirm grass is "medium" length or longer before mowing
- Send a Telegram notification before and after mowing

## Irrigation constraints
- Greenhouse zone: water if any sensor ≤ 58% moisture; target 75%; stop at 80%
- Outdoor zone: water if any sensor ≤ 55% moisture AND no rain expected in next 24h
- Avoid watering between 11:00–16:00 (peak evaporation, poor water efficiency)
- Duration: based on gap between current and target moisture (roughly 1%/min)
- Maximum session: 20 minutes per zone
- Notify owner when opening/closing valves

## Frost alert
- If overnight temperature forecast < 2°C: send a frost warning notification immediately
  (risk of pipe freeze — owner needs to take preventive action)

## Finish
When done with your full assessment and any actions, write a brief summary:
what you observed, what you decided, and what you did (or chose not to do and why).
"""

# ── Environment ───────────────────────────────────────────────────────────────
# "env = Environment()" — represents the state of the garden

class KoloEnvironment:
    """
    The environment is the Pi Zero kolo + cloud sensors.
    It holds the scheduling state (last mow date, etc.) and provides
    the initial context message that seeds the agent loop.
    """

    _STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kolo_state.json")

    def load_schedule(self) -> dict:
        try:
            return json.loads(Path(self._STATE_FILE).read_text())
        except Exception:
            return {}

    def update_last_mow(self, date_iso: str):
        state = self.load_schedule()
        state["last_mow_date"] = date_iso
        Path(self._STATE_FILE).write_text(json.dumps(state, indent=2))

    def as_context(self) -> str:
        """Initial env.state — seeded into the first user message."""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        schedule = self.load_schedule()
        last_mow = schedule.get("last_mow_date") or "never"
        days_since = "unknown"
        if last_mow and last_mow != "never":
            try:
                d = datetime.date.fromisoformat(last_mow)
                days_since = str((datetime.date.today() - d).days)
            except Exception:
                pass
        return (
            f"Current time: {now}\n"
            f"Last mow date: {last_mow} ({days_since} days ago)\n\n"
            "Please assess the garden and take any appropriate actions."
        )


# ── Tool implementations ──────────────────────────────────────────────────────
# "tools = Tools(env)" — the tool functions that update env.state

def _import_kolo_agent():
    """Import kolo_agent module once and cache it."""
    if "kolo_agent" not in sys.modules:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "kolo_agent",
            os.path.join(os.path.dirname(__file__), "kolo_agent.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules["kolo_agent"] = mod
    return sys.modules["kolo_agent"]


def _tool_get_moisture_sensors(use_cache: bool = True) -> dict:
    try:
        from kolo_soil import get_soil_data
        data = get_soil_data(use_cache=use_cache)
        return {k: v for k, v in data.items() if k != "ts"}
    except Exception as e:
        return {"error": str(e)}


def _tool_get_temperatures() -> dict:
    try:
        from kolo_aqara import get_sensor_data
        return get_sensor_data()
    except Exception as e:
        return {"error": str(e)}


def _tool_get_weather_forecast() -> dict:
    try:
        ka = _import_kolo_agent()
        return ka.get_weather()
    except Exception as e:
        return {"error": str(e)}


def _tool_capture_camera_visual(camera: str = "garden") -> dict:
    """Capture RTSP snapshot(s) and return Gemini vision analysis."""
    import tempfile
    try:
        ka = _import_kolo_agent()
        cam_map = {
            "garden":      lambda cs: [c for c in cs if c.get("ptz_ip") == "192.168.1.102"],
            "greenhouse":  lambda cs: [c for c in cs if "green" in c["name"].lower()],
            "all":         lambda cs: cs,
        }
        selector = cam_map.get(camera, cam_map["garden"])
        cams = selector(ka.CAMERAS) or ka.CAMERAS[:1]

        with tempfile.TemporaryDirectory() as tmpdir:
            snaps = []
            for cam in cams:
                path = os.path.join(tmpdir, f"snap_{cam['name'].lower().replace(' ', '_')}.jpg")
                ok = ka.capture_snapshot(cam, path)
                snaps.append({"name": cam["name"], "path": path, "ok": ok})

            if not any(s["ok"] for s in snaps):
                return {"error": "All captures failed — cameras may be offline"}

            analysis = ka.analyse_with_gemini(snaps, {})
            return {
                "camera": camera,
                "captured": sum(1 for s in snaps if s["ok"]),
                "analysis": analysis,
            }
    except Exception as e:
        return {"error": str(e)}


def _tool_get_mower_status() -> dict:
    try:
        ka = _import_kolo_agent()
        return ka.get_indego_state()
    except Exception as e:
        return {"error": str(e)}


def _tool_control_mower(action: str) -> dict:
    try:
        ka = _import_kolo_agent()
        dispatch = {
            "start": ka.start_mowing,
            "pause": ka.pause_mowing,
            "dock":  ka.dock_mower,
        }
        if action not in dispatch:
            return {"error": f"Unknown action '{action}'. Use: start, pause, dock"}
        ok, reason = dispatch[action]()
        return {"action": action, "success": ok, "reason": reason or "OK"}
    except Exception as e:
        return {"error": str(e), "action": action}


def _tool_open_valve(zone: str, duration_min: int) -> dict:
    try:
        from kolo_irrigation import open_valve
        duration_min = max(1, min(20, int(duration_min)))
        ok = open_valve(zone, duration_min)
        return {"zone": zone, "duration_min": duration_min, "success": ok}
    except Exception as e:
        return {"error": str(e), "zone": zone}


def _tool_close_valve(zone: str) -> dict:
    try:
        from kolo_irrigation import close_valve
        ok = close_valve(zone)
        return {"zone": zone, "success": ok}
    except Exception as e:
        return {"error": str(e), "zone": zone}


def _tool_send_notification(message: str) -> dict:
    try:
        import requests
        token = os.environ.get("TELEGRAM_TOKEN", "")
        chat  = os.environ.get("TELEGRAM_CHAT", "")
        if not token or not chat:
            return {"error": "TELEGRAM_TOKEN / TELEGRAM_CHAT not set"}
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        return {"success": r.status_code == 200, "http_status": r.status_code}
    except Exception as e:
        return {"error": str(e)}


# ── Tool dispatch map ─────────────────────────────────────────────────────────

_TOOL_HANDLERS = {
    "get_moisture_sensors":  lambda inp: _tool_get_moisture_sensors(**inp),
    "get_temperatures":      lambda inp: _tool_get_temperatures(),
    "get_weather_forecast":  lambda inp: _tool_get_weather_forecast(),
    "capture_camera_visual": lambda inp: _tool_capture_camera_visual(**inp),
    "get_mower_status":      lambda inp: _tool_get_mower_status(),
    "control_mower":         lambda inp: _tool_control_mower(**inp),
    "open_valve":            lambda inp: _tool_open_valve(**inp),
    "close_valve":           lambda inp: _tool_close_valve(**inp),
    "send_notification":     lambda inp: _tool_send_notification(**inp),
}


def execute_tool(name: str, tool_input: dict) -> str:
    """Dispatch a tool call and return its result as JSON string."""
    log.info("Tool call: %s(%s)", name, json.dumps(tool_input)[:120])
    handler = _TOOL_HANDLERS.get(name)
    if not handler:
        result = {"error": f"Unknown tool: {name}"}
    else:
        try:
            result = handler(tool_input)
        except Exception as e:
            result = {"error": f"Tool execution error: {e}"}
    log.info("  Result: %s", json.dumps(result)[:200])
    return json.dumps(result)


# ── Tool schemas (Claude tool definitions) ────────────────────────────────────
# These tell the LLM what tools are available and how to call them.

TOOLS = [
    {
        "name": "get_moisture_sensors",
        "description": (
            "Read soil moisture, temperature, and battery from all 6 soil sensors via Tuya cloud. "
            "Zones: greenhouse, greenhouse_basil, cassa_bassa (40cm outdoor bed), "
            "cassa_bassa_serra (germination box), cassa_alta (80cm outdoor bed), fragole (strawberries). "
            "Returns moisture %, soil temp °C, battery % per zone."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "use_cache": {
                    "type": "boolean",
                    "description": "Use cached data if <10 min old (default true). Set false to force fresh read."
                }
            },
            "required": []
        }
    },
    {
        "name": "get_temperatures",
        "description": (
            "Read temperature and humidity from all Aqara sensors (Aqara Hub E1). "
            "Locations: kolonihavehus (indoor), outdoor, greenhouse. "
            "Returns temp °C, humidity %, pressure hPa (kolonihavehus only)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_weather_forecast",
        "description": (
            "Fetch current weather and 48h forecast for Hedehusene, Denmark from wttr.in. "
            "Returns: current temp °C, wind km/h, humidity %, rain next 3h, "
            "today/tomorrow min/max temps, overnight min, UV index, cloud cover."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "capture_camera_visual",
        "description": (
            "Capture a photo from a garden camera and get an AI visual assessment. "
            "Returns GRASS_LENGTH (very short/short/medium/long/very long), "
            "NEEDS_MOWING (yes/no/soon), GRASS_COLOR, and OBSERVATIONS about grass condition."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "camera": {
                    "type": "string",
                    "enum": ["garden", "greenhouse", "all"],
                    "description": "'garden' for the main grass area camera, 'greenhouse' for the greenhouse camera, 'all' for all cameras"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_mower_status",
        "description": (
            "Get the current state of the Bosch Indego S+ 400 robot mower. "
            "Returns: state code and description (docked/mowing/charging/error), "
            "mowed percentage, battery level, and last activity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "control_mower",
        "description": (
            "Send a command to the Bosch Indego robot mower via Bosch cloud API. "
            "Only call 'start' after verifying weather conditions AND grass assessment. "
            "Always notify the owner before starting."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "pause", "dock"],
                    "description": (
                        "'start' — send to mow the lawn; "
                        "'pause' — pause mowing in place; "
                        "'dock' — return to charging dock"
                    )
                }
            },
            "required": ["action"]
        }
    },
    {
        "name": "open_valve",
        "description": (
            "Open an irrigation valve to water a zone. The valve auto-closes after duration. "
            "Zones: 'greenhouse' (controls greenhouse + basil pot sensors) or "
            "'outdoor' (controls raised beds + strawberries). "
            "Check moisture sensors and weather forecast before calling. "
            "Avoid 11:00–16:00. Max 20 min per session."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "zone": {
                    "type": "string",
                    "enum": ["greenhouse", "outdoor"]
                },
                "duration_min": {
                    "type": "integer",
                    "description": "Minutes to water (1–20)",
                    "minimum": 1,
                    "maximum": 20
                }
            },
            "required": ["zone", "duration_min"]
        }
    },
    {
        "name": "close_valve",
        "description": "Immediately close an irrigation valve.",
        "input_schema": {
            "type": "object",
            "properties": {
                "zone": {
                    "type": "string",
                    "enum": ["greenhouse", "outdoor"]
                }
            },
            "required": ["zone"]
        }
    },
    {
        "name": "send_notification",
        "description": (
            "Send a Telegram message to the garden owner. "
            "Use for: mowing started/stopped, irrigation started/stopped, "
            "frost warnings, critical moisture alerts, daily summary. "
            "Supports HTML formatting (<b>bold</b>, <i>italic</i>)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to send. HTML formatting supported."
                }
            },
            "required": ["message"]
        }
    }
]


# ── Agent Loop ────────────────────────────────────────────────────────────────
# "while True: action = llm.run(system_prompt + env.state); env.state = tools.run(action)"

def run_agent(dry_run: bool = False) -> str:
    """
    Run one full autonomous garden assessment cycle.

    The loop matches the slide pattern:
        env     = KoloEnvironment()     # garden state (Pi Zero kolo)
        tools   = Tools(env)            # callable garden actions (AgentMex executes)
        prompt  = SYSTEM_PROMPT         # goals, constraints, how to act

        while True:
            action   = llm.run(prompt + env.state)   # Claude decides
            env.state = tools.run(action)             # tools update state

    Returns the final agent summary string.
    """
    env    = KoloEnvironment()
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    # env.state — initial context seeded into the conversation
    initial_context = env.as_context()
    if dry_run:
        initial_context += (
            "\n\n<dry_run>Assess the garden fully but do NOT send notifications, "
            "control the mower, or open/close valves.</dry_run>"
        )

    messages = [{"role": "user", "content": initial_context}]

    log.info("KoloAgent starting — %s%s",
             datetime.datetime.now().isoformat(),
             " [DRY RUN]" if dry_run else "")

    # ── Agent loop ────────────────────────────────────────────────────────────
    while True:
        # action = llm.run(system_prompt + env.state)
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # stable prefix — cache it
            }],
            tools=TOOLS,
            messages=messages,
        )

        log.info("LLM response: stop_reason=%s, blocks=%d",
                 response.stop_reason, len(response.content))

        # Append assistant turn to conversation history
        messages.append({"role": "assistant", "content": response.content})

        # Done — no more tool calls
        if response.stop_reason == "end_turn":
            summary = next(
                (b.text for b in response.content if b.type == "text"),
                "Agent completed — no summary produced."
            )
            log.info("Agent done: %s", summary[:300])
            return summary

        # env.state = tools.run(action) — execute each tool Claude called
        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_blocks:
            break  # no tool calls and not end_turn — shouldn't happen, exit safely

        tool_results = []
        for tb in tool_blocks:
            result_json = execute_tool(tb.name, tb.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tb.id,
                "content": result_json,
            })

        # Feed tool results back — this updates env.state for the next iteration
        messages.append({"role": "user", "content": tool_results})

    return "Agent loop exited unexpectedly."


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(os.path.dirname(__file__), "kolo_agent_loop.log")
            ),
            logging.StreamHandler(),
        ],
    )

    parser = argparse.ArgumentParser(description="KoloAgent — autonomous garden loop")
    parser.add_argument("--dry-run", action="store_true",
                        help="Assess garden but take no actions (no mowing, watering, notifications)")
    args = parser.parse_args()

    summary = run_agent(dry_run=args.dry_run)
    print("\n" + "=" * 60)
    print("AGENT SUMMARY")
    print("=" * 60)
    print(summary)
