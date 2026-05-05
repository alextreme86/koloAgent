#!/usr/bin/env python3
"""
Irrigation controller for KoloAgent.
Controls 2 WOOX R4238 valves + reads soil sensors via Tuya cloud (EU).

Valves:
  outdoor     : bf4ea00ce8114ff4ed9hmj  — outdoor raised beds
  greenhouse  : bf56c5af605339901caaxf  — greenhouse
"""

import json, time, logging

TUYA_REGION = "eu"
TUYA_KEY    = "nysqargfs3nuujxvcppp"
TUYA_SECRET = "901eef5d32514f8e98a715361ea336bb"

VALVES = {
    "outdoor":    ("bf4ea00ce8114ff4ed9hmj", "Outdoor"),
    "greenhouse": ("bf56c5af605339901caaxf", "Greenhouse"),
}

# Moisture thresholds (%)
MOISTURE_LOW      = 30   # below this → watering recommended
MOISTURE_CRITICAL = 15   # below this → alert sent

log = logging.getLogger(__name__)


def _cloud():
    import tinytuya
    return tinytuya.Cloud(
        apiRegion=TUYA_REGION,
        apiKey=TUYA_KEY,
        apiSecret=TUYA_SECRET,
    )


def _valve_id(valve: str) -> str:
    if valve not in VALVES:
        raise ValueError(f"Unknown valve '{valve}'. Choose from: {list(VALVES)}")
    return VALVES[valve][0]


def get_valve_status(valve: str) -> dict:
    """Return current valve state: {open, battery, work_state, countdown}"""
    try:
        did = _valve_id(valve)
        c = _cloud()
        r = c.cloudrequest(f"/v1.0/iot-03/devices/{did}/status")
        items = {i["code"]: i["value"] for i in r.get("result", [])}
        return {
            "valve":      valve,
            "label":      VALVES[valve][1],
            "open":       items.get("switch", False),
            "battery":    items.get("battery_percentage"),
            "work_state": items.get("work_state", "?"),
            "countdown":  items.get("countdown", 0),
        }
    except Exception as e:
        log.error("Valve %s status error: %s", valve, e)
        return {"valve": valve, "label": VALVES[valve][1]}


def get_all_valve_status() -> dict:
    """Return status for both valves."""
    return {v: get_valve_status(v) for v in VALVES}


def set_valve(valve: str, open: bool) -> bool:
    """Open or close a valve. Returns True on success."""
    try:
        did = _valve_id(valve)
        c = _cloud()
        r = c.cloudrequest(
            f"/v1.0/iot-03/devices/{did}/commands",
            action="POST",
            post={"commands": [{"code": "switch", "value": open}]},
        )
        ok = r.get("result") is True or r.get("success") is True
        log.info("Valve %s %s: %s", valve, "open" if open else "close", ok)
        return ok
    except Exception as e:
        log.error("Valve %s command error: %s", valve, e)
        return False


def open_valve(valve: str, duration_min: int = 10) -> bool:
    """Open a valve for a set duration (auto-closes via countdown)."""
    try:
        did = _valve_id(valve)
        c = _cloud()
        r = c.cloudrequest(
            f"/v1.0/iot-03/devices/{did}/commands",
            action="POST",
            post={"commands": [
                {"code": "countdown", "value": duration_min * 60},
                {"code": "switch",    "value": True},
            ]},
        )
        ok = r.get("result") is True or r.get("success") is True
        log.info("Valve %s opened for %d min: %s", valve, duration_min, ok)
        return ok
    except Exception as e:
        log.error("Valve %s open error: %s", valve, e)
        return False


def close_valve(valve: str) -> bool:
    return set_valve(valve, False)


def close_all_valves() -> dict:
    """Close both valves. Returns {valve: success}."""
    return {v: close_valve(v) for v in VALVES}


def get_irrigation_summary() -> str:
    """Multi-line summary for Telegram / bot context."""
    lines = []
    for valve, (did, label) in VALVES.items():
        v = get_valve_status(valve)
        if not v:
            lines.append(f"{label} valve: unavailable")
            continue
        state  = "OPEN" if v.get("open") else "closed"
        bat    = v.get("battery", "?")
        cd     = v.get("countdown", 0)
        cd_str = f", closes in {cd//60}min" if cd and v.get("open") else ""
        lines.append(f"{label} valve: {state}, battery {bat}%{cd_str}")
    return "\n".join(lines)


def check_moisture_alerts() -> list[str]:
    """
    Return list of alert strings for sensors below threshold.
    Called from kolo_agent.py daily run.
    """
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from kolo_soil import get_soil_data
        soil = get_soil_data(use_cache=False)
    except Exception as e:
        log.error("Soil data error: %s", e)
        return []

    alerts = []
    for key, s in soil.items():
        if key == "ts" or not isinstance(s, dict):
            continue
        if not s.get("active", True):
            continue  # skip inactive/empty beds
        m = s.get("moisture")
        label = s.get("label", key)
        if m is None:
            continue
        if m <= MOISTURE_CRITICAL:
            alerts.append(f"\U0001f6a8 {label}: moisture critically low ({m}%) — water immediately")
        elif m <= MOISTURE_LOW:
            alerts.append(f"\u26a0\ufe0f {label}: moisture low ({m}%) — watering recommended")
    return alerts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Valve status ===")
    for v in VALVES:
        print(json.dumps(get_valve_status(v), indent=2))
    print()
    print(get_irrigation_summary())
    print()
    alerts = check_moisture_alerts()
    if alerts:
        print("Moisture alerts:")
        for a in alerts:
            print(" ", a)
    else:
        print("Moisture: all OK")
