#!/usr/bin/env python3
"""
Tuya soil sensor integration for KoloAgent.
Reads moisture, temperature, battery from 6 soil sensors via Tuya cloud (EU).

Sensors:
  greenhouse        : bfb1908669a3d57b32sgnl  — main greenhouse
  greenhouse_basil  : bfccd88f1db54fbf5btzjp  — basil pot in greenhouse
  cassa_bassa       : bfe68a7bf65f54633cnfqv  — raised bed 40cm (outdoor)
  cassa_bassa_serra : bf3fb8f496e11d9777mcoa  — raised bed 40cm (greenhouse)
  cassa_alta        : bf918f92a07055b0f8s5af  — raised bed 80cm (outdoor)
  fragole           : bff8de89ab98a67b9abkzx  — strawberries
"""

import json, time, logging
from pathlib import Path

TUYA_REGION = "eu"
TUYA_KEY    = "nysqargfs3nuujxvcppp"
TUYA_SECRET = "901eef5d32514f8e98a715361ea336bb"

# Each entry: (device_id, label, active)
# Set active=False for empty/unused beds — skips moisture alerts
SENSORS = {
    "greenhouse":        ("bfb1908669a3d57b32sgnl", "Greenhouse",              True),
    "greenhouse_basil":  ("bfccd88f1db54fbf5btzjp", "Greenhouse Basil",        True),
    "cassa_bassa":       ("bfe68a7bf65f54633cnfqv", "Bed 40cm (outdoor)",      False),  # not planted
    "cassa_bassa_serra": ("bf3fb8f496e11d9777mcoa", "Germination Box",          True),
    "cassa_alta":        ("bf918f92a07055b0f8s5af", "Bed 80cm (outdoor)",      True),
    "fragole":           ("bff8de89ab98a67b9abkzx", "Strawberries",            True),
}

_cache: dict = {"data": None, "ts": 0.0}
_CACHE_TTL = 600  # 10 min — sensors update slowly (battery powered)

log = logging.getLogger(__name__)



def _get_cloud():
    import tinytuya
    return tinytuya.Cloud(
        apiRegion=TUYA_REGION,
        apiKey=TUYA_KEY,
        apiSecret=TUYA_SECRET,
    )


def _read_all() -> dict:
    c = _get_cloud()
    result = {}
    for name, (did, label, active) in SENSORS.items():
        try:
            r = c.cloudrequest(f"/v1.0/iot-03/devices/{did}/status")
            items = {i["code"]: i["value"] for i in r.get("result", [])}
            # Sensors report in 0.1°C units (e.g. 171 = 17.1°C)
            temp = items.get("temp_current", None)
            if temp is not None:
                temp = float(temp) / 10
            result[name] = {
                "label":       label,
                "active":      active,
                "moisture":    items.get("humidity"),
                "temperature": round(temp, 1) if temp is not None else None,
                "battery":     items.get("battery_percentage"),
            }
        except Exception as e:
            log.warning("Soil sensor %s error: %s", name, e)
            result[name] = {"label": label, "active": active}
    return result


def get_soil_data(use_cache: bool = True) -> dict:
    """
    Return soil sensor readings:
    {
      "greenhouse": {"moisture": 4, "temperature": 14.0, "battery": 100},
      "new_bed":    {"moisture": 15, "temperature": 9.0, "battery": 100},
      "old_bed":    {"moisture": 28, "temperature": 0.0, "battery": 100},
      "ts": 1744800000.0
    }
    """
    global _cache
    now = time.time()
    if use_cache and _cache["data"] and (now - _cache["ts"]) < _CACHE_TTL:
        return _cache["data"]

    try:
        data = _read_all()
    except Exception as e:
        log.error("Tuya cloud error: %s", e)
        return _cache.get("data") or {}

    if data:
        data["ts"] = now
        _cache = {"data": data, "ts": now}

    return data


def format_for_context() -> str:
    """Return a text block for inclusion in kolo_agent.py context."""
    data = get_soil_data()
    if not data:
        return "Soil sensors: unavailable"

    lines = ["Soil sensors:"]
    for key in SENSORS:
        s = data.get(key, {})
        if not s:
            continue
        label  = s.get("label", key)
        active = s.get("active", True)
        m      = s.get("moisture", "?")
        t      = s.get("temperature", "?")
        bat    = s.get("battery", "?")
        suffix = " (inactive)" if not active else ""
        lines.append(f"  {label}: {m}% moisture, {t}°C, battery {bat}%{suffix}")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = get_soil_data(use_cache=False)
    print(json.dumps(data, indent=2))
    print()
    print(format_for_context())
