#!/usr/bin/env python3
"""
Aqara sensor integration for KoloAgent.
Reads temperature + humidity from Hub E1 sensors via Aqara cloud (EU).

Sensors:
  Kolonihus  : lumi.158d0008c99359  (lumi.weather.v1)
  Greenhouse : lumi.54ef4410009bfffc (lumi.sensor_ht.agl02)

Token file: /home/agentmex/aqara_sdk_tokens.json  (refresh handled automatically)
"""

import json, time, logging
from pathlib import Path

TOKEN_FILE = "/home/kolo/aqara_sdk_tokens.json"

SENSORS = {
    "kolonihavehus": "lumi.158d0008c99359",
    "outdoor":       "lumi.158d0008977de0",
    "greenhouse":    "lumi.54ef4410009bfffc",
}

RESOURCES = {
    "0.1.85": "temperature",   # unit: 0.01 °C
    "0.2.85": "humidity",      # unit: 0.01 %
    "0.3.85": "pressure",      # unit: 0.01 hPa  (kolonihus only)
}

_cache: dict = {"data": None, "ts": 0.0}
_CACHE_TTL = 300   # 5 min

log = logging.getLogger(__name__)


def _refresh_tokens(tokens: dict) -> dict | None:
    """Refresh Aqara access token using the refresh_token.
    The SDK's built-in refresh is broken — it passes the full response to
    AqaraTokenInfo which expects snake_case top-level keys, but the API returns
    camelCase keys nested under 'result'. We do it manually here.
    """
    from aqara_iot import AqaraOpenAPI
    openapi = AqaraOpenAPI(country_code="Europe")
    # No token_info set — refresh request goes unauthenticated (sign uses app key only)
    resp = openapi.post("/v3.0/open/api", {
        "intent": "config.auth.refreshToken",
        "data": {"refreshToken": tokens["refresh_token"]},
    })
    if not resp or resp.get("code") != 0:
        log.error("Aqara token refresh failed: %s", resp)
        return None
    result = resp.get("result", {})
    if not result.get("accessToken"):
        log.error("Aqara token refresh: unexpected response format: %s", resp)
        return None
    new = {
        "access_token":  result["accessToken"],
        "refresh_token": result.get("refreshToken", tokens["refresh_token"]),
        "expire_time":   int(time.time()) + int(result.get("expiresIn", 604800)),
        "uid":           tokens.get("uid", ""),
    }
    Path(TOKEN_FILE).write_text(json.dumps(new, indent=2))
    log.info("Aqara token refreshed: %s...", new["access_token"][:12])
    return new


def _get_openapi():
    from aqara_iot import AqaraOpenAPI, AqaraTokenInfo
    tokens = json.loads(Path(TOKEN_FILE).read_text())

    # Refresh proactively if expired or expiring within 5 minutes
    if int(tokens.get("expire_time", 0)) - int(time.time()) < 300:
        refreshed = _refresh_tokens(tokens)
        if refreshed:
            tokens = refreshed
        else:
            log.error("Could not refresh Aqara token — will try with existing token")

    openapi = AqaraOpenAPI(country_code="Europe")
    openapi.token_info = AqaraTokenInfo({
        "access_token":  tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "expires_in":    max(1, int(tokens.get("expire_time", 0)) - int(time.time())),
        "openId":        tokens.get("uid", ""),
    })
    return openapi, tokens


def _save_tokens(openapi, old_tokens: dict):
    """Persist refreshed tokens if they changed (SDK may also refresh them)."""
    ti = openapi.token_info
    if ti and ti.access_token and ti.access_token != old_tokens.get("access_token"):
        new = {
            "access_token":  ti.access_token,
            "refresh_token": ti.refresh_token,
            "expire_time":   ti.expire_time,
            "uid":           ti.uid,
        }
        Path(TOKEN_FILE).write_text(json.dumps(new, indent=2))
        log.info("Aqara tokens updated by SDK refresh.")


def _query(openapi) -> dict:
    resources = []
    for did in SENSORS.values():
        for rid in RESOURCES:
            resources.append({"subjectId": did, "resourceId": rid})

    resp = openapi.post("/v3.0/open/api", {
        "intent": "query.resource.value",
        "data":   {"resources": resources},
    })

    if resp.get("code") != 0:
        log.warning("Aqara query error: %s", resp)
        return {}

    result = {}
    seen = set()
    for item in resp.get("result", []):
        did = item.get("subjectId")
        rid = item.get("resourceId")
        key = (did, rid)
        if key in seen:
            continue
        seen.add(key)

        raw = item.get("value")
        if raw is None:
            continue
        val = float(raw) / 100

        sensor_name = next((k for k, v in SENSORS.items() if v == did), did)
        field = RESOURCES.get(rid)
        if not field:
            continue  # skip battery/signal etc.
        result.setdefault(sensor_name, {})[field] = round(val, 2)

    return result


def get_sensor_data(use_cache: bool = True) -> dict:
    """
    Return sensor readings dict:
    {
      "kolonihus":  {"temperature": 12.11, "humidity": 64.3, "pressure": 1019.4},
      "greenhouse": {"temperature": -0.58, "humidity": 85.4},
      "ts": 1744800000.0
    }
    Returns {} on error.
    """
    global _cache
    now = time.time()
    if use_cache and _cache["data"] and (now - _cache["ts"]) < _CACHE_TTL:
        return _cache["data"]

    try:
        openapi, old_tokens = _get_openapi()
        data = _query(openapi)
        _save_tokens(openapi, old_tokens)
    except Exception as e:
        log.error("Aqara error: %s", e)
        return _cache.get("data") or {}

    if data:
        data["ts"] = now
        _cache = {"data": data, "ts": now}

    return data


def format_for_context() -> str:
    """Return a text block for inclusion in kolo_agent.py context."""
    data = get_sensor_data()
    if not data:
        return "Aqara sensors: unavailable"

    labels = {
        "kolonihavehus": "Indoor",
        "outdoor":       "Outdoor",
        "greenhouse":    "Greenhouse",
    }
    lines = ["Aqara sensors:"]
    for loc, label in labels.items():
        s = data.get(loc, {})
        if not s:
            continue
        temp = s.get("temperature", "?")
        hum  = s.get("humidity", "?")
        pres = s.get("pressure")
        line = f"  {label}: {temp}°C, {hum}% RH"
        if pres:
            line += f", {pres} hPa"
        lines.append(line)

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = get_sensor_data(use_cache=False)
    print(json.dumps(data, indent=2))
    print()
    print(format_for_context())
