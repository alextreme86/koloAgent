#!/usr/bin/env python3
"""
KoloAgent Web Dashboard — kolo.messina.dk
Runs on Raspberry Pi Zero 2, port 5000.
Serves status, camera snapshots, and mower controls.
"""

import os, sys, io, base64, datetime, tempfile, json, secrets, time
_START = str(int(time.time()))  # cache-buster: changes each server restart
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request, send_file, abort, Response, render_template_string
from functools import wraps

from kolo_agent import (
    capture_all_snapshots, get_weather, get_indego_state,
    analyse_with_gemini, analyse_moisture_with_gemini,
    load_state, should_mow_today,
    days_since_last_mow, start_mowing, pause_mowing, dock_mower,
    record_daily_rain, load_moisture_commentary,
)
from kolo_aqara import get_sensor_data as aqara_data
from kolo_soil import get_soil_data, SENSORS as SOIL_SENSORS
from kolo_irrigation import get_all_valve_status, open_valve, close_valve

app = Flask(__name__)

# ── Pending action (Telegram confirmation flow) ───────────────────────────────
PENDING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kolo_pending.json")
PENDING_TTL  = 120  # seconds before auto-expiry


def _write_pending(action: str, label: str, params: dict = None) -> str:
    token = secrets.token_hex(4)
    data  = {
        "token":      token,
        "action":     action,
        "label":      label,
        "params":     params or {},
        "source":     "web",
        "ts":         time.time(),
        "expires":    time.time() + PENDING_TTL,
        "status":     "pending",
        "result":     None,
        "tg_msg_id":  None,
    }
    with open(PENDING_FILE, "w") as f:
        json.dump(data, f)
    return token


def _read_pending(token: str = None) -> dict | None:
    try:
        if not os.path.exists(PENDING_FILE):
            return None
        with open(PENDING_FILE) as f:
            data = json.load(f)
        if token and data.get("token") != token:
            return None
        return data
    except Exception:
        return None


# ── Auth (disabled — open access, re-enable later) ────────────────────────────
DASHBOARD_TOKEN = os.environ.get("KOLO_DASH_TOKEN", "kolo2026")

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        return f(*args, **kwargs)
    return decorated

# ── API endpoints ─────────────────────────────────────────────────────────────

@app.route("/ping")
def ping():
    return "ok"


@app.route("/live")
def live():
    try:
        status  = _collect_status()
        aqara   = aqara_data(use_cache=True)
        soil    = get_soil_data(use_cache=True)
        valves  = get_all_valve_status()
        w = status.get("weather", {})
        mow = status.get("mower", {})
        dec = status.get("mow_decision", {})
        lines = [
            f"=== KoloDash /live — {status.get('ts','')} ===",
            "",
            f"WEATHER: {w.get('temp_c')}°C feels {w.get('feels_like')}°C  {w.get('desc')}",
            f"  humidity:{w.get('humidity')}%  wind:{w.get('wind_kmh')}km/h  rain_today:{w.get('rain_mm_today')}mm",
            f"  UV:{w.get('uv_index')}  solar:{w.get('solar_rad_wm2')}W/m²  cloud:{w.get('cloud_cover')}%",
            "",
            f"MOWER: {mow.get('state')}  {mow.get('mowed_pct')}%  last_mow:{mow.get('last_mow')} ({mow.get('days_ago')}d ago)",
            f"  decision: {'MOW' if dec.get('will_mow') else 'SKIP'}  reason:{dec.get('reason')}",
            "",
            "SOIL:",
        ]
        for k, v in soil.items():
            if isinstance(v, dict):
                lines.append(f"  {k}: moisture={v.get('moisture')}%  temp={v.get('temperature')}°C  bat={v.get('battery')}%  active={v.get('active')}")
        lines.append("")
        lines.append("CLIMATE:")
        for k, v in aqara.items():
            if isinstance(v, dict):
                lines.append(f"  {k}: temp={v.get('temperature')}°C  hum={v.get('humidity')}%")
        lines.append("")
        lines.append("VALVES:")
        for k, v in valves.items():
            lines.append(f"  {k}: open={v.get('open')}  countdown={v.get('countdown')}s  bat={v.get('battery')}%")
    except Exception as e:
        lines = [f"ERROR: {e}"]
    return Response("\n".join(lines), content_type="text/plain")


def _collect_status() -> dict:
    w       = get_weather()
    indego  = get_indego_state()
    state   = load_state()
    state_map = {258: "Docked", 257: "Charging", 1: "Mowing", 64513: "Offline"}
    mower_state = state_map.get(indego.get("state"), f"Unknown ({indego.get('state','?')})")
    mowed_pct   = indego.get("mowed", 0)
    days_ago    = days_since_last_mow(state)
    grass_needs_mow = state.get("grass_needs_mow", True)
    will_mow, reason = should_mow_today(state, grass_needs_mow, w)
    history     = state.get("mow_history", [])
    consec      = state.get("consecutive_100pct", 0)
    yesterday_rain = state.get("yesterday_rain_mm", 0)
    return {
        "ts": datetime.datetime.now().isoformat(),
        "weather": w,
        "yesterday_rain_mm": yesterday_rain,
        "mower": {
            "state": mower_state,
            "state_code": indego.get("state"),
            "mowed_pct": mowed_pct,
            "last_mow": state.get("last_mow_date"),
            "days_ago": days_ago,
            "consecutive_100pct": consec,
            "runtime_total_h": round(indego.get("runtime", {}).get("total", {}).get("operate", 0) / 60, 1),
        },
        "mow_decision": {
            "will_mow": will_mow,
            "reason": reason,
        },
        "history": history[-5:],
    }


@app.route("/api/status")
@require_auth
def api_status():
    return jsonify(_collect_status())


@app.route("/api/snap", methods=["POST"])
@require_auth
def api_snap():
    token = _write_pending("snap", "📷 Camera snapshots")
    return jsonify({"status": "pending", "token": token})


@app.route("/api/analyse", methods=["POST"])
@require_auth
def api_analyse():
    token = _write_pending("analyse", "🔍 Grass analysis")
    return jsonify({"status": "pending", "token": token})


@app.route("/api/moisture-analysis", methods=["POST"])
@require_auth
def api_moisture_analysis():
    import sqlite3 as _sq
    DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kolo_data.db")
    soil_rows, weather_rows, irr_rows = [], [], []
    if os.path.exists(DB):
        con = _sq.connect(DB)
        con.row_factory = _sq.Row
        soil_rows = con.execute(
            """SELECT zone, moisture, soil_temp, ts FROM sensor_readings
               WHERE ts >= datetime('now','-30 minutes')
               ORDER BY ts DESC"""
        ).fetchall()
        weather_rows = con.execute(
            """SELECT * FROM weather_readings ORDER BY ts DESC LIMIT 1"""
        ).fetchall()
        irr_rows = con.execute(
            """SELECT zone, ts_open, duration_actual_s, moisture_before, moisture_after
               FROM irrigation_events WHERE status='completed'
               ORDER BY id DESC LIMIT 6"""
        ).fetchall()
        con.close()
    soil_live = get_soil_data(use_cache=True) if not soil_rows else {}
    w_live    = get_weather()                  if not weather_rows else {}
    aq_live   = aqara_data(use_cache=True)
    result = analyse_moisture_with_gemini(soil_live, w_live, soil_rows, weather_rows,
                                          irr_rows=irr_rows, aqara=aq_live)
    return jsonify({"result": result, "ts": datetime.datetime.now().isoformat()})


@app.route("/api/moisture-comment")
@require_auth
def api_moisture_comment():
    return jsonify(load_moisture_commentary())


@app.route("/api/mower/<action>", methods=["POST"])
@require_auth
def api_mower_action(action):
    labels = {"start": "▶️ Start mower", "pause": "⏸️ Pause mower", "dock": "🏠 Dock mower"}
    if action not in labels:
        abort(400, "Unknown action")
    token = _write_pending(f"mower_{action}", labels[action])
    return jsonify({"status": "pending", "token": token})


@app.route("/api/log")
@require_auth
def api_log():
    log_path = os.path.join(os.path.dirname(__file__), "kolo_agent.log")
    if not os.path.exists(log_path):
        return jsonify({"lines": []})
    with open(log_path) as f:
        lines = f.readlines()
    return jsonify({"lines": [l.rstrip() for l in lines[-30:]]})


@app.route("/api/action/status/<token>")
@require_auth
def api_action_status(token):
    pending = _read_pending(token)
    if not pending:
        return jsonify({"status": "not_found"})
    if pending["status"] == "pending" and time.time() > pending["expires"]:
        pending["status"] = "expired"
        with open(PENDING_FILE, "w") as f:
            json.dump(pending, f)
    return jsonify(pending)


@app.route("/api/sensors")
@require_auth
def api_sensors():
    aqara = aqara_data(use_cache=True)
    soil  = get_soil_data(use_cache=True)
    return jsonify({"aqara": aqara, "soil": soil, "ts": datetime.datetime.now().isoformat()})


@app.route("/api/valves")
@require_auth
def api_valves():
    return jsonify({"valves": get_all_valve_status(), "ts": datetime.datetime.now().isoformat()})


@app.route("/api/valve/<name>/<action>", methods=["POST"])
@require_auth
def api_valve_action(name, action):
    if name not in ("outdoor", "greenhouse"):
        abort(400, "Unknown valve")
    if action == "open":
        minutes = int(request.args.get("minutes", 10))
        label  = f"💧 Open {name} valve for {minutes} min"
        params = {"valve": name, "minutes": minutes}
    elif action == "close":
        label  = f"🔒 Close {name} valve"
        params = {"valve": name}
    else:
        abort(400, "Unknown action")
    token = _write_pending(f"valve_{action}", label, params)
    return jsonify({"status": "pending", "token": token})


# ── Login page ────────────────────────────────────────────────────────────────

LOGIN_HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KoloDash — Login</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,opsz,wght@0,9..144,500&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#162414;display:flex;align-items:center;justify-content:center;min-height:100vh;font-family:'DM Mono',monospace;color:#dce8da}
.card{background:#1f301d;border:1px solid rgba(90,158,82,0.18);border-radius:16px;padding:36px;width:300px;text-align:center}
h1{font-family:'Fraunces',serif;color:#8dd484;font-size:1.4rem;font-weight:500;margin-bottom:6px}
p{color:#607860;font-size:0.8rem;margin-bottom:24px}
input{width:100%;padding:10px 12px;background:#0c1a0b;border:1px solid rgba(90,158,82,0.32);border-radius:8px;color:#dce8da;font-family:'DM Mono',monospace;font-size:0.9rem;margin-bottom:12px;outline:none}
input:focus{border-color:rgba(141,212,132,0.5)}
button{width:100%;padding:10px;background:#3a6b35;color:#b8edb2;border:none;border-radius:8px;font-family:'DM Mono',monospace;font-size:0.9rem;cursor:pointer}
button:hover{background:#5a9e52}
.err{color:#e87878;font-size:0.8rem;margin-top:8px}
</style></head>
<body>
<div class="card">
  <h1>KoloDash</h1>
  <p>Kolonihave · Hedehusene</p>
  <form method="POST" action="/login">
    <input type="password" name="token" placeholder="Access token" autofocus>
    {% if error %}<div class="err">Wrong token</div>{% endif %}
    <button type="submit">Enter</button>
  </form>
</div>
</body></html>"""


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        resp = Response("", 302, {"Location": "/"})
        return resp
    return render_template_string(LOGIN_HTML, error=False)


@app.route("/logout")
def logout():
    resp = Response("", 302, {"Location": "/login"})
    resp.delete_cookie("kolo_token")
    return resp


# ── Main dashboard HTML ───────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KoloDash — Hedehusene</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;0,9..144,600;1,9..144,300&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --ink:#0c1a0b;--ink2:#162414;--ink3:#1f301d;--ink4:#263d23;
  --leaf:#3a6b35;--leaf2:#5a9e52;--leaf3:#8dd484;--leaf4:#b8edb2;
  --sky:#162a38;--water:#2a6e8e;--water2:#4aa8cc;--water3:#8dd0e8;
  --amber:#b87a08;--amber2:#e8a020;--amber3:#f5c860;
  --red:#7a1818;--red2:#cc3c3c;--red3:#e87878;
  --fog:#dce8da;--fog2:#b8ccb5;--fog3:#8ca889;--fog4:#607860;
  --border:rgba(90,158,82,0.18);--border2:rgba(90,158,82,0.32);
  font-family:'DM Mono',monospace;color:var(--fog)
}
html,body{background:var(--ink2);min-height:100vh}

.topbar{background:var(--ink);border-bottom:1px solid var(--border);padding:0 1.5rem;height:48px;display:flex;align-items:center;gap:16px;position:sticky;top:0;z-index:10}
.logo{font-family:'Fraunces',serif;font-size:16px;font-weight:500;color:var(--leaf3);letter-spacing:0.02em}
.logo span{color:var(--fog3);font-weight:300}
.topbar-meta{font-size:10px;color:var(--fog4);margin-left:auto;display:flex;align-items:center;gap:14px}
.status-pill{display:flex;align-items:center;gap:5px;font-size:10px;padding:3px 9px;border-radius:20px;border:1px solid}
.pill-red{border-color:rgba(204,60,60,0.35);color:var(--red3);background:rgba(204,60,60,0.08)}
.pill-green{border-color:rgba(141,212,132,0.3);color:var(--leaf3);background:rgba(141,212,132,0.07)}
.pill-amber{border-color:rgba(232,160,32,0.35);color:var(--amber3);background:rgba(232,160,32,0.08)}
.pill-dot{width:5px;height:5px;border-radius:50%}
.pill-dot-red{background:var(--red2)}.pill-dot-green{background:var(--leaf3)}.pill-dot-amber{background:var(--amber2)}

.grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(0,1fr) minmax(0,1fr);gap:12px;padding:1.25rem;max-width:1400px;margin:0 auto}
.card{background:var(--ink3);border:1px solid var(--border);border-radius:16px;padding:1.1rem;display:flex;flex-direction:column}
.card-wide{grid-column:span 2}.card-full{grid-column:span 3}
.card-side{display:flex;flex-direction:column;gap:12px}

.ch{display:flex;align-items:center;gap:7px;padding-bottom:10px;margin-bottom:10px;border-bottom:1px solid var(--border)}
.ch-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.ch-dot-red{background:var(--red2)}.ch-dot-green{background:var(--leaf3)}
.ch-dot-amber{background:var(--amber2)}.ch-dot-blue{background:var(--water2)}
.ch-title{font-family:'Fraunces',serif;font-size:10px;font-weight:300;letter-spacing:0.12em;text-transform:uppercase;color:var(--fog3)}
.ch-right{margin-left:auto;display:flex;align-items:center;gap:8px}

.mower-top{display:flex;align-items:center;gap:14px;margin-bottom:14px}
.mower-icon{width:48px;height:48px;border-radius:12px;background:var(--red);border:1px solid rgba(204,60,60,0.4);display:flex;align-items:center;justify-content:center;flex-shrink:0;position:relative}
.mower-icon::after{content:'';position:absolute;inset:-3px;border-radius:15px;border:1px solid rgba(204,60,60,0.25);animation:mring 2.5s ease-in-out infinite}
@keyframes mring{0%,100%{opacity:0.5;transform:scale(1)}50%{opacity:0;transform:scale(1.08)}}
.mower-state{font-family:'Fraunces',serif;font-size:22px;font-weight:500;color:var(--red3);line-height:1}
.mower-sub{font-size:10px;color:var(--fog4);margin-top:3px}
.prog-wrap{margin-bottom:12px}
.prog-label{display:flex;justify-content:space-between;font-size:10px;color:var(--fog4);margin-bottom:5px}
.prog-track{height:3px;background:var(--ink);border-radius:2px;overflow:hidden}
.prog-fill{height:100%;background:var(--leaf2);border-radius:2px;transition:width 0.4s}
.mower-stats{display:grid;grid-template-columns:1fr 1fr 1fr;gap:7px;margin-bottom:14px}
.stat-box{background:var(--ink);border-radius:9px;padding:8px 10px}
.stat-label{font-size:9px;color:var(--fog4);letter-spacing:0.08em;text-transform:uppercase;margin-bottom:3px}
.stat-val{font-size:13px;color:var(--fog2)}

.cmd-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.cmd-btn{border:none;border-radius:10px;padding:10px 6px;font-family:'DM Mono',monospace;font-size:11px;cursor:pointer;display:flex;flex-direction:column;align-items:center;gap:5px;transition:opacity 0.15s;letter-spacing:0.02em}
.cmd-btn:hover{opacity:0.82}.cmd-btn:active{transform:scale(0.96)}.cmd-btn:disabled{opacity:0.4;cursor:not-allowed}
.cmd-icon{width:22px;height:22px}
.btn-start{background:var(--leaf);color:var(--leaf4)}
.btn-pause{background:var(--amber);color:var(--amber3)}
.btn-dock{background:var(--sky);color:var(--water3);border:1px solid var(--water)}
.btn-snap{background:var(--ink4);color:var(--fog3);border:1px solid var(--border2)}
.btn-analyse{background:var(--ink4);color:var(--fog3);border:1px solid var(--border2)}
.btn-moisture{background:rgba(42,110,142,0.18);color:var(--water2);border:1px solid rgba(74,168,204,0.3)}
.analysis-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:12px}

.wx-temp{font-family:'Fraunces',serif;font-size:36px;font-weight:300;color:var(--fog);line-height:1;margin-bottom:2px}
.wx-desc{font-size:11px;color:var(--fog4);margin-bottom:12px}
.wx-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-bottom:2px}
.wx-chip{background:var(--ink);border-radius:9px;padding:8px 10px}
.wx-chip-l{font-size:9px;color:var(--fog4);letter-spacing:0.08em;text-transform:uppercase;margin-bottom:3px}
.wx-chip-v{font-size:13px;color:var(--fog2)}

.dec-header{display:flex;align-items:center;gap:9px;margin-bottom:9px}
.dec-x{width:30px;height:30px;border-radius:50%;background:var(--red);border:1px solid rgba(204,60,60,0.4);display:flex;align-items:center;justify-content:center;font-size:14px;color:var(--red3);flex-shrink:0}
.dec-check{width:30px;height:30px;border-radius:50%;background:var(--leaf);border:1px solid rgba(90,158,82,0.4);display:flex;align-items:center;justify-content:center;font-size:14px;color:var(--leaf3);flex-shrink:0}
.dec-title{font-family:'Fraunces',serif;font-size:17px}
.dec-reason{font-size:11px;color:var(--fog4);line-height:1.6}

.cam-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin-bottom:8px}
.cam-item{border-radius:10px;overflow:hidden;background:var(--ink);position:relative}
.cam-item img{width:100%;display:block}
.cam-label{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,0.6);color:var(--fog3);font-size:10px;padding:4px 8px}
.cam-err{color:var(--fog4);font-size:11px;padding:20px;text-align:center}
.snap-area{background:var(--ink);border-radius:10px;border:1px dashed rgba(90,158,82,0.28);display:flex;flex-direction:column;align-items:center;justify-content:center;padding:24px;gap:8px;cursor:pointer;transition:border-color 0.15s,background 0.15s;min-height:80px}
.snap-area:hover{border-color:var(--border2);background:var(--ink4)}
.snap-icon-wrap{width:32px;height:32px;border-radius:8px;background:var(--ink4);display:flex;align-items:center;justify-content:center}
.snap-label{font-size:10px;color:var(--fog4);text-align:center;line-height:1.6}
.analysis-box{background:var(--ink);border-radius:10px;padding:12px;font-size:11px;color:var(--fog3);line-height:1.7;margin-top:8px;white-space:pre-wrap}

.history-row{display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:10px;color:var(--fog3)}
.history-row:last-child{border-bottom:none}
.history-date{color:var(--fog4);min-width:58px}
.history-badge{font-size:9px;padding:2px 7px;border-radius:5px;margin-left:auto;white-space:nowrap}
.badge-done{background:rgba(74,168,204,0.1);color:var(--water2);border:1px solid rgba(74,168,204,0.2)}
.badge-skip{background:rgba(204,60,60,0.1);color:var(--red3);border:1px solid rgba(204,60,60,0.2)}

.climate-row{display:flex;align-items:center;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:11px}
.climate-row:last-child{border-bottom:none}
.climate-zone{color:var(--fog3);min-width:76px}
.climate-chips{display:flex;gap:5px;flex-wrap:wrap}
.chip{background:var(--ink);padding:2px 7px;border-radius:5px;font-size:10px;color:var(--fog3)}
.chip-warm{background:rgba(232,160,32,0.1);color:var(--amber2);border:1px solid rgba(232,160,32,0.18)}
.chip-cool{background:rgba(74,168,204,0.1);color:var(--water2);border:1px solid rgba(74,168,204,0.18)}
.chip-humid{background:rgba(90,158,82,0.1);color:var(--leaf3);border:1px solid rgba(90,158,82,0.18)}

.zone-lbl{font-size:9px;letter-spacing:0.1em;text-transform:uppercase;color:var(--fog4);margin-bottom:6px}
.valve-row{display:flex;align-items:center;gap:7px;background:var(--ink);border-radius:10px;padding:8px 10px;margin-bottom:10px;flex-wrap:wrap;row-gap:5px}
.v-dot{width:6px;height:6px;border-radius:50%;background:var(--red2);flex-shrink:0}
.v-name{font-size:11px;color:var(--fog2);flex:1;min-width:70px}
.v-badge{font-size:9px;padding:2px 7px;border-radius:6px;background:rgba(204,60,60,0.1);color:var(--red3);border:1px solid rgba(204,60,60,0.2);white-space:nowrap}
.v-badge-open{background:rgba(90,158,82,0.12);color:var(--leaf3);border:1px solid rgba(90,158,82,0.2)}
select{background:var(--ink2);border:1px solid var(--border2);border-radius:6px;color:var(--fog2);font-family:'DM Mono',monospace;font-size:10px;padding:3px 6px;cursor:pointer;outline:none}
.btn-o{background:var(--leaf);border:none;border-radius:6px;color:var(--leaf4);font-size:10px;padding:4px 10px;cursor:pointer;font-family:'DM Mono',monospace;white-space:nowrap;transition:background 0.15s}
.btn-o:hover{background:var(--leaf2)}
.btn-c{background:transparent;border:1px solid rgba(204,60,60,0.4);border-radius:6px;color:var(--red3);font-size:10px;padding:4px 10px;cursor:pointer;font-family:'DM Mono',monospace;white-space:nowrap}
.divider{height:1px;background:var(--border);margin:3px 0 9px}

.sensor-row{display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04)}
.sensor-row:last-child{border-bottom:none}
.s-name{font-size:11px;color:var(--fog2);min-width:90px}
.s-name-dim{color:var(--fog4)}
.bar-t{flex:1;height:3px;background:var(--ink);border-radius:2px;overflow:hidden}
.bar-f{height:100%;border-radius:2px;transition:width 0.4s}
.b-blue{background:var(--water2)}.b-green{background:var(--leaf2)}.b-amber{background:var(--amber2)}.b-red{background:var(--red2)}
.s-pct{font-size:11px;width:32px;text-align:right}
.s-temp{font-size:10px;color:var(--fog4);width:40px;text-align:right}
.s-batt{font-size:9px;color:var(--fog4);width:24px;text-align:right}
.sub-div{display:flex;align-items:center;gap:5px;font-size:9px;color:var(--fog4);letter-spacing:0.08em;text-transform:uppercase;margin:6px 0 4px}
.sub-div::before,.sub-div::after{content:'';flex:1;height:1px;background:var(--border)}
.avg-pill{font-size:10px;padding:2px 8px;border-radius:10px}
.avg-good{background:rgba(90,158,82,0.1);color:var(--leaf3);border:1px solid rgba(90,158,82,0.2)}
.avg-warn{background:rgba(232,160,32,0.1);color:var(--amber2);border:1px solid rgba(232,160,32,0.2)}

.log-box{background:var(--ink);color:var(--leaf3);font-family:'DM Mono',monospace;font-size:10px;padding:12px;border-radius:10px;max-height:180px;overflow-y:auto;line-height:1.6;white-space:pre-wrap}
.a-box{background:var(--ink);border-radius:10px;padding:10px 11px}
.a-label{font-size:9px;color:var(--fog4);letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px}

#toast{position:fixed;bottom:24px;right:24px;background:var(--ink4);color:var(--fog);padding:10px 18px;border-radius:10px;font-size:11px;border:1px solid var(--border2);opacity:0;transition:opacity 0.3s;pointer-events:none;z-index:999}
#toast.show{opacity:1}
@keyframes spin{to{transform:rotate(360deg)}}
.spin{display:inline-block;animation:spin 1s linear infinite}

/* CLIMATE SUB-CARDS */
.climate-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:2px}
.climate-card{border-radius:10px;padding:10px 12px;border:1px solid var(--border)}
.climate-card-label{font-size:9px;color:var(--fog4);letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px}
.climate-card-temp{font-family:'Fraunces',serif;font-size:22px;font-weight:300;line-height:1;margin-bottom:6px}
.climate-card-row{font-size:10px;color:var(--fog3);margin-top:3px}
.cc-indoor{background:rgba(184,122,8,0.1);border-color:rgba(184,122,8,0.18)}
.cc-outdoor{background:rgba(42,110,142,0.1);border-color:rgba(42,110,142,0.2)}
.cc-greenhouse{background:rgba(58,107,53,0.1);border-color:rgba(58,107,53,0.2)}

@media(max-width:960px){.grid{grid-template-columns:1fr 1fr}.card-wide,.card-full{grid-column:span 2}}
@media(max-width:600px){
  .grid{grid-template-columns:1fr;padding:0.75rem;gap:8px}
  .card-wide,.card-full{grid-column:span 1}
  .card-side{gap:8px}
  .topbar{padding:0 1rem;gap:10px}
  .topbar-meta span:first-child{display:none}
  .cmd-grid{gap:6px}
  .cmd-btn{padding:8px 4px;font-size:10px}
  .mower-stats{grid-template-columns:1fr 1fr}
  .wx-grid{grid-template-columns:1fr 1fr}
  .wx-temp{font-size:28px}
  .mower-state{font-size:18px}
  .analysis-grid{grid-template-columns:1fr}
  .cam-grid{grid-template-columns:1fr}
  .valve-row{flex-wrap:wrap}
  .ch-title{font-size:9px}
  .zones-grid{grid-template-columns:1fr !important}
  .climate-cards{grid-template-columns:1fr !important}
}
</style>
</head>
<body>

<div class="topbar">
  <div class="logo">Kolo<span>Dash</span></div>
  <div class="status-pill pill-red" id="mower-pill">
    <div class="pill-dot pill-dot-red" id="mower-pill-dot"></div>
    <span id="mower-pill-txt">Mower —</span>
  </div>
  <div class="status-pill pill-green">
    <div class="pill-dot pill-dot-green"></div>Sensors live
  </div>
  <div class="topbar-meta">
    <span>Hedehusene · AH Bakkevej</span>
    <span id="last-update">—</span>
    <a href="/logout" style="color:var(--fog4);text-decoration:none">logout</a>
  </div>
</div>

<div class="grid">

  <!-- MOWER (2 cols) -->
  <div class="card card-wide">
    <div class="ch">
      <div class="ch-dot" id="mower-ch-dot" style="background:var(--red2)"></div>
      <span class="ch-title">Mower</span>
      <div class="ch-right">
        <span style="font-size:10px;color:var(--fog4)" id="mow-last">Last mow: —</span>
      </div>
    </div>
    <div class="mower-top">
      <div class="mower-icon">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
          <rect x="3" y="13" width="18" height="6" rx="2" stroke="#cc3c3c" stroke-width="1.5"/>
          <circle cx="7" cy="19" r="2" fill="#cc3c3c"/><circle cx="17" cy="19" r="2" fill="#cc3c3c"/>
          <path d="M7 13V9a1 1 0 011-1h4l3 5" stroke="#cc3c3c" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
      </div>
      <div>
        <div class="mower-state" id="mower-state">—</div>
        <div class="mower-sub" id="mower-sub">Loading…</div>
      </div>
    </div>
    <div class="prog-wrap">
      <div class="prog-label"><span>Coverage today</span><span id="mow-pct-lbl">—</span></div>
      <div class="prog-track"><div class="prog-fill" id="mow-fill" style="width:0%"></div></div>
    </div>
    <div class="mower-stats">
      <div class="stat-box"><div class="stat-label">Zone</div><div class="stat-val">Hedehusene</div></div>
      <div class="stat-box"><div class="stat-label">Total runtime</div><div class="stat-val" id="mow-runtime">—</div></div>
      <div class="stat-box"><div class="stat-label">Consec 100%</div><div class="stat-val" id="mow-consec">—</div></div>
    </div>
    <div class="cmd-grid">
      <button class="cmd-btn btn-start" onclick="mowerAction('start')">
        <svg class="cmd-icon" viewBox="0 0 22 22" fill="none"><polygon points="5,3 19,11 5,19" fill="#8dd484"/></svg>
        Start
      </button>
      <button class="cmd-btn btn-pause" onclick="mowerAction('pause')">
        <svg class="cmd-icon" viewBox="0 0 22 22" fill="none"><rect x="4" y="3" width="5" height="16" rx="2" fill="#f5c860"/><rect x="13" y="3" width="5" height="16" rx="2" fill="#f5c860"/></svg>
        Pause
      </button>
      <button class="cmd-btn btn-dock" onclick="mowerAction('dock')">
        <svg class="cmd-icon" viewBox="0 0 22 22" fill="none"><path d="M11 3v10M7 9l4 4 4-4" stroke="#8dd0e8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M3 17h16" stroke="#8dd0e8" stroke-width="1.8" stroke-linecap="round"/></svg>
        Dock
      </button>
    </div>
  </div>

  <!-- WEATHER + DECISION stacked in 3rd col -->
  <div class="card-side">
    <div class="card" style="flex:1">
      <div class="ch">
        <div class="ch-dot ch-dot-green"></div>
        <span class="ch-title">Weather · Hedehusene</span>
      </div>
      <div class="wx-temp" id="wx-temp">—</div>
      <div class="wx-desc" id="wx-desc">—</div>
      <div class="wx-grid">
        <div class="wx-chip"><div class="wx-chip-l">Humidity</div><div class="wx-chip-v" id="wx-hum">—</div></div>
        <div class="wx-chip"><div class="wx-chip-l">Wind</div><div class="wx-chip-v" id="wx-wind">—</div></div>
        <div class="wx-chip"><div class="wx-chip-l">Rain today</div><div class="wx-chip-v" id="wx-rain">—</div></div>
        <div class="wx-chip"><div class="wx-chip-l">Yesterday</div><div class="wx-chip-v" id="wx-yrain">—</div></div>
        <div class="wx-chip"><div class="wx-chip-l">UV index</div><div class="wx-chip-v" id="wx-uv">—</div></div>
        <div class="wx-chip"><div class="wx-chip-l">Solar W/m²</div><div class="wx-chip-v" id="wx-solar">—</div></div>
        <div class="wx-chip"><div class="wx-chip-l">Cloud cover</div><div class="wx-chip-v" id="wx-cloud">—</div></div>
        <div class="wx-chip"><div class="wx-chip-l">Dew point</div><div class="wx-chip-v" id="wx-dew">—</div></div>
      </div>
    </div>
    <div class="card" style="flex:1">
      <div class="ch">
        <div class="ch-dot" id="dec-dot" style="background:var(--red2)"></div>
        <span class="ch-title">Mow decision</span>
      </div>
      <div class="dec-header">
        <div id="dec-icon" class="dec-x">✕</div>
        <div class="dec-title" id="dec-title" style="color:var(--red3)">—</div>
      </div>
      <div class="dec-reason" id="dec-reason">Loading…</div>
    </div>
  </div>

  <!-- ANALYSIS (full width) -->
  <div class="card card-full">
    <div class="ch">
      <div class="ch-dot ch-dot-green"></div>
      <span class="ch-title">Analysis</span>
      <div class="ch-right">
        <span style="font-size:10px;color:var(--fog4)" id="snap-ts"></span>
      </div>
    </div>
    <div class="analysis-grid">
      <button class="cmd-btn btn-snap" onclick="loadSnaps()" style="width:100%">
        <svg class="cmd-icon" viewBox="0 0 22 22" fill="none"><rect x="2" y="5" width="18" height="13" rx="2.5" stroke="#8ca889" stroke-width="1.5"/><circle cx="11" cy="11.5" r="3.5" stroke="#8ca889" stroke-width="1.5"/><path d="M8 5l1.5-2.5h3L14 5" stroke="#8ca889" stroke-width="1.5" stroke-linejoin="round"/></svg>
        Snap
      </button>
      <button class="cmd-btn btn-analyse" onclick="runAnalyse()" style="width:100%">
        <svg class="cmd-icon" viewBox="0 0 22 22" fill="none"><circle cx="10" cy="10" r="6" stroke="#8ca889" stroke-width="1.5"/><path d="M15 15l4 4" stroke="#8ca889" stroke-width="1.8" stroke-linecap="round"/><path d="M7 10h6M10 7v6" stroke="#8ca889" stroke-width="1.3" stroke-linecap="round"/></svg>
        Grass AI
      </button>
      <button class="cmd-btn btn-moisture" onclick="moistureStatus()" style="width:100%">
        <svg class="cmd-icon" viewBox="0 0 22 22" fill="none"><path d="M11 3C11 3 5 10 5 14a6 6 0 0012 0c0-4-6-11-6-11z" stroke="#4aa8cc" stroke-width="1.5" stroke-linejoin="round"/><path d="M8 15a3 3 0 006 0" stroke="#4aa8cc" stroke-width="1.2" stroke-linecap="round"/></svg>
        Moisture AI
      </button>
    </div>
    <div class="cam-grid" id="cam-grid" style="display:none"></div>
    <div id="analysis-card" style="display:none">
      <!-- Watering decision cards (shown when moisture AI returns JSON) -->
      <div id="water-decision" style="display:none;margin-bottom:6px">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
          <div style="background:rgba(42,110,142,0.07);border:1px solid rgba(42,110,142,0.18);border-radius:10px;padding:10px">
            <div style="font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--fog4);margin-bottom:6px">Greenhouse</div>
            <div class="dec-header" style="margin-bottom:4px">
              <div id="water-gh-icon" class="dec-x">&#x2715;</div>
              <div class="dec-title" id="water-gh-title" style="color:var(--red3)">&#x2014;</div>
            </div>
            <div class="dec-reason" id="water-gh-reason" style="color:var(--fog2)">&#x2014;</div>
          </div>
          <div style="background:rgba(42,110,142,0.07);border:1px solid rgba(42,110,142,0.18);border-radius:10px;padding:10px">
            <div style="font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--fog4);margin-bottom:6px">Outdoor</div>
            <div class="dec-header" style="margin-bottom:4px">
              <div id="water-od-icon" class="dec-x">&#x2715;</div>
              <div class="dec-title" id="water-od-title" style="color:var(--red3)">&#x2014;</div>
            </div>
            <div class="dec-reason" id="water-od-reason" style="color:var(--fog2)">&#x2014;</div>
          </div>
        </div>
      </div>
      <div class="analysis-box" id="analysis-txt"></div>
    </div>
  </div>

  <!-- MOISTURE INSIGHT (Ollama commentary, updated 2×/day) -->
  <div class="card card-full">
    <div class="ch">
      <div class="ch-dot ch-dot-blue"></div>
      <span class="ch-title">Moisture insight</span>
      <div class="ch-right">
        <span style="font-size:10px;color:var(--fog4)" id="moisture-comment-ts"></span>
      </div>
    </div>
    <div id="moisture-comment-txt" style="font-size:12px;color:var(--fog2);line-height:1.8;min-height:36px">
      <span style="color:var(--fog4);font-style:italic">Awaiting morning or afternoon run…</span>
    </div>
  </div>

  <!-- CLIMATE + ZONES (full width) -->
  <div class="card card-full">
    <div class="ch">
      <div class="ch-dot ch-dot-blue"></div>
      <span class="ch-title">Climate &amp; irrigation</span>
    </div>
    <!-- Aqara climate sensors -->
    <div id="aqara-body"><div style="color:var(--fog4);font-size:11px">Loading…</div></div>
    <!-- Soil zones side by side -->
    <div class="zones-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:14px">
      <!-- Greenhouse zone -->
      <div style="background:rgba(42,110,142,0.07);border:1px solid rgba(42,110,142,0.18);border-radius:12px;padding:12px">
        <div style="display:flex;align-items:center;gap:7px;padding-bottom:8px;margin-bottom:8px;border-bottom:1px solid rgba(42,110,142,0.2)">
          <div class="ch-dot ch-dot-blue"></div>
          <span class="ch-title">Greenhouse zone</span>
          <span class="avg-pill avg-good" id="gh-avg" style="margin-left:auto;display:none"></span>
        </div>
        <div class="zone-lbl">Soil sensors</div>
        <div style="display:flex;align-items:center;gap:8px;font-size:8px;color:var(--fog4);padding:0 0 3px;letter-spacing:0.05em;text-transform:uppercase">
          <span style="min-width:90px"></span><span style="flex:1">Moisture</span>
          <span style="width:32px;text-align:right">%</span>
          <span style="width:40px;text-align:right">°C</span>
          <span style="width:24px;text-align:right">🔋</span>
        </div>
        <div id="gh-soil"><div style="color:var(--fog4);font-size:11px">Loading…</div></div>
        <div class="divider"></div>
        <div class="zone-lbl">Valve</div>
        <div id="gh-valve"><div style="color:var(--fog4);font-size:11px">Loading…</div></div>
      </div>
      <!-- Outdoor zone -->
      <div style="background:rgba(184,122,8,0.07);border:1px solid rgba(184,122,8,0.18);border-radius:12px;padding:12px">
        <div style="display:flex;align-items:center;gap:7px;padding-bottom:8px;margin-bottom:8px;border-bottom:1px solid rgba(184,122,8,0.2)">
          <div class="ch-dot ch-dot-amber"></div>
          <span class="ch-title">Outdoor zone</span>
          <span class="avg-pill avg-warn" id="out-avg" style="margin-left:auto;display:none"></span>
        </div>
        <div class="zone-lbl">Soil sensors</div>
        <div style="display:flex;align-items:center;gap:8px;font-size:8px;color:var(--fog4);padding:0 0 3px;letter-spacing:0.05em;text-transform:uppercase">
          <span style="min-width:90px"></span><span style="flex:1">Moisture</span>
          <span style="width:32px;text-align:right">%</span>
          <span style="width:40px;text-align:right">°C</span>
          <span style="width:24px;text-align:right">🔋</span>
        </div>
        <div id="out-soil"><div style="color:var(--fog4);font-size:11px">Loading…</div></div>
        <div class="divider"></div>
        <div class="zone-lbl">Valve</div>
        <div id="out-valve"><div style="color:var(--fog4);font-size:11px">Loading…</div></div>
      </div>
    </div>
  </div>

  <!-- HISTORY + LOG -->
  <div class="card card-full">
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
      <div>
        <div class="ch" style="margin-bottom:8px">
          <div class="ch-dot ch-dot-green"></div>
          <span class="ch-title">Recent mow sessions</span>
        </div>
        <div id="hist-body"><div style="color:var(--fog4);font-size:11px">Loading…</div></div>
      </div>
      <div>
        <div class="ch" style="margin-bottom:8px">
          <div class="ch-dot ch-dot-blue"></div>
          <span class="ch-title">Agent log</span>
        </div>
        <div class="log-box" id="log-box">Loading…</div>
      </div>
    </div>
  </div>

</div>

<div id="toast"></div>

<script>
/*__INIT_DATA__*/
const H = {"Content-Type": "application/json"};
const Q = "";

function toast(msg, ok=true) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.style.borderColor = ok ? "rgba(141,212,132,0.3)" : "rgba(204,60,60,0.35)";
  t.style.color = ok ? "var(--leaf3)" : "var(--red3)";
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 3500);
}

function updateClock() {
  const now = new Date();
  const el = document.getElementById("last-update");
  if (el) el.textContent = now.toLocaleDateString("en-GB",{day:"2-digit",month:"short"}) + " · " + now.toLocaleTimeString("en-GB",{hour:"2-digit",minute:"2-digit"});
}
updateClock();
setInterval(updateClock, 30000);

function checkAuth(r) {
  // Do not redirect on 401 — proxy may strip auth headers on API calls.
  // Embedded __INIT__ data keeps the page populated; user stays logged in.
  return r.ok;
}

function renderStatus(d) {
    const ms  = d.mower.state || "Unknown";
    const pct = d.mower.mowed_pct ?? 0;
    const isMowing   = ms === "Mowing";
    const isCharging = ms === "Charging";
    const isDocked   = ms === "Docked";
    const stateColor = isMowing ? "var(--leaf3)" : isCharging ? "var(--amber3)" : "var(--red3)";
    const dotColor   = isMowing ? "var(--leaf3)" : isCharging ? "var(--amber2)" : "var(--red2)";
    document.getElementById("mower-state").textContent = ms;
    document.getElementById("mower-state").style.color = stateColor;
    document.getElementById("mower-ch-dot").style.background = dotColor;
    document.getElementById("mower-sub").textContent =
      (isDocked ? "Docked" : ms) + " · Consecutive 100%: " + (d.mower.consecutive_100pct || 0) + " / 2";
    document.getElementById("mow-pct-lbl").textContent = pct + "%";
    document.getElementById("mow-fill").style.width = pct + "%";
    document.getElementById("mow-consec").textContent = (d.mower.consecutive_100pct || 0) + " / 2";
    document.getElementById("mow-runtime").textContent = (d.mower.runtime_total_h || 0) + " h";
    document.getElementById("mow-last").textContent = d.mower.last_mow
      ? "Last mow: " + d.mower.last_mow + " · " + d.mower.days_ago + "d ago"
      : "Last mow: —";
    const pill    = document.getElementById("mower-pill");
    const pillDot = document.getElementById("mower-pill-dot");
    pill.className = isMowing ? "status-pill pill-green" : isCharging ? "status-pill pill-amber" : "status-pill pill-red";
    pillDot.className = isMowing ? "pill-dot pill-dot-green" : isCharging ? "pill-dot pill-dot-amber" : "pill-dot pill-dot-red";
    document.getElementById("mower-pill-txt").textContent = "Mower " + ms;
    if (d.weather) {
      const w = d.weather;
      document.getElementById("wx-temp").textContent  = w.temp_c + "°C";
      document.getElementById("wx-desc").textContent  = w.desc;
      document.getElementById("wx-hum").textContent   = w.humidity + "%";
      document.getElementById("wx-wind").textContent  = w.wind_kmh + " km/h";
      document.getElementById("wx-rain").textContent  = w.rain_mm_today + " mm";
      document.getElementById("wx-yrain").textContent = (d.yesterday_rain_mm ?? 0) + " mm";
      document.getElementById("wx-uv").textContent    = w.uv_index ?? "—";
      document.getElementById("wx-solar").textContent = w.solar_rad_wm2 != null ? w.solar_rad_wm2 + " W/m²" : "—";
      document.getElementById("wx-cloud").textContent = w.cloud_cover != null ? w.cloud_cover + "%" : "—";
      document.getElementById("wx-dew").textContent   = w.dew_point_c != null ? w.dew_point_c + "°C" : "—";
    }
    const dec = d.mow_decision;
    const decIcon  = document.getElementById("dec-icon");
    const decTitle = document.getElementById("dec-title");
    const decDot   = document.getElementById("dec-dot");
    if (dec.will_mow) {
      decIcon.className = "dec-check"; decIcon.textContent = "✓";
      decTitle.textContent = "Mow today"; decTitle.style.color = "var(--leaf3)";
      decDot.style.background = "var(--leaf3)";
    } else {
      decIcon.className = "dec-x"; decIcon.textContent = "✕";
      decTitle.textContent = "Skip today"; decTitle.style.color = "var(--red3)";
      decDot.style.background = "var(--red2)";
    }
    document.getElementById("dec-reason").textContent = dec.reason;
    const histEl = document.getElementById("hist-body");
    if (d.history && d.history.length) {
      histEl.innerHTML = d.history.slice().reverse().map(h => {
        const done = (h.mowed_pct ?? 0) >= 95;
        return `<div class="history-row">
          <span class="history-date">${h.date || "—"}</span>
          <span>${h.mowed_pct ?? "?"}%</span>
          <span class="history-badge ${done ? "badge-done" : "badge-skip"}">${done ? "done" : "partial"}</span>
        </div>`;
      }).join("");
    } else {
      histEl.innerHTML = '<div style="color:var(--fog4);font-size:11px;padding:6px 0">No history yet</div>';
    }
}

async function loadStatus() {
  try {
    const r = await fetch("/api/status" + Q, {headers: H});
    if (!checkAuth(r)) return;
    renderStatus(await r.json());
  } catch(e) { console.error("status:", e); }
}

function renderLog(d) {
  document.getElementById("log-box").textContent = (d.lines || []).join("\\n");
}

async function loadLog() {
  try {
    const r = await fetch("/api/log" + Q, {headers: H});
    if (!checkAuth(r)) return;
    renderLog(await r.json());
  } catch(e) {}
}

function pollAction(token, onDone) {
  let attempts = 0;
  toast("Awaiting Telegram confirmation…");
  const iv = setInterval(async () => {
    attempts++;
    if (attempts > 70) { clearInterval(iv); toast("No response — action cancelled", false); return; }
    try {
      const r = await fetch(`/api/action/status/${token}` + Q, {headers: H});
      const d = await r.json();
      if (d.status === "done") {
        clearInterval(iv);
        toast((d.result || "Done").split("\\n")[0], true);
        if (onDone) onDone(d);
      } else if (d.status === "denied") {
        clearInterval(iv); toast("Denied via Telegram", false); if (onDone) onDone(null);
      } else if (d.status === "expired") {
        clearInterval(iv); toast("Confirmation timed out", false); if (onDone) onDone(null);
      } else if (d.status === "error") {
        clearInterval(iv); toast(d.result || "Error", false); if (onDone) onDone(null);
      }
    } catch(e) {}
  }, 2000);
}

function renderSnaps(snap_data, grid) {
  if (!snap_data || !snap_data.length) {
    grid.innerHTML = '<div class="cam-err">No snapshots returned</div>'; return;
  }
  grid.innerHTML = snap_data.map(s => s.ok && s.data
    ? `<div class="cam-item"><img src="data:image/jpeg;base64,${s.data}" loading="lazy"><div class="cam-label">${s.name}</div></div>`
    : `<div class="cam-item"><div class="cam-err">${s.name}<br>unavailable</div></div>`
  ).join("");
  document.getElementById("snap-ts").textContent = new Date().toLocaleTimeString("da-DK");
}

async function loadSnaps() {
  const grid = document.getElementById("cam-grid");
  grid.style.display = "grid";
  grid.innerHTML = `<div class="snap-area" style="grid-column:1/-1;pointer-events:none">
    <div class="snap-label"><span class="spin">◌</span> Waiting for Telegram confirmation…</div></div>`;
  try {
    const r = await fetch("/api/snap" + Q, {method:"POST", headers: H});
    const d = await r.json();
    if (d.status === "pending") {
      pollAction(d.token, (res) => {
        if (res && res.snap_data) renderSnaps(res.snap_data, grid);
        else { grid.innerHTML = '<div class="cam-err">Cancelled or no data</div>'; }
      });
    }
  } catch(e) { grid.innerHTML = `<div class="cam-err">Error: ${e}</div>`; }
}

async function runAnalyse() {
  const card = document.getElementById("analysis-card");
  const txt  = document.getElementById("analysis-txt");
  const grid = document.getElementById("cam-grid");
  card.style.display = "block";
  grid.style.display = "grid";
  txt.textContent = "Waiting for Telegram confirmation…";
  grid.innerHTML = `<div class="snap-area" style="grid-column:1/-1;pointer-events:none">
    <div class="snap-label"><span class="spin">◌</span> Capturing…</div></div>`;
  try {
    const r = await fetch("/api/analyse" + Q, {method:"POST", headers: H});
    const d = await r.json();
    if (d.status === "pending") {
      pollAction(d.token, (res) => {
        if (res) {
          txt.textContent = res.result || "Analysis complete";
          if (res.snap_data) renderSnaps(res.snap_data, grid);
        } else { txt.textContent = "Cancelled"; grid.innerHTML = ""; grid.style.display = "none"; }
      });
    }
  } catch(e) { txt.textContent = "Error: " + e; }
}

async function moistureStatus() {
  const card = document.getElementById("analysis-card");
  const txt  = document.getElementById("analysis-txt");
  const grid = document.getElementById("cam-grid");
  const wdec = document.getElementById("water-decision");
  card.style.display = "block";
  grid.style.display = "none";
  wdec.style.display = "none";
  txt.style.display = "block";
  txt.textContent = "\\u23f3 Analysing soil moisture\\u2026";
  try {
    const r = await fetch("/api/moisture-analysis" + Q, {method:"POST", headers: H});
    const d = await r.json();
    const res = d.result;
    if (res && typeof res === "object" && (res.greenhouse || res.outdoor)) {
      txt.style.display = "none";
      wdec.style.display = "block";
      function applyZone(zone, iconId, titleId, reasonId) {
        const icon   = document.getElementById(iconId);
        const title  = document.getElementById(titleId);
        const reason = document.getElementById(reasonId);
        if (!zone) { title.textContent = "No data"; reason.textContent = ""; return; }
        if (zone.water_now) {
          icon.className = "dec-check"; icon.textContent = "\\u2713";
          title.textContent = "Water " + (zone.duration_min || 10) + "min";
          title.style.color = "var(--water2)";
        } else {
          icon.className = "dec-x"; icon.textContent = "\\u2715";
          const days = (zone.reason || "").match(/~?(\\d+)\\s*d/i);
          title.textContent = days ? "Skip (~" + days[1] + "d)" : "Skip";
          title.style.color = "var(--leaf3)";
        }
        reason.textContent = zone.reason || "";
      }
      applyZone(res.greenhouse, "water-gh-icon", "water-gh-title", "water-gh-reason");
      applyZone(res.outdoor,    "water-od-icon", "water-od-title", "water-od-reason");
    } else {
      txt.style.display = "block";
      const resStr = typeof res === "string" ? res : JSON.stringify(res);
      if (resStr.startsWith("Gemini error 503")) {
        txt.textContent = "Gemini is busy \\u2014 please try again in a minute.";
      } else if (resStr.startsWith("Gemini error")) {
        txt.textContent = "AI unavailable: " + resStr.split(":")[1]?.trim().split("\\\\n")[0];
      } else {
        txt.textContent = resStr;
      }
    }
  } catch(e) { txt.style.display = "block"; txt.textContent = "Error: " + e; }
}

async function mowerAction(action) {
  toast("Sending to Telegram for confirmation…");
  try {
    const r = await fetch(`/api/mower/${action}` + Q, {method:"POST", headers: H});
    const d = await r.json();
    if (d.status === "pending") pollAction(d.token, () => setTimeout(loadStatus, 2000));
  } catch(e) { toast("Error: " + e, false); }
}

function renderValve(k, v, containerId) {
  const open = v.open;
  const bat  = v.battery != null ? `<span style="font-size:9px;color:var(--fog4);margin-left:4px">🔋${v.battery}%</span>` : "";
  const cd   = v.countdown > 0 ? " · closes in " + Math.ceil(v.countdown/60) + "min" : "";
  const dotStyle = open ? "style='background:var(--leaf3)'" : "";
  const badge    = open
    ? `<span class="v-badge v-badge-open">open${cd}</span>`
    : `<span class="v-badge">closed</span>`;
  document.getElementById(containerId).innerHTML =
    `<div class="valve-row">
      <div class="v-dot" ${dotStyle}></div>
      <span class="v-name">${v.label || k}${bat}</span>
      ${badge}
      <select id="dur-${k}">
        <option value="5">5 min</option><option value="10" selected>10 min</option>
        <option value="15">15 min</option><option value="20">20 min</option><option value="30">30 min</option>
      </select>
      <button class="btn-o" onclick="valveAction('${k}','open')">Open</button>
      <button class="btn-c" onclick="valveAction('${k}','close')">Close</button>
    </div>`;
}

function renderSoilRows(keys, soil) {
  return keys.map(k => {
    const s = soil[k] || {};
    if (!s.label && !k) return "";
    const lbl = s.label || k;
    const m = s.moisture ?? "?";
    const inactive = !s.active;
    const barW = inactive ? 0 : Math.min(Number(m) || 0, 100);
    const barClass = inactive ? "b-amber" : (m < 15 ? "b-red" : m < 30 ? "b-amber" : m < 60 ? "b-green" : "b-blue");
    const pctColor = inactive ? "var(--fog4)" : (m < 15 ? "var(--red3)" : m < 30 ? "var(--amber2)" : m < 60 ? "var(--leaf3)" : "var(--water2)");
    const nameClass = inactive ? "s-name s-name-dim" : "s-name";
    const tempStr = s.temperature != null ? `<span class="s-temp">${s.temperature}°C</span>` : '<span class="s-temp"></span>';
    const battStr = s.battery != null ? `<span class="s-batt">${s.battery}%🔋</span>` : '<span class="s-batt"></span>';
    return `<div class="sensor-row">
      <span class="${nameClass}">${lbl}</span>
      <div class="bar-t"><div class="bar-f ${barClass}" style="width:${barW}%"></div></div>
      <span class="s-pct" style="color:${pctColor}">${m}%</span>
      ${tempStr}${battStr}
    </div>`;
  }).join("");
}

function renderSensors(d) {
    const aqara = d.aqara || {};
    const soil  = d.soil  || {};
    const zones = [
      {k:"kolonihavehus", lbl:"Indoor",     cls:"cc-indoor",     tempColor:"var(--amber3)"},
      {k:"outdoor",       lbl:"Outdoor",    cls:"cc-outdoor",    tempColor:"var(--water3)"},
      {k:"greenhouse",    lbl:"Greenhouse", cls:"cc-greenhouse", tempColor:"var(--leaf3)"},
    ];
    const cards = zones.map(({k, lbl, cls, tempColor}) => {
      const s = aqara[k] || {};
      const temp = s.temperature != null ? `<div class="climate-card-temp" style="color:${tempColor}">${s.temperature}°C</div>` : `<div class="climate-card-temp" style="color:var(--fog4)">—</div>`;
      const hum  = s.humidity    != null ? `<div class="climate-card-row">${s.humidity}% RH</div>` : "";
      const pres = s.pressure    != null ? `<div class="climate-card-row">${s.pressure} hPa</div>` : "";
      return `<div class="climate-card ${cls}">
        <div class="climate-card-label">${lbl}</div>
        ${temp}${hum}${pres}
      </div>`;
    }).join("");
    document.getElementById("aqara-body").innerHTML = `<div class="climate-cards">${cards}</div>`;
    const ghKeys  = ["greenhouse","greenhouse_basil"];
    const outKeys = ["cassa_alta","fragole","cassa_bassa_serra","cassa_bassa"];
    const ghHtml  = renderSoilRows(ghKeys, soil);
    const outHtml = renderSoilRows(["cassa_alta","fragole"], soil)
      + '<div class="sub-div">Bed 40cm</div>'
      + renderSoilRows(["cassa_bassa_serra","cassa_bassa"], soil);
    document.getElementById("gh-soil").innerHTML  = ghHtml  || '<div style="color:var(--fog4);font-size:11px">Unavailable</div>';
    document.getElementById("out-soil").innerHTML = outHtml || '<div style="color:var(--fog4);font-size:11px">Unavailable</div>';
    const avgFn = keys => {
      const vals = keys.map(k => soil[k]?.moisture).filter(v => v != null && v !== "?" && !isNaN(v));
      return vals.length ? Math.round(vals.reduce((a,b) => a + Number(b), 0) / vals.length) : null;
    };
    const ghAvg = avgFn(ghKeys), outAvg = avgFn(outKeys);
    if (ghAvg != null) {
      const el = document.getElementById("gh-avg");
      el.textContent = "soil avg " + ghAvg + "%";
      el.className = "avg-pill " + (ghAvg >= 40 ? "avg-good" : "avg-warn");
      el.style.display = "";
    }
    if (outAvg != null) {
      const el = document.getElementById("out-avg");
      el.textContent = "soil avg " + outAvg + "%";
      el.className = "avg-pill " + (outAvg >= 40 ? "avg-good" : "avg-warn");
      el.style.display = "";
    }
}

async function loadSensors() {
  try {
    const r = await fetch("/api/sensors" + Q, {headers: H});
    if (!checkAuth(r)) return;
    renderSensors(await r.json());
  } catch(e) { console.error("sensors:", e); }
}

function renderValves(d) {
    const v = d.valves || {};
    if (v.greenhouse) renderValve("greenhouse", v.greenhouse, "gh-valve");
    if (v.outdoor)    renderValve("outdoor",    v.outdoor,    "out-valve");
}

async function loadValves() {
  try {
    const r = await fetch("/api/valves" + Q, {headers: H});
    if (!checkAuth(r)) return;
    renderValves(await r.json());
  } catch(e) { console.error("valves:", e); }
}

async function valveAction(name, action) {
  const minutes = document.getElementById("dur-" + name)?.value || 10;
  const url = `/api/valve/${name}/${action}` + (action === "open" ? `?minutes=${minutes}&token=${encodeURIComponent(TOKEN)}` : Q);
  toast("Sending to Telegram for confirmation…");
  try {
    const r = await fetch(url, {method:"POST", headers: H});
    const d = await r.json();
    if (d.status === "pending") pollAction(d.token, () => setTimeout(loadValves, 2000));
  } catch(e) { toast("Error: " + e, false); }
}

// Render embedded server data immediately — works even if proxy blocks /api/*
(function() {
  var lb = document.getElementById("log-box");
  function dbg(msg) { if (lb) lb.textContent = msg; }
  if (!window.__INIT__) { dbg("ERR: __INIT__ missing"); return; }
  dbg("__INIT__ ok, rendering…");
  try { renderStatus(window.__INIT__.status); dbg("status ok"); } catch(e) { dbg("ERR renderStatus: "+e); return; }
  try { renderSensors(window.__INIT__.sensors); dbg("sensors ok"); } catch(e) { dbg("ERR renderSensors: "+e); return; }
  try { renderValves(window.__INIT__.valves); dbg("valves ok"); } catch(e) { dbg("ERR renderValves: "+e); return; }
  try { renderLog(window.__INIT__.log); } catch(e) { dbg("ERR renderLog: "+e); return; }
  dbg("init done");
})();
async function loadMoistureComment() {
  try {
    const r = await fetch("/api/moisture-comment" + Q, {headers: H});
    const d = await r.json();
    const el = document.getElementById("moisture-comment-txt");
    const ts = document.getElementById("moisture-comment-ts");
    if (d.comment) {
      el.textContent = d.comment;
      if (d.ts) ts.textContent = d.ts.slice(0, 10) + " " + d.ts.slice(11, 16);
    }
  } catch(e) {}
}

// Also try live API fetches (work when proxy forwards /api/*)
loadStatus(); loadLog(); loadSensors(); loadValves(); loadMoistureComment();
setInterval(loadStatus,         60000);
setInterval(loadLog,           120000);
setInterval(loadSensors,       120000);
setInterval(loadValves,         30000);
setInterval(loadMoistureComment, 600000);  // refresh every 10 min
// Full page reload every 5 min keeps embedded data fresh even if API calls fail
setInterval(() => { location.reload(); }, 300000);
</script>
</body>
</html>"""


@app.route("/")
@require_auth
def dashboard():
    # Force fresh URL on every server restart — busts browser/proxy cache
    if request.args.get("v") != _START:
        return Response("", 302, {"Location": f"/?v={_START}",
                                   "Cache-Control": "no-store"})
    try:
        status = _collect_status()
        aqara  = aqara_data(use_cache=True)
        soil   = get_soil_data(use_cache=True)
        valves = get_all_valve_status()
        log_path = os.path.join(os.path.dirname(__file__), "kolo_agent.log")
        log_lines = []
        if os.path.exists(log_path):
            with open(log_path) as f:
                log_lines = [l.rstrip() for l in f.readlines()[-30:]]
        init_data = json.dumps({
            "status":  status,
            "sensors": {"aqara": aqara, "soil": soil},
            "valves":  {"valves": valves},
            "log":     {"lines": log_lines},
        }).replace("</", "<\\/")
    except Exception:
        init_data = "null"
    # Server-side stamp — visible even with zero JS
    try:
        _ms  = status["mower"]["state"]
        _tmp = status["weather"]["temp_c"]
        _ts  = status["ts"][11:16]
    except Exception:
        _ms, _tmp, _ts = "?", "?", "?"
    stamp = (f'<div style="position:fixed;bottom:0;left:0;right:0;background:#1a3a18;'
             f'color:#8dd484;font:12px monospace;padding:4px 12px;z-index:9999;'
             f'border-top:1px solid #3a6b35">'
             f'srv {_ts} · mower:{_ms} · {_tmp}°C</div>')
    inject = f'window.__INIT__={init_data};'
    html = DASHBOARD_HTML.replace("/*__INIT_DATA__*/", inject, 1)
    html = html.replace("</body>", stamp + "</body>", 1)
    resp = Response(html, content_type="text/html")
    resp.headers["Cache-Control"] = "no-store"
    return resp


if __name__ == "__main__":
    port = int(os.environ.get("KOLO_DASH_PORT", 5000))
    print(f"KoloDash starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
