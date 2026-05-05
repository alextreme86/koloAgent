#!/usr/bin/env python3
"""
kolo_logger.py — 5-minute time-series logger for ML training.

Stores: soil moisture, climate, weather, irrigation events.
Future model: predict watering minutes needed to reach target moisture,
              and how weather/temperature accelerates soil drying.

Cron:
  */5 * * * * python3 /home/kolo/kolonihave/kolo_logger.py >> /home/kolo/kolonihave/kolo_logger.log 2>&1
"""

import sqlite3, datetime, os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

from kolo_soil       import get_soil_data, SENSORS as SOIL_SENSORS
from kolo_aqara      import get_sensor_data as aqara_data
from kolo_irrigation import get_all_valve_status
from kolo_agent      import get_weather

DB_PATH    = os.path.join(os.path.dirname(__file__), "kolo_data.db")
STATE_PATH = os.path.join(os.path.dirname(__file__), "kolo_logger_state.json")

# Which soil sensors belong to each valve zone (for avg moisture snapshot)
VALVE_SOIL_ZONES = {
    "greenhouse": ["greenhouse", "greenhouse_basil"],
    "outdoor":    ["cassa_alta", "fragole", "cassa_bassa_serra", "cassa_bassa"],
}


# ── DB setup ──────────────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS sensor_readings (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ts          TEXT    NOT NULL,
        zone        TEXT    NOT NULL,
        moisture    REAL,
        soil_temp   REAL,
        battery     INTEGER
    );

    CREATE TABLE IF NOT EXISTS climate_readings (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ts          TEXT    NOT NULL,
        location    TEXT    NOT NULL,
        temp        REAL,
        humidity    REAL
    );

    CREATE TABLE IF NOT EXISTS weather_readings (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ts              TEXT    NOT NULL,
        temp_c          REAL,
        humidity        INTEGER,
        wind_kmh        INTEGER,
        rain_mm_today   REAL,
        rain_next_3h    REAL,
        max_c_today     INTEGER,
        min_c_today     INTEGER,
        rain_tomorrow   REAL,
        max_c_tomorrow  INTEGER,
        uv_index        INTEGER,
        cloud_cover     INTEGER,
        solar_rad_wm2   REAL,
        dew_point_c     INTEGER
    );

    CREATE TABLE IF NOT EXISTS irrigation_events (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        zone                    TEXT    NOT NULL,
        ts_open                 TEXT,
        ts_close                TEXT,
        duration_requested_s    INTEGER,
        duration_actual_s       REAL,
        moisture_before         REAL,
        moisture_after          REAL,
        soil_temp_at_open       REAL,
        -- weather at open
        outdoor_temp_at_open    REAL,
        outdoor_humidity_at_open INTEGER,
        rain_mm_at_open         REAL,
        -- aqara zone climate at open (greenhouse or outdoor sensor)
        zone_air_temp_at_open   REAL,
        zone_air_humidity_at_open INTEGER,
        status                  TEXT    DEFAULT 'open'
    );

    CREATE INDEX IF NOT EXISTS idx_sensor_ts   ON sensor_readings(ts);
    CREATE INDEX IF NOT EXISTS idx_sensor_zone ON sensor_readings(zone);
    CREATE INDEX IF NOT EXISTS idx_irr_zone    ON irrigation_events(zone);
    CREATE INDEX IF NOT EXISTS idx_irr_status  ON irrigation_events(status);
    """)
    conn.commit()


# ── State (valve open/close transition tracking) ───────────────────────────────

def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


# ── Zone moisture helpers ──────────────────────────────────────────────────────

def zone_avg_moisture(zone: str, soil: dict) -> tuple[float | None, float | None]:
    """Return (avg_moisture, avg_soil_temp) for a valve zone, active sensors only."""
    keys = VALVE_SOIL_ZONES.get(zone, [])
    m_vals, t_vals = [], []
    for k in keys:
        s = soil.get(k, {})
        if s.get("active") and s.get("moisture") is not None:
            m_vals.append(float(s["moisture"]))
        if s.get("active") and s.get("temperature") is not None:
            t_vals.append(float(s["temperature"]))
    avg_m = round(sum(m_vals) / len(m_vals), 1) if m_vals else None
    avg_t = round(sum(t_vals) / len(t_vals), 1) if t_vals else None
    return avg_m, avg_t


# ── Loggers ───────────────────────────────────────────────────────────────────

def log_soil(conn: sqlite3.Connection, ts: str, soil: dict):
    rows = []
    for zone, s in soil.items():
        if not isinstance(s, dict):  # skip "ts" float key added by get_soil_data
            continue
        rows.append((ts, zone, s.get("moisture"), s.get("temperature"), s.get("battery")))
    conn.executemany(
        "INSERT INTO sensor_readings (ts, zone, moisture, soil_temp, battery) VALUES (?,?,?,?,?)",
        rows,
    )
    conn.commit()
    print(f"  Soil: {len(rows)} zones")


def log_climate(conn: sqlite3.Connection, ts: str, aqara: dict):
    rows = []
    for loc, a in aqara.items():
        if not isinstance(a, dict):  # skip "ts" float key
            continue
        rows.append((ts, loc, a.get("temperature"), a.get("humidity")))
    conn.executemany(
        "INSERT INTO climate_readings (ts, location, temp, humidity) VALUES (?,?,?,?)",
        rows,
    )
    conn.commit()
    print(f"  Climate: {len(rows)} locations")


def log_weather(conn: sqlite3.Connection, ts: str, w: dict):
    if not w:
        print("  Weather: unavailable")
        return
    conn.execute(
        """INSERT INTO weather_readings
           (ts, temp_c, humidity, wind_kmh, rain_mm_today, rain_next_3h,
            max_c_today, min_c_today, rain_tomorrow, max_c_tomorrow,
            uv_index, cloud_cover, solar_rad_wm2, dew_point_c)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (ts, w.get("temp_c"), w.get("humidity"), w.get("wind_kmh"),
         w.get("rain_mm_today"), w.get("rain_next_3h"),
         w.get("max_c_today"), w.get("min_c_today"),
         w.get("rain_tomorrow"), w.get("max_c_tomorrow"),
         w.get("uv_index"), w.get("cloud_cover"),
         w.get("solar_rad_wm2"), w.get("dew_point_c")),
    )
    conn.commit()
    print(f"  Weather: {w.get('temp_c')}°C hum={w.get('humidity')}% "
          f"UV={w.get('uv_index')} solar={w.get('solar_rad_wm2')}W/m² cloud={w.get('cloud_cover')}%")


def check_valves(conn: sqlite3.Connection, ts: str, valves: dict,
                 soil: dict, aqara: dict, w: dict, state: dict) -> dict:
    """Detect valve transitions and log irrigation events."""
    prev = state.get("valves", {})

    for zone, v in valves.items():
        is_open  = v.get("open", False)
        was_open = prev.get(zone, {}).get("open", False)

        avg_m, avg_t = zone_avg_moisture(zone, soil)
        outdoor_temp = w.get("temp_c")       if w else None
        outdoor_hum  = w.get("humidity")     if w else None
        rain_mm      = w.get("rain_mm_today") if w else None

        # Aqara zone sensor: greenhouse→"greenhouse", outdoor→"outdoor"
        zone_aqara   = aqara.get(zone, {}) or {}
        zone_air_t   = zone_aqara.get("temperature")
        zone_air_hum = zone_aqara.get("humidity")

        if is_open and not was_open:
            # Valve just opened → new irrigation event
            countdown = v.get("countdown", 0)  # seconds ≈ requested duration at open
            conn.execute(
                """INSERT INTO irrigation_events
                   (zone, ts_open, duration_requested_s,
                    moisture_before, soil_temp_at_open,
                    outdoor_temp_at_open, outdoor_humidity_at_open, rain_mm_at_open,
                    zone_air_temp_at_open, zone_air_humidity_at_open, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,'open')""",
                (zone, ts, countdown, avg_m, avg_t,
                 outdoor_temp, outdoor_hum, rain_mm,
                 zone_air_t, zone_air_hum),
            )
            conn.commit()
            prev.setdefault(zone, {})["open_ts"] = ts
            print(f"  Irrigation OPEN: {zone} requested={countdown}s moisture_before={avg_m}% "
                  f"zone_air={zone_air_t}°C/{zone_air_hum}%")

        elif not is_open and was_open:
            # Valve just closed → compute duration and snapshot moisture_after
            open_ts = prev.get(zone, {}).get("open_ts")
            duration_actual = None
            if open_ts:
                try:
                    dt_open  = datetime.datetime.fromisoformat(open_ts)
                    dt_close = datetime.datetime.fromisoformat(ts)
                    duration_actual = (dt_close - dt_open).total_seconds()
                except Exception:
                    pass

            conn.execute(
                """UPDATE irrigation_events
                   SET ts_close=?, duration_actual_s=?, moisture_after=?, status='completed'
                   WHERE id = (
                       SELECT id FROM irrigation_events
                       WHERE zone=? AND status='open'
                       ORDER BY id DESC LIMIT 1
                   )""",
                (ts, duration_actual, avg_m, zone),
            )
            conn.commit()
            prev.get(zone, {}).pop("open_ts", None)
            print(f"  Irrigation CLOSE: {zone} duration={duration_actual}s moisture_after={avg_m}%")

        # Always record current open state
        prev.setdefault(zone, {})["open"] = is_open

    state["valves"] = prev
    return state


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"\n[{ts}] kolo_logger")

    conn  = get_db()
    init_db(conn)
    state = load_state()

    # Fetch — soil + valves fresh, weather can be slightly cached
    soil   = get_soil_data(use_cache=False)
    aqara  = aqara_data(use_cache=False)
    valves = get_all_valve_status()
    w      = get_weather()

    log_soil(conn, ts, soil)
    log_climate(conn, ts, aqara)
    log_weather(conn, ts, w)
    state = check_valves(conn, ts, valves, soil, aqara, w, state)

    save_state(state)
    conn.close()
    print("  done.")


if __name__ == "__main__":
    main()
