#!/usr/bin/env python3
"""
kolo_moisture_check.py — Ollama moisture commentary, runs 2× per day.

Cron (on Pi):
  0 7  * * * python3 /home/kolo/kolonihave/kolo_moisture_check.py >> /home/kolo/kolonihave/kolo_moisture_check.log 2>&1
  0 17 * * * python3 /home/kolo/kolonihave/kolo_moisture_check.py >> /home/kolo/kolonihave/kolo_moisture_check.log 2>&1
"""

import sys, os, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kolo_agent       import get_moisture_commentary, save_moisture_commentary, get_weather
from kolo_aqara       import get_sensor_data as aqara_data
from kolo_soil        import get_soil_data

def main():
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"\n[{ts}] kolo_moisture_check")

    soil    = get_soil_data(use_cache=False)
    aqara   = aqara_data(use_cache=False)
    weather = get_weather()

    print("  Calling Ollama for moisture commentary (may take several minutes)…")
    comment = get_moisture_commentary(soil, weather, aqara=aqara)

    if comment:
        save_moisture_commentary(comment)
        print(f"  Saved: {comment[:120]}…")
    else:
        print("  No response from Ollama.")

    print("  done.")

if __name__ == "__main__":
    main()
