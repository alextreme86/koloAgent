#!/usr/bin/env python3
"""
Lightweight mow session detector — run via cron at 17:30 daily.
Records completed mow sessions that the noon agent may have missed.
"""
import sys
sys.path.insert(0, "/home/kolo/kolonihave")

import datetime
from kolo_agent import get_indego_state, load_state, record_mow_session, save_state

def main():
    today = datetime.date.today().isoformat()
    state  = load_state()

    if state.get("last_mow_date") == today:
        print(f"[mow_check] Already recorded mow for {today}, skipping.")
        return

    indego    = get_indego_state()
    mowed_pct = indego.get("mowed", 0)
    mow_state = indego.get("state")

    # 257=Charging, 258=Docked — mower returned after completing
    if mowed_pct >= 95 and mow_state in (257, 258):
        print(f"[mow_check] Mow completed detected ({mowed_pct}%) — recording.")
        record_mow_session(state, mowed_pct)
    else:
        print(f"[mow_check] No completed mow (mowed={mowed_pct}%, state={mow_state}).")

if __name__ == "__main__":
    main()
