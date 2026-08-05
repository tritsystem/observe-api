"""
Real SQLite backup for observe_api.db -- uses sqlite3's built-in
Connection.backup() API, which is safe to run against a LIVE database
(including one in WAL mode with an active writer) since it uses SQLite's
own online backup mechanism rather than copying the file bytes directly
(a raw file copy of a WAL-mode DB mid-write can produce a corrupt or
inconsistent backup -- this avoids that class of bug entirely).

Keeps the last KEEP_COUNT backups, prunes older ones -- this is customer
billing/API-key data, worth keeping real history for, but not worth
unbounded disk growth either.
"""
import os
import sqlite3
import time

DB_PATH = "/home/gbran/observe-api/observe_api.db"
BACKUP_DIR = "/home/gbran/observe-api/backups"
KEEP_COUNT = 28  # every 6h => 1 week of history


def main():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest_path = os.path.join(BACKUP_DIR, f"observe_api_{ts}.db")

    src = sqlite3.connect(DB_PATH, timeout=30)
    dest = sqlite3.connect(dest_path)
    with dest:
        src.backup(dest)
    dest.close()
    src.close()
    print(f"backed up to {dest_path}")

    backups = sorted(
        f for f in os.listdir(BACKUP_DIR)
        if f.startswith("observe_api_") and f.endswith(".db")
    )
    for old in backups[:-KEEP_COUNT]:
        os.remove(os.path.join(BACKUP_DIR, old))
        print(f"pruned old backup {old}")


if __name__ == "__main__":
    main()
