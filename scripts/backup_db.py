"""scripts/backup_db.py - helper for backup_db.bat: mysqldump using DATABASE_URL from .env.

The password goes into a temporary option file (never on the command line, never printed).
Usage: python scripts/backup_db.py OUTPUT.sql
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parent.parent


def find_mysqldump() -> str | None:
    found = shutil.which("mysqldump")
    if found:
        return found
    base = os.environ.get("ProgramFiles", r"C:\Program Files")
    for p in sorted(Path(base, "MySQL").glob("MySQL Server*/bin/mysqldump.exe"), reverse=True):
        return str(p)
    return None


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/backup_db.py OUTPUT.sql")
        return 2
    out = Path(sys.argv[1])
    url_text = dotenv_values(ROOT / ".env").get("DATABASE_URL")
    if not url_text:
        print("DATABASE_URL is not set in .env")
        return 1
    url = make_url(url_text)
    if not url.drivername.startswith("mysql"):
        print("DATABASE_URL is not a MySQL URL; nothing to dump.")
        return 1
    exe = find_mysqldump()
    if not exe:
        print("mysqldump not found. Add MySQL Server's bin folder to PATH.")
        return 1

    password = url.password or ""
    fd, cnf = tempfile.mkstemp(suffix=".cnf")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write("[client]\n")
            fh.write(f"user={url.username}\n")
            fh.write('password="' + password.replace("\\", "\\\\").replace('"', '\\"') + '"\n')
            fh.write(f"host={url.host or 'localhost'}\nport={url.port or 3306}\n")
        cmd = [exe, f"--defaults-extra-file={cnf}", "--single-transaction", "--routines",
               "--result-file=" + str(out), url.database]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            msg = res.stderr or ""
            if password:
                msg = msg.replace(password, "***")
            print(msg.strip() or "mysqldump failed")
            return 1
    finally:
        try:
            os.remove(cnf)
        except OSError:
            pass
    print(f"{out.stat().st_size} bytes written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
