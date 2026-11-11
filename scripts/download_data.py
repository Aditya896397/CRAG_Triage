"""Download the PhysioNet/CinC Challenge 2019 sepsis dataset (open access).

PhysioNet serves one pipe-separated file per patient, so this script fetches
them in parallel and can be re-run safely: files already on disk are skipped.

    python scripts/download_data.py                 # full dataset (~40k files)
    python scripts/download_data.py --limit 3000    # first 3000 per hospital (quick dev)
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE = "https://physionet.org/files/challenge-2019/1.0.0/training/"
SETS = {"A": "training_setA", "B": "training_setB"}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "raw"


def list_files(session: requests.Session, set_dir: str) -> list[str]:
    html = session.get(BASE + set_dir + "/", timeout=60).text
    return sorted(set(re.findall(r'href="(p\d+\.psv)"', html)))


def fetch(session: requests.Session, url: str, dest: Path, retries: int = 4) -> bool:
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200 and r.text.startswith("HR|"):
                tmp = dest.with_suffix(".tmp")
                tmp.write_text(r.text, encoding="utf-8")
                tmp.replace(dest)
                return True
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0, help="max files per hospital (0 = all)")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = "crag-triage-dataset-downloader/1.0"
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers)
    session.mount("https://", adapter)

    jobs = []
    for hosp, set_dir in SETS.items():
        out_dir = args.out / set_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        names = list_files(session, set_dir)
        if args.limit:
            names = names[: args.limit]
        print(f"Hospital {hosp}: {len(names)} files listed", flush=True)
        for n in names:
            dest = out_dir / n
            if not dest.exists():
                jobs.append((BASE + set_dir + "/" + n, dest))

    print(f"{len(jobs)} files to download", flush=True)
    failed, done = [], 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch, session, u, d): u for u, d in jobs}
        for fut in as_completed(futures):
            done += 1
            if not fut.result():
                failed.append(futures[fut])
            if done % 1000 == 0:
                print(f"  {done}/{len(jobs)} done", flush=True)
    print(f"Finished. {len(failed)} failed.", flush=True)
    if failed:
        print("Re-run the script to retry failed files.")
        sys.exit(1)


if __name__ == "__main__":
    main()
