"""
00a_download_acled.py
=====================
Pull the full ACLED dataset via the OAuth API into data_raw/acled_full.csv.

The script is RESUMABLE: each paginated response is written to its own
parquet shard under data_raw/acled_shards/. If the run is interrupted, just
re-run — already-completed pages are skipped. When all pages are pulled the
shards are concatenated to a single CSV.

Authentication:
    Set these environment variables before running.
    Do NOT hard-code credentials in this file.
        export ACLED_EMAIL="you@your-institution.edu"
        export ACLED_PASSWORD="your_password"
    On Windows PowerShell:
        $env:ACLED_EMAIL = "..."
        $env:ACLED_PASSWORD = "..."

API spec used (per https://acleddata.com/api-documentation/getting-started):
    Token endpoint: POST https://acleddata.com/oauth/token
        Body (x-www-form-urlencoded):
            username, password, grant_type=password,
            client_id=acled, scope=authenticated
    Read endpoint:  GET  https://acleddata.com/api/acled/read
        Header: Authorization: Bearer <access_token>
        Query: page, limit, plus optional filters
    Token TTL: 86400s (24h). Refresh by re-authenticating when needed.

Outputs:
    data_raw/acled_shards/page_00001.parquet, page_00002.parquet, ...
    data_raw/acled_full.csv
    logs/acled_download.log
"""

import os
import sys
import json
import time
import io
import math
import datetime
import requests
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
master = import_module("00_master")
CONFIG = master.CONFIG
master.ensure_dirs()

# --------------------------------------------------------------------------
# Knobs
# --------------------------------------------------------------------------
PAGE_LIMIT       = 5000     # rows per page; ACLED's effective max is around 5000
SLEEP_BETWEEN    = 0.5      # seconds between successful page calls (be polite)
RETRY_MAX        = 5
RETRY_BACKOFF    = 2.0      # exponential backoff base
TOKEN_REFRESH_S  = 23 * 3600  # refresh token after this many seconds even if not expired

TOKEN_URL = "https://acleddata.com/oauth/token"
READ_URL  = "https://acleddata.com/api/acled/read"

# Optional API-side filters. Leave empty to pull everything; the project uses
# 00b_subset_acled.py for use-case slicing afterwards.
API_FILTERS = {
    # "event_date": "2018-01-01|2026-12-31",  # date range example
    # "event_date_where": "BETWEEN",
    # "country": "Ukraine",
}

SHARD_DIR = os.path.join(CONFIG["data_raw_dir"], "acled_shards")
os.makedirs(SHARD_DIR, exist_ok=True)

LOG_PATH = os.path.join(CONFIG["logs_dir"], "acled_download.log")
def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------
EMAIL = os.environ.get("ACLED_EMAIL")
PASSWORD = os.environ.get("ACLED_PASSWORD")
if not EMAIL or not PASSWORD:
    sys.exit(
        "ERROR: ACLED_EMAIL and ACLED_PASSWORD environment variables are required.\n"
        "On macOS/Linux:\n"
        "    export ACLED_EMAIL='you@institution.edu'\n"
        "    export ACLED_PASSWORD='...'\n"
        "On Windows PowerShell:\n"
        "    $env:ACLED_EMAIL = 'you@institution.edu'\n"
        "    $env:ACLED_PASSWORD = '...'\n"
    )

# --------------------------------------------------------------------------
# OAuth — get a bearer token. Refreshed periodically for long pulls.
# --------------------------------------------------------------------------
def get_access_token():
    log("Requesting OAuth access token...")
    resp = requests.post(
        TOKEN_URL,
        data={
            "username":   EMAIL,
            "password":   PASSWORD,
            "grant_type": "password",
            "client_id":  "acled",
            "scope":      "authenticated",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=60,
    )
    if resp.status_code != 200:
        log(f"Token request failed: {resp.status_code} {resp.text[:500]}")
        sys.exit(1)
    body = resp.json()
    log(f"Got token (expires in {body.get('expires_in', '?')}s).")
    return body["access_token"], time.time()

token, token_started = get_access_token()

# --------------------------------------------------------------------------
# Helper: fetch one page with retries and token refresh
# --------------------------------------------------------------------------
def fetch_page(page_num):
    global token, token_started

    # Refresh token if it's getting old
    if time.time() - token_started > TOKEN_REFRESH_S:
        log("Refreshing access token (preemptive).")
        token, token_started = get_access_token()

    params = {"page": page_num, "limit": PAGE_LIMIT}
    params.update(API_FILTERS)

    last_err = None
    for attempt in range(RETRY_MAX):
        try:
            resp = requests.get(
                READ_URL,
                headers={"Authorization": f"Bearer {token}"},
                params=params,
                timeout=120,
            )
            if resp.status_code == 401:
                # Token expired mid-run; refresh and retry
                log("Got 401, refreshing token and retrying...")
                token, token_started = get_access_token()
                continue
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                log(f"  attempt {attempt+1} failed: {last_err}")
                time.sleep(RETRY_BACKOFF ** attempt)
                continue
            try:
                payload = resp.json()
            except json.JSONDecodeError as e:
                last_err = f"JSON decode error: {e}; first 300 chars: {resp.text[:300]}"
                log(f"  attempt {attempt+1} {last_err}")
                time.sleep(RETRY_BACKOFF ** attempt)
                continue
            return payload
        except requests.RequestException as e:
            last_err = str(e)
            log(f"  attempt {attempt+1} request error: {e}")
            time.sleep(RETRY_BACKOFF ** attempt)
    raise RuntimeError(f"Page {page_num} failed after {RETRY_MAX} attempts: {last_err}")

# --------------------------------------------------------------------------
# Paginated download. Each page is written as its own parquet shard so we
# can resume cleanly.
# --------------------------------------------------------------------------
def shard_path(page_num):
    return os.path.join(SHARD_DIR, f"page_{page_num:05d}.parquet")

log("Beginning paginated pull...")
log(f"  page limit per request: {PAGE_LIMIT}")
log(f"  filters: {API_FILTERS}")

page = 1
total_rows = 0
empty_in_a_row = 0

while True:
    out_path = shard_path(page)
    if os.path.exists(out_path):
        # Already pulled this page; skip
        existing = pd.read_parquet(out_path)
        log(f"  page {page:05d}: already on disk ({len(existing):,} rows). Skipping.")
        total_rows += len(existing)
        if len(existing) < PAGE_LIMIT:
            log(f"  short page detected on resume — assuming complete.")
            break
        page += 1
        continue

    payload = fetch_page(page)
    rows = payload.get("data") or payload.get("results") or payload  # endpoint may vary
    if isinstance(rows, dict):
        # Some responses wrap rows under another key
        for k in ("data", "results", "rows"):
            if k in rows and isinstance(rows[k], list):
                rows = rows[k]
                break

    if not isinstance(rows, list):
        log(f"Unexpected response shape on page {page}: {type(rows)}")
        log(f"Payload keys: {list(payload.keys()) if isinstance(payload, dict) else 'n/a'}")
        log(f"First 500 chars: {json.dumps(payload)[:500]}")
        break

    if len(rows) == 0:
        empty_in_a_row += 1
        log(f"  page {page:05d}: 0 rows.")
        # Two empty pages in a row = end of data
        if empty_in_a_row >= 2:
            log("Two empty pages in a row — assuming end of data.")
            break
        page += 1
        continue
    empty_in_a_row = 0

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    total_rows += len(df)
    log(f"  page {page:05d}: {len(df):,} rows  (running total: {total_rows:,})")

    if len(df) < PAGE_LIMIT:
        # Last page (partial). Stop.
        log("Short page (less than limit) — pull complete.")
        break

    time.sleep(SLEEP_BETWEEN)
    page += 1

log(f"Total rows pulled across {page} pages: {total_rows:,}")

# --------------------------------------------------------------------------
# Concatenate shards into one CSV
# --------------------------------------------------------------------------
log("Concatenating shards into data_raw/acled_full.csv ...")
shards = sorted(f for f in os.listdir(SHARD_DIR) if f.endswith(".parquet"))
if not shards:
    log("No shards on disk — nothing to concatenate.")
    sys.exit(0)

dfs = []
for s in shards:
    dfs.append(pd.read_parquet(os.path.join(SHARD_DIR, s)))
full = pd.concat(dfs, ignore_index=True)
log(f"  combined frame: {full.shape}")

# Some columns may differ across older/newer events; pandas concat handles that with NaN.
# Drop exact duplicates if event_id_cnty is present.
if "event_id_cnty" in full.columns:
    before = len(full)
    full = full.drop_duplicates(subset=["event_id_cnty"], keep="last")
    log(f"  deduplicated on event_id_cnty: {before:,} -> {len(full):,}")

out_csv = os.path.join(CONFIG["data_raw_dir"], "acled_full.csv")
full.to_csv(out_csv, index=False)
log(f"Wrote {out_csv} ({os.path.getsize(out_csv) / 1e6:.1f} MB).")
log("Done. Next: edit the FILTER block in 00b_subset_acled.py and run it.")
