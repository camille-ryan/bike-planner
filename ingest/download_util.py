"""Resumable HTTP file downloader with progress output.

Designed to survive flaky links: a Range-based resume from any leftover
`.part` file plus a retry loop on read timeouts. Progress is logged every
~50 MB so the log stays readable when stdout is captured to a file.
"""
import time
from pathlib import Path

import requests


PROGRESS_STEP = 50 * (1 << 20)   # 50 MB between progress lines
CHUNK = 1 << 20                  # 1 MB read chunk
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 300               # generous — Geofabrik can be slow on large files
MAX_ATTEMPTS = 5


def _attempt(url: str, tmp: Path, label: str, total_hint: int = 0) -> int:
    """One download attempt, supporting Range resume from existing tmp size.

    Returns the final byte count on success (whether resumed or fresh).
    Raises requests exceptions on failure.
    """
    headers = {}
    start = tmp.stat().st_size if tmp.exists() else 0
    if start:
        headers["Range"] = f"bytes={start}-"
    timeout = (CONNECT_TIMEOUT, READ_TIMEOUT)

    with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
        # Range honored → 206; full restart → 200; anything else surface as error.
        if start and r.status_code == 200:
            # Server ignored the Range header; restart from zero.
            print(f"  {tmp.name} server ignored Range, restarting from 0", flush=True)
            tmp.unlink(missing_ok=True)
            start = 0
        elif r.status_code not in (200, 206):
            r.raise_for_status()

        # Total expected file size (post-resume offset added back if resuming).
        clen = int(r.headers.get("Content-Length") or 0)
        total = total_hint or (start + clen if r.status_code == 206 else clen)

        seen = start
        next_threshold = ((seen // PROGRESS_STEP) + 1) * PROGRESS_STEP
        mode = "ab" if start else "wb"
        with tmp.open(mode) as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                seen += len(chunk)
                if seen >= next_threshold or (total and seen >= total):
                    if total:
                        print(f"  {tmp.name} {seen/1e6:>5.0f}/{total/1e6:.0f} MB ({seen/total*100:.0f}%)", flush=True)
                    else:
                        print(f"  {tmp.name} {seen/1e6:.0f} MB", flush=True)
                    next_threshold += PROGRESS_STEP
        return seen


def fetch(url: str, out: Path, label: str = "") -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        print(f"[{label}] cached: {out.name} ({out.stat().st_size/1e6:.0f} MB)")
        return out

    tmp = out.with_suffix(out.suffix + ".part")
    # Cross-invocation resume is unsafe — Geofabrik publishes daily updates,
    # so stale .part bytes from a previous run won't align with bytes the
    # server is serving today, producing a corrupt PBF that osmium fails
    # to decompress. Discard stale .part files; within-run retries below
    # are still resumable since the file version can't change mid-fetch.
    if tmp.exists():
        print(f"[{label}] discarding stale {tmp.name} ({tmp.stat().st_size/1e6:.0f} MB)", flush=True)
        tmp.unlink()

    print(f"[{label}] GET {url}", flush=True)

    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            seen = _attempt(url, tmp, label)
            print(f"  {tmp.name} done ({seen/1e6:.0f} MB)", flush=True)
            tmp.rename(out)
            return out
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_exc = exc
            backoff = min(60, 5 * attempt)
            cur = tmp.stat().st_size if tmp.exists() else 0
            print(f"  {tmp.name} attempt {attempt}/{MAX_ATTEMPTS} failed at {cur/1e6:.0f} MB: {exc.__class__.__name__}; retrying in {backoff}s", flush=True)
            time.sleep(backoff)

    raise RuntimeError(f"[{label}] gave up on {url} after {MAX_ATTEMPTS} attempts: {last_exc}")
