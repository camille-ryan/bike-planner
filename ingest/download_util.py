"""Resumable HTTP file downloader with progress output."""
from pathlib import Path
import requests


def fetch(url: str, out: Path, label: str = "") -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        print(f"[{label}] cached: {out.name} ({out.stat().st_size/1e6:.0f} MB)")
        return out
    print(f"[{label}] GET {url}")
    tmp = out.with_suffix(out.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        seen = 0
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                seen += len(chunk)
                if total:
                    print(f"  {out.name} {seen/1e6:.0f}/{total/1e6:.0f} MB", end="\r")
        print()
    tmp.rename(out)
    return out
