"""Download HAB-1's sources from their official homes into sources/, pinned to the exact versions HAB-1 was built
from (sources.json: path, URL at a fixed commit, SHA-256). A file whose checksum doesn't match stops the download, so
every build starts from the same bytes.

    python download.py

Standard library only. Files already present with the right checksum are skipped. Each source keeps its own licence
(README.md, "Sources and licences").
"""
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "sources"
UA = "HAB-1 benchmark download (https://github.com/nahil444n/hab-1)"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while block := r.read(1 << 20):
            f.write(block)


def unzip(zip_path, to):
    """Extract a zip, refusing any member that would land outside `to`."""
    to.mkdir(parents=True, exist_ok=True)
    root = to.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            name = Path(m.filename).name
            if m.is_dir() or not name:
                continue
            target = (to / name).resolve()
            if target.parent != root:
                raise SystemExit(f"refusing zip member {m.filename!r}")
            target.write_bytes(z.read(m))


def main():
    files = json.load(open(HERE / "sources.json", encoding="utf-8"))
    bad = 0
    for i, f in enumerate(files, 1):
        dest = SRC / f["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not (dest.exists() and sha256(dest) == f["sha256"]):
            fetch(f["url"], dest)
        if sha256(dest) != f["sha256"]:
            print(f"checksum mismatch: {f['path']} from {f['url']}", file=sys.stderr)
            bad += 1
            continue
        if "unzip" in f:
            unzip(dest, SRC / f["unzip"])
        print(f"[{i}/{len(files)}] {f['path']}", flush=True)
    if bad:
        raise SystemExit(f"{bad} file(s) did not match their checksum")
    print(f"all {len(files)} files downloaded and checked into {SRC}")


if __name__ == "__main__":
    main()
