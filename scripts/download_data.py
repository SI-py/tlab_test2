#!/usr/bin/env python3
"""Download the complete public KION dataset."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile


URL = "https://raw.githubusercontent.com/irsafilo/KION_DATASET/main/data_original.zip"
FILES = ("interactions.csv", "users.csv", "items.csv")


def download(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="kion-download-", dir=destination.parent)
    try:
        request = urllib.request.Request(URL, headers={"User-Agent": "recsys-research"})
        with os.fdopen(fd, "wb") as output, urllib.request.urlopen(request, timeout=120) as response:
            shutil.copyfileobj(response, output, 1024 * 1024)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/raw")
    parser.add_argument("--archive", type=Path, help="Use an existing data_original.zip")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if all((output / name).exists() for name in FILES):
        print(f"KION is already available in {output}")
        return
    archive = args.archive or output / "data_original.zip"
    if not archive.exists():
        if args.archive:
            parser.error(f"Archive does not exist: {archive}")
        print(f"Downloading {URL}", flush=True)
        download(archive)
    with zipfile.ZipFile(archive) as package:
        for filename in FILES:
            with package.open(f"data_original/{filename}") as source, (output / filename).open("wb") as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
            print(f"Saved {output / filename}", flush=True)


if __name__ == "__main__":
    main()
