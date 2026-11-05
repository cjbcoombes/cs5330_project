"""Download and extract the raw datasets.

Usage:
    uv run python scripts/download_data.py <dataset_name> [--force]
    uv run python scripts/download_data.py crohme
    uv run python scripts/download_data.py hme100k
    uv run python scripts/download_data.py --all

Each archive goes to data/raw/<dataset>/<archive_name>.zip and is extracted
into data/raw/<dataset>/<archive_name>/.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"

CHUNK_SIZE = 1 << 20  # 1 MiB
MAX_RETRIES = 5
READ_TIMEOUT = 60

# dataset -> {archive_name: (url, md5)}
DATASETS: dict[str, dict[str, tuple[str, str]]] = {
    "crohme": {
        "crohme_2023": (
            "https://zenodo.org/records/8428035/files/CROHME23.zip?download=1",
            "9eb899ddb0e87c80a17f489bf6162c33",
        ),
        "crohme_2016": (
            "https://ndownloader.figshare.com/files/58680376",
            "4ed9689c59b96e3daca2da79e43b1f5b",
        ),
    },
    "hme100k": {
        "hme100k": (
            "https://ndownloader.figshare.com/files/58680373",
            "9246808ed68b0cf71c710a78f3538bc4",
        ),
    },
}


def checksum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")

    for attempt in range(1, MAX_RETRIES + 1):
        resume_from = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

        try:
            with requests.get(url, stream=True, timeout=(10, READ_TIMEOUT), headers=headers) as resp:
                if resume_from and resp.status_code == 200:
                    # Server ignored the Range request -- it'll send the whole
                    # file from byte 0, so truncate instead of appending.
                    resume_from = 0
                resp.raise_for_status()

                total = int(resp.headers.get("content-length", 0)) + resume_from
                written = resume_from
                mode = "ab" if resume_from else "wb"
                with open(dest, mode) as f:
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        f.write(chunk)
                        written += len(chunk)
                        if total:
                            pct = 100 * written / total
                            print(f"\r  {written / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="")
            print()
            return
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"\n  Attempt {attempt}/{MAX_RETRIES} failed ({e}); retrying with resume...")


def process_dataset(name: str, force: bool = False) -> None:
    if name not in DATASETS:
        raise SystemExit(f"Unknown dataset '{name}'. Available: {', '.join(sorted(DATASETS))}")

    dataset_dir = RAW_DATA_DIR / name
    for archive_name, (url, expected_md5) in DATASETS[name].items():
        archive_path = dataset_dir / f"{archive_name}.zip"
        extract_dir = dataset_dir / archive_name

        if extract_dir.exists() and not force:
            print(f"[{name}/{archive_name}] Already extracted at {extract_dir}, skipping.")
            continue

        if archive_path.exists() and not force:
            print(f"[{name}/{archive_name}] Found existing archive at {archive_path}, skipping download.")
        else:
            # Download to a .part sibling and rename only on success, so an
            # interrupted/overwritten download can never corrupt a
            # previously-good archive sitting at the final path.
            tmp_path = archive_path.with_name(archive_path.name + ".part")
            download_file(url, tmp_path)
            tmp_path.replace(archive_path)

        actual = checksum(archive_path)
        if actual != expected_md5:
            raise SystemExit(f"[{name}/{archive_name}] md5 mismatch: expected {expected_md5}, got {actual}")
        print(f"[{name}/{archive_name}] md5 checksum OK")

        print(f"Extracting {archive_path.name} -> {extract_dir}")
        shutil.unpack_archive(str(archive_path), str(extract_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", choices=sorted(DATASETS), help="Dataset to download/extract")
    parser.add_argument("--all", action="store_true", help="Download/extract every dataset")
    parser.add_argument("--force", action="store_true", help="Re-download/re-extract even if files already exist")
    args = parser.parse_args()

    if args.all:
        for name in DATASETS:
            process_dataset(name, force=args.force)
        return

    if not args.dataset:
        parser.print_help()
        raise SystemExit(1)

    process_dataset(args.dataset, force=args.force)


if __name__ == "__main__":
    main()
