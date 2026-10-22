"""Download and extract raw datasets listed in configs/datasets.yaml.

Usage:
    uv run python scripts/download_data.py <dataset_name> [--force]
    uv run python scripts/download_data.py crohme
    uv run python scripts/download_data.py hme100k
    uv run python scripts/download_data.py --list

Each archive is downloaded to data/raw/<dataset>/<archive_name>.<ext> and
extracted into data/raw/<dataset>/<archive_name>/. If an archive's `url` in
the manifest is null, the script assumes you've already placed the file
there manually (e.g. for sources that require a click-through, like a
Google Drive link) and just extracts/verifies it.

Nothing is downloaded by default until you fill in real URLs in
configs/datasets.yaml and run this explicitly -- it is not wired into any
other setup step.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from urllib.parse import urlsplit

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "configs" / "datasets.yaml"
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"

CHUNK_SIZE = 1 << 20  # 1 MiB


def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return yaml.safe_load(f)


def checksum(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


MAX_RETRIES = 5
READ_TIMEOUT = 60


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
                elif resume_from and resp.status_code != 206:
                    resp.raise_for_status()
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
            print(f"\n  Attempt {attempt}/{MAX_RETRIES} failed ({e}); retrying with resume..." if attempt < MAX_RETRIES else "")
            if attempt == MAX_RETRIES:
                raise


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path.name} -> {dest_dir}")
    shutil.unpack_archive(str(archive_path), str(dest_dir))


def process_dataset(name: str, manifest: dict, force: bool = False) -> None:
    if name not in manifest:
        available = ", ".join(sorted(manifest))
        raise SystemExit(f"Unknown dataset '{name}'. Available: {available}")

    entry = manifest[name]
    dataset_dir = RAW_DATA_DIR / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for archive in entry["archives"]:
        archive_name = archive["name"]
        url = archive.get("url")
        expected_sha256 = archive.get("sha256")
        expected_md5 = archive.get("md5")

        if url:
            # Filename is derived from the URL (or `filename`, for sources
            # like figshare's ndownloader whose URLs carry no extension) so
            # re-runs look for the same path they'd download to -- it need
            # not match `archive_name`.
            filename = archive.get("filename") or Path(urlsplit(url).path).name
            archive_path = dataset_dir / filename
        else:
            candidates = sorted(dataset_dir.glob(f"{archive_name}.*"))
            archive_path = candidates[0] if candidates else None

        if archive_path is None:
            print(
                f"[{name}/{archive_name}] No URL in manifest and no local file found. "
                f"Fill in configs/datasets.yaml, or manually place the archive at "
                f"{dataset_dir}/{archive_name}.<ext>. Skipping."
            )
            continue

        if archive_path.exists() and not force:
            print(f"[{name}/{archive_name}] Found existing archive at {archive_path}, skipping download.")
        elif url:
            # Download to a .part sibling and rename only on success, so an
            # interrupted/overwritten download can never corrupt a
            # previously-good archive sitting at the final path.
            tmp_path = archive_path.with_name(archive_path.name + ".part")
            download_file(url, tmp_path)
            tmp_path.replace(archive_path)
        else:
            raise SystemExit(
                f"[{name}/{archive_name}] --force given but there's no URL to re-download from. "
                f"Delete {archive_path} and manually re-place the archive instead."
            )

        if expected_sha256:
            actual = checksum(archive_path, "sha256")
            if actual != expected_sha256:
                raise SystemExit(
                    f"[{name}/{archive_name}] sha256 mismatch: expected {expected_sha256}, got {actual}"
                )
            print(f"[{name}/{archive_name}] sha256 checksum OK")
        elif expected_md5:
            actual = checksum(archive_path, "md5")
            if actual != expected_md5:
                raise SystemExit(
                    f"[{name}/{archive_name}] md5 mismatch: expected {expected_md5}, got {actual}"
                )
            print(f"[{name}/{archive_name}] md5 checksum OK")

        extract_dir = dataset_dir / archive_name
        if extract_dir.exists() and not force:
            print(f"[{name}/{archive_name}] Already extracted at {extract_dir}, skipping.")
            continue
        extract_archive(archive_path, extract_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", help="Dataset name from configs/datasets.yaml (e.g. crohme, hme100k)")
    parser.add_argument("--all", action="store_true", help="Download/extract every dataset in the manifest")
    parser.add_argument("--list", action="store_true", help="List available datasets and exit")
    parser.add_argument("--force", action="store_true", help="Re-download/re-extract even if files already exist")
    args = parser.parse_args()

    manifest = load_manifest()

    if args.list:
        print("Available datasets:")
        for name, entry in manifest.items():
            print(f"  {name}: {entry['description'].strip().splitlines()[0]}")
        sys.exit(0)

    if args.all:
        for name in manifest:
            process_dataset(name, manifest, force=args.force)
        return

    if not args.dataset:
        parser.print_help()
        sys.exit(1)

    process_dataset(args.dataset, manifest, force=args.force)


if __name__ == "__main__":
    main()
