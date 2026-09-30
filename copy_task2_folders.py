#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy folders listed in task2_id.txt to another directory."
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="Directory where the selected folders will be copied",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=Path("task2_id.txt"),
        help="Text file containing one folder ID per line",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset/1219p/glcm/kernel5/non_tumor"),
        help="Directory containing the source folders",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the copy operations without copying files",
    )
    args = parser.parse_args()

    if not args.ids_file.is_file():
        parser.error(f"IDs file does not exist: {args.ids_file}")
    if not args.source.is_dir():
        parser.error(f"Source directory does not exist: {args.source}")

    folder_ids = [
        line.strip()
        for line in args.ids_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not args.dry_run:
        args.destination.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    for folder_id in folder_ids:
        source_folder = args.source / folder_id
        destination_folder = args.destination / folder_id

        if not source_folder.is_dir():
            print(f"MISSING {source_folder}")
            missing += 1
            continue

        print(f"COPY {source_folder} -> {destination_folder}")
        if not args.dry_run:
            shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)
        copied += 1

    action = "Would copy" if args.dry_run else "Copied"
    print(f"{action}: {copied}; missing: {missing}; total IDs: {len(folder_ids)}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
