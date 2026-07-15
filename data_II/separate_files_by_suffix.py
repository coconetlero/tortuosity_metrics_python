"""
separate_files_by_suffix.py

Separates files in a folder into subfolders based on the last
underscore-separated part of their filename (before the extension).

Example:
    01_01_art_fp.txt  -> fp/01_01_art_fp.txt
    01_01_art_ir.txt  -> ir/01_01_art_ir.txt

Usage:
    python3 separate_files_by_suffix.py /path/to/folder
    python3 separate_files_by_suffix.py /path/to/folder --copy
    python3 separate_files_by_suffix.py /path/to/folder --dry-run
    python3 separate_files_by_suffix.py /path/to/folder --dest /path/to/output
"""

import argparse
import os
import shutil


def separate_files_by_suffix(source_dir, dest_root=None, move=True, dry_run=False):
    """
    Sorts files in source_dir into subfolders named after each file's
    suffix (the last '_'-separated token before the extension).

    Parameters:
        source_dir : str
            Folder containing the files to sort.
        dest_root : str, optional
            Folder under which the suffix subfolders are created.
            Defaults to source_dir itself if not given.
        move : bool
            If True (default), files are moved. If False, they are
            copied instead, leaving the originals in place.
        dry_run : bool
            If True, only prints what would happen without touching
            any files.

    Returns:
        dict
            Mapping of suffix -> list of filenames placed under it.
    """
    if dest_root is None:
        dest_root = source_dir

    results = {}

    for fname in sorted(os.listdir(source_dir)):
        src_path = os.path.join(source_dir, fname)
        if not os.path.isfile(src_path):
            continue

        name, ext = os.path.splitext(fname)
        parts = name.split('_')
        suffix = parts[-1] if len(parts) > 1 else "other"

        dest_dir = os.path.join(dest_root, suffix)
        dest_path = os.path.join(dest_dir, fname)

        action = "MOVE" if move else "COPY"
        print(f"[{action}] {fname} -> {suffix}/")

        if not dry_run:
            os.makedirs(dest_dir, exist_ok=True)
            if move:
                shutil.move(src_path, dest_path)
            else:
                shutil.copy2(src_path, dest_path)

        results.setdefault(suffix, []).append(fname)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Separate files into folders based on their filename suffix."
    )
    parser.add_argument("source_dir", help="Folder containing the files to sort")
    parser.add_argument(
        "--dest", default=None,
        help="Folder under which suffix subfolders are created (default: source_dir)"
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy files instead of moving them (originals are kept)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without moving/copying anything"
    )
    args = parser.parse_args()

    results = separate_files_by_suffix(
        source_dir=args.source_dir,
        dest_root=args.dest,
        move=not args.copy,
        dry_run=args.dry_run,
    )

    print()
    print("Summary:")
    for suffix, files in results.items():
        print(f"  {suffix}: {len(files)} file(s)")


if __name__ == "__main__":
    main()
