#!/usr/bin/env python3
"""Normalize NOUS campaign folder structures.

This script finds actual NOUS campaign directories (identified by state.json)
and restructures them so the campaign content is at the top level of each
campaign folder. Non-NOUS files (.boxnote, extra .yaml files, etc.) are removed.

Usage:
    # Dry run (preview changes)
    python normalize_campaigns.py /path/to/campaigns_directory

    # Apply changes
    python normalize_campaigns.py /path/to/campaigns_directory --apply

    # Keep non-NOUS files in a .removed subdirectory instead of deleting
    python normalize_campaigns.py /path/to/campaigns_directory --apply --keep-removed
"""

import argparse
import shutil
from pathlib import Path

NOUS_CAMPAIGN_FILES = {
    "state.json",
    "principles.json",
    "ledger.json",
    "handoff.md",
    "campaign.yaml",
    "retry_log.jsonl",
    "llm_metrics.jsonl",
    "llm_metrics_summary.json",
    "report.md",
}

NOUS_CAMPAIGN_DIRS = {
    "runs",
}

IGNORED_EXTENSIONS = {
    ".boxnote",
    ".zip",
    ".tar",
    ".gz",
    ".DS_Store",
}


def is_nous_campaign_dir(path: Path) -> bool:
    """Check if a directory is a NOUS campaign (has state.json)."""
    return (path / "state.json").exists()


def find_nous_campaign(top_dir: Path) -> Path | None:
    """Find the actual NOUS campaign directory within a top-level folder.

    Returns the path to the directory containing state.json, or None if not found.
    """
    if is_nous_campaign_dir(top_dir):
        return top_dir

    for item in top_dir.iterdir():
        if item.is_dir():
            if is_nous_campaign_dir(item):
                return item
            nested = find_nous_campaign(item)
            if nested:
                return nested

    return None


def is_nous_file(path: Path) -> bool:
    """Check if a file/directory is part of NOUS output."""
    name = path.name

    if path.is_dir():
        return name in NOUS_CAMPAIGN_DIRS or name.startswith("iter-")

    if name in NOUS_CAMPAIGN_FILES:
        return True

    if path.suffix in IGNORED_EXTENSIONS:
        return False

    return False


def get_files_to_remove(campaign_dir: Path, nous_dir: Path) -> list[Path]:
    """Get list of files that are not part of NOUS output."""
    to_remove = []

    for item in campaign_dir.iterdir():
        if item == nous_dir:
            continue

        if item.suffix in IGNORED_EXTENSIONS:
            to_remove.append(item)
            continue

        if item.is_file() and item.name not in NOUS_CAMPAIGN_FILES:
            to_remove.append(item)
        elif item.is_dir() and item.name not in NOUS_CAMPAIGN_DIRS:
            if not is_nous_campaign_dir(item):
                to_remove.append(item)

    return to_remove


def normalize_campaign(
    campaign_dir: Path,
    apply: bool = False,
    keep_removed: bool = False,
) -> dict:
    """Normalize a single campaign directory.

    Returns a dict with:
        - 'status': 'already_normal', 'normalized', 'no_campaign', or 'error'
        - 'nous_dir': path to the NOUS campaign (if found)
        - 'moved': list of moved items
        - 'removed': list of removed items
        - 'error': error message (if any)
    """
    result = {
        "status": "unknown",
        "nous_dir": None,
        "moved": [],
        "removed": [],
        "error": None,
    }

    nous_dir = find_nous_campaign(campaign_dir)

    if not nous_dir:
        result["status"] = "no_campaign"
        return result

    result["nous_dir"] = nous_dir

    if nous_dir == campaign_dir:
        to_remove = []
        for item in campaign_dir.iterdir():
            if item.suffix in IGNORED_EXTENSIONS:
                to_remove.append(item)
            elif item.is_file() and item.name not in NOUS_CAMPAIGN_FILES:
                to_remove.append(item)
            elif item.is_dir() and item.name not in NOUS_CAMPAIGN_DIRS:
                to_remove.append(item)

        if not to_remove:
            result["status"] = "already_normal"
            return result

        result["removed"] = to_remove

        if apply:
            for item in to_remove:
                if keep_removed:
                    removed_dir = campaign_dir / ".removed"
                    removed_dir.mkdir(exist_ok=True)
                    dest = removed_dir / item.name
                    shutil.move(str(item), str(dest))
                else:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()

        result["status"] = "normalized"
        return result

    items_to_move = []
    for item in nous_dir.iterdir():
        items_to_move.append(item)

    to_remove = get_files_to_remove(campaign_dir, nous_dir)

    result["moved"] = items_to_move
    result["removed"] = to_remove

    if apply:
        for item in to_remove:
            if keep_removed:
                removed_dir = campaign_dir / ".removed"
                removed_dir.mkdir(exist_ok=True)
                dest = removed_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
            else:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

        for item in items_to_move:
            dest = campaign_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))

        if nous_dir.exists() and nous_dir != campaign_dir:
            try:
                nous_dir.rmdir()
            except OSError:
                pass

            parent = nous_dir.parent
            while parent != campaign_dir:
                try:
                    parent.rmdir()
                    parent = parent.parent
                except OSError:
                    break

    result["status"] = "normalized"
    return result


def normalize_all_campaigns(
    campaigns_dir: Path,
    apply: bool = False,
    keep_removed: bool = False,
) -> None:
    """Normalize all campaign directories."""
    if not campaigns_dir.is_dir():
        print(f"Error: {campaigns_dir} is not a directory")
        return

    print(f"{'Applying' if apply else 'Preview'} normalization for: {campaigns_dir}\n")

    stats = {"already_normal": 0, "normalized": 0, "no_campaign": 0, "error": 0}

    for item in sorted(campaigns_dir.iterdir()):
        if not item.is_dir():
            continue

        if item.name.startswith("."):
            continue

        print(f"Campaign: {item.name}")

        result = normalize_campaign(item, apply=apply, keep_removed=keep_removed)
        stats[result["status"]] += 1

        if result["status"] == "no_campaign":
            print(f"  No NOUS campaign found (no state.json)\n")
            continue

        if result["status"] == "already_normal":
            print(f"  Already normalized\n")
            continue

        if result["status"] == "error":
            print(f"  Error: {result['error']}\n")
            continue

        if result["nous_dir"] and result["nous_dir"] != item:
            print(f"  NOUS data found in: {result['nous_dir'].relative_to(item)}")

        if result["moved"]:
            print(f"  Moving to top level:")
            for moved in result["moved"]:
                print(f"    - {moved.name}")

        if result["removed"]:
            action = "Moving to .removed" if keep_removed else "Removing"
            print(f"  {action}:")
            for removed in result["removed"]:
                print(f"    - {removed.name}")

        print()

    print("=" * 50)
    print(f"Summary:")
    print(f"  Already normal: {stats['already_normal']}")
    print(f"  Normalized:     {stats['normalized']}")
    print(f"  No campaign:    {stats['no_campaign']}")
    print(f"  Errors:         {stats['error']}")

    if not apply and stats["normalized"] > 0:
        print(f"\nRun with --apply to make these changes.")


def main():
    parser = argparse.ArgumentParser(
        description="Normalize NOUS campaign folder structures."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to campaigns directory",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default is dry run)",
    )
    parser.add_argument(
        "--keep-removed",
        action="store_true",
        help="Keep removed files in .removed subdirectory instead of deleting",
    )

    args = parser.parse_args()
    normalize_all_campaigns(args.path, apply=args.apply, keep_removed=args.keep_removed)


if __name__ == "__main__":
    main()
