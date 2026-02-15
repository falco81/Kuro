#!/usr/bin/env python3
"""
fix_subcategory.py

Scans all .kurodlc.json files in current directory and fixes
ItemTableData entries where category=17 and subcategory=15 → subcategory=16.

Usage:
  python fix_subcategory.py           # dry-run (preview)
  python fix_subcategory.py --apply   # apply changes
"""

import json, os, sys, glob, shutil, datetime


def main():
    apply = '--apply' in sys.argv

    files = sorted(glob.glob('*.kurodlc.json'))
    if not files:
        print("No .kurodlc.json files found in current directory.")
        return

    total_fixed = 0
    files_changed = 0

    for fname in files:
        try:
            with open(fname, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  Error reading {fname}: {e}")
            continue

        if 'ItemTableData' not in data or not data['ItemTableData']:
            continue

        count = 0
        for entry in data['ItemTableData']:
            if (isinstance(entry, dict)
                    and entry.get('category') == 17
                    and entry.get('subcategory') == 15):
                if apply:
                    entry['subcategory'] = 16
                count += 1

        if count > 0:
            print(f"  {fname}: {count} entry(s) with category=17, subcategory=15→16")
            total_fixed += count
            files_changed += 1

            if apply:
                backup = fname + '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.bak'
                shutil.copy2(fname, backup)
                with open(fname, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=True)

    print(f"\nTotal: {total_fixed} entries in {files_changed} file(s)")
    if not apply and total_fixed > 0:
        print("[DRY RUN] No files modified. Use --apply to write changes.")
    elif apply and total_fixed > 0:
        print("All changes applied (backups created).")


if __name__ == "__main__":
    main()
