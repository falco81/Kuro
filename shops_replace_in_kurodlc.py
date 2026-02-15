#!/usr/bin/env python3
"""
shops_replace_in_kurodlc.py - v1.1

Replace shop IDs in .kurodlc.json files. Rebuilds the ShopItem section
with new shop IDs for all item_ids extracted from selected sections.

Supports both full DLC files (CostumeParam + DLCTableData + ItemTableData + ShopItem)
and shop-only files (ShopItem only).

Shop ID validation:
  If t_shop data is available (t_shop.json, t_shop.tbl, .p3a archives),
  the script validates entered shop IDs against the game's shop table and
  displays shop names for confirmation. If t_shop is not available or
  required libraries are missing, validation is silently skipped.

Usage:
  python shops_replace_in_kurodlc.py [file.kurodlc.json] [mode] [options]

  If no file specified, processes ALL .kurodlc.json files in current directory.

EXTRACTION MODES (which sections to get item_ids from):
  all         - Extract from all sections (default)
  shop        - Extract from ShopItem section only
  costume     - Extract from CostumeParam section only
  item        - Extract from ItemTableData section only
  dlc         - Extract from DLCTableData.items section only

  Combinations (use +):
    costume+item   - Extract from CostumeParam and ItemTableData
    shop+costume   - Extract from ShopItem and CostumeParam

Options:
  --new-shop-ids=1,2,3  New shop IDs to use (same for all files)
  --per-file            Prompt for new shop IDs individually per file
  --apply               Apply changes (without this, runs in dry-run mode)
  --dry-run             Explicit dry-run (default behavior)
  --no-backup           Skip backup creation when applying
  --no-interactive      Error out instead of prompting
  --no-ascii-escape     Write UTF-8 directly instead of \\uXXXX escaping
  --help                Show this help message

Examples:
  python shops_replace_in_kurodlc.py FalcoDLC.kurodlc.json --new-shop-ids=21,22,248,258
      Preview shop ID replacement for one file (dry-run).

  python shops_replace_in_kurodlc.py --new-shop-ids=21,22,248,258 --apply
      Replace shop IDs in ALL .kurodlc.json files in current directory.

  python shops_replace_in_kurodlc.py --per-file --apply
      Ask for new shop IDs for each file individually.

  python shops_replace_in_kurodlc.py FalcoDLC.kurodlc.json costume --new-shop-ids=21,22 --apply
      Replace using only item_ids from CostumeParam section.

  python shops_replace_in_kurodlc.py UMat.kurodlc.json shop --new-shop-ids=21,22,248,258
      Preview for shop-only file (extract IDs from ShopItem).
"""

import json
import sys
import os
import shutil
import datetime


# =========================================================================
# Optional t_shop validation (graceful fallback if libs/sources unavailable)
# =========================================================================

try:
    from p3a_lib import p3a_class
    from kurodlc_lib import kuro_tables
    HAS_KURO_LIBS = True
except ImportError:
    HAS_KURO_LIBS = False


def detect_tshop_sources(base_dir):
    """Detect available t_shop sources. Returns list of (type, path)."""
    sources = []
    candidates = [
        ('json',     't_shop.json'),
        ('original', 't_shop.tbl.original'),
        ('tbl',      't_shop.tbl'),
        ('p3a',      'script_en.p3a'),
        ('p3a',      'script_eng.p3a'),
        ('zzz',      'zzz_combined_tables.p3a'),
    ]
    for stype, fname in candidates:
        fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            sources.append((stype, fpath))
    return sources


def collect_shops_recursive(node, result_list):
    """Recursively find dicts with 'id' and 'shop_name' fields."""
    if isinstance(node, dict):
        if 'id' in node and 'shop_name' in node:
            result_list.append(node)
        for value in node.values():
            collect_shops_recursive(value, result_list)
    elif isinstance(node, list):
        for item in node:
            collect_shops_recursive(item, result_list)


def load_tshop_from_json(json_file):
    """Load t_shop from JSON using recursive search."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        shops = []
        collect_shops_recursive(data, shops)
        return shops
    except Exception:
        return None


def load_tshop_from_tbl(tbl_file):
    """Load t_shop from TBL file using recursive search."""
    if not HAS_KURO_LIBS:
        return None
    try:
        kt = kuro_tables()
        table = kt.read_table(tbl_file)
        shops = []
        collect_shops_recursive(table, shops)
        return shops
    except Exception:
        return None


def extract_tshop_from_p3a(p3a_file, out_file):
    """Extract t_shop.tbl from a P3A archive."""
    if not HAS_KURO_LIBS:
        return False
    try:
        p3a = p3a_class()
        with open(p3a_file, 'rb') as p3a.f:
            headers, entries, p3a_dict = p3a.read_p3a_toc()
            for entry in entries:
                if os.path.basename(entry['name']) == 't_shop.tbl':
                    data = p3a.read_file(entry, p3a_dict)
                    with open(out_file, 'wb') as f:
                        f.write(data)
                    return True
        return False
    except Exception:
        return False


def load_tshop_data(base_dir, no_interactive=False):
    """
    Try to load t_shop data. Returns {id: shop_name} dict or None.
    Silently returns None if no sources or libs available (fallback).
    """
    sources = detect_tshop_sources(base_dir)
    if not sources:
        return None

    # Filter: json works without libs, tbl/p3a need HAS_KURO_LIBS
    usable = []
    for stype, path in sources:
        if stype == 'json':
            usable.append((stype, path))
        elif HAS_KURO_LIBS:
            usable.append((stype, path))
    if not usable:
        return None

    # Select source
    if len(usable) == 1 or no_interactive:
        stype, path = usable[0]
        print(f"Loading t_shop from: {os.path.basename(path)}")
    else:
        print(f"\nMultiple t_shop sources detected. Select source:")
        for i, (st, p) in enumerate(usable, 1):
            basename = os.path.basename(p)
            if st in ('p3a', 'zzz'):
                print(f"  {i}) {basename} (extract t_shop.tbl)")
            else:
                print(f"  {i}) {basename}")
        # Show unusable sources (missing libs)
        for st, p in sources:
            if (st, p) not in usable:
                basename = os.path.basename(p)
                print(f"  -) {basename}  [requires p3a_lib/kurodlc_lib]")
        while True:
            try:
                choice = input(f"Enter choice [1-{len(usable)}]: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(usable):
                    stype, path = usable[int(choice) - 1]
                    break
                print("Invalid choice, try again.")
            except (EOFError, KeyboardInterrupt):
                return None
        print(f"Loading t_shop from: {os.path.basename(path)}")

    # Load
    shops_list = None
    if stype == 'json':
        shops_list = load_tshop_from_json(path)
    elif stype in ('tbl', 'original'):
        shops_list = load_tshop_from_tbl(path)
    elif stype in ('p3a', 'zzz'):
        temp_file = os.path.join(base_dir, 't_shop.tbl.tmp')
        try:
            if extract_tshop_from_p3a(path, temp_file):
                shops_list = load_tshop_from_tbl(temp_file)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    if not shops_list:
        return None

    # Build {id: shop_name} dict
    shop_dict = {}
    for shop in shops_list:
        if isinstance(shop, dict) and 'id' in shop and 'shop_name' in shop:
            shop_dict[int(shop['id'])] = shop['shop_name']

    if shop_dict:
        print(f"Loaded t_shop: {len(shop_dict)} shops")
    return shop_dict if shop_dict else None


def validate_shop_ids(shop_ids, shop_dict, no_interactive=False):
    """
    Validate shop IDs against t_shop data. Shows names, asks y/n.
    Returns: 'ok' to proceed, 'retry' to re-prompt, 'abort' to exit.
    If shop_dict is None, always returns 'ok' (fallback: no validation).
    """
    if shop_dict is None:
        return 'ok'

    print(f"\nShop ID validation (t_shop):")
    all_valid = True
    for sid in shop_ids:
        if sid in shop_dict:
            print(f"  {sid:>4} : {shop_dict[sid]}")
        else:
            print(f"  {sid:>4} : [NOT FOUND in t_shop]")
            all_valid = False

    if no_interactive:
        if not all_valid:
            print("Warning: Some shop IDs not found in t_shop. Proceeding (--no-interactive).")
        return 'ok'

    if all_valid:
        prompt = "All shop IDs valid. Proceed? [Y/n]: "
    else:
        prompt = "Warning: Some shop IDs not found. Proceed anyway? [y/N]: "

    try:
        answer = input(prompt).strip().lower()
        if all_valid:
            return 'ok' if answer != 'n' else 'retry'
        else:
            return 'ok' if answer == 'y' else 'retry'
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(1)


def search_tshop_interactive(shop_dict):
    """
    Interactive search mode for t_shop data.
    Supports same search syntax as find_all_shops.py:
      id:NUMBER   - exact ID match
      name:TEXT   - search in shop names (even if TEXT is a number)
      NUMBER      - auto: exact ID lookup + partial ID match
      TEXT        - auto: name search
    """
    print(f"\n  === Shop search ({len(shop_dict)} shops) ===")
    print(f"  id:N = exact ID | name:TEXT = name search | or just type")
    print(f"  Empty line returns to shop ID input.\n")

    while True:
        try:
            query = input("  search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not query:
            return

        results = []

        if query.startswith('id:'):
            # Explicit exact ID search
            id_str = query[3:].strip()
            if not id_str:
                print(f"  Usage: id:NUMBER (e.g. id:21)")
                print()
                continue
            try:
                sid = int(id_str)
                if sid in shop_dict:
                    results.append((sid, shop_dict[sid]))
            except ValueError:
                print(f"  Error: '{id_str}' is not a valid ID")
                print()
                continue

        elif query.startswith('name:'):
            # Explicit name search (even for numbers)
            name_str = query[5:].strip().lower()
            if not name_str:
                print(f"  Usage: name:TEXT (e.g. name:armor)")
                print()
                continue
            for shop_id, name in sorted(shop_dict.items()):
                if name_str in name.lower():
                    results.append((shop_id, name))

        elif query.isdigit():
            # Auto-detect: number → exact ID + partial ID match
            sid = int(query)
            if sid in shop_dict:
                results.append((sid, shop_dict[sid]))
            for shop_id, name in sorted(shop_dict.items()):
                if query in str(shop_id) and shop_id != sid:
                    results.append((shop_id, name))

        else:
            # Auto-detect: text → name search
            query_lower = query.lower()
            for shop_id, name in sorted(shop_dict.items()):
                if query_lower in name.lower():
                    results.append((shop_id, name))

        if not results:
            print(f"  No matches for '{query}'")
        else:
            max_id_len = max(len(str(r[0])) for r in results)
            for shop_id, name in results:
                print(f"  {shop_id:>{max_id_len}} : {name}")
            print(f"  ({len(results)} result(s))")
        print()


def prompt_shop_ids(prompt_text, shop_dict, current_ids=None):
    """
    Prompt user for shop IDs with optional ? search mode.

    Args:
        prompt_text: Prompt message
        shop_dict: Loaded t_shop dict or None
        current_ids: Current shop IDs to display for reference

    Returns:
        List of int shop IDs, or None if empty input (skip).
    """
    if current_ids:
        print(f"\nCurrent shop_ids: {current_ids}")

    search_hint = "  ? = search shops" if shop_dict else ""
    print(f"\n{prompt_text}")
    if search_hint:
        print(search_hint)

    while True:
        try:
            shop_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(1)

        if not shop_input:
            return None

        # Search mode
        if shop_input == '?' and shop_dict:
            search_tshop_interactive(shop_dict)
            print(prompt_text)
            if search_hint:
                print(search_hint)
            continue

        # Parse IDs
        try:
            ids = [int(x.strip()) for x in shop_input.split(",")]
            return ids
        except ValueError:
            print("Error: Invalid format. Enter comma-separated numbers (e.g. 21,22,248)")
            continue

def extract_ids_by_mode(data, modes):
    """
    Extract item IDs based on selected modes.

    Returns:
        (list of IDs, extraction summary lines)
    """
    all_ids = []
    summary = []

    if "shop" in modes and "ShopItem" in data:
        ids = [item["item_id"] for item in data["ShopItem"]
               if isinstance(item, dict) and "item_id" in item]
        all_ids.extend(ids)
        summary.append(f"ShopItem: {len(ids)} IDs")

    if "costume" in modes and "CostumeParam" in data:
        ids = [item["item_id"] for item in data["CostumeParam"]
               if isinstance(item, dict) and "item_id" in item]
        all_ids.extend(ids)
        summary.append(f"CostumeParam: {len(ids)} IDs")

    if "item" in modes and "ItemTableData" in data:
        ids = [item["id"] for item in data["ItemTableData"]
               if isinstance(item, dict) and "id" in item]
        all_ids.extend(ids)
        summary.append(f"ItemTableData: {len(ids)} IDs")

    if "dlc" in modes and "DLCTableData" in data:
        for dlc in data["DLCTableData"]:
            if isinstance(dlc, dict) and "items" in dlc and isinstance(dlc["items"], list):
                all_ids.extend(dlc["items"])
                summary.append(f"DLCTableData: {len(dlc['items'])} IDs")

    return all_ids, summary


def get_available_sections(data):
    """Detect which extractable sections are present."""
    section_map = {
        "shop": "ShopItem",
        "costume": "CostumeParam",
        "item": "ItemTableData",
        "dlc": "DLCTableData",
    }
    available = {}
    for mode, section in section_map.items():
        if section in data and isinstance(data[section], list) and len(data[section]) > 0:
            available[mode] = section
    return available


# =========================================================================
# Validation
# =========================================================================

def is_valid_kurodlc(data):
    """
    Validate .kurodlc.json structure.
    Accepts full DLC or shop-only variants.
    """
    if not isinstance(data, dict):
        return False

    known_sections = {"CostumeParam", "DLCTableData", "ItemTableData", "ShopItem"}
    if not any(k in data for k in known_sections):
        return False

    # Full DLC: has CostumeParam or DLCTableData -> both required
    has_full = "CostumeParam" in data or "DLCTableData" in data
    if has_full:
        if not ("CostumeParam" in data and "DLCTableData" in data):
            return False
        if not isinstance(data["CostumeParam"], list) or not isinstance(data["DLCTableData"], list):
            return False
        return True

    # Shop-only: just ShopItem
    if "ShopItem" in data:
        if not isinstance(data["ShopItem"], list) or len(data["ShopItem"]) == 0:
            return False
        return True

    return False


def get_all_kurodlc_files():
    """Find all .kurodlc.json files in current directory (skip .bak)."""
    files = []
    for name in sorted(os.listdir('.')):
        if not name.lower().endswith('.kurodlc.json'):
            continue
        if '.bak' in name.lower():
            continue
        if os.path.isfile(name):
            files.append(name)
    return files


# =========================================================================
# Shop replacement logic
# =========================================================================

def get_shop_template(data):
    """Get a ShopItem entry template from existing data."""
    if "ShopItem" in data and isinstance(data["ShopItem"], list):
        for entry in data["ShopItem"]:
            if isinstance(entry, dict) and "shop_id" in entry and "item_id" in entry:
                return entry
    return None


def build_new_shop_items(data, item_ids, new_shop_ids, template):
    """
    Build new ShopItem list: one entry per (shop_id, item_id) combination.
    Uses template for unknown fields structure.
    """
    new_entries = []

    # Default template if none found
    if template is None:
        template = {
            "shop_id": 0,
            "item_id": 0,
            "unknown": 1,
            "start_scena_flags": [],
            "empty1": 0,
            "end_scena_flags": [],
            "int2": 0
        }

    for item_id in item_ids:
        for shop_id in new_shop_ids:
            entry = {}
            for key, val in template.items():
                if key == "shop_id":
                    entry[key] = shop_id
                elif key == "item_id":
                    entry[key] = item_id
                elif isinstance(val, list):
                    entry[key] = list(val)  # copy
                else:
                    entry[key] = val
            new_entries.append(entry)

    return new_entries


def process_file(json_file, modes, new_shop_ids, apply_changes, do_backup,
                 ensure_ascii):
    """
    Process a single .kurodlc.json file.

    Returns:
        True if changes were made/would be made, False if nothing to do.
    """
    # Load
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  Error: Invalid JSON in '{json_file}': {e}")
        return False
    except Exception as e:
        print(f"  Error loading '{json_file}': {e}")
        return False

    if not is_valid_kurodlc(data):
        print(f"  Error: '{json_file}' is not a valid .kurodlc.json structure.")
        return False

    # Show available sections
    available = get_available_sections(data)
    sections_str = ", ".join(f"{k}({v})" for k, v in sorted(available.items()))
    print(f"  Sections: {sections_str}")

    # Get current shop IDs
    old_shop_ids = sorted(set(
        item["shop_id"] for item in data.get("ShopItem", [])
        if isinstance(item, dict) and "shop_id" in item
    ))
    if old_shop_ids:
        print(f"  Current shop_ids: {old_shop_ids}")

    # Extract item IDs from requested modes
    all_ids, summary = extract_ids_by_mode(data, modes)
    unique_ids = sorted(set(all_ids))

    if not unique_ids:
        available_modes = ", ".join(sorted(available.keys()))
        requested_modes = ", ".join(sorted(modes))
        print(f"  Warning: No item IDs found in requested sections: {requested_modes}")
        if available_modes:
            print(f"  Available sections: {available_modes}")
        return False

    for line in summary:
        print(f"  Extracted: {line}")
    print(f"  Unique item_ids: {len(unique_ids)}")

    # Get template
    template = get_shop_template(data)

    # Build new ShopItem entries
    new_entries = build_new_shop_items(data, unique_ids, new_shop_ids, template)

    old_count = len(data.get("ShopItem", []))
    new_count = len(new_entries)

    print(f"  New shop_ids: {new_shop_ids}")
    print(f"  ShopItem entries: {old_count} -> {new_count} "
          f"({len(unique_ids)} items x {len(new_shop_ids)} shops)")

    if old_shop_ids == sorted(new_shop_ids) and old_count == new_count:
        print(f"  [SKIP] Shop IDs already match, nothing to change.")
        return False

    if not apply_changes:
        print(f"  [DRY RUN] Would replace ShopItem section.")
        return True

    # Apply changes
    data["ShopItem"] = new_entries

    # Backup
    if do_backup:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = json_file + f'_{timestamp}.bak'
        try:
            shutil.copy2(json_file, backup_file)
            print(f"  Backup: {backup_file}")
        except Exception as e:
            print(f"  Error creating backup: {e}")
            return False

    # Write
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=ensure_ascii)
            f.write('\n')
        print(f"  [APPLIED] Written: {json_file}")
        return True
    except Exception as e:
        print(f"  Error writing '{json_file}': {e}")
        return False


# =========================================================================
# Main
# =========================================================================

def print_usage():
    print(__doc__)


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        print_usage()
        return

    # Parse arguments
    target_file = None
    mode_arg = "all"
    new_shop_ids = None
    apply_changes = False
    do_backup = True
    no_interactive = False
    ensure_ascii = True
    per_file = False

    positional = []
    for arg in sys.argv[1:]:
        if arg == '--apply':
            apply_changes = True
        elif arg == '--dry-run':
            apply_changes = False
        elif arg.startswith('--new-shop-ids='):
            try:
                new_shop_ids = [int(x.strip()) for x in arg.split('=', 1)[1].split(',')]
            except ValueError:
                print(f"Error: Invalid shop IDs in {arg}")
                sys.exit(1)
        elif arg == '--no-backup':
            do_backup = False
        elif arg == '--no-interactive':
            no_interactive = True
        elif arg == '--no-ascii-escape':
            ensure_ascii = False
        elif arg == '--per-file':
            per_file = True
        elif arg.startswith('--'):
            print(f"Error: Unknown option '{arg}'")
            sys.exit(1)
        else:
            positional.append(arg)

    # Parse positional args: [file] [mode]
    for p in positional:
        if p.lower().endswith('.kurodlc.json') or p.lower().endswith('.json'):
            target_file = p
        elif p in ('all', 'shop', 'costume', 'item', 'dlc') or '+' in p:
            mode_arg = p
        else:
            # Could be file or mode
            if os.path.exists(p):
                target_file = p
            else:
                mode_arg = p

    # Parse extraction modes
    valid_modes = {"shop", "costume", "item", "dlc", "all"}
    requested_modes = set()

    if mode_arg == "all":
        requested_modes = {"shop", "costume", "item", "dlc"}
    else:
        parts = mode_arg.split("+")
        for part in parts:
            part = part.strip()
            if part not in valid_modes or part == "all":
                print(f"Error: Unknown mode '{part}'.")
                print("Valid modes: shop, costume, item, dlc, all")
                print("Combine with +: costume+item, shop+costume")
                sys.exit(1)
            requested_modes.add(part)

    # Determine target files
    if target_file:
        if not os.path.exists(target_file):
            print(f"Error: File '{target_file}' not found.")
            sys.exit(1)
        files = [target_file]
    else:
        files = get_all_kurodlc_files()
        if not files:
            print("Error: No .kurodlc.json files found in current directory.")
            sys.exit(1)
        print(f"Found {len(files)} .kurodlc.json file(s):")
        for f in files:
            print(f"  {f}")

    # Determine base directory for t_shop loading
    if target_file:
        base_dir = os.path.dirname(os.path.abspath(target_file)) or '.'
    else:
        base_dir = '.'

    # Try loading t_shop for validation (graceful fallback if unavailable)
    shop_dict = load_tshop_data(base_dir, no_interactive)

    # Get new shop IDs (unless --per-file, which prompts per file)
    if not per_file:
        if new_shop_ids is None:
            if no_interactive:
                print("Error: --new-shop-ids is required with --no-interactive.")
                print("Example: --new-shop-ids=21,22,248,258")
                sys.exit(1)

            # Show current shop IDs from first file for reference
            current = None
            try:
                with open(files[0], 'r', encoding='utf-8') as f:
                    ref_data = json.load(f)
                current = sorted(set(
                    item["shop_id"] for item in ref_data.get("ShopItem", [])
                    if isinstance(item, dict) and "shop_id" in item
                )) or None
            except Exception:
                pass

            # Prompt + validate loop (retry on declined validation)
            while True:
                new_shop_ids = prompt_shop_ids(
                    "Enter new shop IDs (comma-separated, e.g. 21,22,248,258):",
                    shop_dict, current_ids=current)
                if not new_shop_ids:
                    print("Error: No shop IDs provided.")
                    sys.exit(1)

                result = validate_shop_ids(new_shop_ids, shop_dict, no_interactive)
                if result == 'ok':
                    break
                # 'retry': loop back to prompt
        else:
            # --new-shop-ids from command line: validate once, abort on decline
            if not new_shop_ids:
                print("Error: Empty shop ID list.")
                sys.exit(1)
            result = validate_shop_ids(new_shop_ids, shop_dict, no_interactive)
            if result == 'retry':
                print("Aborted.")
                sys.exit(0)

    # Process files
    modes_str = "+".join(sorted(requested_modes))
    print(f"\nExtraction mode: {modes_str}")
    if per_file:
        print(f"Shop IDs: per-file interactive")
    else:
        print(f"New shop_ids: {new_shop_ids}")
    if not apply_changes:
        print("[DRY RUN MODE]")
    print(f"\n{'='*60}")

    changed = 0
    for json_file in files:
        print(f"\nProcessing: {json_file}")

        # Per-file shop ID prompting
        file_shop_ids = new_shop_ids
        if per_file and new_shop_ids is None:
            # Show current shop IDs for this file
            current = None
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    peek_data = json.load(f)
                current = sorted(set(
                    item["shop_id"] for item in peek_data.get("ShopItem", [])
                    if isinstance(item, dict) and "shop_id" in item
                )) or None
            except Exception:
                pass

            # Prompt + validate loop (retry on declined validation, skip on empty)
            while True:
                file_shop_ids = prompt_shop_ids(
                    f"Enter new shop IDs for this file (comma-separated, or Enter to skip):",
                    shop_dict, current_ids=current)
                if not file_shop_ids:
                    print(f"  [SKIP] No shop IDs provided, skipping file.")
                    break

                result = validate_shop_ids(file_shop_ids, shop_dict, no_interactive)
                if result == 'ok':
                    break
                # 'retry': loop back to prompt for this file

            if not file_shop_ids:
                continue

        if process_file(json_file, requested_modes, file_shop_ids, apply_changes,
                        do_backup, ensure_ascii):
            changed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Files processed: {len(files)}")
    print(f"Files {'changed' if apply_changes else 'would change'}: {changed}")
    if not apply_changes and changed > 0:
        print(f"\nUse --apply to write changes.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
