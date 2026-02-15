#!/usr/bin/env python3
"""
kurodlc_add_mdl.py - v2.2

Scan directory for .mdl files not present in a .kurodlc.json file and add
complete entries (CostumeParam, ItemTableData, DLCTableData, ShopItem) for each.

If the target .kurodlc.json file does not exist, creates it from scratch.
When creating a new DLCTableData record, uses t_dlc data for DLC ID assignment.

Uses t_name data for character identification (char_restrict and character names).
Uses t_item data + existing .kurodlc.json files for smart ID assignment.
Uses t_dlc data (optional) for DLC ID validation and available ID suggestion.

NEW in v2.0: Smart ID Assignment
- Searches for free IDs in range 1-5000 (safe game limit)
- Collects used IDs from t_item (game data) + all .kurodlc.json files
- Tries continuous block first, falls back to scattered search
- Clear error if not enough IDs available

NEW in v2.1: Safe defaults
- Default mode is dry-run (preview only), --apply required to write
- Backup files include timestamp: _YYYYMMDD_HHMMSS.bak

Usage:
  python kurodlc_add_mdl.py <file.kurodlc.json> [options]

Requirements (in the same directory):
  - .mdl files to add
  - t_name source (t_name.json, t_name.tbl, or P3A archive)
  - t_item source (t_item.json, t_item.tbl, or P3A archive)
"""

import json
import sys
import os
import re
import copy
import glob
from typing import Dict, List, Optional, Set, Tuple, Any

# -------------------------
# Import required libraries with error handling
# -------------------------
try:
    from p3a_lib import p3a_class
    from kurodlc_lib import kuro_tables
    HAS_LIBS = True
except ImportError as e:
    HAS_LIBS = False
    MISSING_LIB = str(e)


# =========================================================================
# Generic table loading (pattern from resolve_id_conflicts_in_kurodlc.py)
# Supports: t_name, t_item, or any t_*.tbl / t_*.json
# Sources: .json, .tbl.original, .tbl, script_en.p3a, script_eng.p3a,
#          zzz_combined_tables.p3a
# =========================================================================

def detect_all_sources(base_dir, required_prefixes):
    """
    Detect available sources and validate that ALL required table files exist.
    
    For json/tbl/original: each prefix needs its own file (t_name.json, t_item.json)
    For p3a/zzz: single archive contains all tables, no per-file check needed.
    
    Args:
        base_dir: Directory to search in
        required_prefixes: List of table prefixes, e.g. ['t_name', 't_item']
    
    Returns:
        List of (source_type, path, is_valid, missing) tuples.
        is_valid: True if all required files exist for this source type
        missing: List of missing file names (empty if valid)
    """
    results = []
    seen = set()  # Avoid duplicate entries for same path

    # Check table-specific sources (json, original, tbl)
    for stype, ext in [('json', '.json'), ('original', '.tbl.original'), ('tbl', '.tbl')]:
        # Check if at least one file of this type exists
        existing = []
        missing = []
        for prefix in required_prefixes:
            fname = f'{prefix}{ext}'
            fpath = os.path.join(base_dir, fname)
            if os.path.exists(fpath):
                existing.append(fname)
            else:
                missing.append(fname)

        # Only show this source type if at least one file exists
        if existing:
            # Use first existing file as representative path
            rep_path = os.path.join(base_dir, existing[0])
            is_valid = len(missing) == 0
            results.append((stype, rep_path, is_valid, missing))
            seen.add(stype)

    # Check P3A/ZZZ archives (contain all tables)
    archives = [
        ('p3a', 'script_en.p3a'),
        ('p3a', 'script_eng.p3a'),
        ('zzz', 'zzz_combined_tables.p3a'),
    ]
    for stype, fname in archives:
        fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            results.append((stype, fpath, True, []))

    return results


def select_source_interactive(sources):
    """
    Interactive CLI menu for source selection.
    Shows validity status for each source. Invalid sources cannot be selected.
    """
    print(f"\nMultiple data sources detected. Select source:")
    valid_choices = []
    for i, (stype, path, is_valid, missing) in enumerate(sources, 1):
        basename = os.path.basename(path)
        if stype in ('p3a', 'zzz'):
            label = f"{basename} (P3A archive)"
        elif stype == 'json':
            label = f"*.json files"
        elif stype == 'original':
            label = f"*.tbl.original files"
        elif stype == 'tbl':
            label = f"*.tbl files"
        else:
            label = basename

        if is_valid:
            print(f"  {i}) {label}")
            valid_choices.append(i)
        else:
            missing_str = ', '.join(missing)
            print(f"  {i}) {label}  [ERROR: missing {missing_str}]")

    if not valid_choices:
        print("\nError: No valid source available (all have missing files).")
        sys.exit(1)

    while True:
        choice = input(f"\nEnter choice [{valid_choices[0]}-{valid_choices[-1]}]: ").strip()
        if choice.isdigit():
            idx = int(choice)
            if idx in valid_choices:
                return sources[idx - 1]
            elif 1 <= idx <= len(sources):
                stype, path, is_valid, missing = sources[idx - 1]
                missing_str = ', '.join(missing)
                print(f"Cannot use this source - missing: {missing_str}")
                continue
        print("Invalid choice, try again.")


def detect_sources(base_dir, table_prefix):
    """
    Detect available sources for a single table file.
    (Used internally by load_table_data for the actual loading step.)
    
    Returns:
        List of (source_type, full_path) tuples.
    """
    sources = []
    candidates = [
        ('json',     f'{table_prefix}.json'),
        ('original', f'{table_prefix}.tbl.original'),
        ('tbl',      f'{table_prefix}.tbl'),
        ('p3a',      'script_en.p3a'),
        ('p3a',      'script_eng.p3a'),
        ('zzz',      'zzz_combined_tables.p3a'),
    ]
    for stype, fname in candidates:
        fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            sources.append((stype, fpath))
    return sources


def extract_from_p3a(p3a_file, tbl_name, out_file):
    """
    Extract a .tbl file from a P3A archive and write to out_file.
    
    Args:
        p3a_file: Path to .p3a archive
        tbl_name: Name of TBL file to extract (e.g. 't_name.tbl', 't_item.tbl')
        out_file: Path to write extracted data
    
    Returns:
        True on success, False on failure
    """
    if not HAS_LIBS:
        print(f"Error: Required library missing: {MISSING_LIB}")
        print("P3A extraction requires p3a_lib module.")
        return False
    try:
        p3a = p3a_class()
        with open(p3a_file, 'rb') as p3a.f:
            headers, entries, p3a_dict = p3a.read_p3a_toc()
            for entry in entries:
                if os.path.basename(entry['name']) == tbl_name:
                    data = p3a.read_file(entry, p3a_dict)
                    with open(out_file, 'wb') as f:
                        f.write(data)
                    return True
        print(f"Error: {tbl_name} not found in {os.path.basename(p3a_file)}")
        return False
    except Exception as e:
        print(f"Error extracting from P3A: {e}")
        return False


def load_from_json(json_file, section_name):
    """
    Load table data from a JSON file.
    
    Supports two JSON structures:
      1) {"data": [{"name": "SectionName", "data": [...]}]}
      2) {"SectionName": [...]}
    
    Args:
        json_file: Path to JSON file
        section_name: e.g. 'NameTableData', 'ItemTableData'
    
    Returns:
        List of entry dicts, or None on error
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Structure 1: {"data": [{"name": "...", "data": [...]}]}
            if "data" in data and isinstance(data["data"], list):
                for section in data["data"]:
                    if section.get("name") == section_name:
                        return section.get("data", [])
                # Section not found in data array
                print(f"Warning: {section_name} not found in {os.path.basename(json_file)}")
                return None

            # Structure 2: {"SectionName": [...]}
            if section_name in data:
                return data[section_name]

        print(f"Warning: Unexpected JSON structure in {os.path.basename(json_file)}")
        return None
    except Exception as e:
        print(f"Error loading {os.path.basename(json_file)}: {e}")
        return None


def load_from_tbl(tbl_file, section_name):
    """
    Load table data from a TBL file using kuro_tables.
    
    Args:
        tbl_file: Path to .tbl file
        section_name: e.g. 'NameTableData', 'ItemTableData'
    
    Returns:
        List of entry dicts, or None on error
    """
    if not HAS_LIBS:
        print(f"Error: Required library missing: {MISSING_LIB}")
        return None
    try:
        kt = kuro_tables()
        table = kt.read_table(tbl_file)
        if isinstance(table, dict) and section_name in table:
            return table[section_name]
        print(f"Error: {section_name} not found in {os.path.basename(tbl_file)}")
        return None
    except Exception as e:
        print(f"Error loading {os.path.basename(tbl_file)}: {e}")
        return None


def load_table_data(base_dir, table_prefix, section_name, source_type, source_path):
    """
    Load table data from a specific source.
    
    Args:
        base_dir: Directory containing source files
        table_prefix: e.g. 't_name' or 't_item'
        section_name: e.g. 'NameTableData' or 'ItemTableData'
        source_type: 'json', 'original', 'tbl', 'p3a', 'zzz'
        source_path: Path to source file (for p3a/zzz: archive path,
                     for json/tbl/original: ignored, constructed from prefix)
    
    Returns:
        List of entry dicts, or None on error
    """
    tbl_name = f'{table_prefix}.tbl'

    # JSON: load directly
    if source_type == 'json':
        json_path = os.path.join(base_dir, f'{table_prefix}.json')
        print(f"Loading {table_prefix} from: {os.path.basename(json_path)}")
        return load_from_json(json_path, section_name)

    # TBL / TBL.original: load via kuro_tables
    if source_type in ('tbl', 'original'):
        ext = '.tbl.original' if source_type == 'original' else '.tbl'
        tbl_path = os.path.join(base_dir, f'{table_prefix}{ext}')
        print(f"Loading {table_prefix} from: {os.path.basename(tbl_path)}")
        return load_from_tbl(tbl_path, section_name)

    # P3A / ZZZ: extract to temp, load, cleanup
    if source_type in ('p3a', 'zzz'):
        print(f"Loading {table_prefix} from: {os.path.basename(source_path)}")
        temp_file = os.path.join(base_dir, f'{table_prefix}.tbl.original.tmp')
        if not extract_from_p3a(source_path, tbl_name, temp_file):
            return None
        try:
            result = load_from_tbl(temp_file, section_name)
            return result
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    print(f"Error: Unknown source type '{source_type}'")
    return None


# =========================================================================
# Character mapping
# =========================================================================

def extract_chr_prefix(mdl_name):
    """
    Extract chrXXXX prefix from an mdl name.

    Handles patterns like:
      chr5001_c02aa  -> chr5001
      q_chr5001_c56q -> chr5001
      chr5302_c210   -> chr5302
    """
    match = re.search(r'(chr\d+)', mdl_name, re.IGNORECASE)
    return match.group(1).lower() if match else None


def build_char_map(names_data):
    """
    Build mapping: chr_prefix -> {char_id, name}

    Handles both 'char_id' (Kuro 1/2) and 'character_id' (Kai) field names.
    """
    char_map = {}
    if not names_data:
        return char_map

    for entry in names_data:
        if not isinstance(entry, dict):
            continue

        # Get character ID (try both field names)
        char_id = entry.get('char_id', entry.get('character_id'))
        name = entry.get('name', '')
        model = entry.get('model', '')

        if char_id is None or not model:
            continue

        # Extract chr prefix from model field
        prefix = extract_chr_prefix(model)
        if prefix and prefix not in char_map:
            char_map[prefix] = {
                'char_id': char_id,
                'name': name,
            }

    return char_map


# =========================================================================
# Smart ID Assignment Algorithm (from resolve_id_conflicts_in_kurodlc.py)
# =========================================================================

def find_available_ids_in_range(used_ids, count_needed, min_id=1, max_id=5000):
    """
    Find available IDs within specified range using smart search strategy.
    
    Algorithm:
    1. Try to find continuous block first (fast path)
    2. Fall back to scattered search if needed (handles fragmentation)
    3. Start from middle of range for better distribution
    
    Args:
        used_ids: Set of already used IDs
        count_needed: Number of available IDs needed
        min_id: Minimum ID value (inclusive, default=1)
        max_id: Maximum ID value (inclusive, default=5000)
    
    Returns:
        List of available IDs (sorted)
        
    Raises:
        ValueError: If not enough IDs available in range
    """
    if count_needed == 0:
        return []

    total_range = max_id - min_id + 1
    if count_needed > total_range:
        raise ValueError(f"Requested {count_needed} IDs but range only has {total_range} total")

    # Strategy 1: Try to find continuous block first (fast path)
    available = find_continuous_block(used_ids, count_needed, min_id, max_id)
    if available:
        return available

    # Strategy 2: Find scattered IDs (handles fragmented space)
    available = find_scattered_ids(used_ids, count_needed, min_id, max_id)
    if available:
        return available

    # Not enough IDs available
    used_in_range = len([id for id in used_ids if min_id <= id <= max_id])
    available_count = total_range - used_in_range
    raise ValueError(
        f"Not enough available IDs in range [{min_id}, {max_id}].\n"
        f"      Requested: {count_needed}\n"
        f"      Available: {available_count}\n"
        f"      Used in range: {used_in_range}\n"
        f"      Suggestion: Increase --max-id (e.g., --max-id={max_id + 5000})"
    )


def find_continuous_block(used_ids, count_needed, min_id, max_id):
    """
    Try to find a continuous block of available IDs.
    Starts from middle of range and searches outward.
    """
    middle = (min_id + max_id) // 2
    max_offset = max(middle - min_id, max_id - middle)

    for offset in range(max_offset + 1):
        start = middle + offset
        if start + count_needed - 1 <= max_id:
            if all(id not in used_ids for id in range(start, start + count_needed)):
                return list(range(start, start + count_needed))

        if offset > 0:
            start = middle - offset
            if start >= min_id and start + count_needed - 1 <= max_id:
                if all(id not in used_ids for id in range(start, start + count_needed)):
                    return list(range(start, start + count_needed))

    return None


def find_scattered_ids(used_ids, count_needed, min_id, max_id):
    """
    Find scattered available IDs throughout the range.
    Uses middle-out search for better distribution.
    """
    available = []
    middle = (min_id + max_id) // 2
    max_offset = max(middle - min_id, max_id - middle)

    for offset in range(max_offset + 1):
        candidate = middle + offset
        if min_id <= candidate <= max_id and candidate not in used_ids:
            available.append(candidate)
            if len(available) >= count_needed:
                return sorted(available)

        if offset > 0:
            candidate = middle - offset
            if min_id <= candidate <= max_id and candidate not in used_ids:
                available.append(candidate)
                if len(available) >= count_needed:
                    return sorted(available)

    return None


# =========================================================================
# Collect used IDs from all kurodlc files
# =========================================================================

def extract_item_ids_from_kurodlc(json_file):
    """Extract all item_ids from a .kurodlc.json file (all sections)."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return []

    ids = []
    if 'CostumeParam' in data:
        for item in data['CostumeParam']:
            if isinstance(item, dict) and 'item_id' in item:
                ids.append(item['item_id'])
    if 'ItemTableData' in data:
        for item in data['ItemTableData']:
            if isinstance(item, dict) and 'id' in item:
                ids.append(item['id'])
    if 'DLCTableData' in data:
        for dlc in data['DLCTableData']:
            if isinstance(dlc, dict) and 'items' in dlc and isinstance(dlc['items'], list):
                ids.extend(dlc['items'])
    if 'ShopItem' in data:
        for item in data['ShopItem']:
            if isinstance(item, dict) and 'item_id' in item:
                ids.append(item['item_id'])
    return ids


def collect_all_used_ids(base_dir, items_dict, exclude_file=None):
    """
    Build complete set of used IDs from:
    1. t_item game data (items_dict)
    2. All .kurodlc.json files in the directory
    
    Args:
        base_dir: Directory to scan
        items_dict: Dict from t_item {id: name}
        exclude_file: Optionally exclude this file (the target file being modified)
    """
    used_ids = set(items_dict.keys())

    for fname in os.listdir(base_dir):
        if not fname.lower().endswith('.kurodlc.json'):
            continue
        if '.bak' in fname.lower():
            continue
        fpath = os.path.join(base_dir, fname)
        if not os.path.isfile(fpath):
            continue
        # Don't exclude the target file - its existing IDs ARE used
        ids = extract_item_ids_from_kurodlc(fpath)
        used_ids.update(ids)

    return used_ids


# =========================================================================
# DLC ID management (t_dlc)
# =========================================================================

DLC_ID_MIN = 1
DLC_ID_MAX = 350


def load_dlc_ids_from_source(base_dir, source_type, source_path):
    """
    Load used DLC IDs from a t_dlc source.
    Returns dict {dlc_id: dlc_name} or None on failure.
    Handles both JSON generic fields (int1/text1) and named fields (id/name).
    """
    # Determine section name (Kuro uses DLCTableData, Ys X uses DLCTable)
    section_names = ['DLCTableData', 'DLCTable']

    raw_data = None
    section_used = None

    if source_type == 'json':
        json_path = os.path.join(base_dir, 't_dlc.json')
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                jdata = json.load(f)
            if isinstance(jdata, dict):
                # Structure: {"data": [{"name": "DLCTableData", "data": [...]}]}
                if "data" in jdata and isinstance(jdata["data"], list):
                    for section in jdata["data"]:
                        if section.get("name") in section_names:
                            raw_data = section.get("data", [])
                            section_used = section.get("name")
                            break
                # Structure: {"DLCTableData": [...]}
                if raw_data is None:
                    for sn in section_names:
                        if sn in jdata:
                            raw_data = jdata[sn]
                            section_used = sn
                            break
        except Exception:
            return None

    elif source_type in ('tbl', 'original'):
        if not HAS_LIBS:
            return None
        ext = '.tbl.original' if source_type == 'original' else '.tbl'
        tbl_path = os.path.join(base_dir, f't_dlc{ext}')
        try:
            kt = kuro_tables()
            table = kt.read_table(tbl_path)
            if isinstance(table, dict):
                for sn in section_names:
                    if sn in table:
                        raw_data = table[sn]
                        section_used = sn
                        break
        except Exception:
            return None

    elif source_type in ('p3a', 'zzz'):
        if not HAS_LIBS:
            return None
        temp_file = os.path.join(base_dir, 't_dlc.tbl.tmp')
        try:
            if not extract_from_p3a(source_path, 't_dlc.tbl', temp_file):
                return None
            kt = kuro_tables()
            table = kt.read_table(temp_file)
            if isinstance(table, dict):
                for sn in section_names:
                    if sn in table:
                        raw_data = table[sn]
                        section_used = sn
                        break
        except Exception:
            return None
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    if not raw_data:
        return None

    # Build {id: name} dict handling both field naming conventions
    dlc_dict = {}
    for entry in raw_data:
        if not isinstance(entry, dict):
            continue
        # Named fields (from tbl via kurodlc_lib)
        if 'id' in entry:
            dlc_id = entry['id']
            dlc_name = entry.get('name', entry.get('text1', f'DLC #{dlc_id}'))
        # Generic fields (from JSON)
        elif 'int1' in entry:
            dlc_id = entry['int1']
            dlc_name = entry.get('text1', f'DLC #{dlc_id}')
        else:
            continue
        dlc_dict[int(dlc_id)] = dlc_name

    return dlc_dict


def detect_t_dlc_sources(base_dir):
    """Detect available t_dlc sources. Returns list of (type, path)."""
    sources = []
    candidates = [
        ('json',     't_dlc.json'),
        ('original', 't_dlc.tbl.original'),
        ('tbl',      't_dlc.tbl'),
        ('p3a',      'script_en.p3a'),
        ('p3a',      'script_eng.p3a'),
        ('zzz',      'zzz_combined_tables.p3a'),
    ]
    for stype, fname in candidates:
        fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            sources.append((stype, fpath))
    return sources


def load_t_dlc_data(base_dir, source_type=None, source_path=None, no_interactive=False):
    """
    Load t_dlc data. Returns {dlc_id: dlc_name} dict or None.
    
    If source_type/source_path are provided (from unified source selection),
    uses those. Otherwise detects sources independently.
    Silently returns None if unavailable (graceful fallback).
    """
    if source_type and source_path:
        # Use the same source as t_name/t_item
        result = load_dlc_ids_from_source(base_dir, source_type, source_path)
        if result:
            return result
        # Fallback: try t_dlc-specific sources
    
    sources = detect_t_dlc_sources(base_dir)
    if not sources:
        return None

    # Filter usable sources
    usable = []
    for stype, path in sources:
        if stype == 'json':
            usable.append((stype, path))
        elif HAS_LIBS:
            usable.append((stype, path))
    if not usable:
        return None

    # Select source
    if len(usable) == 1 or no_interactive:
        stype, path = usable[0]
    else:
        print(f"\nMultiple t_dlc sources detected. Select source:")
        for i, (st, p) in enumerate(usable, 1):
            basename = os.path.basename(p)
            if st in ('p3a', 'zzz'):
                print(f"  {i}) {basename} (extract t_dlc.tbl)")
            else:
                print(f"  {i}) {basename}")
        while True:
            try:
                choice = input(f"Enter choice [1-{len(usable)}]: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(usable):
                    stype, path = usable[int(choice) - 1]
                    break
                print("Invalid choice, try again.")
            except (EOFError, KeyboardInterrupt):
                return None

    return load_dlc_ids_from_source(base_dir, stype, path)


def collect_used_dlc_ids(base_dir, t_dlc_dict=None):
    """
    Collect all used DLC IDs from:
    1. t_dlc game data (if available)
    2. All .kurodlc.json files in the directory
    
    Returns set of used DLC IDs.
    """
    used = set()

    if t_dlc_dict:
        used.update(t_dlc_dict.keys())

    for fname in os.listdir(base_dir):
        if not fname.lower().endswith('.kurodlc.json'):
            continue
        if '.bak' in fname.lower():
            continue
        fpath = os.path.join(base_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                fdata = json.load(f)
            if 'DLCTableData' in fdata:
                for dlc in fdata['DLCTableData']:
                    if isinstance(dlc, dict) and 'id' in dlc:
                        used.add(int(dlc['id']))
        except Exception:
            continue

    return used


def find_available_dlc_id(used_dlc_ids, min_id=DLC_ID_MIN, max_id=DLC_ID_MAX):
    """
    Find an available DLC ID in range [min_id, max_id).
    Strategy: start after highest used ID within range, then wrap around.
    
    Returns int or None if no IDs available.
    """
    if not used_dlc_ids:
        return min_id + 1

    # Find highest used ID within range
    ids_in_range = [i for i in used_dlc_ids if min_id <= i < max_id]
    if ids_in_range:
        start = max(ids_in_range) + 1
    else:
        start = min_id

    # Search forward from start
    for i in range(start, max_id):
        if i not in used_dlc_ids:
            return i

    # Wrap: search from min_id to start
    for i in range(min_id, start):
        if i not in used_dlc_ids:
            return i

    return None


def search_tdlc_interactive(dlc_dict, used_dlc_ids=None):
    """
    Interactive search mode for t_dlc data.
    Supports same search syntax as find_all_shops.py / shops_replace:
      id:NUMBER   - exact ID match
      name:TEXT   - search in DLC names (even if TEXT is a number)
      NUMBER      - auto: exact ID lookup + partial ID match
      TEXT        - auto: name search
    
    Also marks IDs as [USED] if in used_dlc_ids from .kurodlc.json files
    but not in t_dlc itself.
    """
    print(f"\n  === DLC search ({len(dlc_dict)} entries) ===")
    print(f"  id:N = exact ID | name:TEXT = name search | or just type")
    print(f"  Empty line returns to DLC ID input.\n")

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
            id_str = query[3:].strip()
            if not id_str:
                print(f"  Usage: id:NUMBER (e.g. id:345)")
                print()
                continue
            try:
                did = int(id_str)
                if did in dlc_dict:
                    results.append((did, dlc_dict[did]))
                elif used_dlc_ids and did in used_dlc_ids:
                    results.append((did, '[USED in .kurodlc.json]'))
            except ValueError:
                print(f"  Error: '{id_str}' is not a valid ID")
                print()
                continue

        elif query.startswith('name:'):
            name_str = query[5:].strip().lower()
            if not name_str:
                print(f"  Usage: name:TEXT (e.g. name:costume)")
                print()
                continue
            for dlc_id, name in sorted(dlc_dict.items()):
                if name_str in name.lower():
                    results.append((dlc_id, name))

        elif query.isdigit():
            did = int(query)
            if did in dlc_dict:
                results.append((did, dlc_dict[did]))
            elif used_dlc_ids and did in used_dlc_ids:
                results.append((did, '[USED in .kurodlc.json]'))
            for dlc_id, name in sorted(dlc_dict.items()):
                if query in str(dlc_id) and dlc_id != did:
                    results.append((dlc_id, name))

        else:
            query_lower = query.lower()
            for dlc_id, name in sorted(dlc_dict.items()):
                if query_lower in name.lower():
                    results.append((dlc_id, name))

        if not results:
            print(f"  No matches for '{query}'")
        else:
            max_id_len = max(len(str(r[0])) for r in results)
            for dlc_id, name in results:
                print(f"  {dlc_id:>{max_id_len}} : {name}")
            print(f"  ({len(results)} result(s))")
        print()


# =========================================================================
# kurodlc structure analysis
# =========================================================================

def get_existing_mdl_names(data):
    """Get set of mdl_name values already in CostumeParam."""
    if 'CostumeParam' not in data:
        return set()
    return {
        entry['mdl_name'].lower()
        for entry in data['CostumeParam']
        if isinstance(entry, dict) and 'mdl_name' in entry
    }


def get_existing_shop_ids(data):
    """Extract unique shop_ids from existing ShopItem entries."""
    if 'ShopItem' not in data:
        return []
    shop_ids = set()
    for entry in data['ShopItem']:
        if isinstance(entry, dict) and 'shop_id' in entry:
            shop_ids.add(entry['shop_id'])
    return sorted(shop_ids)


def get_item_template(data):
    """Get a template ItemTableData entry from existing data."""
    if 'ItemTableData' in data and data['ItemTableData']:
        for entry in data['ItemTableData']:
            if isinstance(entry, dict) and 'id' in entry:
                return copy.deepcopy(entry)
    return None


def get_costume_template(data):
    """Get a template CostumeParam entry from existing data."""
    if 'CostumeParam' in data and data['CostumeParam']:
        for entry in data['CostumeParam']:
            if isinstance(entry, dict) and 'item_id' in entry:
                return copy.deepcopy(entry)
    return None


def get_shop_template(data):
    """Get a template ShopItem entry from existing data."""
    if 'ShopItem' in data and data['ShopItem']:
        for entry in data['ShopItem']:
            if isinstance(entry, dict) and 'shop_id' in entry:
                return copy.deepcopy(entry)
    return None


# =========================================================================
# Entry generation
# =========================================================================

def make_costume_entry(template, mdl_name, char_restrict, item_id):
    """Create a new CostumeParam entry."""
    entry = copy.deepcopy(template) if template else {
        "char_restrict": 0,
        "type": 0,
        "item_id": 0,
        "unk0": 0,
        "unk_txt0": "",
        "mdl_name": "",
        "unk1": 0,
        "unk2": 0,
        "attach_name": "",
        "unk_txt1": "",
        "unk_txt2": ""
    }
    entry['char_restrict'] = char_restrict
    entry['item_id'] = item_id
    entry['mdl_name'] = mdl_name
    return entry


def make_item_entry(template, item_id, char_restrict, name, desc):
    """Create a new ItemTableData entry."""
    entry = copy.deepcopy(template) if template else {
        "id": 0,
        "chr_restrict": 0,
        "flags": "",
        "unk_txt": "1",
        "category": 17,
        "subcategory": 16,
        "unk0": 0, "unk1": 0, "unk2": 0, "unk3": 0, "unk4": 0,
        "eff1_id": 0, "eff1_0": 0, "eff1_1": 0, "eff1_2": 0,
        "eff2_id": 0, "eff2_0": 0, "eff2_1": 0, "eff2_2": 0,
        "eff3_id": 0, "eff3_0": 0, "eff3_1": 0, "eff3_2": 0,
        "eff4_id": 0, "eff4_0": 0, "eff4_1": 0, "eff4_2": 0,
        "eff5_id": 0, "eff5_0": 0, "eff5_1": 0, "eff5_2": 0,
        "unk5": 0,
        "hp": 0, "ep": 0, "patk": 0, "pdef": 0, "matk": 0, "mdef": 0,
        "str": 0, "def": 0, "ats": 0, "adf": 0, "agl": 0, "dex": 0,
        "hit": 0, "eva": 0, "meva": 0, "crit": 0, "spd": 0, "mov": 0,
        "stack_size": 1,
        "price": 100,
        "anim": "",
        "name": "",
        "desc": "",
        "unk6": 0, "unk7": 0, "unk8": 0, "unk9": 0
    }
    entry['id'] = item_id
    entry['chr_restrict'] = char_restrict
    entry['name'] = name
    entry['desc'] = desc
    return entry


def make_shop_entries(template, item_id, shop_ids):
    """Create ShopItem entries for all shop_ids."""
    entries = []
    for shop_id in shop_ids:
        entry = copy.deepcopy(template) if template else {
            "shop_id": 0,
            "item_id": 0,
            "unknown": 1,
            "start_scena_flags": [],
            "empty1": 0,
            "end_scena_flags": [],
            "int2": 0
        }
        entry['shop_id'] = shop_id
        entry['item_id'] = item_id
        entries.append(entry)
    return entries


# =========================================================================
# MDL scanning
# =========================================================================

def scan_mdl_files(base_dir):
    """
    Scan directory for .mdl files and return list of mdl names (without extension).
    """
    mdl_files = glob.glob(os.path.join(base_dir, '*.mdl'))
    mdl_names = []
    for f in mdl_files:
        name = os.path.splitext(os.path.basename(f))[0]
        mdl_names.append(name)
    return sorted(mdl_names)


# =========================================================================
# Main logic
# =========================================================================

def print_usage():
    """Print usage information."""
    print(
        "Usage: python kurodlc_add_mdl.py <file.kurodlc.json> [options]\n"
        "\n"
        "Scan directory for .mdl files not yet in the kurodlc config and add\n"
        "complete entries (CostumeParam, ItemTableData, DLCTableData, ShopItem).\n"
        "\n"
        "If the target file does not exist, it is created from scratch.\n"
        "When no DLCTableData record exists, prompts for DLC ID and name.\n"
        "Uses t_dlc data (if available) for DLC ID validation and suggestions.\n"
        "\n"
        "The script looks for .mdl files, t_name, t_item and optionally t_dlc\n"
        "data in the same directory as the .kurodlc.json file.\n"
        "\n"
        "Smart ID Assignment (v2.0):\n"
        "  - Collects used IDs from t_item (game data) + all .kurodlc.json files\n"
        "  - Searches for free IDs in range 1-5000 (configurable)\n"
        "  - Tries continuous block first, falls back to scattered search\n"
        "  - Same algorithm as resolve_id_conflicts_in_kurodlc.py\n"
        "\n"
        "DLC ID Assignment (v2.2):\n"
        "  - Loads t_dlc data for used DLC ID detection\n"
        "  - Collects DLC IDs from all .kurodlc.json files in directory\n"
        "  - Suggests first available ID after highest used in range 1-350\n"
        "  - Validates against known DLC names from t_dlc\n"
        "\n"
        "Options:\n"
        "  --apply             Apply changes (without this, runs in dry-run mode)\n"
        "  --dry-run           Explicit dry-run (default behavior, no changes written)\n"
        "  --shop-ids=1,2,3    Override shop IDs (default: auto-detect from file)\n"
        "  --min-id=N          Minimum ID for search range (default: 1)\n"
        "  --max-id=N          Maximum ID for search range (default: 5000)\n"
        "  --no-interactive    Auto-select sources without prompting\n"
        "  --no-backup         Skip backup creation when applying\n"
        "  --no-ascii-escape   Write UTF-8 directly (e.g. Agnès instead of Agn\\u00e8s)\n"
        "  --help              Show this help message\n"
        "\n"
        "Required files (in same directory as kurodlc.json):\n"
        "  *.mdl               MDL model files to add\n"
        "  t_name source       One of: t_name.json, t_name.tbl, script_en.p3a,\n"
        "                      script_eng.p3a, zzz_combined_tables.p3a\n"
        "  t_item source       One of: t_item.json, t_item.tbl, script_en.p3a,\n"
        "                      script_eng.p3a, zzz_combined_tables.p3a\n"
        "\n"
        "Optional files (for DLC ID validation):\n"
        "  t_dlc source        One of: t_dlc.json, t_dlc.tbl, t_dlc.tbl.original,\n"
        "                      script_en.p3a, script_eng.p3a, zzz_combined_tables.p3a\n"
        "\n"
        "What gets generated for each new MDL:\n"
        "  CostumeParam    - char_restrict from t_name, mdl_name, new item_id\n"
        "  ItemTableData   - name: '<CharName> generated <mdl_name>'\n"
        "                    (placeholder for manual editing)\n"
        "  DLCTableData    - item_id appended to existing DLC record,\n"
        "                    or new record created with prompted DLC ID and name\n"
        "  ShopItem        - entries for each shop_id in the file\n"
        "\n"
        "Examples:\n"
        "  python kurodlc_add_mdl.py FalcoDLC.kurodlc.json\n"
        "      Preview changes (dry-run, default).\n"
        "\n"
        "  python kurodlc_add_mdl.py FalcoDLC.kurodlc.json --apply\n"
        "      Apply changes and write to file (with timestamped backup).\n"
        "\n"
        "  python kurodlc_add_mdl.py NewMod.kurodlc.json --apply\n"
        "      Create new file from scratch (prompts for DLC ID and name).\n"
        "\n"
        "  python kurodlc_add_mdl.py FalcoDLC.kurodlc.json --shop-ids=21,22\n"
        "      Use specific shop IDs instead of auto-detected ones.\n"
        "\n"
        "  python kurodlc_add_mdl.py FalcoDLC.kurodlc.json --min-id=3000 --max-id=4000\n"
        "      Search for free IDs only in range 3000-4000."
    )


def main():
    """Main function."""
    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        print_usage()
        if len(sys.argv) < 2:
            sys.exit(1)
        return

    # Parse arguments
    json_file = sys.argv[1]

    # Ensure filename ends with .kurodlc.json
    if not json_file.lower().endswith('.kurodlc.json'):
        if json_file.lower().endswith('.kurodlc'):
            corrected = json_file + '.json'
        elif json_file.lower().endswith('.json'):
            corrected = json_file[:-5] + '.kurodlc.json'
        else:
            corrected = json_file + '.kurodlc.json'
        print(f"Note: Adding extension → {corrected}")
        json_file = corrected
    apply_changes = False
    manual_shop_ids = None
    min_id = 1
    max_id = 5000
    no_interactive = False
    do_backup = True
    ensure_ascii = True

    args = sys.argv[2:]

    for arg in args:
        if arg == '--apply':
            apply_changes = True
        elif arg == '--dry-run':
            apply_changes = False
        elif arg.startswith('--shop-ids='):
            try:
                manual_shop_ids = [int(x.strip()) for x in arg.split('=', 1)[1].split(',')]
            except ValueError:
                print(f"Error: Invalid shop IDs in {arg}")
                sys.exit(1)
        elif arg.startswith('--min-id='):
            try:
                min_id = int(arg.split('=', 1)[1])
            except ValueError:
                print(f"Error: Invalid min ID in {arg}")
                sys.exit(1)
        elif arg.startswith('--max-id='):
            try:
                max_id = int(arg.split('=', 1)[1])
            except ValueError:
                print(f"Error: Invalid max ID in {arg}")
                sys.exit(1)
        elif arg == '--no-interactive':
            no_interactive = True
        elif arg == '--no-backup':
            do_backup = False
        elif arg == '--no-ascii-escape':
            ensure_ascii = False
        elif arg.startswith('--'):
            print(f"Error: Unknown option '{arg}'")
            sys.exit(1)

    # ---- Load or create kurodlc.json ----
    is_new_file = not os.path.exists(json_file)
    base_dir = os.path.dirname(os.path.abspath(json_file)) or '.'

    if is_new_file:
        print(f"File '{json_file}' does not exist — will create new DLC file.")
        data = {
            "CostumeParam": [],
            "DLCTableData": [],
            "ItemTableData": [],
            "ShopItem": []
        }
    else:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading '{json_file}': {e}")
            sys.exit(1)

        if not isinstance(data, dict):
            print(f"Error: '{json_file}' root must be a JSON object.")
            sys.exit(1)

    # Ensure required sections exist (initialize empty if missing)
    for section in ['CostumeParam', 'ItemTableData', 'DLCTableData', 'ShopItem']:
        if section not in data:
            data[section] = []

    print(f"{'Creating' if is_new_file else 'Loaded'}: {json_file}")
    print(f"  CostumeParam:  {len(data['CostumeParam'])} entries")
    print(f"  ItemTableData: {len(data['ItemTableData'])} entries")
    print(f"  DLCTableData:  {len(data['DLCTableData'])} entries")
    print(f"  ShopItem:      {len(data['ShopItem'])} entries")

    # ---- Scan MDL files ----
    mdl_names = scan_mdl_files(base_dir)
    if not mdl_names:
        print(f"\nNo .mdl files found in {base_dir}")
        sys.exit(0)

    print(f"\nFound {len(mdl_names)} .mdl file(s) in directory")

    # ---- Find missing MDLs ----
    existing_mdls = get_existing_mdl_names(data)
    new_mdls = [m for m in mdl_names if m.lower() not in existing_mdls]

    if not new_mdls:
        print("\nAll .mdl files are already present in the kurodlc config. Nothing to add.")
        sys.exit(0)

    print(f"New .mdl files to add: {len(new_mdls)}")
    for m in new_mdls:
        print(f"  + {m}")

    # ---- Select data source (once for all tables) ----
    required_prefixes = ['t_name', 't_item']
    all_sources = detect_all_sources(base_dir, required_prefixes)

    if not all_sources:
        print("\nError: No data source found.")
        print("Expected one of: t_name/t_item .json, .tbl.original, .tbl,")
        print("  script_en.p3a, script_eng.p3a, zzz_combined_tables.p3a")
        sys.exit(1)

    # Filter to valid sources only for auto-selection
    valid_sources = [s for s in all_sources if s[2]]  # is_valid == True

    if not valid_sources:
        print("\nError: No complete data source found. All sources have missing files:")
        for stype, path, is_valid, missing in all_sources:
            missing_str = ', '.join(missing)
            print(f"  {stype}: missing {missing_str}")
        sys.exit(1)

    if len(valid_sources) == 1 or no_interactive:
        source_type, source_path, _, _ = valid_sources[0]
    elif len(all_sources) == 1 and all_sources[0][2]:
        source_type, source_path, _, _ = all_sources[0]
    else:
        source_type, source_path, _, _ = select_source_interactive(all_sources)

    # ---- Load t_name for character mapping ----
    print(f"\nLoading character data (t_name)...")
    names_data = load_table_data(base_dir, 't_name', 'NameTableData', source_type, source_path)

    if names_data is None:
        print("\nError: Could not load t_name data.")
        sys.exit(1)

    char_map = build_char_map(names_data)
    print(f"Character map: {len(char_map)} characters loaded")

    # ---- Resolve character info for each new MDL ----
    resolved = []  # list of (mdl_name, char_id, char_name)
    unresolved = []

    for mdl_name in new_mdls:
        prefix = extract_chr_prefix(mdl_name)
        if prefix and prefix in char_map:
            info = char_map[prefix]
            resolved.append((mdl_name, info['char_id'], info['name']))
        else:
            unresolved.append((mdl_name, prefix))

    if unresolved:
        print(f"\nWarning: Could not resolve character for {len(unresolved)} MDL(s):")
        for mdl_name, prefix in unresolved:
            print(f"  ? {mdl_name}  (prefix: {prefix or 'none'})")

        if no_interactive:
            print("Skipping unresolved MDLs (--no-interactive).")
        else:
            print("\nYou can manually assign char_restrict for unresolved MDLs.")
            print("Enter char_restrict value, or press Enter to skip, 'q' to abort:\n")
            for mdl_name, prefix in unresolved:
                try:
                    val = input(f"  {mdl_name} char_restrict = ").strip()
                    if val.lower() == 'q':
                        print("Aborted.")
                        sys.exit(0)
                    if val:
                        char_id = int(val)
                        char_name_input = input(f"  {mdl_name} character name = ").strip()
                        resolved.append((mdl_name, char_id, char_name_input or "Unknown"))
                    else:
                        print(f"    Skipping {mdl_name}")
                except (ValueError, EOFError, KeyboardInterrupt):
                    print(f"    Skipping {mdl_name}")

    if not resolved:
        print("\nNo MDL files could be resolved. Nothing to add.")
        sys.exit(0)

    # ---- Load t_item for used ID detection ----
    print(f"\nLoading game item data (t_item)...")
    items_data = load_table_data(base_dir, 't_item', 'ItemTableData', source_type, source_path)

    if items_data is None:
        print("\nError: Could not load t_item data.")
        sys.exit(1)

    # Transform list of entries to {id: name} dict (same as resolve_id_conflicts)
    items_dict = {x['id']: x.get('name', '') for x in items_data if isinstance(x, dict) and 'id' in x}
    print(f"Game items loaded: {len(items_dict)} IDs")

    # ---- Build complete used_ids set ----
    used_ids = collect_all_used_ids(base_dir, items_dict)
    print(f"Total used IDs (game + all DLCs): {len(used_ids)}")

    # ---- Determine shop config ----
    if manual_shop_ids:
        shop_ids = manual_shop_ids
    else:
        shop_ids = get_existing_shop_ids(data)
        if not shop_ids:
            shop_ids = [21, 22, 248, 258]  # Fallback: common Kuro 2 costume shops
            print(f"  No existing shop_ids found, using defaults: {shop_ids}")

    # Get templates from existing entries
    costume_tmpl = get_costume_template(data)
    item_tmpl = get_item_template(data)
    shop_tmpl = get_shop_template(data)

    # ---- Find available IDs using smart algorithm ----
    count_needed = len(resolved)
    print(f"\nSearching for {count_needed} available ID(s) in range [{min_id}, {max_id}]...")

    try:
        available_ids = find_available_ids_in_range(used_ids, count_needed, min_id, max_id)
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    # Report ID assignment strategy
    if len(available_ids) > 1:
        is_continuous = available_ids == list(range(min(available_ids), max(available_ids) + 1))
        id_range_str = f"{min(available_ids)}-{max(available_ids)}"
        strategy = "continuous block" if is_continuous else "scattered (using gaps)"
        print(f"Found {len(available_ids)} IDs: {id_range_str} ({strategy})")
    elif available_ids:
        print(f"Found ID: {available_ids[0]}")

    # ---- Generate entries ----
    print(f"\n{'='*60}")
    print(f"Generating entries")
    print(f"Shop IDs: {shop_ids}")
    print(f"{'='*60}\n")

    new_costume_entries = []
    new_item_entries = []
    new_shop_entries = []
    new_item_ids = []

    for i, (mdl_name, char_id, char_name) in enumerate(resolved):
        item_id = available_ids[i]
        new_item_ids.append(item_id)

        # Generate display name
        item_name = f"{char_name} generated {mdl_name}"
        item_desc = item_name

        # CostumeParam
        costume_entry = make_costume_entry(costume_tmpl, mdl_name, char_id, item_id)
        new_costume_entries.append(costume_entry)

        # ItemTableData
        item_entry = make_item_entry(item_tmpl, item_id, char_id, item_name, item_desc)
        new_item_entries.append(item_entry)

        # ShopItem
        shop_entries = make_shop_entries(shop_tmpl, item_id, shop_ids)
        new_shop_entries.extend(shop_entries)

        print(f"  item_id={item_id}  char={char_id:>3d}  {char_name:<20s}  mdl={mdl_name}")

    # ---- Summary ----
    # ---- DLCTableData: determine DLC ID and name if needed ----
    dlc_id = None
    dlc_name = None
    needs_new_dlc_record = len(data['DLCTableData']) == 0

    if needs_new_dlc_record:
        print(f"\nNo existing DLCTableData record — need to assign a DLC ID.")

        # Load t_dlc for DLC ID detection
        t_dlc_dict = load_t_dlc_data(base_dir, source_type, source_path, no_interactive)
        used_dlc_ids = collect_used_dlc_ids(base_dir, t_dlc_dict)
        suggested_id = find_available_dlc_id(used_dlc_ids)

        if t_dlc_dict or used_dlc_ids:
            # Show loading info
            if t_dlc_dict:
                print(f"Loading t_dlc from: {len(t_dlc_dict)} game DLC entries")

            # Analyze ID distribution
            ids_in_range = [i for i in used_dlc_ids if DLC_ID_MIN <= i < DLC_ID_MAX]
            ids_over = [i for i in used_dlc_ids if i >= DLC_ID_MAX]
            free_count = DLC_ID_MAX - DLC_ID_MIN - len(ids_in_range)

            print(f"\nDLC ID status:")
            print(f"  Current IDs in use: {min(used_dlc_ids)}-{max(used_dlc_ids)} "
                  f"({len(used_dlc_ids)} total)")
            print(f"  Assignable range:   {DLC_ID_MIN}-{DLC_ID_MAX - 1} "
                  f"({len(ids_in_range)} used, {free_count} available)")
            if ids_over:
                print(f"  Note: {len(ids_over)} ID(s) above {DLC_ID_MAX} exist "
                      f"(game/mod IDs outside assignable range)")
        else:
            print(f"\nNo existing DLC IDs found.")
            print(f"  Assignable range: {DLC_ID_MIN}-{DLC_ID_MAX - 1}")

        if suggested_id is not None:
            print(f"  Suggested ID:       {suggested_id}")

        if no_interactive:
            dlc_id = suggested_id if suggested_id is not None else 999
            dlc_name = os.path.splitext(os.path.basename(json_file))[0]
            if dlc_name.endswith('.kurodlc'):
                dlc_name = dlc_name[:-8]
            print(f"Using DLC ID: {dlc_id}, name: {dlc_name}")
        else:
            # Prompt for DLC ID
            search_hint = "\n  ? = search DLCs" if t_dlc_dict else ""
            while True:
                default_str = f" [{suggested_id}]" if suggested_id is not None else ""
                if search_hint:
                    print(search_hint)
                try:
                    id_input = input(f"Enter DLC ID{default_str}: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nAborted.")
                    sys.exit(1)

                if id_input == '?' and t_dlc_dict:
                    search_tdlc_interactive(t_dlc_dict, used_dlc_ids)
                    continue

                if not id_input and suggested_id is not None:
                    dlc_id = suggested_id
                    break
                try:
                    dlc_id = int(id_input)
                    if dlc_id in used_dlc_ids:
                        if t_dlc_dict and dlc_id in t_dlc_dict:
                            print(f"Warning: ID {dlc_id} is assigned to '{t_dlc_dict[dlc_id]}'!")
                        else:
                            print(f"Warning: ID {dlc_id} is already used!")
                        try:
                            confirm = input("Use anyway? [y/N]: ").strip().lower()
                        except (EOFError, KeyboardInterrupt):
                            print("\nAborted.")
                            sys.exit(1)
                        if confirm == 'y':
                            break
                        continue
                    break
                except ValueError:
                    print("Invalid input. Enter a number.")

            # Prompt for DLC name (= desc)
            dlc_name_default = os.path.splitext(os.path.basename(json_file))[0]
            if dlc_name_default.endswith('.kurodlc'):
                dlc_name_default = dlc_name_default[:-8]
            try:
                name_input = input(f"Enter DLC name [{dlc_name_default}]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                sys.exit(1)
            dlc_name = name_input if name_input else dlc_name_default

        print(f"  DLC record: id={dlc_id}, name='{dlc_name}'")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"Summary of changes:")
    print(f"{'='*60}")
    print(f"  CostumeParam:  +{len(new_costume_entries)} entries")
    print(f"  ItemTableData: +{len(new_item_entries)} entries")
    if needs_new_dlc_record:
        print(f"  DLCTableData:  new record (id={dlc_id}) with {len(new_item_ids)} item(s)")
    else:
        print(f"  DLCTableData:  +{len(new_item_ids)} item(s) added to existing record")
    print(f"  ShopItem:      +{len(new_shop_entries)} entries "
          f"({len(resolved)} items x {len(shop_ids)} shops)")
    print(f"{'='*60}")

    if not apply_changes:
        print(f"\n[DRY RUN] No files modified. Use --apply to write changes.")
        return

    # ---- Apply changes ----
    data['CostumeParam'].extend(new_costume_entries)
    data['ItemTableData'].extend(new_item_entries)
    data['ShopItem'].extend(new_shop_entries)

    # DLCTableData: extend existing DLC record or create new
    if needs_new_dlc_record:
        dlc_record = {
            "id": dlc_id,
            "sort_id": dlc_id,
            "items": new_item_ids,
            "unk0": 0,
            "quantity": [1] * len(new_item_ids),
            "unk1": 0,
            "name": dlc_name,
            "desc": dlc_name,
            "unk_txt": "",
            "unk2": 0,
            "unk3": 1,
            "unk4": 0,
            "unk_arr": [],
            "unk5": 0
        }
        data['DLCTableData'].append(dlc_record)
    else:
        dlc_record = data['DLCTableData'][0]
        if 'items' in dlc_record:
            dlc_record['items'].extend(new_item_ids)
        if 'quantity' in dlc_record:
            dlc_record['quantity'].extend([1] * len(new_item_ids))

    # ---- Backup and write ----
    if do_backup and not is_new_file:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = json_file + f'_{timestamp}.bak'
        try:
            import shutil
            shutil.copy2(json_file, backup_file)
            print(f"\nBackup created: {backup_file}")
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")

    try:
        # Ensure section order: CostumeParam, DLCTableData, ItemTableData, ShopItem
        section_order = ['CostumeParam', 'DLCTableData', 'ItemTableData', 'ShopItem']
        ordered = {k: data[k] for k in section_order if k in data}
        for k in data:
            if k not in ordered:
                ordered[k] = data[k]

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(ordered, f, indent=4, ensure_ascii=ensure_ascii)
        print(f"Written: {json_file}")
        print(f"\nDone! {len(resolved)} new MDL(s) added successfully"
              + (f" (new file created)" if is_new_file else "") + ".")
        print(f"\nReminder: Review generated item names in ItemTableData")
        print(f"  (search for 'generated' to find placeholder entries)")
    except Exception as e:
        print(f"Error writing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
