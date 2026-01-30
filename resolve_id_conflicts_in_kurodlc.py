#!/usr/bin/env python3
"""
resolve_id_conflicts_in_kurodlc.py

Description:
Detects and repairs item ID conflicts in .kurodlc.json files against game item data.

NEW in v2.7: Smart ID Assignment Algorithm
- Searches only in range 1-5000 (safe limit)
- Starts from middle (2500) for better distribution
- Finds continuous blocks when possible (fast)
- Falls back to scattered search if needed (handles fragmentation)
- Clear error messages if not enough IDs available

Supported sources:
- t_item.json
- t_item.tbl
- t_item.tbl.original
- script_en.p3a / script_eng.p3a (extracts t_item.tbl.original.tmp)
- zzz_combined_tables.p3a (extracts t_item.tbl.original.tmp)

Modes:
1) checkbydlc
   - Check all .kurodlc.json files for conflicts.
   - [OK] = available, [BAD] = conflict.
   - Shows totals per file and overall.

2) repair
   - Same as checkbydlc, but generates a repair plan for [BAD] IDs.
   - NEW: Uses smart algorithm to find IDs in range 1-5000
   - --apply actually modifies files with backups and verbose logs.
   - --export exports repair plan to id_mapping.json for manual editing
   - --import imports edited id_mapping.json and applies changes

3) repair --export
   - Generates repair plan and exports to id_mapping.json
   - User can manually edit the mapping file
   - Does NOT apply changes yet

4) repair --import
   - Imports id_mapping.json (previously exported and edited by user)
   - Validates all new IDs (checks for conflicts and availability)
   - Applies changes with backups and logs (like --apply)

Options:
--apply                Apply changes to DLC files immediately (automatic repair).
--export               Export repair plan to id_mapping_TIMESTAMP.json for manual editing.
--export-name=NAME     Custom name for exported mapping file.
                       Just provide the custom part, prefix/suffix added automatically.
                       Examples:
                         --export-name=DLC1      -> creates id_mapping_DLC1.json
                         --export-name=test      -> creates id_mapping_test.json
                         --export-name=my_mod    -> creates id_mapping_my_mod.json
                       You can also provide full name if preferred:
                         --export-name=id_mapping_DLC1.json  -> id_mapping_DLC1.json
--import               Import edited id_mapping.json and apply changes.
                       If multiple files exist, shows interactive selection menu.
--mapping-file=PATH    Specify which mapping file to import (must be full filename).
                       Example: --mapping-file=id_mapping_DLC1.json
                       Skips interactive menu if specified.
--keep-extracted       Keep temporary extracted t_item.tbl.original.tmp after P3A extraction.
--no-interactive       Automatically selects first source if multiple found.
                       Also auto-selects newest mapping file when using --import.
--source=<type>        Force a source: json, tbl, original, p3a, zzz.

Workflow examples:

  # Classic automatic repair (NEW: uses smart algorithm, IDs in range 1-5000):
  python resolve_id_conflicts_in_kurodlc.py repair --apply
  
  # NEW: Interactive workflow with manual editing:
  
  # Step 1a: Export with auto-generated timestamp name
  python resolve_id_conflicts_in_kurodlc.py repair --export
  # Creates: id_mapping_20260131_143022.json
  
  # Step 1b: Export with simple custom name (RECOMMENDED)
  python resolve_id_conflicts_in_kurodlc.py repair --export --export-name=DLC1
  # Creates: id_mapping_DLC1.json
  
  # Step 1c: Export with full custom name (if you prefer)
  python resolve_id_conflicts_in_kurodlc.py repair --export --export-name=id_mapping_DLC1.json
  # Creates: id_mapping_DLC1.json
  
  # Step 2: Manually edit the exported file
  # Change suggested new_id values as needed
  
  # Step 3a: Import (interactive selection if multiple files)
  python resolve_id_conflicts_in_kurodlc.py repair --import
  # Shows menu to select file
  
  # Step 3b: Import specific file (skip menu, must use FULL filename)
  python resolve_id_conflicts_in_kurodlc.py repair --import --mapping-file=id_mapping_DLC1.json
  
  # Step 3c: Import with auto-selection (no menu, picks newest)
  python resolve_id_conflicts_in_kurodlc.py repair --import --no-interactive
  
  # You can also combine export with immediate apply:
  python resolve_id_conflicts_in_kurodlc.py repair --export --apply

Other examples:
  python resolve_id_conflicts_in_kurodlc.py checkbydlc
  python resolve_id_conflicts_in_kurodlc.py repair --source=json
  
Algorithm Details:
  The smart algorithm searches for available IDs in range 1-5000:
  1. Starts from middle (2500) for better distribution
  2. Tries to find continuous block first (e.g., 4000-4049)
  3. Falls back to scattered search if needed (e.g., 2499, 2501, 2502, ...)
  4. Gives clear error if not enough IDs available in range
  
  Benefits:
  - Guaranteed to stay within 1-5000 range
  - Better ID distribution (not clustered at end)
  - Handles fragmented ID space efficiently
  - Clear error messages with suggestions
"""

import os, sys, json, shutil, datetime

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

# -------------------------
# Colorama for Windows CMD
# -------------------------
try:
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
except ImportError:
    class Dummy:
        RED=GREEN=RESET_ALL=""
    Fore = Style = Dummy()

# -------------------------
# Smart ID Assignment Algorithm
# -------------------------

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
    # Validate range
    if min_id >= max_id:
        raise ValueError(f"Invalid range: min_id ({min_id}) must be less than max_id ({max_id})")
    
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
        f"      Suggestion: Increase max_id (e.g., --max-id={max_id + 5000}) or remove some items"
    )


def find_continuous_block(used_ids, count_needed, min_id, max_id):
    """
    Try to find a continuous block of available IDs.
    Starts from middle of range and searches outward for better distribution.
    
    Returns:
        List of IDs if continuous block found, None otherwise
    """
    middle = (min_id + max_id) // 2
    max_offset = max(middle - min_id, max_id - middle)
    
    # Search from middle outward
    for offset in range(max_offset + 1):
        # Try middle + offset
        start = middle + offset
        if start + count_needed - 1 <= max_id:
            if all(id not in used_ids for id in range(start, start + count_needed)):
                return list(range(start, start + count_needed))
        
        # Try middle - offset (avoid duplicate at offset=0)
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
    
    Returns:
        Sorted list of IDs if enough found, None otherwise
    """
    available = []
    middle = (min_id + max_id) // 2
    max_offset = max(middle - min_id, max_id - middle)
    
    # Start from middle and spiral outward
    for offset in range(max_offset + 1):
        # Check middle + offset
        candidate = middle + offset
        if min_id <= candidate <= max_id and candidate not in used_ids:
            available.append(candidate)
            if len(available) >= count_needed:
                return sorted(available)
        
        # Check middle - offset (avoid duplicate at offset=0)
        if offset > 0:
            candidate = middle - offset
            if min_id <= candidate <= max_id and candidate not in used_ids:
                available.append(candidate)
                if len(available) >= count_needed:
                    return sorted(available)
    
    return None

    Fore=Style=Dummy()

# -------------------------
# Utilities
# -------------------------
def print_usage():
    print(__doc__)

def get_all_files(cmdlog=False):
    valid_files = []

    for name in os.listdir('.'):
        lname = name.lower()

        if not lname.endswith('.kurodlc.json'):
            continue
        if '.bak_' in lname:
            continue
        if not os.path.isfile(name):
            continue

        ok, reason = is_valid_kurodlc_json(name)
        if not ok:
            if cmdlog:
                print(f"Skipping {name}: {reason}")
            continue

        valid_files.append(name)

    return valid_files


def extract_item_ids(json_file):
    """
    Extract item_ids from all relevant sections.
    
    Returns: list of all IDs (with duplicates if ID is in multiple sections)
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = []
    
    # CostumeParam: item_id field
    if 'CostumeParam' in data:
        for item in data['CostumeParam']:
            if 'item_id' in item:
                ids.append(item['item_id'])
    
    # ItemTableData: id field
    if 'ItemTableData' in data:
        for item in data['ItemTableData']:
            if 'id' in item:
                ids.append(item['id'])
    
    # DLCTableData: items field (list of integers)
    if 'DLCTableData' in data:
        for item in data['DLCTableData']:
            if 'items' in item and isinstance(item['items'], list):
                ids.extend(item['items'])
    
    # ShopItem: item_id field (optional section)
    if 'ShopItem' in data:
        for item in data['ShopItem']:
            if 'item_id' in item:
                ids.append(item['item_id'])
    
    return ids


def extract_item_ids_with_sections(json_file):
    """
    Extract item_ids with information about which sections contain each ID.
    
    Returns: 
        - dict mapping ID -> list of sections where it appears
        - None if file cannot be read or has invalid JSON
    
    Example: {3596: ['CostumeParam', 'ItemTableData', 'DLCTableData']}
    """
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {json_file}:")
        print(f"        {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Cannot read {json_file}: {e}")
        return None
    
    if not isinstance(data, dict):
        print(f"[ERROR] {json_file}: Root element must be a JSON object")
        return None
    
    id_sections = {}  # ID -> [sections]
    
    # CostumeParam: item_id field
    if 'CostumeParam' in data:
        for item in data['CostumeParam']:
            if 'item_id' in item:
                item_id = item['item_id']
                if item_id not in id_sections:
                    id_sections[item_id] = []
                id_sections[item_id].append('CostumeParam')
    
    # ItemTableData: id field
    if 'ItemTableData' in data:
        for item in data['ItemTableData']:
            if 'id' in item:
                item_id = item['id']
                if item_id not in id_sections:
                    id_sections[item_id] = []
                id_sections[item_id].append('ItemTableData')
    
    # DLCTableData: items field (list of integers)
    if 'DLCTableData' in data:
        for dlc in data['DLCTableData']:
            if 'items' in dlc and isinstance(dlc['items'], list):
                for item_id in dlc['items']:
                    if item_id not in id_sections:
                        id_sections[item_id] = []
                    if 'DLCTableData' not in id_sections[item_id]:
                        id_sections[item_id].append('DLCTableData')
    
    # ShopItem: item_id field (optional section)
    if 'ShopItem' in data:
        for item in data['ShopItem']:
            if 'item_id' in item:
                item_id = item['item_id']
                if item_id not in id_sections:
                    id_sections[item_id] = []
                id_sections[item_id].append('ShopItem')
    
    return id_sections

def is_valid_kurodlc_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False, "invalid json"

    if not isinstance(data, dict):
        return False, "root is not object"

    # ---- CostumeParam ----
    if "CostumeParam" not in data or not isinstance(data["CostumeParam"], list):
        return False, "missing or invalid CostumeParam"

    for item in data["CostumeParam"]:
        if not isinstance(item, dict):
            return False, "CostumeParam item not object"
        if "item_id" not in item or not isinstance(item["item_id"], int):
            return False, "CostumeParam.item_id missing or not int"

    # ---- DLCTableData ----
    if "DLCTableData" not in data or not isinstance(data["DLCTableData"], list):
        return False, "missing or invalid DLCTableData"

    for item in data["DLCTableData"]:
        if not isinstance(item, dict):
            return False, "DLCTableData item not object"
        if "items" not in item or not isinstance(item["items"], list):
            return False, "DLCTableData.items missing or not list"
        if not all(isinstance(x, int) for x in item["items"]):
            return False, "DLCTableData.items contains non-int"

    # ---- ItemTableData ---- (OPTIONAL)
    # FIXED: ItemTableData is optional in some DLC files
    if "ItemTableData" in data:
        if not isinstance(data["ItemTableData"], list):
            return False, "invalid ItemTableData (not list)"
        for item in data["ItemTableData"]:
            if not isinstance(item, dict):
                return False, "ItemTableData item not object"
            if "id" not in item or not isinstance(item["id"], int):
                return False, "ItemTableData.id missing or not int"

    # ---- ShopItem ---- (OPTIONAL)
    if "ShopItem" in data:
        if not isinstance(data["ShopItem"], list):
            return False, "invalid ShopItem (not list)"
        for item in data["ShopItem"]:
            if not isinstance(item, dict):
                return False, "ShopItem item not object"
            if "item_id" not in item or not isinstance(item["item_id"], int):
                return False, "ShopItem.item_id missing or not int"

    return True, "ok"

# -------------------------
# Source detection
# -------------------------
def detect_sources():
    sources=[]
    if os.path.exists("t_item.json"): sources.append(("json","t_item.json"))
    if os.path.exists("t_item.tbl.original"): sources.append(("original","t_item.tbl.original"))
    if os.path.exists("t_item.tbl"): sources.append(("tbl","t_item.tbl"))
    if os.path.exists("script_en.p3a"): sources.append(("p3a","script_en.p3a"))
    if os.path.exists("script_eng.p3a"): sources.append(("p3a","script_eng.p3a"))
    if os.path.exists("zzz_combined_tables.p3a"): sources.append(("zzz","zzz_combined_tables.p3a"))
    return sources

def select_source_interactive(sources):
    print("\nMultiple item sources detected. Select source to use for check/repair:")
    for i,(stype,path) in enumerate(sources,1):
        if stype in ("p3a", "zzz"):
            print(f"  {i}) {path} (extract t_item.tbl.original.tmp)")
        else:
            print(f"  {i}) {path}")
    while True:
        choice = input(f"\nEnter choice [1-{len(sources)}]: ").strip()
        if choice.isdigit() and 1<=int(choice)<=len(sources):
            return sources[int(choice)-1]
        print("Invalid choice, try again.")

# -------------------------
# P3A extraction
# -------------------------
def extract_from_p3a(p3a_file, out_file):
    p3a=p3a_class()
    if os.path.exists(p3a_file):
        with open(p3a_file,'rb') as p3a.f:
            headers, entries, p3a_dict = p3a.read_p3a_toc()
            for entry in entries:
                if os.path.basename(entry['name'])=='t_item.tbl':
                    data=p3a.read_file(entry,p3a_dict)
                    with open(out_file,'wb') as f: f.write(data)
                    return True
    return False

# -------------------------
# Load items
# -------------------------
def load_items_from_json():
    with open('t_item.json','r',encoding='utf-8') as f:
        data=json.load(f)
    for section in data.get("data",[]):
        if section.get("name")=="ItemTableData":
            return {x['id']:x['name'] for x in section.get("data",[])}
    return {}

def load_items_from_tbl(tbl_file):
    kt=kuro_tables()
    table=kt.read_table(tbl_file)
    return {x['id']:x['name'] for x in table['ItemTableData']}

# -------------------------
# Print IDs and prepare repair
# -------------------------
def print_ids_for_list(item_ids, items_dict, used_ids_snapshot, mode="check"):
    """
    Print item IDs and their conflict status.
    
    FIXED: Uses immutable snapshot of used_ids to prevent race conditions.
    Returns repair entries without mutating the input used_ids.
    
    Args:
        item_ids: List of item IDs to check
        items_dict: Dict of game items {id: name}
        used_ids_snapshot: Immutable set of IDs to check against
        mode: "check" or "repair"
    
    Returns:
        (ok_count, bad_count, total_count, repair_entries)
    """
    unique_ids = sorted(set(item_ids))
    if not unique_ids:
        print("No item_ids found.")
        return 0, 0, len(unique_ids), []
    
    # Create local copy for generating new IDs
    local_used_ids = set(used_ids_snapshot)
    
    max_id_len = max(len(str(i)) for i in unique_ids)
    max_name_len = max(len(name) for name in items_dict.values()) if items_dict else 0
    ok_count = bad_count = 0
    repair_entries = []
    
    # Collect conflicts first
    conflicts = []
    
    for item_id in unique_ids:
        id_str = str(item_id).rjust(max_id_len)
        if item_id in items_dict:
            name = items_dict[item_id].ljust(max_name_len)
            print(f"{id_str} : {name} {Fore.RED}[BAD]{Style.RESET_ALL}")
            bad_count += 1
            if mode == "repair":
                conflicts.append(item_id)
        else:
            print(f"{id_str} : {'available'.ljust(max_name_len)} {Fore.GREEN}[OK]{Style.RESET_ALL}")
            ok_count += 1
    
    # NEW: Use smart algorithm to find all available IDs at once
    if mode == "repair" and conflicts:
        print(f"\n{'='*60}")
        print(f"Searching for {len(conflicts)} available IDs in range [1, 5000]...")
        print(f"{'='*60}")
        
        try:
            # Use smart algorithm with range 1-5000
            available_ids = find_available_ids_in_range(
                local_used_ids,
                len(conflicts),
                min_id=1,
                max_id=5000
            )
            
            print(f"\n[SUCCESS] Found {len(available_ids)} available IDs")
            if available_ids:
                id_range_str = f"{min(available_ids)}-{max(available_ids)}" if len(available_ids) > 1 else str(available_ids[0])
                print(f"ID range: {id_range_str}")
            
            # Check if IDs form continuous block
            if len(available_ids) > 1:
                is_continuous = available_ids == list(range(min(available_ids), max(available_ids) + 1))
                if is_continuous:
                    print(f"Type: Continuous block")
                else:
                    print(f"Type: Scattered IDs (using gaps in ID space)")
            
            print(f"{'='*60}\n")
            
            # Create repair entries
            print("Repair plan:")
            for old_id, new_id in zip(conflicts, available_ids):
                # Show the mapping
                conflict_name = items_dict.get(old_id, 'Unknown')
                print(f"  {old_id} -> {new_id} ({conflict_name})")
                repair_entries.append((old_id, new_id))
                local_used_ids.add(new_id)
            
        except ValueError as e:
            print(f"\n[ERROR] {e}")
            print("\n" + "="*60)
            print("Cannot proceed with repair. Please choose one of these options:")
            print("="*60)
            print("  1. Remove some items from your DLC mod")
            print("  2. Use manual ID assignment:")
            print("     python resolve_id_conflicts_in_kurodlc.py repair --export")
            print("     (then edit the mapping file to use higher IDs)")
            print("  3. Contact for help if you need assistance")
            print("="*60 + "\n")
            return ok_count, bad_count, len(unique_ids), []
    
    return ok_count, bad_count, len(unique_ids), repair_entries

# -------------------------
# Apply repair
# -------------------------
def apply_repair(repair_entries, timestamp):
    """
    Apply repair changes to DLC files.
    
    FIXED: 
    - Correctly handles DLCTableData.items (list of integers)
    - Only logs when ID is actually found and changed
    - Handles all four sections properly
    """
    repair_per_file = {}
    for file_name, old_id, new_id in repair_entries:
        repair_per_file.setdefault(file_name, []).append((old_id, new_id))

    for file_name, changes in repair_per_file.items():
        if ".bak_" in file_name:
            continue

        backup_file = f"{file_name}.bak_{timestamp}.json"
        shutil.copy2(file_name, backup_file)

        verbose_filename = f"{file_name}.repair_verbose_{timestamp}.txt"
        verbose_lines = []

        # Header at the top
        verbose_lines.append(f"File       : {file_name}")
        verbose_lines.append(f"Backup     : {backup_file}")
        verbose_lines.append(f"Verbose log: {verbose_filename}\n")
        verbose_lines.append("-"*60)

        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for old_id, new_id in changes:
            block_lines = []
            
            # CostumeParam
            if 'CostumeParam' in data:
                for item in data['CostumeParam']:
                    if item.get('item_id') == old_id:
                        mdl_name = item.get('mdl_name', '')
                        item['item_id'] = new_id
                        block_lines.append(f"CostumeParam : {str(old_id).rjust(5)} -> {str(new_id).rjust(5)}, mdl_name: {mdl_name}")
            
            # ShopItem (optional section)
            if 'ShopItem' in data:
                for item in data['ShopItem']:
                    if item.get('item_id') == old_id:
                        shop_id = item.get('shop_id', '')
                        item['item_id'] = new_id
                        block_lines.append(f"ShopItem     : {str(old_id).rjust(5)} -> {str(new_id).rjust(5)}, shop_id : {shop_id}")
            
            # ItemTableData
            if 'ItemTableData' in data:
                for item in data['ItemTableData']:
                    if item.get('id') == old_id:
                        name = item.get('name', '')
                        item['id'] = new_id
                        block_lines.append(f"ItemTableData: {str(old_id).rjust(5)} -> {str(new_id).rjust(5)}, name    : {name}")
            
            # DLCTableData - FIXED: Only log if ID was actually found
            if 'DLCTableData' in data:
                for item in data['DLCTableData']:
                    if 'items' in item and isinstance(item['items'], list):
                        if old_id in item['items']:
                            item['items'] = [new_id if x == old_id else x for x in item['items']]
                            dlc_name = item.get('name', '')
                            block_lines.append(f"DLCTableData : {str(old_id).rjust(5)} -> {str(new_id).rjust(5)}, name    : {dlc_name}")

            if block_lines:
                verbose_lines.append("\n".join(block_lines))
                verbose_lines.append("-"*60)

        # Write updated JSON
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        # Save verbose log
        with open(verbose_filename, 'w', encoding='utf-8') as vf:
            for line in verbose_lines:
                vf.write(line + "\n")

        print(f"File       : {file_name}")
        print(f"Backup     : {backup_file}")
        print(f"Verbose log: {verbose_filename}")
        print("-"*60)

# -------------------------
# Export ID mapping to JSON
# -------------------------
def export_id_mapping(repair_entries, items_dict, timestamp, custom_name=None, source_info=None):
    """
    Export repair plan to JSON file for manual editing.
    
    Args:
        repair_entries: List of (file_name, old_id, new_id) tuples
        items_dict: Dictionary of game items
        timestamp: Timestamp string for filename
        custom_name: Optional custom name (already validated, in id_mapping_*.json format)
        source_info: Optional dict with source information (type, path)
    
    Creates id_mapping.json with:
    - old_id: conflicting ID
    - new_id: suggested replacement (can be edited by user)
    - conflict_name: name of conflicting item in game
    - files: list of affected DLC files
    - source: information about the source used for conflict detection
    
    Returns:
        Created filename or (None, errors) on failure
    """
    # Use custom name if provided, otherwise use timestamp
    if custom_name:
        mapping_file = custom_name
    else:
        # Default: use timestamp
        mapping_file = f"id_mapping_{timestamp}.json"
    
    # Group by old_id and count occurrences
    id_groups = {}
    for file_name, old_id, new_id in repair_entries:
        if old_id not in id_groups:
            # Count how many times this ID appears in the file
            id_sections = extract_item_ids_with_sections(file_name)
            if id_sections is None:
                print(f"[WARNING] Cannot read {file_name}, skipping occurrence count for ID {old_id}")
                occurrences = 0  # Unknown
            else:
                occurrences = len(id_sections.get(old_id, []))
            
            id_groups[old_id] = {
                'old_id': old_id,
                'new_id': new_id,  # User can edit this
                'conflict_name': items_dict.get(old_id, 'Unknown'),
                'occurrences': occurrences,  # NEW: Track number of occurrences
                'files': []
            }
        id_groups[old_id]['files'].append(file_name)
    
    # Convert to list and sort by old_id
    mapping_list = sorted(id_groups.values(), key=lambda x: x['old_id'])
    
    # Create output structure
    output = {
        '_comment': [
            "ID Mapping File - Generated by resolve_id_conflicts_in_kurodlc.py",
            "",
            "INSTRUCTIONS:",
            "1. Review each mapping below",
            "2. Edit 'new_id' values as needed (must be unique and not conflict with game)",
            "3. Save this file",
            "4. Run: python resolve_id_conflicts_in_kurodlc.py repair --import",
            "",
            "IMPORTANT:",
            "- Do NOT change 'old_id' values!",
            "- Do NOT change 'occurrences' values!",
            "- Do NOT change 'files' list!",
            "- You CAN change 'new_id' values (must be unique and not conflict)",
            "- Make sure 'new_id' values are unique",
            "- Make sure 'new_id' values don't conflict with game items",
            "- The script will validate your changes before applying",
            "- Do NOT modify the 'source' section below",
            "- Do NOT manually edit .kurodlc.json files between export and import!",
            "",
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total conflicts: {len(mapping_list)}"
        ],
        'source': source_info if source_info else {
            'type': 'unknown',
            'path': 'unknown',
            'note': 'Source information not available'
        },
        'mappings': mapping_list
    }
    
    try:
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
    except Exception as e:
        return None, [f"Error creating mapping file: {e}"]
    
    print(f"\n{'='*60}")
    print(f"ID mapping exported to: {mapping_file}")
    print(f"{'='*60}")
    print(f"\nFound {len(mapping_list)} conflicting IDs")
    if source_info:
        print(f"Source used: {source_info.get('type', 'unknown')} ({source_info.get('path', 'unknown')})")
    print("\nNext steps:")
    print(f"1. Edit {mapping_file} - change 'new_id' values as needed")
    print(f"2. Run: python {sys.argv[0]} repair --import")
    print(f"\nThe script will:")
    print(f"  - Use the same source automatically (saved in mapping file)")
    print(f"  - Validate your changes before applying them")
    print(f"{'='*60}\n")
    
    return mapping_file, []

# -------------------------
# Validate mapping file structure
# -------------------------
def validate_mapping_structure(data, filename):
    """
    Validate the structure of imported mapping file.
    
    Args:
        data: Parsed JSON data
        filename: Name of the file (for error messages)
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Check root structure
    if not isinstance(data, dict):
        errors.append(f"Error: {filename} root must be a JSON object, got {type(data).__name__}")
        return errors
    
    # Check for 'mappings' key
    if 'mappings' not in data:
        errors.append(f"Error: {filename} missing required 'mappings' key")
        return errors
    
    # Check 'mappings' is a list
    if not isinstance(data['mappings'], list):
        errors.append(f"Error: 'mappings' must be a list, got {type(data['mappings']).__name__}")
        return errors
    
    # Check mappings is not empty
    if len(data['mappings']) == 0:
        errors.append(f"Error: 'mappings' list is empty - nothing to import")
        return errors
    
    # Validate each mapping entry structure
    required_fields = ['old_id', 'new_id', 'files']
    optional_fields = ['conflict_name']
    
    for idx, mapping in enumerate(data['mappings']):
        if not isinstance(mapping, dict):
            errors.append(f"Mapping #{idx+1}: Must be an object, got {type(mapping).__name__}")
            continue
        
        # Check required fields exist
        for field in required_fields:
            if field not in mapping:
                errors.append(f"Mapping #{idx+1}: Missing required field '{field}'")
        
        # Check field types
        if 'old_id' in mapping and not isinstance(mapping['old_id'], int):
            errors.append(f"Mapping #{idx+1}: 'old_id' must be integer, got {type(mapping['old_id']).__name__}")
        
        if 'new_id' in mapping and not isinstance(mapping['new_id'], int):
            errors.append(f"Mapping #{idx+1}: 'new_id' must be integer, got {type(mapping['new_id']).__name__}")
        
        if 'files' in mapping:
            if not isinstance(mapping['files'], list):
                errors.append(f"Mapping #{idx+1}: 'files' must be a list, got {type(mapping['files']).__name__}")
            elif len(mapping['files']) == 0:
                errors.append(f"Mapping #{idx+1}: 'files' list is empty")
            else:
                # Check all files are strings
                for file_idx, file_name in enumerate(mapping['files']):
                    if not isinstance(file_name, str):
                        errors.append(f"Mapping #{idx+1}, file #{file_idx+1}: Must be string, got {type(file_name).__name__}")
    
    if errors:
        errors.insert(0, f"\n{'='*60}")
        errors.insert(1, f"MAPPING FILE STRUCTURE VALIDATION FAILED: {filename}")
        errors.insert(2, f"{'='*60}\n")
        errors.append(f"\n{'='*60}")
        errors.append("Please fix the structure errors above.")
        errors.append("Required format:")
        errors.append("{")
        errors.append('  "mappings": [')
        errors.append("    {")
        errors.append('      "old_id": <integer>,')
        errors.append('      "new_id": <integer>,')
        errors.append('      "files": [<string>, ...]')
        errors.append("    }")
        errors.append("  ]")
        errors.append("}")
        errors.append(f"{'='*60}\n")
    
    return errors

# -------------------------
# Import and validate ID mapping
# -------------------------
def select_mapping_file_interactive(mapping_files):
    """
    Interactive selection of mapping file when multiple files found.
    
    Args:
        mapping_files: List of mapping file paths
        
    Returns:
        Selected file path or None if cancelled
    """
    print("\nMultiple ID mapping files detected.\n")
    print("Select mapping file to import:")
    
    # Sort files by timestamp (newest first for better UX)
    sorted_files = sorted(mapping_files, reverse=True)
    
    for i, file in enumerate(sorted_files, 1):
        # Try to extract timestamp from filename
        try:
            # Format: id_mapping_YYYYMMDD_HHMMSS.json or id_mapping_NAME.json
            if '_20' in file:  # Has timestamp
                parts = file.replace('id_mapping_', '').replace('.json', '').split('_')
                if len(parts) >= 2:
                    date_part = parts[0]
                    time_part = parts[1] if len(parts) > 1 else ''
                    # Format: YYYYMMDD -> YYYY-MM-DD
                    formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                    # Format: HHMMSS -> HH:MM:SS
                    formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}" if time_part else ''
                    display = f"{file} ({formatted_date} {formatted_time})".strip()
                else:
                    display = file
            else:
                display = file
        except:
            display = file
        
        print(f"  {i}) {display}")
    
    while True:
        choice = input(f"\nEnter choice [1-{len(sorted_files)}]: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(sorted_files):
                return sorted_files[idx]
        print("Invalid choice, try again.")

def import_id_mapping(items_dict, used_ids, mapping_file=None, no_interactive=False):
    """
    Import user-edited ID mapping and validate.
    
    Args:
        items_dict: Dictionary of game items
        used_ids: Set of already used IDs
        mapping_file: Optional path to specific mapping file
        no_interactive: If True, auto-select newest file without prompt
    
    Returns:
        (repair_entries, errors) tuple
        - repair_entries: list of (file_name, old_id, new_id) if valid
        - errors: list of error messages if validation failed
    """
    # If specific file not provided, find mapping files
    if mapping_file is None:
        mapping_files = [f for f in os.listdir('.') 
                        if f.startswith('id_mapping_') and f.endswith('.json')]
        
        if not mapping_files:
            return None, ["Error: No id_mapping_*.json file found in current directory"]
        
        if len(mapping_files) == 1:
            # Only one file - use it
            mapping_file = mapping_files[0]
            print(f"\nFound mapping file: {mapping_file}")
        elif no_interactive:
            # Auto-select newest (last in sorted list)
            mapping_file = sorted(mapping_files)[-1]
            print(f"\nAuto-selected most recent mapping file: {mapping_file}")
        else:
            # Multiple files - interactive selection
            mapping_file = select_mapping_file_interactive(mapping_files)
            if mapping_file is None:
                return None, ["Error: No mapping file selected"]
    else:
        # Validate that specified file exists
        if not os.path.exists(mapping_file):
            return None, [f"Error: Specified mapping file not found: {mapping_file}"]
        print(f"\nUsing specified mapping file: {mapping_file}")
    
    print(f"Importing ID mapping from: {mapping_file}\n")
    
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return None, [f"Error: Invalid JSON in {mapping_file}: {e}"]
    except Exception as e:
        return None, [f"Error loading {mapping_file}: {e}"]
    
    # VALIDATE MAPPING FILE STRUCTURE
    validation_errors = validate_mapping_structure(data, mapping_file)
    if validation_errors:
        return None, validation_errors
    
    mappings = data['mappings']
    
    # Validate mappings
    errors = []
    warnings = []
    repair_entries = []
    new_ids_used = set()
    
    print("Validating ID mappings...")
    print(f"{'='*60}\n")
    
    for idx, mapping in enumerate(mappings):
        # Check required fields
        if 'old_id' not in mapping:
            errors.append(f"Mapping #{idx+1}: Missing 'old_id'")
            continue
        if 'new_id' not in mapping:
            errors.append(f"Mapping #{idx+1}: Missing 'new_id'")
            continue
        if 'files' not in mapping:
            errors.append(f"Mapping #{idx+1}: Missing 'files'")
            continue
        
        old_id = mapping['old_id']
        new_id = mapping['new_id']
        files = mapping['files']
        
        # CRITICAL: Validate that old_id EXISTS in ALL listed files
        # AND appears in the SAME NUMBER of sections as at export time
        old_id_found_in_files = []
        old_id_missing_in_files = []
        occurrence_mismatches = []
        
        for file_name in files:
            if not os.path.exists(file_name):
                errors.append(f"ID {old_id}: File does not exist: {file_name}")
                continue
            
            # Get ID sections mapping
            id_sections = extract_item_ids_with_sections(file_name)
            
            # Check if file loading failed
            if id_sections is None:
                errors.append(
                    f"ID {old_id}: Cannot read file {file_name}\n"
                    f"      File may have invalid JSON or be corrupted.\n"
                    f"      Please check the file and fix any JSON syntax errors."
                )
                continue
            
            if old_id in id_sections:
                # ID exists - check number of occurrences
                current_sections = id_sections[old_id]
                num_occurrences = len(current_sections)
                
                # Store expected occurrences from mapping (if available)
                expected_occurrences = mapping.get('occurrences', None)
                
                if expected_occurrences and num_occurrences != expected_occurrences:
                    # Number of occurrences changed!
                    occurrence_mismatches.append({
                        'file': file_name,
                        'expected': expected_occurrences,
                        'found': num_occurrences,
                        'sections': current_sections
                    })
                
                old_id_found_in_files.append(file_name)
            else:
                # ID doesn't exist at all
                old_id_missing_in_files.append(file_name)
        
        # Check for occurrence mismatches
        if occurrence_mismatches:
            error_msg = f"ID {old_id}: Number of occurrences changed!"
            for mismatch in occurrence_mismatches:
                error_msg += f"\n      File: {mismatch['file']}"
                error_msg += f"\n      Expected: {mismatch['expected']} occurrence(s)"
                error_msg += f"\n      Found: {mismatch['found']} occurrence(s)"
                error_msg += f"\n      Current sections: {', '.join(mismatch['sections'])}"
            error_msg += f"\n      Possible cause: Manual changes to file - ID removed from some sections."
            error_msg += f"\n      Solution: Either restore ALL occurrences or create a new export."
            errors.append(error_msg)
            continue
        
        # STRICT CHECK: old_id must exist in ALL listed files
        if old_id_missing_in_files:
            # This is ALWAYS an error - old_id must be in ALL files
            error_msg = f"ID {old_id}: Mismatch detected! This ID is missing in some files."
            error_msg += f"\n      Missing in: {', '.join(old_id_missing_in_files)}"
            if old_id_found_in_files:
                error_msg += f"\n      Found in: {', '.join(old_id_found_in_files)}"
            error_msg += f"\n      Possible cause: Manual changes to files between export and import."
            error_msg += f"\n      Solution: Either restore the original IDs or create a new export."
            errors.append(error_msg)
            continue
        
        if not old_id_found_in_files:
            # old_id not found in ANY file
            errors.append(
                f"ID {old_id}: Not found in any listed file(s)!\n"
                f"      Listed files: {', '.join(files)}\n"
                f"      Possible cause: IDs were manually changed or removed since export.\n"
                f"      Solution: Create a new export to get current state."
            )
            continue
        
        # Validate old_id is actually a conflict with game
        if old_id not in items_dict:
            warnings.append(f"ID {old_id}: Not in conflict with game (may have been manually fixed)")
            continue
        
        # Validate new_id type
        if not isinstance(new_id, int):
            errors.append(f"ID {old_id}: new_id must be an integer, got {type(new_id).__name__}")
            continue
        
        # Validate new_id doesn't conflict with game
        if new_id in items_dict:
            conflict_name = items_dict[new_id]
            errors.append(f"ID {old_id} -> {new_id}: CONFLICT! new_id {new_id} already exists in game as '{conflict_name}'")
            continue
        
        # Validate new_id is not already used in DLCs (except if remapping to same ID)
        if new_id in used_ids and new_id != old_id:
            errors.append(f"ID {old_id} -> {new_id}: CONFLICT! new_id {new_id} already used in DLC files")
            continue
        
        # Validate new_id is not used twice in mapping
        if new_id in new_ids_used and new_id != old_id:
            errors.append(f"ID {old_id} -> {new_id}: DUPLICATE! new_id {new_id} is used multiple times in mapping")
            continue
        
        new_ids_used.add(new_id)
        
        # Add to repair entries
        for file_name in files:
            repair_entries.append((file_name, old_id, new_id))
        
        # Green [OK] like in checkbydlc, without file count
        print(f"\033[92m[OK]\033[0m {old_id} -> {new_id} ('{mapping.get('conflict_name', 'Unknown')}')")
    
    print(f"\n{'='*60}")
    
    # Show warnings
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  [!] {warning}")
    
    # Show errors with detailed summary
    if errors:
        print(f"\n{'='*60}")
        print(f"[ERROR] VALIDATION FAILED - Found {len(errors)} issue(s)")
        print(f"{'='*60}")
        print("\nCannot proceed with import due to inconsistencies between")
        print("mapping file and current state of .kurodlc.json files.")
        print("\nDetails:")
        print("-" * 60)
        
        for i, error in enumerate(errors, 1):
            print(f"\nIssue #{i}:")
            # Print with proper indentation for multi-line errors
            for line in error.split('\n'):
                print(f"  {line}")
        
        print("\n" + "="*60)
        print("POSSIBLE CAUSES:")
        print("  1. Files were manually edited between export and import")
        print("  2. IDs were changed or removed in .kurodlc.json files")
        print("  3. Wrong mapping file selected")
        print("\nRECOMMENDED SOLUTIONS:")
        print("  1. Restore original .kurodlc.json files from backup")
        print("  2. Create a new export with current file state:")
        print(f"     python {sys.argv[0]} repair --export")
        print("  3. Manually fix the IDs mentioned above")
        print("="*60 + "\n")
        
        return None, errors
    
    print(f"\n[SUCCESS] Validation PASSED!")
    print(f"All {len(repair_entries)} ID mappings verified successfully.")
    print(f"Ready to modify {len(set(f for f, _, _ in repair_entries))} file(s).")
    print(f"{'='*60}\n")
    
    return repair_entries, []


# -------------------------
# MAIN
# -------------------------
if len(sys.argv) < 2:
    print_usage()
    sys.exit(0)

arg = sys.argv[1].lower()
options = sys.argv[2:]
files = get_all_files()

if arg not in ("checkbydlc", "repair"):
    print("Only modes 'checkbydlc' and 'repair' are supported.")
    sys.exit(1)

keep_extracted = "--keep-extracted" in options
no_interactive = "--no-interactive" in options
apply_changes = "--apply" in options
export_mapping = "--export" in options
import_mapping = "--import" in options
mapping_file_path = None
export_name = None
forced_source = None

# Validate conflicting options
if import_mapping and export_mapping:
    print("Error: Cannot use both --import and --export at the same time.")
    print("Use --export first to create mapping, then --import to apply it.")
    sys.exit(1)

if import_mapping and apply_changes:
    print("Note: --import already applies changes (--apply is implicit)")
    print("Proceeding with import and apply...\n")
    apply_changes = True  # Make explicit

# FIXED: Proper parsing of --source, --mapping-file, and --export-name parameters
for opt in options:
    if opt.startswith("--source"):
        if "=" in opt:
            _, forced_source = opt.split("=", 1)
        else:
            print("Error: --source requires format: --source=TYPE")
            print("Available types: json, tbl, original, p3a, zzz")
            sys.exit(1)
    elif opt.startswith("--mapping-file"):
        if "=" in opt:
            _, mapping_file_path = opt.split("=", 1)
        else:
            print("Error: --mapping-file requires format: --mapping-file=PATH")
            print("Example: --mapping-file=id_mapping_20260130_143022.json")
            sys.exit(1)
    elif opt.startswith("--export-name"):
        if "=" in opt:
            _, user_input = opt.split("=", 1)
            
            # IMPROVED: Auto-add prefix and suffix if user provides just the custom part
            # User can provide:
            #   - Just name: "DLC1" -> "id_mapping_DLC1.json"
            #   - Full name: "id_mapping_DLC1.json" -> "id_mapping_DLC1.json"
            
            # Validate that user input is not empty
            if not user_input or user_input.isspace():
                print("Error: --export-name cannot be empty")
                print("Example: --export-name=DLC1")
                print("   Will create: id_mapping_DLC1.json")
                sys.exit(1)
            
            # Check for invalid characters
            invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            for char in invalid_chars:
                if char in user_input:
                    print(f"Error: Export name contains invalid character '{char}': {user_input}")
                    print(f"Invalid characters: / \\ : * ? \" < > |")
                    sys.exit(1)
            
            # Auto-construct full filename
            if user_input.startswith('id_mapping_') and user_input.endswith('.json'):
                # User provided full name - use as is
                export_name = user_input
                print(f"Using full export name: {export_name}")
            else:
                # User provided just custom part - auto-add prefix and suffix
                # Remove .json if user added it
                custom_part = user_input.replace('.json', '')
                # Remove id_mapping_ prefix if user added it
                if custom_part.startswith('id_mapping_'):
                    custom_part = custom_part[11:]  # Remove 'id_mapping_'
                
                # Construct full name
                export_name = f"id_mapping_{custom_part}.json"
                print(f"Auto-generated export name: {export_name} (from: {user_input})")
        else:
            print("Error: --export-name requires format: --export-name=NAME")
            print("Examples:")
            print("  --export-name=DLC1              -> id_mapping_DLC1.json")
            print("  --export-name=test              -> id_mapping_test.json")
            print("  --export-name=id_mapping_DLC1.json  -> id_mapping_DLC1.json (full name)")
            sys.exit(1)

# -------------------------
# Source detection and selection (only for non-import modes)
# Import mode handles source loading separately from mapping file
# -------------------------
if not import_mapping:
    sources = detect_sources()
    if not sources:
        print("Error: No valid item source found.")
        sys.exit(1)

    extracted_temp = False
    temp_tbl = "t_item.tbl.original.tmp"
    used_source = None

    if forced_source:
        for stype, path in sources:
            if stype == forced_source:
                used_source = (stype, path)
                break
        if not used_source:
            print(f"Forced source '{forced_source}' not available.")
            print(f"Available sources: {', '.join(s[0] for s in sources)}")
            sys.exit(1)
    else:
        if len(sources) == 1 or no_interactive:
            used_source = sources[0]
        else:
            used_source = select_source_interactive(sources)

    stype, path = used_source

    # Check for required libraries if using P3A
    if stype in ("p3a", "zzz") and not HAS_LIBS:
        print(f"Error: Required library missing: {MISSING_LIB}")
        print("P3A extraction requires p3a_lib module.")
        sys.exit(1)

    # Load items_dict
    if stype == "json":
        items_dict = load_items_from_json()
        source_used = "t_item.json"
        source_info = {'type': 'json', 'path': 't_item.json'}
    elif stype in ("tbl", "original"):
        items_dict = load_items_from_tbl(path)
        source_used = path
        source_info = {'type': stype, 'path': path}
    elif stype in ("p3a", "zzz"):
        if extract_from_p3a(path, temp_tbl):
            extracted_temp = True
            items_dict = load_items_from_tbl(temp_tbl)
            source_used = f"{path} -> {temp_tbl}"
            source_info = {'type': stype, 'path': path, 'extracted': temp_tbl}
        else:
            print("Failed to extract t_item.tbl from P3A.")
            sys.exit(1)

    # Prepare used_ids set (game + DLC)
    # FIXED: Build complete set before processing any files
    used_ids = set(items_dict.keys())
    for f in files:
        used_ids.update(extract_item_ids(f))

# -------------------------
# Handle --import mode separately (loads source from mapping file)
# -------------------------
if arg == "repair" and import_mapping:
    print("\n" + "="*60)
    print("MODE: Import ID mapping from file")
    print("="*60 + "\n")
    
    # First, load the mapping file to get source information
    print("Step 1: Loading mapping file...")
    
    # Find mapping file
    if mapping_file_path is None:
        mapping_files = [f for f in os.listdir('.') 
                        if f.startswith('id_mapping_') and f.endswith('.json')]
        
        if not mapping_files:
            print("\nError: No id_mapping_*.json file found in current directory")
            print("\nPlease run export first:")
            print(f"  python {sys.argv[0]} repair --export")
            sys.exit(1)
        
        if len(mapping_files) == 1:
            mapping_file_path = mapping_files[0]
            print(f"Found single mapping file: {mapping_file_path}")
        elif no_interactive:
            mapping_file_path = sorted(mapping_files)[-1]
            print(f"Auto-selected most recent: {mapping_file_path}")
        else:
            mapping_file_path = select_mapping_file_interactive(mapping_files)
            print(f"Selected: {mapping_file_path}")
    else:
        if not os.path.exists(mapping_file_path):
            print(f"\nError: Specified mapping file not found: {mapping_file_path}")
            
            # Suggest available files
            available_files = [f for f in os.listdir('.') 
                             if f.startswith('id_mapping_') and f.endswith('.json')]
            
            if available_files:
                print("\n" + "="*60)
                print("Available mapping files in current directory:")
                print("="*60)
                sorted_files = sorted(available_files, reverse=True)
                for i, file in enumerate(sorted_files, 1):
                    print(f"  {i}) {file}")
                
                # Simplified selection like source selection
                if not no_interactive:
                    while True:
                        choice = input(f"\nEnter choice [1-{len(sorted_files)}]: ").strip()
                        if not choice:  # Empty = exit
                            print("Exiting.")
                            sys.exit(1)
                        if choice.isdigit():
                            idx = int(choice) - 1
                            if 0 <= idx < len(sorted_files):
                                mapping_file_path = sorted_files[idx]
                                print(f"\nSelected: {mapping_file_path}")
                                break
                        print("Invalid choice, try again.")
                else:
                    print("\nTry one of these files:")
                    print(f"  python {sys.argv[0]} repair --import --mapping-file=FILENAME")
                    print("Or run without --mapping-file for interactive selection:")
                    print(f"  python {sys.argv[0]} repair --import")
                    print("="*60)
                    sys.exit(1)
            else:
                sys.exit(1)
        else:
            print(f"Using specified file: {mapping_file_path}")
    
    # Load mapping file to extract source info
    try:
        with open(mapping_file_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"\nError: Invalid JSON in {mapping_file_path}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError loading {mapping_file_path}: {e}")
        sys.exit(1)
    
    # Validate structure first
    structure_errors = validate_mapping_structure(mapping_data, mapping_file_path)
    if structure_errors:
        for error in structure_errors:
            print(error)
        sys.exit(1)
    
    # Extract source information from mapping file
    saved_source = mapping_data.get('source', {})
    
    if saved_source and saved_source.get('type') != 'unknown':
        print(f"\nStep 2: Loading item source from mapping file...")
        print(f"Source type: {saved_source.get('type')}")
        print(f"Source path: {saved_source.get('path')}")
        
        # Override forced_source and source selection with saved source
        stype = saved_source.get('type')
        path = saved_source.get('path')
        
        # Validate that source still exists
        if stype == 'json':
            if not os.path.exists(path):
                print(f"\nWarning: Saved source '{path}' not found!")
                print("Falling back to source detection...")
                # Fall back to normal source detection
                sources = detect_sources()
                if not sources:
                    print("Error: No valid item source found.")
                    sys.exit(1)
                if len(sources) == 1 or no_interactive:
                    stype, path = sources[0]
                else:
                    stype, path = select_source_interactive(sources)
        elif stype in ('tbl', 'original'):
            if not os.path.exists(path):
                print(f"\nWarning: Saved source '{path}' not found!")
                print("Falling back to source detection...")
                sources = detect_sources()
                if not sources:
                    print("Error: No valid item source found.")
                    sys.exit(1)
                if len(sources) == 1 or no_interactive:
                    stype, path = sources[0]
                else:
                    stype, path = select_source_interactive(sources)
        elif stype in ('p3a', 'zzz'):
            if not os.path.exists(path):
                print(f"\nWarning: Saved source '{path}' not found!")
                print("Falling back to source detection...")
                sources = detect_sources()
                if not sources:
                    print("Error: No valid item source found.")
                    sys.exit(1)
                if len(sources) == 1 or no_interactive:
                    stype, path = sources[0]
                else:
                    stype, path = select_source_interactive(sources)
            
            # Check for required libraries
            if not HAS_LIBS:
                print(f"Error: Required library missing: {MISSING_LIB}")
                print("P3A extraction requires p3a_lib module.")
                sys.exit(1)
    else:
        print(f"\nStep 2: No source information in mapping file, detecting sources...")
        sources = detect_sources()
        if not sources:
            print("Error: No valid item source found.")
            sys.exit(1)
        
        if len(sources) == 1 or no_interactive:
            stype, path = sources[0]
        else:
            stype, path = select_source_interactive(sources)
    
    # Load items_dict from selected source
    extracted_temp = False
    temp_tbl = "t_item.tbl.original.tmp"
    
    if stype == "json":
        items_dict = load_items_from_json()
        source_used = "t_item.json"
    elif stype in ("tbl", "original"):
        items_dict = load_items_from_tbl(path)
        source_used = path
    elif stype in ("p3a", "zzz"):
        if extract_from_p3a(path, temp_tbl):
            extracted_temp = True
            items_dict = load_items_from_tbl(temp_tbl)
            source_used = f"{path} -> {temp_tbl}"
        else:
            print("Failed to extract t_item.tbl from P3A.")
            sys.exit(1)
    
    print(f"Source loaded: {source_used}\n")
    
    # Prepare used_ids set
    used_ids = set(items_dict.keys())
    for f in files:
        all_ids = extract_item_ids(f)
        used_ids.update(all_ids)
    
    # Now validate and import mappings
    print("Step 3: Validating ID mappings...")
    imported_entries, import_errors = import_id_mapping(items_dict, used_ids, mapping_file_path, no_interactive)
    
    if import_errors:
        print("\nImport validation failed:")
        for error in import_errors:
            print(f"  {error}")
        print("\nPlease fix the errors and try again.")
        sys.exit(1)
    
    if not imported_entries:
        print("No valid mappings found to apply.")
        sys.exit(0)
    
    # Apply the imported mapping
    print("\nStep 4: Applying imported ID mappings...\n")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    apply_repair(imported_entries, timestamp)
    print("\n[SUCCESS] All changes from imported mapping applied successfully with backups.")
    
    # Cleanup
    if extracted_temp and not keep_extracted:
        os.remove(temp_tbl)
        print(f"Cleaned up temporary file: {temp_tbl}")
    
    sys.exit(0)

# Normal processing (check or auto-generated repair)
total_ok = total_bad = total_ids = 0
repair_log = []
all_repair_entries = []

for f in files:
    print(f"\nProcessing file: {f}\n")
    all_item_ids = extract_item_ids(f)
    
    # FIXED: Pass immutable snapshot, don't mutate used_ids
    ok, bad, total, repair_entries = print_ids_for_list(
        all_item_ids, items_dict, used_ids, arg
    )
    
    print("\nSummary for this file:")
    print(f"Total IDs : {total}")
    print(f"OK        : {ok}")
    print(f"BAD       : {bad}")
    
    total_ok += ok
    total_bad += bad
    total_ids += total
    
    if repair_entries:
        for old, new in repair_entries:
            repair_log.append(f"{f}: {old} -> {new}")
            all_repair_entries.append((f, old, new))
            # Add new ID to global set for next files
            used_ids.add(new)

print("\nOverall Summary:")
print(f"Total IDs : {total_ids}")
print(f"OK        : {total_ok}")
print(f"BAD       : {total_bad}")
print(f"\nSource used for check: {source_used}")

# Write repair log
if arg == "repair" and repair_log:
    with open("repair_log.txt", "w", encoding="utf-8") as f:
        for line in repair_log:
            f.write(line + "\n")
    print("\n" + "-"*60)
    print("Repair log generated: repair_log.txt")
    print("-"*60 + "\n")

# Export mapping if requested
if arg == "repair" and export_mapping and all_repair_entries:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result = export_id_mapping(all_repair_entries, items_dict, timestamp, export_name, source_info)
    
    # Handle potential errors from export
    if isinstance(result, tuple):
        mapping_file, errors = result
        if errors:
            print("\n" + "="*60)
            print("Export failed:")
            for error in errors:
                print(f"  {error}")
            print("="*60 + "\n")
            sys.exit(1)
    else:
        # Legacy support if no errors (should not happen with new code)
        mapping_file = result

# Apply changes if requested (and not using import mode)
if arg == "repair" and apply_changes and all_repair_entries and not import_mapping:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    apply_repair(all_repair_entries, timestamp)
    print("\nAll changes applied with backups.")

# Cleanup
if extracted_temp and not keep_extracted:
    os.remove(temp_tbl)
    print(f"Cleaned up temporary file: {temp_tbl}")
