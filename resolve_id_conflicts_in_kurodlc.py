#!/usr/bin/env python3
"""
resolve_id_conflicts_in_kurodlc.py

Description:
Detects and repairs item ID conflicts in .kurodlc.json files against game item data.

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
   - --apply actually modifies files with backups and verbose logs.

Options:
--apply            Apply changes to DLC files.
--keep-extracted   Keep temporary extracted t_item.tbl.original.tmp after P3A extraction.
--no-interactive   Automatically selects first source if multiple found.
--source=<type>    Force a source: json, tbl, original, p3a, zzz.

Usage examples:
  python resolve_id_conflicts_in_kurodlc.py checkbydlc
  python resolve_id_conflicts_in_kurodlc.py repair
  python resolve_id_conflicts_in_kurodlc.py repair --apply --keep-extracted
  python resolve_id_conflicts_in_kurodlc.py repair --source=json
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
    
    FIXED: Now correctly handles all four sections:
    - CostumeParam: uses 'item_id' field
    - ItemTableData: uses 'id' field
    - DLCTableData: uses 'items' field (list of integers)
    - ShopItem: uses 'item_id' field (optional section)
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
    
    next_id = max(local_used_ids, default=0) + 1
    
    for item_id in unique_ids:
        id_str = str(item_id).rjust(max_id_len)
        if item_id in items_dict:
            name = items_dict[item_id].ljust(max_name_len)
            print(f"{id_str} : {name} {Fore.RED}[BAD]{Style.RESET_ALL}")
            bad_count += 1
            if mode == "repair":
                # Find next available ID
                while next_id in local_used_ids:
                    next_id += 1
                new_id = next_id
                local_used_ids.add(new_id)
                next_id += 1
                repair_entries.append((item_id, new_id))
        else:
            print(f"{id_str} : {'available'.ljust(max_name_len)} {Fore.GREEN}[OK]{Style.RESET_ALL}")
            ok_count += 1
    
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
forced_source = None

# FIXED: Proper parsing of --source parameter
for opt in options:
    if opt.startswith("--source"):
        if "=" in opt:
            _, forced_source = opt.split("=", 1)
        else:
            print("Error: --source requires format: --source=TYPE")
            print("Available types: json, tbl, original, p3a, zzz")
            sys.exit(1)

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

# Prepare used_ids set (game + DLC)
# FIXED: Build complete set before processing any files
used_ids = set(items_dict.keys())
for f in files:
    used_ids.update(extract_item_ids(f))

# -------------------------
# Process files
# -------------------------
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

# Apply changes if requested
if arg == "repair" and apply_changes and all_repair_entries:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    apply_repair(all_repair_entries, timestamp)
    print("\nAll changes applied with backups.")

# Cleanup
if extracted_temp and not keep_extracted:
    os.remove(temp_tbl)
    print(f"Cleaned up temporary file: {temp_tbl}")
