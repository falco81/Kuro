import json
import sys
import os
from glob import glob

# ------------------------------------------------------------
# Colorama for Windows CMD colors
# ------------------------------------------------------------
try:
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    USE_COLOR = True
except ImportError:
    USE_COLOR = False
    Fore = Style = type('', (), {'RED':'', 'GREEN':'', 'RESET_ALL':''})()

# ------------------------------------------------------------
# Import required libraries with error handling
# ------------------------------------------------------------
try:
    from p3a_lib import p3a_class
    from kurodlc_lib import kuro_tables
    HAS_LIBS = True
except ImportError as e:
    HAS_LIBS = False
    MISSING_LIB = str(e)

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def get_all_files():
    return [f for f in os.listdir('.') if f.lower().endswith('.kurodlc.json')]

def extract_item_ids(json_file, strict=False, cmdlog=False):
    """
    Extract item_ids from all relevant sections.
    
    FIXED: Now correctly extracts from all four sections:
    - CostumeParam: uses 'item_id' field
    - ItemTableData: uses 'id' field
    - DLCTableData: uses 'items' field (list of integers)
    - ShopItem: uses 'item_id' field (optional section)
    """
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        if cmdlog:
            print(f"Skipping {json_file}: invalid JSON ({e})")
        return []

    if not is_valid_kurodlc_structure(data):
        msg = f"{json_file} has invalid kurodlc structure"
        if strict:
            raise ValueError(msg)
        if cmdlog:
            print(f"Skipping {json_file}: {msg}")
        return []

    ids = []
    
    # CostumeParam: item_id field
    if 'CostumeParam' in data:
        for item in data['CostumeParam']:
            if isinstance(item, dict) and 'item_id' in item:
                ids.append(item['item_id'])
    
    # ItemTableData: id field
    if 'ItemTableData' in data:
        for item in data['ItemTableData']:
            if isinstance(item, dict) and 'id' in item:
                ids.append(item['id'])
    
    # DLCTableData: items field (list of integers)
    if 'DLCTableData' in data:
        for item in data['DLCTableData']:
            if isinstance(item, dict) and 'items' in item and isinstance(item['items'], list):
                ids.extend(item['items'])
  
    return ids

def is_valid_kurodlc_structure(data):
    """
    Validate .kurodlc.json structure.
    
    FIXED: ItemTableData and ShopItem are now OPTIONAL.
    Only CostumeParam and DLCTableData are required.
    """
    if not isinstance(data, dict):
        return False

    # Only CostumeParam and DLCTableData are REQUIRED
    required_root_keys = ["CostumeParam", "DLCTableData"]
    if not all(k in data for k in required_root_keys):
        return False

    if not all(isinstance(data[k], list) for k in required_root_keys):
        return False

    # CostumeParam → item_id
    if not any(
        isinstance(x, dict) and "item_id" in x and isinstance(x["item_id"], int)
        for x in data["CostumeParam"]
    ):
        return False

    # DLCTableData → items[]
    if not any(
        isinstance(x, dict)
        and "items" in x
        and isinstance(x["items"], list)
        and all(isinstance(i, int) for i in x["items"])
        for x in data["DLCTableData"]
    ):
        return False

    # ItemTableData → id (OPTIONAL)
    if "ItemTableData" in data:
        if not isinstance(data["ItemTableData"], list):
            return False
        if data["ItemTableData"] and not any(
            isinstance(x, dict) and "id" in x and isinstance(x["id"], int)
            for x in data["ItemTableData"]
        ):
            return False

    # ShopItem → item_id (OPTIONAL)
    if "ShopItem" in data:
        if not isinstance(data["ShopItem"], list):
            return False

    return True


# ------------------------------------------------------------
# Source detection & selection (CHECK MODE)
# ------------------------------------------------------------

def detect_sources():
    sources = []

    if os.path.exists("t_item.json"):
        sources.append(("json", "t_item.json"))

    if os.path.exists("t_item.tbl.original"):
        sources.append(("original", "t_item.tbl.original"))

    if os.path.exists("t_item.tbl"):
        sources.append(("tbl", "t_item.tbl"))

    if os.path.exists("script_en.p3a"):
        sources.append(("p3a", "script_en.p3a"))

    if os.path.exists("script_eng.p3a"):
        sources.append(("p3a", "script_eng.p3a"))

    if os.path.exists("zzz_combined_tables.p3a"):
        sources.append(("zzz", "zzz_combined_tables.p3a"))

    return sources

def select_source_interactive(sources):
    print("\nMultiple item sources detected.\n")
    print("Select source to use for check:")

    for i, (stype, name) in enumerate(sources, 1):
        if stype in ("p3a", "zzz"):
            print(f"  {i}) {name} (extract t_item.tbl.original.tmp)")
        else:
            print(f"  {i}) {name}")

    while True:
        choice = input(f"\nEnter choice [1-{len(sources)}]: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(sources):
                return sources[idx]
        print("Invalid choice, try again.")

# ------------------------------------------------------------
# Extraction from P3A
# ------------------------------------------------------------

def extract_from_p3a(p3a_file, out_file):
    """Extract t_item.tbl from P3A archive."""
    p3a = p3a_class()
    with open(p3a_file, 'rb') as p3a.f:
        headers, entries, p3a_dict = p3a.read_p3a_toc()
        for entry in entries:
            if os.path.basename(entry['name']) == 't_item.tbl':
                data = p3a.read_file(entry, p3a_dict)
                with open(out_file, 'wb') as f:
                    f.write(data)
                return True
    return False

# ------------------------------------------------------------
# Load item table
# ------------------------------------------------------------

def load_items_from_json():
    with open('t_item.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    for section in data.get("data", []):
        if section.get("name") == "ItemTableData":
            return {x['id']: x['name'] for x in section.get("data", [])}
    return {}

def load_items_from_tbl(tbl_file):
    """Load items from .tbl file."""
    kt = kuro_tables()
    table = kt.read_table(tbl_file)
    return {x['id']: x['name'] for x in table['ItemTableData']}

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if len(sys.argv) < 2:
    print("""
Usage: python script.py <mode> [options]

Modes:

  <file.kurodlc.json>
      Process a single .kurodlc.json file and print all unique item_ids.

  searchall
      Process all .kurodlc.json files in the current directory.
      Prints a single sorted list of unique item_ids.

  searchallbydlc
      Process all .kurodlc.json files.
      For each file, prints its item_ids as a list.
      After all files, prints the unique item_ids across all files.

  searchallbydlcline
      Similar to searchallbydlc but prints each item_id on a separate line.
      Also prints unique item_ids at the end, each on a separate line.

  searchallline
      Process all .kurodlc.json files and print each unique item_id on a separate line.
      Simple list without grouping by file.

  check
      Special mode to check if item_ids in .kurodlc.json files are already assigned in the game data.
      Supports multiple sources: t_item.json, t_item.tbl, t_item.tbl.original,
      script_en.p3a, script_eng.p3a, zzz_combined_tables.p3a.

      The script will detect all available sources automatically. If multiple sources are found:
        - Interactive selection is prompted (default)
        - Use --no-interactive to skip prompt and pick the first detected source
        - Use --source=<type> to force a specific source

Check Mode Options:

  --source=<type>
      Force the source to use for check.
      Allowed values:
        json      : use t_item.json
        tbl       : use t_item.tbl
        original  : use t_item.tbl.original
        p3a       : use script_en.p3a or script_eng.p3a (will extract t_item.tbl.original.tmp)
        zzz       : use zzz_combined_tables.p3a (will extract t_item.tbl.original.tmp)

  --no-interactive
      Do not prompt user for source selection.
      If multiple sources exist, the first detected source will be used automatically.

  --keep-extracted
      If using a P3A source, the extracted temporary file t_item.tbl.original.tmp will be kept instead of deleted after check.

Check Mode Output:

  For each unique item_id found in .kurodlc.json files:
    <item_id> : <name> [OK/BAD]

    [OK]   = item_id is available (not present in game data)
    [BAD]  = item_id is assigned (exists in game data)

  Summary at the end:
    Total IDs : <total number of unique item_ids>
    OK        : <number of available IDs>
    BAD       : <number of assigned IDs>
    Source used for check: <actual source file used>

Examples:

  python script.py costume1.kurodlc.json
  python script.py searchall
  python script.py searchallbydlc
  python script.py searchallbydlcline
  python script.py check
  python script.py check --source=json
  python script.py check --source=p3a --keep-extracted
""")
    sys.exit(1)

arg = sys.argv[1].lower()
options = sys.argv[2:]

# ------------------------------------------------------------
# CHECK MODE
# ------------------------------------------------------------

if arg == "check":
    keep_extracted = "--keep-extracted" in options
    no_interactive = "--no-interactive" in options

    forced_source = None
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
    
    # Check if required libraries are available for P3A sources
    if any(stype in ("p3a", "zzz") for stype, _ in sources) and not HAS_LIBS:
        if forced_source in ("p3a", "zzz") or not forced_source:
            print(f"Error: Required library missing: {MISSING_LIB}")
            print("P3A extraction requires p3a_lib and kurodlc_lib modules.")
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
            print("Forced source not available.")
            sys.exit(1)
    else:
        if len(sources) == 1 or no_interactive:
            used_source = sources[0]
        else:
            used_source = select_source_interactive(sources)

    stype, path = used_source

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
            source_used = f"{path} → {temp_tbl}"
        else:
            print("Failed to extract t_item.tbl from P3A.")
            sys.exit(1)

    # Collect IDs
    all_item_ids = []
    for f in get_all_files():
        all_item_ids.extend(extract_item_ids(f))

    unique_ids = sorted(set(all_item_ids))

    max_id_len = max(len(str(i)) for i in unique_ids)
    max_name_len = max(len(name) for name in items_dict.values()) if items_dict else 0

    ok_count = 0
    bad_count = 0

    for item_id in unique_ids:
        id_str = str(item_id).rjust(max_id_len)
        if item_id in items_dict:
            name = items_dict[item_id].ljust(max_name_len)
            print(f"{id_str} : {name} {Fore.RED}[BAD]{Style.RESET_ALL}")
            bad_count += 1
        else:
            print(f"{id_str} : {'available'.ljust(max_name_len)} {Fore.GREEN}[OK]{Style.RESET_ALL}")
            ok_count += 1

    print("\nSummary:")
    print(f"Total IDs : {len(unique_ids)}")
    print(f"OK        : {ok_count}")
    print(f"BAD       : {bad_count}")
    print(f"\nSource used for check: {source_used}")

    if extracted_temp and not keep_extracted:
        os.remove(temp_tbl)
        print(f"Cleaned up temporary file: {temp_tbl}")

    sys.exit(0)

# ------------------------------------------------------------
# OTHER MODES (unchanged)
# ------------------------------------------------------------

files = get_all_files()

if arg == "searchall":
    ids = []
    for f in files:
        ids.extend(extract_item_ids(f))
    print(sorted(set(ids)))

elif arg == "searchallbydlc":
    all_ids = []
    for f in files:
        ids = extract_item_ids(f)
        all_ids.extend(ids)
        print(f"{f}:")
        print(ids)
        print()
    print("Unique item_ids across all files:")
    print(sorted(set(all_ids)))

elif arg == "searchallbydlcline":
    all_ids = []
    for f in files:
        ids = extract_item_ids(f)
        all_ids.extend(ids)
        print(f"{f}:")
        for i in sorted(ids):
            print(i)
        print()
    print("Unique item_ids across all files:")
    for i in sorted(set(all_ids)):
        print(i)

elif arg == "searchallline":
    ids = []
    for f in files:
        ids.extend(extract_item_ids(f))
    for i in sorted(set(ids)):
        print(i)

else:
    print(sorted(set(extract_item_ids(sys.argv[1]))))
