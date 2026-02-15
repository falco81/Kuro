#!/usr/bin/env python3
"""
find_all_items.py - Standalone version with integrated multi-source support

Search and display items from multiple source formats.
Optionally enriches results with CostumeParam model names from t_costume.

Supported sources:
- t_item.json
- t_item.tbl
- t_item.tbl.original
- script_en.p3a / script_eng.p3a (extracts t_item.tbl)
- zzz_combined_tables.p3a (extracts t_item.tbl)

Optional enrichment (auto-detected):
- t_costume.json / t_costume.tbl / t_costume.tbl.original / P3A
  Adds [mdl_name] to costume items in search results.
"""

import sys
import os
import json
import re

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
# Data loading functions (integrated from data_loader.py)
# -------------------------

def detect_sources(base_name='t_item'):
    """Detect available data sources for items."""
    sources = []
    json_file = f"{base_name}.json"
    tbl_original = f"{base_name}.tbl.original"
    tbl_file = f"{base_name}.tbl"
    
    if os.path.exists(json_file):
        sources.append(('json', json_file))
    if os.path.exists(tbl_original):
        sources.append(('original', tbl_original))
    if os.path.exists(tbl_file):
        sources.append(('tbl', tbl_file))
    if os.path.exists("script_en.p3a"):
        sources.append(('p3a', 'script_en.p3a'))
    if os.path.exists("script_eng.p3a"):
        sources.append(('p3a', 'script_eng.p3a'))
    if os.path.exists("zzz_combined_tables.p3a"):
        sources.append(('zzz', 'zzz_combined_tables.p3a'))
    
    return sources


def select_source_interactive(sources):
    """Let user select a source interactively."""
    print("\nMultiple data sources detected. Select source to use:")
    for i, (stype, path) in enumerate(sources, 1):
        if stype in ('p3a', 'zzz'):
            print(f"  {i}) {path} (extract t_item.tbl)")
        else:
            print(f"  {i}) {path}")
    
    while True:
        try:
            choice = input(f"\nEnter choice [1-{len(sources)}]: ").strip()
            idx = int(choice)
            if 1 <= idx <= len(sources):
                return sources[idx - 1]
            print(f"Invalid choice. Please enter a number between 1 and {len(sources)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            sys.exit(0)


def extract_from_p3a(p3a_file, table_name='t_item.tbl', out_file='t_item.tbl.tmp', quiet=False):
    """Extract a TBL file from a P3A archive."""
    if not HAS_LIBS:
        print(f"Error: Required library missing: {MISSING_LIB}")
        print("P3A extraction requires p3a_lib module.")
        return False
    
    try:
        if not os.path.exists(p3a_file):
            print(f"Error: P3A file not found: {p3a_file}")
            return False
        
        p3a = p3a_class()
        
        with open(p3a_file, 'rb') as p3a.f:
            headers, entries, p3a_dict = p3a.read_p3a_toc()
            
            for entry in entries:
                if os.path.basename(entry['name']) == table_name:
                    data = p3a.read_file(entry, p3a_dict)
                    with open(out_file, 'wb') as f:
                        f.write(data)
                    return True
            
            if not quiet:
                print(f"Error: {table_name} not found in {p3a_file}")
            return False
    
    except Exception as e:
        print(f"Error extracting from P3A: {e}")
        return False


def load_items_from_json(json_file='t_item.json'):
    """Load item data from JSON file."""
    try:
        if not os.path.exists(json_file):
            print(f"Error: JSON file not found: {json_file}")
            return None
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict):
            # Structure: {"data": [{"name": "ItemTableData", "data": [...]}]}
            if "data" in data and isinstance(data["data"], list):
                for section in data["data"]:
                    if section.get("name") == "ItemTableData":
                        items = section.get("data", [])
                        if not items:
                            print(f"Warning: No items found in ItemTableData section")
                        return items
                
                print(f"Warning: ItemTableData section not found in {json_file}")
                return []
            
            # Direct structure: {ItemTableData: [...]}
            elif "ItemTableData" in data:
                items = data["ItemTableData"]
                if not isinstance(items, list):
                    print(f"Error: ItemTableData is not a list in {json_file}")
                    return None
                return items
        
        print(f"Error: Unexpected JSON structure in {json_file}")
        return None
    
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_file}: {e}")
        return None
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return None


def load_items_from_tbl(tbl_file):
    """Load item data from TBL file."""
    if not HAS_LIBS:
        print(f"Error: Required library missing: {MISSING_LIB}")
        print("TBL reading requires kurodlc_lib module.")
        return None
    
    try:
        if not os.path.exists(tbl_file):
            print(f"Error: TBL file not found: {tbl_file}")
            return None
        
        kt = kuro_tables()
        table = kt.read_table(tbl_file)
        
        if not isinstance(table, dict):
            print(f"Error: Invalid TBL structure in {tbl_file}")
            return None
        
        if 'ItemTableData' not in table:
            print(f"Error: ItemTableData section not found in {tbl_file}")
            return None
        
        items = table['ItemTableData']
        if not isinstance(items, list):
            print(f"Error: ItemTableData is not a list in {tbl_file}")
            return None
        
        if not items:
            print(f"Warning: No items found in ItemTableData section")
        
        return items
    
    except Exception as e:
        print(f"Error loading {tbl_file}: {e}")
        return None


def load_items(force_source=None, no_interactive=False, keep_extracted=False):
    """
    Load item data from any supported source format.
    
    Returns:
        Tuple of (items_list, source_info) or (None, None) on error
    """
    # Detect available sources
    sources = detect_sources('t_item')
    
    if not sources:
        print(f"Error: No data sources found for t_item")
        print(f"\nLooked for:")
        print(f"  - t_item.json")
        print(f"  - t_item.tbl.original")
        print(f"  - t_item.tbl")
        print(f"  - script_en.p3a / script_eng.p3a")
        print(f"  - zzz_combined_tables.p3a")
        return None, None
    
    # Filter by forced source if specified
    if force_source:
        sources = [(t, p) for t, p in sources if t == force_source]
        if not sources:
            print(f"Error: No sources found matching type '{force_source}'")
            return None, None
    
    # Select source
    if len(sources) == 1 or no_interactive:
        stype, path = sources[0]
    else:
        stype, path = select_source_interactive(sources)
    
    # Load data based on source type
    temp_file = None
    extracted_temp = False
    
    try:
        if stype == 'json':
            items = load_items_from_json(path)
            source_info = {'type': 'json', 'path': path}
        
        elif stype in ('tbl', 'original'):
            items = load_items_from_tbl(path)
            source_info = {'type': stype, 'path': path}
        
        elif stype in ('p3a', 'zzz'):
            # Extract TBL from P3A
            temp_file = 't_item.tbl.tmp'
            if extract_from_p3a(path, 't_item.tbl', temp_file):
                extracted_temp = True
                items = load_items_from_tbl(temp_file)
                source_info = {'type': stype, 'path': path}
            else:
                print(f"Failed to extract t_item.tbl from {path}")
                return None, None
        
        else:
            print(f"Error: Unknown source type '{stype}'")
            return None, None
        
        # Cleanup temporary files
        if extracted_temp and temp_file and not keep_extracted:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return items, source_info
    
    except Exception as e:
        print(f"Error during data loading: {e}")
        
        # Cleanup on error
        if extracted_temp and temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
        
        return None, None


# -------------------------
# Costume data loading (t_costume) — optional enrichment
# -------------------------

def collect_costumes_recursive(node, result_list):
    """Recursively find dicts that look like costume entries.
    Handles both JSON named fields and TBL schema fields.
    
    Known field variants:
      item_id:   item_id (CostumeParam schema, JSON export)
      mdl_name:  mdl_name (CostumeParam schema), name (JSON export),
                 costume_model / base_model (CostumeTable schema)
    Model names can be chr*, equ*, rob*, etc."""
    if isinstance(node, dict):
        item_id = None
        mdl_name = None

        # item_id: try all known field names
        for key in ('item_id', 'int1', 'int2', 'shrt1', 'shrt2'):
            val = node.get(key)
            if isinstance(val, int) and val > 0:
                item_id = val
                break

        # mdl_name: try all known field names
        # CostumeParam schema: 'mdl_name'
        # CostumeTable schema: 'costume_model', 'base_model'
        # JSON export (t_costume.json): 'name'
        # Generic TBL: 'text1', 'text2', 'text3'
        for key in ('mdl_name', 'costume_model', 'base_model', 'name',
                     'text1', 'text2', 'text3'):
            val = node.get(key, '')
            if isinstance(val, str) and re.match(r'^[a-z][a-z0-9_]*\d', val):
                mdl_name = val
                break

        if item_id is not None and isinstance(item_id, int) and mdl_name:
            result_list.append({'item_id': item_id, 'name': mdl_name})

        for value in node.values():
            collect_costumes_recursive(value, result_list)
    elif isinstance(node, list):
        for item in node:
            collect_costumes_recursive(item, result_list)


def load_costumes_from_json(json_file):
    """Load costume data from JSON using recursive search."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        costumes = []
        collect_costumes_recursive(data, costumes)
        return costumes
    except Exception as e:
        print(f"Warning: Failed to read {json_file}: {e}")
        return None


def load_costumes_from_tbl(tbl_file):
    """Load costume data from TBL file using recursive search."""
    if not HAS_LIBS:
        return None
    try:
        kt = kuro_tables()
        table = kt.read_table(tbl_file)
        if not table:
            print(f"Warning: Failed to parse {tbl_file}")
            return None
        costumes = []
        collect_costumes_recursive(table, costumes)
        if not costumes and isinstance(table, dict):
            # Debug: show what sections were found
            sections = [k for k in table.keys() if 'costume' in k.lower() or 'Costume' in k]
            if sections:
                print(f"Warning: Found sections {sections} but no costume entries matched")
                # Show first entry for debugging
                for s in sections:
                    if table[s]:
                        print(f"  {s}[0] keys: {list(table[s][0].keys())}")
                        break
            else:
                print(f"Warning: No costume sections found in {tbl_file}")
                print(f"  Available sections: {list(table.keys())}")
        return costumes
    except Exception as e:
        print(f"Warning: Failed to read {tbl_file}: {e}")
        return None


def load_costume_data(force_source=None, no_interactive=False, keep_extracted=False, preferred_path=None):
    """
    Load t_costume data. Returns {item_id_str: mdl_name} dict or None.
    Silently returns None if unavailable (graceful fallback).
    
    Args:
        preferred_path: If set, prefer this P3A/source file (to match t_item source).
    """
    sources = []
    candidates = [
        ('json',     't_costume.json'),
        ('original', 't_costume.tbl.original'),
        ('tbl',      't_costume.tbl'),
        ('p3a',      'script_en.p3a'),
        ('p3a',      'script_eng.p3a'),
        ('zzz',      'zzz_combined_tables.p3a'),
    ]
    for stype, fname in candidates:
        if os.path.exists(fname):
            sources.append((stype, fname))

    if not sources:
        return None

    # Filter: json works without libs, tbl/p3a need HAS_LIBS
    usable = []
    for stype, path in sources:
        if stype == 'json':
            usable.append((stype, path))
        elif HAS_LIBS:
            usable.append((stype, path))
    if not usable:
        return None

    # Filter by forced source if specified
    if force_source:
        usable = [(t, p) for t, p in usable if t == force_source]
        if not usable:
            return None

    # Auto-select source, preferring same file as t_item
    if preferred_path:
        preferred = [(t, p) for t, p in usable if p == preferred_path]
        if preferred:
            stype, path = preferred[0]
        else:
            stype, path = usable[0]
    else:
        stype, path = usable[0]

    # Load data
    costumes_list = None
    temp_file = None
    extracted_temp = False

    if stype == 'json':
        costumes_list = load_costumes_from_json(path)
    elif stype in ('tbl', 'original'):
        costumes_list = load_costumes_from_tbl(path)
    elif stype in ('p3a', 'zzz'):
        temp_file = 't_costume.tbl.tmp'
        if extract_from_p3a(path, 't_costume.tbl', temp_file):
            extracted_temp = True
            costumes_list = load_costumes_from_tbl(temp_file)

    # Cleanup temp file AFTER reading
    if extracted_temp and temp_file and os.path.exists(temp_file) and not keep_extracted:
        os.remove(temp_file)

    if not costumes_list:
        return None

    # Build {item_id_str: mdl_name} mapping
    costume_map = {}
    for entry in costumes_list:
        item_id = entry.get('item_id')
        mdl_name = entry.get('name', '')
        if item_id is not None and mdl_name:
            costume_map[str(item_id)] = mdl_name

    if costume_map:
        print(f"  Costumes: {len(costume_map)} from {path}")

    return costume_map if costume_map else None

def print_usage():
    """Print usage information."""
    print(
        "Usage: python find_all_items.py [search_query] [options]\n"
        "\n"
        "This script searches through item data from multiple source formats.\n"
        "\n"
        "Supported sources (auto-detected in priority order):\n"
        "  1. t_item.json\n"
        "  2. t_item.tbl.original\n"
        "  3. t_item.tbl\n"
        "  4. script_en.p3a / script_eng.p3a (extracts t_item.tbl)\n"
        "  5. zzz_combined_tables.p3a (extracts t_item.tbl)\n"
        "\n"
        "Optional enrichment:\n"
        "  t_costume (json/tbl/tbl.original/P3A) - adds CostumeParam model names\n"
        "  When available, costume items show their mdl_name in [brackets].\n"
        "  Text search also matches against mdl_name (e.g. 'chr5001_c02').\n"
        "\n"
        "Arguments:\n"
        "  search_query   (Optional) Search query with optional prefix:\n"
        "\n"
        "Search modes:\n"
        "  id:NUMBER      - Search by exact ID (e.g., id:100)\n"
        "  name:TEXT      - Search in item names (e.g., name:100 or name:sword)\n"
        "  mdl:TEXT       - Search in costume model names (e.g., mdl:chr5001_c02)\n"
        "  TEXT           - Auto-detect:\n"
        "                     numbers → ID search\n"
        "                     chr...  → model search (requires t_costume)\n"
        "                     other   → name search (+ mdl_name if t_costume available)\n"
        "\n"
        "Options:\n"
        "  --source=TYPE       Force specific source: json, tbl, original, p3a, zzz\n"
        "  --no-interactive    Auto-select first source if multiple found\n"
        "  --keep-extracted    Keep temporary extracted files from P3A\n"
        "  --help              Show this help message\n"
        "\n"
        "Examples:\n"
        "  python find_all_items.py\n"
        "      Lists all items from auto-detected source.\n"
        "\n"
        "  python find_all_items.py sword\n"
        "      Lists all items with 'sword' in their name (auto-detect).\n"
        "\n"
        "  python find_all_items.py 100\n"
        "      Lists item with ID '100' (auto-detect: it's a number).\n"
        "\n"
        "  python find_all_items.py name:100\n"
        "      Lists all items with '100' in their name (explicit name search).\n"
        "\n"
        "  python find_all_items.py id:100\n"
        "      Lists the item with ID '100' (explicit ID search).\n"
        "\n"
        "  python find_all_items.py chr5001\n"
        "      Lists costumes with 'chr5001' in model name (auto-detect).\n"
        "\n"
        "  python find_all_items.py mdl:c02tow\n"
        "      Lists costumes with 'c02tow' in model name (explicit mdl search).\n"
        "\n"
        "  python find_all_items.py --source=json\n"
        "      Lists all items, forcing JSON source.\n"
        "\n"
        "IMPORTANT:\n"
        "  Use 'name:' prefix when searching for numbers in item names!\n"
        "  Otherwise, auto-detect will treat it as an ID search.\n"
        "  Queries starting with 'chr' auto-detect as model search.\n"
        "  Use 'name:chr...' to search for 'chr...' in item names instead."
    )


def main():
    """Main function."""
    # Parse command line arguments
    search_text = None
    search_id = None
    search_mdl = None
    force_source = None
    no_interactive = False
    keep_extracted = False
    
    args = sys.argv[1:]
    
    # Check for help
    if '--help' in args or '-h' in args:
        print_usage()
        return
    
    # Parse options
    remaining_args = []
    for arg in args:
        if arg.startswith('--source='):
            force_source = arg.split('=', 1)[1]
            if force_source not in ('json', 'tbl', 'original', 'p3a', 'zzz'):
                print(f"Error: Invalid source type '{force_source}'")
                print("Valid types: json, tbl, original, p3a, zzz")
                sys.exit(1)
        elif arg == '--no-interactive':
            no_interactive = True
        elif arg == '--keep-extracted':
            keep_extracted = True
        elif arg.startswith('--'):
            print(f"Error: Unknown option '{arg}'")
            print("Use --help for usage information.")
            sys.exit(1)
        else:
            remaining_args.append(arg)
    
    # Parse search query
    if remaining_args:
        param = remaining_args[0]
        
        # Check for prefix
        if param.startswith('id:'):
            # Explicit ID search
            search_id = param[3:]
            if not search_id:
                print("Error: 'id:' prefix requires a value (e.g., id:100)")
                sys.exit(1)
        
        elif param.startswith('name:'):
            # Explicit name search
            search_text = param[5:].lower()
            if not search_text:
                print("Error: 'name:' prefix requires a value (e.g., name:sword)")
                sys.exit(1)
        
        elif param.startswith('mdl:'):
            # Explicit model name search (t_costume)
            search_mdl = param[4:].lower()
            if not search_mdl:
                print("Error: 'mdl:' prefix requires a value (e.g., mdl:chr5001)")
                sys.exit(1)
        
        else:
            # Auto-detect mode
            if param.isdigit():
                search_id = param
                # Inform user about auto-detection
                print(f"# Auto-detected ID search for '{param}'", file=sys.stderr)
                print(f"# Use 'name:{param}' to search for '{param}' in item names instead", file=sys.stderr)
                print("", file=sys.stderr)
            elif re.match(r'^(chr|equ|rob|fc_)\d*', param, re.IGNORECASE):
                search_mdl = param.lower()
                print(f"# Auto-detected model search for '{param}'", file=sys.stderr)
                print(f"# Use 'name:{param}' to search in item names instead", file=sys.stderr)
                print("", file=sys.stderr)
            else:
                search_text = param.lower()
    
    # Load data
    print("Loading data...")
    items, source_info = load_items(force_source, no_interactive, keep_extracted)
    
    if items is None:
        print("\nFailed to load item data.")
        sys.exit(1)
    
    if not items:
        print("\nNo items found in source.")
        sys.exit(0)
    
    print(f"  Items:    {len(items)} from {source_info['path']}")
    
    # Load costume data (optional enrichment) — prefer same source as items
    costume_map = load_costume_data(force_source, no_interactive, keep_extracted,
                                    preferred_path=source_info.get('path'))
    print()
    
    # Build items dictionary
    items_dict = {}
    for item in items:
        if 'id' in item and 'name' in item:
            items_dict[str(item['id'])] = item['name']
    
    if not items_dict:
        print("No valid items found (missing 'id' or 'name' fields).")
        sys.exit(0)
    
    # Apply filters
    filtered = items_dict
    
    if search_text:
        filtered = {
            item_id: name for item_id, name in filtered.items()
            if search_text in str(name).lower()
            or (costume_map and item_id in costume_map
                and search_text in costume_map[item_id].lower())
        }
    
    if search_id:
        filtered = {
            item_id: name for item_id, name in filtered.items()
            if search_id == item_id
        }
    
    if search_mdl:
        if not costume_map:
            print("Warning: t_costume not available — model search requires t_costume data.")
            print("No matching items found.")
            return
        filtered = {
            item_id: name for item_id, name in filtered.items()
            if item_id in costume_map
            and search_mdl in costume_map[item_id].lower()
        }
    
    # Display results
    if not filtered:
        print("No matching items found.")
        return
    
    max_id_len = max(len(item_id) for item_id in filtered.keys())
    
    # Calculate name padding for costume alignment
    has_costumes = costume_map and any(item_id in costume_map for item_id in filtered)
    if has_costumes:
        max_name_len = max(len(str(name)) for name in filtered.values())
    
    for item_id, item_name in sorted(filtered.items(), key=lambda x: int(x[0])):
        if has_costumes and item_id in costume_map:
            print(f"{item_id.rjust(max_id_len)} : {item_name:<{max_name_len}}  [{costume_map[item_id]}]")
        else:
            print(f"{item_id.rjust(max_id_len)} : {item_name}")
    
    print(f"\nTotal: {len(filtered)} item(s)")


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
