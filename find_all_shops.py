#!/usr/bin/env python3
"""
find_all_shops.py - Standalone version with integrated multi-source support

Search and display shops from multiple source formats.

Supported sources:
- t_shop.json
- t_shop.tbl
- t_shop.tbl.original
- script_en.p3a / script_eng.p3a (extracts t_shop.tbl)
- zzz_combined_tables.p3a (extracts t_shop.tbl)
"""

import sys
import os
import json

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
# Data loading functions
# -------------------------

def detect_sources(base_name='t_shop'):
    """Detect available data sources for shops."""
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
            print(f"  {i}) {path} (extract t_shop.tbl)")
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


def extract_from_p3a(p3a_file, table_name='t_shop.tbl', out_file='t_shop.tbl.tmp'):
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
        print(f"Extracting {table_name} from {p3a_file}...")
        
        with open(p3a_file, 'rb') as p3a.f:
            headers, entries, p3a_dict = p3a.read_p3a_toc()
            
            for entry in entries:
                if os.path.basename(entry['name']) == table_name:
                    data = p3a.read_file(entry, p3a_dict)
                    with open(out_file, 'wb') as f:
                        f.write(data)
                    print(f"Successfully extracted to {out_file}")
                    return True
            
            print(f"Error: {table_name} not found in {p3a_file}")
            return False
    
    except Exception as e:
        print(f"Error extracting from P3A: {e}")
        return False


def collect_shops_recursive(node, result_list):
    """
    Recursively search for shop entries in data structure.
    Looks for objects with 'id' and 'shop_name' fields.
    """
    if isinstance(node, dict):
        # Check if this dict has shop fields
        if 'id' in node and 'shop_name' in node:
            result_list.append(node)
        
        # Recurse into all values
        for value in node.values():
            collect_shops_recursive(value, result_list)
    
    elif isinstance(node, list):
        # Recurse into list items
        for item in node:
            collect_shops_recursive(item, result_list)


def load_shops_from_json(json_file='t_shop.json'):
    """Load shop data from JSON file using recursive search."""
    try:
        if not os.path.exists(json_file):
            print(f"Error: JSON file not found: {json_file}")
            return None
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Recursively search for shop entries
        shops = []
        collect_shops_recursive(data, shops)
        
        return shops
    
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_file}: {e}")
        return None
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return None


def load_shops_from_tbl(tbl_file):
    """Load shop data from TBL file using recursive search."""
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
        
        # Recursively search for shop entries
        shops = []
        collect_shops_recursive(table, shops)
        
        return shops
    
    except Exception as e:
        print(f"Error loading {tbl_file}: {e}")
        return None


def load_shops(force_source=None, no_interactive=False, keep_extracted=False):
    """
    Load shop data from any supported source format using flexible recursive search.
    
    Returns:
        Tuple of (shops_list, source_info) or (None, None) on error
    """
    # Detect sources
    sources = detect_sources('t_shop')
    
    if not sources:
        print(f"Error: No data sources found for t_shop")
        print(f"\nLooked for:")
        print(f"  - t_shop.json")
        print(f"  - t_shop.tbl.original")
        print(f"  - t_shop.tbl")
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
        print(f"Using source: {path}")
    else:
        stype, path = select_source_interactive(sources)
    
    # Load data based on source type
    temp_file = None
    extracted_temp = False
    
    try:
        if stype == 'json':
            shops = load_shops_from_json(path)
            source_info = {'type': 'json', 'path': path}
        
        elif stype in ('tbl', 'original'):
            shops = load_shops_from_tbl(path)
            source_info = {'type': stype, 'path': path}
        
        elif stype in ('p3a', 'zzz'):
            temp_file = 't_shop.tbl.tmp'
            if extract_from_p3a(path, 't_shop.tbl', temp_file):
                extracted_temp = True
                shops = load_shops_from_tbl(temp_file)
                source_info = {'type': stype, 'path': f"{path} -> {temp_file}"}
            else:
                print(f"Failed to extract t_shop.tbl from {path}")
                return None, None
        
        else:
            print(f"Error: Unknown source type '{stype}'")
            return None, None
        
        # Cleanup temporary files
        if extracted_temp and temp_file and not keep_extracted:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Cleaned up temporary file: {temp_file}")
        
        return shops, source_info
    
    except Exception as e:
        print(f"Error during shop data loading: {e}")
        
        # Cleanup on error
        if extracted_temp and temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
        
        return None, None


# -------------------------
# Main script functionality
# -------------------------

def print_usage():
    """Print usage information."""
    print(
        "Usage: python find_all_shops.py [search_query] [options]\n"
        "\n"
        "This script searches through shop data from multiple source formats.\n"
        "\n"
        "Supported sources (auto-detected in priority order):\n"
        "  1. t_shop.json\n"
        "  2. t_shop.tbl.original\n"
        "  3. t_shop.tbl\n"
        "  4. script_en.p3a / script_eng.p3a (extracts t_shop.tbl)\n"
        "  5. zzz_combined_tables.p3a (extracts t_shop.tbl)\n"
        "\n"
        "Arguments:\n"
        "  search_query   (Optional) Search query with optional prefix:\n"
        "\n"
        "Search modes:\n"
        "  id:NUMBER         - Search by exact shop ID (e.g., id:21)\n"
        "  name:TEXT          - Search in shop names (e.g., name:armor or name:21)\n"
        "  TEXT               - Auto-detect (numbers → ID search, text → name search)\n"
        "\n"
        "Options:\n"
        "  --source=TYPE       Force specific source: json, tbl, original, p3a, zzz\n"
        "  --no-interactive    Auto-select first source if multiple found\n"
        "  --keep-extracted    Keep temporary extracted files from P3A\n"
        "  --debug             Show debug information about data structure\n"
        "  --help              Show this help message\n"
        "\n"
        "Examples:\n"
        "  python find_all_shops.py\n"
        "      Lists all shops from auto-detected source.\n"
        "\n"
        "  python find_all_shops.py blacksmith\n"
        "      Lists all shops with 'blacksmith' in their name (auto-detect).\n"
        "\n"
        "  python find_all_shops.py 21\n"
        "      Lists shop with ID '21' (auto-detect: it's a number).\n"
        "\n"
        "  python find_all_shops.py name:21\n"
        "      Lists all shops with '21' in their name (explicit name search).\n"
        "\n"
        "  python find_all_shops.py id:0\n"
        "      Lists the shop with ID '0' (explicit ID search).\n"
        "\n"
        "  python find_all_shops.py --source=json\n"
        "      Lists all shops, forcing JSON source.\n"
        "\n"
        "  python find_all_shops.py armor --source=tbl --no-interactive\n"
        "      Search for 'armor' using TBL source without interactive prompts.\n"
        "\n"
        "IMPORTANT:\n"
        "  Use 'name:' prefix when searching for numbers in shop names!\n"
        "  Otherwise, auto-detect will treat it as an ID search."
    )


def main():
    """Main function."""
    # Parse command line arguments
    search_text = None
    search_id = None
    force_source = None
    no_interactive = False
    keep_extracted = False
    debug = False
    
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
        elif arg == '--debug':
            debug = True
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
                print("Error: 'id:' prefix requires a value (e.g., id:21)")
                sys.exit(1)
        
        elif param.startswith('name:'):
            # Explicit name search
            search_text = param[5:].lower()
            if not search_text:
                print("Error: 'name:' prefix requires a value (e.g., name:armor)")
                sys.exit(1)
        
        else:
            # Auto-detect mode
            if param.isdigit():
                search_id = param
                # Inform user about auto-detection
                print(f"# Auto-detected ID search for '{param}'", file=sys.stderr)
                print(f"# Use 'name:{param}' to search for '{param}' in shop names instead", file=sys.stderr)
                print("", file=sys.stderr)
            else:
                search_text = param.lower()
    
    # Load data
    print("Loading shop data...\n")
    shops, source_info = load_shops(force_source, no_interactive, keep_extracted)
    
    if shops is None:
        print("\nFailed to load shop data.")
        sys.exit(1)
    
    if not shops:
        print("\nNo shops found in source.")
        sys.exit(0)
    
    if debug:
        print(f"\nDEBUG: Loaded {len(shops)} shop entries")
        print(f"DEBUG: Type of shops: {type(shops)}")
        if shops:
            print(f"DEBUG: First shop entry: {shops[0]}")
            print(f"DEBUG: Type of first entry: {type(shops[0])}")
    
    print(f"\nLoaded {len(shops)} shops from: {source_info['path']}\n")
    
    # Build shops list from loaded data
    shop_list = []
    for shop in shops:
        if isinstance(shop, dict):
            if 'id' in shop and 'shop_name' in shop:
                shop_list.append({
                    'id': str(shop['id']),
                    'name': shop['shop_name']
                })
            elif debug:
                print(f"DEBUG: Shop missing fields - available keys: {shop.keys()}")
        else:
            print(f"Warning: Unexpected shop data format: {type(shop)}")
    
    if not shop_list:
        print("No valid shops found (missing 'id' or 'shop_name' fields).")
        if debug and shops:
            print(f"\nDEBUG: Sample shop data structure:")
            print(f"{shops[0]}")
        sys.exit(0)
    
    # Apply filters
    filtered = shop_list
    
    if search_text:
        filtered = [
            shop for shop in filtered
            if search_text in shop['name'].lower()
        ]
    
    if search_id:
        filtered = [
            shop for shop in filtered
            if search_id == shop['id']
        ]
    
    # Display results
    if not filtered:
        print("No matching shops found.")
        return
    
    max_id_len = max(len(shop['id']) for shop in filtered)
    
    for shop in sorted(filtered, key=lambda x: int(x['id']) if x['id'].isdigit() else 999999):
        print(f"{shop['id'].rjust(max_id_len)} : {shop['name']}")
    
    print(f"\nTotal: {len(filtered)} shop(s)")


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
