#!/usr/bin/env python3
"""
find_all_names.py - Standalone version with integrated multi-source support

Search and display character names from multiple source formats.

Supported sources:
- t_name.json
- t_name.tbl
- t_name.tbl.original
- script_en.p3a / script_eng.p3a (extracts t_name.tbl)
- zzz_combined_tables.p3a (extracts t_name.tbl)
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
# Data loading functions (integrated from data_loader.py)
# -------------------------

def detect_sources(base_name='t_name'):
    """Detect available data sources for character names."""
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
            print(f"  {i}) {path} (extract t_name.tbl)")
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


def extract_from_p3a(p3a_file, table_name='t_name.tbl', out_file='t_name.tbl.tmp'):
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


def load_names_from_json(json_file='t_name.json'):
    """Load character name data from JSON file."""
    try:
        if not os.path.exists(json_file):
            print(f"Error: JSON file not found: {json_file}")
            return None
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict):
            # Structure: {"data": [{"name": "NameTableData", "data": [...]}]}
            if "data" in data and isinstance(data["data"], list):
                for section in data["data"]:
                    if section.get("name") == "NameTableData":
                        names = section.get("data", [])
                        if not names:
                            print(f"Warning: No character names found in NameTableData section")
                        return names
                
                print(f"Warning: NameTableData section not found in {json_file}")
                return []
            
            # Direct structure: {NameTableData: [...]}
            elif "NameTableData" in data:
                names = data["NameTableData"]
                if not isinstance(names, list):
                    print(f"Error: NameTableData is not a list in {json_file}")
                    return None
                return names
        
        print(f"Error: Unexpected JSON structure in {json_file}")
        return None
    
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_file}: {e}")
        return None
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return None


def load_names_from_tbl(tbl_file):
    """Load character name data from TBL file."""
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
        
        if 'NameTableData' not in table:
            print(f"Error: NameTableData section not found in {tbl_file}")
            return None
        
        names = table['NameTableData']
        if not isinstance(names, list):
            print(f"Error: NameTableData is not a list in {tbl_file}")
            return None
        
        if not names:
            print(f"Warning: No character names found in NameTableData section")
        
        return names
    
    except Exception as e:
        print(f"Error loading {tbl_file}: {e}")
        return None


def load_names(force_source=None, no_interactive=False, keep_extracted=False):
    """
    Load character name data from any supported source format.
    
    Returns:
        Tuple of (names_list, source_info) or (None, None) on error
    """
    # Detect available sources
    sources = detect_sources('t_name')
    
    if not sources:
        print(f"Error: No data sources found for t_name")
        print(f"\nLooked for:")
        print(f"  - t_name.json")
        print(f"  - t_name.tbl.original")
        print(f"  - t_name.tbl")
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
            names = load_names_from_json(path)
            source_info = {'type': 'json', 'path': path}
        
        elif stype in ('tbl', 'original'):
            names = load_names_from_tbl(path)
            source_info = {'type': stype, 'path': path}
        
        elif stype in ('p3a', 'zzz'):
            # Extract TBL from P3A
            temp_file = 't_name.tbl.tmp'
            if extract_from_p3a(path, 't_name.tbl', temp_file):
                extracted_temp = True
                names = load_names_from_tbl(temp_file)
                source_info = {'type': stype, 'path': f"{path} -> {temp_file}"}
            else:
                print(f"Failed to extract t_name.tbl from {path}")
                return None, None
        
        else:
            print(f"Error: Unknown source type '{stype}'")
            return None, None
        
        # Cleanup temporary files
        if extracted_temp and temp_file and not keep_extracted:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Cleaned up temporary file: {temp_file}")
        
        return names, source_info
    
    except Exception as e:
        print(f"Error during data loading: {e}")
        
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
        "Usage: python find_all_names.py [search_query] [options]\n"
        "\n"
        "This script searches through character name data from multiple source formats.\n"
        "\n"
        "Supported sources (auto-detected in priority order):\n"
        "  1. t_name.json\n"
        "  2. t_name.tbl.original\n"
        "  3. t_name.tbl\n"
        "  4. script_en.p3a / script_eng.p3a (extracts t_name.tbl)\n"
        "  5. zzz_combined_tables.p3a (extracts t_name.tbl)\n"
        "\n"
        "Arguments:\n"
        "  search_query   (Optional) Search query with optional prefix:\n"
        "\n"
        "Search modes:\n"
        "  id:NUMBER         - Search by exact character ID (e.g., id:100)\n"
        "  name:TEXT         - Search in character names (e.g., name:van or name:100)\n"
        "  full_name:TEXT    - Search in full names (e.g., full_name:arkride)\n"
        "  model:TEXT        - Search in model names (e.g., model:chr0000)\n"
        "  TEXT              - Auto-detect (numbers → ID search, text → name search)\n"
        "\n"
        "Options:\n"
        "  --source=TYPE       Force specific source: json, tbl, original, p3a, zzz\n"
        "  --no-interactive    Auto-select first source if multiple found\n"
        "  --keep-extracted    Keep temporary extracted files from P3A\n"
        "  --show-full         Show full names in output\n"
        "  --show-model        Show model names in output\n"
        "  --help              Show this help message\n"
        "\n"
        "Examples:\n"
        "  python find_all_names.py\n"
        "      Lists all characters from auto-detected source.\n"
        "\n"
        "  python find_all_names.py van\n"
        "      Lists all characters with 'van' in their name (auto-detect).\n"
        "\n"
        "  python find_all_names.py 100\n"
        "      Lists character with ID '100' (auto-detect: it's a number).\n"
        "\n"
        "  python find_all_names.py name:100\n"
        "      Lists all characters with '100' in their name (explicit name search).\n"
        "\n"
        "  python find_all_names.py id:0\n"
        "      Lists the character with ID '0' (explicit ID search).\n"
        "\n"
        "  python find_all_names.py full_name:arkride\n"
        "      Lists characters with 'arkride' in their full name.\n"
        "\n"
        "  python find_all_names.py model:chr0000\n"
        "      Lists characters using model 'chr0000'.\n"
        "\n"
        "  python find_all_names.py --source=json --show-full\n"
        "      Lists all characters from JSON source with full names.\n"
        "\n"
        "IMPORTANT:\n"
        "  Use 'name:' prefix when searching for numbers in character names!\n"
        "  Otherwise, auto-detect will treat it as an ID search."
    )


def main():
    """Main function."""
    # Parse command line arguments
    search_text = None
    search_id = None
    search_full_name = None
    search_model = None
    force_source = None
    no_interactive = False
    keep_extracted = False
    show_full = False
    show_model = False
    
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
        elif arg == '--show-full':
            show_full = True
        elif arg == '--show-model':
            show_model = True
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
                print("Error: 'name:' prefix requires a value (e.g., name:van)")
                sys.exit(1)
        
        elif param.startswith('full_name:'):
            # Explicit full name search
            search_full_name = param[10:].lower()
            if not search_full_name:
                print("Error: 'full_name:' prefix requires a value (e.g., full_name:arkride)")
                sys.exit(1)
        
        elif param.startswith('model:'):
            # Explicit model search
            search_model = param[6:].lower()
            if not search_model:
                print("Error: 'model:' prefix requires a value (e.g., model:chr0000)")
                sys.exit(1)
        
        else:
            # Auto-detect mode
            if param.isdigit():
                search_id = param
                # Inform user about auto-detection
                print(f"# Auto-detected ID search for '{param}'", file=sys.stderr)
                print(f"# Use 'name:{param}' to search for '{param}' in character names instead", file=sys.stderr)
                print("", file=sys.stderr)
            else:
                search_text = param.lower()
    
    # Load data
    print("Loading character name data...\n")
    names, source_info = load_names(force_source, no_interactive, keep_extracted)
    
    if names is None:
        print("\nFailed to load character name data.")
        sys.exit(1)
    
    if not names:
        print("\nNo character names found in source.")
        sys.exit(0)
    
    print(f"\nLoaded {len(names)} characters from: {source_info['path']}\n")
    
    # Build character dictionary
    characters = []
    for char in names:
        if 'character_id' in char and 'name' in char:
            char_info = {
                'id': str(char['character_id']),
                'name': char['name'],
                'full_name': char.get('full_name', ''),
                'model': char.get('model', '')
            }
            characters.append(char_info)
    
    if not characters:
        print("No valid characters found (missing 'character_id' or 'name' fields).")
        sys.exit(0)
    
    # Apply filters
    filtered = characters
    
    if search_text:
        filtered = [
            char for char in filtered
            if search_text in char['name'].lower()
        ]
    
    if search_id:
        filtered = [
            char for char in filtered
            if search_id == char['id']
        ]
    
    if search_full_name:
        filtered = [
            char for char in filtered
            if search_full_name in char['full_name'].lower()
        ]
    
    if search_model:
        filtered = [
            char for char in filtered
            if search_model in char['model'].lower()
        ]
    
    # Display results
    if not filtered:
        print("No matching characters found.")
        return
    
    # Calculate column widths
    max_id_len = max(len(char['id']) for char in filtered)
    max_name_len = max(len(char['name']) for char in filtered)
    
    if show_full:
        max_full_len = max(len(char['full_name']) for char in filtered)
    if show_model:
        max_model_len = max(len(char['model']) for char in filtered)
    
    # Print results
    for char in sorted(filtered, key=lambda x: int(x['id']) if x['id'].isdigit() else 999999):
        output = f"{char['id'].rjust(max_id_len)} : {char['name'].ljust(max_name_len)}"
        
        if show_full and char['full_name']:
            output += f" | {char['full_name'].ljust(max_full_len)}"
        
        if show_model and char['model']:
            output += f" | {char['model']}"
        
        print(output)
    
    print(f"\nTotal: {len(filtered)} character(s)")


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
