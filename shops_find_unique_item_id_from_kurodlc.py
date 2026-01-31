#!/usr/bin/env python3
"""
shops_find_unique_item_id_from_kurodlc.py - v2.1 FINAL

Extract unique item IDs from .kurodlc.json files with template generation support.

Version History:
  v2.1 (2026-01-31) - FIXED non-interactive bug
    - Added --no-interactive flag for CI/CD and automated workflows
    - Added --default-shop-ids flag for automatic fallback
    - Better error handling when ShopItem section is missing
    - Clear error messages with actionable solutions
    - Prevents EOFError in non-interactive environments
    
  v2.0 (2026-01-31) - Template generation support
    - Added --generate-template mode
    - Auto-extract shop IDs from ShopItem section
    - Auto-extract template structure
    - Custom output filenames
    
  v1.0 (2025) - Initial release
    - Basic ID extraction from .kurodlc.json files
    - Multiple extraction modes (shop, costume, item, dlc)

GitHub: https://github.com/yourusername/kurodlc-toolkit
"""

import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any, Optional

def print_usage():
    """Print usage information."""
    print("""
Usage: python shops_find_unique_item_id_from_kurodlc.py <file.kurodlc.json> [mode] [options]

BASIC USAGE:
  python shops_find_unique_item_id_from_kurodlc.py my_dlc.kurodlc.json
      Extract all item IDs from all sections (default)

  python shops_find_unique_item_id_from_kurodlc.py my_dlc.kurodlc.json costume
      Extract IDs only from CostumeParam section

EXTRACTION MODES:
  all         - Extract from all sections (default)
  shop        - Extract from ShopItem section only
  costume     - Extract from CostumeParam section only
  item        - Extract from ItemTableData section only
  dlc         - Extract from DLCTableData.items section only
  
  Combinations (use +):
    costume+item   - Extract from CostumeParam and ItemTableData
    shop+costume   - Extract from ShopItem and CostumeParam

TEMPLATE GENERATION (v2.0):
  --generate-template [source]
      Generate template config for shops_create.py
      Optional source: costume, item, dlc, all (default: all)
      Creates: template_<filename>.json

TEMPLATE OPTIONS:
  --shop-ids=1,5,10       Manually specify shop IDs (comma-separated)
  --default-shop-ids      Use [1] as default when ShopItem not found
  --no-interactive        Do not prompt for input (required for CI/CD)
  --output=filename.json  Custom output filename

EXAMPLES:

  1. Extract all IDs:
     python shops_find_unique_item_id_from_kurodlc.py my_dlc.kurodlc.json

  2. Extract only costume IDs:
     python shops_find_unique_item_id_from_kurodlc.py my_dlc.kurodlc.json costume

  3. Generate template (auto-detect shop IDs):
     python shops_find_unique_item_id_from_kurodlc.py my_dlc.kurodlc.json --generate-template costume

  4. Generate template with manual shop IDs:
     python shops_find_unique_item_id_from_kurodlc.py my_dlc.kurodlc.json --generate-template costume --shop-ids=1,5,10,15

  5. For DLC without ShopItem (use default):
     python shops_find_unique_item_id_from_kurodlc.py my_dlc.kurodlc.json --generate-template costume --default-shop-ids

  6. For CI/CD (non-interactive):
     python shops_find_unique_item_id_from_kurodlc.py my_dlc.kurodlc.json --generate-template costume --default-shop-ids --no-interactive

  7. Custom output:
     python shops_find_unique_item_id_from_kurodlc.py my_dlc.kurodlc.json --generate-template --output=my_template.json

COMPLETE WORKFLOW - Creating Shop Assignments:

  Step 1: Generate template from your DLC
    $ python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json --generate-template costume
    Output: template_my_mod.kurodlc.json

  Step 2: (Optional) Edit template_my_mod.kurodlc.json
    - Review item_ids (auto-extracted)
    - Modify shop_ids if needed
    - Customize template structure

  Step 3: Generate shop assignments
    $ python shops_create.py template_my_mod.kurodlc.json
    Output: output_template_my_mod.kurodlc.json

  Step 4: Copy ShopItem section into your DLC
    Copy ShopItem section from output_template_my_mod.kurodlc.json
    Paste into your my_mod.kurodlc.json file

  Result: All costume items now available in all specified shops!

OUTPUT:
  - Standard mode: Prints Python list of IDs to stdout
  - Template mode: Creates template_<filename>.json file
  - Extraction summary: Printed to stderr (use 2>/dev/null to suppress)

NOTES:
  - When ShopItem exists: shop IDs are auto-detected from it
  - When ShopItem missing: use --shop-ids or --default-shop-ids
  - For automation/CI/CD: always use --no-interactive flag
  - Template files can be manually edited before using with shops_create.py
""")

def is_valid_kurodlc_structure(data: Dict) -> bool:
    """
    Validate .kurodlc.json structure.
    
    Required sections: CostumeParam, DLCTableData
    Optional sections: ItemTableData, ShopItem
    """
    if not isinstance(data, dict):
        return False
    
    # Required sections
    required = ["CostumeParam", "DLCTableData"]
    if not all(k in data for k in required):
        return False
    
    if not all(isinstance(data[k], list) for k in required):
        return False
    
    # Validate CostumeParam has item_id
    if not any(isinstance(x, dict) and "item_id" in x for x in data["CostumeParam"]):
        return False
    
    # Validate DLCTableData has items
    if not any(isinstance(x, dict) and "items" in x for x in data["DLCTableData"]):
        return False
    
    return True

def extract_ids_by_mode(data: Dict, modes: Set[str]) -> Tuple[List[int], List[str]]:
    """
    Extract IDs based on selected modes.
    
    Returns:
        (list of IDs, extraction summary lines)
    """
    all_ids = []
    summary = []
    
    if "shop" in modes and "ShopItem" in data:
        shop_ids = [item["item_id"] for item in data["ShopItem"] 
                   if isinstance(item, dict) and "item_id" in item]
        all_ids.extend(shop_ids)
        summary.append(f"ShopItem: {len(shop_ids)} IDs")
    
    if "costume" in modes and "CostumeParam" in data:
        costume_ids = [item["item_id"] for item in data["CostumeParam"]
                      if isinstance(item, dict) and "item_id" in item]
        all_ids.extend(costume_ids)
        summary.append(f"CostumeParam: {len(costume_ids)} IDs")
    
    if "item" in modes and "ItemTableData" in data:
        item_ids = [item["id"] for item in data["ItemTableData"]
                   if isinstance(item, dict) and "id" in item]
        all_ids.extend(item_ids)
        summary.append(f"ItemTableData: {len(item_ids)} IDs")
    
    if "dlc" in modes and "DLCTableData" in data:
        dlc_ids = []
        for dlc in data["DLCTableData"]:
            if isinstance(dlc, dict) and "items" in dlc and isinstance(dlc["items"], list):
                dlc_ids.extend(dlc["items"])
        all_ids.extend(dlc_ids)
        summary.append(f"DLCTableData.items: {len(dlc_ids)} IDs")
    
    return all_ids, summary

def extract_shop_ids(data: Dict) -> Optional[List[int]]:
    """
    Extract unique shop IDs from ShopItem section.
    
    Returns:
        List of unique shop IDs, or None if ShopItem section doesn't exist
    """
    if "ShopItem" not in data or not isinstance(data["ShopItem"], list):
        return None
    
    if len(data["ShopItem"]) == 0:
        return None
    
    shop_ids = set()
    for item in data["ShopItem"]:
        if isinstance(item, dict) and "shop_id" in item:
            shop_ids.add(item["shop_id"])
    
    return sorted(list(shop_ids)) if shop_ids else None

def extract_template_from_shop_item(data: Dict) -> Optional[Dict]:
    """
    Extract template structure from first ShopItem entry.
    
    Returns:
        Template dict with ${shop_id} and ${item_id} placeholders,
        or None if ShopItem section doesn't exist
    """
    if "ShopItem" not in data or not isinstance(data["ShopItem"], list):
        return None
    
    if len(data["ShopItem"]) == 0:
        return None
    
    first_item = data["ShopItem"][0]
    if not isinstance(first_item, dict):
        return None
    
    # Create template from first item
    template = {}
    for key, value in first_item.items():
        if key == "shop_id":
            template[key] = "${shop_id}"
        elif key == "item_id":
            template[key] = "${item_id}"
        else:
            template[key] = value
    
    return template

def generate_template_config(json_file: str, data: Dict, modes: Set[str],
                            manual_shop_ids: Optional[List[int]] = None,
                            output_file: Optional[str] = None,
                            default_shop_ids: bool = False,
                            no_interactive: bool = False) -> None:
    """
    Generate template config file for shops_create.py.
    
    Args:
        json_file: Source .kurodlc.json filename
        data: Parsed JSON data
        modes: Set of extraction modes
        manual_shop_ids: Manually specified shop IDs (overrides auto-detection)
        output_file: Custom output filename
        default_shop_ids: Use [1] as default if no ShopItem found
        no_interactive: Do not prompt for user input
    """
    # Extract item IDs
    all_ids, extraction_summary = extract_ids_by_mode(data, modes)
    unique_item_ids = sorted(set(all_ids))
    
    if not unique_item_ids:
        print("Error: No item IDs found in selected sections.")
        sys.exit(1)
    
    # Determine shop IDs
    final_shop_ids = None
    shop_id_source = None
    
    if manual_shop_ids:
        # Manual shop IDs override everything
        final_shop_ids = manual_shop_ids
        shop_id_source = "manually specified"
    else:
        # Try to extract from ShopItem
        extracted_shop_ids = extract_shop_ids(data)
        if extracted_shop_ids:
            final_shop_ids = extracted_shop_ids
            shop_id_source = "extracted from ShopItem section"
        elif default_shop_ids:
            # Use default [1]
            final_shop_ids = [1]
            shop_id_source = "default [1]"
        else:
            # No shop IDs found and no default requested
            if no_interactive:
                # In non-interactive mode, we cannot prompt
                print("\n" + "="*60)
                print("ERROR: Cannot generate template")
                print("="*60)
                print("\nReason: No shop IDs found in ShopItem section.")
                print("\nThe DLC file does not contain a ShopItem section,")
                print("and no shop IDs were specified.")
                print("\nPossible solutions:")
                print("\n  1. Specify shop IDs manually:")
                print(f"     python {os.path.basename(__file__)} {json_file} \\")
                print("       --generate-template --shop-ids=1,5,10")
                print("\n  2. Use default shop IDs [1]:")
                print(f"     python {os.path.basename(__file__)} {json_file} \\")
                print("       --generate-template --default-shop-ids --no-interactive")
                print("\n  3. Run interactively (without --no-interactive):")
                print(f"     python {os.path.basename(__file__)} {json_file} \\")
                print("       --generate-template")
                print("\n" + "="*60)
                sys.exit(1)
            else:
                # Prompt user for shop IDs
                print("\nNo shop IDs found in ShopItem section.")
                print("Please enter shop IDs (comma-separated, e.g., 1,5,10):")
                print("Or press Enter to use default [1]:")
                try:
                    shop_input = input("> ").strip()
                    if shop_input:
                        final_shop_ids = [int(x.strip()) for x in shop_input.split(",")]
                        shop_id_source = "user input"
                    else:
                        final_shop_ids = [1]
                        shop_id_source = "default [1] (user pressed Enter)"
                except ValueError:
                    print("Error: Invalid shop IDs. Using default [1].")
                    final_shop_ids = [1]
                    shop_id_source = "default [1] (after error)"
                except (EOFError, KeyboardInterrupt):
                    print("\n\nOperation cancelled by user.")
                    sys.exit(1)
    
    # Extract template structure
    template = extract_template_from_shop_item(data)
    
    # Create config
    config = {
        "_comment": [
            "Template config file generated by shops_find_unique_item_id_from_kurodlc.py v2.1",
            f"Source file: {json_file}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "INSTRUCTIONS:",
            "1. Review and modify shop_ids as needed",
            "2. Review item_ids (automatically extracted)",
            "3. Customize template structure if needed",
            "4. Run: python shops_create.py <this_file>",
            "",
            f"Extraction summary:",
        ],
        "item_ids": unique_item_ids,
        "shop_ids": final_shop_ids
    }
    
    # Add extraction details to comment
    for line in extraction_summary:
        config["_comment"].append(f"  - {line}")
    
    # Add shop ID source info
    config["_comment"].append(f"  - Shop IDs: {shop_id_source}")
    
    # Add template if extracted
    if template:
        config["template"] = template
        config["_comment"].append(f"  - Template: extracted from ShopItem")
    else:
        config["_comment"].append(f"  - Template: not included (no ShopItem section)")
        config["_comment"].append(f"    Note: shops_create.py will use default template")
    
    # Determine output filename
    if output_file:
        output_filename = output_file
    else:
        base_name = os.path.basename(json_file)
        output_filename = f"template_{base_name}"
    
    # Write config file
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        print("\n" + "="*60)
        print("Template config file generated successfully!")
        print("="*60)
        print(f"Output file: {output_filename}")
        print(f"\nSummary:")
        print(f"  - Item IDs extracted: {len(unique_item_ids)}")
        print(f"  - Shop IDs: {final_shop_ids} ({shop_id_source})")
        if template:
            print(f"  - Template: Extracted from ShopItem")
        else:
            print(f"  - Template: Not included (will use default)")
        print(f"\nExtraction details:")
        for line in extraction_summary:
            print(f"  - {line}")
        print("="*60)
        
    except Exception as e:
        print(f"Error writing template file: {e}")
        sys.exit(1)

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    # Parse arguments
    json_file = sys.argv[1]
    mode_arg = "all"
    generate_template = False
    template_source = "all"
    output_file = None
    manual_shop_ids = None
    default_shop_ids = False
    no_interactive = False
    
    # Parse options
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == "--generate-template":
            generate_template = True
            # Check if next arg is a source mode
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                template_source = sys.argv[i + 1]
                i += 1
        
        elif arg.startswith("--output="):
            output_file = arg.split("=", 1)[1]
        
        elif arg.startswith("--shop-ids="):
            shop_ids_str = arg.split("=", 1)[1]
            try:
                manual_shop_ids = [int(x.strip()) for x in shop_ids_str.split(",")]
            except ValueError:
                print(f"Error: Invalid shop IDs in {arg}")
                sys.exit(1)
        
        elif arg == "--default-shop-ids":
            default_shop_ids = True
        
        elif arg == "--no-interactive":
            no_interactive = True
        
        elif not arg.startswith("--"):
            # Regular mode argument
            mode_arg = arg
        
        i += 1
    
    # Check if file exists
    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found.")
        sys.exit(1)
    
    # Load JSON
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{json_file}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file '{json_file}': {e}")
        sys.exit(1)
    
    # Validate structure
    if not is_valid_kurodlc_structure(data):
        print(f"Error: JSON file '{json_file}' does not have a valid kurodlc structure.")
        sys.exit(1)
    
    # Parse mode
    if generate_template:
        mode_to_use = template_source
    else:
        mode_to_use = mode_arg
    
    valid_modes = {"shop", "costume", "item", "dlc", "all"}
    requested_modes = set()
    
    if mode_to_use == "all":
        requested_modes = {"shop", "costume", "item", "dlc"}
    else:
        parts = mode_to_use.split("+")
        for part in parts:
            part = part.strip()
            if part not in valid_modes or part == "all":
                print(f"Error: Unknown mode '{part}'.")
                print("Valid modes are: shop, costume, item, dlc, all")
                print("You can combine modes with +, e.g., 'shop+costume'")
                sys.exit(1)
            requested_modes.add(part)
    
    # Execute based on mode
    if generate_template:
        generate_template_config(json_file, data, requested_modes, 
                                manual_shop_ids, output_file, default_shop_ids,
                                no_interactive)
    else:
        # Original behavior - extract and print IDs
        all_ids, extraction_summary = extract_ids_by_mode(data, requested_modes)
        unique_item_ids = sorted(set(all_ids))
        
        # Print extraction summary to stderr
        if extraction_summary:
            print("# Extraction summary:", file=sys.stderr)
            for line in extraction_summary:
                print(f"#   {line}", file=sys.stderr)
            print(f"# Total unique IDs: {len(unique_item_ids)}", file=sys.stderr)
            print("", file=sys.stderr)
        
        # Print result to stdout
        print(unique_item_ids)

if __name__ == "__main__":
    main()
