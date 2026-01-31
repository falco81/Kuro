#!/usr/bin/env python3
"""
shops_find_unique_item_id_from_kurodlc.py - v2.0

Extracts unique item IDs from .kurodlc.json files and can generate
template config files for shops_create.py v2.0.

NEW in v2.0:
- Generate template config for shops_create.py
- Extract shop IDs from ShopItem section
- Support for --generate-template mode
- Template data source selection (same as extraction modes)
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

def print_usage():
    """Print usage information."""
    print("""
Usage: python shops_find_unique_item_id_from_kurodlc.py <file.kurodlc.json> [mode] [options]

This script extracts unique item IDs from a .kurodlc JSON file.

=============================================================================
BASIC MODES (Extract and Print IDs)
=============================================================================

Available modes:
  all         -> Extract from ALL sections (default)
                 Includes: ShopItem, CostumeParam, ItemTableData, DLCTableData.items

  shop        -> Extract only from 'ShopItem' section
  costume     -> Extract only from 'CostumeParam' section
  item        -> Extract only from 'ItemTableData' section (uses 'id' field)
  dlc         -> Extract only from 'DLCTableData.items' section

Combination modes (use + to combine):
  shop+costume     -> ShopItem + CostumeParam
  costume+item     -> CostumeParam + ItemTableData
  item+dlc         -> ItemTableData + DLCTableData.items
  shop+item+dlc    -> ShopItem + ItemTableData + DLCTableData.items
  ... any combination you want!

=============================================================================
TEMPLATE GENERATION MODE (NEW in v2.0)
=============================================================================

Generate a template config file for shops_create.py v2.0:

  --generate-template [source]

This will:
  1. Extract item IDs from specified source (default: all)
  2. Extract shop IDs from ShopItem section (if exists)
  3. Extract template structure from first ShopItem entry
  4. Create a shops_create.py compatible config file

Source options (same as modes above):
  all, shop, costume, item, dlc, or combinations like costume+item

Output file: template_<original_filename>.json

=============================================================================
OPTIONS
=============================================================================

  --generate-template [source]
      Generate template config file
      Source: all, shop, costume, item, dlc, or combinations
      Default source: all

  --output=<filename>
      Custom output filename for generated template
      Default: template_<original_filename>.json

  --shop-ids=<ids>
      Manually specify shop IDs (comma-separated)
      Example: --shop-ids=1,5,10,15
      If not specified, extracts from ShopItem section

  --default-shop-ids
      Use default shop IDs [1] if ShopItem section not found
      Otherwise script will prompt for shop IDs

=============================================================================
EXAMPLES
=============================================================================

Example 1: Basic extraction (print IDs)
  python shops_find_unique_item_id_from_kurodlc.py file.json
  python shops_find_unique_item_id_from_kurodlc.py file.json costume
  python shops_find_unique_item_id_from_kurodlc.py file.json costume+item

Example 2: Generate template from all sections
  python shops_find_unique_item_id_from_kurodlc.py file.json --generate-template

Example 3: Generate template from specific sections
  python shops_find_unique_item_id_from_kurodlc.py file.json --generate-template costume+item

Example 4: Generate template with custom shop IDs
  python shops_find_unique_item_id_from_kurodlc.py file.json --generate-template --shop-ids=1,5,10

Example 5: Generate template with custom output name
  python shops_find_unique_item_id_from_kurodlc.py file.json --generate-template --output=my_config.json

Example 6: Generate template, use default shop ID if not found
  python shops_find_unique_item_id_from_kurodlc.py file.json --generate-template costume --default-shop-ids

=============================================================================
TEMPLATE GENERATION WORKFLOW
=============================================================================

1. Extract item IDs from your DLC:
   python shops_find_unique_item_id_from_kurodlc.py my_dlc.json --generate-template costume

2. Edit generated template_my_dlc.json:
   - Review item_ids (extracted automatically)
   - Modify shop_ids as needed
   - Customize template structure if needed

3. Generate shop assignments:
   python shops_create.py template_my_dlc.json

4. Integrate output into your .kurodlc.json file
""")

def is_valid_kurodlc_structure(data):
    """Validate .kurodlc.json structure."""
    required_keys = ["CostumeParam", "DLCTableData"]
    for key in required_keys:
        if key not in data or not isinstance(data[key], list):
            return False

    for item in data["CostumeParam"]:
        if not isinstance(item, dict) or "item_id" not in item or "mdl_name" not in item:
            return False

    for item in data["DLCTableData"]:
        if not isinstance(item, dict) or "id" not in item or "items" not in item:
            return False
        if not isinstance(item["items"], list) or not all(isinstance(x, int) for x in item["items"]):
            return False

    if "ItemTableData" in data:
        if not isinstance(data["ItemTableData"], list):
            return False
        for item in data["ItemTableData"]:
            if not isinstance(item, dict) or "id" not in item or "name" not in item:
                return False

    if "ShopItem" in data:
        if not isinstance(data["ShopItem"], list):
            return False
        for item in data["ShopItem"]:
            if not isinstance(item, dict) or "item_id" not in item:
                return False

    return True

def extract_ids_by_mode(data, requested_modes):
    """Extract item IDs based on requested modes."""
    all_ids = []
    extraction_summary = []

    if "shop" in requested_modes:
        if "ShopItem" in data:
            shop_item_ids = [item["item_id"] for item in data["ShopItem"] if "item_id" in item]
            all_ids.extend(shop_item_ids)
            extraction_summary.append(f"ShopItem: {len(shop_item_ids)} IDs")
        else:
            extraction_summary.append("ShopItem: not present in file")

    if "costume" in requested_modes:
        costume_item_ids = [item["item_id"] for item in data.get("CostumeParam", []) if "item_id" in item]
        all_ids.extend(costume_item_ids)
        extraction_summary.append(f"CostumeParam: {len(costume_item_ids)} IDs")

    if "item" in requested_modes:
        if "ItemTableData" in data:
            item_table_ids = [item["id"] for item in data["ItemTableData"] if "id" in item]
            all_ids.extend(item_table_ids)
            extraction_summary.append(f"ItemTableData: {len(item_table_ids)} IDs")
        else:
            extraction_summary.append("ItemTableData: not present in file")

    if "dlc" in requested_modes:
        dlc_table_ids = []
        for item in data.get("DLCTableData", []):
            if "items" in item and isinstance(item["items"], list):
                dlc_table_ids.extend(item["items"])
        all_ids.extend(dlc_table_ids)
        extraction_summary.append(f"DLCTableData.items: {len(dlc_table_ids)} IDs")

    return all_ids, extraction_summary

def extract_shop_ids(data):
    """Extract unique shop IDs from ShopItem section."""
    if "ShopItem" not in data:
        return []
    
    shop_ids = [item["shop_id"] for item in data["ShopItem"] if "shop_id" in item]
    return sorted(set(shop_ids))

def extract_template_from_shop_item(data):
    """Extract template structure from first ShopItem entry."""
    if "ShopItem" not in data or not data["ShopItem"]:
        return None
    
    # Get first shop item as template
    first_item = data["ShopItem"][0]
    
    # Create template by replacing values with variables
    template = {}
    for key, value in first_item.items():
        if key == "shop_id":
            template[key] = "${shop_id}"
        elif key == "item_id":
            template[key] = "${item_id}"
        else:
            # Keep other fields as-is
            template[key] = value
    
    return template

def generate_template_config(json_file, data, requested_modes, shop_ids=None, 
                            output_file=None, default_shop_ids=False):
    """Generate template config file for shops_create.py."""
    
    # Extract item IDs
    all_ids, extraction_summary = extract_ids_by_mode(data, requested_modes)
    unique_item_ids = sorted(set(all_ids))
    
    # Determine shop IDs
    if shop_ids is not None:
        # Manually specified
        final_shop_ids = shop_ids
        shop_id_source = "manually specified"
    else:
        # Try to extract from ShopItem
        extracted_shop_ids = extract_shop_ids(data)
        if extracted_shop_ids:
            final_shop_ids = extracted_shop_ids
            shop_id_source = "extracted from ShopItem section"
        elif default_shop_ids:
            final_shop_ids = [1]
            shop_id_source = "default [1]"
        else:
            # Prompt user
            print("\nNo shop IDs found in ShopItem section.")
            print("Please enter shop IDs (comma-separated, e.g., 1,5,10):")
            shop_input = input("> ").strip()
            try:
                final_shop_ids = [int(x.strip()) for x in shop_input.split(",")]
                shop_id_source = "user input"
            except ValueError:
                print("Error: Invalid shop IDs. Using default [1].")
                final_shop_ids = [1]
                shop_id_source = "default [1] (after error)"
    
    # Extract template structure
    template = extract_template_from_shop_item(data)
    
    # Create config
    config = {
        "_comment": [
            "Template config file generated by shops_find_unique_item_id_from_kurodlc.py v2.0",
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
        ] + [f"  - {line}" for line in extraction_summary] + [
            f"  - Total unique item IDs: {len(unique_item_ids)}",
            f"  - Shop IDs: {shop_id_source}",
        ],
        "item_ids": unique_item_ids,
        "shop_ids": final_shop_ids
    }
    
    # Add template if extracted
    if template:
        config["template"] = template
        config["_comment"].append("  - Template: extracted from first ShopItem entry")
    else:
        config["_comment"].append("  - Template: not found (will use shops_create.py default)")
    
    # Determine output filename
    if output_file:
        output_path = Path(output_file)
    else:
        input_path = Path(json_file)
        output_path = input_path.with_name(f"template_{input_path.name}")
    
    # Write config file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"Template config file generated successfully!")
        print(f"{'='*60}")
        print(f"Output file: {output_path}")
        print(f"\nSummary:")
        print(f"  - Item IDs extracted: {len(unique_item_ids)}")
        print(f"  - Shop IDs: {final_shop_ids} ({shop_id_source})")
        if template:
            print(f"  - Template: Extracted from ShopItem")
        else:
            print(f"  - Template: Will use shops_create.py default")
        print(f"\nExtraction details:")
        for line in extraction_summary:
            print(f"  - {line}")
        print(f"\n{'='*60}")
        print(f"Next steps:")
        print(f"  1. Edit {output_path} (modify shop_ids if needed)")
        print(f"  2. Run: python shops_create.py {output_path}")
        print(f"  3. Integrate output into your .kurodlc.json file")
        print(f"{'='*60}\n")
        
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
        
        elif not arg.startswith("--"):
            # Regular mode argument
            mode_arg = arg
        
        i += 1
    
    # Check if file exists
    if not os.path.exists(json_file):
        print(f"Error: The file '{json_file}' does not exist.")
        sys.exit(1)
    
    # Load JSON file
    try:
        with open(json_file, "r", encoding="utf-8") as f:
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
                                manual_shop_ids, output_file, default_shop_ids)
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
