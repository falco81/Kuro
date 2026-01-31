#!/usr/bin/env python3
"""
shops_create.py - v2.0 with Template Support

Generates shop item assignments from configuration with customizable templates.

NEW in v2.0:
- Custom template support in config file
- Variable substitution in templates (${shop_id}, ${item_id}, ${index}, ${count})
- Multiple output sections support
- Default template fallback (backward compatible)
- Template validation
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Default template (backward compatible with v1.0)
DEFAULT_TEMPLATE = {
    "shop_id": "${shop_id}",
    "item_id": "${item_id}",
    "unknown": 1,
    "start_scena_flags": [],
    "empty1": 0,
    "end_scena_flags": [],
    "int2": 0
}

def print_usage():
    """Print usage information."""
    print("""
Usage: python shops_create.py path_to_config_file.json

This script generates a JSON file that assigns items to shops.

=============================================================================
BASIC USAGE (Backward Compatible with v1.0)
=============================================================================

Simple config file structure:
{
    "item_ids": [100, 101, 102],
    "shop_ids": [1, 5, 10]
}

This creates all combinations (item × shop) with default template.

=============================================================================
ADVANCED USAGE (NEW in v2.0 - Template Support)
=============================================================================

Config with custom template:
{
    "item_ids": [100, 101, 102],
    "shop_ids": [1, 5, 10],
    "template": {
        "shop_id": "${shop_id}",
        "item_id": "${item_id}",
        "price": 1000,
        "stock": 99,
        "required_level": 10,
        "flags": []
    }
}

=============================================================================
TEMPLATE VARIABLES
=============================================================================

Available variables for substitution:
  ${shop_id}   - Current shop ID from shop_ids list
  ${item_id}   - Current item ID from item_ids list
  ${index}     - Index of current entry (0-based)
  ${count}     - Total number of entries

Example template with all variables:
{
    "template": {
        "shop_id": "${shop_id}",
        "item_id": "${item_id}",
        "entry_id": "${index}",
        "total_items": "${count}",
        "description": "Shop ${shop_id} - Item ${item_id}"
    }
}

=============================================================================
CUSTOM OUTPUT SECTION
=============================================================================

You can specify output section name:
{
    "item_ids": [100, 101],
    "shop_ids": [1, 2],
    "output_section": "CustomShopItems",
    "template": {
        "shop_id": "${shop_id}",
        "item_id": "${item_id}"
    }
}

Output will be:
{
    "CustomShopItems": [...]
}

Default section name is "ShopItem" if not specified.

=============================================================================
EXAMPLES
=============================================================================

Example 1: Basic usage (v1.0 compatible)
  python shops_create.py basic_config.json

Example 2: Custom template
  python shops_create.py custom_template_config.json

Example 3: Different section name
  python shops_create.py custom_section_config.json

=============================================================================
OUTPUT
=============================================================================

Creates: output_<config_name>.json

Example: If config is "my_shops.json"
Output:  "output_my_shops.json"
""")

def substitute_variables(template: Any, shop_id: int, item_id: int, index: int, total: int) -> Any:
    """
    Recursively substitute variables in template.
    
    Args:
        template: Template structure (can be dict, list, string, or primitive)
        shop_id: Current shop ID
        item_id: Current item ID
        index: Current entry index
        total: Total number of entries
    
    Returns:
        Template with variables substituted
    """
    if isinstance(template, dict):
        return {key: substitute_variables(value, shop_id, item_id, index, total) 
                for key, value in template.items()}
    
    elif isinstance(template, list):
        return [substitute_variables(item, shop_id, item_id, index, total) 
                for item in template]
    
    elif isinstance(template, str):
        # Substitute variables
        result = template
        result = result.replace("${shop_id}", str(shop_id))
        result = result.replace("${item_id}", str(item_id))
        result = result.replace("${index}", str(index))
        result = result.replace("${count}", str(total))
        
        # Try to convert to int if possible (for numeric fields)
        if result.isdigit() or (result.startswith('-') and result[1:].isdigit()):
            return int(result)
        
        return result
    
    else:
        # Primitive types (int, bool, None) - return as-is
        return template

def validate_template(template: Dict) -> bool:
    """
    Validate template structure.
    
    Args:
        template: Template dictionary
    
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(template, dict):
        return False
    
    # Template should have at least shop_id and item_id placeholders
    template_str = json.dumps(template)
    
    if "${shop_id}" not in template_str and "${item_id}" not in template_str:
        print("Warning: Template does not contain ${shop_id} or ${item_id} variables.")
        print("This may not generate the expected output.")
    
    return True

def generate_shop_items(item_ids: List[int], shop_ids: List[int], 
                       template: Dict = None) -> List[Dict]:
    """
    Generate shop items using template.
    
    Args:
        item_ids: List of item IDs
        shop_ids: List of shop IDs
        template: Template dictionary (uses default if None)
    
    Returns:
        List of generated shop items
    """
    if template is None:
        template = DEFAULT_TEMPLATE
    
    shop_items = []
    total = len(item_ids) * len(shop_ids)
    index = 0
    
    for item_id in item_ids:
        for shop_id in shop_ids:
            # Substitute variables in template
            item = substitute_variables(template, shop_id, item_id, index, total)
            shop_items.append(item)
            index += 1
    
    return shop_items

def main():
    """Main function."""
    # Argument check
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    
    if not config_path.exists():
        print(f"Error: File '{config_path}' does not exist.")
        sys.exit(1)
    
    # Load configuration
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{config_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)
    
    # Extract configuration
    item_ids = config.get("item_ids", [])
    shop_ids = config.get("shop_ids", [])
    template = config.get("template", None)
    output_section = config.get("output_section", "ShopItem")
    
    # Validate configuration
    if not item_ids:
        print("Error: 'item_ids' is empty or missing in config.")
        print("Config must contain: {\"item_ids\": [...], \"shop_ids\": [...]}")
        sys.exit(1)
    
    if not shop_ids:
        print("Error: 'shop_ids' is empty or missing in config.")
        print("Config must contain: {\"item_ids\": [...], \"shop_ids\": [...]}")
        sys.exit(1)
    
    if not isinstance(item_ids, list):
        print("Error: 'item_ids' must be a list.")
        sys.exit(1)
    
    if not isinstance(shop_ids, list):
        print("Error: 'shop_ids' must be a list.")
        sys.exit(1)
    
    if not all(isinstance(x, int) for x in item_ids):
        print("Error: All 'item_ids' must be integers.")
        invalid = [x for x in item_ids if not isinstance(x, int)]
        print(f"Invalid values: {invalid}")
        sys.exit(1)
    
    if not all(isinstance(x, int) for x in shop_ids):
        print("Error: All 'shop_ids' must be integers.")
        invalid = [x for x in shop_ids if not isinstance(x, int)]
        print(f"Invalid values: {invalid}")
        sys.exit(1)
    
    # Validate template if provided
    if template is not None:
        if not validate_template(template):
            print("Error: Invalid template structure.")
            sys.exit(1)
        print(f"Using custom template from config file.")
    else:
        print(f"Using default template (v1.0 compatible).")
    
    # Generate shop items
    print(f"\nGenerating shop items...")
    shop_items = generate_shop_items(item_ids, shop_ids, template)
    
    # Create output structure
    result = {
        output_section: shop_items
    }
    
    # Output file name
    output_path = config_path.with_name(f"output_{config_path.name}")
    
    # Write output
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"Success: File '{output_path.name}' was created successfully.")
        print(f"{'='*60}")
        print(f"Generated {len(shop_items)} shop item entries:")
        print(f"  - {len(item_ids)} items")
        print(f"  - {len(shop_ids)} shops")
        print(f"  - Total combinations: {len(item_ids)} × {len(shop_ids)} = {len(shop_items)}")
        print(f"  - Output section: '{output_section}'")
        if template:
            print(f"  - Template: Custom")
        else:
            print(f"  - Template: Default (v1.0 compatible)")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
