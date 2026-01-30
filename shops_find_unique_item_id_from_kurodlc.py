import json
import sys
import os

# Check if an argument was provided
if len(sys.argv) < 2:
    print(
        "Usage: python shops_find_unique_item_id_from_kurodlc.py <.kurodlc.json> [mode]\n"
        "\n"
        "This script extracts all unique item IDs from a .kurodlc JSON file.\n"
        "\n"
        "IMPROVED: Individual modes for each section!\n"
        "\n"
        "Available modes:\n"
        "  all         -> Extract from ALL sections (default)\n"
        "                 Includes: ShopItem, CostumeParam, ItemTableData, DLCTableData.items\n"
        "\n"
        "  shop        -> Extract only from 'ShopItem' section\n"
        "  costume     -> Extract only from 'CostumeParam' section\n"
        "  item        -> Extract only from 'ItemTableData' section (uses 'id' field)\n"
        "  dlc         -> Extract only from 'DLCTableData.items' section\n"
        "\n"
        "Combination modes (use + to combine):\n"
        "  shop+costume     -> ShopItem + CostumeParam\n"
        "  costume+item     -> CostumeParam + ItemTableData\n"
        "  item+dlc         -> ItemTableData + DLCTableData.items\n"
        "  shop+item+dlc    -> ShopItem + ItemTableData + DLCTableData.items\n"
        "  ... any combination you want!\n"
        "\n"
        "Examples:\n"
        "  python shops_find_unique_item_id_from_kurodlc.py file.json\n"
        "      -> Extracts from all sections (default)\n"
        "\n"
        "  python shops_find_unique_item_id_from_kurodlc.py file.json shop\n"
        "      -> Extracts only from ShopItem\n"
        "\n"
        "  python shops_find_unique_item_id_from_kurodlc.py file.json costume+item\n"
        "      -> Extracts from CostumeParam and ItemTableData\n"
        "\n"
        "  python shops_find_unique_item_id_from_kurodlc.py file.json dlc\n"
        "      -> Extracts only from DLCTableData.items\n"
    )
    sys.exit(1)

# File name from argument
json_file = sys.argv[1]

# Optional mode argument
mode_arg = "all"  # Default to all sections
if len(sys.argv) >= 3:
    mode_arg = sys.argv[2].lower()

# Parse mode - can be single mode or combination (e.g., "shop+costume")
valid_modes = {"shop", "costume", "item", "dlc", "all"}
requested_modes = set()

if mode_arg == "all":
    requested_modes = {"shop", "costume", "item", "dlc"}
else:
    # Split by + for combination modes
    parts = mode_arg.split("+")
    for part in parts:
        part = part.strip()
        if part not in valid_modes or part == "all":
            print(f"Error: Unknown mode '{part}'.")
            print("Valid modes are: shop, costume, item, dlc, all")
            print("You can combine modes with +, e.g., 'shop+costume'")
            sys.exit(1)
        requested_modes.add(part)

# Check if the file actually exists
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

def is_valid_kurodlc_structure(data):
    """
    Validate .kurodlc.json structure.
    
    Only CostumeParam and DLCTableData are required.
    ItemTableData and ShopItem are optional.
    """
    # Check required keys
    required_keys = ["CostumeParam", "DLCTableData"]
    for key in required_keys:
        if key not in data or not isinstance(data[key], list):
            return False

    # Validate CostumeParam fields
    for item in data["CostumeParam"]:
        if not isinstance(item, dict) or "item_id" not in item or "mdl_name" not in item:
            return False

    # Validate DLCTableData fields
    for item in data["DLCTableData"]:
        if not isinstance(item, dict) or "id" not in item or "items" not in item:
            return False
        if not isinstance(item["items"], list) or not all(isinstance(x, int) for x in item["items"]):
            return False

    # Validate ItemTableData fields (OPTIONAL)
    if "ItemTableData" in data:
        if not isinstance(data["ItemTableData"], list):
            return False
        for item in data["ItemTableData"]:
            if not isinstance(item, dict) or "id" not in item or "name" not in item:
                return False

    # Validate ShopItem fields (OPTIONAL)
    if "ShopItem" in data:
        if not isinstance(data["ShopItem"], list):
            return False
        for item in data["ShopItem"]:
            if not isinstance(item, dict) or "item_id" not in item:
                return False

    return True

if not is_valid_kurodlc_structure(data):
    print(f"Error: JSON file '{json_file}' does not have a valid kurodlc structure.")
    sys.exit(1)

# Extract item IDs based on requested modes
all_ids = []
extraction_summary = []

# Extract from ShopItem
if "shop" in requested_modes:
    if "ShopItem" in data:
        shop_item_ids = [
            item["item_id"]
            for item in data["ShopItem"]
            if "item_id" in item
        ]
        all_ids.extend(shop_item_ids)
        extraction_summary.append(f"ShopItem: {len(shop_item_ids)} IDs")
    else:
        extraction_summary.append("ShopItem: not present in file")

# Extract from CostumeParam
if "costume" in requested_modes:
    costume_item_ids = [
        item["item_id"]
        for item in data.get("CostumeParam", [])
        if "item_id" in item
    ]
    all_ids.extend(costume_item_ids)
    extraction_summary.append(f"CostumeParam: {len(costume_item_ids)} IDs")

# Extract from ItemTableData
if "item" in requested_modes:
    if "ItemTableData" in data:
        item_table_ids = [
            item["id"]
            for item in data["ItemTableData"]
            if "id" in item
        ]
        all_ids.extend(item_table_ids)
        extraction_summary.append(f"ItemTableData: {len(item_table_ids)} IDs")
    else:
        extraction_summary.append("ItemTableData: not present in file")

# Extract from DLCTableData.items
if "dlc" in requested_modes:
    dlc_table_ids = []
    for item in data.get("DLCTableData", []):
        if "items" in item and isinstance(item["items"], list):
            dlc_table_ids.extend(item["items"])
    all_ids.extend(dlc_table_ids)
    extraction_summary.append(f"DLCTableData.items: {len(dlc_table_ids)} IDs")

# Get unique item_id values
unique_item_ids = sorted(set(all_ids))

# Print extraction summary to stderr so it doesn't interfere with output
if extraction_summary:
    print("# Extraction summary:", file=sys.stderr)
    for line in extraction_summary:
        print(f"#   {line}", file=sys.stderr)
    print(f"# Total unique IDs: {len(unique_item_ids)}", file=sys.stderr)
    print("", file=sys.stderr)

# Print result to stdout (can be piped or redirected)
print(unique_item_ids)
