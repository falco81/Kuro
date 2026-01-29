import json
import sys
import os

# Check if an argument was provided
if len(sys.argv) < 2:
    print(
        "Usage: python shops_find_unique_item_id_from_kurodlc.py <.kurodlc.json> [shop|costume]\n"
        "\n"
        "This script extracts all unique item IDs from a .kurodlc JSON file.\n"
        "Default behavior (no parameter):\n"
        "  - Combines IDs from 'ShopItem' and 'CostumeParam'\n"
        "\n"
        "Optional parameters:\n"
        "  shop     -> search only in 'ShopItem'\n"
        "  costume  -> search only in 'CostumeParam'\n"
    )
    sys.exit(1)

# File name from argument
json_file = sys.argv[1]

# Optional mode argument
mode = None
if len(sys.argv) >= 3:
    mode = sys.argv[2].lower()
    if mode not in ("shop", "costume"):
        print(f"Error: Unknown parameter '{mode}'.")
        print("Allowed parameters are: shop, costume")
        sys.exit(1)

# Check if the file actually exists
if not os.path.exists(json_file):
    print(f"Error: The file '{json_file}' does not exist.")
    sys.exit(1)  # Exit the program with an error

# Load JSON file
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

def is_valid_kurodlc_structure(data):
    # Kontrola hlavních klíčů
    required_keys = ["CostumeParam", "DLCTableData", "ItemTableData"]
    for key in required_keys:
        if key not in data or not isinstance(data[key], list):
            return False

    # Kontrola základních polí v CostumeParam
    for item in data["CostumeParam"]:
        if not isinstance(item, dict) or "item_id" not in item or "mdl_name" not in item:
            return False

    # Kontrola základních polí v DLCTableData
    for item in data["DLCTableData"]:
        if not isinstance(item, dict) or "id" not in item or "items" not in item:
            return False

    # Kontrola základních polí v ItemTableData
    for item in data["ItemTableData"]:
        if not isinstance(item, dict) or "id" not in item or "name" not in item:
            return False

    return True

if not is_valid_kurodlc_structure(data):
    print(f"Error: JSON file '{json_file}' does not have a valid kurodlc structure.")
    sys.exit(1)

shop_item_ids = []
costume_item_ids = []

# Extract item_id values from ShopItem
if mode is None or mode == "shop":
    shop_item_ids = [
        item["item_id"]
        for item in data.get("ShopItem", [])
        if "item_id" in item
    ]

# Extract item_id values from CostumeParam
if mode is None or mode == "costume":
    costume_item_ids = [
        item["item_id"]
        for item in data.get("CostumeParam", [])
        if "item_id" in item
    ]

# Combine and get unique item_id values
unique_item_ids = sorted(set(shop_item_ids + costume_item_ids))

# Print result
print(unique_item_ids)
