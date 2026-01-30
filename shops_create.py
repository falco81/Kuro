import json
import sys
from pathlib import Path

# argument check
if len(sys.argv) < 2:
    print(
        "Usage: python shops_create.py path_to_config_file.json\n"
        "\n"
        "This script generates a JSON file that assigns items to shops.\n"
        "You need to provide a configuration JSON file with the following structure:\n"
        "{\n"
        "    \"item_ids\": [list of item IDs],\n"
        "    \"shop_ids\": [list of shop IDs]\n"
        "}\n"
        "\n"
        "Example:\n"
        "  python shops_create.py my_config.json\n"
        "This will create 'output_my_config.json' containing all combinations of items and shops."
    )
    sys.exit(1)

config_path = Path(sys.argv[1])

if not config_path.exists():
    print(f"Error: File '{config_path}' does not exist.")
    sys.exit(1)

# load configuration
try:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON in '{config_path}': {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading config file: {e}")
    sys.exit(1)

item_ids = config.get("item_ids", [])
shop_ids = config.get("shop_ids", [])

# IMPROVED: Validate configuration
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

# Generate shop items
shop_items = []

for item_id in item_ids:
    for shop_id in shop_ids:
        shop_items.append({
            "shop_id": shop_id,
            "item_id": item_id,
            "unknown": 1,
            "start_scena_flags": [],
            "empty1": 0,
            "end_scena_flags": [],
            "int2": 0
        })

# wrap result
result = {
    "ShopItem": shop_items
}

# output file name
output_path = config_path.with_name(f"output_{config_path.name}")

try:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"Success: File '{output_path.name}' was created successfully.")
    print(f"Generated {len(shop_items)} shop item entries:")
    print(f"  - {len(item_ids)} items")
    print(f"  - {len(shop_ids)} shops")
    print(f"  - Total combinations: {len(item_ids)} × {len(shop_ids)} = {len(shop_items)}")
except Exception as e:
    print(f"Error writing output file: {e}")
    sys.exit(1)
