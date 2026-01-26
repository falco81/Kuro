import json
import sys

# Check if an argument was provided
if len(sys.argv) < 2:
    print("Usage: python find_unique_shop_item_id.py .kurodlc.json")
    sys.exit(1)

# File name from argument
json_file = sys.argv[1]

# Load JSON file
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract all item_id values
item_ids = [item["item_id"] for item in data.get("ShopItem", [])]

# Get unique item_id values
unique_item_ids = sorted(set(item_ids))

# Print result
print(unique_item_ids)
