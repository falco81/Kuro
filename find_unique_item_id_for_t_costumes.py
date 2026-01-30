import json
import sys

def get_unique_ids_by_category(json_path):
    """
    Extract unique item IDs from CostumeParam category.
    
    IMPROVED: Added error handling and None filtering.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{json_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{json_path}': {e}")
        sys.exit(1)

    unique_ids = set()

    for block in data.get("data", []):
        if block.get("name") == "CostumeParam":
            for item in block.get("data", []):
                item_id = item.get("item_id")
                # FIXED: Filter out None values
                if item_id is not None:
                    unique_ids.add(item_id)

    if not unique_ids:
        print("Warning: No item IDs found in CostumeParam category.")
        return []

    return sorted(unique_ids)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python find_unique_item_id_for_t_costumes.py <t_costume.json>\n"
            "\n"
            "This script extracts all unique item IDs from the 'CostumeParam' category\n"
            "in a t_costume JSON file.\n"
            "\n"
            "Arguments:\n"
            "  t_costume.json   Path to the JSON file containing costume data.\n"
            "\n"
            "Example:\n"
            "  python find_unique_item_id_for_t_costumes.py t_costume.json\n"
            "      Outputs a sorted list of unique item IDs from the 'CostumeParam' category."
        )
        sys.exit(1)

    json_file = sys.argv[1]

    ids = get_unique_ids_by_category(json_file)

    # Print result
    print(ids)
