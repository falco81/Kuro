import json
import sys

def get_unique_ids_by_category(json_path, category):
    """
    Extract unique item IDs from a specific category in ItemTableData.
    
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
        if block.get("name") == "ItemTableData":
            for item in block.get("data", []):
                if item.get("category") == category:
                    item_id = item.get("id")
                    # FIXED: Filter out None values
                    if item_id is not None:
                        unique_ids.add(item_id)

    if not unique_ids:
        print(f"Warning: No item IDs found in category {category}.")
        return []

    return sorted(unique_ids)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python find_unique_item_id_for_t_item_category.py <t_item.json> <category>\n"
            "\n"
            "This script extracts all unique item IDs from a specified category in a t_item JSON file.\n"
            "\n"
            "Arguments:\n"
            "  t_item.json   Path to the JSON file containing item data.\n"
            "  category      Category number to filter items by (integer).\n"
            "\n"
            "Example:\n"
            "  python find_unique_item_id_for_t_item_category.py t_item.json 5\n"
            "      Outputs a sorted list of unique item IDs belonging to category 5."
        )
        sys.exit(1)

    json_file = sys.argv[1]
    
    # FIXED: Validate category input
    try:
        category = int(sys.argv[2])
    except ValueError:
        print(f"Error: Category must be an integer, got '{sys.argv[2]}'")
        sys.exit(1)

    ids = get_unique_ids_by_category(json_file, category)

    # Print result
    print(ids)
