import json
import sys

def collect_items(node, result):
    if isinstance(node, dict):
        if "id" in node and "name" in node:
            result[str(node["id"])] = node["name"]

        for value in node.values():
            collect_items(value, result)

    elif isinstance(node, list):
        for item in node:
            collect_items(item, result)


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python find_all_items.py t_item.json [search_query]\n"
            "\n"
            "This script searches through a JSON file containing items and extracts all items with their IDs and names.\n"
            "\n"
            "Arguments:\n"
            "  t_item.json    Path to the JSON file containing items.\n"
            "  search_query   (Optional) Search query with optional prefix:\n"
            "\n"
            "Search modes:\n"
            "  id:NUMBER      - Search by exact ID (e.g., id:100)\n"
            "  name:TEXT      - Search in item names (e.g., name:100 or name:sword)\n"
            "  TEXT           - Auto-detect (numbers → ID search, text → name search)\n"
            "\n"
            "Examples:\n"
            "  python find_all_items.py t_item.json\n"
            "      Lists all items from the file.\n"
            "\n"
            "  python find_all_items.py t_item.json sword\n"
            "      Lists all items with 'sword' in their name (auto-detect).\n"
            "\n"
            "  python find_all_items.py t_item.json 100\n"
            "      Lists item with ID '100' (auto-detect: it's a number).\n"
            "\n"
            "  python find_all_items.py t_item.json name:100\n"
            "      Lists all items with '100' in their name (explicit name search).\n"
            "      Example results: 'Sword of 100', 'Level 100', '100 Gold'\n"
            "\n"
            "  python find_all_items.py t_item.json id:100\n"
            "      Lists the item with ID '100' (explicit ID search).\n"
            "\n"
            "IMPORTANT:\n"
            "  Use 'name:' prefix when searching for numbers in item names!\n"
            "  Otherwise, auto-detect will treat it as an ID search."
        )
        return

    filename = sys.argv[1]
    
    # Parse search query with optional prefix
    search_text = None
    search_id = None
    
    if len(sys.argv) >= 3:
        param = sys.argv[2]
        
        # Check for prefix
        if param.startswith('id:'):
            # Explicit ID search
            search_id = param[3:]
            if not search_id:
                print("Error: 'id:' prefix requires a value (e.g., id:100)")
                return
                
        elif param.startswith('name:'):
            # Explicit name search
            search_text = param[5:].lower()
            if not search_text:
                print("Error: 'name:' prefix requires a value (e.g., name:sword)")
                return
                
        else:
            # Auto-detect mode (original behavior)
            if param.isdigit():
                search_id = param
                # Inform user about auto-detection
                print(f"# Auto-detected ID search for '{param}'", file=sys.stderr)
                print(f"# Use 'name:{param}' to search for '{param}' in item names instead", file=sys.stderr)
                print("", file=sys.stderr)
            else:
                search_text = param.lower()

    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{filename}': {e}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    result = {}
    collect_items(data, result)

    filtered = result

    if search_text:
        filtered = {
            sid: name for sid, name in filtered.items()
            if search_text in str(name).lower()
        }

    if search_id:
        filtered = {
            sid: name for sid, name in filtered.items()
            if search_id == sid
        }

    if not filtered:
        print("No matching items found.")
        return

    max_len = max(len(sid) for sid in filtered.keys())

    for item_id, item_name in filtered.items():
        print(f"{item_id.rjust(max_len)} : {item_name}")


if __name__ == "__main__":
    main()
