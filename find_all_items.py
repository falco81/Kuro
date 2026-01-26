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
        print("Usage: python find_all_items.py t_item.json [search_text] [search_id]")
        return

    filename = sys.argv[1]
    search_text = sys.argv[2].lower() if len(sys.argv) >= 3 else None
    search_id = sys.argv[3] if len(sys.argv) >= 4 else None

    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
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
            if search_id == sid  # přesná shoda
        }

    if not filtered:
        print("No matching items found.")
        return

    max_len = max(len(sid) for sid in filtered.keys())

    for shop_id, item_name in filtered.items():
        print(f"{shop_id.rjust(max_len)} : {item_name}")


if __name__ == "__main__":
    main()
