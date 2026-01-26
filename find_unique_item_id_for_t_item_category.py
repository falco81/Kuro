import json
import sys

def get_unique_ids_by_category(json_path, category):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    unique_ids = set()

    for block in data.get("data", []):
        if block.get("name") == "ItemTableData":
            for item in block.get("data", []):
                if item.get("category") == category:
                    unique_ids.add(item.get("id"))

    return sorted(unique_ids)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Použití: python script.py <t_item.json> <category>")
        sys.exit(1)

    json_file = sys.argv[1]
    category = int(sys.argv[2])

    ids = get_unique_ids_by_category(json_file, category)

    # výstup ve formátu Python listu
    print(ids)
