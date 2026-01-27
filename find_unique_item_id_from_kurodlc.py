import json
import sys
import os

def extract_item_ids(json_file):
    """
    Load a .kurodlc.json file and return a list of item_id values
    from the CostumeParam section.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        item["item_id"]
        for item in data.get("CostumeParam", [])
        if "item_id" in item
    ]

def get_all_kurodlc_files():
    """
    Return all *.kurodlc.json files (case-insensitive) in the current directory.
    """
    files = []
    for f in os.listdir("."):
        if f.lower().endswith(".kurodlc.json") and os.path.isfile(f):
            files.append(f)

    if not files:
        print("No .kurodlc.json files found in the current directory.")
        sys.exit(0)

    return files

# ---- Argument handling -----------------------------------------------------

if len(sys.argv) < 2:
    print("""
Usage:
  python find_unique_item_id_from_kurodlc.py <mode_or_file>

Description:
  Extracts item_id values from the "CostumeParam" section of *.kurodlc.json files.

Modes:
  <file.kurodlc.json>
      Process a single file and print unique item_id values as a sorted list.

  searchall
      Process all *.kurodlc.json files in the current directory and print
      all unique item_id values as a single sorted list.

  searchallline
      Same as searchall, but print each unique item_id on a separate line.

  searchallbydlc
      For each *.kurodlc.json file:
        - print the file name
        - print item_id values found in that file as a list
      Then print all unique item_id values across all files as a list.

  searchallbydlcline
      Same as searchallbydlc, but item_id values are printed line by line.
      The final summary of unique item_id values is also printed line by line.

Examples:
  python find_unique_item_id_from_kurodlc.py costume1.kurodlc.json
  python find_unique_item_id_from_kurodlc.py searchall
  python find_unique_item_id_from_kurodlc.py searchallline
  python find_unique_item_id_from_kurodlc.py searchallbydlc
  python find_unique_item_id_from_kurodlc.py searchallbydlcline
""")
    sys.exit(1)

raw_arg = sys.argv[1]
arg = raw_arg.lower()

# ---- Modes -----------------------------------------------------------------

if arg == "searchall":
    all_item_ids = []
    for f in get_all_kurodlc_files():
        all_item_ids.extend(extract_item_ids(f))

    print(sorted(set(all_item_ids)))

elif arg == "searchallline":
    all_item_ids = []
    for f in get_all_kurodlc_files():
        all_item_ids.extend(extract_item_ids(f))

    for item_id in sorted(set(all_item_ids)):
        print(item_id)

elif arg == "searchallbydlc":
    all_item_ids = []

    for f in get_all_kurodlc_files():
        item_ids = extract_item_ids(f)
        all_item_ids.extend(item_ids)

        print(f"{f}:")
        print(item_ids)
        print()

    print("Unique item_ids across all files:")
    print(sorted(set(all_item_ids)))

elif arg == "searchallbydlcline":
    all_item_ids = []

    for f in get_all_kurodlc_files():
        item_ids = extract_item_ids(f)
        all_item_ids.extend(item_ids)

        print(f"{f}:")
        for item_id in sorted(set(item_ids)):
            print(item_id)
        print()

    print("Unique item_ids across all files:")
    for item_id in sorted(set(all_item_ids)):
        print(item_id)

# ---- Single file -----------------------------------------------------------

else:
    if not os.path.isfile(raw_arg):
        print(f"Error: Unknown parameter or file not found: '{raw_arg}'")
        print("Use one of: searchall, searchallline, searchallbydlc, searchallbydlcline")
        sys.exit(1)

    if not raw_arg.lower().endswith(".kurodlc.json"):
        print(f"Error: Invalid file type: '{raw_arg}' (expected .kurodlc.json)")
        sys.exit(1)

    unique_item_ids = sorted(set(extract_item_ids(raw_arg)))
    print(unique_item_ids)
