# 🐍 KuroDLC shops bulk editing tool

A collection of small Python scripts for **analyzing and modifying `.kurodlc.json` files**, mainly focused on working with *shops* and *items* (IDs, categories, costumes).

The tools are intended for modding / data analysis workflows where you need to inspect existing DLC data or generate shop definitions automatically.

---

## 📦 Repository Contents

This repository contains the following Python scripts:

| Script | Description |
|------|-------------|
| `find_all_shops.py` | Finds and lists all shops defined in a `t_shop.json` file with the ability to search by shop name|
| `find_unique_shop_item_id.py` | Extracts unique `item_id` from `.kurodlc.json` file |
| `create_shops.py` | Generates shop definitions from a template JSON file |
| `find_unique_item_id_for_t_item_category.py` | Finds unique `item_id` values grouped by item category in a `t_item.json` file |
| `find_unique_item_id_for_t_costumes.py` | Extracts unique `item_id` values for costume items in a `t_costume.json` file |

---

## 🧠 Requirements

- Python **3.8+**
- No external dependencies (standard library only)

---

## 🚀 Usage

All scripts are executed from the command line.

### 1️⃣ Find all shops

Lists all shops found in a `t_shop.json` file with the ability to search by shop name case-insensitive.

```bash
python find_all_shops.py path/to/t_shop.json [search_text]
```

**Output:**
- Printed list of shop IDs

---

### 2️⃣ Find unique shop / item ID pairs

Extracts unique `item_id`

```bash
python find_unique_shop_item_id.py path/to/file.kurodlc.json
```

**Output:**
- List of unique item_id

---

### 3️⃣ Find unique item IDs by item category

Finds unique `item_id` values grouped by item category. (outfits category = 17 )

```bash
python find_unique_item_id_for_t_item_category.py path/to/t_item.json <category>
```

**Output:**
- Unique item IDs per category

---

### 4️⃣ Find unique costume item IDs

Extracts all unique `item_id` values for costume items.

```bash
python find_unique_item_id_for_t_costumes.py path/to/t_costume.json
```

**Output:**
- List of costume item IDs

---

### 5️⃣ Create shop definitions

Generates shop definitions using a template JSON file.

```bash
python create_shops.py create_shops_template.json
```

#### Template example

- item_ids from find_unique_shop_item_id.py output
- shop_ids from find_all_shops.py output

```json
{
  "item_ids": [4550, 4551, 4552],
  "shop_ids": [21, 22, 23, 248]
}
```

**Output:**
- `output_create_shops_template.json` containing generated shop definitions

This output can be manually merged into your main `.kurodlc.json` file.

---
## 📝 Notes

- Use to convert tbl to json **`tbl2json.py`**.
- https://github.com/nnguyen259/KuroTools

- Use to extract script_eng.p3a / script_en.p3a **`kurodlc_extract_original_tbls.py`**.
- https://github.com/eArmada8/kuro_dlc_tool
  
---
