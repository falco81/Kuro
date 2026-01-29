# KuroDLC Tools Collection

> A comprehensive toolkit for managing game items, shops, and DLC content in KuroDLC JSON format

[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

---

## 🎯 Overview

The **KuroDLC Tools Collection** is a set of Python utilities designed to help modders and developers work with game item databases, shop configurations, and DLC content files in KuroDLC JSON format. These tools provide functionality for:

- Searching and filtering game items and shops
- Extracting unique item IDs from various sources
- Creating shop configurations programmatically
- Detecting and resolving ID conflicts between game data and DLC mods
- Managing costume and item data across multiple files

### Key Use Cases

- **Mod Development**: Create custom items and shops without ID conflicts
- **Data Analysis**: Search and analyze existing game content
- **Quality Assurance**: Verify DLC content before release
- **Batch Operations**: Automate shop creation and item management

---

## ✨ Features

- 🔍 **Advanced Searching**: Filter items and shops by name, ID, or category
- 🎨 **Color-Coded Output**: Visual feedback for conflicts and status (requires colorama)
- 📊 **Multiple Data Sources**: Support for JSON, TBL, and P3A archive formats
- 🔧 **Automatic Conflict Resolution**: Intelligent ID reassignment with backup creation
- 📝 **Detailed Logging**: Verbose repair logs for tracking all changes
- 🛡️ **Safe Operations**: Automatic backups before any modifications
- 🚀 **Batch Processing**: Handle multiple DLC files simultaneously

---

## 📦 Prerequisites

### Required

- **Python 3.6+**
- Basic understanding of JSON format
- Game data files in supported formats

### Optional Dependencies

- **colorama** - For colored terminal output (Windows CMD support)
  ```bash
  pip install colorama
  ```

- **p3a_lib** - For extracting from P3A archives (required for P3A operations)
  - Must be available in the same directory or Python path
  ```python
    https://github.com/eArmada8/kuro_dlc_tool
  ```

- **kurodlc_lib** - For reading TBL format files
  - Must be available in the same directory or Python path
  ```python
    https://github.com/eArmada8/kuro_dlc_tool
  ```

---

## 🚀 Installation

1. **Clone or Download** this repository:
   ```bash
   git clone https://github.com/yourusername/kurodlc-tools.git
   cd kurodlc-tools
   ```

2. **Install Optional Dependencies** (recommended):
   ```bash
   pip install colorama
   ```

3. **Verify Installation**:
   ```bash
   python find_all_items.py
   ```
   You should see the usage instructions.

---

## 📚 Scripts Documentation

### Item Management

#### `find_all_items.py`

**Purpose**: Search and filter items from game item databases.

**Usage**:
```bash
python find_all_items.py <t_item.json> [search_text] [search_id]
```

**Arguments**:
- `t_item.json` - Path to the JSON file containing item data
- `search_text` - (Optional) Filter items by text in their name (case-insensitive)
- `search_id` - (Optional) Filter items by exact ID

**Features**:
- Recursively searches through nested JSON structures
- Case-insensitive text search
- Exact ID matching
- Formatted output with aligned columns

**Examples**:

```bash
# List all items from the file
python find_all_items.py t_item.json

# Find all items with "sepith" in their name
python find_all_items.py t_item.json sepith

# Find item with specific ID 310
python find_all_items.py t_item.json "" 310
```

**Sample Output**:
```
  310 : Earth Sepith
  311 : Water Sepith
  312 : Fire Sepith
  313 : Wind Sepith
  314 : Time Sepith
  315 : Space Sepith
  316 : Mirage Sepith
```

**Real-World Example**:
```bash
# Search for all tokens
python find_all_items.py t_item.json token

# Output:
  319 : G-Tokens
```

**Technical Details**:
- Uses recursive traversal to handle complex JSON structures
- Automatically formats output based on longest ID length
- Handles missing or malformed data gracefully

---

#### `find_unique_item_id_for_t_item_category.py`

**Purpose**: Extract all unique item IDs from a specific category in the item database.

**Usage**:
```bash
python find_unique_item_id_for_t_item_category.py <t_item.json> <category>
```

**Arguments**:
- `t_item.json` - Path to the JSON file containing item data
- `category` - Category number to filter items by (integer)

**Category Reference**:
Based on the game data:
- **Category 0**: Materials and Components (Sepith, etc.)
- **Category 1**: Food and Consumables
- **Category 2**: Books, Notes, and Quest Items
- **Category 15**: Accessories (Boots, Greaves, etc.)
- **Category 17**: Costumes and Outfits

**Examples**:

```bash
# Get all item IDs in category 0 (Materials)
python find_unique_item_id_for_t_item_category.py t_item.json 0

# Get all item IDs in category 15 (Accessories)
python find_unique_item_id_for_t_item_category.py t_item.json 15

# Get all item IDs in category 17 (Costumes)
python find_unique_item_id_for_t_item_category.py t_item.json 17
```

**Sample Output**:
```bash
# Category 0 (Materials):
[310, 311, 312, 313, 314, 315, 316, 317, 318, 319]

# Category 15 (Accessories):
[1200, 1201, 1210, 1220, 1230]
```

**Use Cases**:
- Finding available ID ranges for specific item types
- Analyzing category distribution
- Preparing data for mod creation

---

#### `find_unique_item_id_for_t_costumes.py`

**Purpose**: Extract all unique item IDs from costume data.

**Usage**:
```bash
python find_unique_item_id_for_t_costumes.py <t_costume.json>
```

**Arguments**:
- `t_costume.json` - Path to the JSON file containing costume data

**Examples**:

```bash
# Extract all costume item IDs
python find_unique_item_id_for_t_costumes.py t_costume.json
```

**Sample Output**:
```
[2500, 2501, 2502, 2503, 2504, 2505, 2506, 2600, 2601, 2602]
```

**Technical Details**:
- Specifically targets the `CostumeParam` section
- Returns sorted list of unique IDs
- Useful for avoiding conflicts when adding new costumes
- Common costume ID range: 2500-2999

---

### Shop Management

#### `find_all_shops.py`

**Purpose**: Search and filter shop data from shop databases.

**Usage**:
```bash
python find_all_shops.py <t_shop.json> [search_text]
```

**Arguments**:
- `t_shop.json` - Path to the JSON file containing shop data
- `search_text` - (Optional) Filter shops by text in their name (case-insensitive)

**Examples**:

```bash
# List all shops
python find_all_shops.py t_shop.json

# Find all newspaper shops
python find_all_shops.py t_shop.json newspaper

# Find bistro shops
python find_all_shops.py t_shop.json bistro
```

**Sample Output**:
```
  5 : Item Shop
  6 : Weapon/Armor Shop
  7 : Shared Table Test
  8 : Modification/Trade Shop
  9 : Kitchen
 10 : Orbments
 21 : Melrose Newspapers & Tobacco
 22 : Melrose Newspapers & Tobacco
 23 : Melrose Newspapers & Tobacco
 24 : Montmart Bistro
 25 : Montmart Bistro
 26 : Montmart Bistro
```

**Real-World Example**:
```bash
# Find all Melrose shops
python find_all_shops.py t_shop.json melrose

# Output:
 21 : Melrose Newspapers & Tobacco
 22 : Melrose Newspapers & Tobacco
 23 : Melrose Newspapers & Tobacco
```

**Features**:
- Recursive search through shop data
- Case-insensitive filtering
- Aligned output formatting
- Handles nested shop structures

---

#### `shops_create.py`

**Purpose**: Generate shop configurations by creating all combinations of items and shops from a configuration file.

**Usage**:
```bash
python shops_create.py <config_file.json>
```

**Input Format**:
The configuration file must have this structure:
```json
{
    "item_ids": [109, 110, 111, 112, 113],
    "shop_ids": [21, 22, 23]
}
```

**Real-World Example**:

Let's say you want to add the DOA Fortune Summer costumes to multiple Melrose shop locations:

**Step 1**: Create `doa_summer_config.json`:
```json
{
    "item_ids": [109, 110, 111, 112, 113],
    "shop_ids": [21, 22, 23]
}
```

**Step 2**: Run the script:
```bash
python shops_create.py doa_summer_config.json
```

**Output**:
Creates `output_doa_summer_config.json` with all item-shop combinations:

```json
{
    "ShopItem": [
        {
            "shop_id": 21,
            "item_id": 109,
            "unknown": 1,
            "start_scena_flags": [],
            "empty1": 0,
            "end_scena_flags": [],
            "int2": 0
        },
        {
            "shop_id": 21,
            "item_id": 110,
            "unknown": 1,
            "start_scena_flags": [],
            "empty1": 0,
            "end_scena_flags": [],
            "int2": 0
        }
        // ... 15 total entries (5 items × 3 shops)
    ]
}
```

**Use Cases**:
- Quickly adding multiple items to multiple shops
- Creating test data for shop systems
- Batch shop configuration for mods (like the DOA Fortune Summer Pack)

**Technical Details**:
- Generates Cartesian product of items × shops
- Creates properly structured ShopItem entries
- Output file is ready for use in DLC files

---

#### `shops_find_unique_item_id_from_kurodlc.py`

**Purpose**: Extract unique item IDs from DLC shop and costume data.

**Usage**:
```bash
python shops_find_unique_item_id_from_kurodlc.py <file.kurodlc.json> [shop|costume]
```

**Arguments**:
- `file.kurodlc.json` - Path to the DLC JSON file
- `shop` - (Optional) Search only in ShopItem section
- `costume` - (Optional) Search only in CostumeParam section
- *No parameter* - Search in both sections (default)

**Real-World Examples**:

```bash
# Extract all item IDs from DOA Fortune Summer Pack 2
python shops_find_unique_item_id_from_kurodlc.py DOAFortuneSummerMod2_kurodlc.json

# Extract only shop item IDs
python shops_find_unique_item_id_from_kurodlc.py DOAFortuneSummerMod2_kurodlc.json shop

# Extract only costume item IDs
python shops_find_unique_item_id_from_kurodlc.py DOAFortuneSummerMod2_kurodlc.json costume
```

**Sample Output**:
```
# All IDs (default):
[109, 110, 111, 112, 113]

# Costume only:
[109, 110, 111, 112, 113]

# Shop only:
[109, 110, 111, 112, 113]
```

**Use Cases**:
- Quick inventory of DLC content
- Verifying item assignments in costume packs
- Preparing data for conflict checking

---

### DLC ID Management

#### `find_unique_item_id_from_kurodlc.py`

**Purpose**: Comprehensive tool for extracting and checking item IDs across DLC files with multiple operation modes.

**Modes**:

##### 1. Single File Mode
```bash
python find_unique_item_id_from_kurodlc.py <file.kurodlc.json>
```
Extracts and prints all unique item IDs from a single DLC file.

**Example**:
```bash
python find_unique_item_id_from_kurodlc.py DOAFortuneSummerMod2_kurodlc.json
```

**Output**:
```
[109, 110, 111, 112, 113]
```

---

##### 2. Search All Mode
```bash
python find_unique_item_id_from_kurodlc.py searchall
```
Processes all `.kurodlc.json` files in the current directory and prints a single sorted list of unique item IDs.

**Example**:
```bash
python find_unique_item_id_from_kurodlc.py searchall
```

**Output** (with both DOA packs present):
```
[100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113]
```

---

##### 3. Search All By DLC Mode
```bash
python find_unique_item_id_from_kurodlc.py searchallbydlc
```
Processes all DLC files, showing item IDs per file, then displays unique IDs across all files.

**Example**:
```bash
python find_unique_item_id_from_kurodlc.py searchallbydlc
```

**Output**:
```
DOAFortuneSummerMod_kurodlc.json:
[100, 101, 102, 103, 104, 105, 106, 107, 108]

DOAFortuneSummerMod2_kurodlc.json:
[109, 110, 111, 112, 113]

Unique item_ids across all files:
[100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113]
```

---

##### 4. Search All By DLC Line Mode
```bash
python find_unique_item_id_from_kurodlc.py searchallbydlcline
```
Similar to `searchallbydlc` but prints each ID on a separate line.

**Example**:
```bash
python find_unique_item_id_from_kurodlc.py searchallbydlcline
```

**Output**:
```
DOAFortuneSummerMod_kurodlc.json:
100
101
102
103
104
105
106
107
108

DOAFortuneSummerMod2_kurodlc.json:
109
110
111
112
113

Unique item_ids across all files:
100
101
102
103
104
105
106
107
108
109
110
111
112
113
```

---

##### 5. Search All Line Mode
```bash
python find_unique_item_id_from_kurodlc.py searchallline
```
Processes all DLC files and prints each unique ID on a separate line without grouping.

**Example**:
```bash
python find_unique_item_id_from_kurodlc.py searchallline
```

**Output**:
```
100
101
102
103
104
105
106
107
108
109
110
111
112
113
```

---

##### 6. Check Mode ⭐ (Most Important)
```bash
python find_unique_item_id_from_kurodlc.py check [options]
```

Checks if item IDs in DLC files conflict with existing game data.

**Supported Data Sources**:
- `t_item.json` - JSON format item database
- `t_item.tbl` - Binary TBL format
- `t_item.tbl.original` - Original backup TBL file
- `script_en.p3a` / `script_eng.p3a` - P3A archives (auto-extracts)
- `zzz_combined_tables.p3a` - P3A archives (auto-extracts)

**Options**:
- `--source=<type>` - Force specific source: `json`, `tbl`, `original`, `p3a` or `zzz`
- `--no-interactive` - Skip source selection prompt (uses first available)
- `--keep-extracted` - Keep temporary files from P3A extraction

**Real-World Examples**:

```bash
# Interactive check (prompts for source if multiple available)
python find_unique_item_id_from_kurodlc.py check

# Force JSON source
python find_unique_item_id_from_kurodlc.py check --source=json

# Use P3A and keep extracted files
python find_unique_item_id_from_kurodlc.py check --source=p3a --keep-extracted

# Non-interactive mode (auto-select first source)
python find_unique_item_id_from_kurodlc.py check --no-interactive
```

**Sample Output** (checking DOA Fortune Summer mods):
```
  100 : available               [OK]
  101 : available               [OK]
  102 : available               [OK]
  103 : available               [OK]
  104 : available               [OK]
  105 : available               [OK]
  106 : available               [OK]
  107 : available               [OK]
  108 : available               [OK]
  109 : available               [OK]
  110 : available               [OK]
  111 : available               [OK]
  112 : available               [OK]
  113 : available               [OK]

Summary:
Total IDs : 14
OK        : 14
BAD       : 0

Source used for check: t_item.json
```

**Example with Conflicts** (if IDs were already used):
```
  109 : Earth Sepith            [BAD]
  110 : Water Sepith            [BAD]
  111 : available               [OK]
  112 : available               [OK]
  113 : available               [OK]

Summary:
Total IDs : 5
OK        : 3
BAD       : 2

Source used for check: t_item.json
```

**Color Coding** (with colorama):
- 🟢 **[OK]** - Item ID is available (not in use)
- 🔴 **[BAD]** - Item ID conflicts with existing game data

**Technical Details**:
- Automatically detects available data sources
- Interactive source selection when multiple sources exist
- Extracts from P3A archives on-the-fly if needed
- Handles cleanup of temporary files
- Comprehensive error handling

---

### Conflict Resolution

#### `resolve_id_conflicts_in_kurodlc.py`

**Purpose**: The most powerful tool in the collection - detects and automatically resolves ID conflicts between DLC content and game data.

**Modes**:

##### 1. Check By DLC Mode
```bash
python resolve_id_conflicts_in_kurodlc.py checkbydlc [options]
```

Checks all DLC files for conflicts, showing detailed results per file.

**Real-World Example**:

```bash
# Interactive check
python resolve_id_conflicts_in_kurodlc.py checkbydlc

# Force specific source
python resolve_id_conflicts_in_kurodlc.py checkbydlc --source=json

# Non-interactive mode
python resolve_id_conflicts_in_kurodlc.py checkbydlc --no-interactive
```

**Sample Output** (checking DOA Fortune Summer packs):
```
Processing file: DOAFortuneSummerMod_kurodlc.json

  100 : available               [OK]
  101 : available               [OK]
  102 : available               [OK]
  103 : available               [OK]
  104 : available               [OK]
  105 : available               [OK]
  106 : available               [OK]
  107 : available               [OK]
  108 : available               [OK]

Summary for this file:
Total IDs : 9
OK        : 9
BAD       : 0

Processing file: DOAFortuneSummerMod2_kurodlc.json

  109 : available               [OK]
  110 : available               [OK]
  111 : available               [OK]
  112 : available               [OK]
  113 : available               [OK]

Summary for this file:
Total IDs : 5
OK        : 5
BAD       : 0

Overall Summary:
Total IDs : 14
OK        : 14
BAD       : 0

Source used for check: t_item.json
```

**Example with Conflicts**:
```
Processing file: DOAFortuneSummerMod2_kurodlc.json

  109 : Earth Sepith            [BAD]
  110 : Water Sepith            [BAD]
  111 : available               [OK]
  112 : available               [OK]
  113 : available               [OK]

Summary for this file:
Total IDs : 5
OK        : 3
BAD       : 2

Overall Summary:
Total IDs : 5
OK        : 3
BAD       : 2

Source used for check: t_item.json
```

---

##### 2. Repair Mode 🔧 (Most Powerful)
```bash
python resolve_id_conflicts_in_kurodlc.py repair [options]
```

Generates a repair plan for all conflicting IDs. Use `--apply` to actually modify files.

**Options**:
- `--apply` - **Actually modify the DLC files** (creates backups first)
- `--source=<type>` - Force specific source
- `--no-interactive` - Skip source selection prompt
- `--keep-extracted` - Keep temporary P3A extraction files

**Workflow**:

**Step 1: Preview Repair Plan** (Safe, no changes)
```bash
python resolve_id_conflicts_in_kurodlc.py repair
```

This shows what would be changed without modifying anything.

**Sample Output** (if conflicts existed):
```
Processing file: DOAFortuneSummerMod2_kurodlc.json

  109 : Earth Sepith            [BAD]
  110 : Water Sepith            [BAD]
  111 : available               [OK]
  112 : available               [OK]
  113 : available               [OK]

Summary for this file:
Total IDs : 5
OK        : 3
BAD       : 2

Overall Summary:
Total IDs : 5
OK        : 3
BAD       : 2

Source used for check: t_item.json

------------------------------------------------------------
Repair log generated: repair_log.txt
------------------------------------------------------------
```

**repair_log.txt** would contain:
```
DOAFortuneSummerMod2_kurodlc.json: 109 -> 3000 (CostumeParam, ShopItem, DLCTableData)
DOAFortuneSummerMod2_kurodlc.json: 110 -> 3001 (CostumeParam, ShopItem, DLCTableData)
```

---

**Step 2: Apply Repairs** (Modifies files)
```bash
python resolve_id_conflicts_in_kurodlc.py repair --apply
```

**What Happens**:
1. ✅ Creates timestamped backups (e.g., `DOAFortuneSummerMod2_kurodlc.json.bak_20260128_143022.json`)
2. ✅ Modifies all affected sections: `CostumeParam`, `ShopItem`, `ItemTableData`, `DLCTableData`
3. ✅ Generates verbose logs showing exactly what changed
4. ✅ Preserves all other data in the files

**Sample Output**:
```
File       : DOAFortuneSummerMod2_kurodlc.json
Backup     : DOAFortuneSummerMod2_kurodlc.json.bak_20260128_143022.json
Verbose log: DOAFortuneSummerMod2_kurodlc.json.repair_verbose_20260128_143022.txt
------------------------------------------------------------

All changes applied with backups.
```

**Verbose Log Example** (`DOAFortuneSummerMod2_kurodlc.json.repair_verbose_20260128_143022.txt`):
```
File       : DOAFortuneSummerMod2_kurodlc.json
Backup     : DOAFortuneSummerMod2_kurodlc.json.bak_20260128_143022.json
Verbose log: DOAFortuneSummerMod2_kurodlc.json.repair_verbose_20260128_143022.txt

------------------------------------------------------------
CostumeParam :   109 ->  3000, mdl_name: chr5001_c107
------------------------------------------------------------
ShopItem     :   109 ->  3000, shop_id : 21
------------------------------------------------------------
ItemTableData:   109 ->  3000, name    : Agnès' Neo Venus Swimsuit
------------------------------------------------------------
DLCTableData :   109 ->  3000
------------------------------------------------------------
CostumeParam :   110 ->  3001, mdl_name: chr5005_c107
------------------------------------------------------------
ShopItem     :   110 ->  3001, shop_id : 21
------------------------------------------------------------
ItemTableData:   110 ->  3001, name    : Risette's Neo Venus Swimsuit
------------------------------------------------------------
```

---

**Advanced Examples**:

```bash
# Repair with JSON source and apply immediately
python resolve_id_conflicts_in_kurodlc.py repair --apply --source=json

# Repair using P3A, apply changes, keep extracted files
python resolve_id_conflicts_in_kurodlc.py repair --apply --source=p3a --keep-extracted

# Non-interactive repair with first available source
python resolve_id_conflicts_in_kurodlc.py repair --apply --no-interactive
```

---

**Safety Features**:

1. **Automatic Backups**:
   - Created before any modifications
   - Timestamped to prevent overwriting
   - Contains exact copy of original file

2. **Verbose Logging**:
   - Shows every single change made
   - Includes context (mdl_name, shop_id, character names)
   - Separate log file per modified DLC file

3. **Comprehensive Coverage**:
   - Updates all sections that reference the ID
   - Maintains data integrity across references
   - Preserves all other file content

4. **Smart ID Assignment**:
   - Finds next available ID in the range
   - Avoids conflicts with both game data and other DLC files
   - Sequentially assigns new IDs for multiple conflicts

---

**Sections Updated by Repair**:

The repair process intelligently updates IDs in all these sections:

| Section | ID Field | Context Logged | Example |
|---------|----------|----------------|----------------------|
| `CostumeParam` | `item_id` | `mdl_name` | chr5001_c107 |
| `ShopItem` | `item_id` | `shop_id` | 21 (Melrose Newspapers) |
| `ItemTableData` | `id` | `name` | Agnès' Neo Venus Swimsuit |
| `DLCTableData` (direct) | `items` array | - | [109, 110, 111, 112, 113] |
| `DLCTableData` (nested) | `ItemTableData[].id` | - | - |

---

**Common Options Summary**:

| Option | Description |
|--------|-------------|
| `--apply` | Actually modify files (creates backups) |
| `--source=json` | Force use of t_item.json |
| `--source=tbl` | Force use of t_item.tbl |
| `--source=original` | Force use of t_item.tbl.original |
| `--source=p3a` | Force use of P3A archive |
| `--source=zzz` | Force use of P3A zzz_combined_tables.p3a archive |
| `--no-interactive` | Skip prompts, use first available source |
| `--keep-extracted` | Keep temporary files from P3A extraction |

---

**File Handling**:

**Ignored Files**:
- Backup files (containing `.bak_` in filename)
- Temporary extraction files (unless `--keep-extracted` used)

**Generated Files**:
- `repair_log.txt` - Summary of all ID changes
- `*.bak_TIMESTAMP.json` - Backup before modifications
- `*.repair_verbose_TIMESTAMP.txt` - Detailed change log per file
- `t_item.tbl.original.tmp` - Temporary P3A extraction (auto-cleaned)

---

## 📄 File Format Reference

### t_item.json Structure

```json
{
    "headers": [
        {
            "name": "ItemTableData",
            "schema": "Kuro1"
        }
    ],
    "data": [
        {
            "name": "ItemTableData",
            "data": [
                {
                    "id": 310,
                    "name": "Earth Sepith",
                    "category": 0,
                    "chr_restrict": 0,
                    "price": 10,
                    "desc": "A crystalline material...",
                    // ... other properties
                }
            ]
        }
    ]
}
```

**Real Item Examples**:
- ID 310: Earth Sepith (Category 0 - Materials)
- ID 311: Water Sepith (Category 0 - Materials)
- ID 319: G-Tokens (Category 0 - Currency)
- ID 1200: Leather Boots (Category 15 - Accessories)
- ID 1201: Rigid Spikes (Category 15 - Accessories)

---

### t_costume.json Structure

```json
{
    "headers": [
        {
            "name": "CostumeParam",
            "schema": "Kuro1"
        }
    ],
    "data": [
        {
            "name": "CostumeParam",
            "data": [
                {
                    "item_id": 2500,
                    "mdl_name": "chr5001_c100",
                    "char_restrict": 1,
                    "type": 0,
                    // ... other properties
                }
            ]
        }
    ]
}
```

**Real Costume ID Range**: 2500-2999 (approximately)

---

### t_shop.json Structure

```json
{
    "headers": [
        {
            "name": "ShopInfo",
            "schema": "Kuro1"
        }
    ],
    "data": [
        {
            "name": "ShopInfo",
            "data": [
                {
                    "id": 21,
                    "shop_name": "Melrose Newspapers & Tobacco",
                    // ... other properties
                }
            ]
        }
    ]
}
```

**Real Shop Examples**:
- ID 5: Item Shop
- ID 6: Weapon/Armor Shop
- ID 8: Modification/Trade Shop
- ID 9: Kitchen
- ID 10: Orbments
- ID 21-23: Melrose Newspapers & Tobacco (multiple locations)
- ID 24-26: Montmart Bistro (multiple locations)
- ID 27-28: Stanley's Factory

---


## 🔄 Workflow Examples

### Example 1: Fixing ID Conflicts in Multiple DLC Files

**Scenario**: You downloaded two mods that have conflicting IDs.

**Step 1: Check All DLCs for Conflicts**
```bash
python resolve_id_conflicts_in_kurodlc.py checkbydlc
```

Output shows conflicts:
```
Processing file: mod1.kurodlc.json:
  310 : Earth Sepith    [BAD]
  311 : Water Sepith    [BAD]

Processing file: mod2.kurodlc.json:
  312 : Fire Sepith     [BAD]
```

**Step 2: Generate Repair Plan**
```bash
python resolve_id_conflicts_in_kurodlc.py repair
```

Review `repair_log.txt`:
```
mod1.kurodlc.json: 310 -> 4000
mod1.kurodlc.json: 311 -> 4001
mod2.kurodlc.json: 312 -> 4002
```

**Step 3: Apply Repairs**
```bash
python resolve_id_conflicts_in_kurodlc.py repair --apply
```

**Step 4: Verify Repairs**
```bash
python resolve_id_conflicts_in_kurodlc.py checkbydlc
```

All items should now show `[OK]`.

**Step 5: Review Changes**

Check the verbose logs:
```bash
cat mod1.kurodlc.json.repair_verbose_20260128_143022.txt
```

---

### Example 2: Adding Items to All Bistro Locations

**Goal**: Add special food items to all Montmart Bistro locations.

**Step 1: Find All Bistro Shop IDs**
```bash
python find_all_shops.py t_shop.json bistro
```

Output:
```
 24 : Montmart Bistro
 25 : Montmart Bistro
 26 : Montmart Bistro
```

**Step 2: Choose Available Item IDs**
```bash
python find_unique_item_id_from_kurodlc.py check
```

Choose item IDs to add to shops.

**Step 3: Create Config File**

`bistro_items_config.json`:
```json
{
    "item_ids": [5000, 5001, 5002],
    "shop_ids": [24, 25, 26]
}
```

**Step 4: Generate Shop File**
```bash
python shops_create.py bistro_items_config.json
```

**Step 5: Use Generated File**

The `output_bistro_items_config.json` now contains the food items in all bistro locations!
You can manually merge with .kurodlc.json

---

### Example 3: Preparing a Mod Release (Complete Example)

**Goal**: Ensure your DOA-style costume mod has no conflicts before releasing.

**Step 1: Collect All DLC Files**

Place all your `.kurodlc.json` files in one directory:
- `my_costume_pack_1.kurodlc.json`
- `my_costume_pack_2.kurodlc.json`

**Step 2: Comprehensive Check**
```bash
python find_unique_item_id_from_kurodlc.py check
```

**Step 3: Fix Any Issues**
```bash
python resolve_id_conflicts_in_kurodlc.py repair --apply
```

**Step 4: Final Verification**
```bash
python resolve_id_conflicts_in_kurodlc.py checkbydlc
```

**Step 5: Extract ID List for Documentation**
```bash
python find_unique_item_id_from_kurodlc.py searchallbydlc
```

Output:
```
my_costume_pack_1.kurodlc.json:
[3100, 3101, 3102, 3103, 3104]

my_costume_pack_2.kurodlc.json:
[3105, 3106, 3107, 3108, 3109]
```

**Step 6: Document Changes**

Include in your README:
```
ID Ranges Used:
- Pack 1: 3100-3104 (5 costumes)
- Pack 2: 3105-3109 (5 costumes)

Shops Modified:
- 21, 22, 23: Melrose Newspapers & Tobacco
```

**Step 7: Package Release**

Create release package:
```
my_costume_mod_v1.0/
  ├── my_costume_pack_1.kurodlc.json
  ├── my_costume_pack_2.kurodlc.json
  ├── README.txt (with ID ranges)
  ├── CHANGELOG.txt (from repair_log.txt if any repairs were made)
  └── screenshots/
```

---

### Example 4: Searching for Specific Item Types

**Goal**: Find all Sepith items in the game.

```bash
python find_all_items.py t_item.json sepith
```

Output:
```
  310 : Earth Sepith
  311 : Water Sepith
  312 : Fire Sepith
  313 : Wind Sepith
  314 : Time Sepith
  315 : Space Sepith
  316 : Mirage Sepith
  317 : Sepith Mass
  318 : All Element Sepith
```

---

## 🐛 Troubleshooting

### Common Issues

#### Issue: "No valid item source found"

**Cause**: None of the supported data files exist in the current directory.

**Solution**:
1. Make sure you have at least one of these files:
   - `t_item.json`
   - `t_item.tbl`
   - `t_item.tbl.original`
   - `script_en.p3a` or `script_eng.p3a`
   - `zzz_combined_tables.p3a`

2. Verify you're running the script in the correct directory

---

#### Issue: "Failed to extract t_item.tbl from P3A"

**Cause**: The P3A library is not available or the P3A file is corrupted.

**Solution**:
1. Ensure `p3a_lib.py` is in the same directory
2. Try using a different source: `--source=json` or `--source=tbl`
3. Verify the P3A file is not corrupted

---

#### Issue: Colors not showing on Windows

**Cause**: Colorama library not installed or not working.

**Solution**:
```bash
pip install colorama
```

If still not working:
```bash
pip uninstall colorama
pip install colorama --upgrade
```

---

#### Issue: "Module 'kurodlc_lib' not found"

**Cause**: Required library for reading TBL files is missing.

**Solution**:
1. Ensure `kurodlc_lib.py` is in the same directory or Python path
2. Use `--source=json` to work with JSON files instead
3. Contact the developer for the library file

---

#### Issue: Script modifies wrong IDs

**Cause**: Backup or snapshot files are being processed.

**Solution**:
The script automatically ignores files with `.bak_` in the name. Make sure your backup naming follows this pattern.

---

#### Issue: JSON decode error

**Cause**: Malformed JSON file.

**Solution**:
1. Validate your JSON using an online validator (jsonlint.com)
2. Check for:
   - Missing commas
   - Trailing commas
   - Unmatched brackets
   - Incorrect escape sequences (especially in character names with special chars like "Agnès")
3. Use a proper JSON editor with syntax highlighting (VS Code, Notepad++)

---

#### Issue: IDs still conflicting after repair

**Cause**: New conflicts introduced by other files or the game was updated.

**Solution**:
1. Run check again: `python resolve_id_conflicts_in_kurodlc.py checkbydlc`
2. Run repair again: `python resolve_id_conflicts_in_kurodlc.py repair --apply`
3. Consider using a higher ID range (e.g., 9000+)

---

#### Issue: Costume not showing in game after adding

**Cause**: Possible issues with model files or character restrictions.

**Solution**:
1. Verify `mdl_name` matches your model file (e.g., `chr5001_c107`)
2. Check `char_restrict` value matches the character ID
3. Ensure the item is added to correct shops
4. Verify the costume category is 17

---

### Getting Help

If you encounter issues not covered here:

1. **Check the Usage**:
   ```bash
   python scriptname.py
   ```
   (Shows detailed usage instructions)

2. **Verify File Formats**:
   - Ensure JSON files are valid
   - Check file encoding is UTF-8
   - Watch for special characters in names (Agnès, etc.)

3. **Enable Verbose Output**:
   - Review generated log files
   - Check verbose repair logs for details

4. **Create an Issue**:
   - Include the command you ran
   - Include error messages
   - Include relevant file snippets (remove sensitive data)
   - Mention which game version you're modding

---

## 📊 Quick Reference

### Command Cheat Sheet

```bash
# Search items
python find_all_items.py t_item.json
python find_all_items.py t_item.json sepith
python find_all_items.py t_item.json "" 310

# Search shops
python find_all_shops.py t_shop.json
python find_all_shops.py t_shop.json melrose
python find_all_shops.py t_shop.json bistro

# Extract IDs by category
python find_unique_item_id_for_t_item_category.py t_item.json 0
python find_unique_item_id_for_t_item_category.py t_item.json 15
python find_unique_item_id_for_t_item_category.py t_item.json 17

# Extract costume IDs
python find_unique_item_id_for_t_costumes.py t_costume.json

# Check DLC for conflicts
python find_unique_item_id_from_kurodlc.py check
python find_unique_item_id_from_kurodlc.py check --source=json

# Create shops
python shops_create.py config.json

# Extract DLC item IDs
python shops_find_unique_item_id_from_kurodlc.py DOAFortuneSummerMod2_kurodlc.json
python shops_find_unique_item_id_from_kurodlc.py DOAFortuneSummerMod2_kurodlc.json shop
python shops_find_unique_item_id_from_kurodlc.py DOAFortuneSummerMod2_kurodlc.json costume

# Check and repair conflicts
python resolve_id_conflicts_in_kurodlc.py checkbydlc
python resolve_id_conflicts_in_kurodlc.py repair
python resolve_id_conflicts_in_kurodlc.py repair --apply
```

---

### File Extensions Guide

| Extension | Description | Used By |
|-----------|-------------|---------|
| `.json` | JSON format data | All scripts |
| `.kurodlc.json` | DLC content file | DLC-specific scripts |
| `.tbl` | Binary table format | Check/repair scripts |
| `.tbl.original` | Original backup TBL | Check/repair scripts |
| `.p3a` | Archive format | Check/repair scripts |
| `.bak_*.json` | Auto-created backup | Repair script |
| `.repair_verbose_*.txt` | Detailed change log | Repair script |
| `repair_log.txt` | Summary change log | Repair script |
| `.tmp` | Temporary extraction | Auto-cleaned |

---

### ID Ranges Reference (Based on Game Data)

| Category | Range | Description | Examples |
|----------|-------|-------------|----------|
| 0 | 300-399 | Materials/Components | 310: Earth Sepith, 319: G-Tokens |
| 1 | 200-299 | Food/Consumables | 285: Soul Chef: Sausage |
| 2 | 300-399, 3000+ | Books/Quest Items | 300: Spriggan Notebook |
| 15 | 1200-1999 | Accessories | 1200: Leather Boots, 1201: Rigid Spikes |
| 17 | 2500-2999 | Costumes | 2500-2602 (base game costumes) |

**Recommended Ranges for Mods**:
- **Costumes**: 3000-3999 (to avoid base game range 2500-2999)
- **Custom Items**: 4000-4999
- **Quest Items**: 5000-5999

---

### Shop ID Reference (Real Game Shops)

| Shop ID | Shop Name | Location/Type |
|---------|-----------|---------------|
| 5 | Item Shop | General items |
| 6 | Weapon/Armor Shop | Equipment |
| 8 | Modification/Trade Shop | Upgrades |
| 9 | Kitchen | Food |
| 10 | Orbments | Quartz/Orbments |
| 21-23 | Melrose Newspapers & Tobacco | Multiple locations |
| 24-26 | Montmart Bistro | Food/Dining |
| 27-28 | Stanley's Factory | Industrial |

---



<div align="center">

**[⬆ Back to Top](#kurodlc-tools-collection)**

---


*Special thanks to the Trails modding community*

</div>
