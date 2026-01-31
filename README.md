# KuroDLC Modding Toolkit

A comprehensive Python toolkit for creating and managing DLC mods for games using the KuroDLC format. This toolkit provides utilities for item discovery, ID management, conflict resolution, and shop assignment automation.

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-GPLv3-green)](LICENSE)


## 📋 Table of Contents

- [Why This Toolkit Exists](#-why-this-toolkit-exists)
- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Scripts Overview](#-scripts-overview)
- [Detailed Documentation](#-detailed-documentation)
  - [Item Discovery Tools](#item-discovery-tools)
  - [ID Extraction Tools](#id-extraction-tools)
  - [Conflict Resolution](#conflict-resolution)
  - [Shop Management](#shop-management)
- [Common Workflows](#-common-workflows)
- [File Formats](#-file-formats)
- [Troubleshooting](#-troubleshooting)
- [Best Practices](#-best-practices)
- [Version History](#-version-history)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Why This Toolkit Exists

### The Problem

When creating DLC mods for Kuro engine games, modders face two major challenges:

**1. ID Conflicts (Primary Problem)**
- DLC mods use item IDs that may conflict with existing game items
- Manual ID conflict detection is tedious and error-prone
- A single conflicting ID can break your entire mod
- Game engine has a hard limit of 5000 IDs that cannot be expanded
- Finding safe, available ID ranges manually is time-consuming

**2. Shop Assignment Tedium (Secondary Problem)**
- Adding items to shops requires editing hundreds of entries manually
- Assigning 50 items to 10 shops = 500 manual entries
- Copy-paste errors are common
- No easy way to bulk-update shop assignments

### The Solution

This toolkit automates both problems:

**Primary: Automatic ID Conflict Resolution**
```bash
# One command to detect and fix all conflicts
python resolve_id_conflicts_in_kurodlc.py repair --apply
```
- Automatically finds safe IDs within the 5000 limit
- Smart distribution algorithm for better ID organization
- Detailed logging and automatic backups
- Manual control option for complex mods

**Secondary: Bulk Shop Assignment**
```bash
# Generate 500 shop assignments in seconds
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json --generate-template
python shops_create.py template_my_mod.json
```
- Batch assign items to multiple shops
- Customizable templates for different shop structures
- 50 items × 10 shops = 500 entries generated instantly

---

## ✨ Features

### Primary Purpose: ID Conflict Resolution
The main reason this toolkit was created - **automatic detection and resolution of item ID conflicts** between your DLC mods and game data. Never worry about conflicting IDs breaking your mods again!

- **⚠️ Conflict Detection**: Automatically detect ID conflicts between DLC and game data
- **🔧 Smart Resolution v2.7**: Intelligent ID assignment algorithm with 1-5000 range limit
- **🎯 Better Distribution**: IDs assigned from middle of range (2500) for optimal spacing
- **💾 Safety First**: Automatic backups and detailed logging for all modifications
- **✅ Validation**: Comprehensive .kurodlc.json structure validation

### Secondary Purpose: Bulk Shop Assignment
Quickly assign items to multiple shops without manual editing - **batch generate shop assignments** for your entire item set with a single command!

- **🛒 Shop Integration v2.0**: Generate shop assignments with customizable templates
- **📦 Bulk Operations**: Assign hundreds of items to multiple shops instantly
- **🎨 Custom Templates**: Define your own shop item structure

### Additional Tools
- **🔍 Item Discovery**: Search and browse game items from JSON, TBL, and P3A sources
- **📋 Multiple Formats**: Support for JSON, TBL, and P3A archive formats
- **🎨 User-Friendly**: Interactive menus and colored output (Windows CMD compatible)

---

## 📦 Requirements

### Core Requirements
- **Python 3.7+**
- Standard library modules (included): `json`, `sys`, `os`, `pathlib`, `shutil`, `datetime`, `glob`

### Optional Dependencies

For enhanced functionality:

[https://github.com/eArmada8/kuro_dlc_tool]

```bash
# For colored terminal output (recommended for Windows)
pip install colorama

# For P3A archive support (optional)
# Requires custom libraries: p3a_lib, kurodlc_lib
# Contact library maintainers or check documentation

```

**Quick Install (Windows):**
```batch
install_python_modules.bat
```

**Note**: The toolkit works fully without optional dependencies using JSON/TBL sources.

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/kurodlc-toolkit.git
cd kurodlc-toolkit

# Optional: Install colorama for colored output
pip install colorama

# Or use the included batch file (Windows)
install_python_modules.bat

# Verify installation
python resolve_id_conflicts_in_kurodlc.py
```

---

## ⚡ Quick Start

### Check for ID Conflicts

```bash
# Check all .kurodlc.json files in current directory
python resolve_id_conflicts_in_kurodlc.py checkbydlc
```

### Automatic Conflict Resolution (NEW v2.7 - Smart Algorithm!)

```bash
# Detect and fix conflicts automatically
# NEW: Uses smart algorithm to assign IDs in range 1-5000
# IDs are distributed from middle (2500) for better spacing
python resolve_id_conflicts_in_kurodlc.py repair --apply
```

**What's New in v2.7:**
- ✅ IDs guaranteed to stay within 1-5000 range
- ✅ Smart distribution starting from middle (2500)
- ✅ Finds continuous blocks when possible
- ✅ Clear errors if not enough IDs available

### Manual Conflict Resolution (Recommended for Production)

```bash
# Step 1: Export conflict report
python resolve_id_conflicts_in_kurodlc.py repair --export --export-name=my_mod

# Step 2: Edit id_mapping_my_mod.json (change new_id values as needed)

# Step 3: Apply your custom mappings
python resolve_id_conflicts_in_kurodlc.py repair --import --mapping-file=id_mapping_my_mod.json
```

### Generate Shop Assignments (NEW v2.0)

```bash
# Step 1: Generate template config from your DLC
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json --generate-template

# Step 2: (Optional) Edit template_my_mod.json to customize shop IDs

# Step 3: Generate shop assignments
python shops_create.py template_my_mod.json

# Step 4: Copy ShopItem section into your .kurodlc.json file
```

---

## 🛠️ Scripts Overview

| Script | Purpose | Version | Key Features |
|--------|---------|---------|--------------|
| **`resolve_id_conflicts_in_kurodlc.py`** | **Main conflict resolver** | v2.7.1 | Smart algorithm, auto/manual repair, validation, backups |
| **`shops_find_unique_item_id_from_kurodlc.py`** | **Extract IDs & generate templates** | v2.0 | Template generation, flexible extraction, shop ID detection |
| **`shops_create.py`** | **Generate shop assignments** | v2.0 | Custom templates, variable substitution |
| `find_all_items.py` | Search game items | v1.0 | ID/name search, auto-detect mode |
| `find_all_shops.py` | Search game shops | v1.0 | Filter by shop name |
| `find_unique_item_id_for_t_costumes.py` | Extract costume IDs | v1.0 | From CostumeParam section |
| `find_unique_item_id_for_t_item_category.py` | Extract category IDs | v1.0 | Filter by item category |
| `find_unique_item_id_from_kurodlc.py` | Extract DLC IDs | v1.0 | Multiple modes, conflict checking |

---

## 📖 Detailed Documentation

### Item Discovery Tools

#### `find_all_items.py` - Item Database Browser

Search and browse items from game data files.

**Basic Usage:**
```bash
python find_all_items.py t_item.json [search_query]
```

**Search Modes:**

| Mode | Syntax | Description | Example |
|------|--------|-------------|---------|
| **Auto-detect** | `TEXT` or `NUMBER` | Numbers → ID search<br>Text → name search | `sword`<br>`100` |
| **Explicit ID** | `id:NUMBER` | Search by exact ID | `id:100` |
| **Explicit name** | `name:TEXT` | Search in item names | `name:100` |

**Examples:**

```bash
# List all items in database
python find_all_items.py t_item.json

# Search by name (auto-detect)
python find_all_items.py t_item.json sword
# Output: All items with "sword" in name

# Search by ID (auto-detect)
python find_all_items.py t_item.json 100
# Output: Item with ID 100

# Search for "100" in item names (explicit)
python find_all_items.py t_item.json name:100
# Output: "Sword of 100", "Level 100 Armor", etc.

# Search by exact ID (explicit)
python find_all_items.py t_item.json id:100
# Output: Only item with ID 100
```

**Sample Output:**
```
 100 : Iron Sword
 101 : Steel Sword
 102 : Mithril Sword
 250 : Legendary Blade
```

**Pro Tip:** Use `name:` prefix when searching for numbers in item names to avoid auto-detection treating them as IDs.

---

#### `find_all_shops.py` - Shop Database Browser

Search and filter game shops.

**Usage:**
```bash
python find_all_shops.py t_shop.json [search_text]
```

**Examples:**

```bash
# List all shops
python find_all_shops.py t_shop.json

# Find shops with "weapon" in name
python find_all_shops.py t_shop.json weapon
```

**Sample Output:**
```
  1 : Weapon Shop - Central District
 15 : Advanced Weaponry
 23 : Rare Weapon Dealer
 87 : Blacksmith's Armory
```

---

### ID Extraction Tools

#### `shops_find_unique_item_id_from_kurodlc.py` - DLC ID Analyzer & Template Generator (v2.0)

Extract IDs and generate template configs for shops_create.py.

**Basic Extraction (v1.0 compatible):**
```bash
python shops_find_unique_item_id_from_kurodlc.py <file.kurodlc.json> [mode]
```

**Extraction Modes:**

| Mode | Description | Output |
|------|-------------|--------|
| `all` | All sections (default) | Complete ID list |
| `shop` | ShopItem only | Shop assignments |
| `costume` | CostumeParam only | Costume items |
| `item` | ItemTableData only | Item definitions |
| `dlc` | DLCTableData.items only | DLC pack contents |
| `costume+item` | Combination modes | Custom selection |

---

**NEW in v2.0: Template Generation**

```bash
python shops_find_unique_item_id_from_kurodlc.py <file.kurodlc.json> --generate-template [source]
```

**What it does:**
1. Extracts item IDs from specified sections
2. Extracts shop IDs from ShopItem section (if exists)
3. Extracts template structure from first ShopItem entry
4. Creates `template_<filename>.json` for shops_create.py

**Template Generation Examples:**

```bash
# Generate from all sections
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json --generate-template

# Generate from specific sections
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json --generate-template costume+item

# With custom shop IDs
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json --generate-template --shop-ids=1,5,10,15

# With custom output name
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json --generate-template --output=my_config.json
```

**Generated Template:**
```json
{
    "_comment": [
        "Template config file generated by shops_find_unique_item_id_from_kurodlc.py v2.0",
        "Source file: my_mod.kurodlc.json",
        "...",
        "Extraction summary:",
        "  - CostumeParam: 10 IDs",
        "  - ItemTableData: 10 IDs",
        "  - Total unique item IDs: 10"
    ],
    "item_ids": [5000, 5001, 5002, ...],
    "shop_ids": [1, 5, 10],
    "template": {
        "shop_id": "${shop_id}",
        "item_id": "${item_id}",
        "unknown": 1,
        "start_scena_flags": [],
        "empty1": 0,
        "end_scena_flags": [],
        "int2": 0
    }
}
```

---

### Conflict Resolution

#### `resolve_id_conflicts_in_kurodlc.py` - Main Conflict Resolver (v2.7.1)

The primary tool for detecting and resolving ID conflicts in DLC files.

**Three Main Modes:**

1. **`checkbydlc`** - Detect conflicts only (no changes)
2. **`repair --apply`** - Automatic conflict resolution with smart algorithm
3. **`repair --export/--import`** - Manual conflict resolution

---

#### Mode 1: Check for Conflicts

```bash
python resolve_id_conflicts_in_kurodlc.py checkbydlc
```

**What it does:**
- Scans all `.kurodlc.json` files in current directory
- Compares against game item database
- Reports conflicts without making changes

**Sample Output:**
```
============================================================
MODE: Check all .kurodlc.json files
============================================================

Source used for check: t_item.json

Processing: custom_items_mod.kurodlc.json
------------------------------------------------------------

3596 : Custom Outfit Alpha      [BAD]
3605 : available                [OK]
3606 : available                [OK]
3607 : Custom Outfit Beta       [BAD]
3622 : available                [OK]

Summary for custom_items_mod.kurodlc.json:
  Total IDs: 5
  [OK]  : 3
  [BAD] : 2

============================================================
OVERALL SUMMARY
============================================================
Total files processed: 1
Total unique IDs: 5
[OK]  : 3  (60.0%)
[BAD] : 2  (40.0%)
============================================================
```

---

#### Mode 2: Automatic Repair (NEW v2.7 - Smart Algorithm!)

```bash
python resolve_id_conflicts_in_kurodlc.py repair --apply
```

**What it does:**
1. Detects conflicts
2. Uses **smart algorithm** to find IDs in range **1-5000**
3. Starts search from **middle (2500)** for better distribution
4. Tries to find **continuous blocks** first (faster)
5. Falls back to **scattered search** if needed (handles fragmentation)
6. Creates backup files (`.bak_TIMESTAMP.json`)
7. Generates detailed logs
8. Modifies files in place

**Sample Output (v2.7):**
```
Processing: custom_items_mod.kurodlc.json
------------------------------------------------------------

3596 : Custom Outfit Alpha      [BAD]
3607 : Custom Outfit Beta       [BAD]

============================================================
Searching for 2 available IDs in range [1, 5000]...
============================================================

[SUCCESS] Found 2 available IDs
ID range: 4000-4001
Type: Continuous block
============================================================

Repair plan:
  3596 -> 4000 (Custom Outfit Alpha)
  3607 -> 4001 (Custom Outfit Beta)

Applying changes...

File       : custom_items_mod.kurodlc.json
Backup     : custom_items_mod.kurodlc.json.bak_20260131_154523.json
Verbose log: custom_items_mod.kurodlc.json.repair_verbose_20260131_154523.txt

Changes applied:
  3596 -> 4000 (Custom Outfit Alpha)
  3607 -> 4001 (Custom Outfit Beta)

[SUCCESS] All changes applied successfully with backups.
```

**Key Improvements in v2.7:**
- ✅ IDs guaranteed within 1-5000 (safe range)
- ✅ Better distribution (not clustered at end)
- ✅ Shows ID range and type (continuous/scattered)
- ✅ Clear error if not enough IDs available

**Files Created:**
- `.bak_TIMESTAMP.json` - Backup of original file
- `.repair_verbose_TIMESTAMP.txt` - Detailed change log

---

#### Smart ID Assignment Algorithm (v2.7)

**How It Works:**

**1. Range Constraint (1-5000)**
```
All assigned IDs are guaranteed to be between 1 and 5000
```
- ✅ **Kuro Engine Limit**: This is a hard limit imposed by the Kuro game engine
- ✅ **Cannot be expanded**: The 5000 ID limit is built into the game engine and cannot be modified
- ✅ Safe limit that avoids engine limitations
- ✅ Better compatibility with game systems
- ✅ Professional ID scheme

**Important**: The 1-5000 range is not arbitrary - it's determined by the Kuro engine's internal limitations. While you can technically use IDs above 5000 with manual assignment, this may cause unpredictable behavior or conflicts with the game engine.

**2. Middle-Out Search Strategy**
```
Instead of: 1 → 2 → 3 → ... → 5000 (sequential from start)
Now uses:   2500 → 2501/2499 → 2502/2498 → ... (from middle)
```
- ✅ Better distribution of IDs
- ✅ Creates buffer from game data (typically 1-3500)
- ✅ More predictable spacing

**3. Two Search Strategies**

**Strategy A: Continuous Block (Fast Path)**
```
Need: 50 IDs
Game uses: 1-3500
Algorithm: Searches from 2500
Finds: 3501-3550 (continuous block)
Result: Clean, sequential IDs ✓
```

**Strategy B: Scattered Search (Fallback)**
```
Need: 50 IDs
Game uses: Every 3rd ID (1, 4, 7, 10, 13...)
Algorithm: Finds gaps
Finds: [2, 3, 5, 6, 8, 9, 11, 12...]
Result: Uses available gaps efficiently ✓
```

**4. Clear Error Handling**

If not enough IDs available in range 1-5000:
```
[ERROR] Not enough available IDs in range [1, 5000].
      Requested: 200
      Available: 150
      Used in range: 4850
      Suggestion: Remove some items or use manual assignment

Cannot proceed with repair. Please choose one of these options:
  1. Remove some items from your DLC mod
  2. Use manual ID assignment (--export/--import)
     WARNING: IDs above 5000 exceed Kuro engine limits
  3. Contact for help if you need assistance

NOTE: The 5000 ID limit is a hard constraint of the Kuro game engine
and cannot be expanded or modified.
```

**Algorithm Examples:**

**Example 1: Small Mod (5 conflicts)**
```
Game IDs: 1-3500
Conflicts: 5 IDs need replacement

Algorithm:
  1. Start search from 2500 (middle of 1-5000)
  2. Find continuous block at 3501-3505
  3. Assign IDs: 3501, 3502, 3503, 3504, 3505

Result: Clean block just after game data ✓
```

**Example 2: Medium Mod (50 conflicts)**
```
Game IDs: 1-3800
Conflicts: 50 IDs need replacement

Algorithm:
  1. Start search from 2500
  2. Find continuous block at 4000-4049
  3. Assign IDs: 4000-4049

Result: Better buffer from game data (200 ID gap) ✓
```

**Example 3: Fragmented Space**
```
Game IDs: Every 3rd ID (1, 4, 7, 10, ...)
Conflicts: 50 IDs need replacement

Algorithm:
  1. Start search from 2500
  2. No continuous block found
  3. Use scattered search
  4. Find gaps: [2, 3, 5, 6, 8, 9, ...]

Result: Efficiently uses available gaps ✓
```

**Benefits Over Old Algorithm:**

| Aspect | Old Algorithm | New Algorithm (v2.7) |
|--------|---------------|----------------------|
| Range | Unlimited (could go 10000+) | **Limited to 1-5000** ✓ |
| Start | From highest ID | **From middle (2500)** ✓ |
| Distribution | Clustered at end | **Well distributed** ✓ |
| Gaps | Ignored | **Utilized efficiently** ✓ |
| Errors | Silent failures | **Clear messages** ✓ |
| Speed | O(n) sequential | O(log n) for blocks ✓ |

---

#### Mode 3: Manual Repair (Recommended for Important Mods)

This three-step workflow gives you full control over ID assignments.

**Step 1: Export Conflict Report**

```bash
# With auto-generated timestamp
python resolve_id_conflicts_in_kurodlc.py repair --export

# With custom name (recommended)
python resolve_id_conflicts_in_kurodlc.py repair --export --export-name=my_mod
```

**Creates: `id_mapping_my_mod.json`**

```json
{
  "_comment": [
    "ID Mapping File - Generated by resolve_id_conflicts_in_kurodlc.py",
    "",
    "INSTRUCTIONS:",
    "1. Review each mapping below",
    "2. Edit 'new_id' values as needed (must be unique)",
    "3. Save this file",
    "4. Run: python resolve_id_conflicts_in_kurodlc.py repair --import",
    "",
    "Generated: 2026-01-31 15:45:23"
  ],
  "source": {
    "type": "json",
    "path": "t_item.json"
  },
  "mappings": [
    {
      "old_id": 3596,
      "new_id": 4001,
      "conflict_name": "Custom Outfit Alpha",
      "occurrences": 3,
      "files": ["custom_items_mod.kurodlc.json"]
    },
    {
      "old_id": 3607,
      "new_id": 4002,
      "conflict_name": "Custom Outfit Beta",
      "occurrences": 3,
      "files": ["custom_items_mod.kurodlc.json"]
    }
  ]
}
```

**Step 2: Edit Mapping File**

Open `id_mapping_my_mod.json` and change `new_id` values:

```json
{
  "old_id": 3596,
  "new_id": 5000,  // Changed from 4001 to your preferred ID
  "conflict_name": "Custom Outfit Alpha",
  "occurrences": 3,
  "files": ["custom_items_mod.kurodlc.json"]
}
```

**Important Notes:**
- ✅ **DO** change `new_id` values
- ❌ **DON'T** change `old_id`, `occurrences`, or `files`
- ❌ **DON'T** manually edit `.kurodlc.json` files between export and import

**Step 3: Import and Apply**

```bash
# Interactive selection (if multiple mapping files exist)
python resolve_id_conflicts_in_kurodlc.py repair --import

# Or specify exact file
python resolve_id_conflicts_in_kurodlc.py repair --import --mapping-file=id_mapping_my_mod.json
```

---

#### Command-Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--apply` | Apply changes immediately (automatic mode) | `repair --apply` |
| `--export` | Export repair plan to mapping file | `repair --export` |
| `--export-name=NAME` | Custom export filename<br>(auto-adds prefix/suffix) | `--export-name=my_mod`<br>→ `id_mapping_my_mod.json` |
| `--import` | Import and apply edited mapping | `repair --import` |
| `--mapping-file=PATH` | Specify mapping file to import | `--mapping-file=id_mapping_my_mod.json` |
| `--source=TYPE` | Force source type | `--source=json` |
| `--no-interactive` | Skip all prompts | `checkbydlc --no-interactive` |
| `--keep-extracted` | Keep temporary P3A extractions | `repair --source=p3a --keep-extracted` |

**Source Types:**
- `json` - t_item.json
- `tbl` - t_item.tbl
- `original` - t_item.tbl.original
- `p3a` - script_en.p3a / script_eng.p3a
- `zzz` - zzz_combined_tables.p3a

---

### Shop Management

#### `shops_create.py` - Shop Assignment Generator (v2.0)

Generate shop item assignments from configuration with customizable templates.

**NEW in v2.0:**
- Custom template support
- Variable substitution (`${shop_id}`, `${item_id}`, `${index}`, `${count}`)
- Custom output section names
- Backward compatible with v1.0

**Basic Usage (v1.0 compatible):**
```bash
python shops_create.py config.json
```

**Input: `config.json`**
```json
{
    "item_ids": [5000, 5001, 5002],
    "shop_ids": [1, 5, 10, 15]
}
```

**Output: `output_config.json`**
```json
{
  "ShopItem": [
    {
      "shop_id": 1,
      "item_id": 5000,
      "unknown": 1,
      "start_scena_flags": [],
      "empty1": 0,
      "end_scena_flags": [],
      "int2": 0
    },
    // ... 11 more entries (3 items × 4 shops = 12 total)
  ]
}
```

---

**Advanced Usage (v2.0 - Custom Templates):**

**Input: `custom_config.json`**
```json
{
    "item_ids": [5000, 5001, 5002],
    "shop_ids": [1, 5, 10],
    "output_section": "CustomShopItems",
    "template": {
        "shop_id": "${shop_id}",
        "item_id": "${item_id}",
        "price": 1000,
        "stock": 99,
        "discount": 0,
        "required_level": 1,
        "metadata": {
            "description": "Item ${item_id} in shop ${shop_id}",
            "entry_number": "${index}"
        }
    }
}
```

**Output:**
```json
{
  "CustomShopItems": [
    {
      "shop_id": 1,
      "item_id": 5000,
      "price": 1000,
      "stock": 99,
      "discount": 0,
      "required_level": 1,
      "metadata": {
        "description": "Item 5000 in shop 1",
        "entry_number": 0
      }
    },
    // ... 8 more entries
  ]
}
```

**Console Output:**
```
Using custom template from config file.

Generating shop items...

============================================================
Success: File 'output_custom_config.json' was created successfully.
============================================================
Generated 9 shop item entries:
  - 3 items
  - 3 shops
  - Total combinations: 3 × 3 = 9
  - Output section: 'CustomShopItems'
  - Template: Custom
============================================================
```

**Template Variables:**

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `${shop_id}` | Current shop ID | `1`, `5`, `10` |
| `${item_id}` | Current item ID | `5000`, `5001` |
| `${index}` | Entry index (0-based) | `0`, `1`, `2` |
| `${count}` | Total number of entries | `9` (3×3) |

**See `example_config_*.json` files for more examples.**

---

## 🔄 Common Workflows

### Workflow 1: Creating a New DLC Mod

```bash
# 1. Find available ID range in game data
python find_all_items.py t_item.json | tail -20
# Note the highest ID (e.g., 3500)

# 2. Create your .kurodlc.json file
# Use IDs starting from 5000 to be safe

# 3. Validate structure and check for conflicts
python find_unique_item_id_from_kurodlc.py check

# 4. If conflicts found (unlikely with 5000+), auto-fix
python resolve_id_conflicts_in_kurodlc.py repair --apply

# 5. Generate shop assignments (optional)
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json --generate-template

# 6. Create shop assignments
python shops_create.py template_my_mod.json

# 7. Integrate shop data
# Copy ShopItem section from output_template_my_mod.json
# into your my_mod.kurodlc.json file

# 8. Final validation
python find_unique_item_id_from_kurodlc.py check
```

---

### Workflow 2: Updating Existing DLC

```bash
# 1. Backup current version
cp my_mod.kurodlc.json my_mod.kurodlc.json.backup

# 2. Check for new conflicts
python resolve_id_conflicts_in_kurodlc.py checkbydlc

# 3. If conflicts found, export for manual review
python resolve_id_conflicts_in_kurodlc.py repair --export --export-name=my_mod_v2

# 4. Edit id_mapping_my_mod_v2.json
# Carefully choose new IDs that fit your mod's ID scheme

# 5. Apply changes
python resolve_id_conflicts_in_kurodlc.py repair --import --mapping-file=id_mapping_my_mod_v2.json

# 6. Verify changes
python find_unique_item_id_from_kurodlc.py check

# 7. Test in game
```

---

### Workflow 3: Adding Items to Shops (NEW v2.0)

```bash
# 1. Generate template from your DLC
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json --generate-template costume

# 2. (Optional) Edit template_my_mod.json
# - Review item_ids (auto-extracted)
# - Modify shop_ids as needed
# - Customize template structure

# 3. Generate shop assignments
python shops_create.py template_my_mod.json

# 4. Copy ShopItem section
# From: output_template_my_mod.json
# To: my_mod.kurodlc.json

# 5. Test in game
```

---

### Workflow 4: Batch Processing Multiple DLCs

```bash
# 1. Place all .kurodlc.json files in same directory

# 2. Check all files at once
python resolve_id_conflicts_in_kurodlc.py checkbydlc

# 3. Export repair plan (covers all files)
python resolve_id_conflicts_in_kurodlc.py repair --export --export-name=batch_fix

# 4. Review and edit id_mapping_batch_fix.json

# 5. Apply to all files
python resolve_id_conflicts_in_kurodlc.py repair --import --mapping-file=id_mapping_batch_fix.json

# 6. Verify all files
python find_unique_item_id_from_kurodlc.py searchallbydlc
python find_unique_item_id_from_kurodlc.py check
```

---

## 📄 File Formats

### .kurodlc.json Structure

```json
{
    "CostumeParam": [
        {
            "item_id": 5000,
            "mdl_name": "custom_costume_01",
            "char_restrict": 1,
            "type": 0
        }
    ],
    "ItemTableData": [
        {
            "id": 5000,
            "name": "Custom Costume",
            "category": 17,
            "subcategory": 15
        }
    ],
    "DLCTableData": [
        {
            "id": 1,
            "name": "Custom Costume Pack",
            "items": [5000, 5001, 5002]
        }
    ],
    "ShopItem": [
        {
            "shop_id": 1,
            "item_id": 5000,
            "unknown": 1,
            "start_scena_flags": [],
            "empty1": 0,
            "end_scena_flags": [],
            "int2": 0
        }
    ]
}
```

**Required Sections:**
- `CostumeParam` - Costume definitions
- `DLCTableData` - DLC pack definitions

**Optional Sections:**
- `ItemTableData` - Item metadata
- `ShopItem` - Shop assignments

---

### Mapping File Structure

```json
{
  "_comment": ["Instructions..."],
  "source": {
    "type": "json",
    "path": "t_item.json"
  },
  "mappings": [
    {
      "old_id": 3596,
      "new_id": 5000,
      "conflict_name": "Custom Outfit Alpha",
      "occurrences": 3,
      "files": ["custom_mod.kurodlc.json"]
    }
  ]
}
```

**Editable Fields:**
- ✅ `new_id` - New ID to assign

**Do NOT Edit:**
- ❌ `old_id` - Original conflicting ID
- ❌ `occurrences` - Number of times ID appears
- ❌ `files` - Files containing this ID

---

### Shop Config File Format (v2.0)

```json
{
    "item_ids": [5000, 5001, 5002],
    "shop_ids": [1, 5, 10],
    "output_section": "ShopItem",
    "template": {
        "shop_id": "${shop_id}",
        "item_id": "${item_id}",
        "unknown": 1,
        "start_scena_flags": [],
        "empty1": 0,
        "end_scena_flags": [],
        "int2": 0
    }
}
```

**Fields:**
- `item_ids` - List of item IDs to add to shops
- `shop_ids` - List of shop IDs where items appear
- `output_section` - Output section name (default: "ShopItem")
- `template` - Template structure with variable substitution

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue: "No valid item source found"

**Error:**
```
Error: No valid item source found.
```

**Cause:** Missing game data files.

**Solution:**
Ensure you have at least one of these files in the current directory:
- `t_item.json`
- `t_item.tbl`
- `t_item.tbl.original`
- `script_en.p3a` or `script_eng.p3a`
- `zzz_combined_tables.p3a`

---

#### Issue: "Invalid JSON" errors

**Error:**
```
[ERROR] Invalid JSON in custom_mod.kurodlc.json:
        Expecting value: line 45 column 5 (char 1234)
```

**Cause:** Malformed JSON syntax.

**Solution:**
1. Use a JSON validator: [jsonlint.com](https://jsonlint.com)
2. Check for:
   - Missing commas between array elements
   - Missing closing brackets/braces
   - Trailing commas (not allowed in JSON)
   - Unescaped quotes in strings

---

#### Issue: "Not enough IDs available" (NEW in v2.7)

**Error:**
```
[ERROR] Not enough available IDs in range [1, 5000].
      Requested: 200
      Available: 150
      Used in range: 4850
```

**Cause:** Your mod has more conflicts than available IDs in the 1-5000 range.

**Why this happens:**
The Kuro game engine has a **hard-coded limit of 5000 item IDs**. This is a fundamental engine limitation that cannot be expanded or modified. If:
- Game uses IDs 1-3500
- Other DLCs use 3501-4850
- You need 200 new IDs
- Only 150 IDs available (4851-5000)
- **The 5000 limit cannot be increased** - it's built into the engine

**Solutions:**

**Option 1: Remove some items (Recommended)**
```bash
# Edit your .kurodlc.json and remove least important items
# This is the safest approach
python resolve_id_conflicts_in_kurodlc.py repair --apply
```

**Option 2: Use manual assignment with IDs > 5000 (⚠️ NOT RECOMMENDED)**
```bash
# WARNING: IDs above 5000 exceed Kuro engine design limits
# This may cause crashes, errors, or save corruption
# Only use if you understand and accept the risks

# Export and manually assign IDs in higher range
python resolve_id_conflicts_in_kurodlc.py repair --export --export-name=custom
# Edit id_mapping_custom.json with higher IDs (e.g., 6000-6199)
python resolve_id_conflicts_in_kurodlc.py repair --import --mapping-file=id_mapping_custom.json

# ⚠️ TEST THOROUGHLY before releasing to users
```

**Option 3: Split into multiple DLC files**
```bash
# Divide items across multiple .kurodlc.json files
# Each file gets separate ID allocation within the 5000 limit
my_mod_part1.kurodlc.json  # Gets IDs 3800-3899
my_mod_part2.kurodlc.json  # Gets IDs 3900-3999
```

**Prevention:**
- Plan your ID usage before creating large mods
- Use ID ranges strategically (e.g., 3800-3999 for costumes)
- Check available space first:
  ```bash
  python find_unique_item_id_from_kurodlc.py check
  ```
- Remember: **5000 is a hard limit** - design your mods accordingly

---

#### Issue: "Number of occurrences changed"

**Error:**
```
ID 3596: Number of occurrences changed!
    Expected: 3 occurrence(s)
    Found: 2 occurrence(s)
```

**Cause:** .kurodlc.json file was manually edited between export and import.

**Solution:**

**Option 1:** Restore from backup
```bash
cp custom_mod.kurodlc.json.bak_TIMESTAMP.json custom_mod.kurodlc.json
```

**Option 2:** Create new export with current state
```bash
python resolve_id_conflicts_in_kurodlc.py repair --export --export-name=fresh
```

**Prevention:** Don't manually edit `.kurodlc.json` files between export and import!

---

## 🎯 Best Practices

### ID Management (Updated for v2.7)

✅ **DO:**
- **Let the algorithm choose IDs** - v2.7's smart algorithm handles distribution
- Use ID ranges appropriate for your mod size:
  - Small mods (< 50 items): Let automatic repair handle it
  - Medium mods (50-200 items): Review suggested IDs via export first
  - Large mods (200+ items): Use manual assignment with custom ranges
- **Start high when manually assigning** - Use 4000+ to avoid game data
- Document your ID schemes (e.g., 4000-4099 = Costumes, 4100-4199 = Weapons)
- Use sequential IDs for related items
- Export mapping files before making changes
- Keep mapping files in version control

❌ **DON'T:**
- Manually specify IDs close to game data (e.g., 3500-3600 if game uses 1-3500)
- Mix ID ranges for different item types without documentation
- Manually edit .kurodlc.json between export/import
- Delete backup files immediately
- Reuse IDs across different mods without checking conflicts
- Ignore the 1-5000 range limit warnings

**NEW in v2.7: Understanding the Range Limit**

The automatic algorithm uses 1-5000 range because:
- ✅ **Kuro Engine Hard Limit**: The game engine has a built-in limit of 5000 IDs
- ✅ **Cannot be expanded**: This limitation is part of the Kuro engine architecture and cannot be modified or extended
- ✅ Provides ~1500-2000 IDs for mods (assuming game uses 1-3500)
- ✅ Safe buffer from game data
- ✅ Professional ID scheme

**⚠️ Warning about IDs > 5000:**
While the toolkit allows manual assignment of IDs above 5000 (via export/import), this exceeds the Kuro engine's design limits and may cause:
- Unpredictable game behavior
- Potential crashes or errors
- Conflicts with engine internals
- Save file corruption

**Only use IDs > 5000 if:**
- You absolutely cannot fit within the limit
- You understand and accept the risks
- You've tested thoroughly

If you need IDs > 5000 (not recommended), use manual assignment:
```bash
python resolve_id_conflicts_in_kurodlc.py repair --export
# Edit mapping to use higher IDs (6000+) - AT YOUR OWN RISK
python resolve_id_conflicts_in_kurodlc.py repair --import
```

---

### Workflow Recommendations

**For Quick Testing:**
```bash
python resolve_id_conflicts_in_kurodlc.py repair --apply
```

**For Production Mods:**
```bash
# Always use export/import workflow
python resolve_id_conflicts_in_kurodlc.py repair --export --export-name=production_v1
# Edit mapping carefully
python resolve_id_conflicts_in_kurodlc.py repair --import
```

**For Team Projects:**
```bash
# Share mapping files, not .kurodlc.json
git add id_mapping_*.json
git commit -m "Add ID mapping for new feature"
```

---

### File Organization

**Recommended structure:**
```
my_mod/
├── src/
│   └── my_mod.kurodlc.json          # Source file
├── mappings/
│   ├── id_mapping_v1.json           # Version 1 mapping
│   ├── id_mapping_v2.json           # Version 2 mapping
│   └── id_mapping_current.json      # Current mapping
├── backups/
│   └── *.bak_*.json                 # Auto-generated backups
├── logs/
│   └── *.repair_verbose_*.txt       # Repair logs
├── templates/
│   └── template_my_mod.json         # Shop config template
└── config/
    └── shop_config.json             # Shop assignments
```

---

### Version Control

**Recommended `.gitignore`:**
```gitignore
# Backups
*.bak_*.json

# Logs
*.repair_verbose_*.txt

# Temporary files
*.tmp
t_item.tbl.original.tmp

# Output files
output_*.json
```

**Track these files:**
```gitignore
# Source files
*.kurodlc.json

# Mapping files (important!)
id_mapping_*.json

# Template files
template_*.json

# Config files
*_config.json

# Schema
kurodlc_schema.json
```
---

## 🙏 Acknowledgments

- Game modding community
- Contributors and testers
- Library developers (p3a_lib, kurodlc_lib)
- [https://github.com/eArmada8/kuro_dlc_tool]
  
---

<p align="center">
  <strong>Made with ❤️ by the modding community</strong>
</p>

<p align="center">
  <a href="#kurodlc-modding-toolkit">Back to Top ↑</a>
</p>
