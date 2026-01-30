# KuroDLC Modding Toolkit

A comprehensive Python toolkit for creating and managing DLC mods for games using the KuroDLC format. This toolkit provides utilities for item discovery, ID management, conflict resolution, and shop assignment automation.

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 📋 Table of Contents

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
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- **🔍 Item Discovery**: Search and browse game items from JSON, TBL, and P3A sources
- **⚠️ Conflict Detection**: Automatically detect ID conflicts between DLC and game data
- **🔧 Smart Resolution**: Automatic or manual ID conflict resolution with validation
- **🛒 Shop Integration**: Generate shop assignments for custom items in bulk
- **📦 Multiple Formats**: Support for JSON, TBL, and P3A archive formats
- **✅ Validation**: Comprehensive .kurodlc.json structure validation
- **💾 Safety First**: Automatic backups and detailed logging for all modifications
- **🎨 User-Friendly**: Interactive menus and colored output (Windows CMD compatible)

---

## 📦 Requirements

### Core Requirements
- **Python 3.7+**
- Standard library modules (included): `json`, `sys`, `os`, `pathlib`, `shutil`, `datetime`, `glob`

### Optional Dependencies

For enhanced functionality:

```bash
# For colored terminal output (recommended for Windows)
pip install colorama

# For P3A archive support (optional)
# Requires custom libraries: p3a_lib, kurodlc_lib
# Contact library maintainers or check documentation
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

### Automatic Conflict Resolution

```bash
# Detect and fix conflicts automatically
python resolve_id_conflicts_in_kurodlc.py repair --apply
```

### Manual Conflict Resolution (Recommended)

```bash
# Step 1: Export conflict report
python resolve_id_conflicts_in_kurodlc.py repair --export --export-name=my_mod

# Step 2: Edit id_mapping_my_mod.json (change new_id values as needed)

# Step 3: Apply your custom mappings
python resolve_id_conflicts_in_kurodlc.py repair --import --mapping-file=id_mapping_my_mod.json
```

---

## 🛠️ Scripts Overview

| Script | Purpose | Key Features |
|--------|---------|--------------|
| **`resolve_id_conflicts_in_kurodlc.py`** | **Main conflict resolver** | Auto/manual repair, validation, backups |
| `find_all_items.py` | Search game items | ID/name search, auto-detect mode |
| `find_all_shops.py` | Search game shops | Filter by shop name |
| `find_unique_item_id_for_t_costumes.py` | Extract costume IDs | From CostumeParam section |
| `find_unique_item_id_for_t_item_category.py` | Extract category IDs | Filter by item category |
| `find_unique_item_id_from_kurodlc.py` | Extract DLC IDs | Multiple modes, conflict checking |
| `shops_create.py` | Generate shop assignments | Batch item-shop combinations |
| `shops_find_unique_item_id_from_kurodlc.py` | Extract shop item IDs | Section-specific extraction |

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

#### `find_unique_item_id_for_t_costumes.py` - Costume ID Extractor

Extract all unique item IDs from costume data.

**Usage:**
```bash
python find_unique_item_id_for_t_costumes.py t_costume.json
```

**Sample Output:**
```
[100, 101, 102, 150, 151, 200, 201, 202, 250]
```

**Use Case:** Quickly identify costume-related items for mod integration.

---

#### `find_unique_item_id_for_t_item_category.py` - Category ID Extractor

Extract item IDs filtered by category number.

**Usage:**
```bash
python find_unique_item_id_for_t_item_category.py t_item.json <category_number>
```

**Examples:**

```bash
# Get all items from category 5 (typically accessories)
python find_unique_item_id_for_t_item_category.py t_item.json 5

# Get all items from category 1 (typically weapons)
python find_unique_item_id_for_t_item_category.py t_item.json 1
```

**Sample Output:**
```
[500, 501, 502, 550, 551, 600, 601]
```

---

### ID Extraction Tools

#### `find_unique_item_id_from_kurodlc.py` - DLC ID Analyzer

Extract and validate item IDs from DLC files with conflict checking.

**Usage:**
```bash
python find_unique_item_id_from_kurodlc.py <mode> [options]
```

**Modes:**

| Mode | Description | Output |
|------|-------------|--------|
| `<file.kurodlc.json>` | Process single file | List of IDs |
| `searchall` | All unique IDs from all files | Single sorted list |
| `searchallbydlc` | IDs grouped by file | Per-file lists + unique total |
| `searchallbydlcline` | IDs per file, one per line | Line-separated output |
| `searchallline` | All unique IDs, one per line | Simple list |
| **`check`** | **Check for conflicts** | **Conflict report with [OK]/[BAD]** |

---

**Check Mode (Most Important):**

```bash
# Interactive source selection
python find_unique_item_id_from_kurodlc.py check

# Force specific source
python find_unique_item_id_from_kurodlc.py check --source=json

# Non-interactive mode (for scripts)
python find_unique_item_id_from_kurodlc.py check --no-interactive

# Use P3A and keep extracted file
python find_unique_item_id_from_kurodlc.py check --source=p3a --keep-extracted
```

**Check Mode Output:**
```
3596 : Custom Outfit Alpha      [BAD]
3605 : available                [OK]
3606 : available                [OK]
3607 : Custom Outfit Beta       [BAD]
3608 : available                [OK]

Summary:
Total IDs : 5
OK        : 3
BAD       : 2

Source used for check: t_item.json
```

**Legend:**
- `[OK]` = ID available (safe to use)
- `[BAD]` = ID conflicts with game data (needs fixing)

**Options:**

| Option | Description |
|--------|-------------|
| `--source=TYPE` | Force source type:<br>`json`, `tbl`, `original`, `p3a`, `zzz` |
| `--no-interactive` | Skip prompts, auto-select first source |
| `--keep-extracted` | Keep temporary P3A extraction file |

---

### Conflict Resolution

#### `resolve_id_conflicts_in_kurodlc.py` - Main Conflict Resolver

The primary tool for detecting and resolving ID conflicts in DLC files.

**Three Main Modes:**

1. **`checkbydlc`** - Detect conflicts only (no changes)
2. **`repair --apply`** - Automatic conflict resolution
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

#### Mode 2: Automatic Repair

```bash
python resolve_id_conflicts_in_kurodlc.py repair --apply
```

**What it does:**
1. Detects conflicts
2. Automatically assigns new non-conflicting IDs
3. Creates backup files (`.bak_TIMESTAMP.json`)
4. Generates detailed logs
5. Modifies files in place

**Sample Output:**
```
Processing: custom_items_mod.kurodlc.json
------------------------------------------------------------

Conflicts detected: 2

3596 : Custom Outfit Alpha
  Suggested new ID: 4001

3607 : Custom Outfit Beta
  Suggested new ID: 4002

Applying changes...

File       : custom_items_mod.kurodlc.json
Backup     : custom_items_mod.kurodlc.json.bak_20260131_154523.json
Verbose log: custom_items_mod.kurodlc.json.repair_verbose_20260131_154523.txt

Changes applied:
  3596 -> 4001 (Custom Outfit Alpha)
  3607 -> 4002 (Custom Outfit Beta)

[SUCCESS] All changes applied successfully with backups.
```

**Files Created:**
- `.bak_TIMESTAMP.json` - Backup of original file
- `.repair_verbose_TIMESTAMP.txt` - Detailed change log

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
    "IMPORTANT:",
    "- Do NOT change 'old_id' values!",
    "- Do NOT change 'occurrences' values!",
    "- Do NOT change 'files' list!",
    "- You CAN change 'new_id' values",
    "- Do NOT manually edit .kurodlc.json files between export and import!",
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

Open `id_mapping_my_mod.json` in your text editor and change `new_id` values:

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

**Validation Output:**
```
============================================================
MODE: Import ID mapping from file
============================================================

Step 1: Loading mapping file...
Found single mapping file: id_mapping_my_mod.json

Step 2: Loading item source from mapping file...
Source type: json
Source path: t_item.json

Step 3: Validating ID mappings...
============================================================

[OK] 3596 -> 5000 ('Custom Outfit Alpha')
[OK] 3607 -> 5001 ('Custom Outfit Beta')

============================================================

[SUCCESS] Validation PASSED!
All 2 ID mappings verified successfully.
Ready to modify 1 file(s).
============================================================

Step 4: Applying imported ID mappings...

File       : custom_items_mod.kurodlc.json
Backup     : custom_items_mod.kurodlc.json.bak_20260131_160123.json
Verbose log: custom_items_mod.kurodlc.json.repair_verbose_20260131_160123.txt
------------------------------------------------------------

[SUCCESS] All changes from imported mapping applied successfully with backups.
```

---

#### Advanced Validation Features

The import process performs comprehensive validation:

**1. Structure Validation**
- Checks JSON format
- Verifies required fields exist
- Validates data types

**2. ID Existence Check**
- Confirms old IDs still exist in files
- Detects if files were manually edited

**3. Occurrence Validation** ⭐ **New!**
- Verifies ID appears same number of times as during export
- Prevents partial modifications

**4. Conflict Check**
- Ensures new IDs don't conflict with game data
- Checks against existing DLC IDs

**5. Duplicate Prevention**
- Prevents assigning same new ID multiple times

**Sample Validation Error:**

```
============================================================
[ERROR] VALIDATION FAILED - Found 1 issue(s)
============================================================

Cannot proceed with import due to inconsistencies between
mapping file and current state of .kurodlc.json files.

Details:
------------------------------------------------------------

Issue #1:
  ID 3596: Number of occurrences changed!
      File: custom_items_mod.kurodlc.json
      Expected: 3 occurrence(s)
      Found: 2 occurrence(s)
      Current sections: ItemTableData, DLCTableData
      
      Possible cause: Manual changes to file - ID removed from some sections.
      Solution: Either restore ALL occurrences or create a new export.

============================================================
POSSIBLE CAUSES:
  1. Files were manually edited between export and import
  2. IDs were changed or removed in .kurodlc.json files
  3. Wrong mapping file selected

RECOMMENDED SOLUTIONS:
  1. Restore original .kurodlc.json files from backup
  2. Create a new export with current file state:
     python resolve_id_conflicts_in_kurodlc.py repair --export
  3. Manually fix the IDs mentioned above
============================================================
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

#### `shops_create.py` - Shop Assignment Generator

Generate shop item assignments from configuration file.

**Usage:**
```bash
python shops_create.py config.json
```

**Input: `config.json`**
```json
{
  "item_ids": [100, 101, 102],
  "shop_ids": [1, 5, 10, 15]
}
```

**What it does:**
- Creates all combinations of items × shops
- Generates properly formatted ShopItem entries

**Output: `output_config.json`**
```json
{
  "ShopItem": [
    {
      "shop_id": 1,
      "item_id": 100,
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

**Console Output:**
```
Success: File 'output_config.json' was created successfully.
Generated 12 shop item entries:
  - 3 items
  - 4 shops
  - Total combinations: 3 × 4 = 12
```

**Next Steps:**
1. Review `output_config.json`
2. Copy `ShopItem` section into your `.kurodlc.json` file
3. Adjust shop IDs/flags if needed

**Example Config Template:**
```json
{
  "item_ids": [5000, 5001, 5002, 5003],
  "shop_ids": [1, 5, 10]
}
```

---

#### `shops_find_unique_item_id_from_kurodlc.py` - Shop ID Extractor

Extract item IDs from specific sections of DLC files.

**Usage:**
```bash
python shops_find_unique_item_id_from_kurodlc.py <file.kurodlc.json> [mode]
```

**Modes:**

| Mode | Sections Extracted | Use Case |
|------|-------------------|----------|
| `all` | All sections (default) | Complete ID inventory |
| `shop` | ShopItem only | Shop assignments |
| `costume` | CostumeParam only | Costume items |
| `item` | ItemTableData only | Item definitions |
| `dlc` | DLCTableData.items only | DLC pack contents |

**Combination Modes (use `+`):**
- `shop+costume` - Shop items + costumes
- `costume+item` - Costumes + item data
- `item+dlc` - Item data + DLC packs
- `shop+item+dlc` - Custom combinations

**Examples:**

```bash
# Extract from all sections
python shops_find_unique_item_id_from_kurodlc.py custom_mod.kurodlc.json

# Extract only shop assignments
python shops_find_unique_item_id_from_kurodlc.py custom_mod.kurodlc.json shop

# Extract costumes and items
python shops_find_unique_item_id_from_kurodlc.py custom_mod.kurodlc.json costume+item

# Extract shop and DLC data
python shops_find_unique_item_id_from_kurodlc.py custom_mod.kurodlc.json shop+dlc
```

**Sample Output:**
```
# Extraction summary:
#   ShopItem: 12 IDs
#   CostumeParam: 8 IDs
#   ItemTableData: 8 IDs
#   DLCTableData.items: 8 IDs
#   Total unique IDs: 15

[100, 101, 102, 103, 104, 105, 106, 107, 200, 201, 202, 203, 204, 205, 206]
```

**Pro Tip:** Use this to verify your shop assignments are correct before testing.

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
# Create shop_config.json with your IDs
python shops_create.py shop_config.json

# 6. Integrate shop data
# Copy ShopItem section from output_shop_config.json
# into your .kurodlc.json file

# 7. Final validation
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

### Workflow 3: Batch Processing Multiple DLCs

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

### Workflow 4: Migrating from Another Mod Format

```bash
# 1. Extract IDs from your existing mod files
python find_unique_item_id_from_kurodlc.py my_old_mod.kurodlc.json > old_ids.txt

# 2. Check which IDs conflict
python find_unique_item_id_from_kurodlc.py check

# 3. Use manual workflow for control
python resolve_id_conflicts_in_kurodlc.py repair --export --export-name=migration

# 4. Plan new ID scheme
# Edit id_mapping_migration.json
# Use sequential IDs like 6000-6999

# 5. Apply migration
python resolve_id_conflicts_in_kurodlc.py repair --import --mapping-file=id_mapping_migration.json

# 6. Update documentation with new ID ranges
```

---

## 📄 File Formats

### Mapping File Structure

The mapping file created during export:

```json
{
  "_comment": [
    "Instructions for editing..."
  ],
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

**Field Descriptions:**

| Field | Type | Editable? | Description |
|-------|------|-----------|-------------|
| `old_id` | Integer | ❌ No | Conflicting ID from your mod |
| `new_id` | Integer | ✅ **Yes** | **New ID to assign** |
| `conflict_name` | String | ❌ No | Item name from game data |
| `occurrences` | Integer | ❌ No | Number of times ID appears |
| `files` | Array | ❌ No | Files containing this ID |

**Validation Rules for `new_id`:**
- Must be an integer
- Must not conflict with game data
- Must not duplicate other new IDs in mapping
- Must be unique across all your mods

---

### Shop Config File Format

Simple configuration for bulk shop assignment:

```json
{
  "item_ids": [5000, 5001, 5002],
  "shop_ids": [1, 5, 10]
}
```

**Field Descriptions:**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `item_ids` | Array of integers | Item IDs to add to shops | `[5000, 5001]` |
| `shop_ids` | Array of integers | Shop IDs where items appear | `[1, 5, 10]` |

**Result:** Every item appears in every shop (Cartesian product)

**Example:** 
- 3 items × 4 shops = 12 shop entries generated

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

**Common mistakes:**
```json
// WRONG - trailing comma
{"items": [100, 101,]}

// CORRECT
{"items": [100, 101]}

// WRONG - missing comma
{"id": 100
 "name": "Item"}

// CORRECT
{"id": 100,
 "name": "Item"}
```

---

#### Issue: "Number of occurrences changed"

**Error:**
```
ID 3596: Number of occurrences changed!
    Expected: 3 occurrence(s)
    Found: 2 occurrence(s)
```

**Cause:** .kurodlc.json file was manually edited between export and import.

**Why this happens:**
IDs typically appear in multiple sections (CostumeParam, ItemTableData, DLCTableData). If you manually edited one section but not others, the count changes.

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

#### Issue: P3A extraction fails

**Error:**
```
Error: Required library missing: No module named 'p3a_lib'
```

**Cause:** Missing optional P3A libraries.

**Solution:**

**Option 1:** Use JSON source instead
```bash
python resolve_id_conflicts_in_kurodlc.py repair --source=json
```

**Option 2:** Install P3A libraries (if available)
```bash
# Contact library maintainers for installation instructions
pip install p3a_lib kurodlc_lib
```

---

#### Issue: Conflicts still appear after repair

**Symptom:** IDs still show `[BAD]` after running repair.

**Possible Causes:**
1. Repair was not applied (only exported)
2. Wrong source file
3. Multiple DLC files with same IDs

**Debug Steps:**

```bash
# 1. Check if backup exists (proves repair was applied)
ls -la *.bak_*.json

# 2. Verify source file used
python find_unique_item_id_from_kurodlc.py check
# Check "Source used for check" line

# 3. Check all DLC files
python find_unique_item_id_from_kurodlc.py searchallbydlc

# 4. Re-run check after repair
python resolve_id_conflicts_in_kurodlc.py repair --apply
python find_unique_item_id_from_kurodlc.py check
```

---

### Debug Information

For detailed troubleshooting, examine these files:

**1. Backup Files**
```
custom_mod.kurodlc.json.bak_20260131_154523.json
```
- Original file before changes
- Use to restore if something goes wrong

**2. Verbose Logs**
```
custom_mod.kurodlc.json.repair_verbose_20260131_154523.txt
```
- Detailed change log
- Lists every ID change
- Shows which sections were modified

**Sample verbose log:**
```
=== REPAIR LOG ===
File: custom_mod.kurodlc.json
Timestamp: 2026-01-31 15:45:23

OLD_ID -> NEW_ID : NAME
3596   -> 5000   : Custom Outfit Alpha

Sections modified:
  - CostumeParam[0]: item_id changed from 3596 to 5000
  - ItemTableData[0]: id changed from 3596 to 5000
  - DLCTableData[0]: items array updated (3596 -> 5000)
```

**3. Mapping Files**
```
id_mapping_my_mod.json
```
- Shows intended changes
- Compare with actual results
- Verify `new_id` values were correctly applied

---

### Getting Help

If issues persist:

1. **Check file structure:**
   ```bash
   python -m json.tool custom_mod.kurodlc.json > /dev/null
   ```

2. **Validate against schema:**
   ```bash
   pip install jsonschema
   python -m jsonschema -i custom_mod.kurodlc.json kurodlc_schema.json
   ```

3. **Generate debug report:**
   ```bash
   python find_unique_item_id_from_kurodlc.py searchallbydlc > debug_report.txt
   python find_unique_item_id_from_kurodlc.py check >> debug_report.txt
   ```

4. **Create issue on GitHub** with:
   - Error messages (full text)
   - Debug report
   - Sample mapping file (anonymize IDs if needed)
   - Python version: `python --version`

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

# Config files
*_config.json

# Schema
kurodlc_schema.json
```

---

<p align="center">
  Made with ❤️ by the modding community
</p>
