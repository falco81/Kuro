# Advanced Documentation - KuroDLC Modding Toolkit

This section provides comprehensive, in-depth documentation for advanced users, including all parameters, real data examples, data structure specifications, and detailed workflows.

## 📋 Table of Contents

- [Script Reference](#script-reference)
  - [resolve_id_conflicts_in_kurodlc.py](#resolve_id_conflicts_in_kurodlcpy)
  - [shops_find_unique_item_id_from_kurodlc.py](#shops_find_unique_item_id_from_kurodlcpy)
  - [shops_create.py](#shops_createpy)
  - [find_all_items.py](#find_all_itemspy)
  - [find_all_shops.py](#find_all_shopspy)
  - [Other Utility Scripts](#other-utility-scripts)
- [Data Structure Specifications](#data-structure-specifications)
- [Real Data Examples](#real-data-examples)
- [Export/Import Formats](#exportimport-formats)
- [Log Files](#log-files)
- [Advanced Workflows](#advanced-workflows)

---

## Script Reference

### resolve_id_conflicts_in_kurodlc.py

**Version:** v2.7.1  
**Purpose:** Detect and resolve ID conflicts between DLC mods and game data

#### All Parameters

```
resolve_id_conflicts_in_kurodlc.py <mode> [options]

MODES:
  checkbydlc          Check all .kurodlc.json files for conflicts (read-only)
                      Shows [OK] for available IDs, [BAD] for conflicts
                      
  repair              Interactive repair mode with conflict resolution
                      Same as checkbydlc but generates repair plan for [BAD] IDs
                      Uses smart algorithm to find IDs in range 1-5000
  
OPTIONS:
  --apply             Apply changes to DLC files immediately (automatic repair)
                      Creates backups and detailed logs
                      
  --export            Export repair plan to id_mapping_TIMESTAMP.json
                      Allows manual editing before applying changes
                      
  --export-name=NAME  Custom name for exported mapping file
                      Examples:
                        --export-name=DLC1  → creates id_mapping_DLC1.json
                        --export-name=test  → creates id_mapping_test.json
                      
  --import            Import edited id_mapping.json and apply changes
                      Shows interactive selection if multiple files exist
                      
  --mapping-file=FILE Specify which mapping file to import (full filename)
                      Example: --mapping-file=id_mapping_DLC1.json
                      Skips interactive menu
                      
  --source=TYPE       Force specific source type
                      Available: json, tbl, original, p3a, zzz
                      
  --keep-extracted    Keep temporary extracted files (for debugging)
  
  --no-interactive    Auto-select first source if multiple found
                      Auto-select newest mapping file when using --import
  
SOURCES (automatically detected):

Game Database Sources (for conflict detection):
  JSON sources:
    - t_item.json
    
  TBL sources (requires kurodlc_lib.py):
    - t_item.tbl
    - t_item.tbl.original
    
  P3A sources (requires kurodlc_lib.py + dependencies):
    - script_en.p3a / script_eng.p3a
    - zzz_combined_tables.p3a
    (automatically extracts t_item.tbl.original.tmp)

DLC Files to Check:
  - All .kurodlc.json files in current directory

ALGORITHM (v2.7):
  Smart ID assignment in range 1-5000:
  1. Starts from middle (2500) for better distribution
  2. Tries continuous blocks first (e.g., 4000-4049)
  3. Falls back to scattered search if needed
  4. Clear error if not enough IDs available
```

#### Examples with Real Data

**Example 1: Check for Conflicts**

```bash
# Check DLC against game database
python resolve_id_conflicts_in_kurodlc.py checkbydlc
```

**Sample Output:**
```
Checking: my_costume_mod.kurodlc.json
Loading game data from: t_item.json

Game database contains 2116 items (ID range: 1-4921)

Analyzing DLC file...

[OK]  3596 - Available
[OK]  3597 - Available
[BAD] 310  - Conflict! Used by: Earth Sepith
[BAD] 311  - Conflict! Used by: Water Sepith
[OK]  5000 - Available

Summary:
  Total IDs in DLC: 5
  Conflicts found: 2
  Safe IDs: 3
```

**Example 2: Repair Mode (Preview)**

```bash
# Preview what would be changed
python resolve_id_conflicts_in_kurodlc.py repair
```

**Sample Output:**
```
=== ID CONFLICT REPAIR ===

Source: t_item.json (2116 items, range 1-4921)
DLC file: my_costume_mod.kurodlc.json

Conflicts detected:
  310 (Earth Sepith) → Will reassign
  311 (Water Sepith) → Will reassign

Smart algorithm will assign from range 1-5000
Starting point: 2500 (middle-out distribution)

Proposed mapping:
  310 → 2500
  311 → 2501

Press Enter to continue, or Ctrl+C to abort...
```

**Example 3: Apply Fixes Automatically**

```bash
# Apply changes without prompts
python resolve_id_conflicts_in_kurodlc.py repair --apply
```

**Sample Output:**
```
=== APPLYING FIXES ===

Backup created: my_costume_mod.kurodlc.json.bak_20260131_143022

Updating IDs:
  ✓ 310 → 2500 (CostumeParam)
  ✓ 310 → 2500 (ItemTableData)
  ✓ 310 → 2500 (DLCTableData.items)
  ✓ 311 → 2501 (CostumeParam)
  ✓ 311 → 2501 (ItemTableData)
  ✓ 311 → 2501 (DLCTableData.items)

Changes applied successfully!
Modified file: my_costume_mod.kurodlc.json

Log saved to: id_conflict_repair_20260131_143022.log
```

**Example 4: Export ID Mapping**

```bash
# Export mapping for manual editing
python resolve_id_conflicts_in_kurodlc.py repair --export
```

**Generated File:** `id_mapping_20260131_143022.json`
```json
{
  "source_file": "my_costume_mod.kurodlc.json",
  "timestamp": "2026-01-31 14:30:22",
  "game_database": "t_item.json",
  "game_id_count": 2116,
  "game_id_range": [1, 4921],
  "mappings": {
    "310": 2500,
    "311": 2501
  },
  "conflicts": [
    {
      "old_id": 310,
      "new_id": 2500,
      "reason": "Conflict with game item: Earth Sepith"
    },
    {
      "old_id": 311,
      "new_id": 2501,
      "reason": "Conflict with game item: Water Sepith"
    }
  ]
}
```

**Example 5: Import Custom Mapping**

```bash
# Edit id_mapping.json manually, then import
python resolve_id_conflicts_in_kurodlc.py repair --import id_mapping_20260131_143022.json --apply
```

**Sample Manual Edit:**
```json
{
  "mappings": {
    "310": 3000,
    "311": 3001
  }
}
```

**Output:**
```
Importing ID mapping from: id_mapping_20260131_143022.json

Custom mappings loaded:
  310 → 3000 (manually specified)
  311 → 3001 (manually specified)

Applying changes...
  ✓ All mappings applied successfully
```

#### Real Data Scenarios

**Scenario 1: Sepith ID Conflicts**

Your DLC uses IDs 310-317 for custom items, but these are used by the game for Sepith:

```
Game Database (t_item.json):
  310: Earth Sepith
  311: Water Sepith
  312: Fire Sepith
  313: Wind Sepith
  314: Time Sepith
  315: Space Sepith
  316: Mirage Sepith
  317: Sepith Mass
```

**Solution:**
```bash
python resolve_id_conflicts_in_kurodlc.py repair --apply
```

Result: Your IDs reassigned to 2500-2507 (safe range)

**Scenario 2: Large DLC with Multiple Conflicts**

```
Your DLC: 50 costume items (IDs 100-149)
Conflicts: IDs 100-120 overlap with game items
Safe range: IDs 121-149 are OK

Smart algorithm will:
- Reassign 100-120 → 2500-2520
- Keep 121-149 unchanged (no conflicts)
```

---

### shops_find_unique_item_id_from_kurodlc.py

**Version:** v2.1  
**Purpose:** Extract item IDs from DLC files and generate template configurations

#### All Parameters

```
shops_find_unique_item_id_from_kurodlc.py <file> [mode] [options]

ARGUMENTS:
  <file>              .kurodlc.json file to process (required)
  [mode]              Extraction mode (default: all)
  
EXTRACTION MODES:
  all                 Extract from all sections (default)
  shop                Extract from ShopItem section only
  costume             Extract from CostumeParam section only
  item                Extract from ItemTableData section only
  dlc                 Extract from DLCTableData.items section only
  
  Combinations:
    costume+item      Extract from CostumeParam and ItemTableData
    shop+costume      Extract from ShopItem and CostumeParam
    
TEMPLATE GENERATION:
  --generate-template [source]
                      Generate template config for shops_create.py
                      Optional source: costume, item, dlc, all (default: all)
                      
TEMPLATE OPTIONS:
  --shop-ids=<list>   Comma-separated shop IDs (e.g., 5,6,10)
                      Overrides auto-detection from ShopItem section
                      
  --default-shop-ids  Use [1] as default when ShopItem section not found
                      Useful for DLCs without existing shop assignments
                      
  --no-interactive    Do not prompt for user input (required for CI/CD)
                      Must be used with --shop-ids or --default-shop-ids
                      
  --output=<file>     Custom output filename for generated template
                      Default: template_<input_filename>.json
```

#### Examples with Real Data

**Example 1: Basic ID Extraction**

```bash
python shops_find_unique_item_id_from_kurodlc.py my_costume_mod.kurodlc.json
```

**Sample DLC Structure:**
```json
{
  "CostumeParam": [
    {"item_id": 3596, "char_restrict": 1, "mdl_name": "c_van01b"},
    {"item_id": 3597, "char_restrict": 2, "mdl_name": "c_agnes01b"},
    {"item_id": 3598, "char_restrict": 3, "mdl_name": "c_feri01b"}
  ],
  "ItemTableData": [
    {"id": 3596, "name": "Van Costume 01", "category": 17},
    {"id": 3597, "name": "Agnes Costume 01", "category": 17},
    {"id": 3598, "name": "Feri Costume 01", "category": 17}
  ],
  "DLCTableData": [
    {"id": 1, "items": [3596, 3597, 3598], "name": "Costume Pack Vol.1"}
  ]
}
```

**Output (stdout):**
```
[3596, 3597, 3598]
```

**Output (stderr):**
```
# Extraction summary:
#   CostumeParam: 3 IDs
#   ItemTableData: 3 IDs
#   DLCTableData.items: 3 IDs
# Total unique IDs: 3
```

**Example 2: Extract Only Costume IDs**

```bash
python shops_find_unique_item_id_from_kurodlc.py my_costume_mod.kurodlc.json costume
```

**Output:**
```
[3596, 3597, 3598]
```

Only extracts from CostumeParam section.

**Example 3: Generate Template (Auto-Detect Shop IDs)**

**DLC with ShopItem section:**
```json
{
  "CostumeParam": [...],
  "ItemTableData": [...],
  "ShopItem": [
    {"shop_id": 21, "item_id": 3596, "unknown": 1},
    {"shop_id": 22, "item_id": 3596, "unknown": 1},
    {"shop_id": 23, "item_id": 3596, "unknown": 1}
  ]
}
```

**Command:**
```bash
python shops_find_unique_item_id_from_kurodlc.py my_costume_mod.kurodlc.json --generate-template costume
```

**Generated Template:** `template_my_costume_mod.kurodlc.json`
```json
{
  "_comment": [
    "Template config file generated by shops_find_unique_item_id_from_kurodlc.py v2.1",
    "Source file: my_costume_mod.kurodlc.json",
    "Generated: 2026-01-31 14:45:00",
    "",
    "INSTRUCTIONS:",
    "1. Review and modify shop_ids as needed",
    "2. Review item_ids (automatically extracted)",
    "3. Customize template structure if needed",
    "4. Run: python shops_create.py <this_file>",
    "",
    "Extraction summary:",
    "  - CostumeParam: 3 IDs",
    "  - Shop IDs: extracted from ShopItem"
  ],
  "item_ids": [3596, 3597, 3598],
  "shop_ids": [21, 22, 23],
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

**Example 4: Generate Template with Manual Shop IDs**

**For real shops from t_shop.json:**
```
Shop ID 5:  Item Shop
Shop ID 6:  Weapon/Armor Shop
Shop ID 10: Orbments
```

**Command:**
```bash
python shops_find_unique_item_id_from_kurodlc.py my_costume_mod.kurodlc.json --generate-template costume --shop-ids=5,6,10
```

**Generated Template:**
```json
{
  "item_ids": [3596, 3597, 3598],
  "shop_ids": [5, 6, 10],
  "template": { ... }
}
```

Result: 3 items × 3 shops = 9 shop assignments will be generated

**Example 5: DLC Without ShopItem Section**

```bash
# Using default shop ID [1]
python shops_find_unique_item_id_from_kurodlc.py my_costume_mod.kurodlc.json --generate-template costume --default-shop-ids

# Or specify shops manually
python shops_find_unique_item_id_from_kurodlc.py my_costume_mod.kurodlc.json --generate-template costume --shop-ids=21,22,23
```

**Example 6: CI/CD Pipeline (Non-Interactive)**

```bash
# For automated workflows
python shops_find_unique_item_id_from_kurodlc.py my_costume_mod.kurodlc.json --generate-template costume --default-shop-ids --no-interactive
```

**Error Handling (No ShopItem + No Flags):**
```
============================================================
ERROR: Cannot generate template
============================================================

Reason: No shop IDs found in ShopItem section.

The DLC file does not contain a ShopItem section,
and no shop IDs were specified.

Possible solutions:

  1. Specify shop IDs manually:
     python shops_find_unique_item_id_from_kurodlc.py my_costume_mod.kurodlc.json --generate-template --shop-ids=5,6,10

  2. Use default shop IDs [1]:
     python shops_find_unique_item_id_from_kurodlc.py my_costume_mod.kurodlc.json --generate-template --default-shop-ids --no-interactive

  3. Run interactively (without --no-interactive):
     python shops_find_unique_item_id_from_kurodlc.py my_costume_mod.kurodlc.json --generate-template
============================================================
```

#### Real Data Integration

**Game Shops (from t_shop.json - ShopInfo section):**
```
ID  5: Item Shop
ID  6: Weapon/Armor Shop
ID  8: Modification/Trade Shop
ID 10: Orbments
ID 21: Melrose Newspapers & Tobacco
ID 22: Melrose Newspapers & Tobacco
ID 23: Melrose Newspapers & Tobacco
```

**Common Shop Assignment Strategies:**

1. **All General Shops:**
```bash
--shop-ids=5,6,8,10
```

2. **Specific Vendor:**
```bash
--shop-ids=21,22,23  # All Melrose locations
```

3. **Costume-Specific:**
```bash
--shop-ids=6  # Only Weapon/Armor Shop
```

---

### shops_create.py

**Version:** v2.0  
**Purpose:** Generate bulk shop assignments from template configurations

#### All Parameters

```
shops_create.py <template_file>

ARGUMENTS:
  <template_file>     Template config file (JSON)
                      Generated by shops_find or created manually
                      
TEMPLATE FORMAT:
  {
    "item_ids": [<list of item IDs>],
    "shop_ids": [<list of shop IDs>],
    "template": {<shop item structure>},
    "output_section": "<section_name>" (optional, default: "ShopItem")
  }
  
OUTPUT:
  Creates: output_<template_file>
  Contains: Complete shop assignments ready to copy into .kurodlc.json
```

#### Examples with Real Data

**Example 1: Basic Shop Assignment Generation**

**Input Template:** `template_my_costume_mod.kurodlc.json`
```json
{
  "item_ids": [3596, 3597, 3598],
  "shop_ids": [5, 6, 10],
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

**Command:**
```bash
python shops_create.py template_my_costume_mod.kurodlc.json
```

**Output File:** `output_template_my_costume_mod.kurodlc.json`
```json
{
  "ShopItem": [
    {
      "shop_id": 5,
      "item_id": 3596,
      "unknown": 1,
      "start_scena_flags": [],
      "empty1": 0,
      "end_scena_flags": [],
      "int2": 0
    },
    {
      "shop_id": 5,
      "item_id": 3597,
      "unknown": 1,
      "start_scena_flags": [],
      "empty1": 0,
      "end_scena_flags": [],
      "int2": 0
    },
    {
      "shop_id": 5,
      "item_id": 3598,
      "unknown": 1,
      "start_scena_flags": [],
      "empty1": 0,
      "end_scena_flags": [],
      "int2": 0
    },
    {
      "shop_id": 6,
      "item_id": 3596,
      "unknown": 1,
      "start_scena_flags": [],
      "empty1": 0,
      "end_scena_flags": [],
      "int2": 0
    },
    {
      "shop_id": 6,
      "item_id": 3597,
      "unknown": 1,
      "start_scena_flags": [],
      "empty1": 0,
      "end_scena_flags": [],
      "int2": 0
    },
    {
      "shop_id": 6,
      "item_id": 3598,
      "unknown": 1,
      "start_scena_flags": [],
      "empty1": 0,
      "end_scena_flags": [],
      "int2": 0
    },
    {
      "shop_id": 10,
      "item_id": 3596,
      "unknown": 1,
      "start_scena_flags": [],
      "empty1": 0,
      "end_scena_flags": [],
      "int2": 0
    },
    {
      "shop_id": 10,
      "item_id": 3597,
      "unknown": 1,
      "start_scena_flags": [],
      "empty1": 0,
      "end_scena_flags": [],
      "int2": 0
    },
    {
      "shop_id": 10,
      "item_id": 3598,
      "unknown": 1,
      "start_scena_flags": [],
      "empty1": 0,
      "end_scena_flags": [],
      "int2": 0
    }
  ]
}
```

**Result:** 3 items × 3 shops = 9 shop assignments generated!

**Example 2: Advanced Template with Variables**

**Template with all variables:**
```json
{
  "item_ids": [3596, 3597],
  "shop_ids": [5, 6],
  "template": {
    "shop_id": "${shop_id}",
    "item_id": "${item_id}",
    "entry_index": "${index}",
    "total_entries": "${count}",
    "unknown": 1,
    "metadata": {
      "created_by": "shops_create.py v2.0",
      "position": "${index}"
    }
  }
}
```

**Output:**
```json
{
  "ShopItem": [
    {
      "shop_id": 5,
      "item_id": 3596,
      "entry_index": 0,
      "total_entries": 4,
      "unknown": 1,
      "metadata": {
        "created_by": "shops_create.py v2.0",
        "position": 0
      }
    },
    {
      "shop_id": 5,
      "item_id": 3597,
      "entry_index": 1,
      "total_entries": 4,
      "unknown": 1,
      "metadata": {
        "created_by": "shops_create.py v2.0",
        "position": 1
      }
    },
    {
      "shop_id": 6,
      "item_id": 3596,
      "entry_index": 2,
      "total_entries": 4,
      "unknown": 1,
      "metadata": {
        "created_by": "shops_create.py v2.0",
        "position": 2
      }
    },
    {
      "shop_id": 6,
      "item_id": 3597,
      "entry_index": 3,
      "total_entries": 4,
      "unknown": 1,
      "metadata": {
        "created_by": "shops_create.py v2.0",
        "position": 3
      }
    }
  ]
}
```

**Variable Substitution:**
- `${shop_id}` → Current shop ID
- `${item_id}` → Current item ID
- `${index}` → Entry index (0-based)
- `${count}` → Total number of entries

**Example 3: Custom Output Section**

```json
{
  "item_ids": [3596, 3597],
  "shop_ids": [5, 6],
  "output_section": "CustomShopAssignments",
  "template": {
    "shop_id": "${shop_id}",
    "item_id": "${item_id}"
  }
}
```

**Output:**
```json
{
  "CustomShopAssignments": [
    {"shop_id": 5, "item_id": 3596},
    {"shop_id": 5, "item_id": 3597},
    {"shop_id": 6, "item_id": 3596},
    {"shop_id": 6, "item_id": 3597}
  ]
}
```

**Example 4: Real-World Scenario - 50 Costumes to 10 Shops**

**Template:**
```json
{
  "item_ids": [3596, 3597, 3598, ..., 3645],  // 50 IDs
  "shop_ids": [5, 6, 8, 10, 21, 22, 23, 24, 26, 27],  // 10 shops
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

**Result:**
- 50 items × 10 shops = **500 shop assignments**
- Generated in < 1 second
- All assignments automatically created with proper structure

---

### find_all_items.py

**Version:** v2.0 (Standalone with multi-source support)  
**Purpose:** Search and browse game items with intelligent auto-detection

#### All Parameters

```
find_all_items.py [search_query] [options]

ARGUMENTS:
  search_query   (Optional) Search query with optional prefix
  
SEARCH MODES:
  id:NUMBER      Search by exact ID (e.g., id:100)
  name:TEXT      Search in item names (e.g., name:sword)
  TEXT           Auto-detect (numbers → ID search, text → name search)
  
OPTIONS:
  --source=TYPE       Force specific source: json, tbl, original, p3a, zzz
  --no-interactive    Auto-select first source if multiple found
  --keep-extracted    Keep temporary extracted files from P3A
  --help              Show this help message
  
SUPPORTED SOURCES (auto-detected in priority order):
  1. t_item.json
  2. t_item.tbl.original
  3. t_item.tbl
  4. script_en.p3a / script_eng.p3a (extracts t_item.tbl)
  5. zzz_combined_tables.p3a (extracts t_item.tbl)
```

#### Examples with Real Data

**Example 1: Search by ID (Auto-Detect)**

```bash
python find_all_items.py 310
```

**Output:**
```
# Auto-detected ID search for '310'
# Use 'name:310' to search for '310' in item names instead

Loading item data...
Using source: t_item.json
Loaded 2116 items from: t_item.json

310 : Earth Sepith

Total: 1 item(s)
```

**Example 2: Search by Name**

```bash
python find_all_items.py sepith
```

**Output:**
```
Loading item data...
Using source: t_item.json
Loaded 2116 items from: t_item.json

310 : Earth Sepith
311 : Water Sepith
312 : Fire Sepith
313 : Wind Sepith
314 : Time Sepith
315 : Space Sepith
316 : Mirage Sepith
317 : Sepith Mass
318 : All Element Sepith

Total: 9 item(s)
```

**Example 3: Explicit Search Modes**

```bash
# Explicit ID search
python find_all_items.py id:310

# Explicit name search (useful for numbers in names)
python find_all_items.py name:100
```

**Example 4: Force Specific Source**

```bash
# Force JSON source
python find_all_items.py sword --source=json

# Force TBL source (requires kurodlc_lib.py)
python find_all_items.py sword --source=tbl

# Extract from P3A archive (requires p3a_lib.py + dependencies)
python find_all_items.py sword --source=p3a
```

**Example 5: Multi-Source Selection (Interactive)**

```bash
python find_all_items.py

# When multiple sources exist:
# 
# Multiple data sources detected. Select source to use:
#   1) t_item.json
#   2) t_item.tbl.original
#   3) t_item.tbl
#   4) script_en.p3a (extract t_item.tbl)
# 
# Enter choice [1-4]:
```

**Example 6: Automation (No Interactive Prompts)**

```bash
# For CI/CD or scripts - auto-select first source
python find_all_items.py sepith --source=json --no-interactive > items.txt
```

**Example 7: Using with P3A Archives**

```bash
# Extract t_item.tbl from P3A and search
python find_all_items.py sword --source=p3a

# Keep extracted file for debugging
python find_all_items.py sword --source=p3a --keep-extracted
```

#### Real Item Categories

From t_item.json, items are organized by category:

```
Category 0:  Sepith items (310-318)
Category 2:  Key items
Category 17: Costume items
Category 30: Casino items
```

---

### find_all_shops.py

**Version:** v2.0 (Standalone with multi-source support)  
**Purpose:** List all shops from game data with flexible source handling

#### All Parameters

```
find_all_shops.py [search_text] [options]

ARGUMENTS:
  search_text   (Optional) Filter shops by text in name (case-insensitive)
  
OPTIONS:
  --source=TYPE       Force specific source: json, tbl, original, p3a, zzz
  --no-interactive    Auto-select first source if multiple found
  --keep-extracted    Keep temporary extracted files from P3A
  --debug             Show debug information about data structure
  --help              Show this help message
  
SUPPORTED SOURCES (auto-detected in priority order):
  1. t_shop.json
  2. t_shop.tbl.original
  3. t_shop.tbl
  4. script_en.p3a / script_eng.p3a (extracts t_shop.tbl)
  5. zzz_combined_tables.p3a (extracts t_shop.tbl)
```

#### Examples with Real Data

**Example 1: List All Shops**

```bash
python find_all_shops.py
```

**Output (first 20 shops):**
```
Loading shop data...
Using source: t_shop.json
Loaded 215 shops from: t_shop.json

  4 : Casino
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
 27 : Stanley's Factory
 28 : Stanley's Factory
...

Total: 215 shop(s)
```

**Example 2: Search for Specific Shop**

```bash
python find_all_shops.py melrose
```

**Output:**
```
Loading shop data...
Using source: t_shop.json
Loaded 215 shops from: t_shop.json

 21 : Melrose Newspapers & Tobacco
 22 : Melrose Newspapers & Tobacco
 23 : Melrose Newspapers & Tobacco

Total: 3 shop(s)
```

**Example 3: Using with Debug Mode**

```bash
python find_all_shops.py --debug
```

**Output:**
```
DEBUG: Loaded 215 shop entries
DEBUG: Type of shops: <class 'list'>
DEBUG: First shop entry: {'id': 5, 'shop_name': 'Item Shop', ...}
DEBUG: Type of first entry: <class 'dict'>

...shop listing...
```

**Example 4: Force TBL Source**

```bash
python find_all_shops.py --source=tbl
```

**Example 5: Automation**

```bash
python find_all_shops.py --source=json --no-interactive > shops_list.txt
```

#### Using Output for Template Generation

1. **Find shops:**
```bash
python find_all_shops.py > shops_list.txt
```

2. **Select shop IDs for your template:**
```
For general items: 5, 6, 8, 10
For specific vendor: 21, 22, 23
```

3. **Use in template generation:**
```bash
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json --generate-template --shop-ids=5,6,8,10
```

---

### find_unique_item_id_for_t_costumes.py

**Version:** v2.0 (Standalone with multi-source support)  
**Purpose:** Extract all unique costume item IDs from game data

#### All Parameters

```
find_unique_item_id_for_t_costumes.py [options]

OPTIONS:
  --source=TYPE       Force specific source: json, tbl, original, p3a, zzz
  --no-interactive    Auto-select first source if multiple found
  --keep-extracted    Keep temporary extracted files from P3A
  --format=FORMAT     Output format: list (default), count, range
  --help              Show this help message
  
OUTPUT FORMATS:
  list   Print Python list of IDs: [100, 101, 102, ...]
  count  Print count of unique IDs: 150 unique item IDs
  range  Print ID range: 100-5000 (150 IDs)
  
SUPPORTED SOURCES (auto-detected in priority order):
  1. t_costume.json
  2. t_costume.tbl.original
  3. t_costume.tbl
  4. script_en.p3a / script_eng.p3a (extracts t_costume.tbl)
  5. zzz_combined_tables.p3a (extracts t_costume.tbl)
```

#### Examples with Real Data

**Example 1: Extract Costume IDs (Default List Format)**

```bash
python find_unique_item_id_for_t_costumes.py
```

**Output:**
```
Loading costume data...
Using source: t_costume.json
Loaded 487 costumes from: t_costume.json

[0, 2350, 2351, 2352, 2353, ..., 4807, 4808, 4809]
```

**Example 2: Count Only**

```bash
python find_unique_item_id_for_t_costumes.py --format=count
```

**Output:**
```
Loading costume data...
Using source: t_costume.json
Loaded 487 costumes from: t_costume.json

487 unique item IDs
```

**Example 3: Range Info**

```bash
python find_unique_item_id_for_t_costumes.py --format=range
```

**Output:**
```
Loading costume data...
Using source: t_costume.json
Loaded 487 costumes from: t_costume.json

0-4809 (487 IDs)
```

**Example 4: Force JSON Source**

```bash
python find_unique_item_id_for_t_costumes.py --source=json
```

**Example 5: Extract from P3A Archive**

```bash
python find_unique_item_id_for_t_costumes.py --source=p3a --no-interactive
```

**Use case:** Find which IDs are used by game costumes to avoid conflicts when creating custom costume mods.

---

### find_unique_item_id_for_t_item_category.py

**Version:** v2.0 (Standalone with multi-source support)  
**Purpose:** Extract item IDs by category from game item database

#### All Parameters

```
find_unique_item_id_for_t_item_category.py <category> [options]

ARGUMENTS:
  category      Category number to filter items by (integer, REQUIRED)

OPTIONS:
  --source=TYPE       Force specific source: json, tbl, original, p3a, zzz
  --no-interactive    Auto-select first source if multiple found
  --keep-extracted    Keep temporary extracted files from P3A
  --format=FORMAT     Output format: list (default), count, range
  --help              Show this help message
  
OUTPUT FORMATS:
  list   Print Python list of IDs: [100, 101, 102, ...]
  count  Print count of unique IDs: 150 unique item IDs in category 5
  range  Print ID range: 100-5000 (150 IDs in category 5)
  
SUPPORTED SOURCES (auto-detected in priority order):
  1. t_item.json
  2. t_item.tbl.original
  3. t_item.tbl
  4. script_en.p3a / script_eng.p3a (extracts t_item.tbl)
  5. zzz_combined_tables.p3a (extracts t_item.tbl)
```

#### Examples with Real Data

**Example 1: Extract Sepith IDs (Category 0)**

```bash
python find_unique_item_id_for_t_item_category.py 0
```

**Output:**
```
Loading item data for category 0...
Using source: t_item.json
Loaded 2116 items from: t_item.json

[310, 311, 312, 313, 314, 315, 316, 317, 318]
```

**Example 2: Count Costume Items (Category 17)**

```bash
python find_unique_item_id_for_t_item_category.py 17 --format=count
```

**Output:**
```
Loading item data for category 17...
Using source: t_item.json
Loaded 2116 items from: t_item.json

487 unique item IDs in category 17
```

**Example 3: Range for Category 5**

```bash
python find_unique_item_id_for_t_item_category.py 5 --format=range
```

**Output:**
```
Loading item data for category 5...
Using source: t_item.json
Loaded 2116 items from: t_item.json

100-450 (78 IDs in category 5)
```

**Example 4: Force TBL Source**

```bash
python find_unique_item_id_for_t_item_category.py 17 --source=tbl
```

**Example 5: Automation**

```bash
# Extract all categories to separate files
for cat in 0 2 5 17 30; do
  python find_unique_item_id_for_t_item_category.py $cat --source=json --no-interactive > category_$cat.txt
done
```

#### Known Item Categories

From t_item.json:

```
Category 0:  Sepith (310-318)
Category 2:  Key Items
Category 5:  Consumables
Category 17: Costumes (487 items, ID range 0-4809)
Category 30: Casino Items
```

**Use case:** Find which IDs are used by specific item categories to avoid conflicts or to understand item organization.

---



## Data Structure Specifications

### .kurodlc.json Structure

**Complete structure with all sections:**

```json
{
  "CostumeParam": [
    {
      "char_restrict": 1,
      "type": 0,
      "item_id": 3596,
      "unk0": 0,
      "unk_txt0": "",
      "mdl_name": "c_van01b",
      "unk1": 0,
      "unk2": 0,
      "attach_name": "",
      "unk_txt1": "",
      "unk_txt2": ""
    }
  ],
  "ItemTableData": [
    {
      "id": 3596,
      "chr_restrict": 1,
      "flags": "",
      "unk_txt": "1",
      "category": 17,
      "subcategory": 15,
      "name": "Costume Pack Vol.1",
      "desc": "Custom costume for party member",
      "visual_name": "",
      "icon_index": 0,
      "int0": 0,
      "int1": 1,
      "price": 0,
      "int3": 0,
      "sell_price": 0,
      "sort_id": 0
    }
  ],
  "DLCTableData": [
    {
      "id": 1,
      "sort_id": 0,
      "items": [3596, 3597, 3598],
      "unk0": 0,
      "quantity": [1, 1, 1],
      "name": "Costume Pack Vol.1",
      "desc": "Custom costume for party member",
      "int3": 0,
      "empty": 0
    }
  ],
  "ShopItem": [
    {
      "shop_id": 21,
      "item_id": 3596,
      "unknown": 1,
      "start_scena_flags": [],
      "empty1": 0,
      "end_scena_flags": [],
      "int2": 0
    }
  ]
}
```

**Field Descriptions:**

**CostumeParam Section:**
- `char_restrict`: Character ID restriction (1=Van, 2=Agnes, etc.)
- `type`: Costume type
- `item_id`: Item ID (must match ItemTableData)
- `mdl_name`: 3D model filename
- `attach_name`: Attachment model (weapons, accessories)

**ItemTableData Section:**
- `id`: Item ID (must be unique, 1-5000)
- `chr_restrict`: Character restriction (0=all, 1=Van, etc.)
- `category`: Item category (17=costume, 0=consumable, etc.)
- `subcategory`: Item subcategory
- `name`: Item display name
- `desc`: Item description
- `price`: Purchase price (0 for non-purchasable)
- `sell_price`: Selling price

**DLCTableData Section:**
- `id`: DLC pack ID
- `items`: Array of item IDs included in DLC
- `quantity`: Quantity of each item (parallel array to items)
- `name`: DLC pack name
- `desc`: DLC pack description

**ShopItem Section:**
- `shop_id`: Shop ID (from t_shop.json ShopInfo)
- `item_id`: Item ID to sell
- `unknown`: Unknown field (usually 1)
- `start_scena_flags`: Scenario flags to enable item
- `end_scena_flags`: Scenario flags to disable item

### t_item.json Structure

**Real structure from game data:**

```json
{
  "headers": [
    {"name": "ItemTableData", "schema": "Kai"}
  ],
  "data": [
    {
      "name": "ItemTableData",
      "data": [
        {
          "id": 310,
          "chr_restrict": 0,
          "flags": "",
          "unk_txt": "0",
          "category": 0,
          "subcategory": 0,
          "name": "Earth Sepith",
          "desc": "A blue-green sepith that embodies the earth element.",
          "visual_name": "",
          "icon_index": 310,
          "int0": 0,
          "int1": 99,
          "price": 10,
          "int3": 0,
          "sell_price": 5,
          "sort_id": 310
        },
        {
          "id": 311,
          "chr_restrict": 0,
          "flags": "",
          "unk_txt": "0",
          "category": 0,
          "subcategory": 0,
          "name": "Water Sepith",
          "desc": "A blue sepith that embodies the water element.",
          "visual_name": "",
          "icon_index": 311,
          "int0": 0,
          "int1": 99,
          "price": 10,
          "int3": 0,
          "sell_price": 5,
          "sort_id": 311
        }
      ]
    }
  ]
}
```

**Game Item ID Ranges:**
- Sepith: 310-318
- Consumables: 100-309, 319+
- Key Items: Various ranges
- Costumes: 2350-4921
- Total items: 2116
- ID range: 1-4921

### t_shop.json Structure

**Real structure from game data:**

```json
{
  "headers": [
    {"name": "ShopInfo", "schema": "Kuro1"},
    {"name": "ShopItem", "schema": "Kuro1"}
  ],
  "data": [
    {
      "name": "ShopInfo",
      "data": [
        {
          "id": 5,
          "shop_name": "Item Shop",
          "long1": 5,
          "flag": "C",
          "empty": 0,
          "shop_price_percent": 100,
          "shop_cam_pos_x": 0.294,
          "shop_cam_pos_y": 1.309,
          "shop_cam_pos_z": -0.135,
          "shop_cam_rotation_1": 4.46,
          "shop_cam_rotation_2": 215.846,
          "shop_cam_rotation_3": 0.0,
          "shop_cam_rotation_4": 2.0,
          "int1": 0,
          "int2": 0,
          "int3": 0,
          "int4": 0
        }
      ]
    },
    {
      "name": "ShopItem",
      "data": [
        {
          "shop_id": 5,
          "item_id": 100,
          "unknown": 1,
          "start_scena_flags": [],
          "empty1": 0,
          "end_scena_flags": [],
          "int2": 0
        }
      ]
    }
  ]
}
```

**Shop ID Reference:**
- General shops: 5, 6, 8, 10
- Vendor chains: 21-23, 24-26, 27-28
- Special: 4 (Casino), 9 (Kitchen)
- Total shops: 215

---

## Export/Import Formats

### ID Mapping Export Format

**File:** `id_mapping_YYYYMMDD_HHMMSS.json`

**Structure:**
```json
{
  "source_file": "my_mod.kurodlc.json",
  "timestamp": "2026-01-31 14:30:22",
  "game_database": "t_item.json",
  "game_id_count": 2116,
  "game_id_range": [1, 4921],
  "engine_limit": 5000,
  "algorithm": "smart_v2.7",
  "mappings": {
    "310": 2500,
    "311": 2501,
    "312": 2502
  },
  "conflicts": [
    {
      "old_id": 310,
      "new_id": 2500,
      "reason": "Conflict with game item: Earth Sepith",
      "sections_affected": ["CostumeParam", "ItemTableData", "DLCTableData.items"]
    },
    {
      "old_id": 311,
      "new_id": 2501,
      "reason": "Conflict with game item: Water Sepith",
      "sections_affected": ["CostumeParam", "ItemTableData", "DLCTableData.items"]
    }
  ],
  "statistics": {
    "total_ids_in_dlc": 50,
    "conflicts_found": 3,
    "conflicts_resolved": 3,
    "safe_ids": 47
  }
}
```

**Manual Editing:**

You can edit the `mappings` section to customize ID assignments:

```json
{
  "mappings": {
    "310": 3000,  // Changed from 2500 to 3000
    "311": 3001,  // Changed from 2501 to 3001
    "312": 2502   // Kept original assignment
  }
}
```

Then import:
```bash
python resolve_id_conflicts_in_kurodlc.py repair --import id_mapping_20260131_143022.json --apply
```

### Template Configuration Format

**File:** `template_<source>.json`

**Minimal Template:**
```json
{
  "item_ids": [3596, 3597, 3598],
  "shop_ids": [5, 6, 10]
}
```

Uses default template structure.

**Custom Template:**
```json
{
  "_comment": ["Optional comment"],
  "item_ids": [3596, 3597, 3598],
  "shop_ids": [5, 6, 10],
  "template": {
    "shop_id": "${shop_id}",
    "item_id": "${item_id}",
    "custom_field": 42,
    "metadata": {
      "index": "${index}",
      "total": "${count}"
    }
  },
  "output_section": "ShopItem"
}
```

**Advanced Template with Conditions:**
```json
{
  "item_ids": [3596, 3597, 3598],
  "shop_ids": [5, 6, 10],
  "template": {
    "shop_id": "${shop_id}",
    "item_id": "${item_id}",
    "unknown": 1,
    "start_scena_flags": [],
    "empty1": 0,
    "end_scena_flags": [],
    "int2": 0,
    "price_multiplier": 1.0,
    "stock_quantity": 99,
    "availability": {
      "always_available": true,
      "required_chapter": 1
    }
  }
}
```

---

## Log Files

### ID Conflict Repair Log

**File:** `id_conflict_repair_YYYYMMDD_HHMMSS.log`

**Sample Content:**
```
==========================================================
ID CONFLICT REPAIR LOG
==========================================================
Timestamp: 2026-01-31 14:30:22
Source File: my_costume_mod.kurodlc.json
Game Database: t_item.json
Algorithm: Smart v2.7 (middle-out distribution)

==========================================================
GAME DATABASE ANALYSIS
==========================================================
Total items in game: 2116
ID range: 1 - 4921
Engine limit: 1 - 5000
Available IDs: 879 (5000 - 2116 - buffer)

==========================================================
DLC ANALYSIS
==========================================================
Total IDs in DLC: 50
ID range: 100 - 149
Sections:
  - CostumeParam: 50 entries
  - ItemTableData: 50 entries
  - DLCTableData: 1 pack with 50 items

==========================================================
CONFLICTS DETECTED
==========================================================
Total conflicts: 21

Conflicting IDs:
  100 → Game item: "Tears" (Category: 1)
  101 → Game item: "Tear Balm" (Category: 1)
  102 → Game item: "Teara Balm" (Category: 1)
  ...
  120 → Game item: "Septium Chunk" (Category: 1)

Safe IDs (no conflicts):
  121-149 (29 IDs)

==========================================================
ID REASSIGNMENT PLAN
==========================================================
Algorithm: Smart v2.7
Starting point: 2500 (middle of 1-5000 range)
Direction: Ascending from middle

Mappings:
  100 → 2500
  101 → 2501
  102 → 2502
  ...
  120 → 2520

IDs to keep unchanged: 121-149

==========================================================
APPLYING CHANGES
==========================================================
Backup created: my_costume_mod.kurodlc.json.bak_20260131_143022

Updating CostumeParam section:
  ✓ Entry 0: item_id 100 → 2500
  ✓ Entry 1: item_id 101 → 2501
  ...
  ✓ Entry 20: item_id 120 → 2520
  - Entries 21-49: No changes (IDs 121-149 are safe)

Updating ItemTableData section:
  ✓ Entry 0: id 100 → 2500
  ✓ Entry 1: id 101 → 2501
  ...
  ✓ Entry 20: id 120 → 2520
  - Entries 21-49: No changes

Updating DLCTableData section:
  ✓ DLC Pack 1: items array updated
    [100,101,...,120] → [2500,2501,...,2520]
    [121,...,149] unchanged

==========================================================
RESULTS
==========================================================
✓ All conflicts resolved
✓ 21 IDs reassigned
✓ 29 IDs kept unchanged
✓ File saved: my_costume_mod.kurodlc.json
✓ Backup available: my_costume_mod.kurodlc.json.bak_20260131_143022

==========================================================
VERIFICATION
==========================================================
✓ JSON structure valid
✓ All ID references updated consistently
✓ No new conflicts introduced
✓ ID range within engine limit (1-5000)

Operation completed successfully!
==========================================================
```

---

## Advanced Workflows

### Workflow 1: Complete Mod Creation Pipeline

**Step 1: Create Initial DLC File**
```json
{
  "CostumeParam": [
    {"item_id": 100, "char_restrict": 1, "mdl_name": "c_van01b"},
    {"item_id": 101, "char_restrict": 2, "mdl_name": "c_agnes01b"}
  ],
  "ItemTableData": [
    {"id": 100, "name": "Custom Costume 01", "category": 17},
    {"id": 101, "name": "Custom Costume 02", "category": 17}
  ],
  "DLCTableData": [
    {"id": 1, "items": [100, 101], "name": "Custom DLC Pack 01"}
  ]
}
```

**Step 2: Check for Conflicts**
```bash
python resolve_id_conflicts_in_kurodlc.py checkbydlc
```

**Output:**
```
[BAD] 100 - Conflict! Used by: Tears
[BAD] 101 - Conflict! Used by: Tear Balm
```

**Step 3: Fix Conflicts**
```bash
python resolve_id_conflicts_in_kurodlc.py repair --apply
```

**Result:** IDs reassigned to 2500, 2501

**Step 4: Generate Shop Template**
```bash
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json --generate-template costume --shop-ids=5,6,10
```

**Step 5: Create Shop Assignments**
```bash
python shops_create.py template_my_mod.kurodlc.json
```

**Step 6: Merge Shop Assignments**
Copy ShopItem section from `output_template_my_mod.kurodlc.json` into `my_mod.kurodlc.json`

**Final Result:**
```json
{
  "CostumeParam": [
    {"item_id": 2500, "char_restrict": 1, "mdl_name": "c_van01b"},
    {"item_id": 2501, "char_restrict": 2, "mdl_name": "c_agnes01b"}
  ],
  "ItemTableData": [
    {"id": 2500, "name": "Custom Costume 01", "category": 17},
    {"id": 2501, "name": "Custom Costume 02", "category": 17}
  ],
  "DLCTableData": [
    {"id": 1, "items": [2500, 2501], "name": "Costume Pack Vol.1"}
  ],
  "ShopItem": [
    {"shop_id": 5, "item_id": 2500, "unknown": 1, ...},
    {"shop_id": 5, "item_id": 2501, "unknown": 1, ...},
    {"shop_id": 6, "item_id": 2500, "unknown": 1, ...},
    {"shop_id": 6, "item_id": 2501, "unknown": 1, ...},
    {"shop_id": 10, "item_id": 2500, "unknown": 1, ...},
    {"shop_id": 10, "item_id": 2501, "unknown": 1, ...}
  ]
}
```

**Ready to use!** ✓ No conflicts, ✓ Items in shops

### Workflow 2: Batch Processing Multiple DLCs

**Scenario:** You have 5 DLC mods to process

```bash
# Create processing script
cat > process_all_dlcs.sh << 'EOF'
#!/bin/bash

for dlc in mod1 mod2 mod3 mod4 mod5; do
  echo "Processing ${dlc}.kurodlc.json..."
  
  # Fix conflicts
  python resolve_id_conflicts_in_kurodlc.py repair --apply
  
  # Generate template
  python shops_find_unique_item_id_from_kurodlc.py ${dlc}.kurodlc.json \
    --generate-template costume \
    --shop-ids=5,6,10 \
    --no-interactive \
    --output=template_${dlc}.json
  
  # Create shop assignments
  python shops_create.py template_${dlc}.json
  
  echo "✓ ${dlc} processed!"
done

echo "All DLCs processed!"
EOF

chmod +x process_all_dlcs.sh
./process_all_dlcs.sh
```

### Workflow 3: CI/CD Integration

**GitHub Actions example:**

```yaml
name: Process DLC Mods

on:
  push:
    paths:
      - '**.kurodlc.json'

jobs:
  process:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install colorama --break-system-packages
      
      - name: Fix ID conflicts
        run: |
          python resolve_id_conflicts_in_kurodlc.py repair --apply
      
      - name: Generate shop templates
        run: |
          for file in *.kurodlc.json; do
            python shops_find_unique_item_id_from_kurodlc.py "$file" \
              --generate-template costume \
              --default-shop-ids \
              --no-interactive
          done
      
      - name: Create shop assignments
        run: |
          for template in template_*.json; do
            python shops_create.py "$template"
          done
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v2
        with:
          name: processed-dlcs
          path: |
            *.kurodlc.json
            output_*.json
```

### Workflow 4: Testing with Real Game Data

**Test your DLC before release:**

```bash
# 1. Extract game data (if you have game files)
# Use kuro_dlc_tool to extract t_item.tbl, t_shop.tbl, etc.

# 2. Convert to JSON (if needed)
# The toolkit works with both JSON and TBL

# 3. Verify no conflicts
python find_all_items.py t_item.json 2500
# Should show: [No results] if ID is available

# 4. Check your DLC IDs
python find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json

# 5. Verify against game database
python resolve_id_conflicts_in_kurodlc.py checkbydlc

# 6. Test shop assignments
# Check if shop IDs exist in game
python find_all_shops.py t_shop.json

# 7. Validate structure
# Make sure all sections are properly formatted
```

### Workflow 5: Manual ID Mapping Control

**For precise control over ID assignments:**

```bash
# 1. Generate ID mapping export
python resolve_id_conflicts_in_kurodlc.py repair --export

# 2. Edit the mapping file
nano id_mapping_20260131_143022.json

# Example edits:
{
  "mappings": {
    "100": 3000,  // Assign to specific range
    "101": 3001,
    "102": 3500,  // Skip some IDs
    "103": 3501
  }
}

# 3. Import custom mapping
python resolve_id_conflicts_in_kurodlc.py repair \
  --import id_mapping_20260131_143022.json \
  --apply

# 4. Verify results
python find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json
```

---

## Real-World Examples

### Example 1: Costume Pack for All Characters

**Goal:** Create costume pack with 1 costume per character (8 characters)

**Game Characters:**
- ID 1: Van
- ID 2: Agnes  
- ID 3: Feri
- ID 4: Aaron
- ID 5: Risette
- ID 6: Quatre
- ID 7: Judith
- ID 8: Bergard

**DLC Structure:**
```json
{
  "CostumeParam": [
    {"item_id": 3596, "char_restrict": 1, "mdl_name": "c_van01b"},
    {"item_id": 3597, "char_restrict": 2, "mdl_name": "c_agnes01b"},
    {"item_id": 3598, "char_restrict": 3, "mdl_name": "c_feri01b"},
    {"item_id": 3599, "char_restrict": 4, "mdl_name": "c_aaron01b"},
    {"item_id": 3600, "char_restrict": 5, "mdl_name": "c_risette01b"},
    {"item_id": 3601, "char_restrict": 6, "mdl_name": "c_quatre01b"},
    {"item_id": 3602, "char_restrict": 7, "mdl_name": "c_judith01b"},
    {"item_id": 3603, "char_restrict": 8, "mdl_name": "c_bergard01b"}
  ],
  "ItemTableData": [
    {"id": 3596, "chr_restrict": 1, "name": "Costume Pack Vol.1", "category": 17},
    {"id": 3597, "chr_restrict": 2, "name": "Costume Pack Vol.1", "category": 17},
    {"id": 3598, "chr_restrict": 3, "name": "Costume Pack Vol.1", "category": 17},
    {"id": 3599, "chr_restrict": 4, "name": "Costume Pack Vol.1", "category": 17},
    {"id": 3600, "chr_restrict": 5, "name": "Costume Pack Vol.1", "category": 17},
    {"id": 3601, "chr_restrict": 6, "name": "Costume Pack Vol.1", "category": 17},
    {"id": 3602, "chr_restrict": 7, "name": "Costume Pack Vol.1", "category": 17},
    {"id": 3603, "chr_restrict": 8, "name": "Costume Pack Vol.1", "category": 17}
  ],
  "DLCTableData": [
    {
      "id": 1,
      "items": [3596, 3597, 3598, 3599, 3600, 3601, 3602, 3603],
      "quantity": [1, 1, 1, 1, 1, 1, 1, 1],
      "name": "Costume Pack Vol.1",
      "desc": "Custom costume for party member"
    }
  ]
}
```

**Processing:**
```bash
# Check conflicts (should be none if using 3596-3603)
python resolve_id_conflicts_in_kurodlc.py checkbydlc

# Generate shop template for Weapon/Armor Shop
python shops_find_unique_item_id_from_kurodlc.py costume_pack.kurodlc.json \
  --generate-template costume --shop-ids=6

# Create shop assignments
python shops_create.py template_costume_pack.kurodlc.json
```

**Result:** 8 costumes available in Weapon/Armor Shop

### Example 2: Large Costume Collection (50+ Items)

**Shops used (from t_shop.json):**
- ID 5: Item Shop
- ID 6: Weapon/Armor Shop
- ID 21-23: Melrose Newspapers & Tobacco (3 locations)

**Commands:**
```bash
# Generate template with multiple shop locations
python shops_find_unique_item_id_from_kurodlc.py large_costume_pack.kurodlc.json \
  --generate-template costume \
  --shop-ids=5,6,21,22,23

# Creates template with:
# - 50 item IDs
# - 5 shop locations
# = 250 shop assignments

python shops_create.py template_large_costume_pack.kurodlc.json

# Result: 250 shop assignments generated in < 1 second
```

---

**End of Advanced Documentation**

This advanced documentation covers all scripts, parameters, real data examples, and workflows. For basic usage, see the main README sections.
