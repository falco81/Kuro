# Advanced Documentation - KuroDLC Modding Toolkit

This section provides comprehensive, in-depth documentation for advanced users, including all parameters, real data examples, data structure specifications, and detailed workflows.

## 📋 Table of Contents

- [Script Reference](#script-reference)
  - [resolve_id_conflicts_in_kurodlc.py](#resolve_id_conflicts_in_kurodlcpy)
  - [shops_find_unique_item_id_from_kurodlc.py](#shops_find_unique_item_id_from_kurodlcpy)
  - [shops_create.py](#shops_createpy)
  - [visualize_id_allocation.py](#visualize_id_allocationpy) ⭐ NEW
  - [convert_kurotools_schemas.py](#convert_kurotools_schemaspy)
  - [3D Model Viewer Scripts](NEW_FEATURES_DOCUMENTATION.md) ⭐ NEW
  - [find_all_items.py](#find_all_itemspy)
  - [find_all_shops.py](#find_all_shopspy)
  - [Other Utility Scripts](#other-utility-scripts)
- [Data Structure Specifications](#data-structure-specifications)
- [Real Data Examples](#real-data-examples)
- [Export/Import Formats](#exportimport-formats)
- [Log Files](#log-files)
- [Advanced Workflows](#advanced-workflows)
- [3D Model Viewing](NEW_FEATURES_DOCUMENTATION.md) ⭐ NEW

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

## convert_kurotools_schemas.py

**Version:** v1.0  
**Purpose:** Convert KuroTools schema definitions to kurodlc_schema.json format

### All Parameters

```
convert_kurotools_schemas.py

No command-line parameters required.

REQUIREMENTS:
  - kurodlc_schema.json in same directory (optional - will create new if missing)
  - schemas/ folder with headers/ subfolder
  - schemas/headers/ must contain KuroTools .json schema files

AUTOMATIC DETECTION:
  - Working directory: Where script is executed
  - Input schema: ./kurodlc_schema.json
  - KuroTools schemas: ./schemas/headers/*.json
  - Output: ./kurodlc_schema_updated.json
  - Report: ./conversion_report.txt

FILE STRUCTURE:
  your_directory/
  ├── convert_kurotools_schemas.py
  ├── kurodlc_schema.json (optional)
  └── schemas/
      └── headers/
          ├── ATBonusParam.json
          ├── ItemTableData.json
          └── ... (280+ files)
```

### What It Does

The converter performs these steps:

1. **Loads Existing Schema** (if exists)
   - Reads `kurodlc_schema.json`
   - Creates index of existing schemas by (table_header, schema_length)

2. **Scans KuroTools Schemas**
   - Searches `schemas/headers/` for all .json files
   - Loads each schema definition

3. **Converts Each Schema**
   - Maps KuroTools data types to Python struct format
   - Calculates total schema size in bytes
   - Flattens nested structures (arrays, objects)
   - Generates keys list and values string
   - Detects primary keys (defaults to "id" if present)

4. **Merges Results**
   - Compares with existing schemas
   - Prevents duplicates (same table + size)
   - Preserves original schemas
   - Adds only new entries

5. **Outputs Files**
   - `kurodlc_schema_updated.json` - Combined schema
   - `conversion_report.txt` - Detailed statistics

### Type Mapping Reference

Complete mapping from KuroTools to kurodlc format:

| KuroTools Type | Python Struct | Value Type | Size (bytes) | Description |
|----------------|---------------|------------|--------------|-------------|
| `byte` | `b` | `n` | 1 | Signed 8-bit integer |
| `ubyte` | `B` | `n` | 1 | Unsigned 8-bit integer |
| `short` | `h` | `n` | 2 | Signed 16-bit integer |
| `ushort` | `H` | `n` | 2 | Unsigned 16-bit integer |
| `int` | `i` | `n` | 4 | Signed 32-bit integer |
| `uint` | `I` | `n` | 4 | Unsigned 32-bit integer |
| `long` | `q` | `n` | 8 | Signed 64-bit integer |
| `ulong` | `Q` | `n` | 8 | Unsigned 64-bit integer |
| `float` | `f` | `n` | 4 | 32-bit floating point |
| `toffset` | `Q` | `t` | 8 | Text offset (pointer to string) |
| `u32array` | `QI` | `a` | 12 | Array of 32-bit values (offset + count) |
| `u16array` | `QI` | `b` | 12 | Array of 16-bit values (offset + count) |

**Value Type Legend:**
- `n` = Number (integer or float)
- `t` = Text (null-terminated string offset)
- `a` = Array of 32-bit values
- `b` = Array of 16-bit values (or byte array)

### Nested Structure Handling

KuroTools supports nested structures with a `size` parameter. These are automatically flattened in the conversion.

**Example: Effects Array**

KuroTools format:
```json
{
    "FALCOM_PS4": {
        "game": "Kuro1",
        "schema": {
            "id": "uint",
            "effects": {
                "size": 5,
                "schema": {
                    "id": "uint",
                    "value1": "uint",
                    "value2": "uint",
                    "value3": "uint"
                }
            },
            "hp": "uint"
        }
    }
}
```

Converted to kurodlc format:
```json
{
    "info_comment": "Kuro1 - Converted from KuroTools",
    "table_header": "ItemTableData",
    "schema_length": 88,
    "schema": {
        "schema": "<I20II",
        "sch_len": 88,
        "keys": [
            "id",
            "eff1_id", "eff1_value1", "eff1_value2", "eff1_value3",
            "eff2_id", "eff2_value1", "eff2_value2", "eff2_value3",
            "eff3_id", "eff3_value1", "eff3_value2", "eff3_value3",
            "eff4_id", "eff4_value1", "eff4_value2", "eff4_value3",
            "eff5_id", "eff5_value1", "eff5_value2", "eff5_value3",
            "hp"
        ],
        "values": "nnnnnnnnnnnnnnnnnnnnnnn",
        "primary_key": "id"
    }
}
```

**Size Calculation:**
- `id` (uint) = 4 bytes
- `effects` (5 × 4 × uint) = 80 bytes
- `hp` (uint) = 4 bytes
- **Total** = 88 bytes

### Multi-Variant Schemas

Many KuroTools schemas have multiple variants for different game versions. Each variant is converted separately.

**Example: ItemTableData with 3 Variants**

KuroTools file structure:
```json
{
    "FALCOM_PS4": {
        "game": "Kuro1",
        "schema": { ... }  // Size: 248 bytes
    },
    "FALCOM_SWITCH": {
        "game": "Ys_X",
        "schema": { ... }  // Size: 176 bytes
    },
    "FALCOM_PC": {
        "game": "Sora1",
        "schema": { ... }  // Size: 232 bytes
    }
}
```

Converted to 3 separate entries:
1. ItemTableData (248) - Kuro1
2. ItemTableData (176) - Ys_X
3. ItemTableData (232) - Sora1

This allows the same table name to have different structures for different games.

### Real Conversion Examples

#### Example 1: Simple Schema - SkillPowerIcon

**Input (KuroTools):**
```json
{
    "FALCOM_PS5": {
        "game": "Kai",
        "schema": {
            "skill_power": "int",
            "icon_id": "int"
        }
    }
}
```

**Output (kurodlc):**
```json
{
    "info_comment": "Kai - Converted from KuroTools",
    "table_header": "SkillPowerIcon",
    "schema_length": 8,
    "schema": {
        "schema": "<2i",
        "sch_len": 8,
        "keys": ["skill_power", "icon_id"],
        "values": "nn",
        "primary_key": "skill_power"
    }
}
```

**Breakdown:**
- 2 × `int` (4 bytes each) = 8 bytes total
- Struct format: `<2i` (little-endian, 2 signed ints)
- Values: `nn` (both are numbers)
- Primary key: Auto-detected "skill_power" (contains "id" but not exact match, so uses first field with power/id pattern)

#### Example 2: Complex Schema - BattleBGM

**Input (KuroTools):**
```json
{
    "FALCOM_PS5": {
        "game": "Kuro2",
        "schema": {
            "id": "uint",
            "bgm_normal": "toffset",
            "bgm_advantage": "toffset",
            "bgm_disadvantage": "toffset",
            "bgm_boss": "toffset",
            "int1": "int",
            "int2": "int",
            "int3": "int"
        }
    }
}
```

**Output (kurodlc):**
```json
{
    "info_comment": "Kuro2 - Converted from KuroTools",
    "table_header": "BattleBGM",
    "schema_length": 56,
    "schema": {
        "schema": "<I4Q3i",
        "sch_len": 56,
        "keys": [
            "id", 
            "bgm_normal", 
            "bgm_advantage", 
            "bgm_disadvantage", 
            "bgm_boss",
            "int1", 
            "int2", 
            "int3"
        ],
        "values": "nttttnnn",
        "primary_key": "id"
    }
}
```

**Breakdown:**
- 1 × `uint` (I) = 4 bytes
- 4 × `toffset` (Q) = 32 bytes
- 3 × `int` (i) = 12 bytes
- **Total** = 56 bytes
- Values: `n` for numbers, `t` for text offsets

#### Example 3: Nested Array - HollowCoreEffParam

**Input (KuroTools):**
```json
{
    "FALCOM_PS4": {
        "game": "Kuro1",
        "schema": {
            "id": "ubyte",
            "name": "toffset",
            "params": {
                "size": 2,
                "schema": {
                    "value": "uint"
                }
            }
        }
    }
}
```

**Output (kurodlc):**
```json
{
    "info_comment": "Kuro1 - Converted from KuroTools",
    "table_header": "HollowCoreEffParam",
    "schema_length": 17,
    "schema": {
        "schema": "<BQ2I",
        "sch_len": 17,
        "keys": [
            "id",
            "name",
            "params_1_value",
            "params_2_value"
        ],
        "values": "ntnn",
        "primary_key": "id"
    }
}
```

**Breakdown:**
- 1 × `ubyte` (B) = 1 byte
- 1 × `toffset` (Q) = 8 bytes
- 2 × `uint` (I) = 8 bytes
- **Total** = 17 bytes

### Conversion Statistics

Based on actual conversion run:

| Metric | Count |
|--------|-------|
| Original kurodlc schemas | 39 |
| KuroTools schema files | 282 |
| Total variants converted | 343 |
| New schemas added | 305 |
| Final schema count | 344 |

**Game Distribution:**
- Kuro1: ~80 schemas
- Kuro2: ~85 schemas
- Kai: ~75 schemas
- Ys_X: ~60 schemas
- Sora1: ~40 schemas

**Table Categories:**
- Battle system: 25+
- Items/Equipment: 30+
- Skills/Abilities: 40+
- Quests/Story: 35+
- UI/Menus: 50+
- Maps/Navigation: 30+
- DLC/Shops: 20+
- Misc: 70+

### Duplicate Detection

The converter prevents duplicates by checking:
- `table_header` (table name)
- `schema_length` (size in bytes)

**Example:**
```
ItemTableData + 248 bytes = Kuro1 variant (kept if new)
ItemTableData + 176 bytes = Ys_X variant (kept if new)
ItemTableData + 248 bytes = Duplicate (skipped)
```

This means the same table can exist multiple times with different sizes (different game versions), but exact duplicates are avoided.

### Output File: kurodlc_schema_updated.json

The output file maintains the same format as the original kurodlc_schema.json:

```json
[
    {
        "info_comment": "Original entry",
        "table_header": "ItemTableData",
        "schema_length": 248,
        "schema": { ... }
    },
    {
        "info_comment": "Kuro2 - Converted from KuroTools",
        "table_header": "BattleBGM",
        "schema_length": 56,
        "schema": { ... }
    },
    ...
]
```

**File size:** ~200 KB (from ~25 KB original)

### Output File: conversion_report.txt

Detailed report with statistics and new schema list:

```
KuroTools Schema Conversion Report
======================================================================

Original schemas: 39
KuroTools schemas found: 282
Converted schemas: 343
New schemas added: 305
Total schemas: 344

New Schema Tables:
----------------------------------------------------------------------
  BattleBGM                    Size:   56  Game: Kuro2 - Converted from KuroTools
  SkillPowerIcon               Size:    8  Game: Kai - Converted from KuroTools
  HollowCoreEffParam           Size:   17  Game: Kuro1 - Converted from KuroTools
  ...
  (305 total new entries)
```

### Error Handling

The converter includes comprehensive error handling:

**Missing Headers Directory:**
```
Headers directory not found: /path/to/schemas/headers
  Found 0 schema file(s)

⚠ WARNING: No schemas found!
  Make sure the 'schemas' folder with 'headers' subfolder exists
  in the same directory as this script.

Press Enter to exit...
```

**Invalid JSON:**
```
Error loading ItemTableData.json: Expecting property name enclosed in double quotes
```

**Unknown Data Type:**
```
Error converting ItemTableData variant FALCOM_PS4: Unknown data type custom_type
```

**Conversion Issues:**
Problematic schemas are skipped with error messages, but conversion continues for other schemas.

### Advanced Workflows

#### Workflow 1: Initial Setup

```bash
# 1. Download KuroTools
git clone https://github.com/nnguyen259/KuroTools.git

# 2. Copy schemas folder
copy KuroTools\schemas .\schemas /s

# 3. Run conversion
python convert_kurotools_schemas.py

# 4. Review report
type conversion_report.txt

# 5. Backup and replace
copy kurodlc_schema.json kurodlc_schema.json.backup
copy kurodlc_schema_updated.json kurodlc_schema.json
```

#### Workflow 2: Update Schemas

When KuroTools adds new schemas:

```bash
# 1. Update KuroTools schemas
cd KuroTools
git pull
cd ..

# 2. Update local schemas
robocopy KuroTools\schemas .\schemas /s /mir

# 3. Re-run converter
python convert_kurotools_schemas.py

# 4. Check what's new
type conversion_report.txt

# 5. Replace if satisfied
copy kurodlc_schema_updated.json kurodlc_schema.json
```

#### Workflow 3: Selective Merge

If you have custom schemas and want to add only specific new ones:

```bash
# 1. Run conversion
python convert_kurotools_schemas.py

# 2. Open both files
# - kurodlc_schema.json (your custom version)
# - kurodlc_schema_updated.json (merged version)

# 3. Check conversion_report.txt for new entries

# 4. Manually copy desired new schemas from updated file to your custom file

# 5. Validate JSON syntax
python -m json.tool kurodlc_schema.json
```

#### Workflow 4: CI/CD Integration

Automate schema updates in CI/CD pipeline:

```bash
#!/bin/bash
# update_schemas.sh

# Clone/update KuroTools
if [ ! -d "KuroTools" ]; then
    git clone https://github.com/nnguyen259/KuroTools.git
else
    cd KuroTools && git pull && cd ..
fi

# Copy schemas
cp -r KuroTools/schemas ./schemas

# Backup existing schema
cp kurodlc_schema.json kurodlc_schema.json.backup

# Run conversion
python convert_kurotools_schemas.py

# Check if conversion succeeded
if [ -f "kurodlc_schema_updated.json" ]; then
    mv kurodlc_schema_updated.json kurodlc_schema.json
    echo "✓ Schema updated successfully"
    cat conversion_report.txt
else
    echo "✗ Schema conversion failed"
    exit 1
fi
```

### Troubleshooting

#### Problem: Schema Size Mismatch

**Symptom:** Converted schema has wrong size when tested with actual TBL file

**Cause:** Complex nested structures or special types not properly handled

**Solution:**
1. Check KuroTools schema for complex nested structures
2. Manually verify size calculation:
   ```
   Each ubyte/byte = 1 byte
   Each ushort/short = 2 bytes
   Each uint/int/float = 4 bytes
   Each ulong/long/toffset = 8 bytes
   Nested structure = size × (sum of inner fields)
   ```
3. Compare with actual TBL file entry size
4. Manually edit schema if needed

**Example Fix:**
```json
// If auto-calculated size is 100 but actual is 104:
"schema_length": 100,  // Change to:
"schema_length": 104,
```

#### Problem: Missing Primary Keys

**Symptom:** Schema works but lacks primary key for duplicate detection

**Solution:** Manually add primary key:
```json
{
    "schema": {
        ...
        "primary_key": "id"  // Add this line
    }
}
```

Or for composite keys:
```json
"primary_key": ["id", "category", "subcategory"]
```

#### Problem: Nested Structure Not Flattened

**Symptom:** Schema has nested array but keys list doesn't show individual elements

**Cause:** Converter missed nested structure

**Solution:** Manually flatten in converted schema:
```json
// Instead of:
"keys": ["id", "effects", "hp"]

// Use:
"keys": ["id", "eff1_id", "eff1_val", "eff2_id", "eff2_val", "hp"]
```

### Best Practices

1. **Always Backup**
   ```bash
   copy kurodlc_schema.json kurodlc_schema.json.backup
   ```

2. **Review Report**
   - Check conversion_report.txt for new schemas
   - Verify game assignments are correct
   - Look for any error messages

3. **Test Incrementally**
   - Don't replace schema immediately
   - Test with a few TBL files first
   - Verify sizes match actual file entries

4. **Keep KuroTools Updated**
   - Periodically check for KuroTools updates
   - Re-run converter to get new schemas

5. **Document Custom Changes**
   - If you manually edit converted schemas, document why
   - Use comments in schema file:
     ```json
     {
         "info_comment": "Kai - Converted from KuroTools - MANUALLY ADJUSTED size",
         ...
     }
     ```

6. **Version Control**
   - Use git to track schema changes
   - Commit before and after conversion
   - Tag stable versions

### Schema Coverage After Conversion

**Before Conversion (39 schemas):**
- Basic tables: ItemTableData, DLCTableData
- Some shop/costume tables
- Limited game-specific support

**After Conversion (344 schemas):**

**Battle System:**
- BattleBGM, BattleLevelField, BattleLevelTurn
- BattleEnemyLevelAdjust, BattleSCraftDamageRatio
- SkillParam, SkillLevelParam, SkillRangeData

**Items & Equipment:**
- Multiple ItemTableData variants (Kuro1, Kuro2, Ys X, Sky)
- CostumeParam, CostumeTable, CostumeAttachTable
- QuartzParam, QuartzLineParam

**Quests & Story:**
- QuestText, QuestTitle, QuestRank
- StoryMissionTable, TimeChartData
- EventTable, EventGroupData

**Shops & Trading:**
- ShopItem, ShopInfo, ProductInfo
- RecipeTableData, TradeItem

**Maps & Navigation:**
- MapInfoTable, MapJumpAreaData, MapSeaTable
- FastTravelTable, LookPointTable

**UI & Menus:**
- HelpTableData, HelpTitle, HelpPage
- NoteMenus (various), SettingMenuData

**Character & Progression:**
- ChrDataParam, CharaLibData, StatusParam
- SupportAbilityParam, ConnectBonusParam

**Minigames:**
- FishingSpot, FishParam, FishInfo
- BlackJackHelp, PokerHelp, CardData

**DLC & Platform:**
- DLCTable, DLCTableData, SteamDlcTableData

**Game-Specific:**
- HollowCore* tables (Kuro series)
- RecaptureIsland* tables (Ys X)
- AbordageTable (Ys X)

### Schema Format Reference

For reference, here's the complete schema format used by kurodlc_lib.py:

```json
{
    "info_comment": "Human-readable description",
    "table_header": "TableName",
    "schema_length": 123,
    "schema": {
        "schema": "<format_string>",
        "sch_len": 123,
        "keys": ["field1", "field2", ...],
        "values": "nttann...",
        "primary_key": "id" or ["id", "category"]
    }
}
```

**Fields:**
- `info_comment`: Description (usually game name + source)
- `table_header`: Table name (must match TBL section name)
- `schema_length`: Total size of one entry in bytes
- `schema.schema`: Python struct format string
- `schema.sch_len`: Same as schema_length (redundant but required)
- `schema.keys`: List of field names
- `schema.values`: String of value types (n/t/a/b for each field)
- `schema.primary_key`: Field(s) used for duplicate detection

**Struct Format String:**
- `<` = Little-endian byte order
- `B` = unsigned byte (8-bit)
- `b` = signed byte (8-bit)
- `H` = unsigned short (16-bit)
- `h` = signed short (16-bit)
- `I` = unsigned int (32-bit)
- `i` = signed int (32-bit)
- `Q` = unsigned long long (64-bit)
- `q` = signed long long (64-bit)
- `f` = float (32-bit)

Numbers can prefix types: `2I` = two unsigned ints = `II`

**Value Types:**
- `n` = Number (decoded as Python int or float)
- `t` = Text (8-byte offset to null-terminated string)
- `a` = Array of 32-bit values (offset + count = 12 bytes)
- `b` = Array of 16-bit values (offset + count = 12 bytes)

---

## Summary

The `convert_kurotools_schemas.py` script is a powerful tool for expanding TBL file support in the KuroDLC toolkit. It automates the tedious process of converting schema definitions from one format to another, saving hours of manual work and reducing errors.

**Key Benefits:**
- ✅ Converts 280+ schemas in seconds
- ✅ Expands coverage from 39 to 344+ tables
- ✅ Supports 5 different games
- ✅ Automatic type mapping and size calculation
- ✅ Prevents duplicates
- ✅ Detailed reporting

**Use Cases:**
- Adding support for new TBL files
- Updating schemas when games are patched
- Supporting multiple game versions
- Building comprehensive modding toolkits

For most users, running the converter once with default settings is sufficient. Advanced users can use the detailed reports and manual editing capabilities for fine-tuned control.

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

---

## New Analysis Tools (v1.0) ⭐

### visualize_id_allocation.py

**Version:** v1.0  
**Purpose:** Analyze and visualize ID allocation patterns to identify free ID ranges and fragmentation

#### All Parameters

```
visualize_id_allocation.py [options]

OPTIONS:
  --source=TYPE         Force specific source type
                        Available: json, tbl, original, p3a, zzz
                        Default: Auto-detect from available sources
                        
  --no-interactive      Auto-select first source if multiple found
                        Useful for automated scripts
                        
  --keep-extracted      Keep temporary extracted files from P3A
                        Default: Delete after use
                        
  --format=FORMAT       Output format selection
                        Available: console, html, both
                        Default: both
                        
  --block-size=N        Block size for console visualization
                        Default: 50 (recommended: 25-100)
                        Larger blocks = better overview
                        Smaller blocks = more detail
                        
  --output=FILE         Custom HTML output filename
                        Default: id_allocation_map.html
                        Example: --output=my_report.html
                        
  --help                Show help message and exit

SOURCES (automatically detected):
  JSON sources:
    - t_item.json
    
  TBL sources (requires kurodlc_lib.py):
    - t_item.tbl
    - t_item.tbl.original
    
  P3A sources (requires kurodlc_lib.py + dependencies):
    - script_en.p3a / script_eng.p3a
    - zzz_combined_tables.p3a
    (automatically extracts t_item.tbl)

OUTPUT:
  Console Format:
    - Color-coded block visualization
    - Statistics table
    - Gap analysis
    - Fragmentation metrics
    
  HTML Format:
    - Interactive grid map (100 columns)
    - Hover tooltips with ID numbers
    - Search functionality
    - Free blocks table (sortable)
    - Full statistics dashboard
    - Responsive design

STATISTICS PROVIDED:
  - Engine Range (1-5000)
  - Highest Used ID
  - Occupied/Free ID counts and percentages
  - Average Gap Size
  - Fragmentation Index (0.0-1.0)
  - Largest Free Block (start-end range)
  - Total Free Blocks count
```

#### Examples

**Example 1: Basic Analysis**

```bash
python visualize_id_allocation.py

# Output:
# Loading item data...
# Loaded 2116 items from: t_item.json
#
# Statistics:
#   Engine Range:       1 - 5000
#   Highest Used ID:    4921
#   Occupied IDs:       2116 / 5000  (42.3%)
#   Free IDs:           2884 / 5000  (57.7%)
#   Fragmentation:      0.73 (High)
#   Largest Free Block: 79 IDs (4922-5000)
#
# HTML report generated: id_allocation_map.html
```

**Example 2: Console Only (CI/CD)**

```bash
python visualize_id_allocation.py --format=console

# Generates only console output, no HTML file
# Useful for automated builds where HTML isn't needed
```

**Example 3: Custom HTML Report**

```bash
python visualize_id_allocation.py --format=html --output=project_allocation.html

# Generates only HTML report with custom filename
# Good for sharing with team
```

**Example 4: Larger Blocks for Overview**

```bash
python visualize_id_allocation.py --block-size=100

# Each block represents 100 IDs instead of 50
# Better for seeing overall patterns
```

**Example 5: Force Specific Source**

```bash
python visualize_id_allocation.py --source=json

# Forces use of t_item.json
# Skips other sources even if available
```

#### Real Data Example

**Input:** t_item.json with 2116 items spread across IDs 1-4921

**Console Output:**
```
ID Allocation Map (Block Size: 50)
═══════════════════════════════════════════════════════════

    0: ████████████████████████████████████████████████ [  0 -  49]  100.0%
   50: ████████████████████████████████████████████████ [ 50 -  99]  100.0%
  100: ████████████████████████████████████████████████ [100 - 149]  100.0%
  150: ███████████████████████████████████░░░░░░░░░░░░░ [150 - 199]   76.0%
  200: ████████████████████████████████████████████████ [200 - 249]  100.0%
  250: ████████████████████████████████████████████████ [250 - 299]  100.0%
  300: ████████████████████████████████████████████████ [300 - 349]  100.0%
  350: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ [350 - 399]    0.0%  ✨
  400: ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ [400 - 449]   28.0%
  ...
 3500: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ [3500-3549]    0.0%  ✨
 3550: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ [3550-3599]    0.0%  ✨
 3600: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ [3600-3649]    0.0%  ✨
  ...
 4950: ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ [4950-4999]   12.0%

Legend: █ Occupied  ░ Free  ✨ Large free block detected
```

**HTML Output Features:**
- Interactive 100×50 grid (5000 cells total)
- Green cells for occupied IDs
- Gray cells for free IDs
- Hover shows exact ID number
- Search box to jump to specific IDs
- Sortable free blocks table

#### Use Cases

**1. Pre-Mod Planning:**
```bash
# Before creating a new mod
python visualize_id_allocation.py

# Review HTML report to find suitable ID ranges
# Example findings:
# - IDs 3500-3650: 151 free IDs (perfect for large weapon pack)
# - IDs 1200-1245: 46 free IDs (good for armor set)
```

**2. Team Coordination:**
```bash
# Generate report for team
python visualize_id_allocation.py --output=team_allocation.html

# Share HTML file
# Team members coordinate ID usage:
# - Modder A: 3500-3599 (weapons)
# - Modder B: 3600-3699 (armor)
# - Modder C: 3700-3799 (accessories)
```

**3. Fragmentation Analysis:**
```bash
# Check fragmentation before major update
python visualize_id_allocation.py

# Fragmentation Index:
# - 0.0-0.3: Low (few large gaps) - Ideal
# - 0.4-0.6: Medium (mixed) - Acceptable
# - 0.7-1.0: High (many small gaps) - Problematic
```

**4. CI/CD Integration:**
```bash
# Automated build script
python visualize_id_allocation.py --format=console --no-interactive > allocation_report.txt

# Parse output for available ID count
# Fail build if insufficient free IDs
```

---

### find_all_names.py

**Version:** v1.0  
**Purpose:** Search and browse character names from game data with intelligent filtering

#### All Parameters

```
find_all_names.py [search_query] [options]

ARGUMENTS:
  search_query          Optional search query with modes:
  
  AUTO-DETECT MODE (no prefix):
    123                 Number → searches by character ID
    van                 Text → searches in character names
    
  EXPLICIT MODES (with prefix):
    id:123              Search by exact character ID
    name:van            Search in character names
    name:123            Search "123" in names (not as ID!)
    full_name:arkride   Search in full names
    model:chr0100       Search in model names

OPTIONS:
  --source=TYPE         Force specific source type
                        Available: json, tbl, original, p3a, zzz
                        Default: Auto-detect from available sources
                        
  --no-interactive      Auto-select first source if multiple found
                        
  --keep-extracted      Keep temporary extracted files from P3A
                        
  --show-full           Show full names in output
                        Adds full_name column to results
                        
  --show-model          Show model names in output
                        Adds model column to results
                        
  --help                Show help message and exit

SOURCES (automatically detected):
  JSON sources:
    - t_name.json
    
  TBL sources (requires kurodlc_lib.py):
    - t_name.tbl
    - t_name.tbl.original
    
  P3A sources (requires kurodlc_lib.py + dependencies):
    - script_en.p3a / script_eng.p3a
    - zzz_combined_tables.p3a
    (automatically extracts t_name.tbl)

OUTPUT FIELDS:
  ID                    Character ID number
  Name                  Character display name
  Full Name             Complete name (with --show-full)
  Model                 3D model identifier (with --show-model)
```

#### Examples

**Example 1: List All Characters**

```bash
python find_all_names.py

# Output:
# Loading character name data...
# Loaded 500 characters from: t_name.json
#
#   0 : (None)
#   1 : Van
#   2 : Agnes
#   3 : Feri
#   ...
#
# Total: 500 character(s)
```

**Example 2: Search by Name (Auto-Detect)**

```bash
python find_all_names.py van

# Output:
# # Auto-detected name search for 'van'
#
# Loading character name data...
# Loaded 500 characters from: t_name.json
#
# 100 : Van
# 101 : Van Arkride
# 225 : Vandaal
#
# Total: 3 character(s)
```

**Example 3: Search by ID (Auto-Detect)**

```bash
python find_all_names.py 100

# Output:
# # Auto-detected ID search for '100'
# # Use 'name:100' to search for '100' in character names instead
#
# Loading character name data...
# Loaded 500 characters from: t_name.json
#
# 100 : Van
#
# Total: 1 character(s)
```

**Example 4: Explicit Name Search (for numbers)**

```bash
python find_all_names.py name:100

# Searches for "100" in character names
# Useful when character names contain numbers
```

**Example 5: Search with Full Names and Models**

```bash
python find_all_names.py van --show-full --show-model

# Output:
# 100 : Van              | Van Arkride           | chr0100_01
# 101 : Van Arkride      | Van Arkride (Full)    | chr0100_02
# 225 : Vandaal          | Vandaal               | chr0225
#
# Total: 3 character(s)
```

**Example 6: Search by Model**

```bash
python find_all_names.py model:chr0100

# Output:
# 100 : Van              | chr0100_01
# 101 : Van Arkride      | chr0100_02
#
# Total: 2 character(s)
```

**Example 7: Search in Full Names**

```bash
python find_all_names.py full_name:arkride

# Searches in full_name field
# Finds all characters with "arkride" in their full name
```

**Example 8: Force Specific Source**

```bash
python find_all_names.py van --source=json

# Forces use of t_name.json
# Ignores TBL/P3A sources
```

#### Real Data Example

**Input:** t_name.json with character data

```json
{
  "data": [
    {
      "name": "NameTableData",
      "data": [
        {
          "character_id": 100,
          "name": "Van",
          "full_name": "Van Arkride",
          "model": "chr0100_01"
        },
        {
          "character_id": 101,
          "name": "Agnes",
          "full_name": "Agnes Claudel",
          "model": "chr0101"
        }
      ]
    }
  ]
}
```

**Query 1: Find Van**
```bash
$ python find_all_names.py van

100 : Van

Total: 1 character(s)
```

**Query 2: Find with Details**
```bash
$ python find_all_names.py van --show-full --show-model

100 : Van | Van Arkride | chr0100_01

Total: 1 character(s)
```

**Query 3: Find by ID**
```bash
$ python find_all_names.py 101 --show-full

101 : Agnes | Agnes Claudel

Total: 1 character(s)
```

#### Use Cases

**1. Character ID Lookup for Scripting:**
```bash
# Need character ID for event script
python find_all_names.py "van"

# Output: 100 : Van
# Use ID 100 in your script
```

**2. Model Reference for 3D Work:**
```bash
# Check which model a character uses
python find_all_names.py id:100 --show-model

# Output: 100 : Van | chr0100_01
# Use chr0100_01 for custom model edits
```

**3. Character Database Export:**
```bash
# Export full character list
python find_all_names.py --show-full --show-model > characters.txt

# Creates complete character reference file
```

**4. Find All Variants:**
```bash
# Find all variations of a character
python find_all_names.py model:chr0100

# Shows all entries using chr0100 models
# Useful for finding costume variants
```

**5. Multi-Language Name Lookup:**
```bash
# Find character by localized name
python find_all_names.py full_name:arkride

# Works across different name fields
```

#### Important Notes

**Auto-Detection Behavior:**
- Pure numbers (e.g., `100`) → ID search
- Text or mixed (e.g., `van`, `chr100`) → Name search
- Use explicit prefixes to override auto-detection

**Search Tips:**
- Searches are case-insensitive
- Partial matches work (e.g., `van` finds `Van`, `Vandaal`)
- Use `--show-full` and `--show-model` for complete information
- Combine with grep/findstr for advanced filtering

**Output Format:**
- Aligned columns for clean display
- Sorted by character ID (ascending)
- Non-numeric IDs sorted last

---

## Visual Guides

For detailed visual examples of the ID allocation map, see:
- **VISUALIZATION_GUIDE.md** - Complete visual guide with examples
- **example_id_allocation_map.html** - Real HTML output example

These guides include:
- Statistics dashboard layout
- Visual ID map examples
- Free blocks table format
- Search functionality examples
- Usage tips and best practices

