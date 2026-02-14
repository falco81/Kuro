# Advanced Documentation - KuroDLC Modding Toolkit

This section provides comprehensive, in-depth documentation for advanced users, including all parameters, real data examples, data structure specifications, and detailed workflows.

## 📋 Table of Contents

- [Script Reference](#script-reference)
  - [resolve_id_conflicts_in_kurodlc.py](#resolve_id_conflicts_in_kurodlcpy)
  - [shops_find_unique_item_id_from_kurodlc.py](#shops_find_unique_item_id_from_kurodlcpy)
  - [shops_create.py](#shops_createpy)
  - [kurodlc_add_mdl.py](#kurodlc_add_mdlpy) ⭐ NEW
  - [visualize_id_allocation.py](#visualize_id_allocationpy)
  - [convert_kurotools_schemas.py](#convert_kurotools_schemaspy)
  - [find_all_items.py](#find_all_itemspy)
  - [find_all_names.py](#find_all_namespy)
  - [find_all_shops.py](#find_all_shopspy)
  - [find_unique_item_id_for_t_costumes.py](#find_unique_item_id_for_t_costumespy)
  - [find_unique_item_id_for_t_item_category.py](#find_unique_item_id_for_t_item_categorypy)
  - [find_unique_item_id_from_kurodlc.py](#find_unique_item_id_from_kurodlcpy)
- [3D Model Viewer Scripts](#3d-model-viewer-scripts)
  - [viewer.py](#viewerpy)
  - [viewer_mdl.py](#viewer_mdlpy)
  - [viewer_mdl_optimized.py](#viewer_mdl_optimizedpy)
  - [viewer_mdl_window.py](#viewer_mdl_windowpy)
  - [viewer_mdl_textured.py](#viewer_mdl_texturedpy)
  - [viewer_mdl_textured_anim.py](#viewer_mdl_textured_animpy) ⭐ MAIN VIEWER
  - [viewer_mdl_textured_scene.py](#viewer_mdl_textured_scenepy) ⭐ SCENE VIEWER
- [Library Files](#library-files)
  - [kurodlc_lib.py](#kurodlc_libpy)
  - [p3a_lib.py](#p3a_libpy)
  - [kuro_mdl_export_meshes.py](#kuro_mdl_export_meshespy)
  - [lib_texture_loader.py](#lib_texture_loaderpy)
  - [lib_fmtibvb.py](#lib_fmtibvbpy)
- [Data Structure Specifications](#data-structure-specifications)
- [Real Data Examples](#real-data-examples)
- [Export/Import Formats](#exportimport-formats)
- [Log Files](#log-files)

---

## Script Reference

### resolve_id_conflicts_in_kurodlc.py

**Version:** v2.7.1  
**Purpose:** Detect and resolve ID conflicts between DLC mods and game data

#### How It Works

The script operates in two phases. In the **detection phase**, it loads a game item database (from any supported source) and builds a dictionary mapping every used item ID to its name. It then scans all `.kurodlc.json` files in the current directory, extracting item IDs from four sections: `CostumeParam` (field `item_id`), `ItemTableData` (field `id`), `DLCTableData` (nested `items` arrays), and `ShopItem` (field `item_id`). Each extracted DLC ID is checked against the game database — if the ID exists in game data, it is flagged as `[BAD]` (conflict), otherwise `[OK]` (safe).

In the **repair phase** (mode `repair`), the script uses a **smart ID assignment algorithm** (v2.7) to find replacement IDs for every conflict. The algorithm works within range 1–5000 (safe game limit). It first attempts to find a **continuous block** of free IDs starting from the midpoint (2500) and searching outward in both directions. If fragmentation prevents a continuous block, it falls back to **scattered search**, collecting individual free IDs using the same middle-out strategy. Once replacement IDs are determined, the script can either preview the mapping, export it to a JSON file for manual editing, or apply it directly with `--apply`.

When applying changes, the script performs ID replacement across **all four sections simultaneously** — it updates `item_id` in CostumeParam, `id` in ItemTableData, values in `DLCTableData.items` arrays, and `item_id` in ShopItem. This ensures internal consistency within each `.kurodlc.json` file. Timestamped backups are always created before modification.

**Source detection** works by scanning the current directory for data files in priority order: `t_item.json` (direct JSON), `t_item.tbl.original` (original binary table), `t_item.tbl` (binary table), then P3A archives (`script_en.p3a`, `script_eng.p3a`, `zzz_combined_tables.p3a`). For P3A sources, the script extracts `t_item.tbl.original` to a temporary `.tmp` file using `p3a_lib` and `kurodlc_lib`, then loads item data from it. Multiple sources trigger interactive selection (or auto-select with `--no-interactive`).

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

**Example 2: Repair with Apply**

```bash
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
Log saved to: id_conflict_repair_20260131_143022.log
```

**Example 3: Export → Edit → Import**

```bash
# Step 1: Export mapping
python resolve_id_conflicts_in_kurodlc.py repair --export --export-name=myfix

# Step 2: Edit id_mapping_myfix.json manually

# Step 3: Import and apply
python resolve_id_conflicts_in_kurodlc.py repair --import --mapping-file=id_mapping_myfix.json --apply
```

**Generated Mapping File:**
```json
{
  "source_file": "my_costume_mod.kurodlc.json",
  "timestamp": "2026-01-31 14:30:22",
  "game_database": "t_item.json",
  "mappings": {
    "310": 2500,
    "311": 2501
  },
  "conflicts": [
    {
      "old_id": 310,
      "new_id": 2500,
      "reason": "Conflict with game item: Earth Sepith"
    }
  ]
}
```

#### Real Data Scenarios

**Scenario 1: Sepith ID Conflicts**

Your DLC uses IDs 310-317, but these are used by the game for Sepith items:

```
Game Database (t_item.json):
  310: Earth Sepith
  311: Water Sepith
  312: Fire Sepith
  313: Wind Sepith
```

**Solution:**
```bash
python resolve_id_conflicts_in_kurodlc.py repair --apply
```

Result: IDs reassigned to 2500-2503 (safe range, middle-out).

**Scenario 2: Large DLC with Multiple Conflicts**

50 costume items with IDs 100-149. IDs 100-120 overlap with game items, 121-149 are safe.

The smart algorithm will:
- Reassign only 100-120 → 2500-2520
- Keep 121-149 unchanged (no conflicts)

---

### shops_find_unique_item_id_from_kurodlc.py

**Version:** v2.2  
**Purpose:** Extract item IDs from DLC files and generate template configurations

#### How It Works

The script reads a `.kurodlc.json` file and extracts item IDs from one or more sections based on the specified extraction mode. It understands the internal structure of kurodlc files, where IDs appear in different fields depending on the section: `item_id` in CostumeParam, `id` in ItemTableData, nested values in `DLCTableData.items` arrays, and `item_id` in ShopItem.

**Extraction logic** works by parsing JSON and iterating through requested sections. When mode is `all` (default), it collects IDs from every section and deduplicates them. Combined modes like `costume+item` merge results from multiple sections. The final output to stdout is a Python-style list of sorted unique IDs, while stderr receives a summary showing per-section counts.

**Validation** (v2.2) recognizes two file types: full DLC files (containing CostumeParam/ItemTableData/DLCTableData) and shop-only files (containing only ShopItem). The `is_valid_kurodlc_structure()` function checks for the presence of at least one recognized section. This allows files like `Daybreak2CostumeShop.kurodlc.json` that only define shop assignments to be processed.

**Template generation** (v2.0+) is an advanced feature that creates a ready-to-use config file for `shops_create.py`. When invoked with `--generate-template`, the script first extracts item IDs from the specified source section (costume, item, dlc, or all), then determines shop IDs either from the existing ShopItem section (auto-detect), from `--shop-ids`, or from `--default-shop-ids`. It also extracts the existing template structure from the first ShopItem entry if available, falling back to a hardcoded default. The generated JSON file contains `item_ids`, `shop_ids`, `template`, and a `_comment` section with usage instructions.

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
                      
  --no-interactive    Do not prompt for user input
                      Must be used with --shop-ids or --default-shop-ids
                      
  --output=<file>     Custom output filename for generated template
                      Default: template_<input_filename>.json
```

#### Examples

**Basic ID Extraction:**
```bash
python shops_find_unique_item_id_from_kurodlc.py my_costume_mod.kurodlc.json
# stdout: [3596, 3597, 3598]
```

**Shop-only File (v2.2):**
```bash
python shops_find_unique_item_id_from_kurodlc.py Daybreak2CostumeShop.kurodlc.json shop
# Extracts item_ids from ShopItem section only — no CostumeParam required
```

**Generate Template with Auto-Detect:**
```bash
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json --generate-template costume
# Creates: template_my_mod.kurodlc.json
```

**Generated Template:**
```json
{
  "_comment": [
    "Template config file generated by shops_find_unique_item_id_from_kurodlc.py v2.2",
    "Source file: my_mod.kurodlc.json",
    "INSTRUCTIONS:",
    "1. Review and modify shop_ids as needed",
    "2. Run: python shops_create.py <this_file>"
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

---

### shops_create.py

**Version:** v2.0  
**Purpose:** Generate bulk shop assignments from template configurations

#### How It Works

The script reads a JSON config file containing three key fields: `item_ids` (array of item IDs), `shop_ids` (array of shop IDs), and optionally `template` (structure for each generated entry). It then generates the **Cartesian product** of all items × all shops — for each combination, it creates a new entry by substituting variables in the template.

**Variable substitution** is performed recursively across the entire template structure (including nested objects). The function `substitute_variables()` walks through every value in the template dict: string values like `"${shop_id}"` are replaced with the current integer value; nested dicts and lists are processed recursively. Four variables are available:

- `${shop_id}` → current shop ID
- `${item_id}` → current item ID  
- `${index}` → 0-based index of the current entry
- `${count}` → total number of entries being generated

**Template validation** checks that the template contains at least `${shop_id}` and `${item_id}` placeholders. If no template is provided, a hardcoded default matching the standard ShopItem structure is used (backward compatible with v1.0 configs).

The output section name defaults to `"ShopItem"` but can be overridden with the `output_section` config key. Output is written to `output_<input_filename>`.

#### All Parameters

```
shops_create.py <template_file>

ARGUMENTS:
  <template_file>     Template config file (JSON)
                      
TEMPLATE FORMAT:
  {
    "item_ids": [<list of item IDs>],
    "shop_ids": [<list of shop IDs>],
    "template": {<shop item structure>},             // optional
    "output_section": "<section_name>"               // optional, default: "ShopItem"
  }
  
TEMPLATE VARIABLES:
  ${shop_id}          Current shop ID (integer)
  ${item_id}          Current item ID (integer)
  ${index}            Entry index, 0-based (integer)
  ${count}            Total number of entries (integer)
  
OUTPUT:
  Creates: output_<template_file>
  Contains: Complete shop assignments ready to copy into .kurodlc.json
```

#### Example

**Input:** `template_my_mod.kurodlc.json`
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

```bash
python shops_create.py template_my_mod.kurodlc.json
```

**Result:** 3 items × 3 shops = 9 shop assignments in `output_template_my_mod.kurodlc.json`.

---

### kurodlc_add_mdl.py

**Version:** v2.1  
**Purpose:** Automatically generate complete DLC entries for new MDL model files

#### How It Works

This script automates the process of adding new costume models to a `.kurodlc.json` file. It solves the problem of manually creating four interconnected data entries for each MDL file — a tedious and error-prone process.

**Step 1: MDL Scanning.** The script calls `scan_mdl_files()` which uses `glob` to find all `*.mdl` files in the same directory as the target `.kurodlc.json`. It extracts just the filename stems (without `.mdl` extension). It then compares these against `get_existing_mdl_names()`, which reads the `CostumeParam` section and collects all `mdl_name` values. Only MDLs not already present in the config are flagged as "new".

**Step 2: Data Source Selection.** The script needs two game data tables: `t_name` (for character identification) and `t_item` (for used ID detection). The `detect_all_sources()` function scans for files matching both prefixes simultaneously. For JSON sources, it requires both `t_name.json` and `t_item.json` to exist. For P3A archives, a single archive contains all tables. Sources where files are missing are marked as invalid. If multiple valid sources exist, interactive selection is shown (unless `--no-interactive`).

**Step 3: Character Resolution.** For each new MDL, the script calls `extract_chr_prefix()` which uses regex `(chr\d+)` to extract the character prefix from filenames like `chr5001_c02aa` → `chr5001`, or `q_chr5001_c56q` → `chr5001`. This prefix is looked up in the character map built by `build_char_map()`, which loads t_name data and maps each `model` field's chr prefix to its `char_id` and `name`. If a character can't be resolved automatically, interactive mode prompts the user to enter `char_restrict` and character name manually. With `--no-interactive`, unresolved MDLs are skipped.

**Step 4: Smart ID Assignment.** The script builds a complete set of used IDs by calling `collect_all_used_ids()`, which merges IDs from three sources: the game item database (t_item), all `.kurodlc.json` files in the directory (scanning CostumeParam, ItemTableData, DLCTableData, and ShopItem sections), and internal DLC IDs. It then calls `find_available_ids_in_range()` (the same algorithm as `resolve_id_conflicts_in_kurodlc.py`) to find free IDs. The algorithm tries a continuous block from the middle first, then falls back to scattered search.

**Step 5: Entry Generation.** For each resolved MDL, the script generates four types of entries:

- **CostumeParam** — Created by `make_costume_entry()`, which uses the first existing CostumeParam entry as a template (or a hardcoded default). It sets `item_id` to the new ID, `mdl_name` to the MDL filename, and `char_restrict` to the resolved character ID.
- **ItemTableData** — Created by `make_item_entry()`, using the first existing ItemTableData entry as template. Sets `id` to the new ID, `name` to `"<CharName> generated <mdl_name>"` (a placeholder for manual editing), `category` to 17 (costume category), and matches `char_restrict`.
- **ShopItem** — Created by `make_shop_entries()` for each shop ID. Shop IDs are auto-detected from existing ShopItem entries or fall back to `[21, 22, 248, 258]` (common Kuro 2 costume shops). Can be overridden with `--shop-ids`.
- **DLCTableData** — New item IDs are appended to the `items` array of the first existing DLC record. If no record exists, a minimal one is created.

**Step 6: Write.** By default the script runs in **dry-run mode** — it shows exactly what would change but writes nothing. Only with `--apply` are changes actually written. Before writing, a timestamped backup (`_YYYYMMDD_HHMMSS.bak`) is created (unless `--no-backup`). The `--no-ascii-escape` flag writes UTF-8 directly (e.g., `Agnès` instead of `Agn\u00e8s`).

#### All Parameters

```
kurodlc_add_mdl.py <file.kurodlc.json> [options]

ARGUMENTS:
  <file>              Target .kurodlc.json file (required)

OPTIONS:
  --apply             Apply changes (without this, runs in dry-run mode)
  --dry-run           Explicit dry-run (default behavior, no changes written)
  --shop-ids=1,2,3    Override shop IDs (default: auto-detect from file)
  --min-id=N          Minimum ID for search range (default: 1)
  --max-id=N          Maximum ID for search range (default: 5000)
  --no-interactive    Auto-select sources without prompting
  --no-backup         Skip backup creation when applying
  --no-ascii-escape   Write UTF-8 directly (e.g. Agnès instead of Agn\u00e8s)
  --help              Show help message

REQUIRED FILES (in same directory as kurodlc.json):
  *.mdl               MDL model files to add
  t_name source       One of: t_name.json, t_name.tbl, script_en.p3a,
                      script_eng.p3a, zzz_combined_tables.p3a
  t_item source       One of: t_item.json, t_item.tbl, script_en.p3a,
                      script_eng.p3a, zzz_combined_tables.p3a
```

#### Examples

**Preview Changes (dry-run):**
```bash
python kurodlc_add_mdl.py FalcoDLC.kurodlc.json
```

**Sample Output:**
```
Loaded: FalcoDLC.kurodlc.json
  CostumeParam:  12 entries
  ItemTableData: 12 entries
  DLCTableData:  1 entries
  ShopItem:      48 entries

Found 15 .mdl file(s) in directory
New .mdl files to add: 3
  + chr5001_c02aa
  + chr5302_c210
  + chr5100_c03bb

Loading character data (t_name)...
Character map: 156 characters loaded

Loading game item data (t_item)...
Game items loaded: 2116 IDs
Total used IDs (game + all DLCs): 2180

Searching for 3 available ID(s) in range [1, 5000]...
Found 3 IDs: 2501-2503 (continuous block)

============================================================
Generating entries
Shop IDs: [21, 22, 248, 258]
============================================================

  item_id=2501  char=  1  Van                   mdl=chr5001_c02aa
  item_id=2502  char= 32  Elaine                mdl=chr5302_c210
  item_id=2503  char= 10  Agnes                 mdl=chr5100_c03bb

============================================================
Summary of changes:
============================================================
  CostumeParam:  +3 entries
  ItemTableData: +3 entries
  DLCTableData:  +3 item(s) added to existing record
  ShopItem:      +12 entries (3 items x 4 shops)
============================================================

[DRY RUN] No files modified. Use --apply to write changes.
```

**Apply with Custom Range:**
```bash
python kurodlc_add_mdl.py FalcoDLC.kurodlc.json --apply --min-id=3000 --max-id=4000
```

**Unresolved Character Handling:**
```
Warning: Could not resolve character for 1 MDL(s):
  ? custom_model_xyz  (prefix: none)

You can manually assign char_restrict for unresolved MDLs.
Enter char_restrict value, or press Enter to skip, 'q' to abort:

  custom_model_xyz char_restrict = 1
  custom_model_xyz character name = Van
```

---

### visualize_id_allocation.py

**Purpose:** Visualize ID allocation patterns, analyze gaps, and find safe ID ranges for modding

#### How It Works

The script loads item data from any supported source (same detection system as other scripts), builds a set of all used IDs, then produces both console and HTML visualizations.

**Console visualization** divides the ID range into configurable blocks (default 50 IDs per block). Each block is represented as a character: `█` for fully occupied, `▓`/`▒`/`░` for partially occupied (75%/50%/25%+), and `·` for empty. Color coding via `colorama` marks game IDs in red and free blocks in green.

**HTML report** generates a standalone HTML file with an interactive ID allocation map. Each ID is represented as a pixel-sized cell in a grid, color-coded by occupancy. The report includes a statistics dashboard (total IDs, used, free, fragmentation index), a visual heat map, a searchable free blocks table, and gap analysis.

**Fragmentation index** is calculated as 1 minus the ratio of the largest free block to total free space. Values 0.0–0.3 indicate low fragmentation (ideal), 0.4–0.6 medium, and 0.7–1.0 high (many scattered small gaps).

**Free block identification** scans the full range and identifies continuous sequences of unused IDs, reporting their start position, length, and percentage of total free space. This helps modders quickly find the best location for new ID allocations.

#### All Parameters

```
visualize_id_allocation.py [options]

OPTIONS:
  --source=TYPE       Force specific source: json, tbl, original, p3a, zzz
  --no-interactive    Auto-select first source if multiple found
  --keep-extracted    Keep temporary extracted files from P3A
  --format=FORMAT     Output format: console, html, both (default: both)
  --block-size=N      Block size for visualization (default: 50)
  --output=FILE       HTML output filename (default: id_allocation_map.html)
  --help              Show help message
```

#### Example

```bash
python visualize_id_allocation.py --format=both
```

**Console Output (example):**
```
ID Allocation Map (block size: 50)
══════════════════════════════════

   0: ████████████████████ ████████████████████ 1000
1000: ████████░░░░░░░░░░░░ ···················· 2000
2000: ···················· ···················· 3000
3000: ··░░████████████████ ████████████████████ 4000
4000: ████████████████████ ████·····░░░░░░···· 5000

Statistics:
  Total range: 0-5000
  Used IDs:    2116
  Free IDs:    2884
  Fragmentation: 0.42 (Medium)
  Largest free block: 1680 IDs at position 1050
```

---

### convert_kurotools_schemas.py

**Purpose:** Convert KuroTools schema definitions to kurodlc_schema.json format

#### How It Works

This is a one-time conversion utility. It reads KuroTools schema definition files (JSON) from a `schemas/headers/` directory, converts each field type to Python struct format equivalents, calculates total byte sizes, and merges the results with the existing `kurodlc_schema.json`.

**Type conversion** is handled by `TYPE_MAPPING`, a dictionary mapping KuroTools type names to `(struct_char, value_type, byte_size)` tuples. For example, `uint` maps to `('I', 'n', 4)` — unsigned 32-bit int, numeric value, 4 bytes. Text offsets (`toffset`) map to `('Q', 't', 8)` — 64-bit pointer. Arrays (`u32array`, `u16array`) are 12 bytes each (8-byte offset + 4-byte count).

**Nested structures** (KuroTools schemas with `size` and `schema` fields) are flattened by repeating inner fields `size` times with numbered prefixes (e.g., `eff1_id`, `eff2_id`).

**Multi-variant handling**: many schemas have variants for different platforms/games (e.g., `FALCOM_PS4`, `FALCOM_SWITCH`). Each variant is converted separately and tagged with the game name in the `info_comment` field.

**Deduplication** uses `(table_header, schema_length)` as a composite key — the same table can exist with different sizes (different game versions), but exact duplicates are skipped.

#### Required File Structure

```
your_directory/
├── convert_kurotools_schemas.py
├── kurodlc_schema.json          (optional — creates new if missing)
└── schemas/
    └── headers/
        ├── ATBonusParam.json
        ├── ItemTableData.json
        └── ... (280+ files)
```

**No command-line parameters.** The script runs interactively, reads from fixed paths relative to the working directory, and outputs `kurodlc_schema_updated.json` + `conversion_report.txt`.

#### Type Mapping Reference

| KuroTools Type | Struct | Value Type | Size | Description |
|----------------|--------|------------|------|-------------|
| `byte` | `b` | `n` | 1 | Signed 8-bit |
| `ubyte` | `B` | `n` | 1 | Unsigned 8-bit |
| `short` | `h` | `n` | 2 | Signed 16-bit |
| `ushort` | `H` | `n` | 2 | Unsigned 16-bit |
| `int` | `i` | `n` | 4 | Signed 32-bit |
| `uint` | `I` | `n` | 4 | Unsigned 32-bit |
| `long` | `q` | `n` | 8 | Signed 64-bit |
| `ulong` | `Q` | `n` | 8 | Unsigned 64-bit |
| `float` | `f` | `n` | 4 | 32-bit float |
| `toffset` | `Q` | `t` | 8 | Text offset (string pointer) |
| `u32array` | `QI` | `a` | 12 | Array of 32-bit values |
| `u16array` | `QI` | `b` | 12 | Array of 16-bit values |

---

### find_all_items.py

**Purpose:** Search and browse items from game data with intelligent filtering

#### How It Works

The script loads item data from `t_item` sources using the same multi-source detection system as other scripts (JSON → TBL → P3A, with interactive source selection). Data is parsed into a dictionary mapping `id` → `name` for each item in the `ItemTableData` section.

**Search** supports three modes: no query (list all), name search (case-insensitive substring match in item names), and ID search (exact match). Auto-detection determines the mode: pure numbers trigger ID search, text triggers name search. Explicit prefixes `id:` and `name:` override auto-detection — this is important when searching for numbers in item names (e.g., `name:100`).

Output is sorted by numeric ID with aligned columns.

#### All Parameters

```
find_all_items.py [search_query] [options]

SEARCH MODES:
  (no query)          List all items
  TEXT                Auto-detect: numbers → ID search, text → name search
  id:NUMBER           Explicit ID search (e.g., id:100)
  name:TEXT           Explicit name search (e.g., name:sword)

OPTIONS:
  --source=TYPE       Force source: json, tbl, original, p3a, zzz
  --no-interactive    Auto-select first source
  --keep-extracted    Keep temporary files from P3A
  --help              Show help
```

#### Examples

```bash
python find_all_items.py sword          # Find items containing "sword"
python find_all_items.py 310            # Find item with ID 310
python find_all_items.py name:310       # Find items with "310" in name
```

---

### find_all_names.py

**Purpose:** Search and browse character names from game data with intelligent filtering

#### How It Works

Functions identically to `find_all_items.py` but operates on `t_name` data (character name database). Loads from `t_name.json`, `t_name.tbl`, or P3A archives.

**Additional search modes:** `full_name:TEXT` searches in the `full_name` field, and `model:TEXT` searches in the `model` field. This makes it easy to find all characters using a specific model prefix (e.g., `model:chr0100` finds all Van variants).

**Additional display options:** `--show-full` adds the `full_name` column, `--show-model` adds the `model` column. Results are formatted with aligned columns.

The `build_char_map()` function in other scripts relies on the same data — `find_all_names.py` is essentially a user-facing browser for the same character database used internally by `kurodlc_add_mdl.py`.

#### All Parameters

```
find_all_names.py [search_query] [options]

SEARCH MODES:
  (no query)          List all characters
  TEXT                Auto-detect: numbers → ID search, text → name search
  id:NUMBER           Search by exact character ID
  name:TEXT           Search in character names
  full_name:TEXT      Search in full names
  model:TEXT          Search in model names

OPTIONS:
  --source=TYPE       Force source: json, tbl, original, p3a, zzz
  --no-interactive    Auto-select first source
  --keep-extracted    Keep temporary files from P3A
  --show-full         Show full names in output
  --show-model        Show model names in output
  --help              Show help
```

#### Examples

```bash
python find_all_names.py van --show-full --show-model
# 100 : Van              | Van Arkride           | chr0100_01

python find_all_names.py model:chr0100
# Shows all model variants for character chr0100
```

---

### find_all_shops.py

**Purpose:** Search and browse shops from game data

#### How It Works

Same pattern as `find_all_items.py` but operates on `t_shop` data. Loads shop information from `t_shop.json`, `t_shop.tbl`, or P3A archives. Searches by shop ID or shop name using the same auto-detect logic. The `--debug` flag shows raw data structure information, useful for understanding the JSON schema.

#### All Parameters

```
find_all_shops.py [search_query] [options]

SEARCH MODES:
  (no query)          List all shops
  TEXT                Auto-detect: numbers → ID search, text → name search
  id:NUMBER           Search by exact shop ID
  name:TEXT           Search in shop names

OPTIONS:
  --source=TYPE       Force source: json, tbl, original, p3a, zzz
  --no-interactive    Auto-select first source
  --keep-extracted    Keep temporary files from P3A
  --debug             Show debug information about data structure
  --help              Show help
```

---

### find_unique_item_id_for_t_costumes.py

**Purpose:** Extract all unique item IDs from costume data (CostumeParam section)

#### How It Works

Loads `t_costume` data from `t_costume.json`, `t_costume.tbl`, or P3A archives. Iterates through the `CostumeParam` section and extracts every `item_id` value. Deduplicates and sorts the results. Supports three output formats: `list` (Python-style list), `count` (just the total), and `range` (min-max with count).

This is useful for understanding what costume IDs are already used by the game, complementing `find_unique_item_id_from_kurodlc.py` which checks DLC files.

#### All Parameters

```
find_unique_item_id_for_t_costumes.py [options]

OPTIONS:
  --source=TYPE       Force source: json, tbl, original, p3a, zzz
  --no-interactive    Auto-select first source
  --keep-extracted    Keep temporary files from P3A
  --format=FORMAT     Output format: list (default), count, range
  --help              Show help
```

---

### find_unique_item_id_for_t_item_category.py

**Purpose:** Extract unique item IDs from a specific category in ItemTableData

#### How It Works

Loads `t_item` data and filters entries by the `category` field matching the specified number. Extracts `id` values from matching entries, deduplicates, and sorts. Useful for understanding what IDs exist in a specific item category (e.g., category 17 = costumes, category 5 = weapons).

#### All Parameters

```
find_unique_item_id_for_t_item_category.py <category> [options]

ARGUMENTS:
  <category>          Category number to filter by (integer, required)

OPTIONS:
  --source=TYPE       Force source: json, tbl, original, p3a, zzz
  --no-interactive    Auto-select first source
  --keep-extracted    Keep temporary files from P3A
  --format=FORMAT     Output format: list (default), count, range
  --help              Show help
```

#### Example

```bash
python find_unique_item_id_for_t_item_category.py 17 --format=range
# Output: 3500-3650 (45 IDs in category 17)
```

---

### find_unique_item_id_from_kurodlc.py

**Purpose:** Extract item IDs from kurodlc files with multiple output modes and game data checking

#### How It Works

This script scans `.kurodlc.json` files and extracts all item IDs. It uses `extract_item_ids()` which reads four sections (CostumeParam → `item_id`, ItemTableData → `id`, DLCTableData → `items` array values, ShopItem → `item_id`). The `is_valid_kurodlc_structure()` function validates files by checking for known section names.

**Check mode** is the most powerful feature. It loads game item data (same multi-source system) and cross-references every DLC item ID against the game database. Each ID is marked `[OK]` (available — not in game data) or `[BAD]` (conflict — exists in game data). Color output uses `colorama` on Windows. This is essentially a read-only version of what `resolve_id_conflicts_in_kurodlc.py` does in its detection phase.

**Output modes** control how results are displayed:

- **Single file** — `<file.kurodlc.json>`: outputs a sorted list of unique IDs from one file
- **searchall** — processes all `.kurodlc.json` files, outputs one combined sorted list
- **searchallbydlc** — shows IDs per file, then combined unique list
- **searchallbydlcline** — same as above but each ID on its own line
- **searchallline** — combined list with each ID on its own line
- **check** — cross-reference DLC IDs against game data

#### All Parameters

```
find_unique_item_id_from_kurodlc.py <mode> [options]

MODES:
  <file.kurodlc.json>     Process single file, print unique item_ids
  searchall                Process all .kurodlc.json, print combined list
  searchallbydlc           Per-file lists, then combined unique list
  searchallbydlcline       Like searchallbydlc but one ID per line
  searchallline            Combined list, one ID per line
  check                    Cross-reference DLC IDs against game data

CHECK MODE OPTIONS:
  --source=TYPE       Force source: json, tbl, original, p3a, zzz
  --no-interactive    Auto-select first source
  --keep-extracted    Keep temporary extracted files
```

#### Example: Check Mode

```bash
python find_unique_item_id_from_kurodlc.py check --source=json
```

**Output:**
```
3596 : available [OK]
3597 : available [OK]
 310 : Earth Sepith [BAD]

Summary:
Total IDs : 3
OK        : 2
BAD       : 1

Source used for check: t_item.json
```

---

## 3D Model Viewer Scripts

All viewer scripts are located in the `viewer_mdl/` directory. They form an evolution from simple to full-featured:

| Script | Renders | Textures | Animation | Window | Key Feature |
|--------|---------|----------|-----------|--------|-------------|
| viewer.py | .fmt/.vb/.ib | ❌ | ❌ | Browser | HTML export with Three.js |
| viewer_mdl.py | .mdl direct | ❌ | ❌ | Browser | Direct MDL parsing |
| viewer_mdl_window.py | .mdl direct | ❌ | ❌ | Native | pywebview, no temp files |
| viewer_mdl_optimized.py | .mdl direct | ❌ | ❌ | Native | Base64 compression for large models |
| viewer_mdl_textured.py | .mdl direct | ✅ DDS | ❌ | Native | Texture loading via Pillow |
| viewer_mdl_textured_anim.py | .mdl direct | ✅ DDS | ✅ Full | Native | **Main viewer** — animations, shaders, gamepad |
| viewer_mdl_textured_scene.py | .mdl direct | ✅ DDS | ✅ Full | Native | Scene mode — maps, terrain, FPS camera |

---

### viewer.py

**Purpose:** Standalone HTML-based 3D viewer using pre-exported .fmt/.vb/.ib files

#### How It Works

This is the simplest viewer. It expects a model directory containing pre-exported geometry files in GPU buffer format: `.fmt` (vertex format descriptor), `.vb` (vertex buffer), and `.ib` (index buffer). These are typically produced by `kuro_mdl_export_meshes.py`.

The `TrailsModelLoader` class reads these files using `InputLayout` to parse vertex formats (semantic names, data types, byte offsets) from `.fmt`, then `_load_vertices()` decodes vertex data from `.vb` and `_load_indices()` reads triangle indices from `.ib`. Each mesh is loaded separately based on numbered file groups (e.g., `0.fmt`, `0.vb`, `0.ib`).

**Normal computation** uses `compute_smooth_normals_with_sharing()` which groups vertices by position (within a tolerance), accumulates face normals across shared positions, and normalizes the result. This produces smooth shading even when the same position appears in multiple vertices (typical for UV seams).

The `export_html()` function generates a self-contained HTML file embedding Three.js inline (from local `three.min.js`). Mesh data is serialized as JSON arrays in the HTML. The viewer implements orbit controls, mesh list toggle, wireframe mode, and basic lighting.

#### Parameters

```
viewer.py <model_path> [--no-original]

  model_path          Path to model directory (containing .fmt/.vb/.ib files)
  --no-original       Use computed smooth normals instead of original normals from .vb
```

---

### viewer_mdl.py

**Purpose:** Direct .mdl file preview without intermediate exported files

#### How It Works

Unlike `viewer.py`, this script reads `.mdl` files directly. It imports `decryptCLE`, `obtain_material_data`, and `obtain_mesh_data` from `kuro_mdl_export_meshes.py` to parse the binary MDL format. CLE-encrypted models are automatically decrypted using `blowfish` and `zstandard`.

The `load_mdl_direct()` function calls the parser, which returns mesh data including vertex positions, normals, UV coordinates, and triangle indices. If the model contains original normals and `--use-original-normals` is set, those are used; otherwise smooth normals are computed.

The `export_html_from_meshes()` function generates HTML similar to `viewer.py` but with MDL-specific mesh naming (using material names from the MDL header). Output is a single HTML file opened in the default browser.

#### Parameters

```
viewer_mdl.py <mdl_path> [--use-original-normals]

  mdl_path                Path to .mdl file
  --use-original-normals  Use normals from MDL data instead of computing smooth normals
```

---

### viewer_mdl_window.py

**Purpose:** Native window .mdl viewer with automatic cleanup (no files left behind)

#### How It Works

Identical rendering to `viewer_mdl.py` but uses `pywebview` to display in a native OS window (Edge WebView2 on Windows, GTK WebKit2 on Linux, WKWebView on macOS) instead of exporting a permanent HTML file. The HTML is written to a temp directory which is automatically cleaned up on exit via `atexit.register()`.

The script creates a `tempfile.mkdtemp()` directory, writes the generated HTML there, launches `webview.create_window()`, and blocks on `webview.start()`. When the window is closed, the temp directory is deleted.

#### Parameters

```
viewer_mdl_window.py <mdl_path> [--use-original-normals]

  Same as viewer_mdl.py. Requires: pip install pywebview
```

---

### viewer_mdl_optimized.py

**Purpose:** Optimized native window viewer for large models

#### How It Works

Same as `viewer_mdl_window.py` but optimized for models with high polygon counts. Large mesh data is **base64-encoded and compressed** before embedding in HTML to avoid WebView2 size limits. The browser-side JavaScript decodes the base64 data back into typed arrays.

This resolves issues where very large models (100K+ vertices) would cause the WebView to fail due to excessive inline JSON data.

#### Parameters

```
viewer_mdl_optimized.py <mdl_path> [--use-original-normals]

  Same as viewer_mdl_window.py. Requires: pip install pywebview
```

---

### viewer_mdl_textured.py

**Purpose:** Textured MDL viewer with DDS support

#### How It Works

Extends `viewer_mdl_window.py` with texture loading. The `load_mdl_with_textures()` function first parses the MDL to extract material information (which textures each material references), then uses `lib_texture_loader.py` to find and convert DDS texture files.

**Texture resolution** searches for DDS files in multiple locations: the model's directory, parent directories, and common game directory structures (`dx11/image/`, `common/image/`, etc.). Found DDS textures are converted to PNG using Pillow (`PIL`) and copied to the temp directory. The HTML references these PNG files via relative paths.

The material-texture mapping connects each mesh's material name to its diffuse texture, normal map, and other texture types parsed from the MDL material data.

A JS API class (`Api`) is exposed to the webview, providing Python ↔ JavaScript bridge for file operations like screenshot saving.

#### Parameters

```
viewer_mdl_textured.py <mdl_file> [--use-original-normals]

  Requires: pip install pywebview Pillow
```

---

### viewer_mdl_textured_anim.py

**Version:** Ver 1.0  
**Purpose:** Full-featured 3D model viewer with textures, skeleton, animations, and gamepad support

This is the **main viewer** — the most capable and feature-rich script.

#### How It Works

**MDL Loading** extends `viewer_mdl_textured.py` with skeleton and animation support. `load_skeleton_from_mdl()` parses bone hierarchy data directly from the raw MDL binary — it scans for bone name strings and parent-child relationships encoded in the file. Each bone has a name, parent index, and local transform matrix.

**Animation Loading** (`load_animations_from_directory()`) scans the model directory for animation files matching the pattern `*_m_*.mdl` (e.g., `chr5001_m_idle.mdl`, `chr5001_m_walk.mdl`). Each animation file contains bone transform keyframes (position, rotation, scale per bone per frame). The loader extracts these and packages them as JSON-serializable animation data. Built-in procedural animations (T-Pose, Idle, Wave, Walk) serve as fallbacks when no external animation files exist.

**Face Animations** (`load_face_animations_from_directory()`) load from `*_face.mdl` files. These contain morph target or bone-based facial expressions (blinking, talking, emotions). Face animations are tagged separately in the UI.

**FXO Shader Support** parses compiled DXBC shader files (`.fxo`) found in a sibling `fxo/` directory. The `parse_fxo_shader()` function reads the shader binary to extract uniform names (constant buffer layouts) and texture slot bindings. When FXO shaders are available, the viewer renders materials using **toon/cel shading** parameters from the game's actual shader data, matching in-game appearance. The `--no-shaders` flag disables this and uses standard PBR materials.

**Gamepad Support** implements a polling system in JavaScript that detects connected controllers and maps their inputs. Supported controller types:

- **DualSense** (PS5) — full support with touchpad/adaptive triggers
- **DualShock** (PS4) — standard button mapping
- **Switch Pro** — Nintendo-style button layout
- **Generic** — fallback XInput mapping
- **Keyboard** — WSAD + mouse orbit

Gamepad controls allow third-person camera orbit, model rotation, animation switching, and speed control. Button mappings include: left stick for camera orbit, right stick for zoom, D-pad for animation selection, triggers for speed adjustment, face buttons for mode toggles (wireframe, bones, screenshot, etc.).

**UI Features** (rendered in HTML/CSS/JS):

- Mesh list with individual show/hide toggles
- Mesh highlighting on hover
- Auto-hide shadow/kage meshes (configurable)
- Background color picker
- Lighting controls (ambient/directional intensity, direction)
- Wireframe overlay toggle
- Skeleton visualization with bone hierarchy
- Animation player with play/pause, speed, frame scrubber
- Face animation controls
- Screenshot capture (saved to Downloads folder)
- Video recording with quality settings (requires `av` module)
- Skybox/environment map support
- FXO shader toggle
- Physics simulation with collision and intensity controls
- FreeCam (free camera flight mode)
- Configurable via `viewer_mdl_textured_config.md`

#### Parameters

```
viewer_mdl_textured_anim.py <path_to_model.mdl> [options]

  --recompute-normals  Recompute smooth normals instead of using originals from MDL
                       (slower loading, typically no visual difference)
  --debug              Enable verbose console logging in browser DevTools
  --skip-popup         Skip loading progress popup on startup
  --no-shaders         Disable toon shader (FXO), use standard PBR materials

  Requires: pip install pywebview Pillow
  Optional: pip install av (for video recording)
```

#### Configuration

The viewer reads `viewer_mdl_textured_config.md` for persistent settings. Runtime configuration is available in the generated HTML via the `CONFIG` object:

```javascript
const CONFIG = {
  CAMERA_ZOOM: 1.0,           // Camera zoom factor (0.8=close, 1.5=far)
  AUTO_HIDE_SHADOW: true,      // Hide shadow meshes on load
  INITIAL_BACKGROUND: 0x1a1a2e // Background color (hex)
};
```

---

### viewer_mdl_textured_scene.py

**Purpose:** Scene viewer for loading entire game maps with multiple MDL models

#### How It Works

This script extends `viewer_mdl_textured_anim.py` with a **scene mode** that parses binary scene JSON files and renders complete 3D game environments. When invoked with `--scene`, it bypasses single-model loading and instead:

**Scene Parsing** (`parse_scene_json()`) reads binary JSON files from the `scene/` directory. These files define actor placements in a game map — each entry contains an actor type (e.g., `StaticMeshActor`, `FieldTerrain`, `PointLight`, `PlantActor`), a model name, and a 4×4 transformation matrix (position, rotation, scale). Non-visual actor types (lights, sounds, navigation) are identified and excluded from model loading.

**Directory Structure Resolution**: The scene expects a specific layout where `scene/` and `asset/` are sibling directories under a game root:
```
game_root/
├── scene/
│   └── mp0010.json          ← Scene file
└── asset/
    ├── common/model/        ← MDL files
    ├── dx11/image/          ← DDS textures
    └── ...
```

**MDL Index Building**: The script recursively scans `asset/common/model/`, `asset/dx11/model/`, and other subdirectories, building a case-insensitive index of all `.mdl` files (excluding animation files `*_m_*.mdl` and queue files `q_*`).

**Model Name Resolution** uses an 8-strategy matching system to connect scene actor names to actual MDL files:

1. **Exact match** — actor name matches MDL filename directly
2. **Prefix stripping** — removes `CP_` and `EV_` prefixes
3. **Door pattern** — `door_mp0010_00` → `ob0010dor00`
4. **Vegetation clusters** — `CP_m0010_grass` → `ob0010plt*`
5. **Field terrain** — `FieldTerrain` → `mp{mapid}` chunk models
6. **Type names** — known non-visual types are skipped
7. **Category hints** — name contains category keyword (cover, kusa, obj, etc.) → `ob{mapid}{cat}*`
8. **Fallback** — unresolved models use placeholder geometry

**Multi-Model Loading**: For each resolved unique model, `load_mdl_with_textures()` is called. Material names are prefixed with the model name (e.g., `chr5001##material_body`) to avoid collisions between models sharing material names. Texture search paths include the `asset/*/image/` directories.

**Terrain Chunks**: `FieldTerrain` actors reference map-specific terrain chunks (e.g., `mp0010`, `mp0010_01`). These are loaded separately and placed according to their scene transforms.

**Scene Rendering**: The generated HTML includes all model data as instances. Each scene actor is rendered by creating a Three.js `Object3D` with the model's geometry and applying the actor's 4×4 transform matrix. This allows hundreds of objects to share a few dozen unique geometries.

**Scene-Specific UI Features**:

- **FreeCam** (FPS-style camera) — always active in scene mode, WSAD + mouse for movement
- **Minimap** — overhead view showing camera position and actor locations
- **Search** — find actors by name in the scene hierarchy
- **Category filters** — show/hide actors by type (terrain, buildings, props, vegetation, etc.)
- **Fog controls** — distance fog for depth perception in large scenes
- **Grid** — ground plane reference grid
- All standard viewer features (textures, shaders, screenshots, gamepad, etc.)

#### Parameters

```
viewer_mdl_textured_scene.py <path_to_model.mdl> [options]
viewer_mdl_textured_scene.py --scene <scene_file.json> [options]

SCENE MODE:
  --scene <file>       Load a scene file (.json binary format)
                       Searches: exact path, scene/ subdirectory
                       Loads MDL models from asset/ directory

STANDARD MODE:
  (same as viewer_mdl_textured_anim.py — single model viewing)

OPTIONS:
  --recompute-normals  Recompute smooth normals
  --debug              Enable verbose console logging
  --skip-popup         Skip loading progress popup
  --no-shaders         Disable FXO toon shaders

  Requires: pip install pywebview Pillow
```

#### Scene Mode Example

```bash
python viewer_mdl_textured_scene.py --scene mp0010.json
```

**Output:**
```
============================================================
SCENE MODE: mp0010.json
============================================================
[+] Parsed 847 actors (124 unique models)
    StaticMeshActor: 612
    FieldTerrain: 45
    PointLight: 89
    PlantActor: 56
    SpotLightActor: 45

[+] Model resolution: 98/124 models found
[+] Terrain chunks in resolved: ['mp0010', 'mp0010_01', 'mp0010_02', ...]

[+] Loading 98 unique MDL models...
  [1/98] Loading ob0010bld001...
    -> 12 meshes loaded
  [2/98] Loading ob0010grd000...
    -> 4 meshes loaded
  ...

[+] Loaded 98 models, 1247 total meshes
[+] Launching viewer...
```

---

## Library Files

### kurodlc_lib.py

**Purpose:** Read and write Kuro DLC binary table files (.tbl) with schema-based serialization

*External library by eArmada8 (GitHub: kuro_dlc_tool)*

#### How It Works

The `kuro_tables` class is the core engine that powers DLC modding. It reads binary `.tbl` game data tables, deserializes them into Python dicts using schema definitions, merges DLC modifications from `.kurodlc.json` files, and writes the modified tables back to binary format.

**Initialization** (`__init__` → `init_schemas()`): On creation, the class loads `kurodlc_schema.json` — a JSON array of schema definitions, each mapping a `(table_header, schema_length)` pair to a struct layout. It then scans all `.tbl` and `.tbl.original` files in the working directory recursively, reading their `#TBL` headers to build two indexes: `schema_dict` (table name → entry byte length) and `crc_dict` (table name → CRC32 checksum). These indexes allow the class to match any table section to its correct schema at runtime.

**Schema lookup** (`get_schema(name, entry_length)`): Returns the schema for a given table section. Schemas are keyed by `(name, entry_length)` tuple because the same table name (e.g., `ItemTableData`) can have different byte layouts across game versions (Kuro 1 vs Kuro 2 vs Kai). If no schema matches, returns empty dict — the section is skipped during read/write.

**Table reading** (`read_table(table_name)`): Parses a `#TBL` binary file. The format starts with magic `#TBL` + section count, followed by section headers (64-byte name string, CRC, offset, entry length, entry count). For each section, raw binary rows are unpacked using `struct.unpack()` with the schema's format string (e.g., `<I4Q3i`). The inner `decode_row()` function then transforms raw values:
- `n` (number) → kept as integer/float
- `t` (text offset) → seeks to the 64-bit offset in the file and reads a null-terminated UTF-8 string
- `a` (u32 array) → seeks to the offset and reads `count` 32-bit unsigned integers
- `b` (u16 array) → seeks to the offset and reads `count` 16-bit unsigned integers

This produces a Python dict per row, with field names from the schema's `keys` array.

**Table writing** (`write_table(table_name)`): The reverse process. First creates a backup (`.original`) if one doesn't exist. Reads the original table, merges DLC data via `update_table_with_kurodlc()`, then serializes back to binary. String and array data are written to a secondary buffer (`data2_buffer`) that follows the fixed-size row data. Text fields become 64-bit offsets pointing into this buffer. Arrays become offset+count pairs. The final binary is: `#TBL` header + section headers + packed rows + variable-length data buffer.

**DLC merging** (`read_kurodlc_json()` → `validate_kurodlc_entries()` → `detect_duplicate_entries()` → `update_table_with_kurodlc()`): This pipeline loads a `.kurodlc.json` file, validates that every entry's keys match the schema (auto-correcting key order if possible), checks for duplicate primary keys across all loaded DLCs (warning the user about conflicts with source file attribution), and finally merges new entries into the table data. When a primary key exists in both the original table and DLC data, the DLC version replaces the original (last-write-wins by primary key).

**Composite primary keys**: Some schemas define `primary_key` as a list (e.g., `["shop_id", "item_id"]`). The class handles this by creating a temporary tuple key `new_primary_key` that joins the component values, enabling correct deduplication even for multi-column primary keys.

#### Key Methods

| Method | Purpose |
|--------|---------|
| `init_schemas()` | Load `kurodlc_schema.json` + scan `.tbl` files for metadata |
| `get_schema(name, length)` | Look up binary layout for a table section |
| `read_table(name)` | Parse `.tbl` binary → Python dicts |
| `write_table(name)` | Serialize Python dicts → `.tbl` binary (with backup) |
| `read_kurodlc_json(name)` | Load + validate + dedup a single `.kurodlc.json` |
| `read_all_kurodlc_jsons()` | Process all `*.kurodlc.json` in current directory |
| `validate_kurodlc_entries()` | Check key names and value types against schema |
| `detect_duplicate_entries()` | Warn about primary key conflicts between DLC files |
| `update_table_with_kurodlc()` | Merge DLC entries into table (replace by primary key) |
| `read_struct_from_json()` | Safe JSON file reader with detailed error messages |
| `write_struct_to_json()` | JSON file writer (UTF-8 encoded) |

#### Dependencies

- `kurodlc_schema.json` — must be in the same directory as the library
- Python standard library only (json, struct, shutil, glob, os)

#### Used By

All scripts that need to read game data from `.tbl` binary files: `resolve_id_conflicts_in_kurodlc.py`, `find_all_items.py`, `find_all_names.py`, `find_all_shops.py`, `find_unique_item_id_for_t_costumes.py`, `find_unique_item_id_for_t_item_category.py`, `find_unique_item_id_from_kurodlc.py`, `kurodlc_add_mdl.py`, `visualize_id_allocation.py`.

---

### p3a_lib.py

**Purpose:** Read, extract, and create P3A archive files (Falcom's proprietary archive format)

*External library by eArmada8 (GitHub: kuro_dlc_tool)*

#### How It Works

The `p3a_class` handles P3A archives — compressed file containers used by Falcom games to package game data (scripts, tables, textures, models). Archives like `script_en.p3a` contain `.tbl` table files that other scripts need to access for game data lookup.

**Archive reading** (`read_p3a_toc()`): Parses the P3A header starting with magic `PH3ARCV\x00`. The header contains flags, version number (1100 or 1200), file count, and archive hash (xxHash64). For version 1200+, additional fields include extended header size and entry size. Each file entry is a 256-byte name string followed by compression type, compressed size, uncompressed size, file offset, compressed hash, and (v1200+) uncompressed hash. If flags bit 0 is set, a Zstandard compression dictionary follows the entries (magic `P3ADICT\x00`).

**File decompression** (`read_file(entry, p3a_dict)`): Reads compressed data at the entry's offset, verifies integrity via xxHash64, then decompresses based on `cmp_type`:
- **0** — uncompressed (raw copy)
- **1** — LZ4 block compression (`lz4.block.decompress()`)
- **2** — Zstandard compression (`zstandard.ZstdDecompressor()`)
- **3** — Zstandard with shared dictionary (uses the archive's `P3ADICT` data for better compression ratios across similar files)

After decompression, the uncompressed hash is verified (v1200+).

**File extraction** (`extract_all_files(p3a_archive, output_dir)`): Reads the TOC, iterates all entries, decompresses each, and writes to disk preserving the internal directory structure. Prompts before overwriting existing files.

**Archive creation** (`p3a_pack_files(file_list, ...)`): Builds a new P3A archive from a list of files. Supports all four compression types. For type 3 (dictionary-based Zstandard), it first trains a compression dictionary from all input files using `zstandard.train_dictionary()` with a 112KB dictionary size. Files are aligned to 64-byte boundaries within the archive. The TOC is written first (header + entries + optional dictionary), followed by the compressed file data block.

**Folder packing** (`pack_folder(folder_name, ...)`): Convenience method that recursively collects all files in a folder and packs them into a P3A with paths relative to the folder root.

#### Compression Types

| Type | Method | Use Case |
|------|--------|----------|
| 0 | None | Small files, debugging |
| 1 | LZ4 | Fast compression/decompression (default) |
| 2 | Zstandard | High compression ratio |
| 3 | Zstandard + Dict | Best ratio for similar files (e.g., table collections) |

#### Dependencies

- `lz4` — LZ4 block compression (`pip install lz4`)
- `zstandard` — Zstandard compression/decompression (`pip install zstandard`)
- `xxhash` — xxHash64 integrity verification (`pip install xxhash`)

#### Used By

All scripts that access game data from P3A archives: when a user has `script_en.p3a` or `zzz_combined_tables.p3a` instead of pre-extracted `.tbl`/`.json` files, the scripts use `p3a_lib` to extract the needed table on-the-fly to a temporary file.

---

### kuro_mdl_export_meshes.py

**Purpose:** Parse ED9/Kuro no Kiseki .mdl model files and export geometry as .fmt/.ib/.vb buffers

*External library by eArmada8 (GitHub: kuro_mdl_tool), based on Uyjulian's parser*

#### How It Works

This is the foundational MDL parser that all viewer scripts depend on. It reads the proprietary binary `.mdl` format used by Falcom's Kuro no Kiseki engine and extracts geometry, material, and skeleton data.

**CLE Decryption** (`decryptCLE(file_content)`): Kuro no Kiseki models can be encrypted and/or compressed. The function checks the file magic and applies transformations in a loop until raw MDL data is obtained:
- Magic `F9BA` or `C9BA` → Blowfish CTR decryption with a hardcoded 16-byte key and 8-byte IV
- Magic `D9BA` → Zstandard decompression

The loop handles chained encryption+compression (decrypt first, then decompress).

**MDL Structure**: The binary format starts with a 12-byte header (magic `MDL ` = `0x204c444d`, version, flags). The remainder is a sequence of typed sections, each with an 8-byte header (type ID, size). Section types:
- **Type 0** — Material data
- **Type 1** — Mesh/geometry data
- **Type 2** — Skeleton/bone hierarchy
- **Type 4** — Primitive data (Kuro 2+)

The `isolate_*_data()` functions seek through sections to find and return the raw bytes for a specific type.

**Material Parsing** (`obtain_material_data(mdl_data)`): Reads the material section as a sequence of material blocks. Each block contains:
- `material_name` — Pascal string (1-byte length prefix + ASCII)
- `shader_name` — the shader program name (e.g., `chr_skin`, `chr_eye`)
- `str3` — additional shader identifier
- `textures[]` — array of texture references, each with: texture image name (DDS filename without extension), texture slot index, wrap modes (S/T), and version-specific unknowns
- `shaders[]` — shader parameter values with typed data (int, float, vec2, vec3, vec4, matrix4x4 as base64)
- `material_switches[]` — named boolean-like switches that control shader behavior
- `uv_map_indices` — which UV channels this material uses

The shader switches are hashed with xxHash64 for quick comparison (stored as `shader_switches_hash_referenceonly`).

**Mesh Parsing** (`obtain_mesh_data(mdl_data, material_struct, trim_for_gpu)`): The most complex function. For each mesh in the MDL:
1. Reads mesh header: mesh name (from skeleton node), node count, submesh count
2. For each submesh: reads vertex buffer layout descriptors (semantic name, DXGI format, byte offset), vertex count, index count, and the raw GPU buffer data
3. Decodes vertex buffers using `lib_fmtibvb` functions — each vertex is unpacked according to its DXGI format (e.g., `R32G32B32_FLOAT` for positions, `R8G8B8A8_UNORM` for blend weights)
4. Reads index buffers (16-bit or 32-bit triangle indices)
5. Associates each submesh with its material via `material_dict`

The `trim_for_gpu` flag controls whether to keep only GPU-relevant vertex attributes (position, normal, UV, blend) or preserve all attributes including debug data.

For Kuro 2+ models (version > 1), vertex and index data is stored in a separate primitive section (type 4) rather than inline in the mesh section. The `isolate_primitive_data()` and `parse_primitive_header()` functions handle this layout.

**Skeleton Parsing** (`obtain_skeleton_data(mdl_data)`): Reads the bone hierarchy as a flat list of nodes. Each node contains:
- `name` — bone name (Pascal string)
- `type` — 0=transform, 1=skin child, 2=mesh
- `mesh_index` — which mesh this bone controls (-1 if none)
- `pos_xyz` — local position (3 floats)
- `unknown_quat` — quaternion rotation (4 floats)
- `skin_mesh` — skin mesh reference
- `rotation_euler_rpy` — Euler angles roll/pitch/yaw (3 floats)
- `scale` — local scale (3 floats)
- `children[]` — list of child node indices

The parent-child relationships form a tree where the root is typically node 0.

**Export** (`process_mdl()`, `write_fmt_ib_vb()`): The main export pipeline reads an MDL file, decrypts it, extracts materials/meshes/skeleton, and writes per-submesh output files:
- `*.fmt` — text-based vertex format descriptor (semantic names, DXGI formats, byte offsets)
- `*.vb` — raw vertex buffer (binary, matches fmt layout)
- `*.ib` — raw index buffer (16-bit or 32-bit triangles)
- `*.vgmap` — vertex group mapping (bone name → index, JSON)
- `material_info.json` — full material data including textures and shader params
- `mesh_info.json` — mesh metadata (names, node counts, submesh info)
- `skeleton.json` — complete bone hierarchy
- `image_list.json` — list of referenced DDS texture filenames

#### Key Functions

| Function | Purpose |
|----------|---------|
| `decryptCLE()` | Decrypt/decompress CLE-encrypted MDL data |
| `obtain_material_data()` | Parse material section → texture refs, shader params |
| `obtain_mesh_data()` | Parse mesh section → vertex/index buffers, layouts |
| `obtain_skeleton_data()` | Parse skeleton section → bone hierarchy |
| `process_mdl()` | Full pipeline: read MDL → export .fmt/.vb/.ib + JSON |
| `isolate_*_data()` | Extract raw bytes for a specific MDL section type |
| `write_fmt_ib_vb()` | Write a single submesh's geometry files |

#### Dependencies

- `blowfish` — Blowfish cipher for CLE decryption (`pip install blowfish`)
- `zstandard` — Zstandard decompression (`pip install zstandard`)
- `xxhash` — hash verification (`pip install xxhash`)
- `numpy` — array operations
- `lib_fmtibvb.py` — GPU buffer format I/O (must be in same directory)

#### Used By

All viewer scripts (`viewer_mdl.py`, `viewer_mdl_textured.py`, `viewer_mdl_textured_anim.py`, `viewer_mdl_textured_scene.py`) import three key functions: `decryptCLE`, `obtain_material_data`, `obtain_mesh_data`. The viewers call these to load MDL data directly into memory (without writing intermediate files to disk), then convert the mesh buffers into Three.js-compatible JSON for browser rendering.

---

### lib_texture_loader.py

**Purpose:** DDS texture loading, path resolution, and format conversion

#### How It Works

**DDSHeader** class parses the 128-byte DDS file header, extracting dimensions (`width`, `height`), mipmap count, and pixel format (FourCC code like `DXT1`, `DXT5`, `BC7`, etc.).

**find_texture_file()** searches for a texture by name across multiple directories. It tries several path variations: exact filename, with `.dds` extension, uppercase/lowercase variants, and common game directory structures. The `search_paths` list is built by the caller based on the model's location and game asset structure.

**convert_dds_to_png_pil()** converts DDS binary data to PNG bytes using Pillow. Pillow handles common DDS formats (DXT1/BC1, DXT5/BC3, BC7, uncompressed RGBA). Returns `None` for unsupported formats.

**convert_dds_to_rgba_raw()** is a fallback converter that decodes DDS to raw RGBA pixel data without Pillow, handling basic uncompressed formats only.

**load_texture_as_data_url()** combines finding and converting: locates a DDS file, converts to PNG, then encodes as a base64 data URL for HTML embedding.

**load_material_textures()** processes a material's texture references (from MDL material data), resolves each texture name to a file path, converts to PNG, and caches results. Returns a map of texture slot → file info.

---

### lib_fmtibvb.py

**Purpose:** Read/write .fmt, .ib, .vb files (GPU buffer format I/O)

*External library by eArmada8 (GitHub: gust_stuff)*

This library provides functions for parsing DXGI vertex format descriptors and reading/writing GPU buffer data. It handles:

- **unpack_dxgi_vector()** / **pack_dxgi_vector()** — Convert between binary GPU data and Python lists. Supports FLOAT, UINT, SINT, UNORM, SNORM formats in 8/16/32-bit widths.
- **read_fmt()** / **write_fmt()** — Parse `.fmt` files (text-based vertex layout descriptors with semantic names, formats, byte offsets).
- **read_vb()** / **write_vb()** — Read/write vertex buffer files, decoding each vertex according to the format descriptor.
- **read_ib()** / **write_ib()** — Read/write index buffer files (16-bit or 32-bit triangle indices).
- **read_struct_from_json()** / **write_struct_to_json()** — JSON serialization helpers for format structures.

Used by `viewer.py` and `kuro_mdl_export_meshes.py` for geometry I/O.

---

## Data Structure Specifications

### .kurodlc.json File Structure

A `.kurodlc.json` file contains DLC mod data organized into sections:

```json
{
  "CostumeParam": [
    {
      "item_id": 3596,
      "char_restrict": 1,
      "mdl_name": "chr5001_c02aa",
      "unk0": 0,
      "unk1": 0,
      "unk2": 0,
      "unk3": "",
      "unk_txt": ""
    }
  ],
  "ItemTableData": [
    {
      "id": 3596,
      "category": 17,
      "sub_category": 0,
      "char_restrict": 1,
      "name": "Custom Costume - Van",
      "desc": "A custom costume for Van.",
      "price": 0,
      "sell_price": 0
    }
  ],
  "DLCTableData": [
    {
      "id": 1,
      "sort_id": 1,
      "items": [3596, 3597, 3598],
      "quantity": [1, 1, 1],
      "name": "Custom Costume Pack",
      "desc": "A collection of custom costumes."
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

**Section roles:**
- **CostumeParam** — Links item IDs to 3D model files (MDL) and restricts to specific characters
- **ItemTableData** — Defines item metadata (name, description, category, price)
- **DLCTableData** — Groups items into DLC packages with quantities
- **ShopItem** — Assigns items to in-game shops

**Shop-only variant** (v2.2): Files with only a `ShopItem` section are valid. These define shop assignments without creating new items (used for adding existing items to additional shops).

### ID Relationships

The `item_id` field connects all sections:

```
CostumeParam.item_id ─┐
ItemTableData.id ──────┼── Same ID value
DLCTableData.items[] ──┤
ShopItem.item_id ──────┘
```

Changing an ID in one section requires changing it in all four.

### kurodlc_schema.json Structure

Defines binary table formats for `.tbl` files:

```json
{
  "info_comment": "Kuro2 - Converted from KuroTools",
  "table_header": "ItemTableData",
  "schema_length": 248,
  "schema": {
    "schema": "<I4Q3i",
    "sch_len": 248,
    "keys": ["id", "name", "desc", "category", ...],
    "values": "nttnnn...",
    "primary_key": "id"
  }
}
```

- `schema` — Python struct format string (little-endian)
- `keys` — Field names matching struct fields
- `values` — Type codes: `n`=number, `t`=text offset, `a`=u32 array, `b`=u16 array
- `primary_key` — Main lookup field

---

## Real Data Examples

### Game Shops (from t_shop.json)

```
ID  5: Item Shop
ID  6: Weapon/Armor Shop
ID  8: Modification/Trade Shop
ID 10: Orbments
ID 21: Melrose Newspapers & Tobacco
ID 22: Melrose Newspapers & Tobacco
ID 23: Melrose Newspapers & Tobacco
```

### Common Shop Assignment Strategies

**All general shops:**
```bash
--shop-ids=5,6,8,10
```

**Specific vendor (all locations):**
```bash
--shop-ids=21,22,23
```

**Costume-specific:**
```bash
--shop-ids=6
```

---

## Export/Import Formats

### id_mapping JSON (resolve_id_conflicts)

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
    }
  ]
}
```

The `mappings` field is editable — modify new_id values before importing with `--import`.

### Template JSON (shops_find / shops_create)

```json
{
  "_comment": ["Usage instructions..."],
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
  },
  "output_section": "ShopItem"
}
```

---

## Log Files

### ID Conflict Repair Log

Generated by `resolve_id_conflicts_in_kurodlc.py repair --apply`:

```
id_conflict_repair_YYYYMMDD_HHMMSS.log
```

Contains: timestamp, source used, per-file conflict details, all ID mappings applied, backup file locations.

### Conversion Report

Generated by `convert_kurotools_schemas.py`:

```
conversion_report.txt
```

Contains: original schema count, converted count, new schemas added, per-table listing of new entries with sizes and game targets.

### Backup Files

All scripts that modify `.kurodlc.json` files create timestamped backups:

```
filename.kurodlc.json_YYYYMMDD_HHMMSS.bak
```

These are full copies of the original file before any modification.
