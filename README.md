# KuroDLC Modding Toolkit

A comprehensive Python toolkit for creating and managing DLC mods for games using the KuroDLC format. This toolkit provides utilities for item discovery, ID management, conflict resolution, and shop assignment automation.

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-GPL--3.0-green)](LICENSE)

> **⚠️ GPL-3.0 License Notice**  
> This project uses libraries from [eArmada8/kuro_dlc_tool](https://github.com/eArmada8/kuro_dlc_tool) which are licensed under GPL-3.0.  
> Therefore, this entire toolkit is also distributed under GPL-3.0 license.  
> See [License](#-license) section for details.

## 📋 Table of Contents

- [Why This Toolkit Exists](#-why-this-toolkit-exists)
- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Scripts Overview](#-scripts-overview)
- [Detailed Documentation](#-detailed-documentation)
- [Common Workflows](#-common-workflows)
- [File Formats](#-file-formats)
- [Troubleshooting](#-troubleshooting)
- [Best Practices](#-best-practices)
- [Version History](#-version-history)
- [External Dependencies](#-external-dependencies)
- [Contributing](#-contributing)
- [License](#-license)
- [Advanced Documentation](#-advanced-documentation) ⭐

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

- **🛒 Shop Integration v2.1**: Generate shop assignments with customizable templates
- **📦 Bulk Operations**: Assign hundreds of items to multiple shops instantly
- **🎨 Custom Templates**: Define your own shop item structure
- **🤖 CI/CD Support**: Non-interactive mode for automated workflows

### Additional Tools
- **🔍 Item Discovery**: Search and browse game items from JSON, TBL, and P3A sources
- **📋 Multiple Formats**: Support for JSON, TBL, and P3A archive formats
- **🎨 User-Friendly**: Interactive menus and colored output (Windows CMD compatible)

---

## 📦 Requirements

### Python Version
- Python 3.7 or higher

### Required Python Libraries
Install via `install_python_modules.bat` (Windows) or manually:
```bash
pip install colorama --break-system-packages
```

### External Libraries (Included in Repository)

This toolkit uses libraries from [eArmada8/kuro_dlc_tool](https://github.com/eArmada8/kuro_dlc_tool):

- **`p3a_lib.py`** - P3A archive handling (GPL-3.0)
- **`kurodlc_lib.py`** - Kuro table (.tbl) file handling (GPL-3.0)

**⚠️ GPL-3.0 License Implications:**

Because this toolkit uses GPL-3.0 licensed libraries, the following applies:
- ✅ **This entire toolkit is licensed under GPL-3.0**
- ✅ **You can freely use, modify, and distribute this toolkit**
- ⚠️ **Any modifications must also be GPL-3.0**
- ⚠️ **Source code must be made available to users**
- ⚠️ **You cannot incorporate this into proprietary software**

See the [License](#-license) section for full details.

### Optional Dependencies (for P3A/TBL Support)

If you want to work with `.p3a` archives or `.tbl` files, install:
```bash
pip install lz4 zstandard xxhash --break-system-packages
```

**Note:** If you only work with JSON files (`.kurodlc.json`, `t_item.json`, etc.), these optional dependencies are not needed. All core functionality works with JSON only.

---

## 🚀 Installation

### Option 1: Download Release (Recommended)
1. Download the latest release from [Releases](https://github.com/falco81/KuroDLCTools/releases)
2. Extract to your desired location
3. Run `install_python_modules.bat` (Windows) or install packages manually

### Option 2: Clone Repository
```bash
git clone https://github.com/falco81/KuroDLCTools.git
cd kurodlc-toolkit
```

### Install Dependencies

**Windows:**
```bash
install_python_modules.bat
```

**Linux/Mac:**
```bash
pip install colorama --break-system-packages

# Optional: for P3A/TBL support
pip install lz4 zstandard xxhash --break-system-packages
```

---

## 🚀 Quick Start

### 1. Fix ID Conflicts in Your DLC (Primary Use Case)
```bash
# Detect conflicts
python resolve_id_conflicts_in_kurodlc.py repair

# Fix conflicts automatically
python resolve_id_conflicts_in_kurodlc.py repair --apply
```

### 2. Generate Shop Assignments (Secondary Use Case)
```bash
# Step 1: Extract IDs and generate template
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json --generate-template

# Step 2: Create shop assignments
python shops_create.py template_my_mod.kurodlc.json

# Step 3: Copy ShopItem section from output_template_my_mod.kurodlc.json into your my_mod.kurodlc.json
```

### 3. Browse Game Items
```bash
# Search for items
python find_all_items.py t_item.json sepith

# Browse all shops
python find_all_shops.py t_shop.json
```

---

## 📚 Scripts Overview

### Core Scripts (Latest Versions)

| Script | Version | Purpose | Main Features |
|--------|---------|---------|---------------|
| **`resolve_id_conflicts_in_kurodlc.py`** | v2.7.1 | ID conflict resolution | Smart algorithm (v2.7), automatic repair, 1-5000 range limit |
| **`shops_find_unique_item_id_from_kurodlc.py`** | v2.1 | Template generation | Extract IDs, generate templates, CI/CD support |
| **`shops_create.py`** | v2.0 | Shop assignment generation | Bulk assignments, custom templates, variable substitution |

### Utility Scripts

| Script | Purpose |
|--------|---------|
| **`find_all_items.py`** | Search and browse game items |
| **`find_all_shops.py`** | List all shops from game data |
| **`find_unique_item_id_for_t_costumes.py`** | Extract costume IDs |
| **`find_unique_item_id_for_t_item_category.py`** | Extract category IDs |
| **`find_unique_item_id_from_kurodlc.py`** | Check DLC IDs against game data |

### Installation Helper

| Script | Purpose |
|--------|---------|
| **`install_python_modules.bat`** | Install Python dependencies (Windows) |

---

## 📖 Detailed Documentation

### resolve_id_conflicts_in_kurodlc.py (v2.7.1) - PRIMARY TOOL

**Purpose:** Automatically detect and fix ID conflicts between DLC mods and game data.

**Key Features (v2.7.1):**
- ✅ Smart ID assignment algorithm (v2.7)
- ✅ Middle-out distribution starting from ID 2500
- ✅ Hard limit enforcement: 1-5000 range only
- ✅ Automatic backup creation
- ✅ Detailed logging and reporting
- ✅ Manual ID mapping import/export

**Basic Usage:**
```bash
# Check for conflicts (read-only)
python resolve_id_conflicts_in_kurodlc.py checkbydlc

# Repair mode with preview
python resolve_id_conflicts_in_kurodlc.py repair

# Apply fixes automatically
python resolve_id_conflicts_in_kurodlc.py repair --apply

# Export ID mappings for manual editing
python resolve_id_conflicts_in_kurodlc.py repair --export

# Import manual ID mappings
python resolve_id_conflicts_in_kurodlc.py repair --import id_mapping.json --apply
```

**What's New in v2.7:**
- Smart ID distribution algorithm
- Better ID spacing for cleaner organization
- Respects 1-5000 engine limit
- Middle-out assignment from 2500

**What's Fixed in v2.7.1:**
- ✅ Removed bare except clause (line 929)
- ✅ 100% code quality score
- ✅ Better error handling

---

### shops_find_unique_item_id_from_kurodlc.py (v2.1) - TEMPLATE GENERATOR

**Purpose:** Extract item IDs from DLC files and generate template configurations for shop assignment.

**Key Features (v2.1):**
- ✅ Extract IDs from DLC files
- ✅ Auto-detect shop IDs from ShopItem section
- ✅ Generate template configs for shops_create.py
- ✅ Support for DLCs without ShopItem section
- ✅ CI/CD friendly (--no-interactive flag)

**Basic Usage:**
```bash
# Extract all IDs
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json

# Extract only costume IDs
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json costume

# Generate template (auto-detect shop IDs)
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json --generate-template

# Generate template with manual shop IDs
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json --generate-template --shop-ids=1,5,10

# For DLC without ShopItem section
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json --generate-template --default-shop-ids

# For CI/CD (non-interactive)
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json --generate-template --default-shop-ids --no-interactive
```

**What's New in v2.0:**
- Template generation feature
- Auto-extract shop IDs from ShopItem
- Auto-extract template structure
- Custom output filenames

**What's New in v2.1:**
- ✅ --no-interactive flag for CI/CD
- ✅ --default-shop-ids flag for automatic fallback
- ✅ Better error messages with actionable solutions
- ✅ No more EOFError in automated environments
- ✅ 100% code quality score

---

### shops_create.py (v2.0) - ASSIGNMENT GENERATOR

**Purpose:** Generate bulk shop assignments from template configurations.

**Key Features:**
- ✅ Bulk assign items to multiple shops
- ✅ Custom template support
- ✅ Variable substitution (${shop_id}, ${item_id}, ${index}, ${count})
- ✅ Custom output sections
- ✅ Backward compatible with v1.0

**Basic Usage:**
```bash
# Generate shop assignments from template
python shops_create.py template_my_mod.kurodlc.json

# Output: output_template_my_mod.kurodlc.json
```

**Template Structure:**
```json
{
  "item_ids": [3596, 3597, 3598],
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

**Result:** 3 items × 3 shops = 9 shop assignments generated automatically!

---

### Utility Scripts

#### find_all_items.py
Search and browse game items with auto-detection.

```bash
# Search by ID (auto-detected)
python find_all_items.py t_item.json 310
# Output: 310 : Earth Sepith

# Search by name (auto-detected)
python find_all_items.py t_item.json sepith
# Output: Multiple sepith items
```

#### find_all_shops.py
List all shops from game data.

```bash
python find_all_shops.py t_shop.json
# Output: List of all shops with IDs and names
```

#### find_unique_item_id_from_kurodlc.py
Check DLC IDs against game data (requires kurodlc_lib.py).

```bash
# Check mode
python find_unique_item_id_from_kurodlc.py check

# Search all .kurodlc.json files
python find_unique_item_id_from_kurodlc.py searchall
```

---

## 🔄 Common Workflows

### Workflow 1: Creating a New DLC Mod (Complete Process)

```bash
# Step 1: Create your .kurodlc.json file
# (Add your CostumeParam, ItemTableData, DLCTableData sections)

# Step 2: Check for ID conflicts
python resolve_id_conflicts_in_kurodlc.py checkbydlc

# Step 3: Fix any conflicts automatically
python resolve_id_conflicts_in_kurodlc.py repair --apply

# Step 4: Generate shop assignment template
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json --generate-template

# Step 5: (Optional) Edit template_my_mod.kurodlc.json
# - Adjust shop_ids if needed
# - Customize template structure

# Step 6: Generate shop assignments
python shops_create.py template_my_mod.kurodlc.json

# Step 7: Copy ShopItem section from output file into your .kurodlc.json

# Done! Your mod is ready with:
# ✅ No ID conflicts
# ✅ All items available in shops
```

### Workflow 2: Updating Existing DLC

```bash
# If you modified your DLC and need to re-check:

# Step 1: Check for new conflicts
python resolve_id_conflicts_in_kurodlc.py repair

# Step 2: Apply fixes if needed
python resolve_id_conflicts_in_kurodlc.py repair --apply

# Step 3: Regenerate shop assignments if items changed
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json --generate-template
python shops_create.py template_my_mod.kurodlc.json
```

### Workflow 3: CI/CD Pipeline Integration

```bash
# Automated build pipeline example

# Step 1: Validate and fix conflicts
python resolve_id_conflicts_in_kurodlc.py repair --apply

# Step 2: Generate shop assignments (non-interactive)
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json --generate-template --default-shop-ids --no-interactive
python shops_create.py template_my_mod.kurodlc.json

# Step 3: Merge output into final .kurodlc.json
# (Your merge script here)
```

---

## 📄 File Formats

### .kurodlc.json Structure

```json
{
  "CostumeParam": [
    {
      "item_id": 3596,
      "char_restrict": 1,
      "type": 0,
      "mdl_name": "c_van01b",
      ...
    }
  ],
  "ItemTableData": [
    {
      "id": 3596,
      "name": "My Costume",
      "desc": "Description",
      "category": 17,
      ...
    }
  ],
  "DLCTableData": [
    {
      "id": 1,
      "items": [3596, 3597, 3598],
      "name": "My DLC Pack",
      ...
    }
  ],
  "ShopItem": [
    {
      "shop_id": 1,
      "item_id": 3596,
      "unknown": 1,
      ...
    }
  ]
}
```

### Template Configuration (for shops_create.py)

```json
{
  "_comment": ["Generated template"],
  "item_ids": [3596, 3597, 3598],
  "shop_ids": [1, 5, 10],
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

## 🔧 Troubleshooting

### Common Issues

**1. "No module named 'colorama'"**
```bash
pip install colorama --break-system-packages
```

**2. "No module named 'lz4'" (when using .tbl or .p3a files)**
```bash
pip install lz4 zstandard xxhash --break-system-packages
```

**3. "EOFError" when generating templates**
- **Solution:** Use `--no-interactive` flag
```bash
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json --generate-template --default-shop-ids --no-interactive
```

**4. DLC has no ShopItem section**
- **Solution:** Use `--shop-ids` or `--default-shop-ids`
```bash
python shops_find_unique_item_id_from_kurodlc.py my_mod.kurodlc.json --generate-template --shop-ids=1,5,10
```

**5. "Invalid .kurodlc.json structure"**
- Ensure your file has required sections: `CostumeParam`, `DLCTableData`
- Validate JSON syntax using a JSON validator

**6. IDs outside 1-5000 range**
- The Kuro engine only supports IDs 1-5000
- Use `resolve_id_conflicts_in_kurodlc.py` to reassign to valid range

---

## 💡 Best Practices

### ID Management
1. **Always check for conflicts** before releasing your mod
2. **Use the smart algorithm** (v2.7) for better ID distribution
3. **Keep backups** - automatic backups are created, but keep your own too
4. **Export ID mappings** if you need to manually adjust specific IDs
5. **Test your mod** after applying ID changes

### Shop Assignments
1. **Review templates** before generating assignments
2. **Use meaningful shop IDs** that match actual game shops
3. **Customize templates** for different item types if needed
4. **Test in-game** - verify items appear in correct shops

### Development Workflow
1. **Work with JSON files** - easier to edit and track changes
2. **Use version control** (git) to track modifications
3. **Document your changes** in commit messages
4. **Test incrementally** - don't change everything at once

---

## 📝 Version History

### v2.7.1 (2026-01-31) - resolve_id_conflicts BUGFIX
- **Fixed:** Removed bare except clause (line 929)
- **Improved:** 100% code quality score
- **Improved:** Better error handling for timestamp parsing
- **Status:** Production ready

### v2.7.0 (2026-01-31) - resolve_id_conflicts SMART ALGORITHM
- **New:** Smart ID distribution algorithm
- **New:** Middle-out assignment starting from ID 2500
- **New:** Better ID spacing for cleaner organization
- **New:** Enforced 1-5000 range limit
- **Improved:** More efficient ID allocation

### v2.1 (2026-01-31) - shops_find BUGFIX & CI/CD SUPPORT
- **Fixed:** EOFError in non-interactive environments
- **New:** `--no-interactive` flag for CI/CD pipelines
- **New:** `--default-shop-ids` flag for automatic fallback
- **Improved:** Clear error messages with actionable solutions
- **Improved:** Better handling of DLCs without ShopItem section
- **Improved:** 100% code quality score
- **Status:** Production ready

### v2.0 (2026-01-31) - shops_find TEMPLATE GENERATION
- **New:** Template generation for shops_create.py
- **New:** Auto-extract item IDs from DLC
- **New:** Auto-extract shop IDs from ShopItem section
- **New:** Auto-extract template structure
- **New:** Custom output filenames
- **Improved:** Workflow integration with shops_create.py

### v2.0 (2026-01-31) - shops_create ENHANCED
- **New:** Variable substitution: ${shop_id}, ${item_id}, ${index}, ${count}
- **New:** Custom output sections
- **New:** Custom templates support
- **Improved:** Backward compatible with v1.0
- **Improved:** Better error messages

### v1.0 (2025) - Initial Release
- Basic ID extraction
- Shop assignment generation
- Item discovery tools
- Conflict detection

---

## 🔗 External Dependencies

This toolkit uses the following external libraries:

### From [eArmada8/kuro_dlc_tool](https://github.com/eArmada8/kuro_dlc_tool)
- **`p3a_lib.py`** - P3A archive handling
- **`kurodlc_lib.py`** - Kuro table (.tbl) file handling

**License:** GPL-3.0  
**Author:** eArmada8  
**Repository:** https://github.com/eArmada8/kuro_dlc_tool

**Note:** These libraries are included in this repository for convenience. All credit for these components goes to the original author.

### Python Packages
- **`colorama`** - Cross-platform colored terminal output
  - License: BSD
  - Used for: User-friendly colored output in Windows CMD

- **`lz4`** (optional) - LZ4 compression
- **`zstandard`** (optional) - Zstandard compression  
- **`xxhash`** (optional) - xxHash hashing
  - Required for: P3A and TBL file support

---

## 🤝 Contributing

Contributions are welcome! Since this project is GPL-3.0, all contributions must also be GPL-3.0 compatible.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature
   ```
3. **Make your changes**
   - Follow existing code style
   - Add comments for complex logic
   - Update documentation if needed
4. **Test your changes**
   - Test with real DLC files
   - Verify backward compatibility
5. **Submit a pull request**
   - Describe your changes clearly
   - Reference any related issues

### Code Style
- Use Python 3.7+ compatible syntax
- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Use `with open()` for file operations
- Avoid bare `except:` clauses

### Testing
- Test with both JSON and TBL sources
- Test with various DLC structures
- Verify error handling
- Check Windows CMD compatibility

---

## 📜 License

### GPL-3.0 License

This project is licensed under the **GNU General Public License v3.0**.

**Why GPL-3.0?**

This toolkit uses libraries from [eArmada8/kuro_dlc_tool](https://github.com/eArmada8/kuro_dlc_tool) which are licensed under GPL-3.0. Under GPL-3.0 terms, any software that incorporates GPL-3.0 licensed code must also be licensed under GPL-3.0.

**What This Means for You:**

✅ **You CAN:**
- Use this toolkit freely for any purpose
- Modify the toolkit for your needs
- Distribute the toolkit to others
- Distribute modified versions
- Use it in your modding projects

⚠️ **You MUST:**
- Keep the GPL-3.0 license with the software
- Make source code available to recipients
- License any modifications under GPL-3.0
- Document your changes
- Include copyright notices

❌ **You CANNOT:**
- Incorporate this into proprietary/closed-source software
- Change the license to a non-GPL-compatible license
- Remove license or copyright notices

**Full License Text:**

See the [LICENSE](LICENSE) file for the complete GNU General Public License v3.0 text.

**External Libraries:**
- `p3a_lib.py` and `kurodlc_lib.py` are from [eArmada8/kuro_dlc_tool](https://github.com/eArmada8/kuro_dlc_tool) (GPL-3.0)
- All credit for these components goes to the original author

**Questions about the License?**

For more information about GPL-3.0, see:
- https://www.gnu.org/licenses/gpl-3.0.html
- https://choosealicense.com/licenses/gpl-3.0/

---

## 🙏 Acknowledgments

- **eArmada8** - For the original [kuro_dlc_tool](https://github.com/eArmada8/kuro_dlc_tool) libraries (p3a_lib.py, kurodlc_lib.py)
- **The Kuro modding community** - For testing and feedback
- **All contributors** - Thank you for your contributions!

---

## 📚 Advanced Documentation

For comprehensive, in-depth documentation including:
- **Complete parameter reference** for all scripts
- **Real data examples** from t_item.json, t_shop.json, t_costume.json
- **Data structure specifications** (.kurodlc.json, exports, imports, logs)
- **Advanced workflows** (CI/CD, batch processing, manual ID mapping)
- **Real-world scenarios** with actual game data

See **[ADVANCED_DOCUMENTATION.md](ADVANCED_DOCUMENTATION.md)** ⭐

**What's included:**
- ✅ All script parameters documented
- ✅ Examples with real game data (Sepith items: 310-318, Shops: 5,6,10,21-23)
- ✅ Complete .kurodlc.json structure specification
- ✅ Export/Import format specifications
- ✅ Log file formats and examples
- ✅ Advanced workflows (CI/CD, batch processing, custom ID mapping)
- ✅ Real-world examples (costume packs, large collections)

**Quick links:**
- [Script Parameter Reference](ADVANCED_DOCUMENTATION.md#script-reference)
- [Data Structure Specs](ADVANCED_DOCUMENTATION.md#data-structure-specifications)
- [Real Data Examples](ADVANCED_DOCUMENTATION.md#real-data-examples)
- [Advanced Workflows](ADVANCED_DOCUMENTATION.md#advanced-workflows)

---

## 📧 Support

- **Issues:** [GitHub Issues](https://github.com/falco81/KuroDLCTools/issues)

---

<p align="center">
  <strong>Happy Modding! 🎮</strong>
</p>

<p align="center">
  Made with ❤️ for the Kuro modding community
</p>
