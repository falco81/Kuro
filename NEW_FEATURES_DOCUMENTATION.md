# Advanced Documentation - NEW FEATURES (v3.0)

This document covers the new features added in version 3.0: ID Allocation Visualization and 3D Model Viewing.

## 📋 Table of Contents

- [visualize_id_allocation.py](#visualize_id_allocationpy)
- [3D Model Viewer Scripts](#3d-model-viewer-scripts)
- [Visualization Workflows](#visualization-workflows)
- [3D Model Viewing Workflows](#3d-model-viewing-workflows)

---

## visualize_id_allocation.py

**Purpose:** Visualize ID allocation patterns, analyze gaps, and find safe ID ranges for modding

**Version:** v3.0

### All Parameters

```
visualize_id_allocation.py [options]

OPTIONS:
  --source=TYPE       Force specific source type
                      Available: json, tbl, original, p3a, zzz
                      
  --no-interactive    Auto-select first source if multiple found
                      
  --keep-extracted    Keep temporary extracted files from P3A
                      
  --format=FORMAT     Output format:
                        console - Terminal visualization only
                        html    - HTML report only
                        both    - Both formats (default)
                      
  --block-size=N      Block size for visualization (default: 50)
                      Larger values = more compact display
                      Smaller values = more granular display
                      
  --output=FILE       HTML output filename (default: id_allocation_map.html)
                      
  --help              Show help message

SOURCES (automatically detected):

Game Database Sources:
  JSON sources:
    - t_item.json
    
  TBL sources (requires kurodlc_lib.py):
    - t_item.tbl
    - t_item.tbl.original
    
  P3A sources (requires kurodlc_lib.py + dependencies):
    - script_en.p3a / script_eng.p3a
    - zzz_combined_tables.p3a
```

### Usage Examples

**Example 1: Default Usage (Both Formats)**

```bash
python visualize_id_allocation.py
```

**Output:**
- Console visualization with statistics
- `id_allocation_map.html` interactive report

**Example 2: Console Only**

```bash
python visualize_id_allocation.py --format=console
```

**Sample Console Output:**
```
ID Allocation Map (Block Size: 50)
═══════════════════════════════════════════════════════════

Engine Range:  1 - 5000 (Kuro Engine limit)
Total IDs:     5000
Occupied:      2116 (42.3%)
Free:          2884 (57.7%)

Statistics:
---------------------------------------------------------------------------
Highest Used ID:        4921
Largest Free Block:     79 IDs (4922-5000)
Total Free Blocks:      370
Average Gap Size:       7.8 IDs
Fragmentation:          0.73 (73%)  ⚠ High fragmentation

ID Allocation Grid:
---------------------------------------------------------------------------
    0: ████████████████████████████████████████████████ [  0 -  49]  100.0%
   50: ████████████████████████████████████████████████ [ 50 -  99]  100.0%
  100: ████████████████████████████████████████████████ [100 - 149]  100.0%
  150: ███████████████████████████████░░░░░░░░░░░░░░░░ [150 - 199]   76.0%
  ...
 3500: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ [3500-3549]    0.0%  ✨
 3550: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ [3550-3599]    0.0%
  ...
 4950: ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ [4950-4999]   12.0%

Legend: █ Occupied  ░ Free  ✨ Large Free Block (>50 IDs)

Free Blocks Analysis:
---------------------------------------------------------------------------
Block Range          Size    Location
---------------------------------------------------------------------------
4922 - 5000          79      End of range (✨ BEST CHOICE)
3500 - 3650          151     Middle range (✨ EXCELLENT)
1200 - 1245          46      Lower-middle range
350 - 425            76      Lower range (✨ GOOD)
150 - 157            8       Lower range
38 - 40              3       Start range
...

Recommendations:
---------------------------------------------------------------------------
✓ For small mods (1-50 items):   Use 4922-5000 (79 free)
✓ For medium mods (50-100 items): Use 3500-3650 (151 free)
✓ For large mods (100+ items):    Multiple blocks needed
⚠ High fragmentation detected - consider coordinated ID planning

Press Enter to exit...
```

**Example 3: HTML Only with Custom Name**

```bash
python visualize_id_allocation.py --format=html --output=team_report.html
```

**Example 4: Custom Block Size**

```bash
# Larger blocks for compact view
python visualize_id_allocation.py --block-size=100

# Smaller blocks for detailed view
python visualize_id_allocation.py --block-size=25
```

**Example 5: Force Specific Source**

```bash
# Use JSON source
python visualize_id_allocation.py --source=json

# Use TBL source
python visualize_id_allocation.py --source=tbl

# Use P3A archive
python visualize_id_allocation.py --source=p3a
```

### HTML Report Features

The generated HTML report includes:

#### 1. Statistics Dashboard

```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│   Engine Range      │  Highest Used ID    │   Occupied IDs      │
│     1 - 5000        │       4921          │   2116 / 5000       │
│  Kuro Engine limit  │   5000 total IDs    │      42.3%          │
└─────────────────────┴─────────────────────┴─────────────────────┘

┌─────────────────────┬─────────────────────┬─────────────────────┐
│     Free IDs        │  Average Gap Size   │  Fragmentation      │
│   2884 / 5000       │      7.8 IDs        │   0.73 (73%)        │
│      57.7%          │                     │   High fragmentation │
└─────────────────────┴─────────────────────┴─────────────────────┘

┌─────────────────────┬─────────────────────┐
│ Largest Free Block  │  Total Free Blocks  │
│   79 IDs            │        370          │
│  (4922-5000)        │                     │
└─────────────────────┴─────────────────────┘
```

#### 2. Interactive ID Map

- **Color Coding:**
  - 🟩 Green cells = Occupied IDs
  - ⬜ Gray cells = Free IDs

- **Interactive Features:**
  - Hover over cells to see exact ID numbers
  - Click to highlight blocks
  - Search box to jump to specific IDs
  - Responsive grid layout

**Example Grid Section (IDs 0-99):**
```
   0-9:   🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩
  10-19:  🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩
  20-29:  🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩
  30-39:  🟩🟩🟩🟩🟩🟩🟩🟩⬜⬜  ← Free IDs: 38-39
  40-49:  ⬜🟩🟩🟩🟩🟩🟩🟩🟩🟩  ← Free ID: 40
  ...
```

#### 3. Free Blocks Table

Sortable table showing all available ID ranges:

| Start ID | End ID | Size | Quality | Actions |
|----------|--------|------|---------|---------|
| 3500 | 3650 | 151 IDs | ✨ EXCELLENT | [Use this block] |
| 4922 | 5000 | 79 IDs | ✨ BEST | [Use this block] |
| 350 | 425 | 76 IDs | ✨ GOOD | [Use this block] |
| 1200 | 1245 | 46 IDs | ✓ Good | [Use this block] |
| 150 | 157 | 8 IDs | • Small | [Use this block] |
| 38 | 40 | 3 IDs | • Tiny | [Use this block] |

**Features:**
- Click column headers to sort (by size, start ID, etc.)
- Quality indicators based on block size
- Direct jump to block in visualization

#### 4. Search Functionality

**Search Examples:**
```
Search: "3500"     → Jumps to ID 3500, shows status
Search: "100-150"  → Highlights range 100-150
Search: "free"     → Highlights all free IDs
Search: "occupied" → Highlights all occupied IDs
```

### Statistics Explained

#### Fragmentation Metric

Fragmentation measures how scattered the free IDs are:

```
Fragmentation = (Total Free Blocks - 1) / Free IDs
```

**Interpretation:**
- **0.0 - 0.3** (Low): Few large blocks, ideal for modding
- **0.3 - 0.7** (Medium): Mix of small and large blocks
- **0.7 - 1.0** (High): Many small gaps, challenging for large mods

**Example:**
```
Total Free IDs:    2884
Total Free Blocks: 370
Fragmentation:     (370 - 1) / 2884 = 0.128 (Low) ✓

vs.

Total Free IDs:    2884
Total Free Blocks: 2100
Fragmentation:     (2100 - 1) / 2884 = 0.727 (High) ⚠
```

#### Average Gap Size

```
Average Gap Size = Total Free IDs / Total Free Blocks
```

Larger values indicate better ID availability.

### Data Loading

The script loads item IDs from the same sources as resolve_id_conflicts_in_kurodlc.py:

1. **t_item.json** (preferred for speed)
2. **t_item.tbl** (requires kurodlc_lib.py)
3. **t_item.tbl.original** (TBL backup)
4. **P3A archives** (script_en.p3a, zzz_combined_tables.p3a)

### Real-World Scenarios

#### Scenario 1: Planning a Costume Pack (50 items)

```bash
# Step 1: Visualize allocation
python visualize_id_allocation.py

# Step 2: Check report - finds block 3500-3650 (151 free)

# Step 3: Use IDs 3500-3549 for your 50 costumes

# Step 4: Reserve 3550-3650 for future additions
```

#### Scenario 2: Team Coordination

```bash
# Team lead generates report
python visualize_id_allocation.py --format=html --output=team_id_map.html

# Share team_id_map.html

# Assign ranges:
# - Team A: 3500-3599 (100 IDs)
# - Team B: 3600-3699 (100 IDs)
# - Team C: 3700-3799 (100 IDs)
```

#### Scenario 3: Finding Safe Range for Large Mod

```bash
# Visualize with smaller block size for detail
python visualize_id_allocation.py --block-size=25

# Look for largest free block
# Report shows: 3500-3650 (151 IDs)

# Plan your mod:
# - Main items: 3500-3600 (100 items)
# - DLC extras: 3601-3650 (50 items)
```

#### Scenario 4: Fragmentation Analysis

```bash
python visualize_id_allocation.py --format=console

# Output shows:
# Fragmentation: 0.73 (High) ⚠

# Analysis:
# - Many small gaps (1-10 IDs)
# - Few large blocks
# - Need to coordinate with other modders
# - Consider using IDs from largest block first
```

### Integration with Other Tools

#### With resolve_id_conflicts_in_kurodlc.py

```bash
# Step 1: Visualize to find safe ranges
python visualize_id_allocation.py
# Note: Block 3500-3650 is free

# Step 2: Create your DLC using IDs in safe range
# Edit my_mod.kurodlc.json - use IDs 3500-3549

# Step 3: Verify no conflicts
python resolve_id_conflicts_in_kurodlc.py checkbydlc
# All OK!
```

#### For Team Projects

```bash
# Generate HTML report for the team
python visualize_id_allocation.py --output=project_id_map.html

# Share the HTML file

# Each team member checks which ranges are free
# Coordinate ID assignments via team communication

# Everyone uses resolve_id_conflicts to verify
python resolve_id_conflicts_in_kurodlc.py checkbydlc
```

---

## 3D Model Viewer Scripts

The toolkit includes four different viewer scripts for .mdl files (Kuro engine 3D models). Each has different trade-offs between convenience, features, and output format.

### Quick Comparison

| Script | Output | Dependencies | Best For |
|--------|--------|--------------|----------|
| `viewer.py` | HTML file | numpy | Standalone viewing, sharing |
| `viewer_mdl.py` | HTML file | numpy, three.js | Standard use, web preview |
| `viewer_mdl_optimized.py` | HTML file | numpy | Large models, performance |
| `viewer_mdl_window.py` | Native window | numpy, pywebview | Quick preview, no files |

### Common Requirements

All viewers require:
```bash
pip install numpy --break-system-packages

# For encrypted CLE assets:
pip install blowfish zstandard --break-system-packages
```

Additionally, viewer_mdl_window.py requires:
```bash
pip install pywebview --break-system-packages
```

### Common Parameters

All viewers support:

```
<model.mdl>              Input .mdl file (required)
--use-original-normals   Use normals from model file instead of computing smooth normals
```

---

### viewer.py

**Purpose:** Standalone core viewer with minimal dependencies

**Features:**
- Complete HTML generation from scratch
- No external Three.js file required (embedded)
- Minimal code footprint
- Good for understanding the basics

**Usage:**
```bash
python viewer_mdl/viewer.py character.mdl
python viewer_mdl/viewer.py character.mdl --use-original-normals
```

**Output:**
- `<model_name>_viewer.html` - Self-contained HTML file

**When to use:**
- Need standalone viewer
- Want to understand viewer implementation
- Minimal dependencies preferred

---

### viewer_mdl.py

**Purpose:** Standard HTML viewer with Three.js integration

**Features:**
- Uses three.min.js from viewer_mdl folder
- Standard feature set
- Balanced performance
- Automatic smooth normal computation

**Usage:**
```bash
python viewer_mdl/viewer_mdl.py character.mdl
python viewer_mdl/viewer_mdl.py weapon.mdl --use-original-normals
```

**Output:**
- `<model_name>_viewer.html`

**Example Output HTML Features:**
- Interactive 3D rotation (mouse drag)
- Zoom (mouse wheel)
- Camera controls
- Lighting setup
- Material rendering

**When to use:**
- Standard 3D model viewing
- Sharing models via HTML
- Web-based model inspection

---

### viewer_mdl_optimized.py

**Purpose:** Performance-optimized version for large models

**Features:**
- Optimized mesh processing
- Faster normal computation
- Better memory usage
- Improved loading times

**Usage:**
```bash
python viewer_mdl/viewer_mdl_optimized.py large_model.mdl
```

**Output:**
- `<model_name>_viewer.html`

**Optimizations:**
- Efficient vertex buffer handling
- Optimized normal averaging algorithm
- Reduced memory allocations
- Faster HTML generation

**When to use:**
- Large models (>100K vertices)
- Batch processing many models
- Performance-critical workflows

---

### viewer_mdl_window.py

**Purpose:** Native window viewer without creating files

**Features:**
- Opens in native window (no HTML file created)
- Automatic cleanup on exit
- No temporary files left behind
- Platform-native WebView integration

**Platform Support:**
- **Windows:** Edge WebView2 (built-in on Windows 10/11)
- **Linux:** GTK + WebKit2
- **macOS:** WKWebView (native)

**Usage:**
```bash
python viewer_mdl/viewer_mdl_window.py character.mdl
python viewer_mdl/viewer_mdl_window.py character.mdl --use-original-normals
```

**Output:**
- Native window with 3D preview
- NO files created
- Automatic cleanup when window closes

**Cleanup Behavior:**
```python
# Temporary files are tracked and removed automatically
# When window closes:
#   1. Delete temp vertex/index buffers
#   2. Delete temp .fmt files
#   3. Remove temp directories
#   4. Exit cleanly
```

**When to use:**
- Quick model preview
- Don't want HTML files
- Native window experience preferred
- Rapid iteration on models

**Installation Note:**
```bash
# Windows
pip install pywebview --break-system-packages
# WebView2 usually pre-installed

# Linux
pip install pywebview --break-system-packages
sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-webkit2-4.0

# macOS
pip install pywebview --break-system-packages
# WKWebView included in macOS
```

---

### Normal Computation

All viewers support two normal modes:

#### Computed Smooth Normals (Default)

```
Algorithm: Position-sharing normal averaging
- Groups vertices by position (within tolerance)
- Computes face normals for all triangles
- Averages normals for vertices at same position
- Produces smooth shading across shared edges
```

**Advantages:**
- Smooth appearance
- Better for curved surfaces
- No artifacts at seams

**Example:**
```bash
python viewer_mdl/viewer_mdl.py character.mdl
# Uses computed smooth normals (default)
```

#### Original Normals

```
Uses normals stored in .mdl file
- Reads NORMAL or NORMAL0 semantic
- Preserves sharp edges if authored
- Shows model exactly as designed
```

**Advantages:**
- Preserves artist intent
- Shows hard edges correctly
- Matches in-game appearance

**Example:**
```bash
python viewer_mdl/viewer_mdl.py character.mdl --use-original-normals
```

**When to use original normals:**
- Model looks wrong with computed normals
- Need to match in-game appearance exactly
- Model has intentional hard edges

---

### Real-World Usage Examples

#### Example 1: Quick Model Check

```bash
# Fast preview without creating files
python viewer_mdl/viewer_mdl_window.py chr5001.mdl

# Window opens → check model → close window
# No files created!
```

#### Example 2: Batch HTML Generation

```bash
# Generate HTML for all models in folder
for file in *.mdl; do
    echo "Processing $file..."
    python viewer_mdl/viewer_mdl.py "$file"
done

# Result: one .html file per .mdl file
```

#### Example 3: Sharing Models

```bash
# Create HTML for sharing
python viewer_mdl/viewer_mdl.py character.mdl

# Share character_viewer.html via:
# - Email attachment
# - Web hosting
# - Team collaboration tools
```

#### Example 4: Model Comparison

```bash
# Generate HTML for both versions
python viewer_mdl/viewer_mdl.py character_v1.mdl
python viewer_mdl/viewer_mdl.py character_v2.mdl

# Open both HTML files side-by-side
# Compare differences visually
```

#### Example 5: Testing Normal Modes

```bash
# Test with computed normals
python viewer_mdl/viewer_mdl_window.py weapon.mdl

# Test with original normals
python viewer_mdl/viewer_mdl_window.py weapon.mdl --use-original-normals

# Compare which looks better
```

---

## Visualization Workflows

### Workflow 1: New Mod Planning

```bash
# Step 1: Generate visualization
python visualize_id_allocation.py

# Step 2: Review HTML report
# Open id_allocation_map.html in browser

# Step 3: Identify safe ranges
# Example: Report shows 3500-3650 (151 free)

# Step 4: Plan your mod
# Use IDs 3500-3549 for 50 costume items
# Reserve 3550-3599 for future DLC
# Leave 3600-3650 as buffer

# Step 5: Create your .kurodlc.json with planned IDs

# Step 6: Verify
python resolve_id_conflicts_in_kurodlc.py checkbydlc
```

### Workflow 2: Finding Emergency ID Space

```bash
# Crisis: Need 20 more IDs for urgent hotfix

# Step 1: Quick console visualization
python visualize_id_allocation.py --format=console

# Step 2: Look for gaps in output
# Example: Found gaps: 2450-2469 (20 IDs)

# Step 3: Use emergency range
# Add items with IDs 2450-2469

# Step 4: Verify no conflicts
python resolve_id_conflicts_in_kurodlc.py checkbydlc
```

### Workflow 3: Multi-Mod Coordination

```bash
# PROJECT: Three teams working on different content

# Team Lead:
# =========
# Generate master allocation map
python visualize_id_allocation.py --format=html --output=master_id_map.html

# Share master_id_map.html with all teams

# Assign ranges:
# - Team A (Costumes): 3500-3599
# - Team B (Accessories): 3600-3699
# - Team C (Weapons): 3700-3799

# Each Team:
# ==========
# Use assigned range in .kurodlc.json

# Verify own range
python resolve_id_conflicts_in_kurodlc.py checkbydlc

# Team Lead (Integration):
# =======================
# Merge all team DLCs

# Final verification
python resolve_id_conflicts_in_kurodlc.py checkbydlc

# Generate final allocation map
python visualize_id_allocation.py --output=final_id_map.html
```

---

## 3D Model Viewing Workflows

### Workflow 1: Rapid Model Inspection

```bash
# During development, quickly check models

# Loop through work-in-progress models
for model in wip/*.mdl; do
    python viewer_mdl/viewer_mdl_window.py "$model"
    # Window opens, check model, press Close
    # Next model...
done
```

### Workflow 2: Model Export Documentation

```bash
# Generate HTML docs for all character models

# Create output directory
mkdir -p model_previews

# Generate HTML for each model
for model in characters/*.mdl; do
    basename=$(basename "$model" .mdl)
    python viewer_mdl/viewer_mdl.py "$model"
    mv "${basename}_viewer.html" "model_previews/"
done

# Result: model_previews/ contains HTML for all models
# Can be hosted on web server or shared with team
```

---

This document covers the new features in v3.0. For the complete advanced documentation, see ADVANCED_DOCUMENTATION.md.
