# Example ID Allocation Map - Visual Guide

## Interactive HTML Report Preview

The `visualize_id_allocation.py` script generates an interactive HTML report with the following sections:

### 1. Statistics Dashboard
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

### 2. Visual ID Map (100-column grid)

**Legend:**
- 🟩 Green cells = Occupied IDs (items exist)
- ⬜ Gray cells = Free IDs (available for use)

**Example Section (IDs 0-99):**
```
   0-9:   🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩
  10-19:  🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩
  20-29:  🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩
  30-39:  🟩🟩🟩🟩🟩🟩🟩🟩⬜⬜
  40-49:  ⬜🟩🟩🟩🟩🟩🟩🟩🟩🟩
  50-59:  🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩
  60-69:  🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩
  70-79:  🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩
  80-89:  🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩
  90-99:  🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩
```

**Interactive Features:**
- Hover over any cell to see exact ID number
- Search box to jump to specific ID
- Color-coded for easy visualization
- Responsive grid layout

### 3. Free Blocks Table

The HTML report includes a sortable table of all available ID ranges:

| Start ID | End ID | Size | Actions |
|----------|--------|------|---------|
| 38 | 40 | 3 IDs | Use this block |
| 150 | 157 | 8 IDs | Use this block |
| 350 | 425 | 76 IDs | Use this block |
| 1200 | 1245 | 46 IDs | Use this block |
| 3500 | 3650 | 151 IDs | ✨ **Best choice!** |
| 4922 | 5000 | 79 IDs | Use this block |

**Features:**
- Click column headers to sort
- Shows block size for easy selection
- Highlights large blocks (>50 IDs)
- Perfect for planning mod ID ranges

### 4. Search Functionality

The HTML includes a search box that allows you to:
- Jump to specific ID numbers
- Find occupied vs free IDs instantly
- Navigate large ID ranges quickly

**Search Examples:**
```
Search: "3500"     → Shows ID 3500 (green if occupied, gray if free)
Search: "100-150"  → Highlights range 100-150
Search: "free"     → Highlights all free IDs
```

## Console Output Example

When running without `--format=html`, you get a color-coded console view:

```
ID Allocation Map (Block Size: 50)
═══════════════════════════════════════════════════════════

    0: ████████████████████████████████████████████████ [  0 -  49]  100.0%
   50: ████████████████████████████████████████████████ [ 50 -  99]  100.0%
  100: ████████████████████████████████████████████████ [100 - 149]  100.0%
  150: ███████████████████████████████████░░░░░░░░░░░░░ [150 - 199]   76.0%
  200: ████████████████████████████████████████████████ [200 - 249]  100.0%
  ...
 3500: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ [3500-3549]    0.0%  ✨ LARGE FREE BLOCK!
 3550: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ [3550-3599]    0.0%
 3600: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ [3600-3649]    0.0%
  ...
 4950: ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ [4950-4999]   12.0%

Legend: █ Occupied  ░ Free
```

## Usage Tips

1. **Planning New Mods:**
   - Run `visualize_id_allocation.py` before starting
   - Look for blocks with >50 free IDs
   - Note the block range (e.g., 3500-3650)
   - Use those IDs in your .kurodlc.json

2. **Team Coordination:**
   - Generate HTML report
   - Share with team members
   - Coordinate ID ranges to avoid conflicts
   - Example: Team A uses 3500-3599, Team B uses 3600-3699

3. **Fragmentation Analysis:**
   - High fragmentation (>0.7) = many small gaps
   - Low fragmentation (<0.3) = few large gaps
   - Prefer low fragmentation for easier management

4. **Finding Safe Ranges:**
   - Look at "Largest Free Block" statistic
   - Check Free Blocks table in HTML
   - Choose blocks that fit your item count
   - Leave some buffer space for future additions
