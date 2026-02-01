# -*- coding: utf-8 -*-
"""
Script: generate_schema_auto.py
Author: M365 Copilot
Purpose (universal):
 • Takes one or more JSON files (or scans the current directory),
 • finds the matching TBL (.tbl or .tbl.txt) with the same base name,
 • from each JSON+TBL pair infers schemas for all common sections,
 • validates schemas and precisely aligns them to the real entry_length from the TBL,
 • and automatically adds/updates them in kurodlc_schema.json.
Usage:
  python generate_schema_auto.py                    # process all *.json with a matching *.tbl / *.tbl.txt
  python generate_schema_auto.py t_name.json       # process only the specified JSONs
  python generate_schema_auto.py a.json b.json     # multiple files
Notes:
 – JSON is expected in the converter format, e.g. {"data":[{"name":"Section","data":[...]}, ...]}.
 – Type heuristics: text-like keys → text (Q), *_id / id / character_id → I, short*→H, byte*→B, float*→f,
   list → a (offset Q + count I). If the computed size doesn't match, the script reduces widths (I→H/B where it makes sense),
   and any missing bytes are padded by adding reservedN (I) at the end.
 – The TBL file is never modified; only kurodlc_schema.json (UTF-8 without BOM) is generated/updated.
"""
import sys
import os
import json
import struct
from typing import Dict, List, Tuple

# ---- constants ----
SIZE = {'I': 4, 'H': 2, 'B': 1, 'Q': 8, 'f': 4}
VALCODE_NUM, VALCODE_TEXT, VALCODE_ARRAY = 'n', 't', 'a'

TEXT_LIKE_PREFIXES = (
    'name', 'model', 'face', 'script', 'text', 'full_name', 'filename', 'desc', 'type_desc',
)
INT16_HINT_PREFIXES = (
    'short',
)
BYTE_HINT_PREFIXES = (
    'byte',
)
PREFER_PK_ORDER = ('id', 'character_id', 'effect_id')

# ---- help ----
HELP_TEXT = r"""Script: generate_schema_auto.py
Author: M365 Copilot

Purpose (universal):
 • Takes one or more JSON files (or scans the current directory),
 • finds the matching TBL (.tbl or .tbl.txt) with the same base name,
 • from each JSON+TBL pair infers schemas for all common sections,
 • validates schemas and precisely aligns them to the real entry_length from the TBL,
 • and automatically adds/updates them in kurodlc_schema.json.

Usage:
  python generate_schema_auto.py                    # process all *.json with a matching *.tbl / *.tbl.txt
  python generate_schema_auto.py t_name.json       # process only the specified JSONs
  python generate_schema_auto.py a.json b.json     # multiple files

Notes:
  – JSON is expected in converter format, e.g. {"data":[{"name":"Section","data":[...]}, ...]}.
  – Type heuristics: text-like → Q, *_id/id/character_id → I, short*→H, byte*→B, float*→f, list → a (Q offset + I count).
    If computed size mismatches, widths may be reduced (I→H/B) and padding reservedN (I) added to the end.
  – TBL is never modified; only kurodlc_schema.json (UTF-8 without BOM) is written/updated.
"""

def print_help_and_exit() -> None:
    print(HELP_TEXT)
    raise SystemExit(0)

# ---- helpers ----
def find_peer_tbl(json_path: str) -> str:
    base = os.path.splitext(os.path.basename(json_path))[0]
    candidates = [base + '.tbl', base + '.TBL', base + '.tbl.txt', base + '.TBL.TXT']
    folder = os.path.dirname(os.path.abspath(json_path))
    lower = {f.lower(): f for f in os.listdir(folder)}
    for c in candidates:
        if os.path.exists(os.path.join(folder, c)):
            return os.path.join(folder, c)
        if c.lower() in lower:
            return os.path.join(folder, lower[c.lower()])
    return ''

def load_tbl_headers(tbl_path: str) -> Dict[str, Tuple[int, int]]:
    headers: Dict[str, Tuple[int, int]] = {}
    with open(tbl_path, 'rb') as f:
        if f.read(4) != b'#TBL':
            raise ValueError(f"File '{tbl_path}' lacks #TBL magic header")
        (num_sections,) = struct.unpack('<I', f.read(4))
        for _ in range(num_sections):
            name_bytes = f.read(64)
            name = name_bytes.split(b'\x00', 1)[0].decode('utf-8', errors='ignore')
            crc, start, entry_len, num_entries = struct.unpack('<4I', f.read(16))
            headers[name] = (entry_len, num_entries)
    return headers

def load_json_sections(json_path: str) -> Dict[str, List[dict]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sections: Dict[str, List[dict]] = {}
    for block in data.get('data', []):
        name = block.get('name')
        rows = block.get('data', [])
        if name and isinstance(rows, list):
            sections[name] = rows
    return sections

def guess_type_from_name(key: str) -> str:
    lk = key.lower()
    if any(lk.startswith(p) for p in TEXT_LIKE_PREFIXES):
        return 'Q'
    if any(lk.startswith(p) for p in INT16_HINT_PREFIXES):
        return 'H'
    if any(lk.startswith(p) for p in BYTE_HINT_PREFIXES):
        return 'B'
    if lk.endswith('_id') or lk in ('id', 'character_id'):
        return 'I'
    if 'float' in lk:
        return 'f'
    if lk.startswith('long'):
        return 'I'
    return ''

def infer_key_types(rows: List[dict]) -> Dict[str, str]:
    types: Dict[str, str] = {}
    # Hints from key names first
    for r in rows:
        for k in r.keys():
            t = guess_type_from_name(k)
            if t:
                types.setdefault(k, t)
    # Infer from values
    for r in rows:
        for k, v in r.items():
            if k in types:
                continue
            if isinstance(v, str):
                types[k] = 'Q'
            elif isinstance(v, list):
                types[k] = 'a'
            elif isinstance(v, float):
                types[k] = 'f'
            elif isinstance(v, int):
                types[k] = 'I'
            elif v is None:
                types[k] = 'Q'
            else:
                types[k] = 'Q'
    return types

def compute_struct_and_values(key_types: Dict[str, str], key_order: List[str]) -> Tuple[str, str, int]:
    struct_parts: List[str] = []
    values_parts: List[str] = []
    total = 0
    for k in key_order:
        t = key_types[k]
        if t == 'a':
            struct_parts.extend(['Q', 'I'])
            total += SIZE['Q'] + SIZE['I']
            values_parts.append(VALCODE_ARRAY)
        elif t == 'Q':
            struct_parts.append('Q'); total += SIZE['Q']; values_parts.append(VALCODE_TEXT)
        elif t == 'f':
            struct_parts.append('f'); total += SIZE['f']; values_parts.append(VALCODE_NUM)
        elif t in ('I', 'H', 'B'):
            struct_parts.append(t); total += SIZE[t]; values_parts.append(VALCODE_NUM)
        else:
            struct_parts.append('Q'); total += SIZE['Q']; values_parts.append(VALCODE_TEXT)
    return ''.join(struct_parts), ''.join(values_parts), total

def adjust_to_entry_len(section: str, key_types: Dict[str, str], key_order: List[str], entry_len: int) -> Tuple[str, str, List[str]]:
    struct_fmt, values, total = compute_struct_and_values(key_types, key_order)
    if total > entry_len:
        # Try to shrink based on hints
        for k in key_order:
            if total <= entry_len: break
            if k.lower().startswith('short') and key_types.get(k) == 'I':
                key_types[k] = 'H'
                struct_fmt, values, total = compute_struct_and_values(key_types, key_order)
        for k in key_order:
            if total <= entry_len: break
            if k.lower().startswith('byte') and key_types.get(k) in ('I', 'H'):
                key_types[k] = 'B'
                struct_fmt, values, total = compute_struct_and_values(key_types, key_order)
        if total > entry_len:
            for k in key_order:
                if total <= entry_len: break
                lk = k.lower()
                if key_types.get(k) == 'I' and lk not in ('id','character_id') and not lk.endswith('_id'):
                    key_types[k] = 'H'
                    struct_fmt, values, total = compute_struct_and_values(key_types, key_order)
    # Pad with reserved* if needed
    r = 0
    while total < entry_len:
        rname = f'reserved{r}'
        while rname in key_types:
            r += 1
            rname = f'reserved{r}'
        key_types[rname] = 'I'
        key_order.append(rname)
        struct_fmt, values, total = compute_struct_and_values(key_types, key_order)
        r += 1
        if r > 128:
            break
    if total != entry_len:
        print(f"[WARN] Section '{section}': calc={total} vs entry_length={entry_len} – please verify mapping.")
    return struct_fmt, values, key_order

def make_schema_block(section: str, entry_len: int, key_types: Dict[str, str], key_order: List[str]) -> dict:
    struct_fmt, values, final_keys = adjust_to_entry_len(section, key_types, key_order[:], entry_len)
    block = {
        'info_comment': 'AutoGen',
        'table_header': section,
        'schema_length': entry_len,
        'schema': {
            'schema': '<' + struct_fmt,
            'sch_len': entry_len,
            'keys': final_keys,
            'values': values,
        }
    }
    for pk in PREFER_PK_ORDER:
        if pk in final_keys:
            block['schema']['primary_key'] = pk
            break
    return block

def merge_into_schema_file(blocks: List[dict], schema_path: str) -> None:
    if os.path.exists(schema_path):
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                base = json.load(f)
        except Exception:
            print('[WARN] kurodlc_schema.json is corrupted – creating a new one.')
            base = []
    else:
        base = []
    if not isinstance(base, list):
        base = []
    index = {(b.get('table_header'), b.get('schema_length')): i for i, b in enumerate(base)}
    for nb in blocks:
        key = (nb.get('table_header'), nb.get('schema_length'))
        if key in index:
            base[index[key]] = nb
            print(f"[UPDATE] {key[0]} (schema_length={key[1]}) – updated.")
        else:
            base.append(nb)
            print(f"[ADD] {key[0]} (schema_length={key[1]}) – added.")
    with open(schema_path, 'w', encoding='utf-8') as f:
        json.dump(base, f, indent=4, ensure_ascii=False)

def process_pair(json_path: str, tbl_path: str, schema_path: str) -> None:
    print('=' * 80)
    print(f'[PAIR] JSON: {json_path}')
    print(f'[PAIR] TBL : {tbl_path}')
    headers = load_tbl_headers(tbl_path)
    print('[INFO] Sections in TBL:')
    for n, (el, cnt) in headers.items():
        print(f' - {n}: entry_length={el}, entries={cnt}')
    json_sections = load_json_sections(json_path)
    common = [s for s in json_sections.keys() if s in headers]
    if not common:
        print('[INFO] No sections in JSON match the TBL – skipped.')
        return
    blocks: List[dict] = []
    for name in common:
        rows = json_sections.get(name, [])
        if not rows:
            print(f"[SKIP] Section '{name}' has empty data – skipped.")
            continue
        entry_len, _ = headers[name]
        key_types = infer_key_types(rows)
        order: List[str] = []
        for pk in PREFER_PK_ORDER:
            if pk in key_types:
                order.append(pk)
        for k in rows[0].keys():
            if k not in order:
                order.append(k)
        for r in rows[1:5]:
            for k in r.keys():
                if k not in order:
                    order.append(k)
        order = [k for k in order if k in key_types]
        block = make_schema_block(name, entry_len, key_types, order)
        calc = 0
        for ch in block['schema']['schema'][1:]:
            calc += SIZE.get(ch, 0)
        if calc == entry_len:
            print(f"[OK] {name}: length {entry_len} B matches.")
        else:
            print(f"[WARN] {name}: calc={calc} vs entry={entry_len} – saving schema anyway (please check).")
        blocks.append(block)
    if blocks:
        merge_into_schema_file(blocks, schema_path)
        print(f'[DONE] Written to {schema_path}.')
    else:
        print('[INFO] No blocks were created.')

def main(argv: List[str]) -> None:
    # --help / -h / /? support (run BEFORE any argument processing)
    if any(a in ('--help', '-h', '/?') for a in argv[1:]):
        print_help_and_exit()

    json_files: List[str] = []
    if len(argv) > 1:
        for a in argv[1:]:
            if not a.lower().endswith('.json'):
                print(f"[SKIP] '{a}' – not a JSON file.")
                continue
            # ignore the target schema file as a data source
            if os.path.basename(a).lower() == 'kurodlc_schema.json':
                print(f"[SKIP] '{a}' – schema file is not a data source.")
                continue
            if os.path.isfile(a):
                json_files.append(a)
            else:
                print(f"[SKIP] '{a}' – file not found.")
    else:
        json_files = [
            f for f in os.listdir('.')
            if f.lower().endswith('.json') and f.lower() != 'kurodlc_schema.json'
        ]
    if not json_files:
        print('[INFO] No JSON files found to process.')
        return

    for jp in json_files:
        tp = find_peer_tbl(jp)
        if not tp:
            print('=' * 80)
            print(f"[MISS] {jp}: Matching TBL not found (looking for .tbl / .tbl.txt in the same directory).")
            continue
        process_pair(jp, tp, schema_path='kurodlc_schema.json')

if __name__ == '__main__':
    main(sys.argv)
