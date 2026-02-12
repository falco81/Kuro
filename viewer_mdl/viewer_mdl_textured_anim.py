#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viewer_mdl_textured_anim.py — Direct .mdl preview with TEXTURE SUPPORT + SKELETON + ANIMATIONS

FEATURES:
- Loads and displays textures from DDS files
- Loads skeleton hierarchy from extracted MDL data
- Loads external MI/JSON files (IK, physics, colliders, dynamic bones)
- Shows skeleton visualization with checkbox toggle
- Basic skeleton animations (T-Pose, Idle, Wave, Walk)
- Animates model, skeleton, and wireframe variants
- Windows 10 CLI compatible output

REQUIREMENTS:
  pip install pywebview Pillow

USAGE:
  python viewer_mdl_textured_anim.py /path/to/model.mdl [--recompute-normals] [--debug] [--skip-popup] [--no-shaders]
  
  --recompute-normals  Recompute smooth normals instead of using originals from MDL
                       (slower loading, typically no visual difference)
  --debug              Enable verbose console logging in browser
  --skip-popup         Skip loading progress popup on startup
  --no-shaders         Disable toon shader rendering, use standard PBR materials
"""

from pathlib import Path
import sys
import json
import struct
import numpy as np
import tempfile
import atexit
import time
import shutil
import fractions

# Import parser functions
from kuro_mdl_export_meshes import decryptCLE, obtain_material_data, obtain_mesh_data  # type: ignore

# Import texture loader
try:
    from lib_texture_loader import find_texture_file, DDSHeader
    TEXTURES_AVAILABLE = True
except ImportError:
    print("Warning: lib_texture_loader not found. Textures will not be loaded.")
    TEXTURES_AVAILABLE = False

# Pillow for DDS conversion
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    print("Warning: PIL not installed. DDS conversion will be limited.")
    PIL_AVAILABLE = False


# -----------------------------
# Temp file cleanup
# -----------------------------
TEMP_FILES = []

def cleanup_temp_files():
    """Delete all temporary files on exit."""
    for filepath in TEMP_FILES:
        try:
            if filepath.exists():
                if filepath.is_dir():
                    shutil.rmtree(filepath)
                    print(f"[CLEANUP] Deleted directory: {filepath}")
                else:
                    filepath.unlink()
                    print(f"[CLEANUP] Deleted: {filepath}")
        except Exception:
            pass

atexit.register(cleanup_temp_files)


# -----------------------------
# DDS to PNG conversion
# -----------------------------
def convert_dds_to_png(dds_path: Path, output_path: Path) -> bool:
    """
    Convert DDS file to PNG.
    
    Args:
        dds_path: Path to DDS file
        output_path: Path where to save PNG
        
    Returns:
        True if successful, False otherwise
    """
    if not PIL_AVAILABLE:
        return False
    
    try:
        img = Image.open(dds_path)
        img.save(output_path, 'PNG')
        return True
    except Exception as e:
        print(f"  [!] Failed to convert {dds_path.name}: {e}")
        return False


# -----------------------------
# Smooth normals
# -----------------------------
def compute_smooth_normals_with_sharing(vertices: np.ndarray, indices: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """Compute smooth normals with position sharing using spatial hashing (O(n) instead of O(n²))."""
    n = len(vertices)
    normals = np.zeros((n, 3), dtype=np.float32)
    
    for i in range(0, len(indices), 3):
        i0, i1, i2 = indices[i:i+3]
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        
        norm = np.linalg.norm(face_normal)
        if norm > 1e-12:
            face_normal = face_normal / norm
        
        normals[i0] += face_normal
        normals[i1] += face_normal
        normals[i2] += face_normal
    
    # Position-based normal sharing using spatial hash (O(n) amortized)
    cell_size = tolerance * 10  # Hash cell size slightly larger than tolerance
    if cell_size < 1e-8:
        cell_size = 1e-5
    
    def hash_pos(v):
        return (int(v[0] / cell_size), int(v[1] / cell_size), int(v[2] / cell_size))
    
    # Build spatial hash
    from collections import defaultdict
    cells = defaultdict(list)
    for i in range(n):
        cells[hash_pos(vertices[i])].append(i)
    
    # For each vertex, check its cell and 26 neighbors
    shared = np.zeros((n, 3), dtype=np.float32)
    visited = np.zeros(n, dtype=bool)
    
    for i in range(n):
        if visited[i]:
            continue
        cx, cy, cz = hash_pos(vertices[i])
        matches = [i]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in cells.get((cx+dx, cy+dy, cz+dz), []):
                        if j != i and not visited[j] and np.linalg.norm(vertices[i] - vertices[j]) < tolerance:
                            matches.append(j)
        
        if len(matches) > 1:
            shared_normal = normals[matches].sum(axis=0)
            norm = np.linalg.norm(shared_normal)
            if norm > 1e-12:
                shared_normal = shared_normal / norm
            for idx in matches:
                normals[idx] = shared_normal
                visited[idx] = True
        else:
            visited[i] = True
    
    # Normalize
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 1e-12
    normals[valid] = normals[valid] / norms[valid][:, None]
    
    return normals


# -----------------------------
# Synthetic bones for missing mesh references
# -----------------------------
def decompose_bind_matrix(mat_4x4):
    """Decompose a 4x4 row-major bind matrix into (pos, quat_xyzw, scale).
    
    Row-major convention (DirectX/Kuro):
      Row 0-2: rotation*scale
      Row 3:   translation
    """
    mat = np.array(mat_4x4, dtype=np.float64)
    
    pos = [float(mat[3][0]), float(mat[3][1]), float(mat[3][2])]
    
    # Scale = row lengths of upper-left 3x3
    sx = float(np.linalg.norm(mat[0][:3]))
    sy = float(np.linalg.norm(mat[1][:3]))
    sz = float(np.linalg.norm(mat[2][:3]))
    scale = [sx if sx > 1e-12 else 1.0, sy if sy > 1e-12 else 1.0, sz if sz > 1e-12 else 1.0]
    
    # Rotation matrix (scale removed)
    rot = np.zeros((3, 3), dtype=np.float64)
    rot[0] = np.array(mat[0][:3]) / scale[0]
    rot[1] = np.array(mat[1][:3]) / scale[1]
    rot[2] = np.array(mat[2][:3]) / scale[2]
    
    # Rotation matrix → quaternion (Shepperd's method)
    tr = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (rot[2, 1] - rot[1, 2]) / s, (rot[0, 2] - rot[2, 0]) / s, (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w, x, y, z = (rot[2, 1] - rot[1, 2]) / s, 0.25 * s, (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w, x, y, z = (rot[0, 2] - rot[2, 0]) / s, (rot[0, 1] + rot[1, 0]) / s, 0.25 * s, (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w, x, y, z = (rot[1, 0] - rot[0, 1]) / s, (rot[0, 2] + rot[2, 0]) / s, (rot[1, 2] + rot[2, 1]) / s, 0.25 * s
    
    quat_xyzw = [float(x), float(y), float(z), float(w)]
    return pos, quat_xyzw, scale


def create_synthetic_bones(skeleton_data, skeleton_name_to_idx, global_bind_matrices):
    """Create skeleton entries for bones referenced by meshes but missing from skeleton.
    
    These are typically costume-specific bones (cloth chains, endpoints) that exist in
    mesh vgmaps and some animations, but not in the base MDL skeleton.
    
    Infers parent-child chain hierarchy from naming conventions:
      BC01 → BC02 → BC03 → BC_Top
      LeftCA01 → LeftCA02 → ... → LeftCA_Top
      Head_Top → child of Head (endpoint)
    """
    import re
    from collections import defaultdict
    
    # Find missing bone names
    missing_names = set()
    for name in global_bind_matrices:
        if name not in skeleton_name_to_idx:
            missing_names.add(name)
    
    if not missing_names:
        return 0
    
    # Group into chains by prefix
    chains = defaultdict(list)
    endpoint_bones = []  # *_Top bones whose parent exists in skeleton
    
    for name in missing_names:
        if name.endswith('_Top'):
            # Check if parent bone (without _Top) exists in skeleton
            parent_name = name[:-4]
            if parent_name in skeleton_name_to_idx:
                endpoint_bones.append((name, parent_name))
            else:
                # Part of a missing chain (e.g., BC_Top where BC04 is also missing)
                prefix = parent_name.rstrip('0123456789')
                if not prefix:
                    prefix = parent_name
                chains[prefix].append((999, name))
        else:
            match = re.match(r'^(.+?)(\d+)$', name)
            if match:
                prefix, num = match.group(1), int(match.group(2))
                chains[prefix].append((num, name))
            else:
                chains[name].append((0, name))
    
    # Sort each chain by number
    for prefix in chains:
        chains[prefix].sort()
    
    next_id = max(b['id_referenceonly'] for b in skeleton_data) + 1
    created = 0
    
    def mat4_from_bind(name):
        """Get 4x4 numpy matrix from bind matrices."""
        m = global_bind_matrices[name]
        return np.array(m, dtype=np.float64)
    
    def add_bone(name, parent_id, local_pos, local_quat_xyzw, local_scale):
        """Add a synthetic bone to skeleton_data."""
        nonlocal next_id, created
        
        bone_entry = {
            'id_referenceonly': next_id,
            'name': name,
            'type': 1,
            'mesh_index': -1,
            'pos_xyz': local_pos,
            'unknown_quat': [0.0, 0.0, 0.0, 1.0],
            'skin_mesh': 0,
            'rotation_euler_rpy': [0.0, 0.0, 0.0],  # Dummy, quat_xyzw is authoritative
            'scale': local_scale,
            'unknown': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'children': [],
            'quat_xyzw': local_quat_xyzw,
            'synthetic': True
        }
        
        # Add to parent's children list
        for b in skeleton_data:
            if b['id_referenceonly'] == parent_id:
                b['children'].append(next_id)
                break
        
        skeleton_data.append(bone_entry)
        skeleton_name_to_idx[name] = next_id
        next_id += 1
        created += 1
    
    # 1. Process endpoint bones (e.g., Head_Top → child of Head)
    for name, parent_name in endpoint_bones:
        parent_id = skeleton_name_to_idx[parent_name]
        parent_world = mat4_from_bind(parent_name) if parent_name in global_bind_matrices else np.eye(4, dtype=np.float64)
        child_world = mat4_from_bind(name)
        
        local_mat = np.linalg.inv(parent_world) @ child_world
        pos, quat, scale = decompose_bind_matrix(local_mat)
        add_bone(name, parent_id, pos, quat, scale)
    
    # 2. Process missing chains (e.g., BC01 → BC02 → BC03 → BC04 → BC_Top)
    for prefix, chain_items in sorted(chains.items()):
        # Find chain root parent: try common skeleton bones
        chain_root_parent_id = 0  # Default to root
        
        # Special case: single _Top bone (e.g., LeftHandIndex_Top)
        # Try to find highest-numbered bone with same prefix in skeleton
        if len(chain_items) == 1 and chain_items[0][0] == 999:
            bone_name = chain_items[0][1]
            base = bone_name[:-4]  # Remove _Top
            best_match = None
            best_num = -1
            for skel_name in skeleton_name_to_idx:
                if skel_name.startswith(base) and skel_name != bone_name:
                    match = re.search(r'(\d+)$', skel_name)
                    if match and int(match.group(1)) > best_num:
                        best_num = int(match.group(1))
                        best_match = skel_name
                    elif not match and best_num < 0:
                        best_match = skel_name
            if best_match:
                chain_root_parent_id = skeleton_name_to_idx[best_match]
        
        if chain_root_parent_id == 0:
            # Try to find parent by prefix pattern
            parent_candidates = []
            # Strip Left/Right prefix for matching
            clean_prefix = prefix
            side = ''
            if prefix.startswith('Left'):
                clean_prefix = prefix[4:]
                side = 'Left'
            elif prefix.startswith('Right'):
                clean_prefix = prefix[5:]
                side = 'Right'
            elif prefix.startswith('L_'):
                clean_prefix = prefix[2:]
                side = 'Left'  # Map L_ to Left for parent search
            elif prefix.startswith('R_'):
                clean_prefix = prefix[2:]
                side = 'Right'  # Map R_ to Right for parent search
            
            # Common parent mappings for costume bones
            if clean_prefix in ('C', 'CA', 'CB', 'CC', 'CD', 'CE'):
                parent_candidates = [f'{side}Shoulder', f'{side}Arm', 'Spine2', 'Spine1', 'Spine']
            elif clean_prefix in ('BC', 'FC'):
                parent_candidates = ['Spine', 'Spine1', 'Hips']
            elif clean_prefix.startswith('Bag'):
                parent_candidates = [f'{side}UpLeg', f'{side}Leg', 'Hips']
            
            for cand in parent_candidates:
                if cand in skeleton_name_to_idx:
                    chain_root_parent_id = skeleton_name_to_idx[cand]
                    break
        
        # Build chain with proper hierarchy
        prev_id = chain_root_parent_id
        prev_world = np.eye(4, dtype=np.float64)
        
        # Get parent world matrix
        for b in skeleton_data:
            if b['id_referenceonly'] == chain_root_parent_id and b['name'] in global_bind_matrices:
                prev_world = mat4_from_bind(b['name'])
                break
        
        for _, bone_name in chain_items:
            if bone_name not in global_bind_matrices:
                continue
            
            child_world = mat4_from_bind(bone_name)
            local_mat = np.linalg.inv(prev_world) @ child_world
            pos, quat, scale = decompose_bind_matrix(local_mat)
            
            add_bone(bone_name, prev_id, pos, quat, scale)
            
            prev_id = skeleton_name_to_idx[bone_name]
            prev_world = child_world
    
    return created


# -----------------------------
# Load Skeleton Data (from MDL directly)
# -----------------------------
def rpy2quat(rot_rpy):
    """Convert Roll-Pitch-Yaw Euler angles to quaternion (wxyz format).
    Exact formula from eArmada8's kuro_mdl_to_basic_gltf.py.
    Corresponds to THREE.js Euler order 'ZYX' (intrinsic ZYX = extrinsic XYZ)."""
    import math as _m
    cr = _m.cos(rot_rpy[0] * 0.5)
    sr = _m.sin(rot_rpy[0] * 0.5)
    cp = _m.cos(rot_rpy[1] * 0.5)
    sp = _m.sin(rot_rpy[1] * 0.5)
    cy = _m.cos(rot_rpy[2] * 0.5)
    sy = _m.sin(rot_rpy[2] * 0.5)
    # wxyz
    return [cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy]


def load_skeleton_from_mdl(mdl_data: bytes) -> list:
    """
    Load skeleton data directly from MDL file using obtain_skeleton_data.
    
    Args:
        mdl_data: Decrypted MDL file bytes
        
    Returns:
        List with skeleton data or None if not found
    """
    try:
        from kuro_mdl_export_meshes import obtain_skeleton_data
        
        print(f"[+] Loading skeleton from MDL...")
        skeleton_data = obtain_skeleton_data(mdl_data)
        
        if skeleton_data == False or skeleton_data is None:
            print(f"    [!] No skeleton data in MDL")
            return None
        
        # Convert tuples to lists for JSON serialization
        skeleton_list = []
        for bone in skeleton_data:
            bone_dict = dict(bone)
            # Convert tuple values to lists
            if 'pos_xyz' in bone_dict and isinstance(bone_dict['pos_xyz'], tuple):
                bone_dict['pos_xyz'] = list(bone_dict['pos_xyz'])
            if 'rotation_euler_rpy' in bone_dict and isinstance(bone_dict['rotation_euler_rpy'], tuple):
                bone_dict['rotation_euler_rpy'] = list(bone_dict['rotation_euler_rpy'])
            if 'scale' in bone_dict and isinstance(bone_dict['scale'], tuple):
                bone_dict['scale'] = list(bone_dict['scale'])
            if 'unknown_quat' in bone_dict and isinstance(bone_dict['unknown_quat'], tuple):
                bone_dict['unknown_quat'] = list(bone_dict['unknown_quat'])
            if 'unknown' in bone_dict and isinstance(bone_dict['unknown'], tuple):
                bone_dict['unknown'] = list(bone_dict['unknown'])
            # Pre-compute quaternion from Euler RPY using eArmada8's exact formula
            # Store as xyzw (THREE.js convention)
            if 'rotation_euler_rpy' in bone_dict:
                q_wxyz = rpy2quat(bone_dict['rotation_euler_rpy'])
                bone_dict['quat_xyzw'] = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
            skeleton_list.append(bone_dict)
        
        print(f"    Found {len(skeleton_list)} bones")
        
        # Count bone types
        type_counts = {}
        for bone in skeleton_list:
            bone_type = bone.get('type', -1)
            type_counts[bone_type] = type_counts.get(bone_type, 0) + 1
        
        print(f"    Bone types: {dict(type_counts)}")
        return skeleton_list
        
    except Exception as e:
        print(f"    [!] Failed to load skeleton: {e}")
        return None


# -----------------------------
# Load Model Info (MI/JSON)
# -----------------------------
def decode_binary_mi(data: bytes) -> dict:
    """
    Decode FDK binary JSON format (.mi files from PS4/NX).
    Based on kuro_decode_bin_json.py by eArmada8.
    """
    import struct, io as _io
    f = _io.BytesIO(data)
    
    def read_null_terminated_string(fh):
        result = b''
        while True:
            b = fh.read(1)
            if b == b'\x00' or not b:
                break
            result += b
        return result.decode('utf-8')
    
    def read_string_from_dict(fh):
        addr, = struct.unpack("<I", fh.read(4))
        return_address = fh.tell()
        fh.seek(addr)
        fh.seek(4, 1)  # skip CRC32
        s = read_null_terminated_string(fh)
        fh.seek(return_address)
        return s
    
    def read_value(fh):
        dat_type, = struct.unpack("<B", fh.read(1))
        name = read_string_from_dict(fh) if dat_type < 0x10 else ''
        if dat_type in (0x02, 0x12):
            return name, read_null_terminated_string(fh)
        elif dat_type in (0x03, 0x13):
            val, = struct.unpack("<d", fh.read(8))
            return name, int(val) if round(val) == val else val
        elif dat_type in (0x04, 0x14):
            num, = struct.unpack("<I", fh.read(4))
            addrs = struct.unpack(f"<{num}I", fh.read(4 * num))
            end_off = fh.tell()
            result = {}
            for a in addrs:
                fh.seek(a)
                k, v = read_value(fh)
                result[k] = v
                end_off = max(fh.tell(), end_off)
            fh.seek(end_off)
            return name, result
        elif dat_type in (0x05, 0x15):
            num, = struct.unpack("<I", fh.read(4))
            addrs = struct.unpack(f"<{num}I", fh.read(4 * num))
            end_off = fh.tell()
            result = []
            for a in addrs:
                fh.seek(a)
                _, v = read_value(fh)
                result.append(v)
                end_off = max(fh.tell(), end_off)
            fh.seek(end_off)
            return name, result
        elif dat_type in (0x06, 0x16):
            return name, bool(struct.unpack("<B", fh.read(1))[0])
        else:
            raise ValueError(f"Unknown binary MI op code: {hex(dat_type)} at {hex(fh.tell()-1)}")
    
    # Parse header
    magic = f.read(4)
    if magic != b'JSON':
        raise ValueError("Not a binary MI file")
    f.read(4)  # unknown
    dat_start, = struct.unpack("<I", f.read(4))
    f.seek(dat_start)
    _, result = read_value(f)
    return result


def parse_fxo_capabilities(fxo_path: Path) -> dict:
    """
    Parse FXO (compiled DXBC shader) and extract cb_local uniform names.
    
    Returns dict with:
      - 'uniforms': set of uniform names from cb_local
      - 'textures': set of bound texture slot names (Tex0, Tex1, etc.)
    """
    try:
        data = fxo_path.read_bytes()
        if len(data) < 32 or data[:4] != b'DXBC':
            return None
        
        chunk_count = struct.unpack_from('<I', data, 28)[0]
        
        # Find RDEF chunk
        rdef_data = None
        for i in range(chunk_count):
            off = struct.unpack_from('<I', data, 32 + i * 4)[0]
            tag = data[off:off+4]
            size = struct.unpack_from('<I', data, off + 4)[0]
            if tag == b'RDEF':
                rdef_data = data[off + 8 : off + 8 + size]
                break
        
        if rdef_data is None:
            return None
        
        def read_string(buf, offset):
            s = ''
            while offset < len(buf) and buf[offset] != 0:
                s += chr(buf[offset]) if 32 <= buf[offset] < 127 else ''
                offset += 1
            return s
        
        # Parse resource bindings for texture slots
        cb_count = struct.unpack_from('<I', rdef_data, 0)[0]
        cb_offset = struct.unpack_from('<I', rdef_data, 4)[0]
        bind_count = struct.unpack_from('<I', rdef_data, 8)[0]
        bind_offset = struct.unpack_from('<I', rdef_data, 12)[0]
        
        textures = set()
        for i in range(bind_count):
            pos = bind_offset + i * 32
            name_off = struct.unpack_from('<I', rdef_data, pos)[0]
            rtype = struct.unpack_from('<I', rdef_data, pos + 4)[0]
            name = read_string(rdef_data, name_off)
            if rtype == 2 and name.startswith('Tex'):  # Texture type
                textures.add(name)
        
        # Parse cb_local uniforms (stride=40 per variable descriptor)
        uniforms = set()
        for i in range(cb_count):
            pos = cb_offset + i * 24
            name_off = struct.unpack_from('<I', rdef_data, pos)[0]
            var_count = struct.unpack_from('<I', rdef_data, pos + 4)[0]
            var_offset = struct.unpack_from('<I', rdef_data, pos + 8)[0]
            name = read_string(rdef_data, name_off)
            
            if name == 'cb_local':
                for v in range(var_count):
                    vpos = var_offset + v * 40
                    if vpos + 12 > len(rdef_data):
                        break
                    vname_off = struct.unpack_from('<I', rdef_data, vpos)[0]
                    vname = read_string(rdef_data, vname_off)
                    if vname and vname.endswith('_g'):
                        uniforms.add(vname)
                break
        
        return {'uniforms': uniforms, 'textures': textures}
    
    except Exception as e:
        print(f"    [!] Failed to parse FXO: {e}")
        return None


def load_model_info(mdl_path: Path) -> dict:
    """
    Load external MI or JSON file containing IK, physics, colliders, dynamic bones.
    
    Automatically detects path:
    model/chr5001_c11.mdl -> model_info/chr5001_c11.mi or chr5001_c11.json
    
    Args:
        mdl_path: Path to original .mdl file
        
    Returns:
        Dictionary with model info or None if not found
    """
    mdl_dir = mdl_path.parent
    mdl_stem = mdl_path.stem
    
    # Check if mdl_path contains 'model' directory
    # Convert path structure: .../model/chr5001_c11.mdl -> .../model_info/chr5001_c11.mi
    possible_paths = []
    
    # Try model_info sibling directory
    if mdl_dir.name == 'model' or 'model' in mdl_dir.parts:
        # Replace 'model' with 'model_info'
        parts = list(mdl_dir.parts)
        for i, part in enumerate(parts):
            if part == 'model':
                model_info_parts = parts[:i] + ['model_info'] + parts[i+1:]
                model_info_dir = Path(*model_info_parts)
                possible_paths.append(model_info_dir / f'{mdl_stem}.mi')
                possible_paths.append(model_info_dir / f'{mdl_stem}.json')
    
    # Also try same directory
    possible_paths.append(mdl_dir / f'{mdl_stem}.mi')
    possible_paths.append(mdl_dir / f'{mdl_stem}.json')
    
    for mi_path in possible_paths:
        if mi_path.exists():
            print(f"[+] Loading model info: {mi_path.name}")
            try:
                with open(mi_path, 'rb') as f:
                    raw = f.read()
                
                # Detect binary MI (FDK binary JSON: starts with b'JSON')
                if raw[:4] == b'JSON':
                    print(f"    (binary MI format detected, decoding...)")
                    mi_data = decode_binary_mi(raw)
                else:
                    mi_data = json.loads(raw.decode('utf-8'))
                
                # Print summary
                print(f"    Model info sections:")
                for key, value in mi_data.items():
                    if isinstance(value, list):
                        count = len(value)
                        if count > 0:
                            print(f"      {key}: {count} items")
                    elif isinstance(value, dict):
                        count = len(value)
                        if count > 0:
                            print(f"      {key}: {count} keys")
                
                return mi_data
                
            except Exception as e:
                print(f"    [!] Failed to load model info: {e}")
                return None
    
    print(f"[!] Model info not found in:")
    for p in possible_paths:
        print(f"    {p}")
    return None


# -----------------------------
# Load animations from _m_*.mdl files
# -----------------------------
def load_animations_from_directory(mdl_path: Path, skeleton_data: list) -> list:
    """
    Scan directory for animation MDL files (e.g. chr5001_m_wait.mdl) and extract keyframe data.
    Converts differential rotations to absolute using bind pose quaternions.
    
    Args:
        mdl_path: Path to the main model .mdl file
        skeleton_data: Skeleton bone list with rotation_euler_rpy
        
    Returns:
        List of animation dicts with channels containing absolute keyframe data
    """
    import struct, io, glob as _glob
    
    mdl_dir = mdl_path.parent
    model_stem = mdl_path.stem  # e.g. "chr5001_c11"
    # Extract base character name (chrXXXX) from any position in filename
    # e.g. "q_chr5121" -> "chr5121", "chr5001_c11" -> "chr5001"
    import re as _re
    chr_match = _re.search(r'(chr\d+)', model_stem)
    base_name = chr_match.group(1) if chr_match else model_stem.split('_')[0]
    
    # Find animation MDL files: *chr5001*_m_*.mdl (covers base, q_ prefix, etc.)
    pattern = str(mdl_dir / f"*{base_name}*_m_*.mdl")
    anim_files = sorted(_glob.glob(pattern))
    
    if not anim_files:
        print(f"[!] No animation files found matching: {pattern}")
        return []
    
    print(f"\n[+] Found {len(anim_files)} animation files")
    
    # Build skeleton bone map for bind pose quaternions
    skel_map = {}
    if skeleton_data:
        for bone in skeleton_data:
            if bone.get('synthetic') and 'quat_xyzw' in bone:
                # Synthetic bones store authoritative quaternion directly (xyzw → wxyz)
                q = bone['quat_xyzw']
                skel_map[bone['name']] = [q[3], q[0], q[1], q[2]]
            elif 'rotation_euler_rpy' in bone:
                skel_map[bone['name']] = rpy2quat(bone['rotation_euler_rpy'])
            else:
                skel_map[bone['name']] = [1, 0, 0, 0]
    
    def qmul(a, b):
        """Multiply two quaternions in wxyz format."""
        w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
        return [w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2]
    
    key_stride = {9: 12, 10: 16, 11: 12, 12: 4, 13: 8}
    animations = []
    
    for anim_path in anim_files:
        anim_file = Path(anim_path)
        # Extract animation name from any prefix pattern
        # e.g. chr5001_m_wait.mdl -> wait, q_chr5001_m_wait.mdl -> q_wait
        anim_stem = anim_file.stem
        # Find _m_ marker and take everything after it
        m_idx = anim_stem.find(f"{base_name}_m_")
        if m_idx >= 0:
            prefix = anim_stem[:m_idx] if m_idx > 0 else ""
            suffix = anim_stem[m_idx + len(base_name) + 3:]  # skip "base_m_"
            anim_name = (prefix + suffix) if prefix else suffix
        else:
            anim_name = anim_stem
        
        try:
            with open(anim_path, 'rb') as f:
                data = f.read()
            
            # Check for CLE encryption
            if data[0:4] in [b"F9BA", b"C9BA", b"D9BA"]:
                try:
                    data = decryptCLE(data)
                except:
                    print(f"    [!] Failed to decrypt {anim_file.name}, skipping")
                    continue
            
            # Find animation section (type 3)
            with io.BytesIO(data) as f:
                magic = struct.unpack("<I", f.read(4))[0]
                if magic != 0x204c444d:
                    print(f"    [!] {anim_file.name}: Invalid MDL magic, skipping")
                    continue
                mdl_ver = struct.unpack("<I", f.read(4))[0]
                f.read(4)  # unknown
                
                # Version >= 1 uses 1-byte string prefix, version 0 uses 4-byte
                def read_string(fh):
                    if mdl_ver >= 1:
                        length, = struct.unpack("<B", fh.read(1))
                    else:
                        length, = struct.unpack("<I", fh.read(4))
                    raw = fh.read(length)
                    try:
                        return raw.decode("utf-8")
                    except UnicodeDecodeError:
                        return raw.decode("latin-1")
                
                ani_offset = None
                ani_size = None
                while True:
                    hdr = f.read(8)
                    if len(hdr) < 8: break
                    stype, ssize = struct.unpack("<II", hdr)
                    if ssize == 0 and stype == 0: break
                    if stype == 3:
                        ani_offset = f.tell()
                        ani_size = ssize
                    f.seek(ssize, 1)
                
                if ani_offset is None:
                    continue
                
                # Parse animation blocks
                f.seek(ani_offset)
                blocks, = struct.unpack("<I", f.read(4))
                channels = []
                
                for _ in range(blocks):
                    name = read_string(f)
                    bone = read_string(f)
                    atype, unk0, unk1, nkf = struct.unpack("<4I", f.read(16))
                    if atype not in key_stride:
                        # Skip unknown types (12=shader, 13=uv scroll)
                        stride = key_stride.get(atype, 0) + 24
                        if stride > 24:
                            f.seek(nkf * stride, 1)
                        continue
                    
                    stride = key_stride[atype] + 24
                    buf = f.read(nkf * stride)
                    
                    times = []
                    values = []
                    for j in range(nkf):
                        t = struct.unpack_from("<f", buf, j * stride)[0]
                        times.append(round(t, 6))
                        
                        val_offset = j * stride + 4
                        val_size = key_stride[atype]
                        raw = list(struct.unpack_from(f"<{val_size//4}f", buf, val_offset))
                        
                        if atype == 10:  # Rotation: differential xyzw -> absolute xyzw
                            bind_q = skel_map.get(bone, [1,0,0,0])
                            diff_wxyz = [raw[3], raw[0], raw[1], raw[2]]
                            abs_wxyz = qmul(bind_q, diff_wxyz)
                            raw = [abs_wxyz[1], abs_wxyz[2], abs_wxyz[3], abs_wxyz[0]]  # xyzw
                        
                        values.extend(raw)
                    
                    channels.append({
                        'bone': bone,
                        'type': atype,  # 9=trans, 10=rot, 11=scale
                        'times': times,
                        'values': values  # flat array
                    })
                
                # Read time range footer
                try:
                    tmin, tmax = struct.unpack("<2f", f.read(8))
                    duration = tmax - tmin
                except:
                    tmin = min((min(c['times']) for c in channels if c['times']), default=0.0)
                    tmax = max((max(c['times']) for c in channels if c['times']), default=1.0)
                    duration = tmax - tmin
                
                # Normalize times to start at 0 (required for proper THREE.js looping)
                if tmin != 0:
                    for c in channels:
                        c['times'] = [round(t - tmin, 6) for t in c['times']]
                
                # Only include channels for types 9, 10, 11
                channels = [c for c in channels if c['type'] in (9, 10, 11)]
                
                animations.append({
                    'name': anim_name,
                    'duration': round(duration, 6),
                    'channels': channels
                })
                
                types = {}
                for c in channels:
                    types[c['type']] = types.get(c['type'], 0) + 1
                print(f"    {anim_name}: {len(channels)} ch, {duration:.1f}s ({types})")
                
        except Exception as e:
            print(f"    [!] Error loading {anim_file.name}: {e}")
            continue
    
    print(f"[+] Loaded {len(animations)} animations")
    return animations


# -----------------------------
# Load MDL with all data
# -----------------------------
def load_mdl_with_textures(mdl_path: Path, temp_dir: Path, recompute_normals: bool = False, no_shaders: bool = False):
    """
    Load MDL file and copy textures to temp directory.
    Also loads skeleton and model info if available.
    
    Returns:
        tuple: (meshes, material_texture_map, skeleton_data, model_info)
    """
    mdl_path = Path(mdl_path).absolute()
    with open(mdl_path, "rb") as f:
        mdl_data = f.read()

    print(f"\n{'='*60}")
    print(f"Loading MDL: {mdl_path.name}")
    print(f"{'='*60}")

    # Decrypt and parse MDL
    mdl_data = decryptCLE(mdl_data)
    material_struct = obtain_material_data(mdl_data)
    mesh_struct = obtain_mesh_data(mdl_data, material_struct=material_struct)

    # Load skeleton data directly from MDL
    skeleton_data = load_skeleton_from_mdl(mdl_data)
    
    # Build skeleton name->index map for BLENDINDICES remapping
    skeleton_name_to_idx = {}
    if skeleton_data:
        for bone in skeleton_data:
            skeleton_name_to_idx[bone['name']] = bone['id_referenceonly']
    
    # Load model info (MI/JSON)
    model_info = load_model_info(mdl_path)

    # Get image list from materials
    image_list = sorted(list(set([x['texture_image_name']+'.dds' for y in material_struct for x in y['textures']])))
    
    print(f"\n[+] Found {len(material_struct)} materials")
    print(f"[+] Found {len(image_list)} unique textures")

    # Search paths for textures
    search_paths = [
        mdl_path.parent,
        mdl_path.parent / 'image',
        mdl_path.parent / 'textures',
        mdl_path.parent.parent,
        mdl_path.parent.parent / 'image',
        mdl_path.parent.parent / 'dx11' / 'image',
        mdl_path.parent.parent / 'dxl1' / 'image',
        mdl_path.parent.parent.parent,
        mdl_path.parent.parent.parent / 'image',
        mdl_path.parent.parent.parent / 'dx11' / 'image',
        mdl_path.parent.parent.parent / 'dxl1' / 'image',
        mdl_path.parent.parent.parent.parent / 'dx11' / 'image',
        mdl_path.parent.parent.parent.parent / 'dxl1' / 'image',
        Path.cwd(),
        Path.cwd() / 'image',
        Path.cwd() / 'textures',
        Path.cwd() / 'dx11' / 'image',
        Path.cwd() / 'dxl1' / 'image',
    ]

    # Create textures subdirectory in temp
    temp_textures_dir = temp_dir / 'textures'
    temp_textures_dir.mkdir(exist_ok=True)

    # Copy and convert textures
    material_texture_map = {}
    texture_success = {}
    
    if TEXTURES_AVAILABLE and PIL_AVAILABLE and len(image_list) > 0:
        print(f"\n[+] Searching for textures in:")
        existing_count = 0
        for p in search_paths:
            if p.exists():
                print(f"  [OK] {p}")
                existing_count += 1
            else:
                print(f"  [ - ] {p} (not found)")
        
        if existing_count == 0:
            print(f"\n[!] None of the texture search paths exist!")
            print(f"  MDL location: {mdl_path.absolute()}")
            print(f"  MDL parent: {mdl_path.parent.absolute()}")
            print(f"  Current dir: {Path.cwd()}")
        
        print(f"\n[+] Converting textures to PNG...")
        for tex_name in image_list:
            dds_path = find_texture_file(tex_name, search_paths)
            
            if dds_path:
                png_name = tex_name.replace('.dds', '.png')
                png_path = temp_textures_dir / png_name
                
                if convert_dds_to_png(dds_path, png_path):
                    print(f"  [OK] {tex_name} -> {png_name}")
                    texture_success[tex_name] = png_name
                else:
                    print(f"  [FAIL] {tex_name} - conversion failed")
                    texture_success[tex_name] = None
            else:
                print(f"  [NOT FOUND] {tex_name}")
                texture_success[tex_name] = None
        
        # Build material texture map
        for material in material_struct:
            mat_name = material['material_name']
            mat_textures = {}
            
            for tex in material.get('textures', []):
                tex_name = tex['texture_image_name']
                if not tex_name.endswith('.dds'):
                    tex_name = tex_name + '.dds'
                
                slot = tex['texture_slot']
                wrapS = tex.get('wrapS', 0)
                wrapT = tex.get('wrapT', 0)
                
                if tex_name in texture_success and texture_success[tex_name]:
                    rel_path = f"textures/{texture_success[tex_name]}"
                    
                    tex_info = {
                        'path': rel_path,
                        'wrapS': wrapS,
                        'wrapT': wrapT
                    }
                    
                    if slot == 0:
                        mat_textures['diffuse'] = tex_info
                    elif slot == 1:
                        mat_textures['detail'] = tex_info
                    elif slot == 3:
                        mat_textures['normal'] = tex_info
                    elif slot == 7:
                        mat_textures['specular'] = tex_info
                    elif slot == 9:
                        mat_textures['toon'] = tex_info
                    else:
                        mat_textures[f'slot_{slot}'] = tex_info
            
            # Extract shader parameters
            shader_type = material.get('shader_name', '')
            mat_textures['_shaderType'] = shader_type
            mat_textures['_shaderHash'] = material.get('shader_switches_hash_referenceonly', '')
            
            shader_params = {}
            for sp in material.get('shaders', []):
                name = sp.get('shader_name', '')
                data = sp.get('data')
                type_int = sp.get('type_int', 0)
                # Extract key rendering params
                if name in (
                    'rimIntensity_g', 'rimLightPower_g', 'rimLightColor_g',
                    'toonEdgeStrength_g', 'toonEdgeColor_g',
                    'shadowColor1_g', 'shadowColor2_g', 'shadowGradSharpness_g',
                    'specularColor_g', 'specularGlossiness_g', 'specularShadowFadeRatio_g',
                    'specularGlossiness0_g', 'specularGlossiness1_g',
                    'emissive_g', 'alphaTestThreshold_g',
                    'shadowBias_g', 'ssaoIntensity_g', 'shadowColor_g',
                    'diffuseMapColor0_g', 'diffuseMapColor1_g',
                ) or name.startswith('Switch_'):
                    shader_params[name] = data
            
            mat_textures['_shaderParams'] = shader_params
            
            if mat_textures:
                material_texture_map[mat_name] = mat_textures
        
        loaded_count = sum(1 for v in texture_success.values() if v is not None)
        print(f"\n[OK] Loaded and converted {loaded_count}/{len(texture_success)} textures")
        
        # Print shader types found in materials
        shader_types = {}
        for material in material_struct:
            st = material.get('shader_name', 'unknown')
            shader_types[st] = shader_types.get(st, 0) + 1
        if shader_types:
            parts_str = ', '.join(f'{k}({v})' for k, v in sorted(shader_types.items()))
            print(f"[+] Material shader types: {parts_str}")
            chr_count = sum(v for k, v in shader_types.items() if k.startswith('chr_'))
            if chr_count:
                if no_shaders:
                    print(f"[!] {chr_count} materials could use toon rendering — disabled by --no-shaders")
                else:
                    print(f"[+] {chr_count} materials will use toon shader rendering")
        
        # Discover and parse FXO shader files
        fxo_dir = None
        fxo_loaded = 0
        
        if no_shaders:
            print(f"[NO-SHADERS] FXO shader loading skipped (--no-shaders)")
        else:
            parts = list(mdl_path.parent.parts)
            for i, part in enumerate(parts):
                if part == 'common':
                    # asset/common/model -> asset/dx11/shader
                    fxo_parts = parts[:i] + ['dx11', 'shader']
                    fxo_dir = Path(*fxo_parts)
                    break
            
            if fxo_dir and fxo_dir.exists():
                # Check if directory has FXO files
                has_fxo = any(fxo_dir.glob("*.fxo"))
                print(f"[+] FXO directory: {fxo_dir}" + (" (scanning...)" if has_fxo else " (empty)")) 
                
                fxo_cache = {}  # shader_name -> caps (parse once per shader type)
                for material in material_struct:
                    mat_name = material['material_name']
                    sname = material.get('shader_name', '')
                    if not sname:
                        continue
                    
                    # Skip if we already loaded an FXO for this shader type
                    # (all variants of same shader_name share the same cb_local uniforms)
                    if sname in fxo_cache:
                        caps = fxo_cache[sname]
                        if caps and mat_name in material_texture_map:
                            material_texture_map[mat_name]['_fxoCaps'] = {
                                'uniforms': sorted(caps['uniforms']),
                                'textures': sorted(caps['textures']),
                            }
                            fxo_loaded += 1
                        continue
                    
                    # FXO naming convention: {shader_name}#{8-char-hash}.fxo
                    # The hash in MDL (16-char) doesn't match FXO filename hash (8-char)
                    # So we match by shader_name prefix and pick first available variant
                    fxo_glob = sorted(fxo_dir.glob(f"{sname}#*.fxo"))
                    if not fxo_glob:
                        # Fallback: try underscore separator (older format)
                        fxo_glob = sorted(fxo_dir.glob(f"{sname}_*.fxo"))
                    
                    if fxo_glob:
                        caps = parse_fxo_capabilities(fxo_glob[0])
                        fxo_cache[sname] = caps
                        if caps:
                            fxo_loaded += 1
                            if mat_name in material_texture_map:
                                material_texture_map[mat_name]['_fxoCaps'] = {
                                    'uniforms': sorted(caps['uniforms']),
                                    'textures': sorted(caps['textures']),
                                }
                            print(f"  [FXO] {sname}: {fxo_glob[0].name} ({len(caps['uniforms'])} uniforms, {len(caps['textures'])} tex, {len(fxo_glob)} variants)")
                        else:
                            print(f"  [FXO] {sname}: {fxo_glob[0].name} — parse failed")
                    else:
                        fxo_cache[sname] = None
                        print(f"  [FXO] {sname}: no matching files found")
                if fxo_loaded > 0:
                    print(f"[+] Loaded {fxo_loaded} FXO shader definitions")
                else:
                    print(f"[!] No FXO matches found — check filenames above vs expected patterns")
            elif fxo_dir:
                print(f"[!] FXO shader directory not found: {fxo_dir}")
            else:
                print(f"[!] Could not determine FXO shader path from MDL location")
    else:
        print("\n[!] Texture loading disabled or dependencies missing")

    # Extract mesh data
    meshes = []
    mesh_blocks = mesh_struct.get("mesh_blocks", [])
    all_buffers = mesh_struct.get("mesh_buffers", [])
    
    # Collect bind-pose matrices from mesh block nodes (for proper skinning)
    # These are 4x4 row-major matrices: bone-local to world-space at bind pose
    global_bind_matrices = {}  # bone_name -> 4x4 matrix (row-major)
    for block in mesh_blocks:
        if 'nodes' in block:
            for node in block['nodes']:
                if node['name'] not in global_bind_matrices:
                    global_bind_matrices[node['name']] = node['matrix']

    print(f"\n[+] Processing {len(all_buffers)} mesh groups...")
    print(f"[+] Collected {len(global_bind_matrices)} unique bind-pose matrices from mesh nodes")

    # Create synthetic skeleton entries for bones in mesh vgmaps but not in skeleton
    # (e.g., costume cloth chains, endpoint bones)
    if skeleton_data and skeleton_name_to_idx:
        synth_count = create_synthetic_bones(skeleton_data, skeleton_name_to_idx, global_bind_matrices)
        if synth_count > 0:
            print(f"[+] Created {synth_count} synthetic bones for mesh references not in MDL skeleton")

    for i, submesh_list in enumerate(all_buffers):
        base_name = mesh_blocks[i].get("name", f"mesh_{i}") if i < len(mesh_blocks) else f"mesh_{i}"
        primitives = mesh_blocks[i].get("primitives", []) if i < len(mesh_blocks) else []
        
        # Get per-mesh-block node list for BLENDINDICES remapping
        mesh_block_nodes = mesh_blocks[i].get("nodes", []) if i < len(mesh_blocks) else []
        
        # Build local-index -> global-skeleton-index remap table
        local_to_global_remap = {}
        if mesh_block_nodes and skeleton_name_to_idx:
            for local_idx, node in enumerate(mesh_block_nodes):
                bone_name = node['name']
                if bone_name in skeleton_name_to_idx:
                    local_to_global_remap[local_idx] = skeleton_name_to_idx[bone_name]
                else:
                    print(f"    [!] WARNING: mesh node '{bone_name}' (local idx {local_idx}) not found in skeleton!")
                    local_to_global_remap[local_idx] = 0  # fallback to root
            
            if local_to_global_remap:
                print(f"    [{i}] {base_name}: {len(local_to_global_remap)} nodes remapped (local -> global skeleton)")

        for j, submesh in enumerate(submesh_list):
            vb = submesh.get("vb", [])
            ib = submesh.get("ib", {}).get("Buffer", [])

            pos_buffer = None
            normal_buffer = None
            uv_buffer = None
            blend_weights_buffer = None
            blend_indices_buffer = None
            tangent_buffer = None

            for element in vb:
                sem = element.get("SemanticName")
                buf = element.get("Buffer")
                if sem == "POSITION":
                    pos_buffer = buf
                elif sem == "NORMAL":
                    normal_buffer = buf
                elif sem == "TEXCOORD" and uv_buffer is None:
                    uv_buffer = buf
                elif sem == "BLENDWEIGHT":
                    blend_weights_buffer = buf
                elif sem == "BLENDINDICES":
                    blend_indices_buffer = buf
                elif sem == "TANGENT":
                    tangent_buffer = buf

            if not pos_buffer:
                continue

            vertices = np.array([p[:3] for p in pos_buffer], dtype=np.float32)

            flat_indices = []
            for tri in ib:
                if len(tri) == 3:
                    flat_indices.extend(tri)
            indices = np.array(flat_indices, dtype=np.uint32)

            uvs = None
            if uv_buffer:
                uvs = np.array([uv[:2] for uv in uv_buffer], dtype=np.float32)

            # Extract skinning data WITH REMAPPING
            skin_weights = None
            skin_indices = None
            if blend_weights_buffer and blend_indices_buffer:
                skin_weights = np.array(blend_weights_buffer, dtype=np.float32)
                raw_indices = np.array(blend_indices_buffer, dtype=np.uint32)
                
                # CRITICAL FIX: Remap BLENDINDICES from mesh-local to global skeleton
                if local_to_global_remap:
                    remapped_indices = np.zeros_like(raw_indices)
                    for vi in range(len(raw_indices)):
                        for bi in range(len(raw_indices[vi])):
                            local_idx = int(raw_indices[vi][bi])
                            remapped_indices[vi][bi] = local_to_global_remap.get(local_idx, 0)
                    skin_indices = remapped_indices
                    
                    if j == 0:  # Log first submesh only
                        sample_raw = raw_indices[0] if len(raw_indices) > 0 else []
                        sample_remap = skin_indices[0] if len(skin_indices) > 0 else []
                        print(f"    Mesh {i}_{j}: Remapped BLENDINDICES (sample: {list(sample_raw)} -> {list(sample_remap)})")
                else:
                    skin_indices = raw_indices
                    print(f"    Mesh {i}_{j}: WARNING - No node remap table, using raw BLENDINDICES")
                
                print(f"    Mesh {i}_{j}: Skinning data - weights:{skin_weights.shape} indices:{skin_indices.shape}")
            else:
                if not blend_weights_buffer:
                    print(f"    Mesh {i}_{j}: No BLENDWEIGHT buffer")
                if not blend_indices_buffer:
                    print(f"    Mesh {i}_{j}: No BLENDINDICES buffer")

            if not recompute_normals and normal_buffer:
                normals = np.array([n[:3] for n in normal_buffer], dtype=np.float32)
                lens = np.linalg.norm(normals, axis=1)
                nonzero = lens > 1e-8
                normals[nonzero] = normals[nonzero] / lens[nonzero][:, None]
            else:
                normals = compute_smooth_normals_with_sharing(vertices, indices) if len(indices) >= 3 else None

            # Extract tangents (vec4: xyz direction + w handedness)
            tangents = None
            if tangent_buffer:
                tangents = np.array([t[:4] if len(t) >= 4 else list(t[:3]) + [1.0] for t in tangent_buffer], dtype=np.float32)
                # Normalize xyz part
                xyz = tangents[:, :3]
                lens_t = np.linalg.norm(xyz, axis=1)
                nonzero_t = lens_t > 1e-8
                tangents[nonzero_t, :3] = xyz[nonzero_t] / lens_t[nonzero_t][:, None]

            material_name = None
            if j < len(primitives):
                material_name = primitives[j].get("material")

            # Shadow mesh detection: match mesh group name from MDL structure
            shadow_keywords = ['shadow', 'kage', 'box']
            gname = base_name.lower()
            is_shadow_mesh = any(kw in gname for kw in shadow_keywords)

            mesh_data = {
                "name": f"{i}_{base_name}_{j:02d}",
                "mesh_group": base_name,
                "is_shadow": is_shadow_mesh,
                "vertices": vertices,
                "normals": normals,
                "uvs": uvs,
                "indices": indices,
                "material": material_name,
                "skin_weights": skin_weights,
                "skin_indices": skin_indices,
                "tangents": tangents
            }

            meshes.append(mesh_data)
            
            # Per-mesh detail log
            mesh_name = mesh_data["name"]
            vert_count = len(vertices)
            tri_count = len(indices) // 3
            has_n = "MDL" if (not recompute_normals and normal_buffer) else "recomputed"
            has_uv = "yes" if uvs is not None else "no"
            has_t = f"vec{len(tangent_buffer[0]) if tangent_buffer else 0}" if tangent_buffer else "no"
            has_skin = "yes" if (skin_weights is not None) else "no"
            shadow_tag = " [SHADOW]" if is_shadow_mesh else ""
            print(f"    {mesh_name}: {vert_count} verts, {tri_count} tris | normals: {has_n} | UV: {has_uv} | tangents: {has_t} | skinning: {has_skin}{shadow_tag}")

    tangent_count = sum(1 for m in meshes if m.get("tangents") is not None)
    uv_no_tangent = sum(1 for m in meshes if m.get("tangents") is None and m.get("uvs") is not None)
    summary = f"[OK] Loaded {len(meshes)} submeshes | tangents: {tangent_count} MDL"
    if uv_no_tangent > 0:
        summary += f", {uv_no_tangent} will be computed in JS"
    print(summary)
    
    # Mesh group summary
    group_counts = {}
    shadow_count = sum(1 for m in meshes if m.get("is_shadow", False))
    for m in meshes:
        g = m.get("mesh_group", "unknown")
        group_counts[g] = group_counts.get(g, 0) + 1
    group_parts = [f"{g}({c})" for g, c in sorted(group_counts.items())]
    print(f"[+] Mesh groups: {', '.join(group_parts)}")
    if shadow_count > 0:
        print(f"[+] Shadow meshes detected: {shadow_count} (mesh group name matches: shadow/kage/box)")
    print(f"{'='*60}\n")

    return meshes, material_texture_map, skeleton_data, model_info, global_bind_matrices


# -----------------------------
# Generate HTML with skeleton support
# -----------------------------
def generate_html_with_skeleton(mdl_path: Path, meshes: list, material_texture_map: dict, 
                                skeleton_data: dict, model_info: dict, debug_mode: bool = False,
                                bind_matrices: dict = None, animations_data: list = None,
                                skip_popup: bool = False, no_shaders: bool = False,
                                recompute_normals: bool = False) -> str:
    """Generate HTML content with texture and skeleton support."""
    
    meshes_data = []
    for m in meshes:
        if m["vertices"] is None or m["indices"] is None:
            continue
        verts = m["vertices"]
        norms = m["normals"]
        uvs = m["uvs"]
        idxs = m["indices"]
        
        if norms is None:
            norms = compute_smooth_normals_with_sharing(verts, idxs)
        
        mesh_info = {
            "name": m["name"],
            "mesh_group": m.get("mesh_group", ""),
            "is_shadow": m.get("is_shadow", False),
            "vertices": verts.astype(np.float32).flatten().tolist(),
            "normals": norms.astype(np.float32).flatten().tolist(),
            "indices": idxs.astype(np.uint32).tolist(),
            "material": m.get("material")
        }
        
        if uvs is not None:
            mesh_info["uvs"] = uvs.astype(np.float32).flatten().tolist()
        
        # Add skinning data if available
        if m.get("skin_weights") is not None and m.get("skin_indices") is not None:
            mesh_info["skinWeights"] = m["skin_weights"].astype(np.float32).flatten().tolist()
            mesh_info["skinIndices"] = m["skin_indices"].astype(np.uint32).flatten().tolist()
        
        # Add tangent data if available
        if m.get("tangents") is not None:
            mesh_info["tangents"] = m["tangents"].astype(np.float32).flatten().tolist()
        
        meshes_data.append(mesh_info)

    materials_json = {}
    for mat_name, textures in material_texture_map.items():
        materials_json[mat_name] = textures

    # Prepare skeleton data for JavaScript
    skeleton_json = json.dumps(skeleton_data) if skeleton_data else "null"
    model_info_json = json.dumps(model_info) if model_info else "null"
    
    # Prepare bind matrices: bone_name -> 4x4 row-major matrix
    # These are the actual bind-pose matrices from the MDL mesh nodes
    bind_matrices_json = json.dumps(bind_matrices) if bind_matrices else "null"
    animations_json = json.dumps(animations_data) if animations_data else "null"


    # SVG path data from reference controller icons (viewBox 0 0 64 64)
    xbox_body_p1 = 'M63.782 42.228c-.325-2.912-1-5.25-1-5.25s-4.664-16.855-7.876-21.475a5.868 5.868 0 0 0-.476-.595s-2.341-2.229-6.018-3.086c-2.829-.66-6.287-.81-7.736-.846-2.064-.018-6.836-.058-8.713-.057-1.86 0-6.584.04-8.637.057-1.449.035-4.908.187-7.738.846-3.677.857-6.018 3.086-6.018 3.086a5.88 5.88 0 0 0-.476.595c-3.212 4.62-7.877 21.474-7.877 21.474s-.674 2.34-.999 5.25c-.823 5.8.911 8.436 1.461 9.118s2.883 3.3 4.885 2.839c2.002-.463 5.452-5.086 5.452-5.086 1-1.238 2.027-2.552 2.975-3.76 1.531-1.95 3.042-2.901 4.205-3.362.311-.124.637-.208.968-.259a10.38 10.38 0 0 1 1.472-.119h20.728c.505.005.995.046 1.472.12.331.05.657.134.968.258 1.163.46 2.674 1.412 4.205 3.362.948 1.208 1.976 2.522 2.975 3.76 0 0 3.45 4.623 5.452 5.086 2.002.462 4.335-2.157 4.885-2.839s2.284-3.317 1.461-9.117zM48.388 15.483a2.23 2.23 0 1 1 0 4.46 2.23 2.23 0 0 1 0-4.46zM15.605 25.505a3.601 3.601 0 1 1 0-7.203 3.601 3.601 0 0 1 0 7.203zM27.85 33.01a4.025 4.025 0 0 1-.218.8.221.221 0 0 1-.208.14h-1.938a.348.348 0 0 0-.348.349v1.938a.221.221 0 0 1-.141.208 3.967 3.967 0 0 1-.8.217 4.418 4.418 0 0 1-2.187-.213.223.223 0 0 1-.145-.21v-1.94a.348.348 0 0 0-.348-.348H19.58a.223.223 0 0 1-.21-.145 4.418 4.418 0 0 1-.214-2.187c.042-.277.12-.542.218-.8a.221.221 0 0 1 .207-.141h1.938a.348.348 0 0 0 .348-.348v-1.938c0-.092.056-.175.142-.208.257-.097.522-.175.799-.217a4.418 4.418 0 0 1 2.187.213.224.224 0 0 1 .145.21v1.94c0 .192.156.348.348.348h1.94c.093 0 .178.057.21.145.243.67.332 1.411.214 2.187zm-.466-9.81a1.379 1.379 0 1 1 0-2.758 1.379 1.379 0 0 1 0 2.758zm5.309 3.385h-1.388a1.085 1.085 0 1 1 0-2.17h1.388a1.085 1.085 0 0 1 0 2.17zM32 17.875a2.435 2.435 0 1 1 0-4.87 2.435 2.435 0 0 1 0 4.87zm3.325 3.946a1.379 1.379 0 1 1 2.758 0 1.379 1.379 0 0 1-2.758 0zm5.054 13.255a3.601 3.601 0 1 1 0-7.203 3.601 3.601 0 0 1 0 7.203zm3.697-10.865a2.23 2.23 0 1 1 0-4.46 2.23 2.23 0 0 1 0 4.46zm4.312 4.269a2.23 2.23 0 1 1 0-4.46 2.23 2.23 0 0 1 0 4.46zm4.283-4.269a2.23 2.23 0 1 1 0-4.46 2.23 2.23 0 0 1 0 4.46z'
    xbox_body_p2 = 'M48.388 15.483a2.23 2.23 0 1 1 0 4.46 2.23 2.23 0 0 1 0-4.46zm-29.182 6.42a3.601 3.601 0 1 0-7.203 0 3.601 3.601 0 0 0 7.203 0zm9.558-.082a1.379 1.379 0 1 0-2.758 0 1.379 1.379 0 0 0 2.758 0zm5.015 3.679c0-.6-.486-1.086-1.085-1.086h-1.388a1.085 1.085 0 0 0 0 2.171h1.388c.6 0 1.085-.486 1.085-1.085zm.656-10.06a2.435 2.435 0 1 0-4.87 0 2.435 2.435 0 0 0 4.87 0zm2.27 7.76a1.379 1.379 0 1 0 0-2.758 1.379 1.379 0 0 0 0 2.758zm7.276 8.275a3.601 3.601 0 1 0-7.203 0 3.601 3.601 0 0 0 7.203 0zm2.324-9.494a2.23 2.23 0 1 0-4.459 0 2.23 2.23 0 0 0 4.46 0zm4.313 4.269a2.23 2.23 0 1 0-4.46 0 2.23 2.23 0 0 0 4.46 0zm4.282-4.27a2.23 2.23 0 1 0-4.46 0 2.23 2.23 0 0 0 4.46 0zm-.47-7.073a4.239 4.239 0 0 0-.67-1.254c-.498-.63-1.216-1.763-3.417-2.728-.93-.407-2.495-.871-3.36-.997-.653-.096-2.632-.231-3.285-.132-.343.052-.66.425-.66.425-.235.161-.587.748-1.379.763l-.985-.009c1.45.035 4.908.187 7.737.846 3.677.857 6.018 3.086 6.018 3.086m-44.86 0s2.341-2.229 6.018-3.086c2.83-.66 6.29-.811 7.739-.846l-.987.009c-.792-.015-1.144-.602-1.379-.763 0 0-.317-.373-.66-.425-.653-.1-2.632.036-3.286.132-.864.126-2.43.59-3.359.997-2.201.965-2.919 2.098-3.418 2.728a4.239 4.239 0 0 0-.668 1.254M25.14 30.33v-1.94a.224.224 0 0 0-.146-.21 4.418 4.418 0 0 0-2.187-.213c-.277.042-.542.12-.8.217a.221.221 0 0 0-.14.208v1.938a.348.348 0 0 1-.349.348H19.58a.221.221 0 0 0-.207.142 3.97 3.97 0 0 0-.218.799 4.418 4.418 0 0 0 .213 2.187.224.224 0 0 0 .211.145h1.94c.192 0 .347.156.347.348v1.94c0 .092.058.178.145.21.67.242 1.412.332 2.188.213.276-.042.541-.12.799-.217a.221.221 0 0 0 .141-.208V34.3c0-.192.156-.348.348-.348h1.938a.221.221 0 0 0 .208-.142c.097-.257.175-.522.218-.799a4.418 4.418 0 0 0-.214-2.187.224.224 0 0 0-.21-.145h-1.94a.348.348 0 0 1-.348-.348z'
    ps_body_p1 = 'M16.383 12.233 7.758 13.76l.144-.996c.9-1.116 3.637-1.802 4.68-1.927a8.833 8.833 0 0 1 3.35.137c.32.061.32.245.32.245ZM4.615 52.569s1.42.541 2.582-3.813a71.235 71.235 0 0 1 7.589-17.293c2.788-4.183 4.665-5.52 5.29-7.323a2.29 2.29 0 0 0 .082-1.031 4.518 4.518 0 0 0 4.378 2.98l14.916.002h.04a4.49 4.49 0 0 0 4.351-3.003 2.296 2.296 0 0 0 .081 1.052c.625 1.803 2.502 3.14 5.29 7.323a71.235 71.235 0 0 1 7.59 17.293c1.16 4.354 2.581 3.813 2.581 3.813l-.034.027h-.002l-3.976.641c-.361.013-.514-.176-.762-.577a66.481 66.481 0 0 1-3.051-7.02c-1.491-3.969-1.927-6.171-4.978-7.226a7.482 7.482 0 0 0-1.446-.299c-.722.026-1.796.323-2.51.342-2.45.05-3.442-.094-7.664-.05v.007l-2.965-.005-2.965.005v-.006c-4.22-.045-5.216.098-7.664.049-.717-.019-1.785-.316-2.51-.342a7.482 7.482 0 0 0-1.446.299c-3.05 1.055-3.487 3.257-4.978 7.226a66.481 66.481 0 0 1-3.051 7.02c-.248.4-.4.59-.762.577Zm16.413-24.456a3.72 3.72 0 1 0 4.375 4.375 3.722 3.722 0 0 0-4.375-4.375Zm12.603 7.072a.382.382 0 0 0-.383-.382h-2.51a.382.382 0 0 0 0 .765h2.51a.382.382 0 0 0 .383-.383Zm7.883-7.072a3.72 3.72 0 1 0 4.375 4.375 3.722 3.722 0 0 0-4.375-4.375Zm6.12-15.88 8.624 1.528-.144-.996c-.9-1.116-3.636-1.802-4.68-1.927a8.833 8.833 0 0 0-3.35.137c-.32.061-.32.245-.32.245Z'
    ps_body_p2 = 'M19.475 15.22a3.63 3.63 0 0 1 .472-3.135 1.533 1.533 0 0 1 .525-.359c2.15-.587 7.074-.485 11.497-.485v.002c4.423 0 9.347-.102 11.497.485a1.533 1.533 0 0 1 .525.359 3.63 3.63 0 0 1 .472 3.134l-.693 3.301-.32 1.487c-.073.345-.14.68-.205.997-.137.679-.259 1.283-.395 1.77-.642 2.307-3.297 2.286-3.437 2.317H24.525c-.14-.033-2.795-.011-3.437-2.318-.136-.487-.258-1.092-.395-1.77-.064-.318-.132-.652-.205-.998l-.32-1.486Zm.594 8.923c-.625 1.803-2.501 3.14-5.29 7.322A71.235 71.235 0 0 0 7.19 48.76c-1.16 4.354-2.582 3.812-2.582 3.812l.035.029-.005-.001c-.536-.086-1.66-.494-2.924-2.3-1.766-2.523-1.72-9.268-1.72-9.268l.01-.002a50.708 50.708 0 0 1 .7-8.463 61.13 61.13 0 0 1 5.904-18.303l.065-.092c.247-.301.728-.308 1.068-.407l8.629-1.528a1.82 1.82 0 0 1 1.732 1.438l2.028 9.19a2.36 2.36 0 0 1-.06 1.279Zm-9.722-5.455s.032.826.088 1.388l.016.252a.932.932 0 0 0 .312.663c.29.339.99 1 .99 1a.646.646 0 0 0 .622 0s.701-.661.99-1a.932.932 0 0 0 .312-.663l.016-.252a27.88 27.88 0 0 0 .088-1.388.914.914 0 0 0-.784-.868 8.18 8.18 0 0 0-.933-.084 8.18 8.18 0 0 0-.933.084.914.914 0 0 0-.784.868Zm-.262 5.612c.339-.288 1-.99 1-.99a.646.646 0 0 0 0-.622s-.661-.7-1-.99a.932.932 0 0 0-.664-.311l-.25-.016a27.872 27.872 0 0 0-1.39-.088.914.914 0 0 0-.867.784A8.183 8.183 0 0 0 6.83 23a8.18 8.18 0 0 0 .084.933.914.914 0 0 0 .868.783s.826-.03 1.388-.087l.251-.016a.932.932 0 0 0 .664-.313Zm3.728 2.926s-.031-.826-.088-1.388l-.016-.252a.932.932 0 0 0-.312-.663c-.289-.339-.99-1-.99-1a.646.646 0 0 0-.622 0s-.7.661-.99 1a.932.932 0 0 0-.312.663l-.016.251a27.872 27.872 0 0 0-.088 1.39.914.914 0 0 0 .784.867 8.18 8.18 0 0 0 .933.084 8.18 8.18 0 0 0 .933-.084.914.914 0 0 0 .784-.868ZM17.298 23a8.18 8.18 0 0 0-.084-.933.914.914 0 0 0-.867-.784s-.826.031-1.389.088l-.251.015a.932.932 0 0 0-.664.313c-.339.289-1 .99-1 .99a.646.646 0 0 0 0 .622s.661.7 1 .99a.931.931 0 0 0 .664.312l.251.015a27.8 27.8 0 0 0 1.389.088.914.914 0 0 0 .867-.784 8.178 8.178 0 0 0 .084-.932Zm.551-6.503-.279-1.415a.73.73 0 1 0-1.432.283l.28 1.415a.73.73 0 1 0 1.431-.283Zm44.424 33.802c-1.264 1.806-2.388 2.214-2.924 2.3h-.005l.035-.028s-1.421.542-2.582-3.813a71.236 71.236 0 0 0-7.59-17.293c-2.788-4.182-4.664-5.52-5.29-7.322a2.36 2.36 0 0 1-.06-1.28l2.028-9.189a1.82 1.82 0 0 1 1.736-1.438l8.625 1.527v.001c.34.1.822.105 1.068.407l.065.092a61.13 61.13 0 0 1 5.903 18.302 50.708 50.708 0 0 1 .701 8.464l.01.002s.046 6.744-1.72 9.268ZM46.035 16.516a.73.73 0 1 0 1.432.283l.279-1.415a.73.73 0 1 0-1.432-.283Zm.701 4.44a2.075 2.075 0 1 0 2.526 2.526 2.06 2.06 0 0 0-2.526-2.526Zm4.688 4.707a2.075 2.075 0 1 0 2.526 2.526 2.06 2.06 0 0 0-2.526-2.526Zm0-9.413a2.075 2.075 0 1 0 2.526 2.525 2.06 2.06 0 0 0-2.526-2.525Zm4.687 4.706a2.075 2.075 0 1 0 2.526 2.526 2.06 2.06 0 0 0-2.526-2.526Z'
    sw_body_p1 = 'M60.152 21.98c-.66-2.872-1.353-5.28-1.945-6.047a4.964 4.964 0 0 0-1.108-.968h-.018a19.062 19.062 0 0 0-5.245-2.586c-3.428-1.161-10.467-1.272-10.467-1.272l.004-.002c-1.467-.023-7.266-.11-9.383-.11-2.125 0-7.959.088-9.401.11l.003.002s-7.038.11-10.467 1.272a19.064 19.064 0 0 0-5.245 2.586 4.964 4.964 0 0 0-1.108.968c-.727.942-1.608 4.369-2.392 8.085.069.107 7.358 11.508 12.85 17.483l.003.005.35-.14a5.484 5.484 0 0 1 1.6-.248l13.807-.06 13.806.06a4.686 4.686 0 0 1 1.95.405l.004-.005c5.511-5.997 12.835-17.461 12.852-17.488ZM24.02 16.712a1.418 1.418 0 1 1-1.428 1.418 1.423 1.423 0 0 1 1.428-1.418Zm-9.752 9.992a3.713 3.713 0 1 1 3.74-3.713 3.726 3.726 0 0 1-3.74 3.713Zm12.696 6.217a.406.406 0 0 1-.406.403h-2.49a.407.407 0 0 0-.405.403v2.47a.404.404 0 0 1-.406.402h-2.381a.404.404 0 0 1-.406-.403v-2.47a.407.407 0 0 0-.406-.402h-2.487a.405.405 0 0 1-.406-.403v-2.365a.405.405 0 0 1 .406-.403h2.487a.407.407 0 0 0 .406-.403v-2.47a.406.406 0 0 1 .406-.402h2.381a.406.406 0 0 1 .406.403v2.47a.406.406 0 0 0 .406.402h2.49a.406.406 0 0 1 .405.403Zm1.747-9.195a.424.424 0 0 1-.425.422h-1.714a.424.424 0 0 1-.425-.422v-1.702a.424.424 0 0 1 .425-.422h1.714a.424.424 0 0 1 .425.422Zm11.161-7.014a1.418 1.418 0 1 1-1.428 1.418 1.423 1.423 0 0 1 1.428-1.418Zm-4.808 6.163a1.428 1.428 0 1 1 1.428 1.418 1.423 1.423 0 0 1-1.428-1.418Zm5.313 12.567a3.713 3.713 0 1 1 3.74-3.714 3.726 3.726 0 0 1-3.74 3.714Zm3.526-10.098a2.353 2.353 0 1 1 2.37-2.353 2.362 2.362 0 0 1-2.37 2.353Zm5.031 4.32a2.353 2.353 0 1 1 2.37-2.352 2.362 2.362 0 0 1-2.37 2.353Zm0-8.64a2.353 2.353 0 1 1 2.37-2.354 2.362 2.362 0 0 1-2.37 2.353Zm5.032 4.32a2.353 2.353 0 1 1 2.37-2.353 2.362 2.362 0 0 1-2.37 2.353Z'
    sw_body_p2 = 'M24.03 16.706a1.418 1.418 0 1 1-1.427 1.417 1.423 1.423 0 0 1 1.427-1.417Zm-6.012 6.278a3.74 3.74 0 1 0-3.74 3.714 3.726 3.726 0 0 0 3.74-3.714Zm8.957 7.566a.406.406 0 0 0-.406-.403h-2.49a.407.407 0 0 1-.406-.403v-2.47a.406.406 0 0 0-.405-.403h-2.382a.406.406 0 0 0-.405.403v2.47a.407.407 0 0 1-.406.403h-2.487a.405.405 0 0 0-.406.403v2.364a.405.405 0 0 0 .406.403h2.487a.407.407 0 0 1 .406.403v2.47a.405.405 0 0 0 .405.403h2.381a.404.404 0 0 0 .406-.403v-2.47a.407.407 0 0 1 .406-.403h2.49a.406.406 0 0 0 .406-.403Zm1.746-8.532a.424.424 0 0 0-.425-.422h-1.714a.424.424 0 0 0-.425.422v1.702a.424.424 0 0 0 .425.422h1.714a.424.424 0 0 0 .425-.422Zm9.733-3.895a1.428 1.428 0 1 0 1.428-1.417 1.423 1.423 0 0 0-1.428 1.417Zm-1.952 6.163a1.418 1.418 0 1 0-1.428-1.417 1.423 1.423 0 0 0 1.428 1.417Zm7.625 7.436a3.74 3.74 0 1 0-3.74 3.713 3.726 3.726 0 0 0 3.74-3.713Zm2.156-8.738a2.37 2.37 0 1 0-2.37 2.354 2.362 2.362 0 0 0 2.37-2.354Zm5.032 4.321a2.37 2.37 0 1 0-2.37 2.354 2.362 2.362 0 0 0 2.37-2.354Zm0-8.641a2.37 2.37 0 1 0-2.37 2.353 2.362 2.362 0 0 0 2.37-2.353Zm5.031 4.32a2.37 2.37 0 1 0-2.37 2.354 2.362 2.362 0 0 0 2.37-2.354ZM3.388 24.01a210.465 210.465 0 0 0-1.455 7.707C.973 37.349-.635 47.06.264 49.186c.748 2.711 3.047 5.075 5.516 5.075a5.287 5.287 0 0 0 4.458-2.401c2.174-3.548 4.254-8.073 4.808-9.143a2.687 2.687 0 0 1 1.198-1.216l-.004-.005C10.728 35.5 3.405 24.036 3.388 24.01ZM47.76 41.513l-.004.005a2.687 2.687 0 0 1 1.198 1.215c.554 1.07 2.634 5.595 4.808 9.144a5.287 5.287 0 0 0 4.458 2.401c2.469 0 4.768-2.364 5.516-5.075.9-2.126-.709-11.837-1.67-17.47-.368-2.165-.886-5.012-1.454-7.708-.017.027-7.34 11.49-12.852 17.488Zm4.06-31.047c-2.077-.673-7.634-1.287-9.289.045-.271-.012-.54-.018-.803-.018 0 0-7.296-.116-9.728-.116s-9.728.116-9.728.116q-.403 0-.818.018v.002c-1.65-1.335-7.214-.72-9.293-.047-2.277.736-4.71 2.579-5.27 4.494a19.063 19.063 0 0 1 5.244-2.586c3.429-1.161 10.468-1.272 10.468-1.272l-.004-.002c1.442-.022 7.276-.11 9.4-.11 2.118 0 7.917.087 9.384.11l-.004.002s7.039.11 10.468 1.272a19.063 19.063 0 0 1 5.244 2.586c-.56-1.915-2.993-3.758-5.27-4.494Z'

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Model Viewer - {mdl_path.name}</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: system-ui, -apple-system, sans-serif; overflow: hidden; background: #1a1a2e; }}
    #container {{ width: 100vw; height: 100vh; }}
    .panel {{ position: absolute; background: rgba(20,20,35,0.95); color: #e0e0e0;
              padding: 18px; border-radius: 10px; box-shadow: 0 8px 32px rgba(0,0,0,0.5); }}
    #info {{ top: 20px; left: 20px; max-width: 220px; }}
    #controls {{ top: 20px; right: 20px; max-height: 85vh; overflow-y: auto; 
                 min-width: 240px; max-width: 400px; width: auto;
                 transition: transform 0.3s ease; }}
    #controls.collapsed {{ transform: translateX(calc(100% + 20px)); }}
    #controls-toggle {{ position: absolute; top: 20px; right: 20px; width: 40px; height: 40px; 
                        background: rgba(124, 58, 237, 0.9); border: none; color: white; 
                        border-radius: 8px; cursor: pointer; font-size: 20px; z-index: 1000;
                        display: none; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
    #controls-toggle:hover {{ background: rgba(168, 85, 247, 0.9); }}
    #controls-toggle.visible {{ display: block; }}
    #overlay-toggle {{ position: absolute; top: 20px; left: 20px; width: 40px; height: 40px; 
                        background: rgba(124, 58, 237, 0.9); border: none; color: white; 
                        border-radius: 8px; cursor: pointer; font-size: 20px; z-index: 1000;
                        display: none; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
    #overlay-toggle:hover {{ background: rgba(168, 85, 247, 0.9); }}
    #overlay-toggle.visible {{ display: block; }}
    #stats {{ bottom: 20px; left: 20px; font-family: monospace; font-size: 12px; width: auto; }}
    h3 {{ margin: 0 0 12px 0; color: #7c3aed; font-size: 16px; }}
    h4 {{ margin: 15px 0 10px 0; padding-bottom: 8px; border-bottom: 1px solid rgba(124, 58, 237, 0.3);
          font-size: 14px; color: #a78bfa; font-weight: 500; }}

    /* === Action buttons (full width gradient) === */
    .btn-action {{
      background: linear-gradient(135deg, #6d28d9, #7c3aed, #9333ea); border: none;
      color: white; padding: 11px 16px; margin: 4px 0; cursor: pointer;
      border-radius: 8px; width: 100%; font-weight: 600; font-size: 13px;
      display: flex; align-items: center; gap: 8px; justify-content: center;
      transition: all 0.15s ease;
    }}
    .btn-action:hover {{ filter: brightness(1.15); transform: translateY(-1px); box-shadow: 0 4px 16px rgba(124, 58, 237, 0.35); }}
    .btn-action.active {{ background: linear-gradient(135deg, #059669, #10b981, #34d399); }}

    /* === Toggle row (label + switch) === */
    .toggle-row {{
      display: flex; align-items: center; justify-content: space-between;
      padding: 9px 14px; margin: 4px 0; background: rgba(124, 58, 237, 0.12);
      border-radius: 8px; cursor: pointer; transition: background 0.15s;
    }}
    .toggle-row:hover {{ background: rgba(124, 58, 237, 0.22); }}
    .toggle-row .label {{ display: flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 500; }}

    /* === Toggle switch === */
    .toggle-switch {{
      position: relative; width: 42px; height: 22px; flex-shrink: 0;
    }}
    .toggle-switch input {{ opacity: 0; width: 0; height: 0; }}
    .toggle-switch .slider {{
      position: absolute; top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(100, 100, 120, 0.5); border-radius: 22px;
      transition: background 0.2s; cursor: pointer;
    }}
    .toggle-switch .slider::before {{
      content: ''; position: absolute; width: 16px; height: 16px;
      left: 3px; bottom: 3px; background: #888; border-radius: 50%;
      transition: all 0.2s;
    }}
    .toggle-switch input:checked + .slider {{
      background: linear-gradient(135deg, #7c3aed, #a855f7);
    }}
    .toggle-switch input:checked + .slider::before {{
      transform: translateX(20px); background: white;
    }}

    /* === Slider rows === */
    .slider-row {{
      display: flex; align-items: center; gap: 8px;
      padding: 6px 14px; margin: 2px 0;
    }}
    .slider-row input[type="range"] {{
      flex: 1; cursor: pointer; accent-color: #7c3aed;
    }}

    /* === Mesh toggles === */
    .mesh-toggle {{
      display: flex; align-items: center; margin: 4px 0; padding: 7px 10px;
      background: rgba(124, 58, 237, 0.1); border-radius: 6px; transition: background 0.2s;
    }}
    .mesh-toggle:hover {{ background: rgba(124, 58, 237, 0.2); }}
    .mesh-toggle input {{ margin-right: 10px; cursor: pointer; width: 16px; height: 16px; accent-color: #7c3aed; }}
    .mesh-toggle label {{ cursor: pointer; flex-grow: 1; font-size: 12px; }}
    .texture-indicator {{
      display: inline-block; width: 12px; height: 12px; border-radius: 3px;
      margin-left: 6px; background: linear-gradient(135deg, #10b981, #34d399);
    }}
    .shadow-indicator {{
      display: inline-block; width: 12px; height: 12px; border-radius: 3px;
      margin-left: 4px; background: linear-gradient(135deg, #6b7280, #9ca3af);
    }}

    /* === Select / dropdown === */
    .styled-select {{
      width: 100%; padding: 9px 12px; margin-bottom: 6px;
      background: #2a2a3e; color: #e0e0e0;
      border: 1px solid rgba(124, 58, 237, 0.3); border-radius: 8px;
      font-size: 13px; cursor: pointer; outline: none;
    }}
    .styled-select:focus {{ border-color: #7c3aed; }}
    .styled-select option {{
      background: #2a2a3e; color: #e0e0e0; padding: 6px;
    }}

    /* === Recording button === */
    #btnRecord.recording {{
      background: linear-gradient(135deg, #dc2626, #ef4444) !important;
      animation: pulse-red 1s infinite;
    }}
    @keyframes pulse-red {{
      0%, 100% {{ opacity: 1; }}
      50% {{ opacity: 0.6; }}
    }}

    /* === Modal === */
    #screenshot-modal {{
      display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(0,0,0,0.8); z-index: 2000; align-items: center; justify-content: center;
    }}
    #screenshot-modal.show {{ display: flex; }}

    /* Converting modal */
    #converting-modal {{
      display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(0,0,0,0.8); z-index: 2001; align-items: center; justify-content: center;
    }}
    #converting-modal.show {{ display: flex; }}

    /* Loading overlay - visible by default */
    #loading-overlay {{
      display: flex; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(15,15,25,0.95); z-index: 3000; align-items: center; justify-content: center;
    }}
    #loading-overlay.hidden {{ display: none; }}
    .progress-bar-container {{
      width: 100%; height: 8px; background: rgba(124, 58, 237, 0.2);
      border-radius: 4px; overflow: hidden; margin-top: 16px;
    }}
    .progress-bar-fill {{
      height: 100%; width: 30%; border-radius: 4px;
      background: linear-gradient(90deg, #7c3aed, #a855f7, #7c3aed);
      background-size: 200% 100%;
      animation: progress-sweep 1.5s ease-in-out infinite;
    }}
    @keyframes progress-sweep {{
      0% {{ margin-left: -30%; }}
      100% {{ margin-left: 100%; }}
    }}
    .modal-content {{
      background: rgba(20,20,35,0.98); padding: 30px; border-radius: 12px;
      box-shadow: 0 12px 48px rgba(0,0,0,0.7); max-width: 500px; text-align: center;
      border: 2px solid rgba(124, 58, 237, 0.5);
    }}
    .modal-content h3 {{ color: #7c3aed; margin-bottom: 20px; font-size: 20px; }}
    .modal-content p {{ color: #e0e0e0; margin: 15px 0; font-size: 14px; line-height: 1.6; }}
    .modal-content .filename {{ 
      background: rgba(124, 58, 237, 0.2); padding: 10px; border-radius: 6px;
      font-family: monospace; color: #a78bfa; word-break: break-all; margin: 15px 0;
      cursor: pointer; transition: all 0.2s;
    }}
    .modal-content .filename:hover {{ background: rgba(124, 58, 237, 0.3); }}
    .modal-content button {{ max-width: 200px; margin: 10px auto; }}

    /* === Info panel hints === */
    .info-badge {{
      background: rgba(124, 58, 237, 0.15); padding: 10px; border-radius: 8px;
      font-size: 11px; margin-bottom: 8px;
    }}
    .info-badge .row {{ display: flex; align-items: center; gap: 8px; }}
    .info-badge .row + .row {{ margin-top: 6px; }}

    /* === Section subtitle === */
    .section-title {{
      font-size: 13px; font-weight: 600; color: #a78bfa; margin: 14px 0 8px 0;
      padding-bottom: 6px; border-bottom: 1px solid rgba(124, 58, 237, 0.2);
      display: flex; align-items: center; gap: 6px;
    }}

    /* === Small info text === */
    .info-text {{ font-size: 11px; color: #9ca3af; }}

    /* Scrollbar styling */
    #controls::-webkit-scrollbar {{ width: 6px; }}
    #controls::-webkit-scrollbar-track {{ background: transparent; }}
    #controls::-webkit-scrollbar-thumb {{ background: rgba(124, 58, 237, 0.3); border-radius: 3px; }}
    #controls::-webkit-scrollbar-thumb:hover {{ background: rgba(124, 58, 237, 0.5); }}
  </style>
</head>
<body>
  <!-- Loading overlay - visible by default -->
  <div id="loading-overlay" {'class="hidden"' if skip_popup else ''}>
    <div class="modal-content" style="min-width:340px;text-align:center;">
      <h3 id="loading-title">⏳ Loading Model...</h3>
      <p id="loading-step" style="color:#9ca3af;font-size:13px;">Initializing...</p>
      <div class="progress-bar-container">
        <div class="progress-bar-fill" id="loading-progress" style="width:0%;margin-left:0;animation:none;transition:width 0.3s ease;"></div>
      </div>
      <p id="loading-detail" style="font-size:11px;color:#6b7280;margin-top:8px;"></p>
    </div>
  </div>

  <button id="controls-toggle" onclick="toggleControlsPanel()">☰</button>
  <button id="overlay-toggle" onclick="toggleOverlayPanels()">☰</button>
  <div id="container"></div>
  <div id="info" class="panel">
    <h3>🎮 Model Viewer</h3>
    <p style="font-size: 13px; color: #b0b0b0; line-height: 1.5; margin-bottom: 12px;">
      <strong style="color: #a78bfa;">{mdl_path.name}</strong>
    </p>
    <div class="info-badge">
      <div class="row">
        <span style="font-size: 14px;">🎨</span>
        <span style="color: #9ca3af;" id="texture-info">Loading textures...</span>
      </div>
    </div>
    <div class="info-badge">
      <div class="row">
        <span style="font-size: 14px;">🦴</span>
        <span style="color: #9ca3af;" id="skeleton-info">No skeleton</span>
      </div>
    </div>
    <div class="info-badge">
      <div class="row"><span>🖱️</span> <span style="color:#9ca3af;">Left: Rotate</span></div>
      <div class="row"><span>🖱️</span> <span style="color:#9ca3af;">Right: Pan</span></div>
      <div class="row"><span>🔄</span> <span style="color:#9ca3af;">Wheel: Zoom</span></div>
    </div>
  </div>
  <div id="controls" class="panel">
    <div class="section-title">🎮 Controls</div>

    <div class="toggle-row" onclick="toggleColors(); document.getElementById('swColors').checked = colorMode;">
      <span class="label">🎨 Colors</span>
      <label class="toggle-switch" onclick="event.stopPropagation()">
        <input type="checkbox" id="swColors" onchange="toggleColors()">
        <span class="slider"></span>
      </label>
    </div>
    <div class="toggle-row" onclick="toggleTextures(); document.getElementById('swTex').checked = textureMode;">
      <span class="label">🖼️ Toggle Textures</span>
      <label class="toggle-switch" onclick="event.stopPropagation()">
        <input type="checkbox" id="swTex" checked onchange="toggleTextures()">
        <span class="slider"></span>
      </label>
    </div>
    <div class="toggle-row" onclick="toggleWireframe(); document.getElementById('swWire').checked = wireframeMode;">
      <span class="label">📐 Wireframe Only</span>
      <label class="toggle-switch" onclick="event.stopPropagation()">
        <input type="checkbox" id="swWire" onchange="toggleWireframe()">
        <span class="slider"></span>
      </label>
    </div>
    <div class="toggle-row" onclick="toggleWireframeOverlay(); document.getElementById('swWireOver').checked = wireframeOverlayMode;">
      <span class="label">🔲 Wireframe Overlay</span>
      <label class="toggle-switch" onclick="event.stopPropagation()">
        <input type="checkbox" id="swWireOver" onchange="toggleWireframeOverlay()">
        <span class="slider"></span>
      </label>
    </div>
    <div class="toggle-row" onclick="toggleGamepad(); document.getElementById('swGamepad').checked = gamepadEnabled;">
      <span class="label">🎮 Controller</span>
      <label class="toggle-switch" onclick="event.stopPropagation()">
        <input type="checkbox" id="swGamepad" onchange="toggleGamepad()">
        <span class="slider"></span>
      </label>
    </div>
    <div id="gamepadSubmenu" style="display:none;padding:4px 12px 8px 20px;background:rgba(124,58,237,0.08);border-radius:0 0 8px 8px;margin-top:-2px;">
      <div id="gamepadStatus" style="font-size:11px;color:#a78bfa;margin-bottom:4px;"></div>
      <div class="toggle-row" style="padding:4px 8px;margin:2px 0;background:rgba(124,58,237,0.08)" onclick="toggleGamepadInvertX(); document.getElementById('swGpInvX').checked = !gamepadInvertX;">
        <span class="label" style="font-size:11px">Invert X</span>
        <label class="toggle-switch" onclick="event.stopPropagation()" style="transform:scale(0.85)">
          <input type="checkbox" id="swGpInvX" onchange="toggleGamepadInvertX()">
          <span class="slider"></span>
        </label>
      </div>
      <div class="toggle-row" style="padding:4px 8px;margin:2px 0;background:rgba(124,58,237,0.08)" onclick="toggleGamepadInvertY(); document.getElementById('swGpInvY').checked = gamepadInvertY;">
        <span class="label" style="font-size:11px">Invert Y</span>
        <label class="toggle-switch" onclick="event.stopPropagation()" style="transform:scale(0.85)">
          <input type="checkbox" id="swGpInvY" onchange="toggleGamepadInvertY()">
          <span class="slider"></span>
        </label>
      </div>
      <div style="border-top:1px solid rgba(124,58,237,0.2);margin:4px 0"></div>
      <div class="toggle-row" style="padding:4px 8px;margin:2px 0;background:rgba(59,130,246,0.08)" onclick="toggleFreeCam(); document.getElementById('swFreeCam').checked = freeCamMode;">
        <span class="label" style="font-size:11px">🎥 3D FreeCam</span>
        <label class="toggle-switch" onclick="event.stopPropagation()" style="transform:scale(0.85)">
          <input type="checkbox" id="swFreeCam" onchange="toggleFreeCam()">
          <span class="slider"></span>
        </label>
      </div>
      <div id="freeCamSubmenu" style="display:none;padding:2px 8px 4px 8px;">
        <div style="display:flex;align-items:center;gap:4px;margin:2px 0">
          <span style="font-size:10px;color:#9ca3af;min-width:62px">Cam Speed:</span>
          <input type="range" id="freeCamSpeedSlider" min="2" max="600" value="100" style="flex:1"
                 oninput="freeCamSpeed = this.value / 100; document.getElementById('freeCamStatus').textContent = freeCamSpeed.toFixed(2) + 'x'">
          <span id="freeCamStatus" style="font-size:10px;color:#60a5fa;min-width:45px;text-align:right">1.00x</span>
        </div>
        <div style="display:flex;align-items:center;gap:4px;margin:2px 0">
          <span style="font-size:10px;color:#9ca3af;min-width:62px">Mouse Sens:</span>
          <input type="range" id="freeCamMouseSens" min="-100" max="100" value="0" style="flex:1"
                 oninput="freeCamMouseSensValue = Math.pow(2, this.value / 50); document.getElementById('freeCamSensStatus').textContent = (this.value > 0 ? '+' : '') + this.value + '%'">
          <span id="freeCamSensStatus" style="font-size:10px;color:#60a5fa;min-width:45px;text-align:right">0%</span>
        </div>
      </div>
    </div>

    <div class="slider-row">
      <span class="info-text" style="min-width:52px;">Opacity:</span>
      <input type="range" id="meshOpacity" min="0" max="1" step="0.05" value="1"
             style="flex:1;" oninput="setMeshOpacity(this.value); document.getElementById('meshOpVal').textContent=parseFloat(this.value).toFixed(2)">
      <span id="meshOpVal" class="info-text" style="min-width:28px;text-align:right;color:#a78bfa;">1</span>
    </div>

    <button class="btn-action" onclick="resetView()">🔄 Reset Camera</button>
    <button class="btn-action" onclick="fitToView()">🎯 Focus Model</button>
    <button class="btn-action" onclick="requestScreenshot()">📸 Screenshot</button>
    <button class="btn-action" id="btnRecord" onclick="toggleRecording()">🔴 Record Video</button>

    <div class="section-title" style="cursor:pointer;user-select:none;" onclick="const el=document.getElementById('captureSettings'); el.style.display=el.style.display==='none'?'block':'none'; this.querySelector('.arrow').textContent=el.style.display==='none'?'▶':'▼';">⚙️ Capture Settings <span class="arrow" style="font-size:10px;margin-left:4px;">▶</span></div>
    <div id="captureSettings" style="display:none;">
      <div class="slider-row">
        <span class="info-text" style="min-width:72px;">Screenshot:</span>
        <select id="screenshotScale" class="styled-select" style="width:auto;flex:1;margin:0;padding:6px 8px;">
          <option value="1">1× (native)</option>
          <option value="2" selected>2× (double)</option>
          <option value="4">4× (ultra)</option>
        </select>
      </div>
      <div class="slider-row">
        <span class="info-text" style="min-width:72px;">Video quality:</span>
        <select id="videoQuality" class="styled-select" style="width:auto;flex:1;margin:0;padding:6px 8px;">
          <option value="4000000">Low (4 Mbps)</option>
          <option value="8000000" selected>Medium (8 Mbps)</option>
          <option value="16000000">High (16 Mbps)</option>
          <option value="32000000">Ultra (32 Mbps)</option>
        </select>
      </div>
      <div class="slider-row">
        <span class="info-text" style="min-width:72px;">Video FPS:</span>
        <select id="videoFps" class="styled-select" style="width:auto;flex:1;margin:0;padding:6px 8px;">
          <option value="30">30 FPS</option>
          <option value="60" selected>60 FPS</option>
        </select>
      </div>
      <div class="slider-row">
        <span class="info-text" style="min-width:72px;">Video format:</span>
        <select id="videoFormat" class="styled-select" style="width:auto;flex:1;margin:0;padding:6px 8px;">
          <option value="webm" selected>WebM (VP9)</option>
        </select>
      </div>
      <div class="toggle-row" onclick="overlayMode = (overlayMode + 1) % 3; document.getElementById('overlayModeLabel').textContent = ['Off','Full','Minimal'][overlayMode];">
        <span class="label">📊 Info Overlay</span>
        <span id="overlayModeLabel" style="font-size:11px;color:#a78bfa;min-width:52px;text-align:right;cursor:pointer;">Off</span>
      </div>
    </div>

    <div class="section-title" style="cursor:pointer;user-select:none;" onclick="const el=document.getElementById('lightingSection'); el.style.display=el.style.display==='none'?'block':'none'; this.querySelector('.arrow').textContent=el.style.display==='none'?'▶':'▼';">💡 Lighting <span class="arrow" style="font-size:10px;margin-left:4px;">▶</span></div>
    <div id="lightingSection" style="display:none;">
      <div class="slider-row">
        <span class="info-text" style="min-width:52px;">Ambient:</span>
        <input type="range" id="lightAmbient" min="0" max="2" step="0.05" value="0.6"
               style="flex:1;" oninput="if(ambientLight)ambientLight.intensity=parseFloat(this.value); document.getElementById('lightAmbVal').textContent=parseFloat(this.value).toFixed(2)">
        <span id="lightAmbVal" class="info-text" style="min-width:28px;text-align:right;color:#fbbf24;">0.60</span>
      </div>
      <div class="slider-row">
        <span class="info-text" style="min-width:52px;">Key:</span>
        <input type="range" id="lightKey" min="0" max="2" step="0.05" value="0.8"
               style="flex:1;" oninput="if(dirLight1)dirLight1.intensity=parseFloat(this.value); document.getElementById('lightKeyVal').textContent=parseFloat(this.value).toFixed(2)">
        <span id="lightKeyVal" class="info-text" style="min-width:28px;text-align:right;color:#fbbf24;">0.80</span>
      </div>
      <div class="slider-row">
        <span class="info-text" style="min-width:52px;">Fill:</span>
        <input type="range" id="lightFill" min="0" max="2" step="0.05" value="0.4"
               style="flex:1;" oninput="if(dirLight2)dirLight2.intensity=parseFloat(this.value); document.getElementById('lightFillVal').textContent=parseFloat(this.value).toFixed(2)">
        <span id="lightFillVal" class="info-text" style="min-width:28px;text-align:right;color:#fbbf24;">0.40</span>
      </div>
      <div id="fxoToggleRow" class="toggle-row" style="display:none;" onclick="document.getElementById('swFxo').checked = !document.getElementById('swFxo').checked; setFxoShaders(document.getElementById('swFxo').checked);">
        <span class="label">✨ FXO Shaders</span>
        <label class="toggle-switch" onclick="event.stopPropagation()">
          <input type="checkbox" id="swFxo" checked onchange="setFxoShaders(this.checked)">
          <span class="slider"></span>
        </label>
      </div>
      <div class="toggle-row" onclick="toggleEmissive(); document.getElementById('swEmissive').checked = emissiveEnabled;">
        <span class="label">✨ Emissive Glow</span>
        <label class="toggle-switch" onclick="event.stopPropagation()">
          <input type="checkbox" id="swEmissive" onchange="toggleEmissive()">
          <span class="slider"></span>
        </label>
      </div>
      <div id="emissiveSliderRow" class="slider-row" style="display:none;">
        <span class="info-text" style="min-width:52px;">Glow:</span>
        <input type="range" id="emissiveGlowSlider" min="-1" max="2" step="0.05" value="0"
               style="flex:1;" oninput="emissiveGlowOffset=parseFloat(this.value); applyEmissiveGlow(); document.getElementById('emissiveGlowVal').textContent=(emissiveGlowOffset>=0?'+':'')+emissiveGlowOffset.toFixed(2)">
        <span id="emissiveGlowVal" class="info-text" style="min-width:36px;text-align:right;color:#fbbf24;">+0.00</span>
      </div>
      <button class="btn-action" onclick="document.getElementById('lightAmbient').value=0.6; document.getElementById('lightKey').value=0.8; document.getElementById('lightFill').value=0.4; if(ambientLight)ambientLight.intensity=0.6; if(dirLight1)dirLight1.intensity=0.8; if(dirLight2)dirLight2.intensity=0.4; document.getElementById('lightAmbVal').textContent='0.60'; document.getElementById('lightKeyVal').textContent='0.80'; document.getElementById('lightFillVal').textContent='0.40'; if(emissiveEnabled){{toggleEmissive();}} emissiveGlowOffset=0; document.getElementById('emissiveGlowSlider').value=0; document.getElementById('emissiveGlowVal').textContent='+0.00'; if(!fxoShadersEnabled){{setFxoShaders(true);}} document.getElementById('swFxo').checked=true;">🔄 Reset Lights</button>
    </div>

    
    <div id="skeleton-controls">
      <div class="section-title" style="cursor:pointer;user-select:none;" onclick="const el=document.getElementById('skeletonSection'); el.style.display=el.style.display==='none'?'block':'none'; this.querySelector('.arrow').textContent=el.style.display==='none'?'▶':'▼';">🦴 Skeleton <span class="arrow" style="font-size:10px;margin-left:4px;">▶</span></div>
      <div id="skeletonSection" style="display:none;">
        <div id="skeleton-available" style="display: none;">
          <div class="toggle-row" onclick="toggleSkeleton(); document.getElementById('swSkel').checked = showSkeleton;">
            <span class="label">🦴 Skeleton</span>
            <label class="toggle-switch" onclick="event.stopPropagation()">
              <input type="checkbox" id="swSkel" onchange="toggleSkeleton()">
              <span class="slider"></span>
            </label>
          </div>
          <div class="toggle-row" onclick="toggleJoints(); document.getElementById('swJoints').checked = showJoints;">
            <span class="label">⚪ Joints</span>
            <label class="toggle-switch" onclick="event.stopPropagation()">
              <input type="checkbox" id="swJoints" onchange="toggleJoints()">
              <span class="slider"></span>
            </label>
          </div>
          <div class="toggle-row" onclick="toggleBoneNames(); document.getElementById('swBoneNames').checked = showBoneNames;">
            <span class="label">🏷️ Bone Names</span>
            <label class="toggle-switch" onclick="event.stopPropagation()">
              <input type="checkbox" id="swBoneNames" onchange="toggleBoneNames()">
              <span class="slider"></span>
            </label>
          </div>
        </div>
        <div id="skeleton-unavailable" style="display: block; padding: 10px; background: rgba(124, 58, 237, 0.08); border-radius: 8px; font-size: 11px; color: #9ca3af;">
          ⚠️ Skeleton not loaded
        </div>
      </div>
      
      <div class="section-title" style="cursor:pointer;user-select:none;" onclick="const el=document.getElementById('animationsSection'); el.style.display=el.style.display==='none'?'block':'none'; this.querySelector('.arrow').textContent=el.style.display==='none'?'▶':'▼';">🎬 Animations <span class="arrow" style="font-size:10px;margin-left:4px;">▶</span></div>
      <div id="animationsSection" style="display:none;">
        <div id="animations-available" style="display: none;">
          <select id="animation-select" class="styled-select" onchange="if(this.value) playAnimation(this.value)">
            <option value="">— Select animation —</option>
          </select>
          <button class="btn-action" id="btnAnimToggle" onclick="toggleAnimPlayback()">⏹️ Stop</button>
          <div class="slider-row">
            <span id="animTimeLabel" class="info-text" style="min-width:60px;">0.00 / 0.00</span>
            <input type="range" id="animTimeline" min="0" max="1" step="0.001" value="0"
                   style="flex:1;" oninput="scrubAnimation(this.value)">
          </div>
          <div class="slider-row">
            <span class="info-text" style="min-width:52px;">Speed:</span>
            <input type="range" id="animSpeedSlider" min="-100" max="100" step="5" value="0"
                   style="flex:1;cursor:pointer;" oninput="updateAnimSpeed(this.value)">
            <span id="animSpeedLabel" class="info-text" style="min-width:32px;text-align:right;color:#a78bfa;">1.0x</span>
          </div>
        </div>
        <div id="animations-unavailable" style="display: block; padding: 10px; background: rgba(124, 58, 237, 0.08); border-radius: 8px; font-size: 11px; color: #9ca3af;">
          ⚠️ No animation files found
        </div>
        
        <div id="dynBonesSection" style="display:none;margin-top:8px;">
          <div class="toggle-row" onclick="toggleDynamicBones(); document.getElementById('swPhysics').checked = dynamicBonesEnabled;">
            <span class="label">⚡ Physics</span>
            <label class="toggle-switch" onclick="event.stopPropagation()">
              <input type="checkbox" id="swPhysics" onchange="toggleDynamicBones()">
              <span class="slider"></span>
            </label>
          </div>
          <span id="dynBonesInfo" class="info-text" style="margin-left:14px;"></span>
          <div id="dynIntensityRow" style="display:none;margin-top:4px;">
            <div class="slider-row">
              <span class="info-text" style="min-width:52px;">Intensity:</span>
              <input type="range" id="dynIntensitySlider" min="-400" max="400" step="5" value="0"
                     style="flex:1;cursor:pointer;" oninput="updateDynIntensity(this.value)">
              <span id="dynIntensityLabel" class="info-text" style="min-width:32px;text-align:right;color:#a78bfa;">+0</span>
            </div>
            <div class="toggle-row" onclick="toggleCollisions(); document.getElementById('swCollisions').checked = dynCollisionsEnabled;" style="margin-top:2px;">
              <span class="label">💥 Collisions</span>
              <label class="toggle-switch" onclick="event.stopPropagation()">
                <input type="checkbox" id="swCollisions" checked onchange="toggleCollisions()">
                <span class="slider"></span>
              </label>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="section-title" style="cursor:pointer;user-select:none;" onclick="const el=document.getElementById('meshSection'); el.style.display=el.style.display==='none'?'block':'none'; this.querySelector('.arrow').textContent=el.style.display==='none'?'▶':'▼';">📦 Meshes <span class="arrow" style="font-size:10px;margin-left:4px;">▶</span></div>
    <div id="meshSection" style="display:none;">
      <button class="btn-action" onclick="toggleAllMeshes(true)">✅ Show All</button>
      <button class="btn-action" onclick="toggleAllMeshes(false)">❌ Hide All</button>
      <div id="shadowToggleRow" class="toggle-row" style="display:none;" onclick="toggleHideShadow(); document.getElementById('swShadow').checked = hideShadowMeshes;">
        <span class="label">👤 Hide Shadow</span>
        <label class="toggle-switch" onclick="event.stopPropagation()">
          <input type="checkbox" id="swShadow" checked onchange="toggleHideShadow()">
          <span class="slider"></span>
        </label>
      </div>
      <div class="toggle-row" onclick="xrayHighlight = !xrayHighlight; document.getElementById('swXray').checked = xrayHighlight;">
        <span class="label">👁 X-Ray Highlight</span>
        <label class="toggle-switch" onclick="event.stopPropagation()">
          <input type="checkbox" id="swXray" checked onchange="xrayHighlight = this.checked">
          <span class="slider"></span>
        </label>
      </div>
      <div class="toggle-row" onclick="document.getElementById('swNormals').checked = !document.getElementById('swNormals').checked; setRecomputeNormals(document.getElementById('swNormals').checked);">
        <span class="label">🔄 Recompute Normals</span>
        <label class="toggle-switch" onclick="event.stopPropagation()">
          <input type="checkbox" id="swNormals" {"checked" if recompute_normals else ""} onchange="setRecomputeNormals(this.checked)">
          <span class="slider"></span>
        </label>
      </div>
      <div id="mesh-list"></div>
    </div>
  </div>
  
  <div id="stats" class="panel">
    <div>FPS: --</div>
  </div>

  <div id="screenshot-modal">
    <div class="modal-content">
      <h3>Screenshot Saved</h3>
      <p>Your screenshot has been saved to:</p>
      <div class="filename" id="screenshot-filename" onclick="openScreenshotFile()">filename.png</div>
      <p style="font-size: 12px; color: #9ca3af;">Click filename to open file</p>
      <button class="btn-action" onclick="closeScreenshotModal()">Close</button>
    </div>
  </div>

  <div id="converting-modal">
    <div class="modal-content" style="min-width:320px;">
      <h3 id="converting-title">⏳ Converting Video...</h3>
      <p id="converting-info" style="color:#9ca3af;">Please wait, encoding to MP4</p>
      <div class="progress-bar-container">
        <div class="progress-bar-fill" id="converting-progress"></div>
      </div>
      <p style="font-size:11px;color:#6b7280;margin-top:8px;">This may take a moment depending on video length</p>
    </div>
  </div>

  <script src="three.min.js"></script>
  <script>
    // Debug mode flag (set by Python --debug parameter)
    const DEBUG = {str(debug_mode).lower()};
    const NO_SHADERS = {str(no_shaders).lower()};
    const RECOMPUTE_NORMALS = {str(recompute_normals).lower()};
    
    // Helper function for conditional logging
    function debug(...args) {{
      if (DEBUG) {{
        debug(...args);
      }}
    }}
    
    // On-screen toast notification
    function showToast(msg, duration) {{
      duration = duration || 3000;
      let t = document.getElementById('_toast');
      if (!t) {{
        t = document.createElement('div');
        t.id = '_toast';
        t.style.cssText = 'position:fixed;bottom:60px;left:50%;transform:translateX(-50%);background:#0a0c14;color:#86efac;font:12px/1.5 system-ui,sans-serif;padding:8px 18px;border-radius:10px;z-index:99999;pointer-events:none;white-space:pre;max-width:90%;text-align:center;border:1px solid rgba(34,197,94,0.6)';
        document.body.appendChild(t);
      }}
      t.textContent = msg;
      t.style.display = 'block';
      clearTimeout(t._timer);
      t._timer = setTimeout(() => {{ t.style.display = 'none'; }}, duration);
    }}
    
    const data = {json.dumps(meshes_data)};
    const materials = {json.dumps(materials_json)};
    const skeletonData = {skeleton_json};
    const modelInfo = {model_info_json};
    const bindMatricesData = {bind_matrices_json};
    const animationsData = {animations_json};

    const CONFIG = {{
      INITIAL_BACKGROUND: 0x1a1a2e,
      CAMERA_ZOOM: 1.5,
      AUTO_HIDE_SHADOW: true
    }};

    let scene, camera, renderer, controls;
    let ambientLight, dirLight1, dirLight2;
    let meshes = [];
    let bones = [];
    let skeleton = null;
    let animationMixer = null;
    let currentAnimation = null;
    let clock = new THREE.Clock();
    let modelCenterY = 0;  // Store model bounding box center Y for skeleton offset
    
    let textureLoader = new THREE.TextureLoader();
    let totalTexturesCount = 0;
    let loadedTexturesCount = 0;
    const shaderStats = {{ toon: 0, standard: 0, fxo: 0, types: {{}} }};
    const tangentStats = {{ mdl: 0, computed: 0, none: 0 }};
    function getTangentInfoStr() {{
      if (tangentStats.mdl > 0 && tangentStats.computed === 0) return 'MDL(' + tangentStats.mdl + ')';
      if (tangentStats.mdl === 0 && tangentStats.computed > 0) return 'computed(' + tangentStats.computed + ')';
      if (tangentStats.mdl > 0 && tangentStats.computed > 0) return 'MDL(' + tangentStats.mdl + ')+comp(' + tangentStats.computed + ')';
      return 'none';
    }}
    function getShaderInfoStr(compact) {{
      const typesStr = Object.entries(shaderStats.types).map(([k,v]) => k + '(' + v + ')').join(', ') || 'none';
      if (compact) {{
        if (NO_SHADERS) return 'FXO: off';
        if (shaderStats.toon > 0) {{
          if (!fxoShadersEnabled) return 'FXO: off (toggle)';
          const fxo = shaderStats.fxo > 0 ? '+FXO(' + shaderStats.fxo + ')' : ' FXO:missing';
          return 'Toon(' + shaderStats.toon + ')' + fxo;
        }}
        return 'Std(' + shaderStats.standard + ')';
      }}
      if (NO_SHADERS) return {{ mode: 'FXO: disabled', types: typesStr }};
      if (shaderStats.toon > 0) {{
        if (!fxoShadersEnabled) return {{ mode: 'FXO: off (toggle)', types: typesStr }};
        const fxo = shaderStats.fxo > 0 ? ' · FXO: ' + shaderStats.fxo : ' · FXO: missing';
        return {{ mode: 'Toon: ' + shaderStats.toon + fxo, types: typesStr }};
      }}
      return {{ mode: 'Std: ' + shaderStats.standard, types: typesStr }};
    }}

    let colorMode = false;
    let textureMode = true;
    let wireframeMode = false;
    let emissiveEnabled = false;  // Emissive glow (default OFF — emissive_g is engine-specific, not PBR)
    let emissiveGlowOffset = 0;   // Emissive glow intensity offset (-1..+2, 0=default)
    let wireframeOverlayMode = false;
    let showSkeleton = false, showJoints = false, showBoneNames = false;
    let currentFps = 0;
    const MODEL_FILENAME = {json.dumps(mdl_path.name)};
    
    // Gamepad controller - third-person mode
    let gamepadEnabled = false;
    let gamepadPrevButtons = [];
    let gamepadDeadzone = 0.15;
    let gamepadType = 'generic';
    let gamepadButtonStates = [];
    let gamepadAxesStates = [0,0,0,0];
    let gamepadTriggerStates = [0,0];
    let gamepadInvertX = true;
    let gamepadInvertY = false;
    let gamepadConnectedShown = false;
    let gamepadCurrentId = '';
    let kbUseKeyboard = true;   // start in keyboard mode
    let lastActiveInput = 'keyboard';  // 'keyboard' | 'gamepad' — switch only on real input
    let gamepadLastTimestamp = 0;  // for ghost detection
    let gamepadStaleFrames = 0;   // frames since timestamp changed
    const GAMEPAD_STALE_THRESHOLD = 120;  // ~2s at 60fps
    let gamepadConfirmed = false;  // true once real input seen — disables ghost detection
    let cachedOverlaySVGImg = null;  // pre-rendered controller/kb SVG for screenshots
    let cachedOverlaySVGStr = '';    // last SVG string (to detect changes)
    const kbKeys = {{}};  // currently held keys
    const kbPrevKeys = {{}};  // previous frame keys (for edge detection)
    // Mouse input for controller keyboard mode
    let mouseDeltaX = 0, mouseDeltaY = 0;  // accumulated per frame
    let mouseRightDown = false;
    let mouseWheelDelta = 0;  // accumulated per frame
    
    // Keyboard → gamepad button mapping
    // Movement: WASD, Camera: Arrow keys, Zoom: QE
    const KB_MAP = {{
      // key → gamepad button index for "justPressed" actions
      'Space': 0,       // Play/Pause
      'Escape': 1,      // Stop
      'KeyR': 2,        // Reset position
      'KeyF': 3,        // Dynamic bones
      'BracketLeft': 4, // [ → prev anim
      'BracketRight': 5,// ] → next anim
      'KeyP': 9,        // Screenshot
      'Equal': 12,      // + → speed up
      'Minus': 13,      // - → speed down
      'KeyV': 14,       // V → cycle visual
      'KeyB': 15,       // B → cycle bones
      'KeyO': 8,        // O → overlay
      'KeyC': 11,       // C → toggle FreeCam
      'KeyG': 16,       // G → focus/fit to view
    }};
    const KB_LABELS = {{
      0: 'Space', 1: 'Esc', 2: 'R', 3: 'F',
      4: '[', 5: ']', 6: 'Q', 7: 'E',
      8: 'O', 9: 'P', 11: 'C', 12: '+', 13: '-', 14: 'V', 15: 'B',
    }};

    // Button labels per controller type
    // Controller type: 'xbox', 'playstation', 'switch', 'generic'
    const GP_LABELS = {{
      xbox: {{
        0: 'A', 1: 'B', 2: 'X', 3: 'Y',
        4: 'LB', 5: 'RB', 6: 'LT', 7: 'RT',
        8: 'View', 9: 'Menu', 10: 'LS', 11: 'RS',
        12: '↑', 13: '↓', 14: '←', 15: '→', 16: 'Xbox',
      }},
      playstation: {{
        0: '✕', 1: '○', 2: '□', 3: '△',
        4: 'L1', 5: 'R1', 6: 'L2', 7: 'R2',
        8: 'Share', 9: 'Opt', 10: 'L3', 11: 'R3',
        12: '↑', 13: '↓', 14: '←', 15: '→', 16: 'PS',
      }},
      switch: {{
        0: 'B', 1: 'A', 2: 'Y', 3: 'X',
        4: 'L', 5: 'R', 6: 'ZL', 7: 'ZR',
        8: '−', 9: '+', 10: 'LS', 11: 'RS',
        12: '↑', 13: '↓', 14: '←', 15: '→', 16: '⌂',
      }},
      generic: {{
        0: '①', 1: '②', 2: '③', 3: '④',
        4: 'L1', 5: 'R1', 6: 'L2', 7: 'R2',
        8: 'Sel', 9: 'Start', 10: 'L3', 11: 'R3',
        12: '↑', 13: '↓', 14: '←', 15: '→',
      }},
      keyboard: {{
        0: 'Spc', 1: 'Esc', 2: 'R', 3: 'F',
        4: '[', 5: ']', 6: 'Q', 7: 'E',
        8: 'O', 9: 'P', 10: '', 11: 'C',
        12: '+', 13: '-', 14: 'V', 15: 'B',
        16: 'G',
      }}
    }};

    // Button colors per type (face buttons 0-3)
    const GP_COLORS = {{
      xbox: {{ 0: '#3ddc84', 1: '#f44336', 2: '#2196f3', 3: '#ffc107' }},
      playstation: {{ 0: '#5c9dff', 1: '#f44336', 2: '#e991d0', 3: '#4cdfad' }},
      switch: {{ 0: '#ffdc00', 1: '#ff4136', 2: '#39cccc', 3: '#0074d9' }},
      generic: {{ 0: '#888', 1: '#888', 2: '#888', 3: '#888' }},
      keyboard: {{ 0: '#60a5fa', 1: '#f87171', 2: '#4ade80', 3: '#fbbf24' }}
    }};

    // Action mapping (button index → description)
    const GP_ACTIONS = {{
      0: 'Play/Pause', 1: 'Stop', 2: 'Reset Pos', 3: 'Physics',
      4: 'Prev Anim', 5: 'Next Anim', 6: 'Zoom Out', 7: 'Zoom In',
      8: 'Overlay', 9: 'Screenshot', 10: 'Focus', 11: 'FreeCam', 12: 'Speed+', 13: 'Speed-', 14: 'Visual', 15: 'Bones', 16: 'Focus'
    }};
    const GP_STICK_ACTIONS = {{ ls: 'Move', rs: 'Camera' }};

    // Display names per type
    const GP_TYPE_NAMES = {{
      xbox: 'Xbox Controller',
      playstation: 'PlayStation Controller',
      switch: 'Switch Pro Controller',
      keyboard: 'Keyboard',
      generic: 'Gamepad Connected'
    }};

    // ── SVG body paths from reference controller icons ──
    const CTRL_BODY = {{
      xbox: ['{xbox_body_p1}', '{xbox_body_p2}'],
      ps: ['{ps_body_p1}', '{ps_body_p2}'],
      sw: ['{sw_body_p1}', '{sw_body_p2}']
    }};

    function renderControllerSVG(tp, bs, ax, tr, labels, colors) {{
      const isPS = (tp === 'playstation');
      const isSW = (tp === 'switch');
      const isXB = (tp === 'xbox' || tp === 'generic');
      // viewBox 0 0 64 64, display 420×280 → controller centered
      const W = 420, H = 280;
      let s = '<svg xmlns="http://www.w3.org/2000/svg" width="'+W+'" height="'+H+'" viewBox="0 7 64 45" preserveAspectRatio="xMidYMid meet" style="display:block;margin:2px auto">';
      s += '<defs>';
      s += '<radialGradient id="stG"><stop offset="0%" stop-color="#4a4a5a"/><stop offset="100%" stop-color="#1a1a2a"/></radialGradient>';
      s += '<radialGradient id="stGA"><stop offset="0%" stop-color="#a78bfa"/><stop offset="100%" stop-color="#4c1d95"/></radialGradient>';
      s += '</defs>';

      // Body from reference SVG paths
      const bk = isPS ? 'ps' : isSW ? 'sw' : 'xbox';
      s += '<path d="'+CTRL_BODY[bk][0]+'" fill="#1c1c30" stroke="#2a2a40" stroke-width="0.2"/>';
      s += '<path d="'+CTRL_BODY[bk][1]+'" fill="#2a2a42" opacity="0.35"/>';

      // ═══ Helpers (all coords in 64-unit space) ═══
      const SR = 3.2, CR = 2.0, MV = 1.8;  // stick socket, cap, movement
      const FR = 2.0;   // face btn radius
      const SW = 0.2;   // stroke width
      const FS1 = 1.6, FS2 = 1.2;  // font sizes

      function stickVis(cx, cy, axX, axY, concave, pressed) {{
        const act = Math.abs(axX) > 0 || Math.abs(axY) > 0;
        const dx = cx + axX * MV, dy = cy + axY * MV;
        s += '<circle cx="'+cx+'" cy="'+cy+'" r="'+SR+'" fill="'+(pressed?'rgba(124,58,237,0.25)':'#0a0a18')+'" stroke="'+(pressed?'#a78bfa':(act?'#5b21b6':'#2a2a3a'))+'" stroke-width="'+(pressed?SW*1.5:SW)+'"/>';
        s += '<circle cx="'+dx+'" cy="'+dy+'" r="'+CR+'" fill="url(#'+(act||pressed?'stGA':'stG')+')" stroke="'+(act||pressed?'#a78bfa':'#444')+'" stroke-width="'+SW+'"/>';
        if (concave) {{
          s += '<circle cx="'+dx+'" cy="'+dy+'" r="'+(CR*0.6)+'" fill="none" stroke="'+(act||pressed?'rgba(167,139,250,0.3)':'rgba(60,60,80,0.5)')+'" stroke-width="0.1"/>';
        }} else {{
          s += '<circle cx="'+dx+'" cy="'+dy+'" r="'+(CR*0.55)+'" fill="none" stroke="'+(act||pressed?'rgba(167,139,250,0.15)':'rgba(60,60,80,0.2)')+'" stroke-width="0.1"/>';
        }}
      }}

      function faceBtn(cx, cy, idx, r) {{
        const p = bs[idx]; const fc = colors[idx];
        const fill = p ? (fc || '#7c3aed') : 'rgba(37,37,64,0.7)';
        const stroke = p ? (fc ? '#fff' : '#a78bfa') : (fc || 'rgba(68,68,68,0.6)');
        const tf = p ? (fc ? '#000' : '#fff') : (fc || '#777');
        s += '<circle cx="'+cx+'" cy="'+cy+'" r="'+r+'" fill="'+fill+'" stroke="'+stroke+'" stroke-width="'+(SW*0.8)+'"/>';
        s += '<text x="'+cx+'" y="'+(cy+r*0.35)+'" text-anchor="middle" fill="'+tf+'" font-size="'+FS2+'" font-family="sans-serif" font-weight="bold">'+(labels[idx]||'')+'</text>';
      }}

      function dpadVis(cx, cy, style) {{
        const aw = 1.2, al = 3.5;  // arm width, length
        if (style === 'ps') {{
          const dd = 2.5;
          [[12,0,-dd],[13,0,dd],[14,-dd,0],[15,dd,0]].forEach(function(d) {{
            const idx=d[0], ox=d[1], oy=d[2]; const p = bs[idx];
            s += '<rect x="'+(cx+ox-aw*0.7)+'" y="'+(cy+oy-aw*0.7)+'" width="'+(aw*1.4)+'" height="'+(aw*1.4)+'" rx="0.3" fill="'+(p?'rgba(124,58,237,0.7)':'rgba(30,30,54,0.6)')+'" stroke="'+(p?'#a78bfa':'rgba(58,58,80,0.5)')+'" stroke-width="0.08"/>';
          }});
        }} else {{
          s += '<rect x="'+(cx-aw/2)+'" y="'+(cy-al)+'" width="'+aw+'" height="'+(al*2)+'" rx="0.3" fill="rgba(26,26,46,0.6)" stroke="rgba(58,58,80,0.4)" stroke-width="0.08"/>';
          s += '<rect x="'+(cx-al)+'" y="'+(cy-aw/2)+'" width="'+(al*2)+'" height="'+aw+'" rx="0.3" fill="rgba(26,26,46,0.6)" stroke="rgba(58,58,80,0.4)" stroke-width="0.08"/>';
          [[12,0,-2.2],[13,0,2.2],[14,-2.2,0],[15,2.2,0]].forEach(function(d) {{
            const idx=d[0], ox=d[1], oy=d[2]; const p = bs[idx];
            if (p) s += '<rect x="'+(cx+ox-0.6)+'" y="'+(cy+oy-0.6)+'" width="1.2" height="1.2" rx="0.15" fill="rgba(124,58,237,0.6)"/>';
          }});
        }}
      }}

      function trigBar(x, y, w, idx, val, side) {{
        const pct = Math.max(0, Math.min(1, val)) * w; const act = val > 0.05;
        s += '<rect x="'+x+'" y="'+y+'" width="'+w+'" height="1.8" rx="0.5" fill="#0d0d1a" stroke="rgba(58,58,80,0.4)" stroke-width="0.08"/>';
        if (pct > 0.2) {{
          const fx = (side==='r') ? (x + w - pct) : x;
          s += '<rect x="'+fx+'" y="'+y+'" width="'+Math.max(1,pct)+'" height="1.8" rx="0.5" fill="rgba(124,58,237,'+(0.3+val*0.65)+')"/>';
        }}
        s += '<text x="'+(x+w/2)+'" y="'+(y+1.3)+'" text-anchor="middle" fill="'+(act?'#ddd':'#555')+'" font-size="1" font-family="sans-serif">'+(labels[idx]||'')+'</text>';
      }}

      function bumperBar(x, y, w, idx) {{
        const p = bs[idx];
        s += '<rect x="'+x+'" y="'+y+'" width="'+w+'" height="2" rx="1" fill="'+(p?'#7c3aed':'rgba(37,37,64,0.6)')+'" stroke="'+(p?'#a78bfa':'rgba(68,68,68,0.4)')+'" stroke-width="0.1"/>';
        s += '<text x="'+(x+w/2)+'" y="'+(y+1.4)+'" text-anchor="middle" fill="'+(p?'#fff':'#777')+'" font-size="1" font-family="sans-serif">'+(labels[idx]||'')+'</text>';
      }}

      function smallBtn(cx, cy, idx, shape, extra) {{
        const p = bs[idx]; const lbl = labels[idx] || '';
        if (shape === 'circle') {{
          s += '<circle cx="'+cx+'" cy="'+cy+'" r="1.1" fill="'+(p?'#7c3aed':'rgba(37,37,64,0.6)')+'" stroke="'+(p?'#a78bfa':'rgba(68,68,68,0.4)')+'" stroke-width="0.08"/>';
          s += '<text x="'+cx+'" y="'+(cy+0.4)+'" text-anchor="middle" fill="'+(p?'#fff':'#555')+'" font-size="0.9">'+(extra||lbl)+'</text>';
        }} else {{
          s += '<rect x="'+(cx-1.5)+'" y="'+(cy-0.7)+'" width="3" height="1.4" rx="0.4" fill="'+(p?'#7c3aed':'rgba(37,37,64,0.6)')+'" stroke="'+(p?'#a78bfa':'rgba(68,68,68,0.4)')+'" stroke-width="0.08"/>';
          s += '<text x="'+cx+'" y="'+(cy+0.4)+'" text-anchor="middle" fill="'+(p?'#fff':'#666')+'" font-size="0.8">'+lbl+'</text>';
        }}
      }}

      // ═══ LAYOUTS from SVG element positions ═══
      if (isPS) {{
        // ── DualSense: SVG positions with small correction (−1.2x, +1.2y) ──
        trigBar(7, 8.5, 13, 6, tr[0], 'l');
        trigBar(44, 8.5, 13, 7, tr[1], 'r');
        bumperBar(7, 11, 13, 4);
        bumperBar(44, 11, 13, 5);
        // D-pad
        dpadVis(12.0, 23.0, 'ps');
        // Face buttons △○□✕: shifted ~(-1.2, +1.2) from SVG icon positions
        faceBtn(51.5, 18.7, 3, FR); faceBtn(46.8, 23.4, 2, FR);
        faceBtn(56.2, 23.4, 1, FR); faceBtn(51.5, 28.1, 0, FR);
        // Sticks
        stickVis(21.7, 31.8, ax[0], ax[1], true, bs[10]);
        stickVis(42.2, 31.8, ax[2], ax[3], true, bs[11]);
        // Create / Options (horizontal pills)
        smallBtn(17.8, 16.5, 8, 'pill', null);
        smallBtn(46.0, 16.5, 9, 'pill', null);
        // PS button
        smallBtn(32.0, 35.0, 16, 'circle', 'PS');
      }} else if (isSW) {{
        trigBar(6, 8, 14, 6, tr[0], 'l');
        trigBar(44, 8, 14, 7, tr[1], 'r');
        bumperBar(6, 10.5, 14, 4);
        bumperBar(44, 10.5, 14, 5);
        // Left stick upper-left
        stickVis(16.1, 23.0, ax[0], ax[1], true, bs[10]);
        // D-pad lower-left
        dpadVis(22.0, 31.0, 'cross');
        // Face buttons upper-right
        faceBtn(50.1, 18.8, 3, FR); faceBtn(45.1, 23.2, 2, FR);
        faceBtn(55.2, 23.2, 1, FR); faceBtn(50.1, 27.5, 0, FR);
        // Right stick lower inward
        stickVis(42.2, 33.0, ax[2], ax[3], true, bs[11]);
        // −/+
        smallBtn(23.3, 17.4, 8, 'circle', '−');
        smallBtn(39.2, 17.4, 9, 'circle', '+');
        smallBtn(37.0, 24.0, 16, 'circle', '⌂');
      }} else {{
        // Xbox
        trigBar(6, 8, 14, 6, tr[0], 'l');
        trigBar(44, 8, 14, 7, tr[1], 'r');
        bumperBar(6, 10.5, 14, 4);
        bumperBar(44, 10.5, 14, 5);
        // Left stick upper-left
        stickVis(15.6, 21.9, ax[0], ax[1], false, bs[10]);
        // D-pad lower-left
        dpadVis(23.5, 31.5, 'cross');
        // Face YXBA upper-right
        faceBtn(48.4, 17.7, 3, 2.1); faceBtn(44.1, 22.0, 2, 2.1);
        faceBtn(52.7, 22.0, 1, 2.1); faceBtn(48.4, 26.2, 0, 2.1);
        // Right stick lower inward
        stickVis(40.4, 31.5, ax[2], ax[3], false, bs[11]);
        // Xbox guide
        s += '<circle cx="32" cy="15.4" r="1.8" fill="#107c10" stroke="#2dd42d" stroke-width="0.15" opacity="0.7"/>';
        s += '<text x="32" y="15.9" text-anchor="middle" fill="#4ade80" font-size="1.4" font-weight="bold">X</text>';
        // View & Menu
        smallBtn(27.4, 21.8, 8, 'circle', '⧉');
        smallBtn(36.7, 21.8, 9, 'circle', '≡');
      }}

      s += '</svg>';
      return s;
    }}
    
    function renderKeyboardSVG(bs, ax, tr) {{
      const W = 410, H = 100;
      const k = 27, g = 3, rh = 24, step = 30;
      const mx = 25;
      function col(i) {{ return mx + i * step; }}
      const y1 = 11, y2 = y1+rh+g, y3 = y2+rh+g;
      let s = '<svg xmlns="http://www.w3.org/2000/svg" width="'+W+'" height="'+H+'" viewBox="0 0 '+W+' '+H+'" style="display:block;margin:4px auto 2px">';
      s += '<rect x="2" y="2" width="'+(W-4)+'" height="'+(H-4)+'" rx="8" fill="#181828" stroke="#3a3a50" stroke-width="1.2"/>';
      function key(x, y, w, h, label, pressed) {{
        const bg = pressed ? '#7c3aed' : '#222238';
        const stk = pressed ? '#a78bfa' : '#2e2e42';
        const tf = pressed ? '#fff' : '#aaa';
        s += '<rect x="'+x+'" y="'+(y+2)+'" width="'+w+'" height="'+h+'" rx="4" fill="#0e0e1a"/>';
        s += '<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+(h-1)+'" rx="4" fill="'+bg+'" stroke="'+stk+'" stroke-width="0.8"/>';
        s += '<rect x="'+(x+1)+'" y="'+(y+1)+'" width="'+(w-2)+'" height="2.5" rx="1" fill="rgba(255,255,255,'+(pressed?'0.18':'0.06')+')"/>';
        const fs = label.length > 2 ? 8 : (label.length > 1 ? 9 : 11);
        s += '<text x="'+(x+w/2)+'" y="'+(y+h/2+3)+'" text-anchor="middle" fill="'+tf+'" font-size="'+fs+'" font-family="monospace" font-weight="bold">'+label+'</text>';
      }}
      key(col(1),y1,k,rh,'W',ax[1]<0);
      key(col(4),y1,k,rh,'↑',ax[3]<0);
      key(col(6),y1,k,rh,'Spc',bs[0]); key(col(7),y1,k,rh,'Esc',bs[1]);
      key(col(8),y1,k,rh,'R',bs[2]); key(col(9),y1,k,rh,'F',bs[3]);
      key(col(10),y1,k,rh,'V',bs[14]); key(col(11),y1,k,rh,'B',bs[15]);
      key(col(0),y2,k,rh,'A',ax[0]<0); key(col(1),y2,k,rh,'S',ax[1]>0); key(col(2),y2,k,rh,'D',ax[0]>0);
      key(col(3),y2,k,rh,'←',ax[2]<0); key(col(4),y2,k,rh,'↓',ax[3]>0); key(col(5),y2,k,rh,'→',ax[2]>0);
      key(col(6),y2,k,rh,'[',bs[4]); key(col(7),y2,k,rh,']',bs[5]);
      key(col(8),y2,k,rh,'+',bs[12]); key(col(9),y2,k,rh,'−',bs[13]);
      key(col(10),y2,k,rh,'O',bs[8]); key(col(11),y2,k,rh,'P',bs[9]);
      key(col(0),y3,k,rh,'Q',tr[0]>0.5); key(col(1),y3,k,rh,'C',bs[11]); key(col(2),y3,k,rh,'E',tr[1]>0.5);
      key(col(8),y3,k,rh,'G',bs[16]);
      s += '</svg>';
      return s;
    }}

    function renderMappingLegend(tp, labels) {{
      const isKB = (tp === 'keyboard');
      const items = [[0,'Play/Pause'],[1,'Stop'],[2,'Reset Pos'],[3,'Physics'],[4,'Prev Anim'],[5,'Next Anim'],[6,'Zoom Out'],[7,'Zoom In'],[14,'Visual Mode'],[15,'Bone Cycle'],[12,'Speed +'],[13,'Speed −'],[8,'Overlay'],[9,'Screenshot']];
      if (isKB) {{ items.push([16,'Focus'],[11,'FreeCam']); }}
      else {{ items.push([10,'Focus'],[11,'FreeCam']); }}
      let h = '<div style="margin-top:4px;font-size:10px;line-height:17px;color:#888;columns:2;column-gap:12px">';
      items.forEach(function(it) {{
        const lbl = labels[it[0]] || '';
        if (!lbl) return;
        h += '<div style="white-space:nowrap"><span style="display:inline-block;min-width:32px;padding:1px 4px;background:#252540;border:1px solid #333;border-radius:3px;color:#ccc;text-align:center;font-family:monospace;font-size:9px">' + lbl + '</span> ' + it[1] + '</div>';
      }});
      if (!isKB) {{
        h += '<div style="white-space:nowrap"><span style="display:inline-block;min-width:32px;padding:1px 4px;background:#252540;border:1px solid #333;border-radius:3px;color:#ccc;text-align:center;font-size:9px">L🕹</span> Move</div>';
        h += '<div style="white-space:nowrap"><span style="display:inline-block;min-width:32px;padding:1px 4px;background:#252540;border:1px solid #333;border-radius:3px;color:#ccc;text-align:center;font-size:9px">R🕹</span> Camera</div>';
      }}
      h += '</div>';
      return h;
    }}

    let characterGroup = null;
    let characterYaw = 0;
    let characterMoveSpeed = 0;
    let characterCenterY = 0;
    let tpCamTheta = Math.PI;  // behind character
    let tpCamPhi = 1.2;        // slight top-down angle
    let tpCamDist = 1;
    let groundGrid = null;
    let tpAutoAnimWalk = null;
    let tpAutoAnimIdle = null;
    let tpCurrentAutoAnim = null;
    let tpIsMoving = false;
    let freeCamMode = false;      // free camera fly mode
    let freeCamSpeed = 1.0;       // speed multiplier (adjusted by D-pad)
    let freeCamMouseSensValue = 1.0; // mouse sensitivity multiplier
    let overlayMode = 0; // 0=off, 1=full, 2=minimal
    let freeCamBaseSpeed = 0.1;   // computed from model size
    let skeletonGroup = null, jointsGroup = null;

    class OrbitControls {{
      constructor(camera, domElement) {{
        this.camera = camera;
        this.domElement = domElement;
        this.target = new THREE.Vector3();
        this.spherical = new THREE.Spherical();
        this.sphericalDelta = new THREE.Spherical();
        this.scale = 1;
        this.panOffset = new THREE.Vector3();
        this.isMouseDown = false;
        this.rotateSpeed = 0.5;
        this.zoomSpeed = 1;
        this.panSpeed = 1;
        this.mouseButtons = {{LEFT: 0, MIDDLE: 1, RIGHT: 2}};
        
        this.domElement.addEventListener('contextmenu', e => e.preventDefault());
        this.domElement.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.domElement.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.domElement.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.domElement.addEventListener('wheel', this.onMouseWheel.bind(this));
      }}
      
      onMouseDown(e) {{
        if (this._tpMode && !freeCamMode) return;
        this.isMouseDown = true;
        this.mouseButton = e.button;
      }}
      
      onMouseUp() {{
        this.isMouseDown = false;
      }}
      
      onMouseMove(e) {{
        if (!this.isMouseDown) return;
        
        // FreeCam mouse handling
        if (freeCamMode && this._tpMode) {{
          const mx = e.movementX || 0;
          const my = e.movementY || 0;
          if (mx === 0 && my === 0) return;
          const sens = 0.0015 * freeCamMouseSensValue;
          const invX = gamepadInvertX ? -1 : 1;
          const invY = gamepadInvertY ? -1 : 1;
          if (this.mouseButton === 0) {{
            tpCamTheta += mx * sens * invX;
            tpCamPhi = Math.max(0.1, Math.min(Math.PI - 0.1, tpCamPhi + my * sens * invY));
          }} else if (this.mouseButton === 2) {{
            const speed = freeCamBaseSpeed * freeCamSpeed * 0.08 * freeCamMouseSensValue;
            const phi = tpCamPhi, theta = tpCamTheta;
            const lookDir = new THREE.Vector3(
              -Math.sin(phi) * Math.sin(theta), -Math.cos(phi),
              -Math.sin(phi) * Math.cos(theta)).normalize();
            const strafeDir = new THREE.Vector3(lookDir.z, 0, -lookDir.x).normalize();
            this.camera.position.addScaledVector(strafeDir, mx * speed * invX);
            this.camera.position.y += my * speed * invY;
          }}
          return;
        }}
        
        if (this._tpMode) return;
        
        if (this.mouseButton === this.mouseButtons.LEFT) {{
          const dx = e.movementX * this.rotateSpeed * 0.01;
          const dy = e.movementY * this.rotateSpeed * 0.01;
          this.sphericalDelta.theta -= dx;
          this.sphericalDelta.phi -= dy;
        }} else if (this.mouseButton === this.mouseButtons.RIGHT) {{
          const cam = this.camera;
          const right = new THREE.Vector3(cam.matrix.elements[0], cam.matrix.elements[1], cam.matrix.elements[2]);
          const up = new THREE.Vector3(cam.matrix.elements[4], cam.matrix.elements[5], cam.matrix.elements[6]);
          const distScale = cam.position.distanceTo(this.target) * 0.001;
          this.panOffset.add(right.multiplyScalar(-e.movementX * this.panSpeed * distScale));
          this.panOffset.add(up.multiplyScalar(e.movementY * this.panSpeed * distScale));
        }}
      }}
      
      onMouseWheel(e) {{
        if (freeCamMode && this._tpMode) {{
          e.preventDefault();
          const speed = freeCamBaseSpeed * freeCamSpeed;
          const phi = tpCamPhi, theta = tpCamTheta;
          const lookDir = new THREE.Vector3(
            -Math.sin(phi) * Math.sin(theta), -Math.cos(phi),
            -Math.sin(phi) * Math.cos(theta)).normalize();
          this.camera.position.addScaledVector(lookDir, -e.deltaY * speed * 0.02 * freeCamMouseSensValue);
          return;
        }}
        if (this._tpMode) return;
        e.preventDefault();
        this.scale *= Math.pow(0.95, -e.deltaY * this.zoomSpeed * 0.05);
      }}
      
      update() {{
        const offset = new THREE.Vector3();
        const quat = new THREE.Quaternion().setFromUnitVectors(this.camera.up, new THREE.Vector3(0, 1, 0));
        
        offset.copy(this.camera.position).sub(this.target);
        offset.applyQuaternion(quat);
        
        this.spherical.setFromVector3(offset);
        this.spherical.theta += this.sphericalDelta.theta;
        this.spherical.phi += this.sphericalDelta.phi;
        this.spherical.phi = Math.max(0.01, Math.min(Math.PI - 0.01, this.spherical.phi));
        this.spherical.radius *= this.scale;
        
        this.target.add(this.panOffset);
        
        offset.setFromSpherical(this.spherical);
        offset.applyQuaternion(quat.invert());
        
        this.camera.position.copy(this.target).add(offset);
        this.camera.lookAt(this.target);
        
        this.sphericalDelta.set(0, 0, 0);
        this.scale = 1;
        this.panOffset.set(0, 0, 0);
      }}
    }}

    function loadTexture(url, wrapS, wrapT, onLoad, onError) {{
      const wrapModes = [
        THREE.RepeatWrapping,
        THREE.MirroredRepeatWrapping,
        THREE.ClampToEdgeWrapping
      ];
      
      const wrapSMode = wrapModes[wrapS] || THREE.RepeatWrapping;
      const wrapTMode = wrapModes[wrapT] || THREE.RepeatWrapping;
      
      textureLoader.load(url, 
        texture => {{
          texture.wrapS = wrapSMode;
          texture.wrapT = wrapTMode;
          texture.needsUpdate = true;
          loadedTexturesCount++;
          updateTextureStatus();
          if (onLoad) onLoad(texture);
        }},
        undefined,
        error => {{
          if (DEBUG) console.error('Error loading texture:', url, error);
          if (onError) onError(error);
        }}
      );
    }}

    function createMaterial(materialName, meshName) {{
      const matData = materials[materialName];
      
      if (!matData) {{
        return new THREE.MeshStandardMaterial({{
          color: 0x808080, roughness: 0.7, metalness: 0.2, skinning: true
        }});
      }}

      const shaderType = matData._shaderType || '';
      const sp = matData._shaderParams || {{}};
      const isChrShader = shaderType.startsWith('chr_');
      
      if (isChrShader && !NO_SHADERS) {{
        // Use MeshPhongMaterial with onBeforeCompile for toon effects
        // This preserves native skinning support while adding rim light + toon shading
        
        // FXO capabilities: if present, only apply effects confirmed by compiled shader
        const fxoCaps = matData._fxoCaps;
        const fxoUniforms = fxoCaps ? new Set(fxoCaps.uniforms) : null;
        const hasFxo = fxoUniforms !== null;
        
        // Check if effect is enabled: FXO present → must be in uniforms; no FXO → always enabled
        const hasRim = !hasFxo || fxoUniforms.has('rimIntensity_g');
        const hasToonEdge = !hasFxo || fxoUniforms.has('toonEdgeStrength_g');
        const hasShadowColors = !hasFxo || fxoUniforms.has('shadowColor1_g');
        const hasSpecular = !hasFxo || fxoUniforms.has('specularColor_g');
        const hasEmissive = !hasFxo || fxoUniforms.has('emissive_g');
        
        const rimColor = hasRim ? (sp.rimLightColor_g || [0.3, 0.3, 0.4]) : [0,0,0];
        const rimIntensity = hasRim ? (sp.rimIntensity_g != null ? sp.rimIntensity_g : 0.8) : 0;
        const rimPower = hasRim ? (sp.rimLightPower_g || 4.0) : 1;
        const shadowCol1 = hasShadowColors ? (sp.shadowColor1_g || sp.shadowColor_g || [0.15, 0.1, 0.2]) : [0,0,0];
        const shadowCol2 = hasShadowColors ? (sp.shadowColor2_g || sp.shadowColor_g || [0.08, 0.05, 0.12]) : [0,0,0];
        const shadowSharpness = hasShadowColors ? (sp.shadowGradSharpness_g || 0.5) : 0;
        const toonEdge = hasToonEdge ? (sp.toonEdgeStrength_g || 0.0) : 0;
        const toonEdgeCol = hasToonEdge ? (sp.toonEdgeColor_g || [0.0, 0.0, 0.0]) : [0,0,0];
        
        const matParams = {{
          color: 0xffffff,
          shininess: hasSpecular ? (sp.specularGlossiness_g || sp.specularGlossiness0_g || 25.0) : 30,
          specular: hasSpecular ? new THREE.Color().fromArray(sp.specularColor_g || [0.2, 0.2, 0.3]) : new THREE.Color(0x333344),
          side: THREE.DoubleSide,
          skinning: true,
        }};
        
        // Note: emissive_g in Trails engine shaders is NOT PBR emission —
        // it's a shader-specific brightness parameter. Store for optional toggle.
        if (hasEmissive && sp.emissive_g && sp.emissive_g > 0) {{
          matParams._emissiveGlow = Math.min(sp.emissive_g * 0.2, 0.5);
        }}
        
        if (sp.Switch_AlphaTest === 1) {{
          matParams.alphaTest = sp.alphaTestThreshold_g || 0.5;
          matParams.transparent = true;
        }}
        
        const mat = new THREE.MeshPhongMaterial(matParams);
        mat.userData.isToonMaterial = true;
        mat.userData.shaderType = shaderType;
        mat.userData.hasFxo = hasFxo;
        if (matParams._emissiveGlow) mat.userData.emissiveGlow = matParams._emissiveGlow;
        shaderStats.toon++;
        shaderStats.types[shaderType] = (shaderStats.types[shaderType] || 0) + 1;
        if (hasFxo) shaderStats.fxo = (shaderStats.fxo || 0) + 1;
        
        // Inject rim light + toon shading via onBeforeCompile (only for effects confirmed by FXO or fallback)
        const needsCompileHook = hasRim || hasToonEdge;
        if (needsCompileHook) {{
        mat.onBeforeCompile = function(shader) {{
          // Add custom uniforms
          shader.uniforms.uRimIntensity = {{ value: rimIntensity }};
          shader.uniforms.uRimPower = {{ value: rimPower }};
          shader.uniforms.uRimColor = {{ value: new THREE.Color(rimColor[0], rimColor[1], rimColor[2]) }};
          shader.uniforms.uToonEdge = {{ value: toonEdge }};
          shader.uniforms.uToonEdgeColor = {{ value: new THREE.Color(toonEdgeCol[0], toonEdgeCol[1], toonEdgeCol[2]) }};
          
          // Inject uniforms into fragment shader
          shader.fragmentShader = shader.fragmentShader.replace(
            'uniform float opacity;',
            `uniform float opacity;
            uniform float uRimIntensity;
            uniform float uRimPower;
            uniform vec3 uRimColor;
            uniform float uToonEdge;
            uniform vec3 uToonEdgeColor;`
          );
          
          // Inject toon + rim effects after output_fragment (where gl_FragColor is set)
          shader.fragmentShader = shader.fragmentShader.replace(
            '#include <output_fragment>',
            `#include <output_fragment>
            vec3 toonViewDir = normalize(vViewPosition);
            ${{hasRim ? `
            // Rim light
            float rimDot = 1.0 - max(dot(normal, toonViewDir), 0.0);
            float rimFactor = pow(rimDot, uRimPower) * uRimIntensity;
            gl_FragColor.rgb += uRimColor * rimFactor;` : ''}}
            ${{hasToonEdge ? `
            // Toon edge darkening
            if (uToonEdge > 0.0) {{
              float edgeFactor = 1.0 - smoothstep(0.0, uToonEdge * 0.3, max(dot(normal, toonViewDir), 0.0));
              gl_FragColor.rgb = mix(gl_FragColor.rgb, uToonEdgeColor, edgeFactor * uToonEdge);
            }}` : ''}}`
          );
        }};
        }}
        
        // Load diffuse texture
        if (matData.diffuse) {{
          totalTexturesCount++;
          const texInfo = matData.diffuse;
          const texPath = typeof texInfo === 'string' ? texInfo : texInfo.path;
          const wrapS = typeof texInfo === 'object' ? (texInfo.wrapS || 0) : 0;
          const wrapT = typeof texInfo === 'object' ? (texInfo.wrapT || 0) : 0;
          
          loadTexture(texPath, wrapS, wrapT, texture => {{
            const mesh = meshes.find(m => m.userData.meshName === meshName);
            if (mesh) {{
              mesh.material.map = texture;
              mesh.userData.originalMap = texture;
              mesh.material.needsUpdate = true;
            }}
          }});
        }} else {{
          mat.color.setHex(0x808080);
        }}
        
        // Load normal map
        if (matData.normal) {{
          totalTexturesCount++;
          const texInfo = matData.normal;
          const texPath = typeof texInfo === 'string' ? texInfo : texInfo.path;
          const wrapS = typeof texInfo === 'object' ? (texInfo.wrapS || 0) : 0;
          const wrapT = typeof texInfo === 'object' ? (texInfo.wrapT || 0) : 0;
          loadTexture(texPath, wrapS, wrapT, texture => {{
            const mesh = meshes.find(m => m.userData.meshName === meshName);
            if (mesh) {{
              mesh.material.normalMap = texture;
              mesh.userData.originalNormalMap = texture;
              mesh.material.needsUpdate = true;
            }}
          }});
        }}
        
        const effects = [];
        if (hasRim) effects.push('rim');
        if (hasToonEdge) effects.push('edge');
        if (hasShadowColors) effects.push('shadow');
        if (hasSpecular) effects.push('spec');
        if (hasEmissive && sp.emissive_g > 0) effects.push('emissive');
        debug('Toon material:', materialName, 'type:', shaderType,
              hasFxo ? '(FXO)' : '(fallback)',
              'effects:', effects.join('+') || 'none');
        
        return mat;
      }}
      
      // Water shader — semi-transparent with color, fresnel, animated waves
      if (shaderType === 'water') {{
        const waterCol = sp.waterColor_g || [0.1, 0.3, 0.5];
        const waterIntensity = sp.waterColorIntensity_g != null ? sp.waterColorIntensity_g : 0.8;
        const fresnelVal = sp.fresnel_g != null ? sp.fresnel_g : 1.5;
        const specCol = sp.specularColor_g || [1, 1, 1];
        const specGloss = sp.specularGlossiness_g || 100;
        const waveFreq = sp.waveFreq_g || 10.0;
        const waveScale = sp.waveScale_g || 0.05;
        const waveVel = sp.waveVelocity_g || [0, -0.005];

        const waterMat = new THREE.MeshPhongMaterial({{
          color: new THREE.Color(waterCol[0] * waterIntensity, waterCol[1] * waterIntensity, waterCol[2] * waterIntensity),
          specular: new THREE.Color(specCol[0], specCol[1], specCol[2]),
          shininess: specGloss,
          transparent: true,
          opacity: 0.7,
          side: THREE.DoubleSide,
          skinning: true,
        }});
        waterMat.userData.shaderType = 'water';
        waterMat.userData.isWaterMaterial = true;

        // Inject fresnel transparency + wave vertex animation via onBeforeCompile
        waterMat.onBeforeCompile = function(shader) {{
          shader.uniforms.uTime = {{ value: 0 }};
          shader.uniforms.uFresnelPower = {{ value: fresnelVal }};
          shader.uniforms.uWaveFreq = {{ value: waveFreq }};
          shader.uniforms.uWaveScale = {{ value: waveScale }};
          shader.uniforms.uWaveVelocity = {{ value: new THREE.Vector2(waveVel[0], waveVel[1]) }};

          // Vertex shader: wave displacement
          shader.vertexShader = shader.vertexShader.replace(
            'void main() {{',
            `uniform float uTime;
            uniform float uWaveFreq;
            uniform float uWaveScale;
            uniform vec2 uWaveVelocity;
            void main() {{`
          );
          shader.vertexShader = shader.vertexShader.replace(
            '#include <begin_vertex>',
            `#include <begin_vertex>
            float wt = uTime * 0.5;
            float wave1 = sin(position.x * uWaveFreq + wt + uWaveVelocity.x * uTime * 100.0) * uWaveScale;
            float wave2 = sin(position.z * uWaveFreq * 0.7 + wt * 1.3 + uWaveVelocity.y * uTime * 100.0) * uWaveScale * 0.7;
            transformed.y += wave1 + wave2;`
          );

          // Fragment shader: fresnel-based alpha
          shader.fragmentShader = shader.fragmentShader.replace(
            'uniform float opacity;',
            `uniform float opacity;
            uniform float uFresnelPower;`
          );
          shader.fragmentShader = shader.fragmentShader.replace(
            '#include <output_fragment>',
            `#include <output_fragment>
            vec3 waterViewDir = normalize(vViewPosition);
            float fresnelDot = max(dot(normal, waterViewDir), 0.0);
            float fresnelFactor = pow(1.0 - fresnelDot, uFresnelPower);
            gl_FragColor.a = mix(0.3, 0.85, fresnelFactor);
            // Specular highlight boost at grazing angles
            gl_FragColor.rgb += vec3(fresnelFactor * 0.15);`
          );

          waterMat.userData._shader = shader;
        }};

        // Load detail texture (slot 1) as normal map for wave ripples
        if (matData.detail) {{
          totalTexturesCount++;
          const texInfo = matData.detail;
          const texPath = typeof texInfo === 'string' ? texInfo : texInfo.path;
          const wrapS = typeof texInfo === 'object' ? (texInfo.wrapS || 0) : 0;
          const wrapT = typeof texInfo === 'object' ? (texInfo.wrapT || 0) : 0;
          loadTexture(texPath, wrapS, wrapT, texture => {{
            texture.wrapS = THREE.RepeatWrapping;
            texture.wrapT = THREE.RepeatWrapping;
            const mesh = meshes.find(m => m.userData.meshName === meshName);
            if (mesh) {{
              mesh.material.normalMap = texture;
              mesh.material.normalScale = new THREE.Vector2(0.5, 0.5);
              mesh.userData.originalNormalMap = texture;
              mesh.material.needsUpdate = true;
            }}
          }});
        }}

        shaderStats.standard++;
        if (shaderType) shaderStats.types[shaderType] = (shaderStats.types[shaderType] || 0) + 1;
        debug('Water material:', materialName, 'color:', waterCol, 'fresnel:', fresnelVal, 'waveFreq:', waveFreq);
        return waterMat;
      }}

      // Fallback: standard material for non-character shaders
      const matParams = {{ roughness: 0.7, metalness: 0.2, side: THREE.DoubleSide, skinning: true }};

      if (matData.diffuse) {{
        totalTexturesCount++;
        const texInfo = matData.diffuse;
        const texPath = typeof texInfo === 'string' ? texInfo : texInfo.path;
        const wrapS = typeof texInfo === 'object' ? (texInfo.wrapS || 0) : 0;
        const wrapT = typeof texInfo === 'object' ? (texInfo.wrapT || 0) : 0;
        
        loadTexture(texPath, wrapS, wrapT, texture => {{
          const mesh = meshes.find(m => m.userData.meshName === meshName);
          if (mesh) {{
            mesh.material.map = texture;
            mesh.userData.originalMap = texture;
            mesh.material.needsUpdate = true;
          }}
        }});
      }} else {{
        matParams.color = 0x808080;
      }}

      if (matData.normal) {{
        totalTexturesCount++;
        const texInfo = matData.normal;
        const texPath = typeof texInfo === 'string' ? texInfo : texInfo.path;
        const wrapS = typeof texInfo === 'object' ? (texInfo.wrapS || 0) : 0;
        const wrapT = typeof texInfo === 'object' ? (texInfo.wrapT || 0) : 0;
        
        loadTexture(texPath, wrapS, wrapT, texture => {{
          const mesh = meshes.find(m => m.userData.meshName === meshName);
          if (mesh) {{
            mesh.material.normalMap = texture;
            mesh.userData.originalNormalMap = texture;
            mesh.material.needsUpdate = true;
          }}
        }});
      }}

      // Apply shader params to standard material too
      // Note: emissive_g in non-chr shaders (e.g. 'monster') is NOT PBR emission —
      // it's a shader-specific brightness/crystal parameter. Store for optional toggle.
      let stdEmissiveGlow = 0;
      if (sp.emissive_g && sp.emissive_g > 0) {{
        stdEmissiveGlow = Math.min(sp.emissive_g * 0.2, 0.5);
      }}
      if (sp.Switch_AlphaTest === 1) {{
        matParams.alphaTest = sp.alphaTestThreshold_g || 0.5;
        matParams.transparent = true;
      }}

      shaderStats.standard++;
      if (shaderType) shaderStats.types[shaderType] = (shaderStats.types[shaderType] || 0) + 1;
      const stdMat = new THREE.MeshStandardMaterial(matParams);
      if (stdEmissiveGlow > 0) stdMat.userData.emissiveGlow = stdEmissiveGlow;
      return stdMat;
    }}

    let loadingStartTime = Date.now();
    let loadingTimerInterval = setInterval(() => {{
      const el = document.getElementById('loading-detail');
      if (el) {{
        const s = ((Date.now() - loadingStartTime) / 1000).toFixed(0);
        el.textContent = s + 's elapsed';
      }}
    }}, 500);

    function updateLoadingProgress(step, detail, percent) {{
      const stepEl = document.getElementById('loading-step');
      const detailEl = document.getElementById('loading-detail');
      const progressEl = document.getElementById('loading-progress');
      if (stepEl) stepEl.textContent = step;
      // Keep elapsed timer in detail but append extra info
      const s = ((Date.now() - loadingStartTime) / 1000).toFixed(0);
      if (detailEl) detailEl.textContent = (detail ? detail + ' · ' : '') + s + 's elapsed';
      if (progressEl) progressEl.style.width = percent + '%';
    }}

    function hideLoadingOverlay() {{
      if (loadingTimerInterval) {{ clearInterval(loadingTimerInterval); loadingTimerInterval = null; }}
      const overlay = document.getElementById('loading-overlay');
      if (overlay) overlay.classList.add('hidden');
    }}

    function init() {{
      updateLoadingProgress('Setting up renderer...', '', 5);

      scene = new THREE.Scene();
      scene.background = new THREE.Color(CONFIG.INITIAL_BACKGROUND);

      camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
      camera.position.set(0, 2, 5);

      renderer = new THREE.WebGLRenderer({{ antialias: true, preserveDrawingBuffer: true }});
      renderer.setSize(window.innerWidth, window.innerHeight);
      document.getElementById('container').appendChild(renderer.domElement);

      ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
      scene.add(ambientLight);
      
      dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
      dirLight1.position.set(5, 10, 7);
      scene.add(dirLight1);
      
      dirLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
      dirLight2.position.set(-5, 5, -7);
      scene.add(dirLight2);

      controls = new OrbitControls(camera, renderer.domElement);

      window.addEventListener('resize', onWindowResize);

      // Chain loading steps with yields to allow UI updates
      setTimeout(() => {{
        updateLoadingProgress('Loading skeleton...', skeletonData ? skeletonData.length + ' bones' : '', 15);
        setTimeout(() => {{
          loadSkeleton();
          updateLoadingProgress('Loading meshes...', data.length + ' meshes', 30);
          setTimeout(() => {{
            loadMeshes();
            populateMeshList();
            const animCount = animationsData ? animationsData.length : 0;
            if (animCount > 0) {{
              updateLoadingProgress('Building animations...', '0 / ' + animCount, 35);
              buildAnimationClipsAsync((done, total) => {{
                const pct = 35 + Math.round((done / total) * 50);
                updateLoadingProgress('Building animations...', done + ' / ' + total, pct);
              }}).then(() => {{
                finishLoading();
              }});
            }} else {{
              finishLoading();
            }}
          }}, 0);
        }}, 0);
      }}, 0);
    }}

    function finishLoading() {{
      updateStats();
      updateTextureStatus();
      updateLoadingProgress('Finalizing...', '', 95);
      setTimeout(() => {{
        document.getElementById('controls-toggle').classList.add('visible');
        document.getElementById('overlay-toggle').classList.add('visible');
        updateLoadingProgress('Ready!', '', 100);
        setTimeout(() => {{
          hideLoadingOverlay();
          animate();
        }}, 200);
      }}, 0);
    }}

    function loadMeshes() {{
      const colors = [0xff6b6b, 0x4ecdc4, 0xffe66d, 0x95e1d3, 0xf38181];
      
      data.forEach((meshData, idx) => {{
        const geometry = new THREE.BufferGeometry();
        const verts = new Float32Array(meshData.vertices);
        const norms = new Float32Array(meshData.normals);
        const indices = new Uint32Array(meshData.indices);

        geometry.setAttribute('position', new THREE.BufferAttribute(verts, 3));
        geometry.setAttribute('normal', new THREE.BufferAttribute(norms, 3));
        // Store original normals for toggle
        geometry.userData = geometry.userData || {{}};
        geometry.userData.originalNormals = new Float32Array(norms);
        
        if (meshData.uvs) {{
          const uvs = new Float32Array(meshData.uvs);
          geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
        }}
        
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        
        // Tangents: use MDL data or compute fallback (needs index set first)
        if (meshData.tangents) {{
          const tangents = new Float32Array(meshData.tangents);
          geometry.setAttribute('tangent', new THREE.BufferAttribute(tangents, 4));
          tangentStats.mdl++;
        }} else if (meshData.uvs) {{
          computeTangents(geometry);
          tangentStats.computed++;
        }} else {{
          tangentStats.none++;
        }}
        
        // Add skinning data if available
        const hasSkinning = meshData.skinWeights && meshData.skinIndices;
        if (hasSkinning) {{
          debug('Mesh has skinning data:', meshData.name);
          debug('  Weights:', meshData.skinWeights.length, 'values');
          debug('  Indices:', meshData.skinIndices.length, 'values');
          debug('  Sample weights:', meshData.skinWeights.slice(0, 4));
          debug('  Sample indices:', meshData.skinIndices.slice(0, 4));
          
          const skinWeights = new Float32Array(meshData.skinWeights);
          const skinIndices = new Uint16Array(meshData.skinIndices);
          
          // CRITICAL: Validate skin indices are within bone range
          if (DEBUG && skeleton) {{
            const maxBoneIndex = skeleton.bones.length - 1;
            const indices = Array.from(skinIndices);
            const uniqueIndices = [...new Set(indices)].sort((a,b) => a-b);
            const outOfRange = indices.filter(idx => idx > maxBoneIndex);
            
            debug('Skin indices stats:');
            debug('  Total skeleton bones:', skeleton.bones.length);
            debug('  Max valid index:', maxBoneIndex);
            debug('  Unique indices used:', uniqueIndices.length, '/', skeleton.bones.length);
            debug('  Index range:', Math.min(...indices), '-', Math.max(...indices));
            if (outOfRange.length > 0) {{
              if (DEBUG) console.warn('⚠️ INVALID BONE INDICES:', outOfRange.length, 'indices > ', maxBoneIndex);
              if (DEBUG) console.warn('  Sample invalid indices:', outOfRange.slice(0, 10));
            }} else {{
              debug('  ✅ All bone indices valid!');
            }}
          }}
          
          // Normalize weights to ensure they sum to 1.0
          const numVertices = skinWeights.length / 4;
          for (let i = 0; i < numVertices; i++) {{
            const idx = i * 4;
            let sum = skinWeights[idx] + skinWeights[idx+1] + skinWeights[idx+2] + skinWeights[idx+3];
            if (sum > 0.0001) {{
              skinWeights[idx] /= sum;
              skinWeights[idx+1] /= sum;
              skinWeights[idx+2] /= sum;
              skinWeights[idx+3] /= sum;
            }}
          }}
          
          geometry.setAttribute('skinWeight', new THREE.BufferAttribute(skinWeights, 4));
          geometry.setAttribute('skinIndex', new THREE.BufferAttribute(skinIndices, 4));
        }} else {{
          debug('Mesh has NO skinning data:', meshData.name);
        }}
        
        geometry.computeBoundingSphere();

        const material = createMaterial(meshData.material, meshData.name);
        
        // Create SkinnedMesh if has skinning data, otherwise regular Mesh
        let mesh;
        if (hasSkinning && skeleton) {{
          mesh = new THREE.SkinnedMesh(geometry, material);
          mesh.frustumCulled = false;  // Important for skeletal animation
          
          // CRITICAL: Pass explicit identity matrix as bindMatrix!
          // This prevents skeleton.calculateInverses() from being called,
          // preserving our correct MDL inverse bind matrices.
          // Identity is correct because mesh vertices are already in world space.
          const identityBindMatrix = new THREE.Matrix4();  // identity
          mesh.bind(skeleton, identityBindMatrix);
          
          // Verify skinning is properly set up
          debug('✅ SkinnedMesh bound:', meshData.name,
                '| isSkinnedMesh:', mesh.isSkinnedMesh,
                '| skeleton bones:', mesh.skeleton ? mesh.skeleton.bones.length : 'NONE',
                '| boneTexture:', mesh.skeleton && mesh.skeleton.boneTexture ? 'YES' : 'NO',
                '| skinIndex type:', geometry.attributes.skinIndex ? geometry.attributes.skinIndex.array.constructor.name : 'MISSING');
          
          debug('Created SkinnedMesh:', meshData.name,
                'bones:', skeleton.bones.length,
                'bindMode:', mesh.bindMode);
        }} else {{
          mesh = new THREE.Mesh(geometry, material);
          if (hasSkinning && !skeleton) {{
            if (DEBUG) console.warn('Mesh has skinning but NO skeleton!', meshData.name);
          }}
        }}
        
        mesh.userData.meshName = meshData.name;
        mesh.userData.materialName = meshData.material;
        mesh.userData.originalColor = colors[idx % colors.length];
        mesh.userData.hasTexture = !!meshData.material && !!materials[meshData.material];
        mesh.userData.hasSkinning = hasSkinning;
        // Store FXO material for toggle
        if (material.userData && material.userData.isToonMaterial) {{
          mesh.userData.fxoMaterial = material;
        }}
        
        mesh.userData.isShadowMesh = !!meshData.is_shadow;
        mesh.userData.meshGroup = meshData.mesh_group || '';
        if (CONFIG.AUTO_HIDE_SHADOW && mesh.userData.isShadowMesh) {{
          mesh.visible = false;
        }}
        
        scene.add(mesh);
        meshes.push(mesh);
      }});

      const box = new THREE.Box3();
      meshes.forEach(m => box.expandByObject(m));
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      
      // Store model center Y for skeleton offset
      modelCenterY = center.y;
      debug('Model bounding box center Y:', modelCenterY);
      debug('Model bounding box:', 'min:', box.min.toArray(), 'max:', box.max.toArray());
      
      // DEBUG: Check first mesh world position
      if (meshes.length > 0) {{
        const firstMesh = meshes[0];
        debug('First mesh world position:', firstMesh.position.toArray());
        debug('First mesh world matrix translation:', firstMesh.matrixWorld.elements.slice(12, 15));
      }}
      
      // DEBUG MARKERS: Add visible spheres at key positions
      if (DEBUG) {{
        // Red sphere at world origin [0,0,0]
        const originMarker = new THREE.Mesh(
          new THREE.SphereGeometry(0.05),
          new THREE.MeshBasicMaterial({{ color: 0xff0000 }})
        );
        originMarker.position.set(0, 0, 0);
        scene.add(originMarker);
        
        // Blue sphere at model bounding box center
        const centerMarker = new THREE.Mesh(
          new THREE.SphereGeometry(0.05),
          new THREE.MeshBasicMaterial({{ color: 0x0000ff }})
        );
        centerMarker.position.copy(center);
        scene.add(centerMarker);
        
        debug('DEBUG MARKERS: Red sphere at world [0,0,0], Blue sphere at model center', center.toArray());
      }}
      
      const maxDim = Math.max(size.x, size.y, size.z);
      const fov = camera.fov * (Math.PI / 180);
      
      // Adapt camera clipping planes to model scale
      camera.near = Math.max(0.01, maxDim * 0.0001);
      camera.far = maxDim * 20;
      camera.updateProjectionMatrix();
      
      // Adapt controls limits to model scale
      controls.minDistance = maxDim * 0.01;
      controls.maxDistance = maxDim * 10;
      
      const aspect = camera.aspect;
      const vFOV = fov;
      const hFOV = 2 * Math.atan(Math.tan(vFOV / 2) * aspect);
      
      const distanceV = maxDim / (2 * Math.tan(vFOV / 2));
      const distanceH = maxDim / (2 * Math.tan(hFOV / 2));
      const cameraDistance = Math.max(distanceV, distanceH);
      
      const dist = cameraDistance * CONFIG.CAMERA_ZOOM;
      
      const direction = new THREE.Vector3(0.5, 0.5, 1).normalize();
      camera.position.copy(center).add(direction.multiplyScalar(dist));
      camera.lookAt(center);
      
      controls.target.copy(center);
      
      const offset = camera.position.clone().sub(center);
      controls.spherical.setFromVector3(offset);
      controls.panOffset.set(0, 0, 0);
      
      // Scale directional lights relative to model size
      scene.children.forEach(c => {{
        if (c.isDirectionalLight) {{
          c.position.normalize().multiplyScalar(maxDim * 2);
        }}
      }});
      
      controls.update();
    }}

    // Focus camera on all visible meshes (or all meshes if none visible)
    function fitToView() {{
      const box = new THREE.Box3();
      const visible = meshes.filter(m => m.visible);
      (visible.length > 0 ? visible : meshes).forEach(m => box.expandByObject(m));
      if (box.isEmpty()) return;
      
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      
      // Adapt camera clipping planes
      camera.near = Math.max(0.01, maxDim * 0.0001);
      camera.far = maxDim * 20;
      camera.updateProjectionMatrix();
      
      // Adapt controls limits
      controls.minDistance = maxDim * 0.01;
      controls.maxDistance = maxDim * 10;
      
      // Calculate framing distance
      const fov = camera.fov * (Math.PI / 180);
      const aspect = camera.aspect;
      const hFOV = 2 * Math.atan(Math.tan(fov / 2) * aspect);
      const distV = maxDim / (2 * Math.tan(fov / 2));
      const distH = maxDim / (2 * Math.tan(hFOV / 2));
      const dist = Math.max(distV, distH) * CONFIG.CAMERA_ZOOM;
      
      // Position camera so it faces the center from current viewing direction
      const forward = new THREE.Vector3();
      camera.getWorldDirection(forward);
      camera.position.copy(center).addScaledVector(forward, -dist);
      camera.lookAt(center);
      
      controls.target.copy(center);
      const offset = camera.position.clone().sub(center);
      controls.spherical.setFromVector3(offset);
      controls.panOffset.set(0, 0, 0);
      controls.update();
      
      // Update FreeCam base speed for model scale
      freeCamBaseSpeed = maxDim * 0.02;
      
      // Update FreeCam angles if active
      if (freeCamMode) {{
        tpCamPhi = Math.acos(-forward.y);
        tpCamTheta = Math.atan2(-forward.x, -forward.z);
      }}
      
      // Scale directional lights relative to model
      scene.children.forEach(c => {{
        if (c.isDirectionalLight) {{
          c.position.normalize().multiplyScalar(maxDim * 2);
        }}
      }});
      
      debug('fitToView: maxDim=' + maxDim.toFixed(1) + ' dist=' + dist.toFixed(1) + ' center=', center.toArray());
    }}

    // Focus camera on a specific mesh by index
    let focusLockUntil = 0;  // timestamp to temporarily disable TP camera override
    let xrayHighlight = true;  // whether blink shows through other meshes
    let hideShadowMeshes = true;  // hide shadow meshes (default ON)
    let fxoShadersEnabled = true;  // FXO toon shaders active
    let recomputeNormalsEnabled = RECOMPUTE_NORMALS;  // recomputed normals active
    
    function focusMesh(idx, withZoom) {{
      try {{
      if (idx < 0 || idx >= meshes.length) return;
      const mesh = meshes[idx];
      
      // Make mesh visible if hidden
      if (!mesh.visible) {{
        mesh.visible = true;
        if (wireframeOverlayMode && mesh.userData.wireframeOverlay) {{
          mesh.userData.wireframeOverlay.visible = true;
        }}
        const cb = document.getElementById(`mesh-${{idx}}`);
        if (cb) cb.checked = true;
        updateStats();
      }}
      
      const box = new THREE.Box3().expandByObject(mesh);
      if (box.isEmpty()) return;
      
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      
      // In controller mode (third-person) or no-zoom mode — only blink, no camera move
      const skipZoom = !withZoom || (gamepadEnabled && !freeCamMode);
      
      if (!skipZoom) {{
        // Calculate framing distance
        const fov = camera.fov * (Math.PI / 180);
        const aspect = camera.aspect;
        const hFOV = 2 * Math.atan(Math.tan(fov / 2) * aspect);
        const distV = maxDim / (2 * Math.tan(fov / 2));
        const distH = maxDim / (2 * Math.tan(hFOV / 2));
        const dist = Math.max(distV, distH, 0.1) * CONFIG.CAMERA_ZOOM;
        
        // Adapt camera clipping planes and controls limits
        camera.near = Math.max(0.001, maxDim * 0.0001);
        camera.far = Math.max(camera.far, maxDim * 40);
        camera.updateProjectionMatrix();
        controls.maxDistance = Math.max(controls.maxDistance, dist * 5);
        
        // Position camera: place it looking at mesh center from a standard angle
        const direction = new THREE.Vector3(0.5, 0.5, 1).normalize();
        camera.position.copy(center).add(direction.multiplyScalar(dist));
        camera.lookAt(center);
        
        controls.target.copy(center);
        const offset = camera.position.clone().sub(center);
        controls.spherical.setFromVector3(offset);
        controls.panOffset.set(0, 0, 0);
        controls.update();
        
        if (freeCamMode) {{
          const lookDir = new THREE.Vector3().subVectors(center, camera.position).normalize();
          tpCamPhi = Math.acos(-lookDir.y);
          tpCamTheta = Math.atan2(-lookDir.x, -lookDir.z);
        }}
        
        // Prevent TP camera from overriding for 500ms
        focusLockUntil = Date.now() + 500;
      }}
      
      // Always show info toast and blink
      const triCount = mesh.geometry && mesh.geometry.index ? mesh.geometry.index.count / 3 : '?';
      showToast('🎯 ' + mesh.userData.meshName + '\\n' + triCount + ' tris · size ' + maxDim.toFixed(1), 5000);
      
      // Blink mesh red for ~5 seconds
      blinkMesh(mesh);
      }} catch(err) {{
        console.error('focusMesh error:', err);
      }}
    }}
    
    let blinkTimers = new Map();  // mesh -> timer id
    function blinkMesh(mesh) {{
      // Cancel previous blink on this mesh
      if (blinkTimers.has(mesh)) {{
        clearInterval(blinkTimers.get(mesh).interval);
        clearTimeout(blinkTimers.get(mesh).timeout);
        // Restore original materials
        const prev = blinkTimers.get(mesh);
        mesh.material = prev.origMaterial;
        mesh.renderOrder = 0;
        blinkTimers.delete(mesh);
      }}
      
      // Store original material
      const origMaterial = mesh.material;
      
      // Create highlight wireframe material (bright green, always visible)
      const wireMat = new THREE.MeshBasicMaterial({{
        color: 0x00ff88,
        wireframe: true,
        depthTest: !xrayHighlight
      }});
      
      let blinkOn = true;
      
      const interval = setInterval(() => {{
        if (blinkOn) {{
          mesh.material = wireMat;
          wireMat.depthTest = !xrayHighlight;
          if (xrayHighlight) mesh.renderOrder = 9999;
        }} else {{
          mesh.material = origMaterial;
          mesh.renderOrder = 0;
        }}
        blinkOn = !blinkOn;
      }}, 300);
      
      const timeout = setTimeout(() => {{
        clearInterval(interval);
        mesh.material = origMaterial;
        mesh.renderOrder = 0;
        wireMat.dispose();
        blinkTimers.delete(mesh);
      }}, 5000);
      
      blinkTimers.set(mesh, {{ interval, timeout, origMaterial }});
    }}

    function loadSkeleton() {{
      if (!skeletonData || !Array.isArray(skeletonData) || skeletonData.length === 0) {{
        document.getElementById('skeleton-info').textContent = 'No skeleton data';
        return;
      }}

      debug('=== LOADING SKELETON ===');
      debug('Bones:', skeletonData.length);
      
      document.getElementById('skeleton-info').textContent = `${{skeletonData.length}} bones loaded`;
      document.getElementById('skeleton-available').style.display = 'block';
      document.getElementById('skeleton-unavailable').style.display = 'none';
      if (animationsData && animationsData.length > 0) {{
        document.getElementById('animations-available').style.display = 'block';
        document.getElementById('animations-unavailable').style.display = 'none';
      }}
      // Show dynamic bones controls if MI data has physics chains
      if (modelInfo && modelInfo.DynamicBone && modelInfo.DynamicBone.length > 0) {{
        document.getElementById('dynBonesSection').style.display = 'block';
      }}

      // STEP 1: Create all bones with LOCAL transforms
      bones = skeletonData.map((boneData, idx) => {{
        const bone = new THREE.Bone();
        bone.name = boneData.name || `Bone_${{idx}}`;
        
        // Set LOCAL position (relative to parent)
        bone.position.set(boneData.pos_xyz[0], boneData.pos_xyz[1], boneData.pos_xyz[2]);
        
        // Set LOCAL rotation from pre-computed quaternion (eArmada8 rpy2quat formula)
        if (boneData.quat_xyzw) {{
          bone.quaternion.set(boneData.quat_xyzw[0], boneData.quat_xyzw[1], boneData.quat_xyzw[2], boneData.quat_xyzw[3]);
        }} else {{
          const euler = new THREE.Euler(
            boneData.rotation_euler_rpy[0],
            boneData.rotation_euler_rpy[1],
            boneData.rotation_euler_rpy[2],
            'ZYX'
          );
          bone.quaternion.setFromEuler(euler);
        }}
        
        // Set LOCAL scale
        bone.scale.set(boneData.scale[0], boneData.scale[1], boneData.scale[2]);
        
        bone.userData.boneData = boneData;
        bone.userData.boneIndex = idx;
        return bone;
      }});
      
      // STEP 2: Build parent-child hierarchy
      skeletonData.forEach((boneData, idx) => {{
        boneData.children.forEach(childIdx => {{
          if (childIdx < bones.length) {{
            bones[idx].add(bones[childIdx]);
          }}
        }});
      }});
      
      // STEP 3: Add root bone directly to scene and compute all world matrices
      scene.add(bones[0]);
      bones[0].updateMatrixWorld(true);
      
      // STEP 4: Create skeleton - let Three.js calculate inverse bind matrices
      // from bone.matrixWorld (guaranteed to match since Three.js computed them)
      // Passing no boneInverses triggers calculateInverses() in constructor
      skeleton = new THREE.Skeleton(bones);
      // Ensure bone texture is created (required by some THREE.js versions)
      if (typeof skeleton.computeBoneTexture === 'function') {{
        skeleton.computeBoneTexture();
      }}
      debug('Skeleton created with', skeleton.bones.length, 'bones');
      debug('Inverse bind matrices: calculated from Three.js bone world matrices');
      
      // STEP 5: Diagnostic - verify bind pose + compare with MDL matrices
      if (DEBUG) {{
        skeleton.update();
        debug('=== BIND POSE VERIFICATION ===');
        let allIdentity = true;
        for (let i = 0; i < Math.min(10, bones.length); i++) {{
          const bm = new THREE.Matrix4();
          bm.multiplyMatrices(bones[i].matrixWorld, skeleton.boneInverses[i]);
          const t = [bm.elements[12], bm.elements[13], bm.elements[14]];
          const isId = Math.abs(bm.elements[0] - 1) < 0.001 && 
                       Math.abs(bm.elements[5] - 1) < 0.001 && 
                       Math.abs(bm.elements[10] - 1) < 0.001 &&
                       Math.abs(t[0]) < 0.001 && Math.abs(t[1]) < 0.001 && Math.abs(t[2]) < 0.001;
          if (!isId) allIdentity = false;
          debug('  Bone[' + i + '] ' + bones[i].name + ': ~identity? ' + isId);
        }}
        debug('All first 10 bones identity at bind pose:', allIdentity);
        
        // Compare Three.js world matrices with MDL bind matrices
        if (bindMatricesData) {{
          debug('=== THREE.JS vs MDL MATRIX COMPARISON ===');
          ['Root', 'Hips', 'Head', 'LeftArm', 'LeftUpLeg'].forEach(name => {{
            const bone = bones.find(b => b.name === name);
            const mdl = bindMatricesData[name];
            if (bone && mdl) {{
              const wm = bone.matrixWorld.elements;
              // Three.js col-major: translation at [12,13,14]
              // MDL row-major: translation at row3 = [mat[3][0], mat[3][1], mat[3][2]]
              const tjsPos = [wm[12], wm[13], wm[14]];
              const mdlPos = [mdl[3][0], mdl[3][1], mdl[3][2]];
              const posDiff = Math.sqrt(
                Math.pow(tjsPos[0]-mdlPos[0],2) + Math.pow(tjsPos[1]-mdlPos[1],2) + Math.pow(tjsPos[2]-mdlPos[2],2));
              // Check if Three.js col0 matches MDL row0 (would mean MDL^T == ThreeJS)
              const col0Match = Math.abs(wm[0]-mdl[0][0]) < 0.01 && Math.abs(wm[1]-mdl[0][1]) < 0.01;
              // Check if Three.js col0 matches MDL col0 (would mean MDL == ThreeJS, no transpose needed)
              const noTrMatch = Math.abs(wm[0]-mdl[0][0]) < 0.01 && Math.abs(wm[1]-mdl[1][0]) < 0.01;
              debug('  ' + name + ': posDiff=' + posDiff.toFixed(6) + 
                ' col0=MDLrow0?' + col0Match + ' col0=MDLcol0?' + noTrMatch +
                ' TJS=[' + tjsPos.map(v=>v.toFixed(4)) + '] MDL=[' + mdlPos.map(v=>v.toFixed(4)) + ']');
            }}
          }});
        }}
      }}

      // STEP 7: Create skeleton visualization groups
      skeletonGroup = new THREE.Group();
      skeletonGroup.name = 'skeleton_lines';
      jointsGroup = new THREE.Group();
      jointsGroup.name = 'skeleton_joints';
      scene.add(skeletonGroup);
      scene.add(jointsGroup);
      skeletonGroup.visible = showSkeleton;
      jointsGroup.visible = showJoints;

      // Precompute parent index map for skeleton visualization
      window._boneParentMap = {{}};
      skeletonData.forEach((bd, idx) => {{
        if (bd.children) {{
          bd.children.forEach(childIdx => {{
            window._boneParentMap[childIdx] = idx;
          }});
        }}
      }});

      debug('=== SKELETON READY ===');
      
      // Animation clips are built separately in async init flow
    }}

    function updateSkeletonVis() {{
      if (!skeletonGroup || !jointsGroup) return;

      while (skeletonGroup.children.length) skeletonGroup.remove(skeletonGroup.children[0]);
      while (jointsGroup.children.length) jointsGroup.remove(jointsGroup.children[0]);

      const lineMat = new THREE.LineBasicMaterial({{ color: 0x00ff88, depthTest: false }});
      const jointGeo = new THREE.SphereGeometry(0.005, 6, 6);
      const jointMat = new THREE.MeshBasicMaterial({{ color: 0xff4444, depthTest: false }});

      // When in third-person mode, skeleton/joints groups are children of characterGroup
      // so we need local positions, not world positions
      const needLocal = !!characterGroup;
      const invMatrix = needLocal ? new THREE.Matrix4().copy(characterGroup.matrixWorld).invert() : null;

      for (let i = 0; i < skeletonData.length; i++) {{
        const bd = skeletonData[i];
        const bone = bones[i];
        if (!bone) continue;
        if (bd.type !== 1) continue;

        const worldPos = new THREE.Vector3();
        bone.getWorldPosition(worldPos);
        const drawPos = needLocal ? worldPos.clone().applyMatrix4(invMatrix) : worldPos;

        // Joint sphere
        if (showJoints) {{
          const jm = new THREE.Mesh(jointGeo, jointMat);
          jm.position.copy(drawPos);
          jm.renderOrder = 999;
          jointsGroup.add(jm);
        }}

        // Line to parent
        const parentIdx = window._boneParentMap[i];
        if (parentIdx !== undefined && bones[parentIdx]) {{
          const parentPos = new THREE.Vector3();
          bones[parentIdx].getWorldPosition(parentPos);
          const parentDraw = needLocal ? parentPos.applyMatrix4(invMatrix) : parentPos;
          const geo = new THREE.BufferGeometry().setFromPoints([parentDraw, drawPos]);
          const line = new THREE.Line(geo, lineMat);
          line.renderOrder = 998;
          skeletonGroup.add(line);
        }}
      }}

      skeletonGroup.visible = showSkeleton;
      jointsGroup.visible = showJoints;

      if (showBoneNames) updateBoneLabels();
    }}

    function toggleSkeleton() {{
      showSkeleton = !showSkeleton;
      skeletonGroup.visible = showSkeleton;
      const sw = document.getElementById('swSkel'); if (sw) sw.checked = showSkeleton;
      if (showSkeleton) updateSkeletonVis();
    }}

    function toggleJoints() {{
      showJoints = !showJoints;
      jointsGroup.visible = showJoints;
      const sw = document.getElementById('swJoints'); if (sw) sw.checked = showJoints;
      updateSkeletonVis();
    }}

    function toggleBoneNames() {{
      showBoneNames = !showBoneNames;
      const sw = document.getElementById('swBoneNames'); if (sw) sw.checked = showBoneNames;
      let overlay = document.getElementById('bone-names-overlay');
      if (showBoneNames) {{
        if (!overlay) {{
          overlay = document.createElement('div');
          overlay.id = 'bone-names-overlay';
          overlay.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:10;';
          document.body.appendChild(overlay);
        }}
        updateBoneLabels();
      }} else if (overlay) {{
        overlay.innerHTML = '';
      }}
    }}

    function updateBoneLabels() {{
      const overlay = document.getElementById('bone-names-overlay');
      if (!overlay || !showBoneNames) return;
      overlay.innerHTML = '';
      const w = window.innerWidth, h = window.innerHeight;
      for (let i = 0; i < skeletonData.length; i++) {{
        const bd = skeletonData[i];
        if (bd.type !== 1) continue;
        const bone = bones[i];
        if (!bone) continue;
        const pos = new THREE.Vector3();
        bone.getWorldPosition(pos);
        pos.project(camera);
        if (pos.z > 1) continue;
        const x = (pos.x * 0.5 + 0.5) * w;
        const y = (-pos.y * 0.5 + 0.5) * h;
        const label = document.createElement('div');
        label.style.cssText = 'position:absolute;left:'+x+'px;top:'+y+'px;color:#0f8;font-size:9px;font-family:monospace;transform:translate(-50%,-100%);white-space:nowrap;text-shadow:0 0 3px #000;';
        label.textContent = bd.name;
        overlay.appendChild(label);
      }}
    }}

    function setMeshOpacity(val) {{
      const v = parseFloat(val);
      meshes.forEach(m => {{
        m.material.opacity = v;
        m.material.transparent = v < 1;
        m.material.needsUpdate = true;
      }});
    }}

    // =========================================================
    // ANIMATION SYSTEM - Real MDL animations
    // =========================================================
    let animationClips = {{}};
    let currentAnimName = null;

    function buildAnimationClipsAsync(onProgress) {{
      return new Promise((resolve) => {{
        if (!animationsData || !bones || bones.length === 0) {{ resolve(); return; }}
        
        const boneNameMap = {{}};
        bones.forEach(b => {{ boneNameMap[b.name] = b; }});
        
        const total = animationsData.length;
        let idx = 0;
        const batchSize = 5; // Process N animations per frame
        
        function processBatch() {{
          const end = Math.min(idx + batchSize, total);
          for (; idx < end; idx++) {{
            const anim = animationsData[idx];
            const tracks = [];
            
            anim.channels.forEach(ch => {{
              if (!boneNameMap[ch.bone]) return;
              
              const times = new Float32Array(ch.times);
              const values = new Float32Array(ch.values);
              
              let track;
              if (ch.type === 10) {{
                track = new THREE.QuaternionKeyframeTrack(ch.bone + '.quaternion', times, values);
              }} else if (ch.type === 9) {{
                track = new THREE.VectorKeyframeTrack(ch.bone + '.position', times, values);
              }} else if (ch.type === 11) {{
                track = new THREE.VectorKeyframeTrack(ch.bone + '.scale', times, values);
              }}
              if (track) tracks.push(track);
            }});
            
            if (tracks.length > 0) {{
              animationClips[anim.name] = new THREE.AnimationClip(anim.name, anim.duration, tracks);
            }}
          }}
          
          if (onProgress) onProgress(idx, total);
          
          if (idx < total) {{
            setTimeout(processBatch, 0);
          }} else {{
            debug('Built', Object.keys(animationClips).length, 'animation clips');
            populateAnimationList();
            resolve();
          }}
        }}
        
        processBatch();
      }});
    }}

    // Kept for compatibility but prefer async version
    function buildAnimationClips() {{
      if (!animationsData || !bones || bones.length === 0) return;
      
      const boneNameMap = {{}};
      bones.forEach(b => {{ boneNameMap[b.name] = b; }});
      
      animationsData.forEach(anim => {{
        const tracks = [];
        
        anim.channels.forEach(ch => {{
          // Only process bones that exist in our skeleton
          if (!boneNameMap[ch.bone]) return;
          
          const times = new Float32Array(ch.times);
          const values = new Float32Array(ch.values);
          
          let track;
          if (ch.type === 10) {{ // Rotation (quaternion xyzw)
            track = new THREE.QuaternionKeyframeTrack(
              ch.bone + '.quaternion', times, values
            );
          }} else if (ch.type === 9) {{ // Translation xyz
            track = new THREE.VectorKeyframeTrack(
              ch.bone + '.position', times, values
            );
          }} else if (ch.type === 11) {{ // Scale xyz
            track = new THREE.VectorKeyframeTrack(
              ch.bone + '.scale', times, values
            );
          }}
          
          if (track) tracks.push(track);
        }});
        
        if (tracks.length > 0) {{
          animationClips[anim.name] = new THREE.AnimationClip(anim.name, anim.duration, tracks);
          debug('Built clip:', anim.name, tracks.length, 'tracks,', anim.duration.toFixed(1) + 's');
        }}
      }});
      
      debug('Built', Object.keys(animationClips).length, 'animation clips');
    }}

    function populateAnimationList() {{
      const select = document.getElementById('animation-select');
      if (!select) return;
      
      const names = Object.keys(animationClips);
      if (names.length === 0) return;
      
      names.forEach(name => {{
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
      }});
    }}

    function playAnimation(animName) {{
      if (!bones || bones.length === 0) return;
      const clip = animationClips[animName];
      if (!clip) {{ DEBUG && console.warn('Clip not found:', animName); return; }}

      // Stop current
      if (animationMixer) {{
        animationMixer.stopAllAction();
      }}

      // Create mixer on root bone
      const target = bones[0];
      animationMixer = new THREE.AnimationMixer(target);
      // Apply current speed slider
      const speedSlider = document.getElementById('animSpeedSlider');
      if (speedSlider) animationMixer.timeScale = Math.pow(2, parseInt(speedSlider.value) / 50);
      currentAnimName = animName;

      currentAnimation = animationMixer.clipAction(clip);
      currentAnimation.play();
      
      // Update toggle button
      const btn = document.getElementById('btnAnimToggle');
      if (btn) {{ btn.textContent = '⏸️ Pause'; btn.className = 'btn-action active'; }}
      
      // Reset dynamic bone particles to new animated pose after one frame
      if (dynamicBonesEnabled) {{
        setTimeout(() => {{ resetDynamicBones(); }}, 50);
      }}
      
      debug('Playing:', animName, 'duration:', clip.duration.toFixed(2) + 's');
    }}

    // Smooth crossfade for third-person auto-animations (reuses same mixer)
    function playAnimationCrossfade(animName, fadeDuration) {{
      if (!bones || bones.length === 0) return;
      const clip = animationClips[animName];
      if (!clip) return;
      
      fadeDuration = fadeDuration || 0.3;
      
      // Create mixer if needed (first call)
      if (!animationMixer) {{
        const target = bones[0];
        animationMixer = new THREE.AnimationMixer(target);
        const speedSlider = document.getElementById('animSpeedSlider');
        if (speedSlider) animationMixer.timeScale = Math.pow(2, parseInt(speedSlider.value) / 50);
      }}
      
      const prevAction = currentAnimation;
      const newAction = animationMixer.clipAction(clip);
      
      newAction.reset();
      newAction.play();
      
      if (prevAction && prevAction !== newAction) {{
        // Crossfade: old action fades out, new fades in
        newAction.crossFadeFrom(prevAction, fadeDuration, true);
      }}
      
      currentAnimation = newAction;
      currentAnimName = animName;
      
      const btn = document.getElementById('btnAnimToggle');
      if (btn) {{ btn.textContent = '⏸️ Pause'; btn.className = 'btn-action active'; }}
    }}

    function updateAnimSpeed(val) {{
      const v = parseInt(val);
      // -100→0.25x, 0→1.0x, +100→4.0x (exponential for natural feel)
      const speed = Math.pow(2, v / 50);
      if (animationMixer) {{
        animationMixer.timeScale = speed;
      }}
      const label = document.getElementById('animSpeedLabel');
      if (label) label.textContent = speed.toFixed(1) + 'x';
    }}

    function toggleAnimPlayback() {{
      if (!currentAnimation) {{
        // Nothing playing - try to play selected
        const sel = document.getElementById('animation-select');
        if (sel && sel.value) playAnimation(sel.value);
        return;
      }}
      
      const btn = document.getElementById('btnAnimToggle');
      if (currentAnimation.paused) {{
        currentAnimation.paused = false;
        if (btn) {{ btn.textContent = '⏸️ Pause'; btn.className = 'btn-action active'; }}
      }} else {{
        currentAnimation.paused = true;
        if (btn) {{ btn.textContent = '▶️ Play'; btn.className = 'btn-action'; }}
      }}
    }}

    function stopAnimation() {{
      if (animationMixer) {{
        animationMixer.stopAllAction();
        animationMixer = null;
      }}
      currentAnimation = null;
      currentAnimName = null;

      // Reset dropdown and button
      const sel = document.getElementById('animation-select');
      if (sel) sel.value = '';
      const btn = document.getElementById('btnAnimToggle');
      if (btn) {{ btn.textContent = '⏹️ Stop'; btn.className = 'btn-action'; }}
      const slider = document.getElementById('animTimeline');
      if (slider) slider.value = 0;
      const label = document.getElementById('animTimeLabel');
      if (label) label.textContent = '0.00 / 0.00';

      // Reset to bind pose
      bones.forEach((bone, idx) => {{
        const boneData = skeletonData[idx];
        bone.position.set(boneData.pos_xyz[0], boneData.pos_xyz[1], boneData.pos_xyz[2]);
        
        if (boneData.quat_xyzw) {{
          bone.quaternion.set(boneData.quat_xyzw[0], boneData.quat_xyzw[1], boneData.quat_xyzw[2], boneData.quat_xyzw[3]);
        }} else {{
          const euler = new THREE.Euler(
            boneData.rotation_euler_rpy[0],
            boneData.rotation_euler_rpy[1],
            boneData.rotation_euler_rpy[2],
            'ZYX'
          );
          bone.quaternion.setFromEuler(euler);
        }}
        
        if (boneData.scale) {{
          bone.scale.set(boneData.scale[0], boneData.scale[1], boneData.scale[2]);
        }}
      }});
      
      debug('Reset to bind pose');
      
      // Reset dynamic bones state to match bind pose
      // Without this, savedLocalQuat holds stale animation quats
      // which overwrites the bind pose on next frame, freezing physics
      if (dynamicBonesEnabled && dynChains.length > 0) {{
        if (bones.length > 0) bones[0].updateMatrixWorld(true);
        resetDynamicBones();
      }}
    }}

    function scrubAnimation(val) {{
      if (!currentAnimation || !animationMixer) return;
      const clip = currentAnimation.getClip();
      const t = parseFloat(val) * clip.duration;
      // Pause while scrubbing
      if (!currentAnimation.paused) {{
        currentAnimation.paused = true;
        const btn = document.getElementById('btnAnimToggle');
        if (btn) {{ btn.textContent = '▶️ Play'; btn.className = 'btn-action'; }}
      }}
      currentAnimation.time = t;
      animationMixer.update(0);  // Force pose update at new time
      const label = document.getElementById('animTimeLabel');
      if (label) label.textContent = t.toFixed(2) + ' / ' + clip.duration.toFixed(2);
    }}

    function updateTimeline() {{
      if (!currentAnimation || !animationMixer) return;
      const clip = currentAnimation.getClip();
      const t = currentAnimation.time % clip.duration;
      const slider = document.getElementById('animTimeline');
      const label = document.getElementById('animTimeLabel');
      if (slider && !currentAnimation.paused) {{
        slider.value = t / clip.duration;
      }}
      if (label) {{
        label.textContent = t.toFixed(2) + ' / ' + clip.duration.toFixed(2);
      }}
    }}

    // =========================================================
    // DYNAMIC BONE PHYSICS (Jiggle Physics / Spring Simulation)
    // =========================================================
    let dynamicBonesEnabled = false;
    let dynChains = [];
    let dynAccum = 0;
    let dynLastAnimTime = -1;
    const DYN_FIXED_DT = 1/60;
    let dynProcessedBones = new Set();
    let dynIntensityMult = 1.0;
    let dynCollisionsEnabled = true;  // Toggle for collision debugging

    function initDynamicBones() {{
      dynChains = [];
      if (!modelInfo || !modelInfo.DynamicBone || !bones || bones.length === 0) return;
      
      const boneMap = {{}};
      bones.forEach(b => {{ boneMap[b.name] = b; }});
      
      // Build skeleton parent name map from THREE.js hierarchy
      const skelParentName = {{}};
      bones.forEach(b => {{
        if (b.parent && b.parent.isBone) skelParentName[b.name] = b.parent.name;
      }});
      
      // Build colliders
      const colliders = [];
      if (modelInfo.DynamicBoneCollider) {{
        modelInfo.DynamicBoneCollider.forEach(cd => {{
          const attachBone = boneMap[cd.node];
          if (!attachBone) return;
          // Rotation offset (degrees → radians)
          const DEG = Math.PI / 180;
          const rotQ = new THREE.Quaternion().setFromEuler(
            new THREE.Euler(
              (cd.offset_rot_x || 0) * DEG,
              (cd.offset_rot_y || 0) * DEG,
              (cd.offset_rot_z || 0) * DEG
            )
          );
          colliders.push({{
            bone: attachBone, name: cd.name, type: cd.type,
            radius: cd.param0, height: cd.param1,
            param2: cd.param2 || 0, param3: cd.param3 || 0,
            offset: new THREE.Vector3(cd.offset_x, cd.offset_y, cd.offset_z),
            offsetRot: rotQ,
            needSpecific: cd.need_specific
          }});
        }});
      }}
      
      let chainCount = 0;
      modelInfo.DynamicBone.forEach((db, dbIdx) => {{
        if (!db.Joint || db.Joint.length < 2) return;
        
        const nodeNames = db.Joint.map(j => j.node);
        
        const jointBones = [];
        const jointParams = [];
        const parentIdx = [];
        let valid = true;
        
        for (let i = 0; i < db.Joint.length; i++) {{
          const jd = db.Joint[i];
          const b = boneMap[jd.node];
          if (!b) {{ valid = false; break; }}
          
          jointBones.push(b);
          jointParams.push({{
            damping: jd.damping,
            dampingMin: jd.damping_min || jd.damping,
            dampingMax: jd.damping_max || jd.damping,
            dampingVelRatio: jd.damping_velocity_ratio || 1,
            isDynamicDamping: jd.is_dynamic_damping || false,
            resilience: jd.resilience,
            gravity: jd.gravity,
            rotLimit: jd.rotation_limit,
            colRadius: jd.collision_radius || 0,
            isDisable: jd.is_disable,
            freezeAxis: jd.freeze_axis || 0
          }});
          
          // Find actual skeleton parent within this chain
          if (i === 0) {{
            parentIdx.push(-1);
          }} else {{
            let pName = skelParentName[jd.node];
            let pIdx = -1;
            while (pName) {{
              const idx = nodeNames.indexOf(pName);
              if (idx >= 0) {{ pIdx = idx; break; }}
              pName = skelParentName[pName];
            }}
            parentIdx.push(pIdx >= 0 ? pIdx : 0);
          }}
        }}
        if (!valid || jointBones.length < 2) return;
        
        // Resolve colliders
        const chainColliders = [];
        if (!db.ignore_collision) {{
          if (db.SpecificCollider && db.SpecificCollider.length > 0) {{
            db.SpecificCollider.forEach(name => {{
              const c = colliders.find(x => x.name === name);
              if (c) chainColliders.push(c);
            }});
          }} else {{
            colliders.forEach(c => {{
              if (!c.needSpecific) chainColliders.push(c);
            }});
          }}
        }}
        
        // Initialize particles at current world positions
        const particles = jointBones.map(b => {{
          const wp = new THREE.Vector3();
          b.getWorldPosition(wp);
          return {{ pos: wp.clone(), prevPos: wp.clone(), restLen: 0 }};
        }});
        
        // Rest lengths from actual skeleton parent
        for (let i = 1; i < particles.length; i++) {{
          const pi = parentIdx[i];
          particles[i].restLen = new THREE.Vector3().subVectors(
            particles[i].pos, particles[pi].pos).length();
        }}
        
        // Pre-allocate caches
        const animPos = jointBones.map(() => new THREE.Vector3());
        // Save current (clean, post-init) local quaternions for restore
        const savedLocalQuat = jointBones.map(b => b.quaternion.clone());
        
        dynChains.push({{
          bones: jointBones, params: jointParams, particles,
          parentIdx, colliders: chainColliders, animPos, savedLocalQuat
        }});
        chainCount++;
      }});
      
      // === CHAIN PRIORITY DEDUPLICATION ===
      // Some bones appear in multiple chains (e.g. Left_Rib01_Top is in the master
      // back-hair chain [2] with weak defaults r=1, AND in dedicated sub-chain [9]
      // with tuned r=20). Without dedup, dynProcessedBones gives priority to whichever
      // chain runs first (the master), ignoring the tuned sub-chain entirely.
      // Fix: for each duplicate bone, keep it active only in the chain where it has
      // the strongest/most specific params; disable it in the other chain(s).
      const boneChainMap = new Map(); // boneName → [entry objects]
      dynChains.forEach((chain, ci) => {{
        chain.bones.forEach((b, ji) => {{
          if (!boneChainMap.has(b.name)) boneChainMap.set(b.name, []);
          boneChainMap.get(b.name).push({{ chainIdx: ci, jointIdx: ji }});
        }});
      }});
      let dedupCount = 0;
      boneChainMap.forEach((entries, boneName) => {{
        if (entries.length < 2) return;
        // Pick the "best" entry: prefer the chain where this bone has highest
        // non-default resilience, or highest damping, or smallest chain size (= dedicated)
        let bestIdx = 0;
        let bestScore = -1;
        entries.forEach((e, ei) => {{
          const p = dynChains[e.chainIdx].params[e.jointIdx];
          const chainLen = dynChains[e.chainIdx].bones.length;
          // Score: resilience * 1000 + damping * 100 + (1/chainLen) to prefer dedicated chains
          const score = p.resilience * 1000 + p.damping * 100 + (100 / chainLen);
          if (score > bestScore) {{ bestScore = score; bestIdx = ei; }}
        }});
        // Disable this bone in all OTHER chains
        entries.forEach((e, ei) => {{
          if (ei !== bestIdx) {{
            dynChains[e.chainIdx].params[e.jointIdx].isDisable = true;
            dedupCount++;
          }}
        }});
      }});
      if (dedupCount > 0) debug('Chain dedup: disabled', dedupCount, 'duplicate bone entries');
      
      const info = document.getElementById('dynBonesInfo');
      if (info) info.textContent = `${{chainCount}} chains`;
      debug('Dynamic bones initialized:', chainCount, 'chains');
    }}

    function toggleDynamicBones() {{
      dynamicBonesEnabled = !dynamicBonesEnabled;
      const sw = document.getElementById('swPhysics'); if (sw) sw.checked = dynamicBonesEnabled;
      if (dynamicBonesEnabled && dynChains.length === 0) initDynamicBones();
      
      // Show/hide intensity slider
      const row = document.getElementById('dynIntensityRow');
      if (row) row.style.display = dynamicBonesEnabled ? 'block' : 'none';
      
      if (!dynamicBonesEnabled && dynChains.length > 0) {{
        // Turning off: restore bones to clean animated state
        dynChains.forEach(chain => {{
          chain.bones.forEach((b, i) => {{
            b.quaternion.copy(chain.savedLocalQuat[i]);
          }});
        }});
        if (bones.length > 0) bones[0].updateMatrixWorld(true);
      }}
      
      resetDynamicBones();
      dynAccum = 0;
    }}

    function toggleCollisions() {{
      dynCollisionsEnabled = !dynCollisionsEnabled;
      const sw = document.getElementById('swCollisions'); if (sw) sw.checked = dynCollisionsEnabled;
    }}

    function updateDynIntensity(val) {{
      const v = parseInt(val);
      // Negative: linear 1→0 (fast cutoff, -50→0x)
      // Positive: gentle curve, max 1.4x at +400
      if (v <= 0) {{
        dynIntensityMult = Math.max(0, 1.0 + v / 50);
      }} else {{
        dynIntensityMult = 1.0 + (v / 400) * 0.4;
      }}
      const label = document.getElementById('dynIntensityLabel');
      if (label) {{
        label.textContent = (v >= 0 ? '+' : '') + v;
      }}
    }}

    function resetDynamicBones() {{
      // Restore bones to clean state
      dynChains.forEach(chain => {{
        chain.bones.forEach((b, i) => {{
          b.quaternion.copy(chain.savedLocalQuat[i]);
        }});
      }});
      if (bones.length > 0) bones[0].updateMatrixWorld(true);
      // Reset particles to current (clean) positions
      dynChains.forEach(chain => {{
        chain.bones.forEach((b, i) => {{
          b.getWorldPosition(chain.particles[i].pos);
          chain.particles[i].prevPos.copy(chain.particles[i].pos);
        }});
      }});
      dynLastAnimTime = -1;
      dynPrevCamPos = null;
      
      // Compute collision exemptions: particles that START inside a collider
      // in their natural bind/animated pose should never be pushed out by that collider.
      // This is computed once at init, NOT per-frame (unlike the old animPos check).
      // During animation, collisions work normally for non-exempt pairs.
      dynChains.forEach(chain => {{
        const exempt = new Set();
        const chainColl = chain.colliders;
        for (let pi = 0; pi < chain.particles.length; pi++) {{
          const ppos = chain.particles[pi].pos;
          const colRad = chain.params[pi].colRadius;
          for (let ci = 0; ci < chainColl.length; ci++) {{
            const col = chainColl[ci];
            const colWP = new THREE.Vector3();
            col.bone.getWorldPosition(colWP);
            const boneQ = col.bone.getWorldQuaternion(new THREE.Quaternion());
            if (col.offset.lengthSq() > 0) {{
              colWP.add(col.offset.clone().applyQuaternion(boneQ));
            }}
            const colQ = boneQ.clone();
            if (col.offsetRot) colQ.multiply(col.offsetRot);
            
            let inside = false;
            if (col.type === 0) {{
              // Sphere
              const d = ppos.distanceTo(colWP);
              inside = d < col.radius + colRad;
            }} else if (col.type === 2) {{
              // Capsule - use correct axis from param2
              const axisVec = [new THREE.Vector3(1,0,0), new THREE.Vector3(0,1,0), new THREE.Vector3(0,0,1)];
              const axis = (axisVec[col.param2] || axisVec[2]).clone().applyQuaternion(colQ);
              const halfH = col.height * 0.5;
              const diff = new THREE.Vector3().subVectors(ppos, colWP);
              let proj = diff.dot(axis);
              proj = Math.max(-halfH, Math.min(halfH, proj));
              const cp = colWP.clone().addScaledVector(axis, proj);
              const d = ppos.distanceTo(cp);
              inside = d < col.radius + colRad;
            }} else if (col.type === 1) {{
              // Finite plane: normal = local Y rotated by colQ
              const normal = new THREE.Vector3(0, 1, 0).applyQuaternion(colQ);
              const diff = new THREE.Vector3().subVectors(ppos, colWP);
              const dist = diff.dot(normal);
              inside = dist < col.height + colRad;
            }} else if (col.type === 4) {{
              // Oriented plane
              const normal = new THREE.Vector3(0, 1, 0).applyQuaternion(colQ);
              const diff = new THREE.Vector3().subVectors(ppos, colWP);
              const dist = diff.dot(normal);
              inside = dist < colRad;
            }}
            if (inside) exempt.add(pi + '_' + ci);
          }}
        }}
        chain.collisionExempt = exempt;
      }});
    }}

    // Track camera for virtual inertia (physics reacts to mouse rotation)
    let dynPrevCamPos = null;
    
    function updateDynamicBones(dt) {{
      if (!dynamicBonesEnabled || dynChains.length === 0 || dt <= 0) return;
      
      // ============================================================
      // Virtual inertia: when camera orbits, model appears to rotate.
      // Shift particle prevPos to simulate the model moving, creating
      // Verlet velocity that makes physics react to mouse dragging.
      // ============================================================
      if (dynPrevCamPos) {{
        const curCamPos = camera.position.clone();
        const pivot = controls.target;
        
        // Compute rotation from prev camera to current camera around pivot
        const prevDir = new THREE.Vector3().subVectors(dynPrevCamPos, pivot).normalize();
        const curDir = new THREE.Vector3().subVectors(curCamPos, pivot).normalize();
        const dot = THREE.MathUtils.clamp(prevDir.dot(curDir), -1, 1);
        
        if (dot < 0.99999) {{
          // Camera moved — compute rotation quaternion
          const axis = new THREE.Vector3().crossVectors(prevDir, curDir).normalize();
          const angle = Math.acos(dot);
          // Inverse rotation: if camera went right, model "swung" left
          const invQ = new THREE.Quaternion().setFromAxisAngle(axis, -angle);
          
          dynChains.forEach(chain => {{
            for (let i = 0; i < chain.particles.length; i++) {{
              if (chain.parentIdx[i] < 0) continue; // skip root
              // Rotate prevPos around pivot by inverse camera rotation
              chain.particles[i].prevPos.sub(pivot).applyQuaternion(invQ).add(pivot);
            }}
          }});
        }}
      }}
      dynPrevCamPos = camera.position.clone();
      
      // ============================================================
      // Phase 1: Capture clean animated state
      // Bones are already clean: animate() restored savedLocalQuat
      // before mixer ran, then mixer set animated bones to frame N.
      // ============================================================
      dynChains.forEach(chain => {{
        chain.bones.forEach((b, i) => {{
          chain.savedLocalQuat[i].copy(b.quaternion);  // Save clean for next frame restore
          b.getWorldPosition(chain.animPos[i]);
        }});
      }});
      
      // Detect animation loop — reset particles only
      if (currentAnimation) {{
        const clip = currentAnimation.getClip();
        const curTime = currentAnimation.time % clip.duration;
        if (dynLastAnimTime >= 0 && curTime < dynLastAnimTime - 0.1) {{
          dynChains.forEach(chain => {{
            chain.bones.forEach((b, i) => {{
              chain.particles[i].pos.copy(chain.animPos[i]);
              chain.particles[i].prevPos.copy(chain.animPos[i]);
            }});
          }});
        }}
        dynLastAnimTime = curTime;
      }}
      
      // Phase 2: Physics
      dynAccum += Math.min(dt, 0.05);
      while (dynAccum >= DYN_FIXED_DT) {{
        dynAccum -= DYN_FIXED_DT;
        dynPhysicsStep(DYN_FIXED_DT);
      }}
      
      // Phase 3: Apply rotation deltas to bones
      dynProcessedBones.clear();
      dynChains.forEach(chain => dynApplyChain(chain));
    }}

    function dynPhysicsStep(dt) {{
      dynChains.forEach(chain => {{
        const {{ params, particles, parentIdx, animPos, colliders: chainColl }} = chain;
        
        for (let i = 0; i < particles.length; i++) {{
          const p = particles[i];
          const pp = params[i];
          const pi = parentIdx[i];
          
          // Root and disabled: follow animation
          if (pi < 0 || pp.isDisable) {{
            p.prevPos.copy(p.pos);
            p.pos.copy(animPos[i]);
            continue;
          }}
          
          // Dynamic damping: adjust based on particle velocity
          let effectiveDamping = pp.damping;
          if (pp.isDynamicDamping) {{
            // Velocity magnitude (from verlet)
            const velMag = Math.sqrt(
              (p.pos.x - p.prevPos.x) ** 2 +
              (p.pos.y - p.prevPos.y) ** 2 +
              (p.pos.z - p.prevPos.z) ** 2
            ) / dt;
            // Blend between dampingMin (at rest) and dampingMax (at high velocity)
            const velFactor = Math.min(velMag / Math.max(pp.dampingVelRatio, 0.001), 1.0);
            effectiveDamping = pp.dampingMin + (pp.dampingMax - pp.dampingMin) * velFactor;
          }}
          
          // Verlet integration with dynamic damping
          // Intensity scales ALL physics (velocity + gravity), not just gravity
          // Velocity clamp below prevents runaway oscillation
          let vx = (p.pos.x - p.prevPos.x) * (1 - effectiveDamping) * dynIntensityMult;
          let vy = (p.pos.y - p.prevPos.y) * (1 - effectiveDamping) * dynIntensityMult;
          let vz = (p.pos.z - p.prevPos.z) * (1 - effectiveDamping) * dynIntensityMult;
          
          // Clamp velocity magnitude to prevent excessive oscillation
          // Max displacement per step = 50% of bone length
          const vMag = Math.sqrt(vx*vx + vy*vy + vz*vz);
          const maxV = p.restLen * 0.5;
          if (vMag > maxV && vMag > 0.0001) {{
            const vs = maxV / vMag;
            vx *= vs; vy *= vs; vz *= vs;
          }}
          
          p.prevPos.copy(p.pos);
          
          // Velocity
          p.pos.x += vx;
          p.pos.y += vy;
          p.pos.z += vz;
          
          // Gravity (acceleration: pos += g * dt², scaled by intensity)
          p.pos.y += pp.gravity * dt * dt * dynIntensityMult;
          
          // Elasticity: spring force pulling toward animated position
          // resilience/100 = spring coefficient per step
          const elasticity = pp.resilience / 100;
          if (elasticity > 0) {{
            p.pos.x += (animPos[i].x - p.pos.x) * elasticity;
            p.pos.y += (animPos[i].y - p.pos.y) * elasticity;
            p.pos.z += (animPos[i].z - p.pos.z) * elasticity;
          }}
          
          // === CONSTRAINTS (applied in order) ===
          
          // 1. Distance constraint: maintain bone length from parent
          const parentPos = particles[pi].pos;
          let dx = p.pos.x - parentPos.x;
          let dy = p.pos.y - parentPos.y;
          let dz = p.pos.z - parentPos.z;
          let len = Math.sqrt(dx*dx + dy*dy + dz*dz);
          if (len > 0.0001 && p.restLen > 0.0001) {{
            const s = p.restLen / len;
            p.pos.x = parentPos.x + dx * s;
            p.pos.y = parentPos.y + dy * s;
            p.pos.z = parentPos.z + dz * s;
          }}
          
          // 2. Rotation limit: constrain angle from animated direction
          // rotLimit=0 is special: means ZERO deviation allowed (snap to animated direction)
          // rotLimit>0 && <3.0: constrain within angle
          // rotLimit>=3.0 (pi): unconstrained
          if (pp.rotLimit >= 0 && pp.rotLimit < 3.0) {{
            const ax = animPos[i].x - animPos[pi].x;
            const ay = animPos[i].y - animPos[pi].y;
            const az = animPos[i].z - animPos[pi].z;
            const aLen = Math.sqrt(ax*ax + ay*ay + az*az);
            if (aLen > 0.0001) {{
              if (pp.rotLimit < 0.001) {{
                // rotLimit≈0: snap particle to animated direction from CURRENT parent
                const invA = p.restLen / aLen;
                p.pos.x = parentPos.x + ax * invA;
                p.pos.y = parentPos.y + ay * invA;
                p.pos.z = parentPos.z + az * invA;
              }} else {{
                dx = p.pos.x - parentPos.x;
                dy = p.pos.y - parentPos.y;
                dz = p.pos.z - parentPos.z;
                len = Math.sqrt(dx*dx + dy*dy + dz*dz);
                if (len > 0.0001) {{
                  const anx = ax/aLen, any_ = ay/aLen, anz = az/aLen;
                  const cnx = dx/len, cny = dy/len, cnz = dz/len;
                  const dot = Math.max(-1, Math.min(1, anx*cnx + any_*cny + anz*cnz));
                  const angle = Math.acos(dot);
                  if (angle > pp.rotLimit) {{
                    const crossX = any_*cnz - anz*cny;
                    const crossY = anz*cnx - anx*cnz;
                    const crossZ = anx*cny - any_*cnx;
                    const crossLen = Math.sqrt(crossX*crossX + crossY*crossY + crossZ*crossZ);
                    if (crossLen > 1e-6) {{
                      const tmpV = new THREE.Vector3(anx, any_, anz);
                      tmpV.applyAxisAngle(new THREE.Vector3(crossX/crossLen, crossY/crossLen, crossZ/crossLen), pp.rotLimit);
                      p.pos.x = parentPos.x + tmpV.x * p.restLen;
                      p.pos.y = parentPos.y + tmpV.y * p.restLen;
                      p.pos.z = parentPos.z + tmpV.z * p.restLen;
                    }}
                  }}
                }}
              }}
            }}
          }}
          
          // 3. Collision
          // Colliders represent body geometry. They push particles out when physics
          // simulation moves them inside. Particles that naturally START inside a
          // collider (computed once at init in resetDynamicBones) are exempt.
          if (dynCollisionsEnabled) {{
          const colRad = pp.colRadius;
          const exempt = chain.collisionExempt || new Set();
          for (let ci = 0; ci < chainColl.length; ci++) {{
            if (exempt.has(i + '_' + ci)) continue; // Natural position, skip
            
            const col = chainColl[ci];
            const colWP = new THREE.Vector3();
            col.bone.getWorldPosition(colWP);
            const boneQ = col.bone.getWorldQuaternion(new THREE.Quaternion());
            if (col.offset.lengthSq() > 0) {{
              colWP.add(col.offset.clone().applyQuaternion(boneQ));
            }}
            const colQ = boneQ.clone();
            if (col.offsetRot) colQ.multiply(col.offsetRot);
            
            if (col.type === 0) {{
              // Sphere: param0 = radius
              const r = col.radius + colRad;
              if (r > 0) {{
                dx = p.pos.x - colWP.x;
                dy = p.pos.y - colWP.y;
                dz = p.pos.z - colWP.z;
                const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
                if (dist < r && dist > 0.0001) {{
                  const push = r / dist;
                  p.pos.x = colWP.x + dx * push;
                  p.pos.y = colWP.y + dy * push;
                  p.pos.z = colWP.z + dz * push;
                }}
              }}
            }} else if (col.type === 2) {{
              // Capsule: param0 = radius, param1 = height, param2 = axis (0=X, 1=Y, 2=Z)
              const axisVecs = [new THREE.Vector3(1,0,0), new THREE.Vector3(0,1,0), new THREE.Vector3(0,0,1)];
              const axis = (axisVecs[col.param2] || axisVecs[2]).clone().applyQuaternion(colQ);
              const halfH = col.height * 0.5;
              const r = col.radius + colRad;
              if (r > 0) {{
                dx = p.pos.x - colWP.x;
                dy = p.pos.y - colWP.y;
                dz = p.pos.z - colWP.z;
                let proj = dx*axis.x + dy*axis.y + dz*axis.z;
                proj = Math.max(-halfH, Math.min(halfH, proj));
                const cpx = colWP.x + axis.x * proj;
                const cpy = colWP.y + axis.y * proj;
                const cpz = colWP.z + axis.z * proj;
                dx = p.pos.x - cpx;
                dy = p.pos.y - cpy;
                dz = p.pos.z - cpz;
                const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
                if (dist < r && dist > 0.0001) {{
                  const push = r / dist;
                  p.pos.x = cpx + dx * push;
                  p.pos.y = cpy + dy * push;
                  p.pos.z = cpz + dz * push;
                }}
              }}
            }} else if (col.type === 1) {{
              // Finite plane: normal = local Y rotated by colQ
              // param0 = half-width, param1 = margin/thickness, param2 = half-depth
              const normal = new THREE.Vector3(0, 1, 0).applyQuaternion(colQ);
              const r = col.height + colRad;
              dx = p.pos.x - colWP.x;
              dy = p.pos.y - colWP.y;
              dz = p.pos.z - colWP.z;
              const dist = dx*normal.x + dy*normal.y + dz*normal.z;
              if (dist < r) {{
                // Check finite extent: project onto local X and Z axes
                const localX = new THREE.Vector3(1, 0, 0).applyQuaternion(colQ);
                const localZ = new THREE.Vector3(0, 0, 1).applyQuaternion(colQ);
                const projX = Math.abs(dx*localX.x + dy*localX.y + dz*localX.z);
                const projZ = Math.abs(dx*localZ.x + dy*localZ.y + dz*localZ.z);
                const extentX = col.radius > 0 ? col.radius : 999;  // param0
                const extentZ = col.param2 > 0 ? col.param2 : 999;  // param2
                if (projX < extentX && projZ < extentZ) {{
                  const push = r - dist;
                  p.pos.x += normal.x * push;
                  p.pos.y += normal.y * push;
                  p.pos.z += normal.z * push;
                }}
              }}
            }} else if (col.type === 4) {{
              // Oriented plane: infinite plane through collider position
              const normal = new THREE.Vector3(0, 1, 0).applyQuaternion(colQ);
              const margin = colRad;
              dx = p.pos.x - colWP.x;
              dy = p.pos.y - colWP.y;
              dz = p.pos.z - colWP.z;
              const dist = dx*normal.x + dy*normal.y + dz*normal.z;
              if (dist < margin) {{
                const push = margin - dist;
                p.pos.x += normal.x * push;
                p.pos.y += normal.y * push;
                p.pos.z += normal.z * push;
              }}
            }}
          }}
          }} // end dynCollisionsEnabled
          
          // 4. Freeze axis: constrain particle to a plane in bone's local space
          // freeze_axis=1 → freeze local X: particle can only deviate in parent's YZ plane
          if (pp.freezeAxis === 1) {{
            // Get parent bone's local X axis in world space
            const parentBone = chain.bones[pi];
            const localX = new THREE.Vector3(1, 0, 0);
            const parentWorldQ = parentBone.getWorldQuaternion(new THREE.Quaternion());
            localX.applyQuaternion(parentWorldQ);
            
            // Project particle displacement onto the freeze plane (remove X component)
            dx = p.pos.x - parentPos.x;
            dy = p.pos.y - parentPos.y;
            dz = p.pos.z - parentPos.z;
            const projOnX = dx * localX.x + dy * localX.y + dz * localX.z;
            // Get animated direction's X component to preserve it
            const aDx = animPos[i].x - animPos[pi].x;
            const aDy = animPos[i].y - animPos[pi].y;
            const aDz = animPos[i].z - animPos[pi].z;
            const animProjOnX = aDx * localX.x + aDy * localX.y + aDz * localX.z;
            // Replace physics X-proj with animated X-proj (freeze)
            const correction = animProjOnX - projOnX;
            p.pos.x += localX.x * correction;
            p.pos.y += localX.y * correction;
            p.pos.z += localX.z * correction;
            // Re-apply distance constraint
            dx = p.pos.x - parentPos.x;
            dy = p.pos.y - parentPos.y;
            dz = p.pos.z - parentPos.z;
            len = Math.sqrt(dx*dx + dy*dy + dz*dz);
            if (len > 0.0001 && p.restLen > 0.0001) {{
              const s3 = p.restLen / len;
              p.pos.x = parentPos.x + dx * s3;
              p.pos.y = parentPos.y + dy * s3;
              p.pos.z = parentPos.z + dz * s3;
            }}
          }}
        }}
      }});
    }}

    function dynApplyChain(chain) {{
      const {{ bones: cBones, particles, parentIdx }} = chain;
      
      // Build child map
      const childMap = new Map();
      for (let i = 0; i < cBones.length; i++) {{
        const pi = parentIdx[i];
        if (pi >= 0) {{
          if (!childMap.has(pi)) childMap.set(pi, []);
          childMap.get(pi).push(i);
        }}
      }}
      
      // Process root-to-tip (BFS)
      const queue = [];
      for (let i = 0; i < cBones.length; i++) {{
        if (parentIdx[i] < 0) queue.push(i);
      }}
      
      while (queue.length > 0) {{
        const i = queue.shift();
        const children = childMap.get(i) || [];
        children.forEach(ci => queue.push(ci));
        
        const boneName = cBones[i].name;
        if (dynProcessedBones.has(boneName)) continue;
        if (children.length === 0) continue;
        if (chain.params[i].isDisable) continue;  // Disabled bones should not be rotated
        
        dynProcessedBones.add(boneName);
        
        const ci = children[0];
        const bone = cBones[i];
        
        // Use LIVE bone positions (post-parent-rotation) for reference direction.
        // After parent bones are rotated + updateWorldMatrix, this bone's world
        // position already reflects parent rotations. Using live positions means
        // we only compute the ADDITIONAL rotation needed for THIS bone, preventing
        // cumulative over-rotation down the chain (which caused U-fold artifacts).
        // dynProcessedBones prevents double-processing across chains.
        const boneWP = new THREE.Vector3();
        const childWP = new THREE.Vector3();
        cBones[i].getWorldPosition(boneWP);
        cBones[ci].getWorldPosition(childWP);
        const curDir = new THREE.Vector3().subVectors(childWP, boneWP);
        const curLen = curDir.length();
        if (curLen < 0.0001) continue;
        curDir.divideScalar(curLen);
        
        // Simulated direction from particles
        const simDir = new THREE.Vector3().subVectors(particles[ci].pos, particles[i].pos);
        const simLen = simDir.length();
        if (simLen < 0.0001) continue;
        simDir.divideScalar(simLen);
        
        // Skip if nearly identical (no rotation needed)
        const dot = curDir.dot(simDir);
        if (dot > 0.9999) continue;
        
        // Note: rotation limits are already enforced on particle positions in
        // dynPhysicsStep. Applying them again here would double-constrain non-root
        // bones (parent rotation shifts live curDir, making the effective limit tighter).
        // So we just faithfully rotate the bone toward where the particle ended up.
        
        // World-space rotation delta: current bone direction → simulated
        const worldRot = new THREE.Quaternion().setFromUnitVectors(curDir, simDir);
        
        // Convert to local space delta:
        // We want: newWorldQ = worldRot * curWorldQ
        // Since: curWorldQ = parentWorldQ * localQ
        // Then: newWorldQ = worldRot * parentWorldQ * localQ  
        //                 = parentWorldQ * [inv(parentWorldQ) * worldRot * parentWorldQ] * localQ
        //                 = parentWorldQ * localDelta * localQ
        // So: bone.quaternion = localDelta * bone.quaternion  (premultiply)
        const skelParent = bone.parent;
        const parentWorldQ = skelParent ?
          skelParent.getWorldQuaternion(new THREE.Quaternion()) :
          new THREE.Quaternion();
        
        const localDelta = parentWorldQ.clone().invert().multiply(worldRot).multiply(parentWorldQ);
        bone.quaternion.premultiply(localDelta);
        
        // Propagate so children see this change
        bone.updateWorldMatrix(false, true);
      }}
    }}

    function populateMeshList() {{
      const list = document.getElementById('mesh-list');
      meshes.forEach((mesh, idx) => {{
        const div = document.createElement('div');
        div.className = 'mesh-toggle';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = mesh.visible;
        checkbox.id = `mesh-${{idx}}`;
        checkbox.addEventListener('change', () => {{
          mesh.visible = checkbox.checked;
          if (wireframeOverlayMode && mesh.userData.wireframeOverlay) {{
            mesh.userData.wireframeOverlay.visible = checkbox.checked;
          }}
          updateStats();
        }});
        
        const label = document.createElement('label');
        label.htmlFor = `mesh-${{idx}}`;
        label.textContent = mesh.userData.meshName;
        label.style.flex = '1';
        label.style.overflow = 'hidden';
        label.style.textOverflow = 'ellipsis';
        
        if (mesh.userData.hasTexture) {{
          const indicator = document.createElement('span');
          indicator.className = 'texture-indicator';
          indicator.title = 'Has texture';
          label.appendChild(indicator);
        }}
        
        if (mesh.userData.isShadowMesh) {{
          const shadowIcon = document.createElement('span');
          shadowIcon.className = 'shadow-indicator';
          shadowIcon.title = 'Shadow mesh (group: ' + mesh.userData.meshGroup + ')';
          label.appendChild(shadowIcon);
        }}
        
        const spotBtn = document.createElement('span');
        spotBtn.textContent = '💡';
        spotBtn.title = 'Highlight this mesh (no zoom)';
        spotBtn.style.cssText = 'cursor:pointer;font-size:12px;padding:1px 3px;border-radius:3px;opacity:0.4;flex-shrink:0;transition:opacity 0.15s';
        spotBtn.addEventListener('mouseenter', () => {{ spotBtn.style.opacity = '1'; }});
        spotBtn.addEventListener('mouseleave', () => {{ spotBtn.style.opacity = '0.4'; }});
        spotBtn.addEventListener('click', (e) => {{ e.preventDefault(); e.stopPropagation(); focusMesh(idx, false); }});
        
        const focusBtn = document.createElement('span');
        focusBtn.textContent = '🎯';
        focusBtn.title = 'Focus on this mesh';
        focusBtn.style.cssText = 'cursor:pointer;font-size:12px;padding:1px 3px;border-radius:3px;opacity:0.4;flex-shrink:0;transition:opacity 0.15s';
        focusBtn.addEventListener('mouseenter', () => {{ focusBtn.style.opacity = '1'; }});
        focusBtn.addEventListener('mouseleave', () => {{ focusBtn.style.opacity = '0.4'; }});
        focusBtn.addEventListener('click', (e) => {{ e.preventDefault(); e.stopPropagation(); focusMesh(idx, true); }});
        
        div.appendChild(checkbox);
        div.appendChild(label);
        div.appendChild(spotBtn);
        div.appendChild(focusBtn);
        list.appendChild(div);
      }});
      // Show FXO Shaders toggle if toon materials are present
      if (shaderStats.toon > 0 && !NO_SHADERS) {{
        const fxoRow = document.getElementById('fxoToggleRow');
        if (fxoRow) fxoRow.style.display = '';
      }}
      // Show Hide Shadow toggle if any shadow meshes exist
      if (meshes.some(m => m.userData.isShadowMesh)) {{
        const shadowRow = document.getElementById('shadowToggleRow');
        if (shadowRow) shadowRow.style.display = '';
      }}
    }}

    function toggleAllMeshes(visible) {{
      meshes.forEach((m, idx) => {{
        m.visible = visible;
        if (wireframeOverlayMode && m.userData.wireframeOverlay) {{
          m.userData.wireframeOverlay.visible = visible;
        }}
        const checkbox = document.getElementById(`mesh-${{idx}}`);
        if (checkbox) checkbox.checked = visible;
      }});
      if (visible) {{
        hideShadowMeshes = false;
        const sw = document.getElementById('swShadow'); if (sw) sw.checked = false;
      }}
      updateStats();
    }}

    function toggleHideShadow() {{
      hideShadowMeshes = !hideShadowMeshes;
      const sw = document.getElementById('swShadow'); if (sw) sw.checked = hideShadowMeshes;
      meshes.forEach((m, idx) => {{
        if (!m.userData.isShadowMesh) return;
        m.visible = !hideShadowMeshes;
        if (wireframeOverlayMode && m.userData.wireframeOverlay) {{
          m.userData.wireframeOverlay.visible = !hideShadowMeshes;
        }}
        const cb = document.getElementById(`mesh-${{idx}}`);
        if (cb) cb.checked = !hideShadowMeshes;
      }});
      updateStats();
    }}

    function setFxoShaders(enabled) {{
      fxoShadersEnabled = enabled;
      meshes.forEach(m => {{
        if (!m.userData.fxoMaterial) return;  // not a toon-shaded mesh
        
        // Resolve textures based on current textureMode
        const tex = textureMode ? (m.userData.originalMap || null) : null;
        const nmap = textureMode ? (m.userData.originalNormalMap || null) : null;
        
        if (fxoShadersEnabled) {{
          // Restore FXO material
          const fxo = m.userData.fxoMaterial;
          fxo.map = tex;
          fxo.normalMap = nmap;
          if (colorMode) {{
            fxo.color.setHex(m.userData.originalColor);
          }} else {{
            fxo.color.setHex(tex ? 0xffffff : 0x808080);
          }}
          fxo.wireframe = wireframeMode;
          // Apply emissive state
          if (emissiveEnabled && m.userData.fxoMaterial.userData.emissiveGlow > 0) {{
            fxo.emissive = new THREE.Color(0xffffff);
            fxo.emissiveIntensity = Math.max(0, Math.min(m.userData.fxoMaterial.userData.emissiveGlow + emissiveGlowOffset, 3.0));
          }} else {{
            fxo.emissive = new THREE.Color(0x000000);
            fxo.emissiveIntensity = 0;
          }}
          fxo.needsUpdate = true;
          m.material = fxo;
        }} else {{
          // Switch to default MeshStandardMaterial
          if (!m.userData.defaultMaterial) {{
            const fxo = m.userData.fxoMaterial;
            m.userData.defaultMaterial = new THREE.MeshStandardMaterial({{
              roughness: 0.7, metalness: 0.2, side: THREE.DoubleSide, skinning: true,
              transparent: fxo.transparent || false,
              alphaTest: fxo.alphaTest || 0
            }});
            // Copy emissive glow eligibility
            if (fxo.userData.emissiveGlow) m.userData.defaultMaterial.userData.emissiveGlow = fxo.userData.emissiveGlow;
          }}
          const def = m.userData.defaultMaterial;
          def.map = tex;
          def.normalMap = nmap;
          if (colorMode) {{
            def.color.setHex(m.userData.originalColor);
          }} else {{
            def.color.setHex(tex ? 0xffffff : 0x808080);
          }}
          def.wireframe = wireframeMode;
          // Apply emissive state
          if (emissiveEnabled && def.userData.emissiveGlow > 0) {{
            def.emissive = new THREE.Color(0xffffff);
            def.emissiveIntensity = Math.max(0, Math.min(def.userData.emissiveGlow + emissiveGlowOffset, 3.0));
          }} else {{
            def.emissive = new THREE.Color(0x000000);
            def.emissiveIntensity = 0;
          }}
          def.needsUpdate = true;
          m.material = def;
        }}
      }});
      updateStats();
    }}

    function setRecomputeNormals(enabled) {{
      recomputeNormalsEnabled = enabled;
      meshes.forEach(m => {{
        const geom = m.geometry;
        if (!geom || !geom.userData || !geom.userData.originalNormals) return;
        if (enabled) {{
          computeSmoothNormals(geom);
        }} else {{
          const origNormals = geom.userData.originalNormals;
          geom.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(origNormals), 3));
        }}
        geom.attributes.normal.needsUpdate = true;
      }});
      updateStats();
    }}

    function computeSmoothNormals(geometry) {{
      const posAttr = geometry.attributes.position;
      const indexAttr = geometry.index;
      if (!posAttr || !indexAttr) return;
      
      const n = posAttr.count;
      const positions = posAttr.array;
      const indices = indexAttr.array;
      const normals = new Float32Array(n * 3);
      
      // Step 1: Accumulate face normals per vertex
      for (let i = 0; i < indices.length; i += 3) {{
        const i0 = indices[i], i1 = indices[i+1], i2 = indices[i+2];
        const ax = positions[i0*3], ay = positions[i0*3+1], az = positions[i0*3+2];
        const bx = positions[i1*3], by = positions[i1*3+1], bz = positions[i1*3+2];
        const cx = positions[i2*3], cy = positions[i2*3+1], cz = positions[i2*3+2];
        
        const e1x = bx-ax, e1y = by-ay, e1z = bz-az;
        const e2x = cx-ax, e2y = cy-ay, e2z = cz-az;
        let fnx = e1y*e2z - e1z*e2y;
        let fny = e1z*e2x - e1x*e2z;
        let fnz = e1x*e2y - e1y*e2x;
        
        const len = Math.sqrt(fnx*fnx + fny*fny + fnz*fnz);
        if (len > 1e-12) {{ fnx /= len; fny /= len; fnz /= len; }}
        
        normals[i0*3] += fnx; normals[i0*3+1] += fny; normals[i0*3+2] += fnz;
        normals[i1*3] += fnx; normals[i1*3+1] += fny; normals[i1*3+2] += fnz;
        normals[i2*3] += fnx; normals[i2*3+1] += fny; normals[i2*3+2] += fnz;
      }}
      
      // Step 2: Spatial hash for position sharing
      const tol = 1e-6;
      const cellSize = tol * 10 || 1e-5;
      const cells = new Map();
      
      function hashKey(x, y, z) {{
        return Math.floor(x/cellSize) + ',' + Math.floor(y/cellSize) + ',' + Math.floor(z/cellSize);
      }}
      
      for (let i = 0; i < n; i++) {{
        const key = hashKey(positions[i*3], positions[i*3+1], positions[i*3+2]);
        if (!cells.has(key)) cells.set(key, []);
        cells.get(key).push(i);
      }}
      
      // Step 3: Share normals between vertices at same position
      const visited = new Uint8Array(n);
      for (let i = 0; i < n; i++) {{
        if (visited[i]) continue;
        const px = positions[i*3], py = positions[i*3+1], pz = positions[i*3+2];
        const cx = Math.floor(px/cellSize), cy = Math.floor(py/cellSize), cz = Math.floor(pz/cellSize);
        const matches = [i];
        
        for (let dx = -1; dx <= 1; dx++) {{
          for (let dy = -1; dy <= 1; dy++) {{
            for (let dz = -1; dz <= 1; dz++) {{
              const key = (cx+dx) + ',' + (cy+dy) + ',' + (cz+dz);
              const bucket = cells.get(key);
              if (!bucket) continue;
              for (let k = 0; k < bucket.length; k++) {{
                const j = bucket[k];
                if (j === i || visited[j]) continue;
                const dx2 = positions[j*3]-px, dy2 = positions[j*3+1]-py, dz2 = positions[j*3+2]-pz;
                if (Math.sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2) < tol) {{
                  matches.push(j);
                }}
              }}
            }}
          }}
        }}
        
        if (matches.length > 1) {{
          let sx = 0, sy = 0, sz = 0;
          for (let m = 0; m < matches.length; m++) {{
            const idx = matches[m];
            sx += normals[idx*3]; sy += normals[idx*3+1]; sz += normals[idx*3+2];
          }}
          const slen = Math.sqrt(sx*sx + sy*sy + sz*sz);
          if (slen > 1e-12) {{ sx /= slen; sy /= slen; sz /= slen; }}
          for (let m = 0; m < matches.length; m++) {{
            const idx = matches[m];
            normals[idx*3] = sx; normals[idx*3+1] = sy; normals[idx*3+2] = sz;
            visited[idx] = 1;
          }}
        }} else {{
          visited[i] = 1;
        }}
      }}
      
      // Step 4: Normalize remaining
      for (let i = 0; i < n; i++) {{
        const nx = normals[i*3], ny = normals[i*3+1], nz = normals[i*3+2];
        const len = Math.sqrt(nx*nx + ny*ny + nz*nz);
        if (len > 1e-12) {{
          normals[i*3] /= len; normals[i*3+1] /= len; normals[i*3+2] /= len;
        }}
      }}
      
      geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    }}

    function computeTangents(geometry) {{
      const posAttr = geometry.attributes.position;
      const normAttr = geometry.attributes.normal;
      const uvAttr = geometry.attributes.uv;
      const indexAttr = geometry.index;
      if (!posAttr || !normAttr || !uvAttr || !indexAttr) return;
      
      const n = posAttr.count;
      const pos = posAttr.array;
      const norm = normAttr.array;
      const uv = uvAttr.array;
      const idx = indexAttr.array;
      
      // Accumulate tangent and bitangent per vertex
      const tan1 = new Float32Array(n * 3);  // tangent
      const tan2 = new Float32Array(n * 3);  // bitangent
      
      for (let i = 0; i < idx.length; i += 3) {{
        const i0 = idx[i], i1 = idx[i+1], i2 = idx[i+2];
        
        const x1 = pos[i1*3] - pos[i0*3], y1 = pos[i1*3+1] - pos[i0*3+1], z1 = pos[i1*3+2] - pos[i0*3+2];
        const x2 = pos[i2*3] - pos[i0*3], y2 = pos[i2*3+1] - pos[i0*3+1], z2 = pos[i2*3+2] - pos[i0*3+2];
        
        const s1 = uv[i1*2] - uv[i0*2], t1 = uv[i1*2+1] - uv[i0*2+1];
        const s2 = uv[i2*2] - uv[i0*2], t2 = uv[i2*2+1] - uv[i0*2+1];
        
        const r = 1.0 / (s1 * t2 - s2 * t1 || 1e-12);
        
        const sx = (t2 * x1 - t1 * x2) * r, sy = (t2 * y1 - t1 * y2) * r, sz = (t2 * z1 - t1 * z2) * r;
        const tx = (s1 * x2 - s2 * x1) * r, ty = (s1 * y2 - s2 * y1) * r, tz = (s1 * z2 - s2 * z1) * r;
        
        tan1[i0*3] += sx; tan1[i0*3+1] += sy; tan1[i0*3+2] += sz;
        tan1[i1*3] += sx; tan1[i1*3+1] += sy; tan1[i1*3+2] += sz;
        tan1[i2*3] += sx; tan1[i2*3+1] += sy; tan1[i2*3+2] += sz;
        
        tan2[i0*3] += tx; tan2[i0*3+1] += ty; tan2[i0*3+2] += tz;
        tan2[i1*3] += tx; tan2[i1*3+1] += ty; tan2[i1*3+2] += tz;
        tan2[i2*3] += tx; tan2[i2*3+1] += ty; tan2[i2*3+2] += tz;
      }}
      
      // Gram-Schmidt orthogonalize + compute handedness
      const tangents = new Float32Array(n * 4);
      for (let i = 0; i < n; i++) {{
        const nx = norm[i*3], ny = norm[i*3+1], nz = norm[i*3+2];
        const tx = tan1[i*3], ty = tan1[i*3+1], tz = tan1[i*3+2];
        
        // t - n * dot(n, t)
        const dot = nx*tx + ny*ty + nz*tz;
        let ox = tx - nx*dot, oy = ty - ny*dot, oz = tz - nz*dot;
        const len = Math.sqrt(ox*ox + oy*oy + oz*oz);
        if (len > 1e-12) {{ ox /= len; oy /= len; oz /= len; }}
        
        // Handedness: sign of dot(cross(n, t), tan2)
        const cx = ny*tz - nz*ty, cy = nz*tx - nx*tz, cz = nx*ty - ny*tx;
        const w = (cx*tan2[i*3] + cy*tan2[i*3+1] + cz*tan2[i*3+2]) < 0 ? -1.0 : 1.0;
        
        tangents[i*4] = ox; tangents[i*4+1] = oy; tangents[i*4+2] = oz; tangents[i*4+3] = w;
      }}
      
      geometry.setAttribute('tangent', new THREE.BufferAttribute(tangents, 4));
    }}

    function toggleColors() {{
      colorMode = !colorMode;
      const sw = document.getElementById('swColors'); if (sw) sw.checked = colorMode;
      meshes.forEach(m => {{
        if (colorMode) {{
          m.material.color.setHex(m.userData.originalColor);
        }} else if (m.material.map) {{
          m.material.color.setHex(0xffffff);
        }} else {{
          m.material.color.setHex(0x808080);
        }}
        m.material.needsUpdate = true;
      }});
    }}

    function toggleTextures() {{
      textureMode = !textureMode;
      const sw = document.getElementById('swTex'); if (sw) sw.checked = textureMode;
      meshes.forEach(m => {{
        if (textureMode) {{
          if (m.userData.originalMap) {{
            m.material.map = m.userData.originalMap;
            m.material.color.setHex(colorMode ? m.userData.originalColor : 0xffffff);
          }}
          if (m.userData.originalNormalMap) {{
            m.material.normalMap = m.userData.originalNormalMap;
          }}
        }} else {{
          m.material.map = null;
          m.material.normalMap = null;
          m.material.color.setHex(colorMode ? m.userData.originalColor : 0x808080);
        }}
        m.material.needsUpdate = true;
      }});
    }}

    function toggleWireframe() {{
      wireframeMode = !wireframeMode;
      const sw = document.getElementById('swWire'); if (sw) sw.checked = wireframeMode;
      // Turn off overlay if wireframe on
      if (wireframeMode && wireframeOverlayMode) {{
        wireframeOverlayMode = false;
        const sw2 = document.getElementById('swWireOver'); if (sw2) sw2.checked = false;
        meshes.forEach(m => {{ if (m.userData.wireframeOverlay) m.userData.wireframeOverlay.visible = false; }});
      }}
      meshes.forEach(m => {{
        m.material.wireframe = wireframeMode;
        m.material.needsUpdate = true;
      }});
    }}

    function toggleWireframeOverlay() {{
      wireframeOverlayMode = !wireframeOverlayMode;
      const sw = document.getElementById('swWireOver'); if (sw) sw.checked = wireframeOverlayMode;
      // Turn off wireframe if overlay on
      if (wireframeOverlayMode && wireframeMode) {{
        wireframeMode = false;
        const sw2 = document.getElementById('swWire'); if (sw2) sw2.checked = false;
        meshes.forEach(m => {{ m.material.wireframe = false; m.material.needsUpdate = true; }});
      }}
      if (wireframeOverlayMode) {{
        meshes.forEach(m => {{
          if (!m.userData.wireframeOverlay) {{
            const wireGeom = m.geometry.clone();
            const wireMat = new THREE.MeshBasicMaterial({{
              color: 0x00ffff,
              wireframe: true,
              transparent: true,
              opacity: 0.3,
              skinning: true
            }});
            let wireMesh;
            if (m.isSkinnedMesh && m.skeleton) {{
              wireMesh = new THREE.SkinnedMesh(wireGeom, wireMat);
              wireMesh.bind(m.skeleton, new THREE.Matrix4());
              wireMesh.frustumCulled = false;
              if (characterGroup) {{ characterGroup.add(wireMesh); }} else {{ scene.add(wireMesh); }}
            }} else {{
              wireMesh = new THREE.Mesh(wireGeom, wireMat);
              m.add(wireMesh);
            }}
            wireMesh.userData.isWireframeOverlay = true;
            m.userData.wireframeOverlay = wireMesh;
          }}
          m.userData.wireframeOverlay.visible = m.visible;
        }});
      }} else {{
        meshes.forEach(m => {{
          if (m.userData.wireframeOverlay) {{
            m.userData.wireframeOverlay.visible = false;
          }}
        }});
      }}
    }}

    function toggleEmissive() {{
      emissiveEnabled = !emissiveEnabled;
      const sw = document.getElementById('swEmissive'); if (sw) sw.checked = emissiveEnabled;
      const sliderRow = document.getElementById('emissiveSliderRow');
      if (sliderRow) sliderRow.style.display = emissiveEnabled ? '' : 'none';
      applyEmissiveGlow();
    }}

    function applyEmissiveGlow() {{
      meshes.forEach(m => {{
        const mat = m.material;
        const glow = mat.userData.emissiveGlow;
        if (glow && glow > 0) {{
          if (emissiveEnabled) {{
            const intensity = Math.max(0, Math.min(glow + emissiveGlowOffset, 3.0));
            mat.emissive = new THREE.Color(0xffffff);
            mat.emissiveIntensity = intensity;
          }} else {{
            mat.emissive = new THREE.Color(0x000000);
            mat.emissiveIntensity = 0;
          }}
          mat.needsUpdate = true;
        }}
      }});
    }}

    function resetView() {{
      // Disable gamepad if active
      if (gamepadEnabled) {{
        gamepadEnabled = false;
        freeCamMode = false;
        const fcSw = document.getElementById('swFreeCam'); if (fcSw) fcSw.checked = false;
        const fcSub = document.getElementById('freeCamSubmenu'); if (fcSub) fcSub.style.display = 'none';
        disableThirdPerson();
        gamepadType = 'generic';
        gamepadPrevButtons = [];
        gamepadButtonStates = [];
        gamepadAxesStates = [0,0,0,0];
        gamepadTriggerStates = [0,0];
        gamepadConnectedShown = false;
        gamepadCurrentId = '';
        const sw = document.getElementById('swGamepad'); if (sw) sw.checked = false;
        const statusEl = document.getElementById('gamepadStatus'); if (statusEl) statusEl.textContent = '';
        const submenu = document.getElementById('gamepadSubmenu'); if (submenu) submenu.style.display = 'none';
      }}
      // Reset all toggle states to defaults
      // Textures ON
      if (!textureMode) toggleTextures();
      // Colors OFF
      if (colorMode) toggleColors();
      // Wireframe OFF
      if (wireframeMode) toggleWireframe();
      // Wireframe Overlay OFF
      if (wireframeOverlayMode) toggleWireframeOverlay();
      // Skeleton OFF
      if (showSkeleton) toggleSkeleton();
      // Joints OFF
      if (showJoints) toggleJoints();
      // Bone Names OFF
      if (showBoneNames) toggleBoneNames();
      // Stop any running animation
      stopAnimation();
      // Dynamic bones OFF
      if (dynamicBonesEnabled) toggleDynamicBones();
      // Reset intensity slider
      dynIntensityMult = 1.0;
      const dynSlider = document.getElementById('dynIntensitySlider');
      if (dynSlider) dynSlider.value = 0;
      const dynLabel = document.getElementById('dynIntensityLabel');
      if (dynLabel) dynLabel.textContent = '+0';
      // Reset speed slider
      const speedSlider = document.getElementById('animSpeedSlider');
      if (speedSlider) speedSlider.value = 0;
      const speedLabel = document.getElementById('animSpeedLabel');
      if (speedLabel) speedLabel.textContent = '1.0x';
      // Mesh opacity 1
      document.getElementById('meshOpacity').value = 1;
      document.getElementById('meshOpVal').textContent = '1';
      setMeshOpacity(1);
      // Show all meshes, then re-hide shadow meshes (matching initial state)
      toggleAllMeshes(true);
      if (CONFIG.AUTO_HIDE_SHADOW) {{
        meshes.forEach((m, idx) => {{
          if (m.userData.isShadowMesh) {{
            m.visible = false;
            const cb = document.getElementById(`mesh-${{idx}}`);
            if (cb) cb.checked = false;
          }}
        }});
        hideShadowMeshes = true;
        const swShadow = document.getElementById('swShadow');
        if (swShadow) swShadow.checked = true;
      }}

      // Reset camera to frame model
      fitToView();
    }}

    // ============================================
    // FreeCam Mode
    // ============================================
    function toggleFreeCam() {{
      freeCamMode = !freeCamMode;
      const sw = document.getElementById('swFreeCam'); if (sw) sw.checked = freeCamMode;
      const submenu = document.getElementById('freeCamSubmenu');
      
      if (freeCamMode) {{
        if (submenu) submenu.style.display = 'block';
        
        // Compute base speed from model size
        const box = new THREE.Box3();
        meshes.filter(m => m.visible).forEach(m => box.expandByObject(m));
        if (box.isEmpty()) meshes.forEach(m => box.expandByObject(m));
        const size = box.getSize(new THREE.Vector3());
        freeCamBaseSpeed = Math.max(size.x, size.y, size.z) * 0.02;
        
        // Wider vertical look range for freecam
        tpCamPhi = 1.4;
        
        debug('FreeCam ON. BaseSpeed:', freeCamBaseSpeed.toFixed(4));
      }} else {{
        if (submenu) submenu.style.display = 'none';
        // Return camera to third-person orbit
        tpCamPhi = Math.max(0.3, Math.min(Math.PI * 0.45, tpCamPhi));
        if (characterGroup) updateThirdPersonCamera();
        debug('FreeCam OFF');
      }}
    }}

    // ============================================
    // Gamepad Controller Support
    // ============================================
    function toggleGamepad() {{
      gamepadEnabled = !gamepadEnabled;
      const sw = document.getElementById('swGamepad'); if (sw) sw.checked = gamepadEnabled;
      const statusEl = document.getElementById('gamepadStatus');
      const submenu = document.getElementById('gamepadSubmenu');
      
      if (gamepadEnabled) {{
        if (submenu) submenu.style.display = 'block';
        enableThirdPerson();
        gamepadType = 'generic';
        gamepadPrevButtons = [];
        gamepadConnectedShown = false;
        gamepadCurrentId = '';
        lastActiveInput = 'keyboard';  // start in keyboard, switch on real button press
        gamepadConfirmed = false;
        gamepadStaleFrames = 0;
        gamepadLastTimestamp = 0;
        if (statusEl) {{
          statusEl.textContent = '⌨️🖱️ Keyboard+Mouse · Press controller button to switch';
          statusEl.style.color = '#60a5fa';
        }}
      }} else {{
        // Reset FreeCam if active
        if (freeCamMode) {{
          freeCamMode = false;
          const fcSw = document.getElementById('swFreeCam'); if (fcSw) fcSw.checked = false;
          const fcSub = document.getElementById('freeCamSubmenu'); if (fcSub) fcSub.style.display = 'none';
        }}
        disableThirdPerson();
        gamepadType = 'generic';
        gamepadPrevButtons = [];
        gamepadButtonStates = [];
        gamepadAxesStates = [0,0,0,0];
        gamepadTriggerStates = [0,0];
        gamepadConnectedShown = false;
        gamepadCurrentId = '';
        kbUseKeyboard = false;
        lastActiveInput = 'keyboard';
        gamepadConfirmed = false;
        gamepadStaleFrames = 0;
        gamepadLastTimestamp = 0;
        Object.keys(kbKeys).forEach(k => kbKeys[k] = false);
        Object.keys(kbPrevKeys).forEach(k => kbPrevKeys[k] = false);
        mouseDeltaX = 0; mouseDeltaY = 0; mouseWheelDelta = 0; mouseRightDown = false;
        if (submenu) submenu.style.display = 'none';
        if (statusEl) statusEl.textContent = '';
      }}
    }}

    function toggleGamepadInvertX() {{
      gamepadInvertX = !gamepadInvertX;
      const sw = document.getElementById('swGpInvX'); if (sw) sw.checked = !gamepadInvertX;
    }}

    function toggleGamepadInvertY() {{
      gamepadInvertY = !gamepadInvertY;
      const sw = document.getElementById('swGpInvY'); if (sw) sw.checked = gamepadInvertY;
    }}

    function detectGamepadType(id) {{
      const lo = id.toLowerCase();
      // Xbox: Microsoft controllers, XInput, 045e vendor
      if (lo.includes('xbox') || lo.includes('045e') || lo.includes('xinput') || lo.includes('x-box')) return 'xbox';
      // PlayStation: Sony controllers, 054c vendor, DualSense/DualShock
      if (lo.includes('dualsense') || lo.includes('dualshock') || lo.includes('054c') || lo.includes('playstation')) return 'playstation';
      // "Wireless Controller" without other identifiers = likely PS (DualSense in Chrome)
      if (lo === 'wireless controller' || lo.startsWith('wireless controller (')) return 'playstation';
      // Switch: Nintendo, 057e vendor, Pro Controller
      if (lo.includes('nintendo') || lo.includes('057e') || lo.includes('pro controller') || lo.includes('switch')) return 'switch';
      // 8BitDo controllers often mimic Xbox or have their own IDs
      if (lo.includes('8bitdo')) {{
        if (lo.includes('pro') || lo.includes('sn30')) return 'switch';
        return 'xbox';  // most 8BitDo default to xinput
      }}
      // Logitech gamepads
      if (lo.includes('logitech') || lo.includes('046d')) return 'generic';
      // PowerA, Hori, PDP - usually Xbox or Switch layout
      if (lo.includes('hori')) {{
        if (lo.includes('nintendo') || lo.includes('switch')) return 'switch';
        return 'generic';
      }}
      // Standard gamepad (Chrome fallback label)
      if (lo.includes('standard gamepad') && lo.includes('vendor: 045e')) return 'xbox';
      if (lo.includes('standard gamepad') && lo.includes('vendor: 054c')) return 'playstation';
      if (lo.includes('standard gamepad') && lo.includes('vendor: 057e')) return 'switch';
      return 'generic';
    }}

    function updateGamepadStatusLabel() {{
      const statusEl = document.getElementById('gamepadStatus');
      if (!statusEl || !gamepadEnabled) return;
      // Just display based on current gamepadType (set by updateGamepad scan)
      statusEl.textContent = '🟢 ' + (GP_TYPE_NAMES[gamepadType] || 'Gamepad Connected');
      statusEl.style.color = '#4ade80';
    }}

    function enableThirdPerson() {{
      // Create character group and reparent model objects
      characterGroup = new THREE.Group();
      scene.add(characterGroup);
      
      // Compute model bounding box BEFORE reparenting
      const box = new THREE.Box3();
      meshes.filter(m => m.visible).forEach(m => box.expandByObject(m));
      if (box.isEmpty()) meshes.forEach(m => box.expandByObject(m));
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      const modelHeight = size.y;
      characterCenterY = center.y;
      characterMoveSpeed = modelHeight * 0.015;  // ~1.5% of model height per frame
      
      // Set camera distance based on model size
      tpCamDist = modelHeight * 2.5;
      tpCamTheta = Math.PI;
      tpCamPhi = 1.2;
      
      // Reparent: root bone
      if (bones.length > 0) {{
        scene.remove(bones[0]);
        characterGroup.add(bones[0]);
      }}
      // Reparent: all meshes
      meshes.forEach(m => {{
        scene.remove(m);
        characterGroup.add(m);
      }});
      // Reparent: wireframe overlays
      meshes.forEach(m => {{
        if (m.userData.wireframeOverlay) {{
          scene.remove(m.userData.wireframeOverlay);
          characterGroup.add(m.userData.wireframeOverlay);
        }}
      }});
      // Reparent: skeleton visualization
      if (skeletonGroup) {{ scene.remove(skeletonGroup); characterGroup.add(skeletonGroup); }}
      if (jointsGroup) {{ scene.remove(jointsGroup); characterGroup.add(jointsGroup); }}
      
      // Initial character rotation: face away from camera (towards -Z)
      characterYaw = 0;
      characterGroup.rotation.y = characterYaw;
      
      // Add ground grid
      const gridSize = modelHeight * 60;
      const gridDivs = 120;
      groundGrid = new THREE.GridHelper(gridSize, gridDivs, 0x444466, 0x333355);
      groundGrid.position.y = box.min.y;
      scene.add(groundGrid);
      
      // Try to find walk/idle animations
      tpAutoAnimWalk = null;
      tpAutoAnimIdle = null;
      tpCurrentAutoAnim = null;
      const animNames = Object.keys(animationClips);
      
      // Find best match for idle and walk/run
      for (const name of animNames) {{
        const lower = name.toLowerCase();
        // Idle: prefer "wait", fallback to idle/stand
        if (lower === 'wait') tpAutoAnimIdle = name;
        else if (!tpAutoAnimIdle && (lower.includes('wait') || lower.includes('idle') || lower.includes('stand'))) tpAutoAnimIdle = name;
        // Walk/Run: prefer "run" (exact), ignore stop_run
        if (lower === 'run') tpAutoAnimWalk = name;
        else if (!tpAutoAnimWalk && (lower.startsWith('run') || lower.includes('run')) && !lower.includes('stop')) tpAutoAnimWalk = name;
        else if (!tpAutoAnimWalk && (lower.includes('walk') || lower.includes('move'))) tpAutoAnimWalk = name;
      }}
      
      // Auto-play idle if found
      if (tpAutoAnimIdle && !currentAnimName) {{
        playAnimationCrossfade(tpAutoAnimIdle, 0.1);
        tpCurrentAutoAnim = tpAutoAnimIdle;
      }}
      
      // Disable mouse orbit (camera is now gamepad-controlled)
      controls._tpMode = true;
      
      // Position camera behind character
      updateThirdPersonCamera();
      
      debug('Third-person mode ON. Walk:', tpAutoAnimWalk, 'Idle:', tpAutoAnimIdle, 'Speed:', characterMoveSpeed.toFixed(4));
    }}

    function disableThirdPerson() {{
      if (!characterGroup) return;
      
      // Reparent everything back to scene
      if (bones.length > 0) {{
        characterGroup.remove(bones[0]);
        scene.add(bones[0]);
      }}
      meshes.forEach(m => {{
        characterGroup.remove(m);
        scene.add(m);
        if (m.userData.wireframeOverlay) {{
          characterGroup.remove(m.userData.wireframeOverlay);
          scene.add(m.userData.wireframeOverlay);
        }}
      }});
      if (skeletonGroup) {{ characterGroup.remove(skeletonGroup); scene.add(skeletonGroup); }}
      if (jointsGroup) {{ characterGroup.remove(jointsGroup); scene.add(jointsGroup); }}
      
      // Reset character group transform so model returns to origin
      scene.remove(characterGroup);
      characterGroup = null;
      
      // Remove grid
      if (groundGrid) {{
        scene.remove(groundGrid);
        groundGrid.geometry.dispose();
        if (Array.isArray(groundGrid.material)) groundGrid.material.forEach(m => m.dispose());
        else groundGrid.material.dispose();
        groundGrid = null;
      }}
      
      // Re-enable mouse controls
      controls._tpMode = false;
      
      tpCurrentAutoAnim = null;
      tpIsMoving = false;
      
      // Reset camera to orbit view
      fitToView();
      
      debug('Third-person mode OFF');
    }}

    function updateThirdPersonCamera() {{
      if (!characterGroup) return;
      if (Date.now() < focusLockUntil) return;  // focusMesh lock active
      
      // Camera orbits around character center
      const targetPos = characterGroup.position.clone();
      targetPos.y += characterCenterY;
      
      // Spherical to cartesian
      const x = tpCamDist * Math.sin(tpCamPhi) * Math.sin(tpCamTheta);
      const y = tpCamDist * Math.cos(tpCamPhi);
      const z = tpCamDist * Math.sin(tpCamPhi) * Math.cos(tpCamTheta);
      
      camera.position.set(targetPos.x + x, targetPos.y + y, targetPos.z + z);
      camera.lookAt(targetPos);
      
      // Keep orbit controls in sync (so overlays etc. still work)
      controls.target.copy(targetPos);
      const offset = camera.position.clone().sub(targetPos);
      controls.spherical.setFromVector3(offset);
      controls.panOffset.set(0, 0, 0);
    }}

    // Listen for gamepad connection/disconnection - ALWAYS store, even when disabled
    window.addEventListener('gamepadconnected', (e) => {{
      debug('Gamepad connected:', e.gamepad.index, e.gamepad.id);
      // Don't auto-switch lastActiveInput — wait for real button press
      gamepadStaleFrames = 0;
      gamepadLastTimestamp = 0;
      gamepadConfirmed = false;  // need to see real button press to confirm
      if (gamepadEnabled) {{
        const statusEl = document.getElementById('gamepadStatus');
        if (statusEl) {{
          statusEl.textContent = '🟡 Detecting...';
          statusEl.style.color = '#f59e0b';
        }}
      }}
    }});

    window.addEventListener('gamepaddisconnected', (e) => {{
      debug('Gamepad disconnected:', e.gamepad.index, e.gamepad.id);
      lastActiveInput = 'keyboard';  // fall back to keyboard
      gamepadStaleFrames = GAMEPAD_STALE_THRESHOLD + 1;  // force ghost state
      gamepadLastTimestamp = 0;
      gamepadConfirmed = false;
      if (gamepadEnabled) {{
        gamepadPrevButtons = [];
        const statusEl = document.getElementById('gamepadStatus');
        if (statusEl) {{
          statusEl.textContent = '⌨️ Keyboard Mode';
          statusEl.style.color = '#60a5fa';
        }}
      }}
    }});

    // Keyboard input for third-person fallback
    window.addEventListener('keydown', (e) => {{
      if (!gamepadEnabled) return;
      kbKeys[e.code] = true;
      lastActiveInput = 'keyboard';  // user is using keyboard
      e.preventDefault();
    }});
    window.addEventListener('keyup', (e) => {{
      if (!gamepadEnabled) return;
      kbKeys[e.code] = false;
    }});

    // Mouse events for controller keyboard mode (right-drag = camera, wheel = zoom)
    // Use document-level listeners since renderer may not exist yet
    document.addEventListener('mousedown', (e) => {{
      if (!gamepadEnabled || lastActiveInput !== 'keyboard') return;
      if (e.target && e.target.tagName === 'CANVAS' && e.button === 2) {{ mouseRightDown = true; e.preventDefault(); }}
    }});
    window.addEventListener('mouseup', (e) => {{
      if (e.button === 2) mouseRightDown = false;
    }});
    document.addEventListener('mousemove', (e) => {{
      if (!gamepadEnabled || lastActiveInput !== 'keyboard' || !mouseRightDown) return;
      mouseDeltaX += e.movementX;
      mouseDeltaY += e.movementY;
    }});
    document.addEventListener('wheel', (e) => {{
      if (!gamepadEnabled || lastActiveInput !== 'keyboard') return;
      if (e.target && e.target.tagName === 'CANVAS') {{
        mouseWheelDelta += e.deltaY;
        e.preventDefault();
      }}
    }}, {{ passive: false }});
    document.addEventListener('contextmenu', (e) => {{
      if (gamepadEnabled && e.target && e.target.tagName === 'CANVAS') e.preventDefault();
    }});

    // ── FreeCam camera update helper ──
    function updateFreeCamCamera() {{
      const lookDir = new THREE.Vector3(
        -Math.sin(tpCamPhi) * Math.sin(tpCamTheta),
        -Math.cos(tpCamPhi),
        -Math.sin(tpCamPhi) * Math.cos(tpCamTheta)
      ).normalize();
      const lookTarget = camera.position.clone().add(lookDir);
      camera.lookAt(lookTarget);
      controls.target.copy(lookTarget);
    }}

    function applyStick(val) {{
      return Math.abs(val) < gamepadDeadzone ? 0 : val;
    }}

    function updateGamepad() {{
      if (!gamepadEnabled) return;
      
      // Scan all gamepad slots every frame
      const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
      let gp = null;
      let bestPriority = -1;
      
      for (let i = 0; i < gamepads.length; i++) {{
        const c = gamepads[i];
        if (!c || !c.connected) continue;
        if (!c.buttons || c.buttons.length < 4 || !c.axes || c.axes.length < 2) continue;
        
        const type = detectGamepadType(c.id || '');
        const hasInput = c.axes.some(a => Math.abs(a) > 0.01) ||
                         Array.from(c.buttons).some(b => b.pressed || b.value > 0.01);
        let priority = 0;
        if (type !== 'generic') priority += 10;
        if (hasInput) priority += 5;
        if (c.timestamp > 0) priority += 1;
        
        if (priority > bestPriority) {{
          bestPriority = priority;
          gp = c;
        }}
      }}
      
      // ── Ghost detection + last-active-input switching ──
      // Key principle: only a BUTTON PRESS switches to gamepad mode.
      // Axis movement alone doesn't count (ghost BT devices can have axis drift).
      // Ghost: no button ever pressed + frozen timestamp → stay on keyboard.
      
      if (gp) {{
        // Track timestamp freshness
        if (gp.timestamp !== gamepadLastTimestamp) {{
          gamepadLastTimestamp = gp.timestamp;
          gamepadStaleFrames = 0;
        }} else {{
          gamepadStaleFrames++;
        }}
        
        // Only a real BUTTON PRESS confirms the gamepad and switches input
        const hasButtonPress = Array.from(gp.buttons).some(b => b.pressed || b.value > 0.5);
        
        if (hasButtonPress) {{
          if (!gamepadConfirmed) {{
            gamepadConfirmed = true;
            debug('Gamepad confirmed by button press');
          }}
          lastActiveInput = 'gamepad';
        }}
        
        // Once confirmed, significant axis movement also keeps gamepad active
        if (gamepadConfirmed) {{
          const hasAxisInput = gp.axes.some(a => Math.abs(a) > gamepadDeadzone);
          if (hasAxisInput) {{
            lastActiveInput = 'gamepad';
          }}
        }}
        
        // Ghost: never confirmed + timestamp frozen → force keyboard
        if (!gamepadConfirmed && gamepadStaleFrames > GAMEPAD_STALE_THRESHOLD) {{
          lastActiveInput = 'keyboard';
        }}
      }}
      
      // Use keyboard if: no gamepad at all, OR user's last input was keyboard
      const useKeyboard = (!gp || lastActiveInput === 'keyboard');
      
      // ── Keyboard path: no gamepad, or last active was keyboard ──
      if (useKeyboard) {{
        kbUseKeyboard = true;
        if (gamepadConnectedShown && !gp) {{
          gamepadConnectedShown = false;
          gamepadCurrentId = '';
        }}
        // Update status to show keyboard mode
        if (gamepadType !== 'keyboard') {{
          gamepadType = 'keyboard';
          const statusEl = document.getElementById('gamepadStatus');
          if (statusEl) {{
            statusEl.textContent = '⌨️🖱️ Keyboard+Mouse Mode' + (gp ? ' (controller idle)' : '');
            statusEl.style.color = '#60a5fa';
          }}
        }}
        
        // Build virtual axes from WASD / Arrows / QE
        const lx = (kbKeys['KeyD'] ? 1 : 0) - (kbKeys['KeyA'] ? 1 : 0);
        const ly = (kbKeys['KeyW'] ? -1 : 0) + (kbKeys['KeyS'] ? 1 : 0);  // W=forward=-1
        const rx = (kbKeys['ArrowRight'] ? 1 : 0) - (kbKeys['ArrowLeft'] ? 1 : 0);
        const ry = (kbKeys['ArrowDown'] ? 1 : 0) - (kbKeys['ArrowUp'] ? 1 : 0);
        let lt = kbKeys['KeyQ'] ? 1 : 0;  // Q = zoom out
        let rt = kbKeys['KeyE'] ? 1 : 0;  // E = zoom in
        
        // Mouse wheel → zoom
        if (mouseWheelDelta !== 0) {{
          if (mouseWheelDelta > 0) lt = Math.min(1, lt + 0.5);
          else rt = Math.min(1, rt + 0.5);
        }}
        // Capture mouse deltas for this frame, then consume
        const mDx = mouseDeltaX, mDy = mouseDeltaY;
        mouseDeltaX = 0; mouseDeltaY = 0; mouseWheelDelta = 0;
        
        // Build virtual button states from key map
        const vButtons = new Array(17).fill(false);
        for (const [code, idx] of Object.entries(KB_MAP)) {{
          if (kbKeys[code]) vButtons[idx] = true;
        }}
        
        // Store for overlay
        gamepadButtonStates = vButtons;
        gamepadAxesStates = [lx, ly, rx, ry];
        gamepadTriggerStates = [lt, rt];
        
        // Apply invert
        const invX = gamepadInvertX ? -1 : 1;
        const invY = gamepadInvertY ? -1 : 1;
        
        if (freeCamMode) {{
          // ── FreeCam: fly-like free movement ──
          // Look: right stick / arrows
          if (rx !== 0) tpCamTheta += rx * 0.03 * invX;
          if (ry !== 0) tpCamPhi = Math.max(0.1, Math.min(Math.PI - 0.1, tpCamPhi + ry * 0.02 * invY));
          // Mouse: right-drag camera look
          if (mDx !== 0) tpCamTheta += mDx * 0.003 * invX;
          if (mDy !== 0) tpCamPhi = Math.max(0.1, Math.min(Math.PI - 0.1, tpCamPhi + mDy * 0.003 * invY));
          
          const speed = freeCamBaseSpeed * freeCamSpeed;
          // Full 3D look direction (includes pitch)
          const lookDir = new THREE.Vector3(
            -Math.sin(tpCamPhi) * Math.sin(tpCamTheta),
            -Math.cos(tpCamPhi),
            -Math.sin(tpCamPhi) * Math.cos(tpCamTheta)
          ).normalize();
          // Strafe: perpendicular on XZ plane
          const strafeDir = new THREE.Vector3(lookDir.z, 0, -lookDir.x).normalize();
          // Move forward/back along look direction (fly into the view)
          if (ly !== 0) camera.position.addScaledVector(lookDir, -ly * speed * invY);
          // Strafe left/right
          if (lx !== 0) camera.position.addScaledVector(strafeDir, lx * speed * invX);
          // Up/Down in world space: triggers / QE
          if (lt > 0.05) camera.position.y -= lt * speed;
          if (rt > 0.05) camera.position.y += rt * speed;
          
          // Apply camera look direction
          const lookTarget = camera.position.clone().add(lookDir);
          camera.lookAt(lookTarget);
          controls.target.copy(lookTarget);
          
        }} else {{
        // ── Third-person character movement ──
        // Camera orbit
        if (rx !== 0) tpCamTheta += rx * 0.03 * invX;
        if (ry !== 0) tpCamPhi = Math.max(0.3, Math.min(Math.PI * 0.45, tpCamPhi + ry * 0.02 * invY));
        // Mouse: right-drag camera orbit
        if (mDx !== 0) tpCamTheta += mDx * 0.003 * invX;
        if (mDy !== 0) tpCamPhi = Math.max(0.3, Math.min(Math.PI * 0.45, tpCamPhi + mDy * 0.002 * invY));
        
        // Zoom
        if (lt > 0.05 || rt > 0.05) {{
          tpCamDist *= 1 + (lt - rt) * 0.02;
          tpCamDist = Math.max(characterMoveSpeed * 10, tpCamDist);
        }}
        
        // Movement
        const isMovingNow = (lx !== 0 || ly !== 0);
        if (isMovingNow && characterGroup) {{
          const camForward = new THREE.Vector3(-Math.sin(tpCamTheta), 0, -Math.cos(tpCamTheta)).normalize();
          const camRight = new THREE.Vector3(camForward.z, 0, -camForward.x);
          const moveDir = new THREE.Vector3();
          moveDir.addScaledVector(camRight, lx * invX);
          moveDir.addScaledVector(camForward, -ly * invY);
          if (moveDir.lengthSq() > 0) {{
            const stickMag = Math.min(1, moveDir.length());
            moveDir.normalize();
            characterGroup.position.addScaledVector(moveDir, characterMoveSpeed * stickMag);
            const targetYaw = Math.atan2(moveDir.x, moveDir.z);
            let yawDiff = targetYaw - characterYaw;
            while (yawDiff > Math.PI) yawDiff -= Math.PI * 2;
            while (yawDiff < -Math.PI) yawDiff += Math.PI * 2;
            characterYaw += yawDiff * 0.15;
            characterGroup.rotation.y = characterYaw;
          }}
        }}
        
        // Auto animation
        if (isMovingNow && !tpIsMoving) {{
          if (tpAutoAnimWalk && tpCurrentAutoAnim !== tpAutoAnimWalk) {{
            playAnimationCrossfade(tpAutoAnimWalk, 0.2);
            tpCurrentAutoAnim = tpAutoAnimWalk;
          }}
        }} else if (!isMovingNow && tpIsMoving) {{
          if (tpAutoAnimIdle && tpCurrentAutoAnim !== tpAutoAnimIdle) {{
            playAnimationCrossfade(tpAutoAnimIdle, 0.35);
            tpCurrentAutoAnim = tpAutoAnimIdle;
          }}
        }}
        tpIsMoving = isMovingNow;
        
        }} // end if freeCamMode else
        
        // Button edge detection
        function kbJustPressed(idx) {{
          return vButtons[idx] && !kbPrevKeys[idx];
        }}
        
        if (kbJustPressed(0)) {{ toggleAnimPlayback(); tpCurrentAutoAnim = null; }}
        if (kbJustPressed(1)) {{ stopAnimation(); tpCurrentAutoAnim = null; }}
        if (kbJustPressed(2)) {{ 
          if (freeCamMode) {{
            fitToView();
          }} else if (characterGroup) {{ characterGroup.position.set(0,0,0); characterYaw = 0; characterGroup.rotation.y = 0; }} }}
        if (kbJustPressed(3)) {{ toggleDynamicBones(); document.getElementById('swPhysics').checked = dynamicBonesEnabled; }}
        if (kbJustPressed(4)) {{
          const sel = document.getElementById('animation-select');
          if (sel && sel.selectedIndex > 0) {{ sel.selectedIndex--; playAnimationCrossfade(sel.value, 0.25); tpCurrentAutoAnim = null; }}
        }}
        if (kbJustPressed(5)) {{
          const sel = document.getElementById('animation-select');
          if (sel && sel.selectedIndex < sel.options.length - 1) {{ sel.selectedIndex++; playAnimationCrossfade(sel.value, 0.25); tpCurrentAutoAnim = null; }}
        }}
        if (freeCamMode) {{
          // FreeCam: D-pad with accelerating hold
          const dpadUp = vButtons[12];
          const dpadDn = vButtons[13];
          if (dpadUp || dpadDn) {{
            if (!window._fcKbHold) window._fcKbHold = 0;
            window._fcKbHold++;
            // Frame 1: immediate step. Then pause. Then slow, then accelerate.
            const h = window._fcKbHold;
            let doStep = false;
            if (h === 1) doStep = true;                    // instant first
            else if (h < 30) doStep = (h % 12 === 0);     // slow phase: ~5fps equiv
            else if (h < 90) doStep = (h % 6 === 0);      // medium: ~10fps
            else doStep = (h % 3 === 0);                   // fast: ~20fps
            if (doStep) {{
              const s = document.getElementById('freeCamSpeedSlider');
              if (s) {{
                const step = h < 90 ? 1 : 2;
                s.value = dpadUp ? Math.min(600, parseInt(s.value) + step) : Math.max(2, parseInt(s.value) - step);
                freeCamSpeed = s.value / 100;
                document.getElementById('freeCamStatus').textContent = freeCamSpeed.toFixed(2) + 'x';
              }}
            }}
          }} else {{
            window._fcKbHold = 0;
          }}
        }} else {{
        if (kbJustPressed(12)) {{
            const s = document.getElementById('animSpeedSlider'); if (s) {{ s.value = Math.min(100, parseInt(s.value) + 10); updateAnimSpeed(s.value); }}
        }}
        if (kbJustPressed(13)) {{
            const s = document.getElementById('animSpeedSlider'); if (s) {{ s.value = Math.max(-100, parseInt(s.value) - 10); updateAnimSpeed(s.value); }}
        }}
        }}
        if (kbJustPressed(9)) requestScreenshot();
        if (kbJustPressed(8)) {{ overlayMode = (overlayMode + 1) % 3; document.getElementById('overlayModeLabel').textContent = ['Off','Full','Minimal'][overlayMode]; }}
        if (kbJustPressed(11)) {{ toggleFreeCam(); document.getElementById('swFreeCam').checked = freeCamMode; }}
        if (kbJustPressed(16)) {{ fitToView(); }}
        
        // V (14) → cycle visual modes
        if (kbJustPressed(14)) {{
          let mode = 0;
          if (textureMode && !wireframeOverlayMode && !wireframeMode) mode = 0;
          else if (textureMode && wireframeOverlayMode && !wireframeMode) mode = 1;
          else if (colorMode && !wireframeOverlayMode && !wireframeMode) mode = 2;
          else if (colorMode && wireframeOverlayMode) mode = 3;
          else if (wireframeMode && !textureMode) mode = 4;
          else if (wireframeMode && textureMode) mode = 5;
          const next = (mode + 1) % 6;
          const states = [
            [false,true,false,false],[false,true,false,true],[true,false,false,false],
            [true,false,false,true],[false,false,true,false],[false,true,true,false],
          ];
          const [wC,wT,wW,wO] = states[next];
          if (colorMode!==wC) toggleColors(); if (textureMode!==wT) toggleTextures();
          if (wireframeMode!==wW) toggleWireframe(); if (wireframeOverlayMode!==wO) toggleWireframeOverlay();
          document.getElementById('swColors').checked=colorMode; document.getElementById('swTex').checked=textureMode;
          document.getElementById('swWire').checked=wireframeMode; document.getElementById('swWireOver').checked=wireframeOverlayMode;
        }}
        // B (15) → cycle skeleton modes
        if (kbJustPressed(15)) {{
          if (!showSkeleton) {{ toggleSkeleton(); document.getElementById('swSkel').checked=showSkeleton; }}
          else if (!showJoints) {{ toggleJoints(); document.getElementById('swJoints').checked=showJoints; }}
          else if (!showBoneNames) {{ toggleBoneNames(); document.getElementById('swBoneNames').checked=showBoneNames; }}
          else {{ if (showBoneNames) {{ toggleBoneNames(); document.getElementById('swBoneNames').checked=false; }}
            if (showJoints) {{ toggleJoints(); document.getElementById('swJoints').checked=false; }}
            if (showSkeleton) {{ toggleSkeleton(); document.getElementById('swSkel').checked=false; }}
          }}
        }}
        
        // Save prev states for edge detection (store by button index)
        for (let bi = 0; bi < 17; bi++) kbPrevKeys[bi] = vButtons[bi];
        
        if (!freeCamMode) updateThirdPersonCamera();
        return;
      }}
      
      // ── Real gamepad path ──
      kbUseKeyboard = false;
      
      // Detect type; update UI when type changes or first connect or ID changes
      const detected = detectGamepadType(gp.id || '');
      const idChanged = (gp.id || '') !== gamepadCurrentId;
      if (detected !== gamepadType || !gamepadConnectedShown || idChanged) {{
        gamepadType = detected;
        gamepadCurrentId = gp.id || '';
        gamepadConnectedShown = true;
        if (idChanged) gamepadPrevButtons = [];  // reset edge detection on gamepad switch
        updateGamepadStatusLabel();
        debug('Gamepad:', gamepadType, 'ID:', gp.id, 'priority:', bestPriority);
      }}
      
      // ── Sticks ──
      const lx = applyStick(gp.axes[0] || 0);
      const ly = applyStick(gp.axes[1] || 0);
      const rx = applyStick(gp.axes[2] || 0);
      const ry = applyStick(gp.axes[3] || 0);
      
      // Store states for overlay rendering
      gamepadButtonStates = gp.buttons.map(b => b.pressed);
      gamepadAxesStates = [lx, ly, rx, ry];
      gamepadTriggerStates = [gp.buttons[6] ? gp.buttons[6].value : 0, gp.buttons[7] ? gp.buttons[7].value : 0];
      
      // ── Invert axes ──
      const invX = gamepadInvertX ? -1 : 1;
      const invY = gamepadInvertY ? -1 : 1;
      
      // ── Triggers ──
      const lt = gp.buttons[6] ? gp.buttons[6].value : 0;
      const rt = gp.buttons[7] ? gp.buttons[7].value : 0;
      
      if (freeCamMode) {{
        // ── FreeCam: fly-like free movement ──
        // Look: right stick
        if (rx !== 0) tpCamTheta += rx * 0.04 * invX;
        if (ry !== 0) tpCamPhi = Math.max(0.1, Math.min(Math.PI - 0.1, tpCamPhi + ry * 0.03 * invY));
        
        const speed = freeCamBaseSpeed * freeCamSpeed;
        // Full 3D look direction (includes pitch)
        const lookDir = new THREE.Vector3(
          -Math.sin(tpCamPhi) * Math.sin(tpCamTheta),
          -Math.cos(tpCamPhi),
          -Math.sin(tpCamPhi) * Math.cos(tpCamTheta)
        ).normalize();
        // Strafe: perpendicular on XZ plane
        const strafeDir = new THREE.Vector3(lookDir.z, 0, -lookDir.x).normalize();
        // Move forward/back along look direction (fly into the view)
        if (ly !== 0) camera.position.addScaledVector(lookDir, -ly * speed * invY);
        // Strafe left/right
        if (lx !== 0) camera.position.addScaledVector(strafeDir, lx * speed * invX);
        // Up/Down in world space: triggers
        if (lt > 0.05) camera.position.y -= lt * speed;
        if (rt > 0.05) camera.position.y += rt * speed;
        
        // Apply camera look direction
        const lookTarget = camera.position.clone().add(lookDir);
        camera.lookAt(lookTarget);
        controls.target.copy(lookTarget);
        
      }} else {{
      // ── Right stick → orbit camera around character ──
      if (rx !== 0) {{
        tpCamTheta += rx * 0.04 * invX;
      }}
      if (ry !== 0) {{
        tpCamPhi = Math.max(0.3, Math.min(Math.PI * 0.45, tpCamPhi + ry * 0.03 * invY));
      }}
      
      // ── Triggers → zoom camera in/out ──
      if (lt > 0.05 || rt > 0.05) {{
        tpCamDist *= 1 + (lt - rt) * 0.02;
        tpCamDist = Math.max(characterMoveSpeed * 10, tpCamDist);
      }}
      
      // ── Left stick → move character in camera-relative direction ──
      const isMovingNow = (lx !== 0 || ly !== 0);
      
      if (isMovingNow && characterGroup) {{
        // Get camera forward direction on XZ plane
        const camForward = new THREE.Vector3(
          -Math.sin(tpCamTheta),
          0,
          -Math.cos(tpCamTheta)
        ).normalize();
        const camRight = new THREE.Vector3(camForward.z, 0, -camForward.x);
        
        // Movement vector in world space (apply invert to left stick too)
        const moveDir = new THREE.Vector3();
        moveDir.addScaledVector(camRight, lx * invX);
        moveDir.addScaledVector(camForward, -ly * invY);  // -ly because stick forward = negative
        
        if (moveDir.lengthSq() > 0) {{
          const stickMagnitude = Math.min(1, moveDir.length());
          moveDir.normalize();
          
          // Move character
          characterGroup.position.addScaledVector(moveDir, characterMoveSpeed * stickMagnitude);
          
          // Smoothly rotate character to face movement direction
          const targetYaw = Math.atan2(moveDir.x, moveDir.z);
          let yawDiff = targetYaw - characterYaw;
          // Normalize to [-PI, PI]
          while (yawDiff > Math.PI) yawDiff -= Math.PI * 2;
          while (yawDiff < -Math.PI) yawDiff += Math.PI * 2;
          characterYaw += yawDiff * 0.15;  // Smooth rotation
          characterGroup.rotation.y = characterYaw;
        }}
      }}
      
      // ── Auto animation: crossfade between run/wait ──
      if (isMovingNow && !tpIsMoving) {{
        // Started moving → crossfade to run
        if (tpAutoAnimWalk && tpCurrentAutoAnim !== tpAutoAnimWalk) {{
          playAnimationCrossfade(tpAutoAnimWalk, 0.2);
          tpCurrentAutoAnim = tpAutoAnimWalk;
        }}
      }} else if (!isMovingNow && tpIsMoving) {{
        // Stopped moving → crossfade to idle
        if (tpAutoAnimIdle && tpCurrentAutoAnim !== tpAutoAnimIdle) {{
          playAnimationCrossfade(tpAutoAnimIdle, 0.35);
          tpCurrentAutoAnim = tpAutoAnimIdle;
        }}
      }}
      tpIsMoving = isMovingNow;
      }} // end else (third-person mode)
      
      // ── Buttons (rising edge) ──
      const buttons = gp.buttons.map(b => b.pressed);
      function justPressed(idx) {{
        return idx < buttons.length && buttons[idx] && !(gamepadPrevButtons[idx]);
      }}
      
      // Ensure prevButtons array matches current length
      if (gamepadPrevButtons.length !== buttons.length) {{
        gamepadPrevButtons = new Array(buttons.length).fill(false);
      }}
      
      // A / Cross (0) → play/pause animation
      if (justPressed(0)) {{
        toggleAnimPlayback();
        tpCurrentAutoAnim = null;  // Disable auto-anim after manual control
      }}
      
      // B / Circle (1) → stop animation
      if (justPressed(1)) {{
        stopAnimation();
        tpCurrentAutoAnim = null;
      }}
      
      // X / Square (2) → reset character to origin
      if (justPressed(2)) {{
        if (freeCamMode) {{
          fitToView();
        }} else if (characterGroup) {{
          characterGroup.position.set(0, 0, 0);
          characterYaw = 0;
          characterGroup.rotation.y = 0;
        }}
      }}
      
      // Y / Triangle (3) → toggle dynamic bones
      if (justPressed(3)) {{
        toggleDynamicBones();
        document.getElementById('swPhysics').checked = dynamicBonesEnabled;
      }}
      
      // LB (4) → previous animation
      if (justPressed(4)) {{
        const sel = document.getElementById('animation-select');
        if (sel && sel.selectedIndex > 0) {{
          sel.selectedIndex--;
          playAnimationCrossfade(sel.value, 0.25);
          tpCurrentAutoAnim = null;
        }}
      }}
      
      // RB (5) → next animation
      if (justPressed(5)) {{
        const sel = document.getElementById('animation-select');
        if (sel && sel.selectedIndex < sel.options.length - 1) {{
          sel.selectedIndex++;
          playAnimationCrossfade(sel.value, 0.25);
          tpCurrentAutoAnim = null;
        }}
      }}
      
      // D-pad Up (12) / Down (13) → speed (accelerating hold in FreeCam)
      if (freeCamMode) {{
        const gpUp = buttons[12];
        const gpDn = buttons[13];
        if (gpUp || gpDn) {{
          if (!window._fcGpHold) window._fcGpHold = 0;
          window._fcGpHold++;
          const h = window._fcGpHold;
          let doStep = false;
          if (h === 1) doStep = true;
          else if (h < 30) doStep = (h % 12 === 0);
          else if (h < 90) doStep = (h % 6 === 0);
          else doStep = (h % 3 === 0);
          if (doStep) {{
            const s = document.getElementById('freeCamSpeedSlider');
            if (s) {{
              const step = h < 90 ? 1 : 2;
              s.value = gpUp ? Math.min(600, parseInt(s.value) + step) : Math.max(2, parseInt(s.value) - step);
              freeCamSpeed = s.value / 100;
              document.getElementById('freeCamStatus').textContent = freeCamSpeed.toFixed(2) + 'x';
            }}
          }}
        }} else {{
          window._fcGpHold = 0;
        }}
      }} else {{
      if (justPressed(12)) {{
          const slider = document.getElementById('animSpeedSlider');
          if (slider) {{ slider.value = Math.min(100, parseInt(slider.value) + 10); updateAnimSpeed(slider.value); }}
      }}
      if (justPressed(13)) {{
          const slider = document.getElementById('animSpeedSlider');
          if (slider) {{ slider.value = Math.max(-100, parseInt(slider.value) - 10); updateAnimSpeed(slider.value); }}
      }}
      }}
      
      // D-pad Left (14) → cycle display modes:
      // Textured → Tex+Overlay → Colors → Colors+Overlay → Wireframe → Wire+Tex → back
      if (justPressed(14)) {{
        let mode = 0;
        if (textureMode && !wireframeOverlayMode && !wireframeMode) mode = 0;
        else if (textureMode && wireframeOverlayMode && !wireframeMode) mode = 1;
        else if (colorMode && !wireframeOverlayMode && !wireframeMode) mode = 2;
        else if (colorMode && wireframeOverlayMode) mode = 3;
        else if (wireframeMode && !textureMode) mode = 4;
        else if (wireframeMode && textureMode) mode = 5;
        
        const next = (mode + 1) % 6;
        //                     [color, tex,   wire,  overlay]
        const states = [
          [false, true, false, false],   // 0: Textured
          [false, true, false, true],    // 1: Tex+Overlay
          [true, false, false, false],   // 2: Colors
          [true, false, false, true],    // 3: Colors+Overlay
          [false, false, true, false],   // 4: Wireframe
          [false, true, true, false],    // 5: Wire+Textured
        ];
        const [wantColor, wantTex, wantWire, wantOver] = states[next];
        if (colorMode !== wantColor) toggleColors();
        if (textureMode !== wantTex) toggleTextures();
        if (wireframeMode !== wantWire) toggleWireframe();
        if (wireframeOverlayMode !== wantOver) toggleWireframeOverlay();
        document.getElementById('swColors').checked = colorMode;
        document.getElementById('swTex').checked = textureMode;
        document.getElementById('swWire').checked = wireframeMode;
        document.getElementById('swWireOver').checked = wireframeOverlayMode;
      }}
      
      // D-pad Right (15) → cycle skeleton modes: OFF → Skel → +Joints → +Names → OFF
      if (justPressed(15)) {{
        if (!showSkeleton) {{
          // Step 1: skeleton ON
          if (!showSkeleton) toggleSkeleton();
          document.getElementById('swSkel').checked = showSkeleton;
        }} else if (!showJoints) {{
          // Step 2: +joints
          if (!showJoints) toggleJoints();
          document.getElementById('swJoints').checked = showJoints;
        }} else if (!showBoneNames) {{
          // Step 3: +bone names
          if (!showBoneNames) toggleBoneNames();
          document.getElementById('swBoneNames').checked = showBoneNames;
        }} else {{
          // Step 4: all OFF
          if (showBoneNames) {{ toggleBoneNames(); document.getElementById('swBoneNames').checked = false; }}
          if (showJoints) {{ toggleJoints(); document.getElementById('swJoints').checked = false; }}
          if (showSkeleton) {{ toggleSkeleton(); document.getElementById('swSkel').checked = false; }}
        }}
      }}
      
      // Start (9) → screenshot
      if (justPressed(9)) {{
        requestScreenshot();
      }}
      
      // Back / View / Share (8) → cycle info overlay mode
      if (justPressed(8)) {{
        overlayMode = (overlayMode + 1) % 3;
        document.getElementById('overlayModeLabel').textContent = ['Off','Full','Minimal'][overlayMode];
      }}
      
      // R3 (11) → toggle FreeCam
      if (justPressed(11)) {{
        toggleFreeCam();
        document.getElementById('swFreeCam').checked = freeCamMode;
      }}
      
      // L3 (10) → Focus / fit to view
      if (justPressed(10)) {{
        fitToView();
      }}
      
      gamepadPrevButtons = buttons;
      
      // ── Update third-person camera ──
      if (!freeCamMode) updateThirdPersonCamera();
    }}

    function updateStats() {{
      let totalTris = 0;
      let totalVerts = 0;
      let visibleCount = 0;
      
      meshes.forEach(m => {{
        if (m.visible) {{
          visibleCount++;
          if (m.geometry.index) {{
            totalTris += m.geometry.index.count / 3;
          }}
          totalVerts += m.geometry.attributes.position.count;
        }}
      }});
      
      const statsEl = document.getElementById('stats');
      const extended = overlayMode > 0;
      
      if (!extended) {{
        // Basic mode — compact width
        statsEl.style.minWidth = '';
        const sMode = getShaderInfoStr(true);
        statsEl.innerHTML = '<div>FPS: ' + currentFps + '</div>' +
          '<div>Triangles: ' + totalTris.toLocaleString() + '</div>' +
          '<div>Vertices: ' + totalVerts.toLocaleString() + '</div>' +
          '<div>Visible: ' + visibleCount + '/' + meshes.length + '</div>' +
          '<div style="color:#9ca3af">Shaders: ' + sMode + '</div>' +
          '<div style="color:#9ca3af">Tangents: ' + getTangentInfoStr() + '</div>';
        // Show lighting in basic overlay only if non-default
        const la = ambientLight ? ambientLight.intensity : 0.6;
        const lk = dirLight1 ? dirLight1.intensity : 0.8;
        const lf = dirLight2 ? dirLight2.intensity : 0.4;
        if (Math.abs(la-0.6)>0.01 || Math.abs(lk-0.8)>0.01 || Math.abs(lf-0.4)>0.01) {{
          statsEl.innerHTML += '<div style="color:#fbbf24">💡 A:' + la.toFixed(2) + ' K:' + lk.toFixed(2) + ' F:' + lf.toFixed(2) + '</div>';
        }}
        return;
      }}
      
      // Extended mode — wider for controller SVG (full only)
      statsEl.style.minWidth = overlayMode === 1 ? '440px' : '';
      const boneCount = bones ? bones.length : 0;
      const animCount = Object.keys(animationClips).length;
      const opacity = parseFloat(document.getElementById('meshOpacity').value);
      const isRec = mediaRecorder && mediaRecorder.state === 'recording';
      
      let animName = currentAnimName || 'None';
      let animState = '';
      if (currentAnimName) {{
        const paused = currentAnimation && currentAnimation.paused;
        const speed = animationMixer ? animationMixer.timeScale.toFixed(1) : '1.0';
        animState = (paused ? 'paused' : 'playing') + ' · ' + speed + 'x';
      }}

      function dot(on) {{ return on ? '<span style="color:#4ade80">●</span>' : '<span style="color:#555">○</span>'; }}
      
      const w = renderer.domElement.width;
      const h = renderer.domElement.height;
      
      let html = '<div style="color:#a78bfa;font-weight:bold">' + MODEL_FILENAME + '</div>';
      html += '<div style="color:#555;font-size:10px;margin-bottom:4px">' + new Date().toLocaleTimeString() + '</div>';
      html += '<div style="border-top:1px solid rgba(124,58,237,0.3);margin:3px 0"></div>';
      html += '<table style="border-collapse:collapse;font-size:inherit;color:#e0e0e0">';
      html += '<tr><td style="padding:0 6px 0 0">' + w + ' × ' + h + '</td><td style="border-left:1px solid #444;padding:0 0 0 6px">FPS: ' + currentFps + '</td></tr>';
      html += '<tr><td style="padding:0 6px 0 0">Tris: ' + totalTris.toLocaleString() + '</td><td style="border-left:1px solid #444;padding:0 0 0 6px">Verts: ' + totalVerts.toLocaleString() + '</td></tr>';
      html += '<tr><td style="padding:0 6px 0 0">Meshes: ' + visibleCount + '/' + meshes.length + '</td><td style="border-left:1px solid #444;padding:0 0 0 6px">Bones: ' + boneCount + '</td></tr>';
      html += '<tr><td style="padding:0 6px 0 0">Textures: ' + loadedTexturesCount + '/' + totalTexturesCount + '</td><td style="border-left:1px solid #444;padding:0 0 0 6px">Anims: ' + animCount + '</td></tr>';
      // Shader info row
      const shInfo = getShaderInfoStr(false);
      html += '<tr><td colspan="2" style="padding:0;color:#9ca3af">Shaders: ' + shInfo.mode + ' · ' + shInfo.types + '</td></tr>';
      html += '<tr><td colspan="2" style="padding:0;color:#9ca3af">Tangents: ' + getTangentInfoStr() + '</td></tr>';
      html += '</table>';
      html += '<div style="border-top:1px solid rgba(124,58,237,0.3);margin:3px 0"></div>';
      html += '<div>' + dot(colorMode) + ' Colors  ' + dot(textureMode) + ' Textures  ' + dot(wireframeMode) + ' Wire  ' + dot(wireframeOverlayMode) + ' Overlay' + (shaderStats.toon > 0 && !NO_SHADERS ? '  ' + dot(fxoShadersEnabled) + ' FXO' : '') + (recomputeNormalsEnabled ? '  ' + dot(true) + ' Normals' : '') + '  ' + dot(emissiveEnabled) + ' Emissive</div>';
      html += '<div>' + dot(showSkeleton) + ' Skeleton  ' + dot(showJoints) + ' Joints  ' + dot(showBoneNames) + ' Names  ' + dot(dynamicBonesEnabled) + ' Physics  ' + dot(dynCollisionsEnabled) + ' Collisions</div>';
      if (freeCamMode) {{
        html += '<div>' + dot(true) + ' FreeCam · Speed: ' + freeCamSpeed.toFixed(2) + 'x</div>';
      }}
      if (opacity < 1.0) {{
        html += '<div style="color:#9ca3af">Opacity: ' + opacity.toFixed(2) + '</div>';
      }}
      // Lighting info (show when non-default or emissive active)
      const la = ambientLight ? ambientLight.intensity : 0.6;
      const lk = dirLight1 ? dirLight1.intensity : 0.8;
      const lf = dirLight2 ? dirLight2.intensity : 0.4;
      const lightsNonDefault = Math.abs(la-0.6)>0.01 || Math.abs(lk-0.8)>0.01 || Math.abs(lf-0.4)>0.01;
      if (lightsNonDefault || emissiveEnabled) {{
        let lightStr = '💡';
        if (lightsNonDefault) {{
          lightStr += ' Ambient: ' + la.toFixed(2) + '  Key: ' + lk.toFixed(2) + '  Fill: ' + lf.toFixed(2);
        }}
        if (emissiveEnabled) {{
          lightStr += (lightsNonDefault ? '  · ' : ' ') + 'Glow: ' + (emissiveGlowOffset >= 0 ? '+' : '') + emissiveGlowOffset.toFixed(2);
        }}
        html += '<div style="color:#fbbf24">' + lightStr + '</div>';
      }}
      if (gamepadEnabled) {{
        const hasGamepad = gamepadButtonStates.length > 0;
        const tp = gamepadType;
        const labels = GP_LABELS[tp] || GP_LABELS.generic;
        const colors = GP_COLORS[tp] || {{}};
        const bs = gamepadButtonStates;
        const ax = gamepadAxesStates;
        const tr = gamepadTriggerStates;
        const typeName = GP_TYPE_NAMES[tp] || 'Gamepad';
        
        html += '<div style="border-top:1px solid rgba(124,58,237,0.3);margin:3px 0"></div>';
        html += '<div style="color:#a78bfa;font-weight:bold;text-align:center">🎮 ' + typeName + (hasGamepad ? '' : ' · Waiting') + '</div>';
        
        if (overlayMode === 1) {{
          if (tp === 'keyboard') {{
            html += renderKeyboardSVG(bs, ax, tr);
          }} else {{
            html += renderControllerSVG(tp, bs, ax, tr, labels, colors);
          }}
          html += renderMappingLegend(tp, labels);
        }}
      }}
      html += '<div style="border-top:1px solid rgba(124,58,237,0.3);margin:3px 0"></div>';
      html += '<div style="color:#60a5fa">Anim: ' + animName + '</div>';
      if (animState) {{
        html += '<div style="color:#60a5fa;padding-left:6ch">' + animState + '</div>';
      }}
      if (isRec) {{
        const recS = ((Date.now() - recordingStartTime) / 1000).toFixed(1);
        html += '<div style="color:#ef4444;font-weight:bold">🔴 REC ' + recS + 's · ' + recordingFps + 'fps</div>';
      }}
      
      statsEl.innerHTML = html;
      
      // ── Cache controller/keyboard SVG as Image for screenshot/video ──
      if (gamepadEnabled) {{
        const tp = gamepadType;
        const labels = GP_LABELS[tp] || GP_LABELS.generic;
        const colors = GP_COLORS[tp] || {{}};
        const bs = gamepadButtonStates;
        const ax = gamepadAxesStates;
        const tr = gamepadTriggerStates;
        let svgStr;
        if (tp === 'keyboard') {{
          svgStr = renderKeyboardSVG(bs, ax, tr);
        }} else {{
          svgStr = renderControllerSVG(tp, bs, ax, tr, labels, colors);
        }}
        // Only re-render image when SVG actually changed
        if (svgStr !== cachedOverlaySVGStr) {{
          cachedOverlaySVGStr = svgStr;
          const img = new Image();
          const blob = new Blob([svgStr], {{type: 'image/svg+xml'}});
          const url = URL.createObjectURL(blob);
          img.onload = function() {{ cachedOverlaySVGImg = img; URL.revokeObjectURL(url); }};
          img.onerror = function() {{ URL.revokeObjectURL(url); }};
          img.src = url;
        }}
      }} else {{
        cachedOverlaySVGImg = null;
        cachedOverlaySVGStr = '';
      }}
    }}

    function updateTextureStatus() {{
      if (totalTexturesCount === 0) {{
        document.getElementById('texture-info').textContent = 'No textures';
      }} else {{
        document.getElementById('texture-info').textContent = 
          `${{loadedTexturesCount}}/${{totalTexturesCount}} loaded`;
      }}
    }}

    function toggleControlsPanel() {{
      const panel = document.getElementById('controls');
      panel.classList.toggle('collapsed');
    }}

    let overlayPanelsHidden = false;
    function toggleOverlayPanels() {{
      overlayPanelsHidden = !overlayPanelsHidden;
      const info = document.getElementById('info');
      const stats = document.getElementById('stats');
      if (info) info.style.display = overlayPanelsHidden ? 'none' : '';
      if (stats) stats.style.display = overlayPanelsHidden ? 'none' : '';
    }}

    function requestScreenshot() {{
      const scale = parseInt(document.getElementById('screenshotScale').value) || 2;
      const w = window.innerWidth;
      const h = window.innerHeight;
      const targetW = w * scale;
      const targetH = h * scale;
      
      if (scale <= 1) {{
        // Native: just capture current frame
        renderer.render(scene, camera);
        const tmpCanvas = document.createElement('canvas');
        tmpCanvas.width = renderer.domElement.width;
        tmpCanvas.height = renderer.domElement.height;
        const tmpCtx = tmpCanvas.getContext('2d');
        drawComposite(tmpCtx, tmpCanvas.width, tmpCanvas.height);
        finishScreenshot(tmpCanvas.toDataURL('image/png'));
        return;
      }}
      
      // Save original state
      const origW = renderer.domElement.width;
      const origH = renderer.domElement.height;
      const origStyleW = renderer.domElement.style.width;
      const origStyleH = renderer.domElement.style.height;
      const gl = renderer.getContext();
      const aspect = w / h;
      
      // Try requested size, then check if GPU actually gave us that size
      let renderW = targetW;
      let renderH = targetH;
      
      renderer.setSize(renderW, renderH, false);
      
      // Check actual drawing buffer - browser may silently clamp
      let actualW = gl.drawingBufferWidth;
      let actualH = gl.drawingBufferHeight;
      
      if (actualW < renderW || actualH < renderH) {{
        // GPU clamped - use actual buffer size, preserving aspect ratio
        renderW = actualW;
        renderH = Math.round(actualW / aspect);
        if (renderH > actualH) {{
          renderH = actualH;
          renderW = Math.round(actualH * aspect);
        }}
        renderer.setSize(renderW, renderH, false);
        actualW = gl.drawingBufferWidth;
        actualH = gl.drawingBufferHeight;
      }}
      
      camera.aspect = renderW / renderH;
      camera.updateProjectionMatrix();
      
      // Update skeleton for this frame
      if (bones.length > 0) bones[0].updateMatrixWorld(true);
      if (skeleton) skeleton.update();
      
      renderer.render(scene, camera);
      
      // Composite render + overlays at actual render resolution
      const renderCanvas = document.createElement('canvas');
      renderCanvas.width = actualW;
      renderCanvas.height = actualH;
      const renderCtx = renderCanvas.getContext('2d');
      drawComposite(renderCtx, actualW, actualH);
      
      // Restore original renderer size immediately
      renderer.setSize(origW, origH, false);
      renderer.domElement.style.width = origStyleW;
      renderer.domElement.style.height = origStyleH;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.render(scene, camera);
      
      // If render was clamped, upscale to target resolution
      if (actualW >= targetW && actualH >= targetH) {{
        finishScreenshot(renderCanvas.toDataURL('image/png'));
      }} else {{
        const outCanvas = document.createElement('canvas');
        outCanvas.width = targetW;
        outCanvas.height = targetH;
        const outCtx = outCanvas.getContext('2d');
        outCtx.imageSmoothingEnabled = true;
        outCtx.imageSmoothingQuality = 'high';
        outCtx.drawImage(renderCanvas, 0, 0, targetW, targetH);
        finishScreenshot(outCanvas.toDataURL('image/png'));
      }}
    }}
    
    function finishScreenshot(dataURL) {{
      
      if (window.pywebview) {{
        window.pywebview.api.save_screenshot(dataURL).then(result => {{
          if (result.success) {{
            showScreenshotModal(result.filepath);
          }} else {{
            alert('Screenshot failed: ' + result.error);
          }}
        }});
      }} else {{
        // Fallback: browser download
        const a = document.createElement('a');
        a.href = dataURL;
        a.download = 'screenshot_' + new Date().toISOString().replace(/[:.]/g, '-').slice(0,19) + '.png';
        a.click();
      }}
    }}

    function showScreenshotModal(filepath) {{
      // Store filepath for openScreenshotFile()
      window.lastScreenshotPath = filepath;
      document.getElementById('screenshot-filename').textContent = filepath;
      document.getElementById('screenshot-modal').classList.add('show');
    }}

    function closeScreenshotModal() {{
      document.getElementById('screenshot-modal').classList.remove('show');
    }}

    function openScreenshotFile() {{
      if (window.lastScreenshotPath && window.pywebview) {{
        window.pywebview.api.open_file(window.lastScreenshotPath).then(result => {{
          if (!result.success) {{
            alert('Could not open file: ' + window.lastScreenshotPath);
          }}
        }});
      }}
    }}

    function copyFilename() {{
      // Deprecated - kept for compatibility
      const filename = document.getElementById('screenshot-filename').textContent;
      navigator.clipboard.writeText(filename).then(() => {{
        const elem = document.getElementById('screenshot-filename');
        const original = elem.textContent;
        elem.textContent = 'Copied!';
        setTimeout(() => elem.textContent = original, 1000);
      }});
    }}

    // ============================================
    // Video Recording (WebM via MediaRecorder API)
    // ============================================
    let mediaRecorder = null;
    let recordedChunks = [];
    let recordingStartTime = 0;
    let compositeCanvas = null;
    let compositeCtx = null;
    let recordingFps = 60;
    let lastFrameTime = 0;
    let recordingVideoTrack = null;

    function toggleRecording() {{
      if (mediaRecorder && mediaRecorder.state === 'recording') {{
        stopRecording();
      }} else {{
        startRecording();
      }}
    }}

    function startRecording() {{
      try {{
        // Create composite canvas for recording (WebGL + overlays)
        compositeCanvas = document.createElement('canvas');
        compositeCanvas.width = renderer.domElement.width;
        compositeCanvas.height = renderer.domElement.height;
        compositeCtx = compositeCanvas.getContext('2d');
        
        recordingFps = parseInt(document.getElementById('videoFps').value) || 60;
        const bitrate = parseInt(document.getElementById('videoQuality').value) || 8000000;
        
        // Use captureStream(0) = manual frame mode for precise FPS control
        const stream = compositeCanvas.captureStream(0);
        recordingVideoTrack = stream.getVideoTracks()[0];
        lastFrameTime = 0;
        
        const mimeTypes = [
          'video/webm;codecs=vp9',
          'video/webm;codecs=vp8',
          'video/webm'
        ];
        let selectedMime = '';
        for (const mime of mimeTypes) {{
          if (MediaRecorder.isTypeSupported(mime)) {{
            selectedMime = mime;
            break;
          }}
        }}
        
        if (!selectedMime) {{
          alert('Video recording not supported in this browser');
          return;
        }}
        
        recordedChunks = [];
        mediaRecorder = new MediaRecorder(stream, {{
          mimeType: selectedMime,
          videoBitsPerSecond: bitrate
        }});
        
        mediaRecorder.ondataavailable = (e) => {{
          if (e.data.size > 0) recordedChunks.push(e.data);
        }};
        
        mediaRecorder.onstop = () => {{
          const blob = new Blob(recordedChunks, {{ type: selectedMime }});
          saveRecording(blob);
          compositeCanvas = null;
          compositeCtx = null;
          recordingVideoTrack = null;
        }};
        
        mediaRecorder.start(100);
        recordingStartTime = Date.now();
        
        const btn = document.getElementById('btnRecord');
        if (btn) {{
          btn.classList.add('recording');
          btn.textContent = '⏹ Stop Recording';
        }}
        updateRecordTimer();
        const outFormat = document.getElementById('videoFormat').value || 'webm';
        debug('Recording started:', selectedMime, recordingFps + 'fps', (bitrate/1000000) + 'Mbps', '→', outFormat);
      }} catch (e) {{
        alert('Recording failed: ' + e.message);
      }}
    }}

    // Called each frame from animate() - throttles to recording FPS
    function updateCompositeFrame() {{
      if (!compositeCtx || !compositeCanvas || !recordingVideoTrack) return;
      
      // Throttle to recording FPS
      const now = performance.now();
      const frameInterval = 1000 / recordingFps;
      if (now - lastFrameTime < frameInterval * 0.9) return;
      lastFrameTime = now;
      
      const c = compositeCanvas;
      const ctx = compositeCtx;
      // Resize if needed
      if (c.width !== renderer.domElement.width || c.height !== renderer.domElement.height) {{
        c.width = renderer.domElement.width;
        c.height = renderer.domElement.height;
      }}
      drawComposite(ctx, c.width, c.height);
      
      // Manually push frame to the stream
      recordingVideoTrack.requestFrame();
    }}

    // Composite WebGL canvas + joints + bone names onto a 2D context
    function drawComposite(ctx, w, h) {{
      // Scale factor relative to native window size
      const sf = w / window.innerWidth || 1;
      // Draw WebGL canvas
      ctx.drawImage(renderer.domElement, 0, 0);
      
      // Draw joints if visible
      if (showJoints) {{
        for (let i = 0; i < skeletonData.length; i++) {{
          const bd = skeletonData[i];
          if (bd.type !== 1) continue;
          const bone = bones[i];
          if (!bone) continue;
          const pos = new THREE.Vector3();
          bone.getWorldPosition(pos);
          pos.project(camera);
          if (pos.z > 1) continue;
          const x = (pos.x * 0.5 + 0.5) * w;
          const y = (-pos.y * 0.5 + 0.5) * h;
          ctx.beginPath();
          ctx.arc(x, y, 3 * sf, 0, Math.PI * 2);
          ctx.fillStyle = '#ff4444';
          ctx.fill();
          ctx.lineWidth = 1 * sf;
          ctx.strokeStyle = '#aa0000';
          ctx.stroke();
        }}
      }}
      
      // Draw skeleton lines if visible
      if (showSkeleton) {{
        ctx.strokeStyle = '#00ff88';
        ctx.lineWidth = 1 * sf;
        for (let i = 0; i < skeletonData.length; i++) {{
          const bd = skeletonData[i];
          if (bd.type !== 1) continue;
          const bone = bones[i];
          if (!bone) continue;
          const parentIdx = window._boneParentMap[i];
          if (parentIdx === undefined || !bones[parentIdx]) continue;
          
          const pos = new THREE.Vector3();
          bone.getWorldPosition(pos);
          pos.project(camera);
          if (pos.z > 1) continue;
          
          const ppos = new THREE.Vector3();
          bones[parentIdx].getWorldPosition(ppos);
          ppos.project(camera);
          if (ppos.z > 1) continue;
          
          const x1 = (pos.x * 0.5 + 0.5) * w;
          const y1 = (-pos.y * 0.5 + 0.5) * h;
          const x2 = (ppos.x * 0.5 + 0.5) * w;
          const y2 = (-ppos.y * 0.5 + 0.5) * h;
          
          ctx.beginPath();
          ctx.moveTo(x2, y2);
          ctx.lineTo(x1, y1);
          ctx.stroke();
        }}
      }}
      
      // Draw bone names if visible
      if (showBoneNames) {{
        ctx.font = Math.round(9 * sf) + 'px monospace';
        ctx.textAlign = 'center';
        for (let i = 0; i < skeletonData.length; i++) {{
          const bd = skeletonData[i];
          if (bd.type !== 1) continue;
          const bone = bones[i];
          if (!bone) continue;
          const pos = new THREE.Vector3();
          bone.getWorldPosition(pos);
          pos.project(camera);
          if (pos.z > 1) continue;
          const x = (pos.x * 0.5 + 0.5) * w;
          const y = (-pos.y * 0.5 + 0.5) * h;
          // Text shadow
          ctx.fillStyle = '#000';
          ctx.fillText(bd.name, x + sf, y + sf);
          ctx.fillText(bd.name, x - sf, y - sf);
          // Text
          ctx.fillStyle = '#00ff88';
          ctx.fillText(bd.name, x, y);
        }}
      }}

      // Info overlay (if enabled)
      if (overlayMode > 0) {{
        drawInfoOverlay(ctx, w, h, sf);
      }}
      
      // Toast notification (only when overlay is active)
      if (overlayMode > 0) {{
        const toast = document.getElementById('_toast');
        if (toast && toast.style.display !== 'none' && toast.textContent) {{
          const text = toast.textContent;
          const lines = text.split('\\n');
          const fontSize = Math.round(12 * sf);
          ctx.font = fontSize + 'px system-ui, sans-serif';
          const lineH = fontSize * 1.5;
          const maxLineW = Math.max(...lines.map(l => ctx.measureText(l).width));
          const padX = 18 * sf;
          const padY = 8 * sf;
          const boxW = maxLineW + padX * 2;
          const boxH = lines.length * lineH + padY * 2;
          const bx = (w - boxW) / 2;
          const by = h - 60 * sf - boxH;
          
          ctx.save();
          ctx.fillStyle = '#0a0c14';
          ctx.strokeStyle = 'rgba(34,197,94,0.6)';
          ctx.lineWidth = 1 * sf;
          ctx.beginPath();
          const r = 10 * sf;
          ctx.moveTo(bx + r, by); ctx.lineTo(bx + boxW - r, by);
          ctx.quadraticCurveTo(bx + boxW, by, bx + boxW, by + r);
          ctx.lineTo(bx + boxW, by + boxH - r);
          ctx.quadraticCurveTo(bx + boxW, by + boxH, bx + boxW - r, by + boxH);
          ctx.lineTo(bx + r, by + boxH);
          ctx.quadraticCurveTo(bx, by + boxH, bx, by + boxH - r);
          ctx.lineTo(bx, by + r);
          ctx.quadraticCurveTo(bx, by, bx + r, by);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
          
          ctx.fillStyle = '#86efac';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          lines.forEach((line, i) => {{
            ctx.fillText(line, w / 2, by + padY + (i + 0.5) * lineH);
          }});
          ctx.restore();
        }}
      }}
    }}

    function drawInfoOverlay(ctx, w, h, sf) {{
      // Gather info
      let totalTris = 0, totalVerts = 0, visibleMeshes = 0;
      meshes.forEach(m => {{
        if (m.visible) {{
          visibleMeshes++;
          if (m.geometry.index) totalTris += m.geometry.index.count / 3;
          totalVerts += m.geometry.attributes.position.count;
        }}
      }});

      const boneCount = bones ? bones.length : 0;
      const animCount = Object.keys(animationClips).length;
      const opacity = parseFloat(document.getElementById('meshOpacity').value);
      const isRecording = mediaRecorder && mediaRecorder.state === 'recording';

      // Current animation info
      let animName = currentAnimName || 'None';
      let animState = '';
      if (currentAnimName) {{
        const paused = currentAnimation && currentAnimation.paused;
        const speed = animationMixer ? animationMixer.timeScale.toFixed(1) : '1.0';
        animState = (paused ? 'paused' : 'playing') + '  ' + speed + 'x';
      }}

      // Timestamp
      const now = new Date();
      const ts = now.getFullYear() + '-' +
        String(now.getMonth()+1).padStart(2,'0') + '-' +
        String(now.getDate()).padStart(2,'0') + ' ' +
        String(now.getHours()).padStart(2,'0') + ':' +
        String(now.getMinutes()).padStart(2,'0') + ':' +
        String(now.getSeconds()).padStart(2,'0');

      // Helper: pad string to fixed width (monospace)
      function pad(s, len) {{
        s = String(s);
        while (s.length < len) s += ' ';
        return s;
      }}

      // Build two-column rows with aligned │
      const col = 16; // left column char width
      const rows = [
        [pad(w + ' × ' + h, col) + '│  FPS: ' + currentFps, 'value'],
        [pad('Tris: ' + totalTris.toLocaleString(), col) + '│  Verts: ' + totalVerts.toLocaleString(), 'value'],
        [pad('Meshes: ' + visibleMeshes + '/' + meshes.length, col) + '│  Bones: ' + boneCount, 'value'],
        [pad('Textures: ' + loadedTexturesCount + '/' + totalTexturesCount, col) + '│  Anims: ' + animCount, 'value'],
      ];
      // Shader info
      const shInfoOvl = getShaderInfoStr(false);
      rows.push([pad('Shaders: ' + shInfoOvl.mode, col) + '│  ' + shInfoOvl.types, 'value']);
      rows.push(['Tangents: ' + getTangentInfoStr(), 'value']);

      // Build lines
      const divLen = 30;
      const div = '─'.repeat(divLen);
      const lines = [];
      lines.push([MODEL_FILENAME, 'title']);
      lines.push([ts, 'dim']);
      lines.push([div, 'divider']);
      rows.forEach(r => lines.push(r));
      lines.push([div, 'divider']);

      // Options
      const opts = [
        ['Colors', colorMode], ['Textures', textureMode],
        ['Wireframe', wireframeMode], ['Wire Overlay', wireframeOverlayMode],
      ];
      if (shaderStats.toon > 0 && !NO_SHADERS) opts.push(['FXO', fxoShadersEnabled]);
      if (recomputeNormalsEnabled) opts.push(['Normals', true]);
      opts.push(['Emissive', emissiveEnabled]);
      const optLine = opts.map(o => (o[1] ? '●' : '○') + ' ' + o[0]).join('  ');
      lines.push([optLine, 'mixed']);

      const opts2 = [
        ['Skeleton', showSkeleton], ['Joints', showJoints],
        ['Names', showBoneNames], ['Physics', dynamicBonesEnabled],
        ['Collisions', dynCollisionsEnabled],
      ];
      const optLine2 = opts2.map(o => (o[1] ? '●' : '○') + ' ' + o[0]).join('  ');
      lines.push([optLine2, 'mixed']);

      if (opacity < 1.0) {{
        lines.push(['Opacity: ' + opacity.toFixed(2), 'value']);
      }}

      // Lighting info (show when non-default or emissive active)
      const laO = ambientLight ? ambientLight.intensity : 0.6;
      const lkO = dirLight1 ? dirLight1.intensity : 0.8;
      const lfO = dirLight2 ? dirLight2.intensity : 0.4;
      const lightsNonDefaultO = Math.abs(laO-0.6)>0.01 || Math.abs(lkO-0.8)>0.01 || Math.abs(lfO-0.4)>0.01;
      if (lightsNonDefaultO || emissiveEnabled) {{
        let lightStrO = '💡';
        if (lightsNonDefaultO) {{
          lightStrO += ' A:' + laO.toFixed(2) + '  K:' + lkO.toFixed(2) + '  F:' + lfO.toFixed(2);
        }}
        if (emissiveEnabled) {{
          lightStrO += (lightsNonDefaultO ? '  · ' : ' ') + 'Glow:' + (emissiveGlowOffset >= 0 ? '+' : '') + emissiveGlowOffset.toFixed(2);
        }}
        lines.push([lightStrO, 'value']);
      }}

      if (freeCamMode) {{
        lines.push(['● FreeCam · Speed: ' + freeCamSpeed.toFixed(2) + 'x', 'value']);
      }}

      if (gamepadEnabled) {{
        const tp = gamepadType;
        const typeName = GP_TYPE_NAMES[tp] || 'Gamepad';
        const labels = GP_LABELS[tp] || GP_LABELS.generic;
        lines.push([div, 'divider']);
        lines.push(['🎮 ' + typeName, 'title']);
        // Insert cached SVG controller image only in full mode
        if (overlayMode === 1 && cachedOverlaySVGImg) {{
          lines.push([cachedOverlaySVGImg, 'svgimg']);
          // Button mapping legend (structured for badge rendering)
          const baseMappings = [[0,'Play/Pause'],[1,'Stop'],[2,'Reset Pos'],[3,'Physics'],[4,'Prev Anim'],[5,'Next Anim'],[6,'Zoom Out'],[7,'Zoom In'],[14,'Visual'],[15,'Bones'],[12,'Speed +'],[13,'Speed −'],[8,'Overlay'],[9,'Screenshot']];
          if (tp === 'keyboard') {{
            baseMappings.push([16,'Focus'],[11,'FreeCam']);
          }} else {{
            baseMappings.push([10,'Focus'],[11,'FreeCam']);
          }}
          const legendItems = baseMappings.filter(m => labels[m[0]]).map(m => ({{ key: labels[m[0]], action: m[1] }}));
          if (tp !== 'keyboard') {{ legendItems.push({{ key: 'L🕹', action: 'Move' }}); legendItems.push({{ key: 'R🕹', action: 'Camera' }}); }}
          else {{ legendItems.push({{ key: 'WASD', action: 'Move' }}); legendItems.push({{ key: '🖱️RMB', action: 'Camera' }}); legendItems.push({{ key: '⬆⬇⬅➡', action: 'Camera' }}); legendItems.push({{ key: '🖱️Whl', action: 'Zoom' }}); }}
          // Two items per row
          for (let i = 0; i < legendItems.length; i += 2) {{
            const row = [legendItems[i]];
            if (legendItems[i+1]) row.push(legendItems[i+1]);
            lines.push([row, 'legend']);
          }}
        }}
      }}

      // Animation info (always show)
      lines.push([div, 'divider']);
      lines.push(['Anim: ' + animName, 'anim']);
      if (animState) {{
        lines.push(['      ' + animState, 'anim']);
      }}

      if (isRecording) {{
        const recElapsed = ((Date.now() - recordingStartTime) / 1000).toFixed(1);
        lines.push(['🔴 REC ' + recElapsed + 's · ' + recordingFps + 'fps', 'record']);
      }}

      // Draw background box
      const fontSize = Math.round(11 * sf);
      const lineHeight = fontSize * 1.4;
      const padding = Math.round(10 * sf);
      const margin = Math.round(12 * sf);

      ctx.font = fontSize + 'px monospace';

      // Measure max width
      let maxW = 0;
      let svgImgH = 0;  // extra height for SVG controller image
      const svgImgTargetW = Math.round(280 * sf);  // target width for SVG in overlay
      
      lines.forEach(l => {{
        if (l[1] === 'svgimg' && l[0]) {{
          const img = l[0];
          const aspect = img.naturalWidth / (img.naturalHeight || 1);
          svgImgH = Math.round(svgImgTargetW / aspect);
          if (svgImgTargetW > maxW) maxW = svgImgTargetW;
        }} else if (l[1] === 'legend') {{
          // Estimate width for two badge+action pairs
          const badgeFontSize = Math.round(9 * sf);
          const row = l[0];
          let rowW = 0;
          row.forEach((item, idx) => {{
            ctx.font = badgeFontSize + 'px monospace';
            const kw = ctx.measureText(item.key).width + Math.round(8 * sf);
            ctx.font = fontSize + 'px monospace';
            const aw = ctx.measureText(' ' + item.action).width;
            rowW += kw + aw;
            if (idx < row.length - 1) rowW += Math.round(16 * sf);
          }});
          if (rowW > maxW) maxW = rowW;
        }} else {{
          const m = ctx.measureText(String(l[0]));
          if (m.width > maxW) maxW = m.width;
        }}
      }});

      const boxW = maxW + padding * 2;
      const textLines = lines.filter(l => l[1] !== 'svgimg').length;
      const boxH = textLines * lineHeight + svgImgH + padding * 2;
      const boxX = margin;
      const boxY = h - boxH - margin;

      // Semi-transparent background with rounded corners
      ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
      ctx.beginPath();
      const r = Math.round(6 * sf);
      ctx.moveTo(boxX + r, boxY);
      ctx.lineTo(boxX + boxW - r, boxY);
      ctx.quadraticCurveTo(boxX + boxW, boxY, boxX + boxW, boxY + r);
      ctx.lineTo(boxX + boxW, boxY + boxH - r);
      ctx.quadraticCurveTo(boxX + boxW, boxY + boxH, boxX + boxW - r, boxY + boxH);
      ctx.lineTo(boxX + r, boxY + boxH);
      ctx.quadraticCurveTo(boxX, boxY + boxH, boxX, boxY + boxH - r);
      ctx.lineTo(boxX, boxY + r);
      ctx.quadraticCurveTo(boxX, boxY, boxX + r, boxY);
      ctx.fill();

      // Border
      ctx.strokeStyle = 'rgba(124, 58, 237, 0.4)';
      ctx.lineWidth = 1 * sf;
      ctx.stroke();

      // Draw text
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      let ty = boxY + padding;

      const colorMap = {{
        'title': ['#a78bfa', true],
        'divider': ['rgba(124,58,237,0.4)', false],
        'value': ['#e0e0e0', false],
        'dim': ['#6b7280', false],
        'anim': ['#60a5fa', false],
        'record': ['#ef4444', true],
        'mixed': ['#d1d5db', false],
      }};

      lines.forEach(([text, type]) => {{
        if (type === 'svgimg' && text) {{
          // Draw cached controller/keyboard SVG image
          const img = text;
          const aspect = img.naturalWidth / (img.naturalHeight || 1);
          const drawW = svgImgTargetW;
          const drawH = Math.round(drawW / aspect);
          const ix = boxX + padding + (maxW - drawW) / 2;  // center in box
          ctx.drawImage(img, ix, ty, drawW, drawH);
          ty += drawH;
          return;
        }}
        
        if (type === 'legend') {{
          // Draw badge-styled button mappings
          const row = text;
          const badgeFontSize = Math.round(9 * sf);
          const badgeH = Math.round(fontSize * 1.1);
          const badgePadX = Math.round(4 * sf);
          const badgeR = Math.round(3 * sf);
          let tx = boxX + padding;
          row.forEach((item, idx) => {{
            // Draw badge background
            ctx.font = badgeFontSize + 'px monospace';
            const kw = ctx.measureText(item.key).width + badgePadX * 2;
            const bx = tx;
            const by = ty + (lineHeight - badgeH) / 2 - Math.round(1 * sf);
            ctx.fillStyle = '#252540';
            ctx.beginPath();
            ctx.moveTo(bx + badgeR, by);
            ctx.lineTo(bx + kw - badgeR, by);
            ctx.quadraticCurveTo(bx + kw, by, bx + kw, by + badgeR);
            ctx.lineTo(bx + kw, by + badgeH - badgeR);
            ctx.quadraticCurveTo(bx + kw, by + badgeH, bx + kw - badgeR, by + badgeH);
            ctx.lineTo(bx + badgeR, by + badgeH);
            ctx.quadraticCurveTo(bx, by + badgeH, bx, by + badgeH - badgeR);
            ctx.lineTo(bx, by + badgeR);
            ctx.quadraticCurveTo(bx, by, bx + badgeR, by);
            ctx.fill();
            // Badge border
            ctx.strokeStyle = '#444';
            ctx.lineWidth = 1 * sf;
            ctx.stroke();
            // Badge text
            ctx.fillStyle = '#ccc';
            ctx.font = badgeFontSize + 'px monospace';
            ctx.fillText(item.key, bx + badgePadX, by + (badgeH - badgeFontSize) / 2);
            tx += kw + Math.round(3 * sf);
            // Action text
            ctx.fillStyle = '#888';
            ctx.font = fontSize + 'px monospace';
            ctx.fillText(item.action, tx, ty);
            tx += ctx.measureText(item.action).width + Math.round(12 * sf);
          }});
          ty += lineHeight;
          return;
        }}
        
        const [color, bold] = colorMap[type] || ['#e0e0e0', false];
        ctx.fillStyle = color;
        ctx.font = (bold ? 'bold ' : '') + fontSize + 'px monospace';

        if (type === 'mixed') {{
          let tx = boxX + padding;
          const parts = text.split(/(●|○)/);
          parts.forEach(p => {{
            if (p === '●') ctx.fillStyle = '#4ade80';
            else if (p === '○') ctx.fillStyle = '#6b7280';
            else ctx.fillStyle = '#d1d5db';
            ctx.fillText(p, tx, ty);
            tx += ctx.measureText(p).width;
          }});
        }} else {{
          ctx.fillText(text, boxX + padding, ty);
        }}
        ty += lineHeight;
      }});
    }}

    function updateRecordTimer() {{
      if (!mediaRecorder || mediaRecorder.state !== 'recording') return;
      const elapsed = ((Date.now() - recordingStartTime) / 1000).toFixed(1);
      const btn = document.getElementById('btnRecord');
      if (btn) btn.textContent = '⏹ Stop (' + elapsed + 's)';
      requestAnimationFrame(updateRecordTimer);
    }}

    function stopRecording() {{
      if (mediaRecorder && mediaRecorder.state === 'recording') {{
        mediaRecorder.stop();
      }}
      const btn = document.getElementById('btnRecord');
      if (btn) {{
        btn.classList.remove('recording');
        btn.textContent = '🔴 Record Video';
      }}
    }}

    let convertingTimerInterval = null;
    function showConvertingModal(format) {{
      const title = document.getElementById('converting-title');
      const info = document.getElementById('converting-info');
      if (title) title.textContent = '⏳ Converting Video...';
      if (info) info.textContent = 'Encoding to ' + format.toUpperCase();
      document.getElementById('converting-modal').classList.add('show');
      // Start elapsed timer
      const startTime = Date.now();
      convertingTimerInterval = setInterval(() => {{
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
        if (info) info.textContent = 'Encoding to ' + format.toUpperCase() + ' (' + elapsed + 's)';
      }}, 500);
    }}

    function hideConvertingModal() {{
      document.getElementById('converting-modal').classList.remove('show');
      if (convertingTimerInterval) {{ clearInterval(convertingTimerInterval); convertingTimerInterval = null; }}
    }}

    function saveRecording(blob) {{
      const rawFormat = document.getElementById('videoFormat').value || 'webm';
      const container = rawFormat.split(':')[0]; // 'mp4:h264_nvenc' → 'mp4'
      const needsConversion = container !== 'webm';

      if (window.pywebview) {{
        // Show converting overlay immediately for non-webm
        if (needsConversion) showConvertingModal(container);

        const reader = new FileReader();
        reader.onload = () => {{
          const base64 = reader.result.split(',')[1];
          window.pywebview.api.save_video(base64, rawFormat).then(result => {{
            hideConvertingModal();
            if (result.success) {{
              showScreenshotModal(result.filepath);
              if (result.warning) DEBUG && console.warn('[Recording]', result.warning);
            }} else {{
              alert('Save failed: ' + result.error);
            }}
          }}).catch(err => {{
            hideConvertingModal();
            alert('Save failed: ' + err);
          }});
        }};
        reader.readAsDataURL(blob);
      }} else {{
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'recording_' + new Date().toISOString().replace(/[:.]/g, '-').slice(0,19) + '.' + container;
        a.click();
        URL.revokeObjectURL(url);
      }}
    }}

    function onWindowResize() {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }}

    let lastTime = Date.now();
    function animate() {{
      requestAnimationFrame(animate);
      const delta = clock.getDelta();
      
      // RESTORE dynamic bones to clean state BEFORE mixer runs
      // This undoes Phase 3 modifications from last frame.
      // Then mixer overwrites animated bones with current frame values.
      // Trackless bones stay at their clean (bind/rest) state.
      if (dynamicBonesEnabled && dynChains.length > 0) {{
        dynChains.forEach(chain => {{
          chain.bones.forEach((b, i) => {{
            b.quaternion.copy(chain.savedLocalQuat[i]);
          }});
        }});
      }}
      
      // Update animation mixer
      if (animationMixer) {{
        animationMixer.update(delta);
        updateTimeline();
      }}
      
      // Update bone world matrices (ALL bones now in clean animated state)
      if (bones.length > 0) {{
        bones[0].updateMatrixWorld(true);
      }}
      
      // Dynamic bones physics (after clean animation, before skinning)
      // Works both with and without animation (gravity affects static pose too)
      if (dynamicBonesEnabled) {{
        updateDynamicBones(delta);
        if (bones.length > 0) {{
          bones[0].updateMatrixWorld(true);
        }}
      }}
      
      if (skeleton) {{
        skeleton.update();
      }}
      
      // Update skeleton visualization
      if (showSkeleton || showJoints) {{
        updateSkeletonVis();
      }}
      if (showBoneNames) {{
        updateBoneLabels();
      }}
      
      updateGamepad();
      if (freeCamMode) {{
        updateFreeCamCamera();
      }} else {{
        controls.update();
      }}
      // Update water shader time uniforms
      meshes.forEach(m => {{
        if (m.material && m.material.userData.isWaterMaterial && m.material.userData._shader) {{
          m.material.userData._shader.uniforms.uTime.value = clock.elapsedTime;
        }}
      }});

      renderer.render(scene, camera);
      
      // Composite frame for video recording (WebGL + bone names overlay)
      if (mediaRecorder && mediaRecorder.state === 'recording') {{
        updateCompositeFrame();
      }}
      
      // Update stats (FPS, triangles, etc)
      const now = Date.now();
      const fps = Math.round(1000 / (now - lastTime));
      currentFps = fps;
      lastTime = now;
      updateStats();
    }}

    // Populate video format dropdown from Python API
    let videoFormatsLoaded = false;
    function populateVideoFormats() {{
      if (videoFormatsLoaded) return;
      if (!window.pywebview || !window.pywebview.api || !window.pywebview.api.get_video_formats) {{
        // Retry until pywebview API is fully ready
        if (!window._vfRetries) window._vfRetries = 0;
        window._vfRetries++;
        if (window._vfRetries < 50) {{ // try for 10 seconds
          setTimeout(populateVideoFormats, 200);
        }} else {{
          debug('pywebview API not available for video formats');
        }}
        return;
      }}
      window.pywebview.api.get_video_formats().then(result => {{
        const sel = document.getElementById('videoFormat');
        if (!sel || !result || !result.formats) return;
        videoFormatsLoaded = true;
        sel.innerHTML = '';
        result.formats.forEach((f, i) => {{
          const opt = document.createElement('option');
          opt.value = f.value;
          opt.textContent = f.label;
          if (i === 0) opt.selected = true;
          sel.appendChild(opt);
        }});
        debug('Video formats:', result.formats.map(f => f.label).join(', '));
      }}).catch(err => {{
        debug('Could not load video formats:', err);
      }});
    }}
    setTimeout(populateVideoFormats, 500);
    // Also listen for pywebview ready event
    window.addEventListener('pywebviewready', populateVideoFormats);

    init();
  </script>
</body>
</html>
"""
    
    return html_content


# -----------------------------
# API Class for pywebview
# -----------------------------
class API:
    def __init__(self, screenshot_dir: str, temp_dir: str):
        # Store as string to avoid pywebview serialization issues with Path objects
        self.screenshot_dir_str = screenshot_dir
        self.temp_dir_str = temp_dir
        # Ensure directory exists
        Path(screenshot_dir).mkdir(parents=True, exist_ok=True)
        self.screenshot_counter = 0
        self.video_counter = 0
        
        # Detect available video codecs
        self.available_codecs = []
        self._detect_codecs()

    def _detect_codecs(self):
        """Detect available video encoders via PyAV."""
        try:
            import av
            import tempfile, os
            
            # Test codecs by actually encoding a tiny video
            codecs_to_check = [
                ('libx264', 'H.264 (x264)'),
                ('libopenh264', 'H.264 (OpenH264)'),
                ('h264_nvenc', 'H.264 (NVIDIA NVENC)'),
                ('h264_amf', 'H.264 (AMD AMF)'),
                ('h264_qsv', 'H.264 (Intel QSV)'),
                ('h264_mf', 'H.264 (Media Foundation)'),
                ('mpeg4', 'MPEG-4'),
            ]
            
            for codec_name, label in codecs_to_check:
                tmp_path = None
                try:
                    # Check if codec exists first
                    try:
                        av.Codec(codec_name, 'w')
                    except Exception:
                        continue
                    
                    # Try encoding 1 frame into a real container
                    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.mp4')
                    os.close(tmp_fd)
                    
                    out = av.open(tmp_path, mode='w')
                    stream = out.add_stream(codec_name, rate=30)
                    stream.width = 64
                    stream.height = 64
                    stream.pix_fmt = 'yuv420p'
                    
                    frame = av.VideoFrame(64, 64, 'yuv420p')
                    for pkt in stream.encode(frame):
                        out.mux(pkt)
                    for pkt in stream.encode():
                        out.mux(pkt)
                    out.close()
                    
                    self.available_codecs.append({'codec': codec_name, 'label': label})
                except Exception:
                    pass
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass
            
            names = [c['codec'] for c in self.available_codecs]
            print(f"[+] PyAV video codecs: {', '.join(names) if names else 'none'}")
        except ImportError:
            print(f"[+] PyAV not installed (MP4/MKV export unavailable)")

    def get_video_formats(self) -> dict:
        """Return available video formats for the UI."""
        formats = [{'value': 'webm', 'label': 'WebM (VP9)', 'codec': 'vp9'}]
        
        if self.available_codecs:
            # Find best H.264 codec for MP4/MKV
            best = self.available_codecs[0]
            formats.append({'value': 'mp4', 'label': f'MP4 ({best["label"]})', 'codec': best['codec']})
            formats.append({'value': 'mkv', 'label': f'MKV ({best["label"]})', 'codec': best['codec']})
            
            # Add additional codec options for MP4 if multiple available
            for c in self.available_codecs[1:]:
                formats.append({'value': f'mp4:{c["codec"]}', 'label': f'MP4 ({c["label"]})', 'codec': c['codec']})
        
        return {"formats": formats}

    def save_screenshot(self, image_data: str) -> dict:
        """Save screenshot from base64 data URL."""
        try:
            import base64
            import re
            
            # Extract base64 data
            match = re.match(r'data:image/png;base64,(.+)', image_data)
            if not match:
                return {"success": False, "error": "Invalid data URL format"}
            
            img_data = base64.b64decode(match.group(1))
            
            # Generate filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.screenshot_counter += 1
            filename = f"screenshot_{timestamp}_{self.screenshot_counter:03d}.png"
            filepath = Path(self.screenshot_dir_str) / filename
            
            # Save file
            with open(filepath, 'wb') as f:
                f.write(img_data)
            
            print(f"[Screenshot] Saved: {filepath}")
            return {"success": True, "filepath": str(filepath.absolute())}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def save_video(self, base64_data: str, format: str = 'webm') -> dict:
        """Save video recording. Converts from WebM to MP4/MKV via PyAV if needed."""
        try:
            import base64
            
            video_data = base64.b64decode(base64_data)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.video_counter += 1
            
            # Parse format - can be 'webm', 'mp4', 'mkv', or 'mp4:h264_nvenc' etc.
            forced_codec = None
            if ':' in format:
                format, forced_codec = format.split(':', 1)
            
            if format not in ('webm', 'mp4', 'mkv'):
                format = 'webm'
            
            if format == 'webm':
                # Direct save, no conversion needed
                filename = f"recording_{timestamp}_{self.video_counter:03d}.webm"
                filepath = Path(self.screenshot_dir_str) / filename
                with open(filepath, 'wb') as f:
                    f.write(video_data)
                print(f"[Recording] Saved: {filepath} ({len(video_data) / 1024 / 1024:.1f} MB)")
                return {"success": True, "filepath": str(filepath.absolute())}
            
            # MP4/MKV: convert via PyAV
            try:
                import av
            except ImportError:
                return {"success": False, "error": "PyAV not installed. Run: pip install av"}
            
            # Save WebM to temp file in script's temp directory
            tmp_webm = Path(self.temp_dir_str) / f"_tmp_{timestamp}.webm"
            with open(tmp_webm, 'wb') as f:
                f.write(video_data)
            
            filename = f"recording_{timestamp}_{self.video_counter:03d}.{format}"
            filepath = Path(self.screenshot_dir_str) / filename
            
            try:
                input_container = av.open(str(tmp_webm))
                input_video = input_container.streams.video[0]
                
                # Use forced codec, or best available from startup detection
                if forced_codec:
                    codec = forced_codec
                elif self.available_codecs:
                    codec = self.available_codecs[0]['codec']
                else:
                    codec = 'mpeg4'
                
                print(f"[Recording] Encoding with: {codec}")
                
                pix_fmt = 'yuv420p'
                
                output_container = av.open(str(filepath), mode='w')
                output_stream = output_container.add_stream(codec, rate=input_video.average_rate or 60)
                # H.264 requires even dimensions
                out_w = input_video.width if input_video.width % 2 == 0 else input_video.width - 1
                out_h = input_video.height if input_video.height % 2 == 0 else input_video.height - 1
                output_stream.width = out_w
                output_stream.height = out_h
                output_stream.pix_fmt = pix_fmt
                out_fps = int(input_video.average_rate or 60)
                output_stream.time_base = fractions.Fraction(1, out_fps)
                # Quality: CRF-like via bit_rate
                if input_video.bit_rate:
                    output_stream.bit_rate = input_video.bit_rate
                else:
                    output_stream.bit_rate = 8_000_000
                
                # Copy audio if present
                input_audio = None
                output_audio = None
                if len(input_container.streams.audio) > 0:
                    input_audio = input_container.streams.audio[0]
                    output_audio = output_container.add_stream('aac', rate=input_audio.rate or 44100)
                
                frame_count = 0
                for packet in input_container.demux():
                    if packet.stream == input_video:
                        for frame in packet.decode():
                            # Reformat to target pix_fmt and dimensions
                            out_frame = frame.reformat(width=out_w, height=out_h, format=pix_fmt)
                            # Reset pts to ensure monotonic increase
                            out_frame.pts = frame_count
                            out_frame.time_base = fractions.Fraction(1, out_fps)
                            frame_count += 1
                            for out_packet in output_stream.encode(out_frame):
                                output_container.mux(out_packet)
                    elif input_audio and packet.stream == input_audio:
                        for frame in packet.decode():
                            for out_packet in output_audio.encode(frame):
                                output_container.mux(out_packet)
                
                # Flush encoders
                for out_packet in output_stream.encode():
                    output_container.mux(out_packet)
                if output_audio:
                    for out_packet in output_audio.encode():
                        output_container.mux(out_packet)
                
                output_container.close()
                input_container.close()
                
                file_size = filepath.stat().st_size / 1024 / 1024
                print(f"[Recording] Converted to {format.upper()}: {filepath} ({file_size:.1f} MB, {frame_count} frames)")
                return {"success": True, "filepath": str(filepath.absolute())}
                
            except Exception as conv_err:
                # Close any open containers before moving files
                for c in ('input_container', 'output_container'):
                    try:
                        obj = locals().get(c)
                        if obj: obj.close()
                    except Exception:
                        pass
                # Conversion failed - copy WebM to downloads as fallback
                import shutil
                fallback = Path(self.screenshot_dir_str) / f"recording_{timestamp}_{self.video_counter:03d}.webm"
                shutil.move(str(tmp_webm), str(fallback))
                print(f"[Recording] Conversion failed ({conv_err}), saved as WebM fallback: {fallback}")
                return {"success": True, "filepath": str(fallback.absolute()), 
                        "warning": f"Conversion to {format} failed: {conv_err}. Saved as WebM."}
            finally:
                try:
                    if tmp_webm.exists():
                        tmp_webm.unlink()
                except Exception:
                    pass
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def open_file(self, filepath: str) -> dict:
        """Open file in system default application."""
        try:
            import subprocess
            import platform
            
            filepath = Path(filepath)
            if not filepath.exists():
                return {"success": False, "error": "File not found"}
            
            system = platform.system()
            if system == 'Windows':
                subprocess.run(['start', '', str(filepath)], shell=True)
            elif system == 'Darwin':  # macOS
                subprocess.run(['open', str(filepath)])
            else:  # Linux
                subprocess.run(['xdg-open', str(filepath)])
            
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}


# -----------------------------
# Main function
# -----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python viewer_mdl_textured.py <path_to_model.mdl> [--recompute-normals] [--debug] [--skip-popup] [--no-shaders]")
        sys.exit(1)

    mdl_path = Path(sys.argv[1])
    recompute_normals = '--recompute-normals' in sys.argv
    debug_mode = '--debug' in sys.argv
    skip_popup = '--skip-popup' in sys.argv
    no_shaders = '--no-shaders' in sys.argv
    
    if debug_mode:
        print("[DEBUG MODE] Verbose console logging enabled")

    if not mdl_path.exists():
        print(f"Error: File not found: {mdl_path}")
        sys.exit(1)

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="mdl_viewer_"))
    TEMP_FILES.append(temp_dir)
    print(f"[+] Created temp directory: {temp_dir}")

    # Copy three.min.js to temp directory
    # Get absolute path to script directory
    script_dir = Path(__file__).resolve().parent
    
    # Try multiple locations
    three_js_locations = [
        script_dir / "three_min.js",           # Same directory as script
        script_dir / "three.min.js",           # Alternative name
        Path.cwd() / "three_min.js",           # Current working directory
        Path.cwd() / "three.min.js",           # Alternative name in CWD
    ]
    
    three_js_source = None
    for loc in three_js_locations:
        if loc.exists():
            three_js_source = loc
            break
    
    if three_js_source:
        three_js_dest = temp_dir / "three.min.js"
        shutil.copy(three_js_source, three_js_dest)
        print(f"[+] Copied {three_js_source.name} to temp directory")
    else:
        print("[!] Warning: three_min.js not found!")
        print(f"    Searched in:")
        print(f"      {script_dir}")
        print(f"      {Path.cwd()}")

    # Load MDL with skeleton and model info
    meshes, material_texture_map, skeleton_data, model_info, bind_matrices = load_mdl_with_textures(
        mdl_path, temp_dir, recompute_normals, no_shaders
    )

    if not meshes:
        print("[!] No meshes loaded!")
        sys.exit(1)

    # Load animations from _m_*.mdl files in same directory
    animations_data = []
    if skeleton_data:
        animations_data = load_animations_from_directory(mdl_path, skeleton_data)

    # Generate HTML
    html_content = generate_html_with_skeleton(
        mdl_path, meshes, material_texture_map, skeleton_data, model_info, debug_mode, bind_matrices,
        animations_data=animations_data, skip_popup=skip_popup, no_shaders=no_shaders,
        recompute_normals=recompute_normals
    )

    # Save HTML to temp
    html_path = temp_dir / "viewer.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"[+] Generated HTML: {html_path}")

    # Launch viewer
    try:
        import webview
        
        # Create screenshots directory in Downloads
        screenshots_dir = Path.home() / "Downloads"
        screenshots_dir.mkdir(exist_ok=True)
        
        # Create API instance
        api = API(str(screenshots_dir), str(temp_dir))
        
        print(f"\n[+] Launching viewer...")
        print(f"[+] Screenshots will be saved to: {screenshots_dir.absolute()}")
        print(f"{'='*60}\n")
        
        window = webview.create_window(
            title=f"MDL Viewer - {mdl_path.name}",
            url=str(html_path),
            width=1400,
            height=900,
            resizable=True,
            maximized=True,
            js_api=api
        )
        
        webview.start(debug=debug_mode)
        
    except ImportError:
        print("\n[!] pywebview not installed.")
        print(f"[+] HTML saved to: {html_path}")
        print(f"[+] Open this file in a web browser to view the model")


if __name__ == "__main__":
    main()
