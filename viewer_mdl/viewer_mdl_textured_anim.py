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
  python viewer_mdl_textured_anim.py /path/to/model.mdl [--recompute-normals] [--debug] [--skip-popup]
  
  --recompute-normals  Recompute smooth normals instead of using originals from MDL
                       (slower loading, typically no visual difference)
  --debug              Enable verbose console logging in browser
  --skip-popup         Skip loading progress popup on startup
"""

from pathlib import Path
import sys
import json
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
def load_mdl_with_textures(mdl_path: Path, temp_dir: Path, recompute_normals: bool = False):
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
            
            if mat_textures:
                material_texture_map[mat_name] = mat_textures
        
        loaded_count = sum(1 for v in texture_success.values() if v is not None)
        print(f"\n[OK] Loaded and converted {loaded_count}/{len(texture_success)} textures")
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

            material_name = None
            if j < len(primitives):
                material_name = primitives[j].get("material")

            mesh_data = {
                "name": f"{i}_{base_name}_{j:02d}",
                "vertices": vertices,
                "normals": normals,
                "uvs": uvs,
                "indices": indices,
                "material": material_name,
                "skin_weights": skin_weights,
                "skin_indices": skin_indices
            }

            meshes.append(mesh_data)

    print(f"[OK] Loaded {len(meshes)} submeshes")
    print(f"{'='*60}\n")

    return meshes, material_texture_map, skeleton_data, model_info, global_bind_matrices


# -----------------------------
# Generate HTML with skeleton support
# -----------------------------
def generate_html_with_skeleton(mdl_path: Path, meshes: list, material_texture_map: dict, 
                                skeleton_data: dict, model_info: dict, debug_mode: bool = False,
                                bind_matrices: dict = None, animations_data: list = None,
                                skip_popup: bool = False) -> str:
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
    #stats {{ bottom: 20px; left: 20px; font-family: monospace; font-size: 12px; }}
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
    </div>

    <div class="slider-row">
      <span class="info-text" style="min-width:52px;">Opacity:</span>
      <input type="range" id="meshOpacity" min="0" max="1" step="0.05" value="1"
             style="flex:1;" oninput="setMeshOpacity(this.value); document.getElementById('meshOpVal').textContent=parseFloat(this.value).toFixed(2)">
      <span id="meshOpVal" class="info-text" style="min-width:28px;text-align:right;color:#a78bfa;">1</span>
    </div>

    <button class="btn-action" onclick="resetView()">🔄 Reset Camera</button>
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
      <div class="toggle-row" onclick="document.getElementById('swInfoOverlay').checked = !document.getElementById('swInfoOverlay').checked;">
        <span class="label">📊 Info Overlay</span>
        <label class="toggle-switch" onclick="event.stopPropagation()">
          <input type="checkbox" id="swInfoOverlay">
          <span class="slider"></span>
        </label>
      </div>
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
          <div class="toggle-row" onclick="toggleDynamicBones(); document.getElementById('swDynBones').checked = dynamicBonesEnabled;">
            <span class="label">⚡ Dynamic Bones</span>
            <label class="toggle-switch" onclick="event.stopPropagation()">
              <input type="checkbox" id="swDynBones" onchange="toggleDynamicBones()">
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
    
    // Helper function for conditional logging
    function debug(...args) {{
      if (DEBUG) {{
        console.log(...args);
      }}
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

    let colorMode = false;
    let textureMode = true;
    let wireframeMode = false;
    let wireframeOverlayMode = false;
    let showSkeleton = false, showJoints = false, showBoneNames = false;
    let currentFps = 0;
    const MODEL_FILENAME = {json.dumps(mdl_path.name)};
    
    // Gamepad controller - third-person mode
    let gamepadEnabled = false;
    let gamepadIndex = -1;
    let gamepadPrevButtons = [];
    let gamepadDeadzone = 0.15;
    let gamepadType = 'xbox';  // 'xbox', 'playstation', 'generic'
    let gamepadButtonStates = [];  // current frame pressed states
    let gamepadAxesStates = [0,0,0,0];  // current axes
    let gamepadTriggerStates = [0,0];  // LT, RT values
    let gamepadInvertX = true;   // X camera invert ON by default
    let gamepadInvertY = false;  // Y camera invert OFF by default

    // Button labels per controller type
    const GP_LABELS = {{
      xbox: {{
        0: 'A', 1: 'B', 2: 'X', 3: 'Y',
        4: 'LB', 5: 'RB', 6: 'LT', 7: 'RT',
        8: 'Back', 9: 'Start', 10: 'LS', 11: 'RS',
        12: '↑', 13: '↓', 14: '←', 15: '→',
      }},
      playstation: {{
        0: '✕', 1: '○', 2: '□', 3: '△',
        4: 'L1', 5: 'R1', 6: 'L2', 7: 'R2',
        8: 'Share', 9: 'Opt', 10: 'L3', 11: 'R3',
        12: '↑', 13: '↓', 14: '←', 15: '→',
      }},
      generic: {{
        0: '1', 1: '2', 2: '3', 3: '4',
        4: 'L1', 5: 'R1', 6: 'L2', 7: 'R2',
        8: 'Sel', 9: 'Start', 10: 'L3', 11: 'R3',
        12: '↑', 13: '↓', 14: '←', 15: '→',
      }}
    }};

    // Button colors per type (for face buttons only)
    const GP_COLORS = {{
      xbox: {{ 0: '#3ddc84', 1: '#f44336', 2: '#2196f3', 3: '#ffc107' }},
      playstation: {{ 0: '#5c9dff', 1: '#f44336', 2: '#e991d0', 3: '#4cdfad' }},
      generic: {{}}
    }};

    // Action mapping (button index → description)
    const GP_ACTIONS = {{
      0: 'Play/Pause', 1: 'Stop', 2: 'Reset Pos', 3: 'DynBones',
      4: 'Prev Anim', 5: 'Next Anim', 6: 'Zoom Out', 7: 'Zoom In',
      9: 'Screenshot', 12: 'Speed+', 13: 'Speed-', 14: 'Wireframe', 15: 'Skeleton'
    }};
    const GP_STICK_ACTIONS = {{ ls: 'Move', rs: 'Camera' }};
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
        if (this._tpMode) return;  // Disabled in third-person mode
        this.isMouseDown = true;
        this.mouseButton = e.button;
      }}
      
      onMouseUp() {{
        this.isMouseDown = false;
      }}
      
      onMouseMove(e) {{
        if (!this.isMouseDown || this._tpMode) return;
        
        if (this.mouseButton === this.mouseButtons.LEFT) {{
          const dx = e.movementX * this.rotateSpeed * 0.01;
          const dy = e.movementY * this.rotateSpeed * 0.01;
          this.sphericalDelta.theta -= dx;
          this.sphericalDelta.phi -= dy;
        }} else if (this.mouseButton === this.mouseButtons.RIGHT) {{
          const cam = this.camera;
          const right = new THREE.Vector3(cam.matrix.elements[0], cam.matrix.elements[1], cam.matrix.elements[2]);
          const up = new THREE.Vector3(cam.matrix.elements[4], cam.matrix.elements[5], cam.matrix.elements[6]);
          this.panOffset.add(right.multiplyScalar(-e.movementX * this.panSpeed * 0.0008));
          this.panOffset.add(up.multiplyScalar(e.movementY * this.panSpeed * 0.0008));
        }}
      }}
      
      onMouseWheel(e) {{
        if (this._tpMode) return;  // Disabled in third-person mode
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
          console.error('Error loading texture:', url, error);
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

      return new THREE.MeshStandardMaterial(matParams);
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

      const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
      scene.add(ambientLight);
      
      const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
      dirLight1.position.set(5, 10, 7);
      scene.add(dirLight1);
      
      const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
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
        
        if (meshData.uvs) {{
          const uvs = new Float32Array(meshData.uvs);
          geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
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
              console.warn('⚠️ INVALID BONE INDICES:', outOfRange.length, 'indices > ', maxBoneIndex);
              console.warn('  Sample invalid indices:', outOfRange.slice(0, 10));
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
        
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
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
          console.log('✅ SkinnedMesh bound:', meshData.name,
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
            console.warn('Mesh has skinning but NO skeleton!', meshData.name);
          }}
        }}
        
        mesh.userData.meshName = meshData.name;
        mesh.userData.materialName = meshData.material;
        mesh.userData.originalColor = colors[idx % colors.length];
        mesh.userData.hasTexture = !!meshData.material && !!materials[meshData.material];
        mesh.userData.hasSkinning = hasSkinning;
        
        const hideKeywords = ['shadow', 'kage'];
        if (CONFIG.AUTO_HIDE_SHADOW && 
            hideKeywords.some(keyword => meshData.name.toLowerCase().includes(keyword))) {{
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
      
      controls.update();
    }}

    function loadSkeleton() {{
      if (!skeletonData || !Array.isArray(skeletonData) || skeletonData.length === 0) {{
        document.getElementById('skeleton-info').textContent = 'No skeleton data';
        return;
      }}

      console.log('=== LOADING SKELETON ===');
      console.log('Bones:', skeletonData.length);
      
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
      console.log('Skeleton created with', skeleton.bones.length, 'bones');
      console.log('Inverse bind matrices: calculated from Three.js bone world matrices');
      
      // STEP 5: Diagnostic - verify bind pose + compare with MDL matrices
      if (DEBUG) {{
        skeleton.update();
        console.log('=== BIND POSE VERIFICATION ===');
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
          console.log('  Bone[' + i + '] ' + bones[i].name + ': ~identity? ' + isId);
        }}
        console.log('All first 10 bones identity at bind pose:', allIdentity);
        
        // Compare Three.js world matrices with MDL bind matrices
        if (bindMatricesData) {{
          console.log('=== THREE.JS vs MDL MATRIX COMPARISON ===');
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
              console.log('  ' + name + ': posDiff=' + posDiff.toFixed(6) + 
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

      console.log('=== SKELETON READY ===');
      
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
            console.log('Built', Object.keys(animationClips).length, 'animation clips');
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
      
      console.log('Built', Object.keys(animationClips).length, 'animation clips');
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
      if (!clip) {{ console.warn('Clip not found:', animName); return; }}

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
      const sw = document.getElementById('swDynBones'); if (sw) sw.checked = dynamicBonesEnabled;
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
        
        if (mesh.userData.hasTexture) {{
          const indicator = document.createElement('span');
          indicator.className = 'texture-indicator';
          indicator.title = 'Has texture';
          label.appendChild(indicator);
        }}
        
        div.appendChild(checkbox);
        div.appendChild(label);
        list.appendChild(div);
      }});
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
      updateStats();
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

    function resetView() {{
      // Disable gamepad if active
      if (gamepadEnabled) {{
        gamepadEnabled = false;
        disableThirdPerson();
        gamepadIndex = -1;
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
        const hideKeywords = ['shadow', 'kage'];
        meshes.forEach((m, idx) => {{
          if (hideKeywords.some(kw => m.userData.meshName.toLowerCase().includes(kw))) {{
            m.visible = false;
            const cb = document.getElementById(`mesh-${{idx}}`);
            if (cb) cb.checked = false;
          }}
        }});
      }}

      // Reset camera
      const box = new THREE.Box3();
      meshes.filter(m => m.visible).forEach(m => box.expandByObject(m));
      
      if (box.isEmpty()) {{
        meshes.forEach(m => box.expandByObject(m));
      }}
      
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      const fov = camera.fov * (Math.PI / 180);
      const dist = (maxDim / (2 * Math.tan(fov / 2))) * CONFIG.CAMERA_ZOOM;
      
      const direction = new THREE.Vector3(0.5, 0.5, 1).normalize();
      camera.position.copy(center).add(direction.multiplyScalar(dist));
      camera.lookAt(center);
      
      controls.target.copy(center);
      const offset = camera.position.clone().sub(center);
      controls.spherical.setFromVector3(offset);
      controls.panOffset.set(0, 0, 0);
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
        // Try to find a connected gamepad
        const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
        let found = false;
        for (let i = 0; i < gamepads.length; i++) {{
          if (gamepads[i] && gamepads[i].connected) {{
            gamepadIndex = i;
            gamepadPrevButtons = new Array(gamepads[i].buttons.length).fill(false);
            found = true;
            break;
          }}
        }}
        updateGamepadStatusLabel();
        if (!found && statusEl) {{
          statusEl.textContent = '⏳ Waiting for controller...';
          statusEl.style.color = '#f59e0b';
        }}
      }} else {{
        disableThirdPerson();
        gamepadIndex = -1;
        if (submenu) submenu.style.display = 'none';
        if (statusEl) {{ statusEl.textContent = ''; }}
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

    function updateGamepadStatusLabel() {{
      const statusEl = document.getElementById('gamepadStatus');
      if (!statusEl || !gamepadEnabled) return;
      const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
      const gp = (gamepadIndex >= 0) ? gamepads[gamepadIndex] : null;
      if (gp && gp.connected) {{
        const id = gp.id.toLowerCase();
        if (id.includes('xbox') || id.includes('045e') || id.includes('xinput')) {{
          gamepadType = 'xbox';
          statusEl.textContent = '🟢 Xbox Controller';
        }} else if (id.includes('dualsense') || id.includes('dualshock') || id.includes('054c') || id.includes('playstation')) {{
          gamepadType = 'playstation';
          statusEl.textContent = '🟢 PlayStation Controller';
        }} else {{
          gamepadType = 'generic';
          statusEl.textContent = '🟢 Gamepad Connected';
        }}
        statusEl.style.color = '#4ade80';
      }}
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
      const box = new THREE.Box3();
      meshes.filter(m => m.visible).forEach(m => box.expandByObject(m));
      if (box.isEmpty()) meshes.forEach(m => box.expandByObject(m));
      if (!box.isEmpty()) {{
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const fov = camera.fov * (Math.PI / 180);
        const dist = (maxDim / (2 * Math.tan(fov / 2))) * CONFIG.CAMERA_ZOOM;
        const direction = new THREE.Vector3(0.5, 0.5, 1).normalize();
        camera.position.copy(center).add(direction.multiplyScalar(dist));
        camera.lookAt(center);
        controls.target.copy(center);
        const offset = camera.position.clone().sub(center);
        controls.spherical.setFromVector3(offset);
        controls.panOffset.set(0, 0, 0);
        controls.update();
      }}
      
      debug('Third-person mode OFF');
    }}

    function updateThirdPersonCamera() {{
      if (!characterGroup) return;
      
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

    // Listen for gamepad connection/disconnection
    window.addEventListener('gamepadconnected', (e) => {{
      if (!gamepadEnabled) return;
      gamepadIndex = e.gamepad.index;
      gamepadPrevButtons = new Array(e.gamepad.buttons.length).fill(false);
      updateGamepadStatusLabel();
      debug('Gamepad connected:', e.gamepad.id);
    }});

    window.addEventListener('gamepaddisconnected', (e) => {{
      if (e.gamepad.index === gamepadIndex) {{
        gamepadIndex = -1;
        const statusEl = document.getElementById('gamepadStatus');
        if (statusEl && gamepadEnabled) {{
          statusEl.textContent = '🔴 Disconnected';
          statusEl.style.color = '#ef4444';
        }}
      }}
    }});

    function applyStick(val) {{
      return Math.abs(val) < gamepadDeadzone ? 0 : val;
    }}

    function updateGamepad() {{
      if (!gamepadEnabled || gamepadIndex < 0) return;
      
      const gamepads = navigator.getGamepads();
      const gp = gamepads[gamepadIndex];
      if (!gp || !gp.connected) return;
      
      // ── Sticks ──
      const lx = applyStick(gp.axes[0] || 0);  // Left stick X
      const ly = applyStick(gp.axes[1] || 0);  // Left stick Y
      const rx = applyStick(gp.axes[2] || 0);  // Right stick X
      const ry = applyStick(gp.axes[3] || 0);  // Right stick Y
      
      // Store states for overlay rendering
      gamepadButtonStates = gp.buttons.map(b => b.pressed);
      gamepadAxesStates = [lx, ly, rx, ry];
      gamepadTriggerStates = [gp.buttons[6] ? gp.buttons[6].value : 0, gp.buttons[7] ? gp.buttons[7].value : 0];
      
      // ── Invert axes ──
      const invX = gamepadInvertX ? -1 : 1;
      const invY = gamepadInvertY ? -1 : 1;
      
      // ── Right stick → orbit camera around character ──
      if (rx !== 0) {{
        tpCamTheta += rx * 0.04 * invX;
      }}
      if (ry !== 0) {{
        tpCamPhi = Math.max(0.3, Math.min(Math.PI * 0.45, tpCamPhi + ry * 0.03 * invY));
      }}
      
      // ── Triggers → zoom camera in/out ──
      const lt = gp.buttons[6] ? gp.buttons[6].value : 0;
      const rt = gp.buttons[7] ? gp.buttons[7].value : 0;
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
      
      // ── Buttons (rising edge) ──
      const buttons = gp.buttons.map(b => b.pressed);
      function justPressed(idx) {{
        return idx < buttons.length && buttons[idx] && !gamepadPrevButtons[idx];
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
        if (characterGroup) {{
          characterGroup.position.set(0, 0, 0);
          characterYaw = 0;
          characterGroup.rotation.y = 0;
        }}
      }}
      
      // Y / Triangle (3) → toggle dynamic bones
      if (justPressed(3)) {{
        toggleDynamicBones();
        document.getElementById('swDynBones').checked = dynamicBonesEnabled;
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
      
      // D-pad Up (12) → speed up
      if (justPressed(12)) {{
        const slider = document.getElementById('animSpeedSlider');
        if (slider) {{ slider.value = Math.min(100, parseInt(slider.value) + 10); updateAnimSpeed(slider.value); }}
      }}
      
      // D-pad Down (13) → slow down
      if (justPressed(13)) {{
        const slider = document.getElementById('animSpeedSlider');
        if (slider) {{ slider.value = Math.max(-100, parseInt(slider.value) - 10); updateAnimSpeed(slider.value); }}
      }}
      
      // D-pad Left (14) → toggle wireframe
      if (justPressed(14)) {{
        toggleWireframe();
        document.getElementById('swWire').checked = wireframeMode;
      }}
      
      // D-pad Right (15) → toggle skeleton
      if (justPressed(15)) {{
        toggleSkeleton();
        document.getElementById('swSkel').checked = showSkeleton;
      }}
      
      // Start (9) → screenshot
      if (justPressed(9)) {{
        requestScreenshot();
      }}
      
      gamepadPrevButtons = buttons;
      
      // ── Update third-person camera ──
      updateThirdPersonCamera();
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
      const extended = document.getElementById('swInfoOverlay') && document.getElementById('swInfoOverlay').checked;
      
      if (!extended) {{
        // Basic mode
        statsEl.innerHTML = '<div>FPS: ' + currentFps + '</div>' +
          '<div>Triangles: ' + totalTris.toLocaleString() + '</div>' +
          '<div>Vertices: ' + totalVerts.toLocaleString() + '</div>' +
          '<div>Visible: ' + visibleCount + '/' + meshes.length + '</div>';
        return;
      }}
      
      // Extended mode
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
      html += '</table>';
      html += '<div style="border-top:1px solid rgba(124,58,237,0.3);margin:3px 0"></div>';
      html += '<div>' + dot(colorMode) + ' Colors  ' + dot(textureMode) + ' Textures  ' + dot(wireframeMode) + ' Wire  ' + dot(wireframeOverlayMode) + ' Overlay</div>';
      html += '<div>' + dot(showSkeleton) + ' Skeleton  ' + dot(showJoints) + ' Joints  ' + dot(showBoneNames) + ' Names  ' + dot(dynamicBonesEnabled) + ' DynBones</div>';
      if (gamepadEnabled) {{
        const gps = navigator.getGamepads ? navigator.getGamepads() : [];
        const gp = (gamepadIndex >= 0 && gps[gamepadIndex]) ? gps[gamepadIndex] : null;
        const tp = gamepadType;
        const labels = GP_LABELS[tp] || GP_LABELS.generic;
        const colors = GP_COLORS[tp] || {{}};
        const bs = gamepadButtonStates;  // current pressed states
        const ax = gamepadAxesStates;
        const tr = gamepadTriggerStates;
        const typeName = tp === 'xbox' ? 'Xbox' : tp === 'playstation' ? 'PlayStation' : 'Gamepad';
        
        html += '<div style="border-top:1px solid rgba(124,58,237,0.3);margin:3px 0"></div>';
        html += '<div style="color:#a78bfa;font-weight:bold">🎮 ' + typeName + (gp ? '' : ' · Waiting') + '</div>';
        
        // Button helper: renders a small styled button tag
        function btn(idx, action) {{
          const pressed = bs[idx];
          const label = labels[idx] || idx;
          const faceColor = colors[idx];
          let bg = pressed ? 'rgba(124,58,237,0.6)' : 'rgba(50,50,70,0.6)';
          let border = pressed ? '#a78bfa' : '#555';
          let fg = faceColor || (pressed ? '#fff' : '#999');
          if (pressed && faceColor) {{ bg = faceColor; fg = '#000'; border = faceColor; }}
          return '<span style="display:inline-block;padding:1px 4px;margin:1px;border-radius:3px;' +
            'background:' + bg + ';border:1px solid ' + border + ';color:' + fg + ';font-size:10px;' +
            'min-width:18px;text-align:center;font-weight:' + (pressed ? 'bold' : 'normal') + '">' +
            label + '</span><span style="color:' + (pressed ? '#e0e0e0' : '#777') + ';font-size:10px;margin-right:6px">' + action + '</span>';
        }}
        
        // Stick indicator helper
        function stick(label, action, xVal, yVal) {{
          const active = (Math.abs(xVal) > 0 || Math.abs(yVal) > 0);
          const fg = active ? '#e0e0e0' : '#777';
          const bg = active ? 'rgba(124,58,237,0.4)' : 'rgba(50,50,70,0.6)';
          // Mini stick visualization: dot position
          const dotX = Math.round(xVal * 5);
          const dotY = Math.round(yVal * 5);
          return '<span style="display:inline-block;position:relative;width:16px;height:16px;' +
            'border-radius:50%;background:' + bg + ';border:1px solid ' + (active ? '#a78bfa' : '#555') + ';' +
            'vertical-align:middle;margin:1px 2px">' +
            '<span style="position:absolute;width:4px;height:4px;border-radius:50%;background:' + (active ? '#a78bfa' : '#888') + ';' +
            'left:' + (5 + dotX) + 'px;top:' + (5 + dotY) + 'px"></span></span>' +
            '<span style="color:' + fg + ';font-size:10px;margin-right:6px">' + action + '</span>';
        }}
        
        // Trigger helper
        function trig(idx, action, val) {{
          const active = val > 0.05;
          const label = labels[idx] || idx;
          const pct = Math.round(val * 100);
          let bg = active ? 'rgba(124,58,237,' + (0.2 + val * 0.5) + ')' : 'rgba(50,50,70,0.6)';
          return '<span style="display:inline-block;padding:1px 4px;margin:1px;border-radius:3px;' +
            'background:' + bg + ';border:1px solid ' + (active ? '#a78bfa' : '#555') + ';color:' + (active ? '#e0e0e0' : '#999') + ';' +
            'font-size:10px;min-width:18px;text-align:center">' + label + '</span>' +
            '<span style="color:' + (active ? '#e0e0e0' : '#777') + ';font-size:10px;margin-right:6px">' + action + '</span>';
        }}
        
        html += '<div style="line-height:18px">';
        html += stick('LS', 'Move', ax[0], ax[1]) + stick('RS', 'Camera', ax[2], ax[3]) + '<br>';
        html += btn(0, 'Play') + btn(1, 'Stop') + '<br>';
        html += btn(2, 'Reset') + btn(3, 'DynB') + '<br>';
        html += btn(4, '◀Anim') + btn(5, 'Anim▶') + '<br>';
        html += trig(6, 'Zoom-', tr[0]) + trig(7, 'Zoom+', tr[1]) + '<br>';
        html += btn(12, 'Spd+') + btn(13, 'Spd-') + '<br>';
        html += btn(14, 'Wire') + btn(15, 'Skel') + '<br>';
        html += btn(9, '📸 Shot');
        html += '</div>';
      }}
      if (opacity < 1.0) {{
        html += '<div style="color:#9ca3af">Opacity: ' + opacity.toFixed(2) + '</div>';
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
      if (document.getElementById('swInfoOverlay') && document.getElementById('swInfoOverlay').checked) {{
        drawInfoOverlay(ctx, w, h, sf);
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
      const optLine = opts.map(o => (o[1] ? '●' : '○') + ' ' + o[0]).join('  ');
      lines.push([optLine, 'mixed']);

      const opts2 = [
        ['Skeleton', showSkeleton], ['Joints', showJoints],
        ['Names', showBoneNames], ['DynBones', dynamicBonesEnabled],
      ];
      const optLine2 = opts2.map(o => (o[1] ? '●' : '○') + ' ' + o[0]).join('  ');
      lines.push([optLine2, 'mixed']);

      if (gamepadEnabled) {{
        const tp = gamepadType;
        const labels = GP_LABELS[tp] || GP_LABELS.generic;
        const typeName = tp === 'xbox' ? 'Xbox' : tp === 'playstation' ? 'PlayStation' : 'Gamepad';
        lines.push([div, 'divider']);
        lines.push(['🎮 ' + typeName, 'title']);
        // gprow: array of [btnIdx, action] pairs per row (-1 = stick, -2 = trigger)
        lines.push([[[{{'t':'stick','x':gamepadAxesStates[0],'y':gamepadAxesStates[1]}}, 'Move'],
                      [{{'t':'stick','x':gamepadAxesStates[2],'y':gamepadAxesStates[3]}}, 'Camera']], 'gprow']);
        lines.push([[[0, 'Play'], [1, 'Stop']], 'gprow']);
        lines.push([[[2, 'Reset'], [3, 'DynB']], 'gprow']);
        lines.push([[[4, '◀Anim'], [5, 'Anim▶']], 'gprow']);
        lines.push([[[{{'t':'trig','idx':6,'v':gamepadTriggerStates[0]}}, 'Zoom-'],
                      [{{'t':'trig','idx':7,'v':gamepadTriggerStates[1]}}, 'Zoom+']], 'gprow']);
        lines.push([[[12, 'Spd+'], [13, 'Spd-']], 'gprow']);
        lines.push([[[14, 'Wire'], [15, 'Skel']], 'gprow']);
        lines.push([[[9, '📸Shot']], 'gprow']);
      }}

      if (opacity < 1.0) {{
        lines.push(['Opacity: ' + opacity.toFixed(2), 'value']);
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
      const gpBtnW = Math.round(28 * sf);  // button width
      const gpActW = Math.round(40 * sf);  // action label width
      const gpPairW = gpBtnW + gpActW + Math.round(6 * sf);  // one btn+action pair
      
      lines.forEach(l => {{
        if (l[1] === 'gprow') {{
          const rowW = l[0].length * gpPairW;
          if (rowW > maxW) maxW = rowW;
        }} else {{
          const m = ctx.measureText(l[0]);
          if (m.width > maxW) maxW = m.width;
        }}
      }});

      const boxW = maxW + padding * 2;
      const boxH = lines.length * lineHeight + padding * 2;
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
        if (type === 'gprow') {{
          // Render gamepad button row with graphical icons
          const pairs = text;  // array of [btnData, actionLabel]
          let tx = boxX + padding;
          const labels = GP_LABELS[gamepadType] || GP_LABELS.generic;
          const fColors = GP_COLORS[gamepadType] || {{}};
          const btnH = Math.round(14 * sf);
          const btnR = Math.round(3 * sf);
          const gap = Math.round(4 * sf);
          const smallFont = Math.round(9 * sf) + 'px monospace';
          const tinyFont = Math.round(7 * sf) + 'px monospace';
          
          pairs.forEach(([btnData, action]) => {{
            if (typeof btnData === 'object' && btnData.t === 'stick') {{
              // ── Draw stick circle ──
              const stR = Math.round(8 * sf);
              const cx = tx + stR;
              const cy = ty + lineHeight / 2;
              const active = (Math.abs(btnData.x) > 0 || Math.abs(btnData.y) > 0);
              
              // Outer ring
              ctx.beginPath();
              ctx.arc(cx, cy, stR, 0, Math.PI * 2);
              ctx.fillStyle = active ? 'rgba(124,58,237,0.4)' : 'rgba(50,50,70,0.6)';
              ctx.fill();
              ctx.strokeStyle = active ? '#a78bfa' : '#555';
              ctx.lineWidth = sf;
              ctx.stroke();
              
              // Inner dot (position based on axes)
              const dotR = Math.round(2.5 * sf);
              const dx = cx + btnData.x * (stR - dotR - sf);
              const dy = cy + btnData.y * (stR - dotR - sf);
              ctx.beginPath();
              ctx.arc(dx, dy, dotR, 0, Math.PI * 2);
              ctx.fillStyle = active ? '#a78bfa' : '#888';
              ctx.fill();
              
              // Action label
              ctx.font = smallFont;
              ctx.fillStyle = active ? '#e0e0e0' : '#777';
              ctx.fillText(action, cx + stR + gap, ty + (lineHeight - 9 * sf) / 2);
              tx += stR * 2 + gap + gpActW + gap;
              
            }} else if (typeof btnData === 'object' && btnData.t === 'trig') {{
              // ── Draw trigger bar ──
              const barW = gpBtnW;
              const barH = btnH;
              const bx = tx;
              const by = ty + (lineHeight - barH) / 2;
              const val = btnData.v;
              const active = val > 0.05;
              const label = labels[btnData.idx] || '';
              
              // Background
              ctx.fillStyle = 'rgba(50,50,70,0.6)';
              ctx.beginPath();
              ctx.roundRect(bx, by, barW, barH, btnR);
              ctx.fill();
              
              // Fill bar
              if (active) {{
                ctx.fillStyle = 'rgba(124,58,237,' + (0.3 + val * 0.5) + ')';
                ctx.beginPath();
                ctx.roundRect(bx, by, barW * val, barH, btnR);
                ctx.fill();
              }}
              
              // Border
              ctx.strokeStyle = active ? '#a78bfa' : '#555';
              ctx.lineWidth = sf;
              ctx.beginPath();
              ctx.roundRect(bx, by, barW, barH, btnR);
              ctx.stroke();
              
              // Label centered
              ctx.font = tinyFont;
              ctx.fillStyle = active ? '#e0e0e0' : '#999';
              ctx.textAlign = 'center';
              ctx.fillText(label, bx + barW / 2, by + (barH - 7 * sf) / 2);
              ctx.textAlign = 'left';
              
              // Action label
              ctx.font = smallFont;
              ctx.fillStyle = active ? '#e0e0e0' : '#777';
              ctx.fillText(action, bx + barW + gap, ty + (lineHeight - 9 * sf) / 2);
              tx += barW + gap + gpActW + gap;
              
            }} else {{
              // ── Draw regular button ──
              const idx = btnData;
              const pressed = gamepadButtonStates[idx];
              const label = labels[idx] || String(idx);
              const faceColor = fColors[idx];
              const bx = tx;
              const by = ty + (lineHeight - btnH) / 2;
              
              // Button shape: face buttons round, others rounded rect
              const isFace = (idx >= 0 && idx <= 3);
              const isDpad = (idx >= 12 && idx <= 15);
              
              if (isFace) {{
                // Circular face button
                const cr = btnH / 2;
                const cx = bx + cr;
                const cy = by + cr;
                ctx.beginPath();
                ctx.arc(cx, cy, cr, 0, Math.PI * 2);
                if (pressed && faceColor) {{
                  ctx.fillStyle = faceColor;
                }} else {{
                  ctx.fillStyle = pressed ? 'rgba(124,58,237,0.6)' : 'rgba(50,50,70,0.6)';
                }}
                ctx.fill();
                ctx.strokeStyle = pressed ? (faceColor || '#a78bfa') : '#555';
                ctx.lineWidth = sf;
                ctx.stroke();
                
                // Label
                ctx.font = 'bold ' + tinyFont;
                ctx.fillStyle = (pressed && faceColor) ? '#000' : (faceColor || (pressed ? '#fff' : '#999'));
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(label, cx, cy);
                ctx.textAlign = 'left';
                ctx.textBaseline = 'top';
                
                tx += btnH + gap;
              }} else if (isDpad) {{
                // D-pad: square-ish with arrow
                ctx.fillStyle = pressed ? 'rgba(124,58,237,0.6)' : 'rgba(50,50,70,0.6)';
                ctx.beginPath();
                ctx.roundRect(bx, by, gpBtnW, btnH, btnR);
                ctx.fill();
                ctx.strokeStyle = pressed ? '#a78bfa' : '#555';
                ctx.lineWidth = sf;
                ctx.beginPath();
                ctx.roundRect(bx, by, gpBtnW, btnH, btnR);
                ctx.stroke();
                
                ctx.font = tinyFont;
                ctx.fillStyle = pressed ? '#fff' : '#999';
                ctx.textAlign = 'center';
                ctx.fillText(label, bx + gpBtnW / 2, by + (btnH - 7 * sf) / 2);
                ctx.textAlign = 'left';
                tx += gpBtnW + gap;
              }} else {{
                // Bumper / Start / etc
                ctx.fillStyle = pressed ? 'rgba(124,58,237,0.6)' : 'rgba(50,50,70,0.6)';
                ctx.beginPath();
                ctx.roundRect(bx, by, gpBtnW, btnH, btnR);
                ctx.fill();
                ctx.strokeStyle = pressed ? '#a78bfa' : '#555';
                ctx.lineWidth = sf;
                ctx.beginPath();
                ctx.roundRect(bx, by, gpBtnW, btnH, btnR);
                ctx.stroke();
                
                ctx.font = tinyFont;
                ctx.fillStyle = pressed ? '#fff' : '#999';
                ctx.textAlign = 'center';
                ctx.fillText(label, bx + gpBtnW / 2, by + (btnH - 7 * sf) / 2);
                ctx.textAlign = 'left';
                tx += gpBtnW + gap;
              }}
              
              // Action label next to button
              ctx.font = smallFont;
              ctx.fillStyle = pressed ? '#e0e0e0' : '#777';
              ctx.fillText(action, tx, ty + (lineHeight - 9 * sf) / 2);
              tx += gpActW + gap;
            }}
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
              if (result.warning) console.warn('[Recording]', result.warning);
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
      controls.update();
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
        print("Usage: python viewer_mdl_textured.py <path_to_model.mdl> [--recompute-normals] [--debug] [--skip-popup]")
        sys.exit(1)

    mdl_path = Path(sys.argv[1])
    recompute_normals = '--recompute-normals' in sys.argv
    debug_mode = '--debug' in sys.argv
    skip_popup = '--skip-popup' in sys.argv
    
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
        mdl_path, temp_dir, recompute_normals
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
        animations_data=animations_data, skip_popup=skip_popup
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
