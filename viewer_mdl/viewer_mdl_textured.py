#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viewer_mdl_textured.py — Direct .mdl preview with TEXTURE SUPPORT (FIXED)

FEATURES:
- Loads and displays textures from DDS files
- Copies textures to temp directory (NOT embedded as data URLs)
- Properly manages three.min.js in temp
- Auto cleanup on exit

REQUIREMENTS:
  pip install pywebview Pillow

USAGE:
  python viewer_mdl_textured.py /path/to/model.mdl [--use-original-normals]
"""

from pathlib import Path
import sys
import json
import numpy as np
import tempfile
import atexit
import time
import shutil

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
        print(f"  ⚠ Failed to convert {dds_path.name}: {e}")
        return False


# -----------------------------
# Smooth normals
# -----------------------------
def compute_smooth_normals_with_sharing(vertices: np.ndarray, indices: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """Compute smooth normals with position sharing."""
    from collections import defaultdict

    position_map = {}
    vertex_to_position = {}
    for idx, v in enumerate(vertices):
        key = tuple(np.round(v / tolerance) * tolerance)
        position_map.setdefault(key, []).append(idx)
        vertex_to_position[idx] = key

    position_normals = defaultdict(lambda: np.zeros(3, dtype=np.float32))

    for i in range(0, len(indices), 3):
        i0, i1, i2 = int(indices[i]), int(indices[i + 1]), int(indices[i + 2])
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        position_normals[vertex_to_position[i0]] += face_normal
        position_normals[vertex_to_position[i1]] += face_normal
        position_normals[vertex_to_position[i2]] += face_normal

    for k in position_normals:
        n = position_normals[k]
        L = np.linalg.norm(n)
        if L > 0:
            position_normals[k] = n / L

    normals = np.zeros_like(vertices, dtype=np.float32)
    for idx in range(len(vertices)):
        normals[idx] = position_normals[vertex_to_position[idx]]
    return normals


# -----------------------------
# Load MDL with textures
# -----------------------------
def load_mdl_with_textures(mdl_path: Path, temp_dir: Path, use_original_normals: bool = False):
    """
    Load MDL file and copy textures to temp directory.
    
    Returns:
        tuple: (meshes, material_texture_map)
    """
    mdl_path = Path(mdl_path).absolute()  # Convert to absolute path!
    with open(mdl_path, "rb") as f:
        mdl_data = f.read()

    print(f"\n{'='*60}")
    print(f"Loading MDL: {mdl_path.name}")
    print(f"{'='*60}")

    # Decrypt and parse MDL
    mdl_data = decryptCLE(mdl_data)
    material_struct = obtain_material_data(mdl_data)
    mesh_struct = obtain_mesh_data(mdl_data, material_struct=material_struct)

    # Get image list from materials
    image_list = sorted(list(set([x['texture_image_name']+'.dds' for y in material_struct for x in y['textures']])))
    
    print(f"\nFound {len(material_struct)} materials")
    print(f"Found {len(image_list)} unique textures")

    # Search paths for textures
    # Priority order:
    # 1. Same level as MDL
    # 2. Subdirectories of MDL location
    # 3. Parent directories going up
    search_paths = [
        # Same level as MDL
        mdl_path.parent,                                        # ./
        mdl_path.parent / 'image',                             # ./image/
        mdl_path.parent / 'textures',                          # ./textures/
        
        # One level up from MDL
        mdl_path.parent.parent,                                # ../
        mdl_path.parent.parent / 'image',                      # ../image/
        mdl_path.parent.parent / 'dx11' / 'image',            # ../dx11/image/
        mdl_path.parent.parent / 'dxl1' / 'image',            # ../dxl1/image/
        
        # Two levels up from MDL (common case: model/subdir/*.mdl, textures at model level)
        mdl_path.parent.parent.parent,                         # ../../
        mdl_path.parent.parent.parent / 'image',              # ../../image/
        mdl_path.parent.parent.parent / 'dx11' / 'image',     # ../../dx11/image/  ← Your case!
        mdl_path.parent.parent.parent / 'dxl1' / 'image',     # ../../dxl1/image/
        
        # Three levels up (for deeply nested structures)
        mdl_path.parent.parent.parent.parent / 'dx11' / 'image',  # ../../../dx11/image/
        mdl_path.parent.parent.parent.parent / 'dxl1' / 'image',  # ../../../dxl1/image/
        
        # Current working directory
        Path.cwd(),                                            # cwd
        Path.cwd() / 'image',                                  # cwd/image/
        Path.cwd() / 'textures',                               # cwd/textures/
        Path.cwd() / 'dx11' / 'image',                        # cwd/dx11/image/
        Path.cwd() / 'dxl1' / 'image',                        # cwd/dxl1/image/
    ]

    # Create textures subdirectory in temp
    temp_textures_dir = temp_dir / 'textures'
    temp_textures_dir.mkdir(exist_ok=True)

    # Copy and convert textures
    material_texture_map = {}
    texture_success = {}
    
    if TEXTURES_AVAILABLE and PIL_AVAILABLE and len(image_list) > 0:
        print(f"\nSearching for textures in:")
        existing_count = 0
        for p in search_paths:
            if p.exists():
                print(f"  [OK] {p}")
                existing_count += 1
            else:
                print(f"  [ - ] {p} (not found)")
        
        if existing_count == 0:
            print(f"\n[WARNING] None of the texture search paths exist!")
            print(f"  MDL location: {mdl_path.absolute()}")
            print(f"  MDL parent: {mdl_path.parent.absolute()}")
            print(f"  Current dir: {Path.cwd()}")
        
        print(f"\nConverting textures to PNG...")
        for tex_name in image_list:
            # Find DDS file
            dds_path = find_texture_file(tex_name, search_paths)
            
            if dds_path:
                # Convert to PNG name
                png_name = tex_name.replace('.dds', '.png')
                png_path = temp_textures_dir / png_name
                
                # Convert DDS to PNG
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
                wrapS = tex.get('wrapS', 0)  # Default to REPEAT
                wrapT = tex.get('wrapT', 0)  # Default to REPEAT
                
                if tex_name in texture_success and texture_success[tex_name]:
                    # Use relative path: textures/filename.png
                    rel_path = f"textures/{texture_success[tex_name]}"
                    
                    # Create texture info with wrap modes
                    tex_info = {
                        'path': rel_path,
                        'wrapS': wrapS,
                        'wrapT': wrapT
                    }
                    
                    # Map to material property
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
        print("\n[WARNING] Texture loading disabled or dependencies missing")

    # Extract mesh data
    meshes = []
    mesh_blocks = mesh_struct.get("mesh_blocks", [])
    all_buffers = mesh_struct.get("mesh_buffers", [])

    print(f"\nProcessing {len(all_buffers)} mesh groups...")

    for i, submesh_list in enumerate(all_buffers):
        base_name = mesh_blocks[i].get("name", f"mesh_{i}") if i < len(mesh_blocks) else f"mesh_{i}"
        primitives = mesh_blocks[i].get("primitives", []) if i < len(mesh_blocks) else []

        for j, submesh in enumerate(submesh_list):
            vb = submesh.get("vb", [])
            ib = submesh.get("ib", {}).get("Buffer", [])

            pos_buffer = None
            normal_buffer = None
            uv_buffer = None

            for element in vb:
                sem = element.get("SemanticName")
                buf = element.get("Buffer")
                if sem == "POSITION":
                    pos_buffer = buf
                elif sem == "NORMAL":
                    normal_buffer = buf
                elif sem == "TEXCOORD" and uv_buffer is None:
                    uv_buffer = buf

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

            if use_original_normals and normal_buffer:
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
                "material": material_name
            }

            meshes.append(mesh_data)

    print(f"[OK] Loaded {len(meshes)} submeshes\n")

    return meshes, material_texture_map


# -----------------------------
# Generate HTML
# -----------------------------
def generate_html_with_textures(mdl_path: Path, meshes: list, material_texture_map: dict) -> str:
    """Generate HTML content with texture support."""
    
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
        
        meshes_data.append(mesh_info)

    materials_json = {}
    for mat_name, textures in material_texture_map.items():
        materials_json[mat_name] = textures

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
                 min-width: 200px; max-width: 400px; width: auto;
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
    button {{ background: linear-gradient(135deg, #7c3aed, #a855f7); border: none;
             color: white; padding: 10px; margin: 5px 0; cursor: pointer;
             border-radius: 6px; width: 100%; font-weight: 600; font-size: 13px; }}
    button:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4); }}
    .toggle-btn {{
      display: flex; align-items: center; justify-content: space-between;
      background: rgba(124, 58, 237, 0.15); border: 1px solid rgba(124, 58, 237, 0.3);
      color: #e0e0e0; padding: 10px 12px; margin: 5px 0; cursor: pointer;
      border-radius: 6px; width: 100%; font-weight: 500; font-size: 13px;
      transition: all 0.2s ease;
    }}
    .toggle-btn:hover {{ background: rgba(124, 58, 237, 0.25); }}
    .toggle-btn.active {{
      background: rgba(124, 58, 237, 0.35); border-color: rgba(124, 58, 237, 0.6);
    }}
    .toggle-btn .toggle-label {{ display: flex; align-items: center; gap: 6px; }}
    .toggle-btn .toggle-indicator {{
      width: 36px; height: 20px; border-radius: 10px; position: relative;
      background: rgba(100, 100, 120, 0.5); transition: background 0.2s ease; flex-shrink: 0;
    }}
    .toggle-btn.active .toggle-indicator {{ background: #7c3aed; }}
    .toggle-btn .toggle-indicator::after {{
      content: ''; position: absolute; width: 16px; height: 16px; border-radius: 50%;
      background: white; top: 2px; left: 2px; transition: transform 0.2s ease;
    }}
    .toggle-btn.active .toggle-indicator::after {{ transform: translateX(16px); }}
    .cycle-btn {{
      display: flex; align-items: center; justify-content: space-between;
      background: rgba(124, 58, 237, 0.15); border: 1px solid rgba(124, 58, 237, 0.3);
      color: #e0e0e0; padding: 10px 12px; margin: 5px 0; cursor: pointer;
      border-radius: 6px; width: 100%; font-weight: 500; font-size: 13px;
      transition: all 0.2s ease;
    }}
    .cycle-btn:hover {{ background: rgba(124, 58, 237, 0.25); }}
    .cycle-btn.active {{ background: rgba(124, 58, 237, 0.35); border-color: rgba(124, 58, 237, 0.6); }}
    .cycle-btn .toggle-label {{ display: flex; align-items: center; gap: 6px; }}
    .cycle-dots {{ display: flex; gap: 6px; align-items: center; flex-shrink: 0; }}
    .cycle-dot {{
      width: 18px; height: 18px; border-radius: 50%; border: 2px solid transparent;
      transition: all 0.2s ease; opacity: 0.4;
    }}
    .cycle-dot.dot-off {{ background: #808080; }}
    .cycle-dot.dot-color {{ background: linear-gradient(135deg, #ff6b6b, #4ecdc4, #ffe66d); }}
    .cycle-dot.dot-white {{ background: #ffffff; }}
    .cycle-dot.current {{ opacity: 1; border-color: #a78bfa; transform: scale(1.15); box-shadow: 0 0 8px rgba(167, 139, 250, 0.5); }}
    .mesh-toggle {{
      display: flex; align-items: center; margin: 8px 0; padding: 8px;
      background: rgba(124, 58, 237, 0.1); border-radius: 6px; transition: background 0.2s;
    }}
    .mesh-toggle:hover {{ background: rgba(124, 58, 237, 0.2); }}
    .mesh-toggle input {{ margin-right: 10px; cursor: pointer; width: 18px; height: 18px; }}
    .mesh-toggle label {{ cursor: pointer; flex-grow: 1; font-size: 13px; }}
    .texture-indicator {{
      display: inline-block; width: 12px; height: 12px; border-radius: 3px;
      margin-left: 6px; background: linear-gradient(135deg, #10b981, #34d399);
    }}
    #screenshot-modal {{
      display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(0,0,0,0.8); z-index: 2000; align-items: center; justify-content: center;
    }}
    #screenshot-modal.show {{ display: flex; }}
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
    .modal-content .filename:hover {{
      background: rgba(124, 58, 237, 0.3); color: #c4b5fd;
    }}
    .modal-content button {{
      margin-top: 20px; padding: 12px 30px; font-size: 14px; width: auto;
      min-width: 120px;
    }}
  </style>
</head>
<body>
  <button id="controls-toggle" onclick="toggleControlsPanel()">☰</button>
  <div id="container"></div>
  <div id="info" class="panel">
    <h3>📦 Model Viewer</h3>
    <p style="font-size: 13px; color: #b0b0b0; line-height: 1.5; margin-bottom: 12px;">
      <strong style="color: #7c3aed;">{mdl_path.name}</strong>
    </p>
    <div id="texture-status" style="background: rgba(124, 58, 237, 0.15); padding: 10px; border-radius: 8px; font-size: 11px; margin-bottom: 12px;">
      <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 16px;">🎨</span>
        <span style="color: #9ca3af;" id="texture-info">Loading textures...</span>
      </div>
    </div>
    <div style="background: rgba(124, 58, 237, 0.15); padding: 10px; border-radius: 8px; font-size: 11px;">
      <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
        <span style="font-size: 16px;">🖱️</span>
        <span style="color: #9ca3af;">Left: Rotate</span>
      </div>
      <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
        <span style="font-size: 16px;">🖱️</span>
        <span style="color: #9ca3af;">Right: Pan</span>
      </div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 16px;">🔄</span>
        <span style="color: #9ca3af;">Wheel: Zoom</span>
      </div>
    </div>
  </div>
  <div id="controls" class="panel">
    <h4>🎮 Controls</h4>
    <button onclick="toggleAllMeshes(true)">✅ Show All</button>
    <button onclick="toggleAllMeshes(false)">❌ Hide All</button>
    <div class="cycle-btn" id="btn-colors" onclick="toggleColors()">
      <span class="toggle-label">🎨 Colors</span>
      <span class="cycle-dots">
        <span class="cycle-dot dot-off current" title="Off (gray)"></span>
        <span class="cycle-dot dot-color" title="Per-mesh colors"></span>
        <span class="cycle-dot dot-white" title="White"></span>
      </span>
    </div>
    <div class="toggle-btn active" id="btn-textures" onclick="toggleTextures()">
      <span class="toggle-label">🖼️ Toggle Textures</span>
      <span class="toggle-indicator"></span>
    </div>
    <div class="toggle-btn" id="btn-wireframe" onclick="toggleWireframe()">
      <span class="toggle-label">📐 Wireframe Only</span>
      <span class="toggle-indicator"></span>
    </div>
    <div class="toggle-btn" id="btn-wireframe-overlay" onclick="toggleWireframeOverlay()">
      <span class="toggle-label">🔲 Wireframe Overlay</span>
      <span class="toggle-indicator"></span>
    </div>
    <button onclick="resetCamera()">🎯 Reset Camera</button>
    <button onclick="takeScreenshot()">📷 Screenshot</button>
    <h4>👁️ Meshes</h4>
    <div id="mesh-list"></div>
  </div>
  <div id="stats" class="panel">
    <div id="vertices"></div>
    <div id="triangles"></div>
    <div id="visible"></div>
    <div id="fps"></div>
  </div>

  <!-- Screenshot Modal -->
  <div id="screenshot-modal">
    <div class="modal-content">
      <h3>📷 Screenshot Saved</h3>
      <p>Screenshot has been successfully saved to:</p>
      <div class="filename" id="screenshot-path"></div>
      <button onclick="closeScreenshotModal()">OK</button>
    </div>
  </div>

  <script src="three.min.js"></script>
  <script>
    // ============================================================
    // CONFIGURATION - Adjust these values as needed
    // ============================================================
    const CONFIG = {{
      CAMERA_ZOOM: 1.2,           // Camera zoom factor (lower = closer, higher = farther)
                                   // 1.0 = tight fit, 1.2 = 20% padding, 0.9 = zoom in 10%
      AUTO_HIDE_SHADOW: true,      // Automatically hide meshes with "shadow" in name
      INITIAL_BACKGROUND: 0x1a1a2e // Background color (hex)
    }};
    // ============================================================
    
    const data = {json.dumps(meshes_data)};
    const materials = {json.dumps(materials_json)};
    
    let scene, camera, renderer, controls, meshes = [];
    let wireframeMode = false, wireframeOverlayMode = false, colorMode = 0;
    let wireframeMeshes = [];
    let textureLoader = new THREE.TextureLoader();
    let loadedTexturesCount = 0;
    let totalTexturesCount = 0;
    let texturesEnabled = true;

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
        this.isMouseDown = true;
        this.mouseButton = e.button;
      }}
      
      onMouseUp() {{
        this.isMouseDown = false;
      }}
      
      onMouseMove(e) {{
        if (!this.isMouseDown) return;
        
        if (this.mouseButton === this.mouseButtons.LEFT) {{
          // LEFT = ROTATE
          const dx = e.movementX * this.rotateSpeed * 0.01;
          const dy = e.movementY * this.rotateSpeed * 0.01;
          this.sphericalDelta.theta -= dx;
          this.sphericalDelta.phi -= dy;
        }} else if (this.mouseButton === this.mouseButtons.RIGHT) {{
          // RIGHT = PAN (použít e.movementX/Y přímo!)
          const cam = this.camera;
          const right = new THREE.Vector3(cam.matrix.elements[0], cam.matrix.elements[1], cam.matrix.elements[2]);
          const up = new THREE.Vector3(cam.matrix.elements[4], cam.matrix.elements[5], cam.matrix.elements[6]);
          this.panOffset.add(right.multiplyScalar(-e.movementX * this.panSpeed * 0.0008));
          this.panOffset.add(up.multiplyScalar(e.movementY * this.panSpeed * 0.0008));
        }}
      }}
      
      onMouseWheel(e) {{
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
      // Konverze wrap mode hodnot:
      // 0 = REPEAT, 1 = MIRROR (MirroredRepeatWrapping), 2 = CLAMP (ClampToEdgeWrapping)
      const wrapModes = [
        THREE.RepeatWrapping,        // 0
        THREE.MirroredRepeatWrapping, // 1
        THREE.ClampToEdgeWrapping     // 2
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
          color: 0x808080, roughness: 0.7, metalness: 0.2
        }});
      }}

      const matParams = {{ roughness: 0.7, metalness: 0.2, side: THREE.DoubleSide }};

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
            mesh.userData.originalMap = texture; // Uložit pro toggle
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
            mesh.userData.originalNormalMap = texture; // Uložit pro toggle
            mesh.material.needsUpdate = true;
          }}
        }});
      }}

      return new THREE.MeshStandardMaterial(matParams);
    }}

    function updateTextureStatus() {{
      const status = totalTexturesCount > 0 
        ? `${{loadedTexturesCount}}/${{totalTexturesCount}} textures loaded`
        : 'No textures';
      document.getElementById('texture-info').textContent = status;
    }}

    function init() {{
      scene = new THREE.Scene();
      scene.background = new THREE.Color(CONFIG.INITIAL_BACKGROUND);

      camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
      camera.position.set(0, 2, 5);

      renderer = new THREE.WebGLRenderer({{ antialias: true }});
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

      loadMeshes();
      populateMeshList();
      updateStats();
      updateTextureStatus();

      window.addEventListener('resize', onWindowResize);
      document.getElementById('controls-toggle').classList.add('visible');
      animate();
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
        
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        geometry.computeBoundingSphere();

        const material = createMaterial(meshData.material, meshData.name);
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.userData.meshName = meshData.name;
        mesh.userData.materialName = meshData.material;
        mesh.userData.originalColor = colors[idx % colors.length];
        mesh.userData.hasTexture = !!meshData.material && !!materials[meshData.material];
        
        // Auto-hide shadow meshes if enabled in config
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
      
      // Calculate proper camera distance to fill viewport
      const maxDim = Math.max(size.x, size.y, size.z);
      const fov = camera.fov * (Math.PI / 180);
      
      // Take into account both vertical and horizontal FOV
      const aspect = camera.aspect;
      const vFOV = fov;
      const hFOV = 2 * Math.atan(Math.tan(vFOV / 2) * aspect);
      
      // Calculate distance needed to fit model in both dimensions
      const distanceV = maxDim / (2 * Math.tan(vFOV / 2));
      const distanceH = maxDim / (2 * Math.tan(hFOV / 2));
      const cameraDistance = Math.max(distanceV, distanceH);
      
      // Apply zoom factor from config
      const dist = cameraDistance * CONFIG.CAMERA_ZOOM;
      
      // Position camera
      const direction = new THREE.Vector3(0.5, 0.5, 1).normalize();
      camera.position.copy(center).add(direction.multiplyScalar(dist));
      camera.lookAt(center);
      
      // Initialize controls properly
      controls.target.copy(center);
      
      // Set spherical from camera position
      const offset = camera.position.clone().sub(center);
      controls.spherical.setFromVector3(offset);
      // radius is already set by setFromVector3
      controls.panOffset.set(0, 0, 0);
      
      controls.update();
    }}

    function populateMeshList() {{
      const list = document.getElementById('mesh-list');
      meshes.forEach((mesh, idx) => {{
        const div = document.createElement('div');
        div.className = 'mesh-toggle';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = mesh.visible;  // Use actual visibility state
        checkbox.id = `mesh-${{idx}}`;
        checkbox.addEventListener('change', () => {{
          mesh.visible = checkbox.checked;
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
      meshes.forEach((mesh, idx) => {{
        mesh.visible = visible;
        document.getElementById(`mesh-${{idx}}`).checked = visible;
      }});
      updateStats();
    }}

    function updateColorDots() {{
      const dots = document.querySelectorAll('#btn-colors .cycle-dot');
      dots.forEach((dot, i) => {{
        dot.classList.toggle('current', i === colorMode);
      }});
      document.getElementById('btn-colors').classList.toggle('active', colorMode !== 0);
    }}

    function toggleColors() {{
      colorMode = (colorMode + 1) % 3;
      updateColorDots();

      // Colors going ON → disable textures if they're on
      if (colorMode === 1 && texturesEnabled) {{
        texturesEnabled = false;
        document.getElementById('btn-textures').classList.remove('active');
        meshes.forEach(mesh => {{
          if (mesh.userData.hasTexture) {{
            mesh.material.map = null;
            mesh.material.normalMap = null;
            mesh.material.needsUpdate = true;
          }}
        }});
      }}
      // Apply color mode to all meshes
      meshes.forEach(mesh => {{
        if (colorMode === 0) mesh.material.color.set(0x808080);
        else if (colorMode === 1) mesh.material.color.setHex(mesh.userData.originalColor);
        else mesh.material.color.set(0xffffff);
      }});
    }}

    function toggleTextures() {{
      texturesEnabled = !texturesEnabled;
      document.getElementById('btn-textures').classList.toggle('active', texturesEnabled);

      // Textures ON → reset colors to OFF
      if (texturesEnabled && colorMode !== 0) {{
        colorMode = 0;
        updateColorDots();
      }}

      meshes.forEach(mesh => {{
        if (mesh.userData.hasTexture) {{
          if (texturesEnabled) {{
            if (mesh.userData.originalMap) mesh.material.map = mesh.userData.originalMap;
            if (mesh.userData.originalNormalMap) mesh.material.normalMap = mesh.userData.originalNormalMap;
            mesh.material.color.set(0xffffff);
          }} else {{
            mesh.material.map = null;
            mesh.material.normalMap = null;
            mesh.material.color.set(0x808080);
          }}
          mesh.material.needsUpdate = true;
        }}
      }});
    }}

    function toggleWireframe() {{
      // If overlay is active, turn it off first
      if (wireframeOverlayMode) {{
        wireframeOverlayMode = false;
        document.getElementById('btn-wireframe-overlay').classList.remove('active');
        wireframeMeshes.forEach(wf => wf.parent.remove(wf));
        wireframeMeshes = [];
      }}
      wireframeMode = !wireframeMode;
      document.getElementById('btn-wireframe').classList.toggle('active', wireframeMode);
      meshes.forEach(mesh => mesh.material.wireframe = wireframeMode);
    }}

    function toggleWireframeOverlay() {{
      // If wireframe is active, turn it off first
      if (wireframeMode) {{
        wireframeMode = false;
        document.getElementById('btn-wireframe').classList.remove('active');
        meshes.forEach(mesh => mesh.material.wireframe = false);
      }}
      wireframeOverlayMode = !wireframeOverlayMode;
      document.getElementById('btn-wireframe-overlay').classList.toggle('active', wireframeOverlayMode);
      
      if (wireframeOverlayMode) {{
        meshes.forEach(mesh => {{
          const wf = new THREE.WireframeGeometry(mesh.geometry);
          const line = new THREE.LineSegments(wf, 
            new THREE.LineBasicMaterial({{ color: 0x000000, linewidth: 1 }}));
          mesh.add(line);
          wireframeMeshes.push(line);
        }});
      }} else {{
        wireframeMeshes.forEach(wf => wf.parent.remove(wf));
        wireframeMeshes = [];
      }}
    }}

    function resetCamera() {{
      // Reset toggle states to defaults
      // Reset wireframe overlay first (removes overlay meshes)
      if (wireframeOverlayMode) toggleWireframeOverlay();
      // Reset wireframe mode
      if (wireframeMode) toggleWireframe();
      // Reset colors to default (OFF, colorMode=0)
      if (colorMode !== 0) {{
        colorMode = 0;
        updateColorDots();
        meshes.forEach(mesh => {{
          if (!(mesh.material.map && texturesEnabled)) {{
            mesh.material.color.set(0x808080);
          }}
        }});
      }}
      // Reset textures to ON
      if (!texturesEnabled) toggleTextures();

      const box = new THREE.Box3();
      meshes.forEach(m => box.expandByObject(m));
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      
      // Calculate proper camera distance to fill viewport
      const maxDim = Math.max(size.x, size.y, size.z);
      const fov = camera.fov * (Math.PI / 180);
      
      // Take into account both vertical and horizontal FOV
      const aspect = camera.aspect;
      const vFOV = fov;
      const hFOV = 2 * Math.atan(Math.tan(vFOV / 2) * aspect);
      
      // Calculate distance for both dimensions
      const distanceV = maxDim / (2 * Math.tan(vFOV / 2));
      const distanceH = maxDim / (2 * Math.tan(hFOV / 2));
      const cameraDistance = Math.max(distanceV, distanceH);
      const dist = cameraDistance * CONFIG.CAMERA_ZOOM;  // Use config zoom
      
      // Reset to default position
      const direction = new THREE.Vector3(0.5, 0.5, 1).normalize();
      camera.position.copy(center).add(direction.multiplyScalar(dist));
      camera.lookAt(center);
      
      // Reset controls properly
      controls.target.copy(center);
      
      // Set spherical from camera position
      const offset = camera.position.clone().sub(center);
      controls.spherical.setFromVector3(offset);
      controls.panOffset.set(0, 0, 0);
      
      controls.update();
    }}

    function updateStats() {{
      const visibleMeshes = meshes.filter(m => m.visible);
      let totalVerts = 0, totalTris = 0;
      
      visibleMeshes.forEach(m => {{
        totalVerts += m.geometry.attributes.position.count;
        if (m.geometry.index) totalTris += m.geometry.index.count / 3;
      }});
      
      document.getElementById('vertices').textContent = `Vertices: ${{totalVerts.toLocaleString()}}`;
      document.getElementById('triangles').textContent = `Triangles: ${{totalTris.toLocaleString()}}`;
      document.getElementById('visible').textContent = `Visible: ${{visibleMeshes.length}}/${{meshes.length}}`;
    }}

    let currentScreenshotPath = null;

    function takeScreenshot() {{
      // Render scény pro čistý screenshot
      renderer.render(scene, camera);
      
      // Získat data z canvasu jako PNG
      const dataURL = renderer.domElement.toDataURL('image/png');
      
      // Check if pywebview API is available
      if (window.pywebview && window.pywebview.api) {{
        // Use Python API to save file
        window.pywebview.api.save_screenshot(dataURL).then(result => {{
          const modal = document.getElementById('screenshot-modal');
          const pathElement = document.getElementById('screenshot-path');
          
          if (result.success) {{
            currentScreenshotPath = result.filepath;
            pathElement.textContent = result.filepath;
            pathElement.onclick = openScreenshot;
            pathElement.style.cursor = 'pointer';
            pathElement.title = 'Click to open';
            modal.classList.add('show');
            console.log(`Screenshot saved to: ${{result.filepath}}`);
          }} else {{
            alert(`Error saving screenshot: ${{result.error}}`);
          }}
        }}).catch(error => {{
          alert(`Error: ${{error}}`);
          console.error('Screenshot error:', error);
        }});
      }} else {{
        // Fallback: download via browser (for non-pywebview environments)
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
        const filename = `screenshot_${{timestamp}}.png`;
        
        const link = document.createElement('a');
        link.download = filename;
        link.href = dataURL;
        link.click();
        
        const modal = document.getElementById('screenshot-modal');
        const pathElement = document.getElementById('screenshot-path');
        pathElement.textContent = `Downloads/${{filename}}`;
        pathElement.onclick = null;
        pathElement.style.cursor = 'default';
        modal.classList.add('show');
        
        console.log(`Screenshot downloaded as: ${{filename}}`);
      }}
    }}

    function openScreenshot() {{
      if (currentScreenshotPath && window.pywebview && window.pywebview.api) {{
        window.pywebview.api.open_file(currentScreenshotPath).then(result => {{
          if (!result.success) {{
            alert(`Error opening file: ${{result.error}}`);
          }}
        }}).catch(error => {{
          console.error('Error opening screenshot:', error);
        }});
      }}
    }}

    function closeScreenshotModal() {{
      const modal = document.getElementById('screenshot-modal');
      modal.classList.remove('show');
    }}


    let lastTime = Date.now();
    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
      
      const now = Date.now();
      const fps = Math.round(1000 / (now - lastTime));
      document.getElementById('fps').textContent = `FPS: ${{fps}}`;
      lastTime = now;
    }}

    function onWindowResize() {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }}

    function toggleControlsPanel() {{
      const panel = document.getElementById('controls');
      const btn = document.getElementById('controls-toggle');
      panel.classList.toggle('collapsed');
      btn.textContent = panel.classList.contains('collapsed') ? '☰' : '✕';
    }}

    init();
  </script>
</body>
</html>
"""

    return html_content


# -----------------------------
# WebView API for screenshots
# -----------------------------
class Api:
    """API for communication between JavaScript and Python."""
    
    def __init__(self, screenshot_dir: str):
        # Use Downloads folder instead of MDL location
        import os
        if os.name == 'nt':  # Windows
            downloads = Path.home() / 'Downloads'
        else:  # Linux/Mac
            downloads = Path.home() / 'Downloads'
        
        self.screenshot_dir = str(downloads)
    
    def save_screenshot(self, image_data: str) -> dict:
        """
        Save screenshot to Downloads folder.
        
        Args:
            image_data: Base64 encoded PNG image data (with data:image/png;base64, prefix)
        
        Returns:
            dict with 'success', 'filepath', and 'filename'
        """
        try:
            import base64
            
            # Remove data URL prefix
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            
            # Generate filename with timestamp
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"screenshot_{timestamp}.png"
            filepath = Path(self.screenshot_dir) / filename
            
            # Save file
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
            
            print(f"[Screenshot] Saved to: {filepath}")
            
            return {
                'success': True,
                'filepath': str(filepath.absolute()),
                'filename': filename
            }
        
        except Exception as e:
            print(f"[Screenshot] Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def open_file(self, filepath: str) -> dict:
        """
        Open file in default system viewer.
        
        Args:
            filepath: Path to the file to open
            
        Returns:
            dict with 'success'
        """
        try:
            import subprocess
            import os
            
            filepath = Path(filepath)
            
            if not filepath.exists():
                return {'success': False, 'error': 'File not found'}
            
            # Open file with default application
            if os.name == 'nt':  # Windows
                os.startfile(filepath)
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', str(filepath)])
            else:  # Linux
                subprocess.run(['xdg-open', str(filepath)])
            
            print(f"[Screenshot] Opened: {filepath}")
            return {'success': True}
            
        except Exception as e:
            print(f"[Screenshot] Error opening file: {e}")
            return {'success': False, 'error': str(e)}



# -----------------------------
# Main
# -----------------------------
def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="View MDL 3D models with texture support")
    parser.add_argument("mdl_file", type=str, help="Path to the .mdl file to view")
    parser.add_argument("--use-original-normals", action="store_true", 
                       help="Use original normals from model instead of computing smooth normals")
    
    args = parser.parse_args()
    mdl_path = Path(args.mdl_file)
    
    if not mdl_path.exists():
        print(f"Error: File not found: {mdl_path}")
        sys.exit(1)
    
    if not mdl_path.suffix.lower() == '.mdl':
        print(f"Error: File must be a .mdl file, got: {mdl_path.suffix}")
        sys.exit(1)
    
    # Get script directory
    if getattr(sys, 'frozen', False):
        # Running as compiled executable (PyInstaller)
        script_dir = Path(sys._MEIPASS)  # PyInstaller temp directory
    else:
        # Running as normal Python script
        script_dir = Path(__file__).parent
    
    print(f"\n{'='*60}")
    print(f"MDL Viewer with Texture Support")
    print(f"{'='*60}")
    print(f"Model: {mdl_path.name}")
    print(f"Path: {mdl_path.parent}")
    
    # Create temp directory
    temp_dir = Path(tempfile.gettempdir()) / f"mdl_viewer_{int(time.time())}"
    temp_dir.mkdir(exist_ok=True)
    TEMP_FILES.append(temp_dir)
    
    print(f"\nTemp directory: {temp_dir}")
    
    # Copy three.min.js if exists
    local_threejs = script_dir / "three.min.js"
    if local_threejs.exists():
        temp_threejs = temp_dir / "three.min.js"
        shutil.copy2(local_threejs, temp_threejs)
        print(f"[OK] Copied three.min.js to temp")
    else:
        print(f"[WARNING] three.min.js not found - will use CDN (requires internet)")
    
    # Load MDL with textures
    try:
        meshes, material_texture_map = load_mdl_with_textures(
            mdl_path, 
            temp_dir,
            use_original_normals=args.use_original_normals
        )
    except Exception as e:
        print(f"\n[ERROR] Error loading MDL file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if not meshes:
        print("\n[ERROR] No meshes found in MDL file")
        sys.exit(1)
    
    # Generate HTML
    html_content = generate_html_with_textures(mdl_path, meshes, material_texture_map)
    
    # Write HTML to temp
    temp_file = temp_dir / f"{mdl_path.stem}_viewer.html"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"\n{'='*60}")
    print(f"Opening viewer...")
    print(f"{'='*60}\n")
    
    # Open in browser using webview
    try:
        import webview
        
        # Create API instance with string path (to avoid serialization issues)
        api = Api(str(mdl_path.parent.absolute()))
        
        window = webview.create_window(
            title=f"Model Viewer - {mdl_path.name}",
            url=f'file:///{temp_file.absolute().as_posix()}',
            width=1280,
            height=800,
            resizable=True,
            fullscreen=False,
            maximized=True,
            js_api=api
        )
        
        print("[OK] Window opened. Close it to exit.")
        webview.start()
        print("\n[OK] Window closed. Cleaning up...")
        
    except ImportError:
        print("[WARNING] pywebview not installed, using default browser")
        import webbrowser
        webbrowser.open(f'file://{temp_file.absolute()}')
        
        print("\nPress Ctrl+C to exit and clean up...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting...")
    
    print("\n[Cleanup] Removing temporary files...")


if __name__ == "__main__":
    main()
