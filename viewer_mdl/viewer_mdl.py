#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viewer_mdl.py — Direct .mdl preview without intermediate .fmt/.vb/.ib

Dependencies (shared with your MDL utility):
  - kuro_mdl_export_meshes.py  (in the same directory)
  - lib_fmtibvb.py             (in the same directory, imported by the parser)
  - blowfish, zstandard        (for CLE assets: pip install blowfish zstandard)

Usage:
  python viewer_mdl.py /path/to/model.mdl [--use-original-normals]

Output:
  - Generates HTML preview:  <mdl_basename>_viewer.html
  - Tries to open it in the default web browser
"""

from pathlib import Path
import sys
import json
import numpy as np

# Import ONLY the necessary functions from your parser
from kuro_mdl_export_meshes import decryptCLE, obtain_material_data, obtain_mesh_data  # type: ignore


# -----------------------------
# Smooth normals (lightweight version)
# -----------------------------
def compute_smooth_normals_with_sharing(vertices: np.ndarray, indices: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """Compute smooth normals with position sharing (within given tolerance)."""
    from collections import defaultdict

    position_map = {}
    vertex_to_position = {}
    for idx, v in enumerate(vertices):
        key = tuple(np.round(v / tolerance) * tolerance)
        position_map.setdefault(key, []).append(idx)
        vertex_to_position[idx] = key

    position_normals = defaultdict(lambda: np.zeros(3, dtype=np.float32))

    # Accumulate face normals per shared position
    for i in range(0, len(indices), 3):
        i0, i1, i2 = int(indices[i]), int(indices[i + 1]), int(indices[i + 2])
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        position_normals[vertex_to_position[i0]] += face_normal
        position_normals[vertex_to_position[i1]] += face_normal
        position_normals[vertex_to_position[i2]] += face_normal

    # Normalize and expand back to per-vertex array
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
# Load MDL → vertices/normals/indices (via your parser)
# -----------------------------
def load_mdl_direct(mdl_path: Path, use_original_normals: bool = False):
    """
    Returns a list of meshes of the form:
      {
        'name': str,
        'vertices': np.ndarray [N,3],
        'normals':  np.ndarray [N,3] or None,
        'indices':  np.ndarray [M] (uint32)
      }
    """
    mdl_path = Path(mdl_path)
    with open(mdl_path, "rb") as f:
        mdl_data = f.read()

    # Decrypt/Decompress CLE container (if applicable)
    mdl_data = decryptCLE(mdl_data)

    # Materials: required by the parser for proper element mapping (not used in the preview itself)
    material_struct = obtain_material_data(mdl_data)

    # Extract mesh structures (contain 'mesh_buffers' with IB/VB and fmt)
    mesh_struct = obtain_mesh_data(mdl_data, material_struct=material_struct)

    meshes = []
    mesh_blocks = mesh_struct.get("mesh_blocks", [])
    all_buffers = mesh_struct.get("mesh_buffers", [])

    for i, submesh_list in enumerate(all_buffers):
        # Mesh block name (e.g., "body", "hair") if available
        base_name = mesh_blocks[i].get("name", f"mesh_{i}") if i < len(mesh_blocks) else f"mesh_{i}"

        for j, submesh in enumerate(submesh_list):
            vb = submesh.get("vb", [])
            ib = submesh.get("ib", {}).get("Buffer", [])

            # Find POSITION and (optionally) NORMAL
            pos_buffer = None
            normal_buffer = None

            for element in vb:
                sem = element.get("SemanticName")
                buf = element.get("Buffer")
                if sem == "POSITION":
                    pos_buffer = buf
                elif sem == "NORMAL":
                    normal_buffer = buf

            if not pos_buffer:
                # Without positions we can't render anything
                continue

            # Positions — take first 3 components
            vertices = np.array([p[:3] for p in pos_buffer], dtype=np.float32)

            # Indices — stored in triangles; flatten to 1D index buffer
            flat_indices = []
            for tri in ib:
                if len(tri) == 3:
                    flat_indices.extend(tri)
            indices = np.array(flat_indices, dtype=np.uint32)

            # Normals — use original if requested and present, otherwise compute
            if use_original_normals and normal_buffer:
                normals = np.array([n[:3] for n in normal_buffer], dtype=np.float32)
                # Ensure unit-length (SNORM may be pre-normalized, normalize just in case)
                lens = np.linalg.norm(normals, axis=1)
                nonzero = lens > 1e-8
                normals[nonzero] = normals[nonzero] / lens[nonzero][:, None]
            else:
                normals = compute_smooth_normals_with_sharing(vertices, indices) if len(indices) >= 3 else None

            meshes.append({
                "name": f"{i}_{base_name}_{j:02d}",
                "vertices": vertices,
                "normals": normals,
                "indices": indices
            })

    return meshes


# -----------------------------
# Export HTML (Three.js) – fixed script tag + fixed toggleColors()
# -----------------------------
def export_html_from_meshes(mdl_path: Path, meshes: list):
    """
    Create an HTML file <mdl_basename>_viewer.html with (positions, normals, indices).
    The template uses safe JS string concatenation (no backtick template strings).
    """

    # Build the JSON payload for the client-side renderer
    meshes_data = []
    for m in meshes:
        if m["vertices"] is None or m["indices"] is None:
            continue
        verts = m["vertices"]
        norms = m["normals"]
        idxs = m["indices"]
        if norms is None:
            # As a fallback, compute normals here (just in case)
            norms = compute_smooth_normals_with_sharing(verts, idxs)
        meshes_data.append({
            "name": m["name"],
            "vertices": verts.astype(np.float32).flatten().tolist(),
            "normals": norms.astype(np.float32).flatten().tolist(),
            "indices": idxs.astype(np.uint32).tolist()
        })

    mdl_name = Path(mdl_path).name

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Model Viewer - {mdl_name}</title>
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

    /* === Action buttons === */
    .btn-action {{
      background: linear-gradient(135deg, #6d28d9, #7c3aed, #9333ea); border: none;
      color: white; padding: 11px 16px; margin: 4px 0; cursor: pointer;
      border-radius: 8px; width: 100%; font-weight: 600; font-size: 13px;
      display: flex; align-items: center; gap: 8px; justify-content: center;
      transition: all 0.15s ease;
    }}
    .btn-action:hover {{ filter: brightness(1.15); transform: translateY(-1px); box-shadow: 0 4px 16px rgba(124, 58, 237, 0.35); }}

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

    /* === Cycle dots (3-state Colors) === */
    .cycle-dots {{ display: flex; gap: 6px; align-items: center; flex-shrink: 0; }}
    .cycle-dot {{
      width: 18px; height: 18px; border-radius: 50%; border: 2px solid transparent;
      transition: all 0.2s ease; opacity: 0.4;
    }}
    .cycle-dot.dot-off {{ background: #808080; }}
    .cycle-dot.dot-color {{ background: linear-gradient(135deg, #ff6b6b, #4ecdc4, #ffe66d); }}
    .cycle-dot.dot-white {{ background: #ffffff; }}
    .cycle-dot.current {{ opacity: 1; border-color: #a78bfa; transform: scale(1.15); box-shadow: 0 0 8px rgba(167, 139, 250, 0.5); }}

    /* === Mesh toggles === */
    .mesh-toggle {{
      display: flex; align-items: center; margin: 4px 0; padding: 7px 10px;
      background: rgba(124, 58, 237, 0.1); border-radius: 6px; transition: background 0.2s;
    }}
    .mesh-toggle:hover {{ background: rgba(124, 58, 237, 0.2); }}
    .mesh-toggle input {{ margin-right: 10px; cursor: pointer; width: 16px; height: 16px; accent-color: #7c3aed; }}
    .mesh-toggle label {{ cursor: pointer; flex-grow: 1; font-size: 12px; }}

    /* === Select / dropdown === */
    .styled-select {{
      width: 100%; padding: 9px 12px; margin-bottom: 6px;
      background: #2a2a3e; color: #e0e0e0;
      border: 1px solid rgba(124, 58, 237, 0.3); border-radius: 8px;
      font-size: 13px; cursor: pointer; outline: none;
    }}
    .styled-select:focus {{ border-color: #7c3aed; }}
    .styled-select option {{ background: #2a2a3e; color: #e0e0e0; padding: 6px; }}

    /* === Slider rows === */
    .slider-row {{
      display: flex; align-items: center; gap: 8px;
      padding: 6px 14px; margin: 2px 0;
    }}

    /* === Section title (collapsible) === */
    .section-title {{
      font-size: 13px; font-weight: 600; color: #a78bfa; margin: 14px 0 8px 0;
      padding-bottom: 6px; border-bottom: 1px solid rgba(124, 58, 237, 0.2);
      display: flex; align-items: center; gap: 6px;
    }}

    /* === Info helpers === */
    .info-text {{ font-size: 11px; color: #9ca3af; }}
    .info-badge {{
      background: rgba(124, 58, 237, 0.15); padding: 10px; border-radius: 8px;
      font-size: 11px; margin-bottom: 8px;
    }}
    .info-badge .row {{ display: flex; align-items: center; gap: 8px; }}
    .info-badge .row + .row {{ margin-top: 6px; }}

    /* === Modal === */
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
    .modal-content .filename:hover {{ background: rgba(124, 58, 237, 0.3); }}
    .modal-content button {{ max-width: 200px; margin: 10px auto; }}

    /* === Scrollbar === */
    #controls::-webkit-scrollbar {{ width: 6px; }}
    #controls::-webkit-scrollbar-track {{ background: transparent; }}
    #controls::-webkit-scrollbar-thumb {{ background: rgba(124, 58, 237, 0.3); border-radius: 3px; }}
    #controls::-webkit-scrollbar-thumb:hover {{ background: rgba(124, 58, 237, 0.5); }}
  </style>
</head>
<body>
  <button id="controls-toggle" onclick="toggleControlsPanel()">☰</button>
  <div id="container"></div>
  <div id="info" class="panel">
    <h3>🎮 Model Viewer</h3>
    <p style="font-size: 13px; color: #b0b0b0; line-height: 1.5; margin-bottom: 12px;">
      <strong style="color: #a78bfa;">{mdl_name}</strong><br>
      <span style="font-size: 11px; color: #9ca3af;">direct .mdl (no .fmt/.ib/.vb)</span>
    </p>
    <div class="info-badge">
      <div class="row">
        <span style="font-size: 16px;">🖱️</span>
        <span style="color: #9ca3af;">Left: Rotate</span>
      </div>
      <div class="row">
        <span style="font-size: 16px;">🖱️</span>
        <span style="color: #9ca3af;">Right: Pan</span>
      </div>
      <div class="row">
        <span style="font-size: 16px;">🔄</span>
        <span style="color: #9ca3af;">Wheel: Zoom</span>
      </div>
    </div>
  </div>
  <div id="controls" class="panel">
    <div class="section-title">🎮 Controls</div>

    <div class="toggle-row" onclick="toggleColors()">
      <span class="label">🎨 Colors</span>
      <span class="cycle-dots">
        <span class="cycle-dot dot-color current" title="Per-mesh colors"></span>
        <span class="cycle-dot dot-off" title="Gray"></span>
        <span class="cycle-dot dot-white" title="HSL rainbow"></span>
      </span>
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

    <button class="btn-action" onclick="resetCamera()">🔄 Reset Camera</button>
    <button class="btn-action" onclick="takeScreenshot()">📸 Screenshot</button>

    <div class="section-title" style="cursor:pointer;user-select:none;" onclick="var el=document.getElementById('captureSettings'); el.style.display=el.style.display==='none'?'block':'none'; this.querySelector('.arrow').textContent=el.style.display==='none'?'▶':'▼';">⚙️ Capture Settings <span class="arrow" style="font-size:10px;margin-left:4px;">▶</span></div>
    <div id="captureSettings" style="display:none;">
      <div class="slider-row">
        <span class="info-text" style="min-width:72px;">Screenshot:</span>
        <select id="screenshotScale" class="styled-select" style="width:auto;flex:1;margin:0;padding:6px 8px;">
          <option value="1">1× (native)</option>
          <option value="2" selected>2× (double)</option>
          <option value="4">4× (ultra)</option>
        </select>
      </div>
    </div>

    <div class="section-title" style="cursor:pointer;user-select:none;" onclick="var el=document.getElementById('meshSection'); el.style.display=el.style.display==='none'?'block':'none'; this.querySelector('.arrow').textContent=el.style.display==='none'?'▶':'▼';">📦 Meshes <span class="arrow" style="font-size:10px;margin-left:4px;">▶</span></div>
    <div id="meshSection" style="display:none;">
      <button class="btn-action" onclick="toggleAllMeshes(true)">✅ Show All</button>
      <button class="btn-action" onclick="toggleAllMeshes(false)">❌ Hide All</button>
      <div id="mesh-list"></div>
    </div>
  </div>
  <div id="stats" class="panel">
    <div id="fps">FPS: 60</div>
    <div>Vertices: <span id="vertices">0</span></div>
    <div>Triangles: <span id="triangles">0</span></div>
    <div>Visible: <span id="visible">0</span></div>
  </div>

  <div id="screenshot-modal">
    <div class="modal-content">
      <h3>📸 Screenshot Saved</h3>
      <p>Your screenshot has been downloaded:</p>
      <div class="filename" id="screenshot-filename">filename.png</div>
      <button class="btn-action" onclick="closeScreenshotModal()">Close</button>
    </div>
  </div>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script>
    var CONFIG = {{
      CAMERA_ZOOM: 1.5,
      AUTO_HIDE_SHADOW: true,
      INITIAL_BACKGROUND: 0x1a1a2e
    }};

    var data = {json.dumps(meshes_data)};
    var scene, camera, renderer, controls, meshes = [];
    var wireframeMode = false, wireframeOverlayMode = false, colorMode = 0;
    var wireframeMeshes = [];

    function toggleControlsPanel() {{
      var panel = document.getElementById('controls');
      panel.classList.toggle('collapsed');
    }}

    // Minimal OrbitControls (custom, independent of three/examples)
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

        this.domElement.addEventListener('contextmenu', function(e) {{ e.preventDefault(); }});
        this.domElement.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.domElement.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.domElement.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.domElement.addEventListener('wheel', this.onMouseWheel.bind(this));
      }}

      onMouseDown(e) {{
        this.isMouseDown = true;
        this.mouseButton = e.button;
        this.lastX = e.clientX;
        this.lastY = e.clientY;
      }}

      onMouseUp() {{
        this.isMouseDown = false;
      }}

      onMouseMove(e) {{
        if (!this.isMouseDown) return;
        var dx = e.clientX - this.lastX;
        var dy = e.clientY - this.lastY;
        this.lastX = e.clientX;
        this.lastY = e.clientY;

        if (this.mouseButton === 0) {{
          this.sphericalDelta.theta -= 2 * Math.PI * dx / this.domElement.clientHeight * this.rotateSpeed;
          this.sphericalDelta.phi -= 2 * Math.PI * dy / this.domElement.clientHeight * this.rotateSpeed;
          this.update();
        }} else if (this.mouseButton === 2) {{
          var dist = this.camera.position.distanceTo(this.target);
          var factor = dist * Math.tan((this.camera.fov / 2) * Math.PI / 180.0);
          var left = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 0);
          left.multiplyScalar(-2 * dx * factor / this.domElement.clientHeight * this.panSpeed);
          var up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 1);
          up.multiplyScalar(2 * dy * factor / this.domElement.clientHeight * this.panSpeed);
          this.panOffset.add(left).add(up);
          this.update();
        }}
      }}

      onMouseWheel(e) {{
        e.preventDefault();
        this.scale *= (e.deltaY < 0) ? 0.95 : 1.05;
        this.update();
      }}

      update() {{
        var offset = new THREE.Vector3();
        var quat = new THREE.Quaternion().setFromUnitVectors(this.camera.up, new THREE.Vector3(0,1,0));
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
        this.sphericalDelta.set(0,0,0);
        this.scale = 1;
        this.panOffset.set(0,0,0);
      }}
    }}

    function init() {{
      scene = new THREE.Scene();
      scene.background = new THREE.Color(CONFIG.INITIAL_BACKGROUND);
      camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.1, 1000);
      camera.position.set(2, 1.2, 2);
      renderer = new THREE.WebGLRenderer({{antialias: true, preserveDrawingBuffer: true}});
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      document.getElementById('container').appendChild(renderer.domElement);

      scene.add(new THREE.AmbientLight(0xffffff, 0.5));
      var light = new THREE.DirectionalLight(0xffffff, 0.8);
      light.position.set(5, 5, 5);
      scene.add(light);

      var colors = [0xE8B4B8, 0xB8D4E8, 0xC8E8B8, 0xE8D4B8, 0xD8B8E8, 0xE8E8B8];
      var totalVerts = 0;
      var totalTriangles = 0;
      var meshListDiv = document.getElementById('mesh-list');

      data.forEach(function(m, i) {{
        var geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(m.vertices), 3));
        geo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(m.normals), 3));
        geo.setIndex(new THREE.BufferAttribute(new Uint32Array(m.indices), 1));

        var mat = new THREE.MeshPhongMaterial({{
          color: colors[i % colors.length],
          side: THREE.DoubleSide,
          flatShading: false,
          shininess: 30
        }});

        var mesh = new THREE.Mesh(geo, mat);
        mesh.name = m.name;
        mesh.userData.baseColor = colors[i % colors.length];

        // Auto-hide shadow meshes
        if (CONFIG.AUTO_HIDE_SHADOW && m.name.toLowerCase().indexOf('shadow') !== -1) {{
          mesh.visible = false;
        }}

        scene.add(mesh);
        meshes.push(mesh);
        totalVerts += m.vertices.length / 3;
        totalTriangles += m.indices.length / 3;

        var toggleDiv = document.createElement('div');
        toggleDiv.className = 'mesh-toggle';
        toggleDiv.innerHTML =
          '<input type="checkbox" id="mesh-' + i + '" ' + (mesh.visible ? 'checked' : '') + ' onchange="toggleMesh(' + i + ')">' +
          '<label for="mesh-' + i + '">' + m.name + '</label>';
        meshListDiv.appendChild(toggleDiv);
      }});

      controls = new OrbitControls(camera, renderer.domElement);
      scene.add(new THREE.GridHelper(10, 10, 0x444444, 0x222222));

      var box = new THREE.Box3();
      meshes.forEach(function(m) {{ if (m.visible) box.expandByObject(m); }});
      var center = box.getCenter(new THREE.Vector3());
      var size = box.getSize(new THREE.Vector3());
      var dist = Math.max(size.x, size.y, size.z) * CONFIG.CAMERA_ZOOM;
      camera.position.set(center.x + dist*0.5, center.y + dist*0.5, center.z + dist*0.5);
      camera.lookAt(center);
      controls.target.copy(center);
      controls.update();

      document.getElementById('vertices').textContent = totalVerts.toLocaleString();
      document.getElementById('triangles').textContent = totalTriangles.toLocaleString();
      var visibleCount = meshes.filter(function(m) {{ return m.visible; }}).length;
      document.getElementById('visible').textContent = visibleCount + '/' + meshes.length;

      window.addEventListener('resize', function() {{
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      }});

      animate();
    }}

    function toggleMesh(index) {{
      meshes[index].visible = document.getElementById('mesh-' + index).checked;
      var visibleCount = meshes.filter(function(m) {{ return m.visible; }}).length;
      document.getElementById('visible').textContent = visibleCount + '/' + meshes.length;
    }}

    function toggleAllMeshes(visible) {{
      meshes.forEach(function(mesh, index) {{
        mesh.visible = visible;
        document.getElementById('mesh-' + index).checked = visible;
      }});
      document.getElementById('visible').textContent = (visible ? meshes.length : 0) + '/' + meshes.length;
    }}

    function toggleColors() {{
      colorMode = (colorMode + 1) % 3;
      var dots = document.querySelectorAll('.cycle-dot');
      dots.forEach(function(d) {{ d.classList.remove('current'); }});
      dots[colorMode].classList.add('current');

      meshes.forEach(function(mesh, index) {{
        if (colorMode === 0) {{
          mesh.material.color.setHex(mesh.userData.baseColor);
        }} else if (colorMode === 1) {{
          mesh.material.color.setHex(0xCCCCCC);
        }} else {{
          mesh.material.color.setHSL(index / meshes.length, 0.7, 0.6);
        }}
      }});
    }}

    function toggleWireframe() {{
      wireframeMode = !wireframeMode;
      document.getElementById('swWire').checked = wireframeMode;
      meshes.forEach(function(m) {{ m.material.wireframe = wireframeMode; }});
      // Disable overlay if wireframe-only is on
      if (wireframeMode && wireframeOverlayMode) {{
        wireframeOverlayMode = false;
        document.getElementById('swWireOver').checked = false;
        wireframeMeshes.forEach(function(wm) {{ scene.remove(wm); wm.geometry.dispose(); wm.material.dispose(); }});
        wireframeMeshes = [];
      }}
    }}

    function toggleWireframeOverlay() {{
      wireframeOverlayMode = !wireframeOverlayMode;
      document.getElementById('swWireOver').checked = wireframeOverlayMode;
      if (wireframeOverlayMode) {{
        // Disable wireframe-only if overlay is on
        if (wireframeMode) {{
          wireframeMode = false;
          document.getElementById('swWire').checked = false;
          meshes.forEach(function(m) {{ m.material.wireframe = false; }});
        }}
        meshes.forEach(function(m) {{
          var wGeo = m.geometry.clone();
          var wMat = new THREE.MeshBasicMaterial({{
            color: 0x000000, wireframe: true, transparent: true, opacity: 0.15
          }});
          var wMesh = new THREE.Mesh(wGeo, wMat);
          wMesh.position.copy(m.position);
          wMesh.rotation.copy(m.rotation);
          wMesh.scale.copy(m.scale);
          wMesh.visible = m.visible;
          scene.add(wMesh);
          wireframeMeshes.push(wMesh);
        }});
      }} else {{
        wireframeMeshes.forEach(function(wm) {{ scene.remove(wm); wm.geometry.dispose(); wm.material.dispose(); }});
        wireframeMeshes = [];
      }}
    }}

    function resetCamera() {{
      var box = new THREE.Box3();
      meshes.forEach(function(m) {{ if (m.visible) box.expandByObject(m); }});
      var center = box.getCenter(new THREE.Vector3());
      var size = box.getSize(new THREE.Vector3());
      var dist = Math.max(size.x, size.y, size.z) * CONFIG.CAMERA_ZOOM;
      camera.position.set(center.x + dist*0.5, center.y + dist*0.5, center.z + dist*0.5);
      camera.lookAt(center);
      controls.target.copy(center);
      controls.update();
    }}

    function takeScreenshot() {{
      var scale = parseInt(document.getElementById('screenshotScale').value) || 2;
      var w = window.innerWidth;
      var h = window.innerHeight;
      var targetW = w * scale;
      var targetH = h * scale;

      if (scale <= 1) {{
        renderer.render(scene, camera);
        finishScreenshot(renderer.domElement.toDataURL('image/png'));
        return;
      }}

      // High-res: render to offscreen WebGLRenderTarget
      var rt = new THREE.WebGLRenderTarget(targetW, targetH, {{
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat
      }});

      renderer.setRenderTarget(rt);
      renderer.render(scene, camera);

      // Read pixels from render target
      var pixels = new Uint8Array(targetW * targetH * 4);
      renderer.readRenderTargetPixels(rt, 0, 0, targetW, targetH, pixels);
      renderer.setRenderTarget(null);
      rt.dispose();

      // Flip Y (WebGL is bottom-up) and write to canvas
      var tmpCanvas = document.createElement('canvas');
      tmpCanvas.width = targetW;
      tmpCanvas.height = targetH;
      var tmpCtx = tmpCanvas.getContext('2d');
      var imageData = tmpCtx.createImageData(targetW, targetH);
      for (var y = 0; y < targetH; y++) {{
        var srcRow = (targetH - 1 - y) * targetW * 4;
        var dstRow = y * targetW * 4;
        imageData.data.set(pixels.subarray(srcRow, srcRow + targetW * 4), dstRow);
      }}
      tmpCtx.putImageData(imageData, 0, 0);

      finishScreenshot(tmpCanvas.toDataURL('image/png'));
    }}

    function finishScreenshot(dataURL) {{
      var timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
      var filename = '{Path(mdl_path).stem}_' + timestamp + '.png';
      var link = document.createElement('a');
      link.download = filename;
      link.href = dataURL;
      link.click();
      document.getElementById('screenshot-filename').textContent = filename;
      document.getElementById('screenshot-modal').classList.add('show');
    }}

    function closeScreenshotModal() {{
      document.getElementById('screenshot-modal').classList.remove('show');
    }}

    var lastTime = performance.now(), frames = 0;
    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      // Sync wireframe overlay visibility
      if (wireframeOverlayMode) {{
        wireframeMeshes.forEach(function(wm, i) {{ if (i < meshes.length) wm.visible = meshes[i].visible; }});
      }}
      renderer.render(scene, camera);
      frames++;
      var time = performance.now();
      if (time >= lastTime + 1000) {{
        document.getElementById('fps').textContent = 'FPS: ' + Math.round(frames * 1000 / (time - lastTime));
        frames = 0; lastTime = time;
      }}
    }}

    init();
  </script>
</body>
</html>"""

    output = Path(mdl_path).with_suffix("")  # strip .mdl
    output = output.parent / f"{output.name}_viewer.html"
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)
    return output


# -----------------------------
# main
# -----------------------------
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("mdl_path", help="Path to .mdl file")
    p.add_argument("--use-original-normals", action="store_true",
                   help="If the .mdl contains normals, use them (otherwise smooth normals are computed).")
    args = p.parse_args()

    mdl_path = Path(args.mdl_path)
    if not mdl_path.exists() or mdl_path.suffix.lower() != ".mdl":
        print("Error: please provide an existing .mdl file.")
        sys.exit(2)

    try:
        print(f"[1/3] Reading MDL: {mdl_path}")
        meshes = load_mdl_direct(mdl_path, use_original_normals=args.use_original_normals)
        if not meshes:
            print("No renderable mesh was found in the file.")
            sys.exit(1)

        print(f"[2/3] Mesh count: {len(meshes)}")
        total_verts = sum(len(m["vertices"]) for m in meshes if m["vertices"] is not None)
        total_tris = sum(len(m["indices"]) // 3 for m in meshes if m["indices"] is not None)
        print(f"       Vertices: {total_verts:,} | Triangles: {total_tris:,}")

        print("[3/3] Generating HTML…")
        out = export_html_from_meshes(mdl_path, meshes)
        print(f"[OK] Created: {out}")

        try:
            import webbrowser
            webbrowser.open(f"file://{out.absolute()}")
            print("[OK] Opened in default browser.")
        except Exception:
            print(f"Please open manually: {out.absolute()}")

    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
