#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viewer_mdl.py — Direct .mdl preview in native window (NO FILES LEFT BEHIND!)

Requires: pip install pywebview
- Windows: automatically uses Edge WebView2
- Linux: uses GTK + WebKit2
- macOS: uses WKWebView

Dependencies (shared with your MDL utility):
  - kuro_mdl_export_meshes.py  (in the same directory)
  - lib_fmtibvb.py             (in the same directory, imported by the parser)
  - blowfish, zstandard        (for CLE assets: pip install blowfish zstandard)

Usage:
  pip install pywebview
  python viewer_mdl_window.py /path/to/model.mdl [--use-original-normals]

Output:
  - Opens native window with WebView
  - Automatically exits when window is closed
  - NO HTML file on disk!
"""

from pathlib import Path
import sys
import json
import numpy as np
import tempfile
import atexit
import time

# Import ONLY the necessary functions from your parser
from kuro_mdl_export_meshes import decryptCLE, obtain_material_data, obtain_mesh_data  # type: ignore


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
                    import shutil
                    shutil.rmtree(filepath)
                    print(f"[CLEANUP] Deleted directory: {filepath}")
                else:
                    filepath.unlink()
                    print(f"[CLEANUP] Deleted: {filepath}")
        except Exception:
            pass

atexit.register(cleanup_temp_files)


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
# Load MDL
# -----------------------------
def load_mdl_direct(mdl_path: Path, use_original_normals: bool = False):
    """Returns a list of meshes."""
    mdl_path = Path(mdl_path)
    with open(mdl_path, "rb") as f:
        mdl_data = f.read()

    mdl_data = decryptCLE(mdl_data)
    material_struct = obtain_material_data(mdl_data)
    mesh_struct = obtain_mesh_data(mdl_data, material_struct=material_struct)

    meshes = []
    mesh_blocks = mesh_struct.get("mesh_blocks", [])
    all_buffers = mesh_struct.get("mesh_buffers", [])

    for i, submesh_list in enumerate(all_buffers):
        base_name = mesh_blocks[i].get("name", f"mesh_{i}") if i < len(mesh_blocks) else f"mesh_{i}"

        for j, submesh in enumerate(submesh_list):
            vb = submesh.get("vb", [])
            ib = submesh.get("ib", {}).get("Buffer", [])

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
                continue

            vertices = np.array([p[:3] for p in pos_buffer], dtype=np.float32)

            flat_indices = []
            for tri in ib:
                if len(tri) == 3:
                    flat_indices.extend(tri)
            indices = np.array(flat_indices, dtype=np.uint32)

            if use_original_normals and normal_buffer:
                normals = np.array([n[:3] for n in normal_buffer], dtype=np.float32)
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
# Generate HTML content (as string)
# -----------------------------
def generate_html_content(mdl_path: Path, meshes: list, script_dir: Path = None) -> str:
    """Generate HTML content without writing to file.
    
    Args:
        mdl_path: Path to the MDL file
        meshes: List of mesh data
        script_dir: Directory where the script is located (for finding local three.min.js)
    """
    
    meshes_data = []
    for m in meshes:
        if m["vertices"] is None or m["indices"] is None:
            continue
        verts = m["vertices"]
        norms = m["normals"]
        idxs = m["indices"]
        if norms is None:
            norms = compute_smooth_normals_with_sharing(verts, idxs)
        meshes_data.append({
            "name": m["name"],
            "vertices": verts.astype(np.float32).flatten().tolist(),
            "normals": norms.astype(np.float32).flatten().tolist(),
            "indices": idxs.astype(np.uint32).tolist()
        })

    html = f"""<!DOCTYPE html>
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
    #info {{ top: 20px; left: 20px; max-width: 200px; }}
    #controls {{ top: 20px; right: 20px; max-height: 85vh; overflow-y: auto; min-width: 180px; transition: transform 0.3s ease; }}
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
             border-radius: 6px; width: 100%; font-weight: 600; }}
    button:hover {{ transform: translateY(-1px); }}
    .mesh-toggle {{
      display: flex; align-items: center; margin: 8px 0; padding: 8px;
      background: rgba(124, 58, 237, 0.1); border-radius: 6px; transition: background 0.2s;
    }}
    .mesh-toggle:hover {{ background: rgba(124, 58, 237, 0.2); }}
    .mesh-toggle input {{ margin-right: 10px; cursor: pointer; width: 18px; height: 18px; }}
    .mesh-toggle label {{ cursor: pointer; flex-grow: 1; font-size: 13px; }}
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
    <button onclick="toggleColors()">🎨 Toggle Colors</button>
    <button onclick="toggleWireframe()">📐 Wireframe Only</button>
    <button onclick="toggleWireframeOverlay()">🔲 Wireframe Overlay</button>
    <button onclick="resetCamera()">🎯 Reset Camera</button>
    <h4>👁️ Meshes</h4>
    <div id="mesh-list"></div>
  </div>
  <div id="stats" class="panel">
    <div id="vertices"></div>
    <div id="triangles"></div>
    <div id="visible"></div>
    <div id="fps"></div>
  </div>

  <script src="three.min.js"></script>
  <script>
    const data = {json.dumps(meshes_data)};
    let scene, camera, renderer, controls, meshes = [];
    let wireframeMode = false, wireframeOverlayMode = false, colorMode = 0;
    let wireframeMeshes = [];

    class OrbitControls {{
      constructor(camera, domElement) {{
        this.camera = camera; this.domElement = domElement;
        this.target = new THREE.Vector3();
        this.spherical = new THREE.Spherical();
        this.sphericalDelta = new THREE.Spherical();
        this.scale = 1; this.panOffset = new THREE.Vector3();
        this.isMouseDown = false; this.rotateSpeed = 0.5; this.zoomSpeed = 1; this.panSpeed = 1;
        this.mouseButtons = {{LEFT: 0, MIDDLE: 1, RIGHT: 2}};
        this.domElement.addEventListener('contextmenu', function(e) {{ e.preventDefault(); }});
        this.domElement.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.domElement.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.domElement.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.domElement.addEventListener('wheel', this.onMouseWheel.bind(this));
      }}
      onMouseDown(e) {{ this.isMouseDown = true; this.mouseButton = e.button; }}
      onMouseUp() {{ this.isMouseDown = false; }}
      onMouseMove(e) {{
        if (!this.isMouseDown) return;
        const dx = e.movementX * this.rotateSpeed * 0.01;
        const dy = e.movementY * this.rotateSpeed * 0.01;
        if (this.mouseButton === this.mouseButtons.LEFT) {{
          this.sphericalDelta.theta -= dx; this.sphericalDelta.phi -= dy;
        }} else if (this.mouseButton === this.mouseButtons.RIGHT) {{
          const cam = this.camera, right = new THREE.Vector3(cam.matrix.elements[0], cam.matrix.elements[1], cam.matrix.elements[2]);
          const up = new THREE.Vector3(cam.matrix.elements[4], cam.matrix.elements[5], cam.matrix.elements[6]);
          this.panOffset.add(right.multiplyScalar(-e.movementX * this.panSpeed * 0.003));
          this.panOffset.add(up.multiplyScalar(e.movementY * this.panSpeed * 0.003));
        }}
      }}
      onMouseWheel(e) {{
        e.preventDefault(); this.scale *= Math.pow(0.95, -e.deltaY * this.zoomSpeed * 0.05);
      }}
      update() {{
        const offset = new THREE.Vector3(), quat = new THREE.Quaternion().setFromUnitVectors(this.camera.up, new THREE.Vector3(0,1,0));
        offset.copy(this.camera.position).sub(this.target); offset.applyQuaternion(quat);
        this.spherical.setFromVector3(offset);
        this.spherical.theta += this.sphericalDelta.theta; this.spherical.phi += this.sphericalDelta.phi;
        this.spherical.phi = Math.max(0.01, Math.min(Math.PI - 0.01, this.spherical.phi));
        this.spherical.radius *= this.scale;
        this.target.add(this.panOffset);
        offset.setFromSpherical(this.spherical); offset.applyQuaternion(quat.invert());
        this.camera.position.copy(this.target).add(offset); this.camera.lookAt(this.target);
        this.sphericalDelta.set(0,0,0); this.scale = 1; this.panOffset.set(0,0,0);
      }}
    }}

    function init() {{
      scene = new THREE.Scene(); scene.background = new THREE.Color(0x2a2a3e);
      camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.1, 1000);
      camera.position.set(2, 1.2, 2);

      renderer = new THREE.WebGLRenderer({{antialias: true}});
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      document.getElementById('container').appendChild(renderer.domElement);

      scene.add(new THREE.AmbientLight(0xffffff, 0.5));
      const light = new THREE.DirectionalLight(0xffffff, 0.8);
      light.position.set(5, 5, 5); scene.add(light);

      const colors = [0xE8B4B8, 0xB8D4E8, 0xC8E8B8, 0xE8D4B8, 0xD8B8E8, 0xE8E8B8];
      let totalVerts = 0, totalTriangles = 0;
      const meshListDiv = document.getElementById('mesh-list');

      data.forEach(function(m, i) {{
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(m.vertices), 3));
        geo.setAttribute('normal',   new THREE.BufferAttribute(new Float32Array(m.normals), 3));
        geo.setIndex(new THREE.BufferAttribute(new Uint32Array(m.indices), 1));

        const mat = new THREE.MeshPhongMaterial({{
          color: colors[i % colors.length],
          side: THREE.DoubleSide,
          flatShading: false,
          shininess: 30
        }});
        const mesh = new THREE.Mesh(geo, mat);
        mesh.name = m.name;
        scene.add(mesh); meshes.push(mesh);

        // Create wireframe overlay mesh (initially hidden)
        const wireGeo = new THREE.BufferGeometry();
        wireGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(m.vertices), 3));
        wireGeo.setIndex(new THREE.BufferAttribute(new Uint32Array(m.indices), 1));
        const wireMat = new THREE.LineBasicMaterial({{ color: 0x000000, linewidth: 1 }});
        const wireframe = new THREE.LineSegments(new THREE.WireframeGeometry(wireGeo), wireMat);
        wireframe.visible = false;
        scene.add(wireframe);
        wireframeMeshes.push(wireframe);

        totalVerts += m.vertices.length / 3;
        totalTriangles += m.indices.length / 3;

        const toggleDiv = document.createElement('div');
        toggleDiv.className = 'mesh-toggle';
        toggleDiv.innerHTML =
          '<input type="checkbox" id="mesh-' + i + '" checked onchange="toggleMesh(' + i + ')">' +
          '<label for="mesh-' + i + '">' + m.name + '</label>';
        meshListDiv.appendChild(toggleDiv);
      }});

      // Adjust controls panel width based on longest mesh name
      setTimeout(function() {{
        const labels = document.querySelectorAll('.mesh-toggle label');
        let maxWidth = 180;
        labels.forEach(function(label) {{
          const width = label.offsetWidth + 60; // Add padding for checkbox and margins
          if (width > maxWidth) maxWidth = width;
        }});
        document.getElementById('controls').style.width = Math.min(maxWidth, 400) + 'px';
      }}, 100);

      controls = new OrbitControls(camera, renderer.domElement);
      scene.add(new THREE.GridHelper(10, 10, 0x444444, 0x222222));

      const box = new THREE.Box3();
      meshes.forEach(function(m) {{ box.expandByObject(m); }});
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const dist = Math.max(size.x, size.y, size.z) * 2;
      camera.position.set(center.x + dist*0.5, center.y + dist*0.5, center.z + dist*0.5);
      camera.lookAt(center);
      controls.target.copy(center); controls.update();

      document.getElementById('vertices').textContent = 'Vertices: ' + totalVerts.toLocaleString();
      document.getElementById('triangles').textContent = 'Triangles: ' + totalTriangles.toLocaleString();
      document.getElementById('visible').textContent = 'Visible: ' + meshes.length + '/' + meshes.length;

      window.addEventListener('resize', function() {{
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      }});

      // Show toggle button after initialization
      document.getElementById('controls-toggle').classList.add('visible');

      animate();
    }}

    function toggleMesh(index) {{
      const cb = document.getElementById('mesh-' + index);
      if (cb) {{
        meshes[index].visible = cb.checked;
        if (wireframeOverlayMode) {{
          wireframeMeshes[index].visible = cb.checked;
        }}
      }}
      const visibleCount = meshes.filter(function(m) {{ return m.visible; }}).length;
      document.getElementById('visible').textContent = 'Visible: ' + visibleCount + '/' + meshes.length;
    }}

    function toggleAllMeshes(visible) {{
      meshes.forEach(function(mesh, index) {{
        mesh.visible = visible;
        if (wireframeOverlayMode) {{
          wireframeMeshes[index].visible = visible;
        }}
        const cb = document.getElementById('mesh-' + index);
        if (cb) cb.checked = visible;
      }});
      document.getElementById('visible').textContent = 'Visible: ' + (visible ? meshes.length : 0) + '/' + meshes.length;
    }}

    function toggleColors() {{
      colorMode = (colorMode + 1) % 3;
      meshes.forEach(function (mesh, index) {{
        if (colorMode === 0) {{
          const palette = [0xE8B4B8, 0xB8D4E8, 0xC8E8B8, 0xE8D4B8, 0xD8B8E8, 0xE8E8B8];
          mesh.material.color.setHex(palette[index % palette.length]);
        }} else if (colorMode === 1) {{
          mesh.material.color.setHex(0xCCCCCC);
        }} else {{
          const hue = index / Math.max(1, meshes.length);
          const color = new THREE.Color().setHSL(hue, 0.7, 0.6);
          mesh.material.color.copy(color);
        }}
      }});
    }}

    function toggleWireframe() {{
      wireframeMode = !wireframeMode;
      if (wireframeMode && wireframeOverlayMode) {{
        wireframeOverlayMode = false;
        wireframeMeshes.forEach(function(wf) {{ wf.visible = false; }});
      }}
      meshes.forEach(function(m) {{ m.material.wireframe = wireframeMode; }});
    }}

    function toggleWireframeOverlay() {{
      wireframeOverlayMode = !wireframeOverlayMode;
      if (wireframeOverlayMode && wireframeMode) {{
        wireframeMode = false;
        meshes.forEach(function(m) {{ m.material.wireframe = false; }});
      }}
      wireframeMeshes.forEach(function(wf, i) {{ 
        wf.visible = wireframeOverlayMode && meshes[i].visible; 
      }});
    }}

    function toggleControlsPanel() {{
      const controls = document.getElementById('controls');
      const toggle = document.getElementById('controls-toggle');
      if (controls.classList.contains('collapsed')) {{
        controls.classList.remove('collapsed');
        toggle.textContent = '☰';
      }} else {{
        controls.classList.add('collapsed');
        toggle.textContent = '☰';
      }}
      toggle.classList.add('visible');
    }}

    function resetCamera() {{
      const box = new THREE.Box3();
      meshes.forEach(function(m) {{ box.expandByObject(m); }});
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const dist = Math.max(size.x, size.y, size.z) * 2;
      camera.position.set(center.x + dist*0.5, center.y + dist*0.5, center.z + dist*0.5);
      camera.lookAt(center);
      controls.target.copy(center); controls.update();
    }}

    let lastTime = performance.now(), frames = 0;
    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
      frames++;
      const time = performance.now();
      if (time >= lastTime + 1000) {{
        document.getElementById('fps').textContent = 'FPS: ' + Math.round(frames * 1000 / (time - lastTime));
        frames = 0; lastTime = time;
      }}
    }}
    init();
  </script>
</body>
</html>"""
    
    return html


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
        print(f"[1/4] Reading MDL: {mdl_path}")
        meshes = load_mdl_direct(mdl_path, use_original_normals=args.use_original_normals)
        if not meshes:
            print("No renderable mesh was found in the file.")
            sys.exit(1)

        print(f"[2/4] Mesh count: {len(meshes)}")
        total_verts = sum(len(m["vertices"]) for m in meshes if m["vertices"] is not None)
        total_tris = sum(len(m["indices"]) // 3 for m in meshes if m["indices"] is not None)
        print(f"       Vertices: {total_verts:,} | Triangles: {total_tris:,}")

        print("[3/4] Generating HTML content…")
        
        # Get script directory and check for local three.min.js
        script_dir = Path(__file__).parent
        local_threejs = script_dir / "three.min.js"
        
        if not local_threejs.exists():
            print("\n[WARNING] three.min.js not found in script directory!")
            print(f"Expected location: {local_threejs}")
            print("\nTo use this viewer offline:")
            print("1. Run: python download_libs.py")
            print("2. Keep three.min.js in the same directory as this script")
            print("\nContinuing anyway (will fail if no internet connection)...\n")
        
        html_content = generate_html_content(mdl_path, meshes, script_dir)
        
        print("[4/4] Opening viewer window…")
        try:
            import webview
            import shutil
            
            # Create temp directory for HTML and JS
            temp_dir = Path(tempfile.gettempdir()) / f"mdl_viewer_{int(time.time())}"
            temp_dir.mkdir(exist_ok=True)
            
            # Copy three.min.js to temp directory if it exists locally
            if local_threejs.exists():
                temp_threejs = temp_dir / "three.min.js"
                shutil.copy2(local_threejs, temp_threejs)
                TEMP_FILES.append(temp_threejs)
                print(f"     Using local three.min.js")
            else:
                print(f"     [WARNING] Local three.min.js not found, HTML will try to load it anyway")
            
            # WebView2 has ~2MB limit for NavigateToString, so we need to use a temp file
            # for large models (but it will still be auto-deleted)
            temp_file = temp_dir / f"{mdl_path.stem}_viewer.html"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(html_content)
            TEMP_FILES.append(temp_file)
            TEMP_FILES.append(temp_dir)
            
            # Create window with file URL
            window = webview.create_window(
                f'Model Viewer - {mdl_path.name}',
                url=f'file:///{temp_file.absolute().as_posix()}',
                width=1280,
                height=800,
                resizable=True,
                fullscreen=False,
		maximized=True
            )
            
            print("[OK] Window opened. Close it to exit.")
            print("     Temp file will be auto-deleted on exit.")
            
            # Start event loop
            webview.start()
            
            print("\n[OK] Window closed. Cleaning up...")
            
        except ImportError:
            print("\n[ERROR] pywebview is not installed!")
            print("Install: pip install pywebview")
            print("\nAlternatively, creating temporary HTML file...")
            
            # Fallback: create temporary file
            temp_file = Path(tempfile.gettempdir()) / f"{mdl_path.stem}_viewer.html"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(html_content)
            TEMP_FILES.append(temp_file)
            
            import webbrowser
            webbrowser.open(f"file://{temp_file.absolute()}")
            print(f"[OK] Opened in browser: {temp_file}")
            print("     File will be deleted on exit.")
            
            input("\nPress Enter to close and cleanup...")

    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
