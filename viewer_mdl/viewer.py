#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STANDALONE 3D Model Viewer
Integrates only necessary loading functions from mdl_viewer.py
HTML generation unchanged from FINAL_FIXED_viewer.py
"""

import struct
import numpy as np
from pathlib import Path
import re
import json
import sys
from collections import defaultdict

# Patterns for format detection
f32_pattern = re.compile(r'''^R32G32B32A32_FLOAT|^R32G32B32_FLOAT|^R32G32_FLOAT|^R32_FLOAT''')
f16_pattern = re.compile(r'''^R16G16B16A16_FLOAT|^R16G16_FLOAT''')
u32_pattern = re.compile(r'''^R32G32B32A32_UINT|^R32G32B32_UINT|^R32G32_UINT|^R32_UINT''')
u16_pattern = re.compile(r'''^R16G16B16A16_UINT|^R16G16_UINT''')
s32_pattern = re.compile(r'''^R32G32B32A32_SINT|^R32G32B32_SINT|^R32G32_SINT|^R32_SINT''')
s16_pattern = re.compile(r'''^R16G16B16A16_SINT|^R16G16_SINT''')


# ============================================================================
# MODEL LOADING (minimal necessary code from mdl_viewer.py)
# ============================================================================

class InputLayoutElement:
    """Reprezentuje jeden element vertex layoutu"""
    
    def __init__(self):
        self.SemanticName = None
        self.SemanticIndex = 0
        self.Format = None
        self.InputSlot = 0
        self.AlignedByteOffset = 0
        self.InputSlotClass = None
        self.InstanceDataStepRate = 0

class InputLayoutElement:
    """Reprezentuje jeden element vertex layoutu"""
    
    def __init__(self):
        self.SemanticName = None
        self.SemanticIndex = 0
        self.Format = None
        self.InputSlot = 0
        self.AlignedByteOffset = 0
        self.InputSlotClass = None
        self.InstanceDataStepRate = 0
        
    def parse_from_file(self, lines):
        """Parsuje element z řádků .fmt souboru"""
        for line in lines:
            line = line.strip()
            if not line or line.startswith('element['):
                break
                
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'SemanticName':
                    self.SemanticName = value
                elif key == 'SemanticIndex':
                    self.SemanticIndex = int(value)
                elif key == 'Format':
                    self.Format = value
                elif key == 'InputSlot':
                    self.InputSlot = int(value)
                elif key == 'AlignedByteOffset':
                    self.AlignedByteOffset = int(value)
                elif key == 'InputSlotClass':
                    self.InputSlotClass = value
                elif key == 'InstanceDataStepRate':
                    self.InstanceDataStepRate = int(value)
        
        return self
    
    @property
    def name(self):
        if self.SemanticIndex:
            return f'{self.SemanticName}{self.SemanticIndex}'
        return self.SemanticName
    
    def get_decoder(self):
        """Vrátí decoder funkci pro tento formát"""
        fmt = self.Format
        
        if f32_pattern.match(fmt):
            return lambda data: np.frombuffer(data, np.float32).tolist()
        if u32_pattern.match(fmt):
            return lambda data: np.frombuffer(data, np.uint32).tolist()
        if u16_pattern.match(fmt):
            return lambda data: np.frombuffer(data, np.uint16).tolist()
        if s32_pattern.match(fmt):
            return lambda data: np.frombuffer(data, np.int32).tolist()
        if s16_pattern.match(fmt):
            return lambda data: np.frombuffer(data, np.int16).tolist()
        
        raise ValueError(f'Nepodporovaný formát: {fmt}')
    
    def get_size(self):
        """Vrátí velikost dat v bajtech"""
        components_pattern = re.compile(r'(?<![0-9])[0-9]+(?![0-9])')
        matches = components_pattern.findall(self.Format)
        return sum(map(int, matches)) // 8


class InputLayout:
    """Layout pro vertex buffer"""
    
    def __init__(self):
        self.elements = []
        self.stride = 0
        self.topology = 'trianglelist'
        self.index_format = 'R32_UINT'
        
    def parse_from_file(self, fmt_path):
        """Parsuje .fmt soubor"""
        with open(fmt_path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('stride:'):
                self.stride = int(line.split(':')[1].strip())
            elif line.startswith('topology:'):
                self.topology = line.split(':')[1].strip()
            elif line.startswith('format:'):
                self.index_format = line.split(':')[1].strip()
            elif line.startswith('element['):
                elem = InputLayoutElement()
                elem.parse_from_file(lines[i+1:])
                self.elements.append(elem)
            
            i += 1
        
        return self
    
    def decode_vertex(self, data):
        """Dekóduje jeden vertex z binárních dat"""
        vertex = {}
        for elem in self.elements:
            start = elem.AlignedByteOffset
            end = start + elem.get_size()
            elem_data = data[start:end]
            
            decoder = elem.get_decoder()
            vertex[elem.name] = decoder(elem_data)
        
        return vertex


class TrailsModelLoader:
    """Načítá 3D modely z Legend of Heroes Trails"""
    
    def __init__(self, model_directory):
        self.model_dir = Path(model_directory)
        self.meshes = []
        
    def load(self):
        """Načte všechny meshe v adresáři"""
        print(f"Načítám model z: {self.model_dir}")
        
        # Najdi všechny .fmt soubory
        fmt_files = sorted(self.model_dir.glob("*.fmt"))
        
        for fmt_file in fmt_files:
            base_name = fmt_file.stem  # Název bez přípony
            vb_file = self.model_dir / f"{base_name}.vb"
            ib_file = self.model_dir / f"{base_name}.ib"
            
            if not vb_file.exists():
                print(f"  Varování: Nenalezen {vb_file.name}")
                continue
            
            print(f"  Načítám mesh: {base_name}")
            
            try:
                # Načtení layoutu
                layout = InputLayout()
                layout.parse_from_file(fmt_file)
                
                # Načtení vertex bufferu
                vertices = self._load_vertices(vb_file, layout)
                
                # Načtení index bufferu
                indices = None
                if ib_file.exists():
                    indices = self._load_indices(ib_file, layout.index_format)
                
                # Extrakce pozic a normál
                positions = []
                normals = []
                
                for vertex in vertices:
                    if 'POSITION' in vertex:
                        positions.append(vertex['POSITION'][:3])  # Pouze XYZ
                    if 'NORMAL' in vertex:
                        normals.append(vertex['NORMAL'][:3])
                
                mesh = {
                    'name': base_name,
                    'vertices': np.array(positions) if positions else None,
                    'normals': np.array(normals) if normals else None,
                    'indices': indices,
                    'raw_vertices': vertices
                }
                
                self.meshes.append(mesh)
                
            except Exception as e:
                print(f"  Chyba při načítání {base_name}: {e}")
                continue
        
        print(f"Načteno {len(self.meshes)} meshů")
        return self
    
    def _load_vertices(self, vb_path, layout):
        """Načte vertex buffer"""
        with open(vb_path, 'rb') as f:
            data = f.read()
        
        vertices = []
        num_vertices = len(data) // layout.stride
        
        for i in range(num_vertices):
            start = i * layout.stride
            end = start + layout.stride
            vertex_data = data[start:end]
            
            vertex = layout.decode_vertex(vertex_data)
            vertices.append(vertex)
        
        return vertices
    
    def _load_indices(self, ib_path, index_format):
        """Načte index buffer"""
        with open(ib_path, 'rb') as f:
            data = f.read()
        
        # Určení formátu indexů
        if 'R32' in index_format:
            indices = np.frombuffer(data, dtype=np.uint32)
        elif 'R16' in index_format:
            indices = np.frombuffer(data, dtype=np.uint16)
        else:
            raise ValueError(f'Nepodporovaný index formát: {index_format}')
        
        return indices
    
    def get_all_vertices(self):
        """Vrátí všechny vertices ze všech meshů"""
        all_verts = []
        for mesh in self.meshes:
            if mesh['vertices'] is not None:
                all_verts.append(mesh['vertices'])
        
        if all_verts:
            return np.vstack(all_verts)
        return np.array([])
    
    def get_mesh_bounds(self):
        """Vrátí bounding box celého modelu"""
        verts = self.get_all_vertices()
        if len(verts) == 0:
            return None
        
        return {
            'min': verts.min(axis=0),
            'max': verts.max(axis=0),
            'center': verts.mean(axis=0)
        }
    
    def export_to_obj(self, output_path):
        """Exportuje model do OBJ formátu"""
        with open(output_path, 'w') as f:
            f.write("# Legend of Heroes Trails Model Export\n")
            f.write(f"# Generated from {self.model_dir.name}\n\n")
            
            vertex_offset = 1  # OBJ indexuje od 1
            
            for mesh in self.meshes:
                vertices = mesh['vertices']
                normals = mesh['normals']
                indices = mesh['indices']
                
                if vertices is None or len(vertices) == 0:
                    continue
                
                # Zápis názvu objektu
                f.write(f"\no {mesh['name']}\n")
                
                # Zápis vertexů
                for v in vertices:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                
                # Zápis normál
                if normals is not None:
                    for n in normals:
                        f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
                
                # Zápis faces
                if indices is not None and len(indices) >= 3:
                    for i in range(0, len(indices) - 2, 3):
                        try:
                            i0 = int(indices[i]) + vertex_offset
                            i1 = int(indices[i+1]) + vertex_offset
                            i2 = int(indices[i+2]) + vertex_offset
                            
                            if normals is not None:
                                f.write(f"f {i0}//{i0} {i1}//{i1} {i2}//{i2}\n")
                            else:
                                f.write(f"f {i0} {i1} {i2}\n")
                        except:
                            continue
                
                vertex_offset += len(vertices)
                f.write("\n")
        
        print(f"Model exportován do: {output_path}")


def visualize_model(model_loader, show_wireframe=True, show_points=False, elevation=20, azimuth=45):
    """Vizualizuje 3D model pomocí matplotlib"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("Instaluji matplotlib...")
        import subprocess
        subprocess.check_call(["pip", "install", "matplotlib", "--break-system-packages"])
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(model_loader.meshes)))
    
    for idx, mesh in enumerate(model_loader.meshes):
        vertices = mesh['vertices']
        indices = mesh['indices']
        
        if vertices is None or len(vertices) == 0:
            continue
        
        color = colors[idx]
        
        # Vykreslení bodů (volitelné)
        if show_points:
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                      c=[color], s=1, alpha=0.3)
        
        # Vykreslení trojúhelníků
        if indices is not None and len(indices) >= 3:
            triangles = []
            for i in range(0, len(indices) - 2, 3):
                try:
                    i0, i1, i2 = int(indices[i]), int(indices[i+1]), int(indices[i+2])
                    if i0 < len(vertices) and i1 < len(vertices) and i2 < len(vertices):
                        tri = [vertices[i0], vertices[i1], vertices[i2]]
                        triangles.append(tri)
                except:
                    continue
            
            if triangles:
                poly = Poly3DCollection(triangles, alpha=0.6, linewidths=0.5, 
                                       edgecolors='k' if show_wireframe else None)
                poly.set_facecolor(color)
                ax.add_collection3d(poly)
    
    # Nastavení os
    bounds = model_loader.get_mesh_bounds()
    if bounds:
        center = bounds['center']
        size = (bounds['max'] - bounds['min']).max() / 2
        
        ax.set_xlim([center[0] - size, center[0] + size])
        ax.set_ylim([center[1] - size, center[1] + size])
        ax.set_zlim([center[2] - size, center[2] + size])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Model Preview - Legend of Heroes Trails')
    ax.view_init(elev=elevation, azim=azimuth)
    
    plt.tight_layout()
    return fig, ax


def main():
    """Hlavní funkce"""
    import sys
    
    if len(sys.argv) < 2:
        print("Použití: python mdl_viewer.py <cesta_k_adresáři> [--export-obj]")
        print("\nPříklad:")
        print("  python mdl_viewer.py ./chr5000_c00")
        print("  python mdl_viewer.py ./chr5000_c00 --export-obj")
        sys.exit(1)
    
    model_path = sys.argv[1]
    export_obj = '--export-obj' in sys.argv
    
    try:
        # Načtení modelu
        loader = TrailsModelLoader(model_path)
        loader.load()
        
        # Zobrazení informací
        print("\n" + "="*60)
        print("INFORMACE O MODELU:")
        print("="*60)
        print(f"Počet meshů: {len(loader.meshes)}")
        
        total_verts = 0
        total_tris = 0
        
        for mesh in loader.meshes:
            verts = len(mesh['vertices']) if mesh['vertices'] is not None else 0
            tris = len(mesh['indices']) // 3 if mesh['indices'] is not None else 0
            total_verts += verts
            total_tris += tris
            print(f"  - {mesh['name']}: {verts} vertices, {tris} trojúhelníků")
        
        print(f"\nCelkem: {total_verts} vertices, {total_tris} trojúhelníků")
        
        bounds = loader.get_mesh_bounds()
        if bounds:
            print(f"\nBounding Box:")
            print(f"  Min: {bounds['min']}")
            print(f"  Max: {bounds['max']}")
            print(f"  Center: {bounds['center']}")
        
        print("\n" + "="*60)
        
        # Export do OBJ (volitelné)
        if export_obj:
            obj_file = Path(model_path) / "model_export.obj"
            loader.export_to_obj(obj_file)
        
        # Vizualizace
        print("\nGeneruji 3D náhled...")
        import matplotlib.pyplot as plt
        fig, ax = visualize_model(loader, show_wireframe=True)
        
        # Uložení náhledu
        output_file = Path(model_path) / "model_preview.png"
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Náhled uložen do: {output_file}")
        
        # Zobrazení interaktivního okna
        plt.show()
        
    except Exception as e:
        print(f"Chyba: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ============================================================================
# NORMAL COMPUTATION (from position_shared_normals_WINDOWS.py)
# ============================================================================

def compute_smooth_normals_with_sharing(vertices, indices, tolerance=1e-6):
    """
    THE FIX: Compute smooth normals with proper position sharing
    """
    print(f"    Computing normals with position sharing...")
    
    position_map = {}
    vertex_to_position = {}
    
    for idx, vertex in enumerate(vertices):
        pos_key = tuple(np.round(vertex / tolerance) * tolerance)
        
        if pos_key not in position_map:
            position_map[pos_key] = []
        position_map[pos_key].append(idx)
        vertex_to_position[idx] = pos_key
    
    num_unique = len(position_map)
    num_total = len(vertices)
    num_duplicates = num_total - num_unique
    
    print(f"      Total vertices: {num_total}")
    print(f"      Unique positions: {num_unique}")
    print(f"      Duplicate vertices: {num_duplicates} ({100*num_duplicates/num_total:.1f}%)")
    
    position_normals = defaultdict(lambda: np.zeros(3, dtype=np.float32))
    
    for i in range(0, len(indices), 3):
        i0, i1, i2 = int(indices[i]), int(indices[i+1]), int(indices[i+2])
        
        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        
        pos0 = vertex_to_position[i0]
        pos1 = vertex_to_position[i1]
        pos2 = vertex_to_position[i2]
        
        position_normals[pos0] += face_normal
        position_normals[pos1] += face_normal
        position_normals[pos2] += face_normal
    
    for pos_key in position_normals:
        length = np.linalg.norm(position_normals[pos_key])
        if length > 0:
            position_normals[pos_key] /= length
    
    normals = np.zeros_like(vertices, dtype=np.float32)
    for idx in range(len(vertices)):
        pos_key = vertex_to_position[idx]
        normals[idx] = position_normals[pos_key]
    
    print(f"      [OK] Normals computed and shared")
    
    return normals


# ============================================================================
# HTML EXPORT (UNCHANGED from FINAL_FIXED_viewer.py)
# ============================================================================

def export_html(model_path, use_original_normals=False):
    """Export with proper normal computation"""
    
    print(f"Loading: {model_path}")
    loader = TrailsModelLoader(model_path)
    loader.load()
    
    meshes_data = []
    
    for mesh in loader.meshes:
        vertices = mesh['vertices']
        indices = mesh['indices']
        
        if vertices is None or indices is None:
            continue
        
        print(f"\n  {mesh['name']}:")
        
        if use_original_normals and mesh['normals'] is not None:
            normals = mesh['normals']
            print(f"    Using ORIGINAL normals from .vb")
        else:
            normals = compute_smooth_normals_with_sharing(vertices, indices)
        
        mesh_data = {
            'name': mesh['name'],
            'vertices': vertices.flatten().tolist(),
            'normals': normals.flatten().tolist(),
            'indices': indices.tolist()
        }
        
        meshes_data.append(mesh_data)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Model Viewer - Position-Shared Normals</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: system-ui, -apple-system, sans-serif; overflow: hidden; background: #1a1a2e; }}
        #container {{ width: 100vw; height: 100vh; }}
        .panel {{ position: absolute; background: rgba(20,20,35,0.95); color: #e0e0e0; 
                 padding: 18px; border-radius: 10px; box-shadow: 0 8px 32px rgba(0,0,0,0.5); }}
        #info {{ top: 20px; left: 20px; max-width: 300px; }}
        #controls {{ top: 20px; right: 20px; max-height: 85vh; overflow-y: auto; }}
        #stats {{ bottom: 20px; left: 20px; font-family: monospace; font-size: 12px; }}
        h3 {{ margin: 0 0 12px 0; color: #7c3aed; font-size: 16px; }}
        h4 {{ margin: 15px 0 10px 0; padding-bottom: 8px; border-bottom: 1px solid rgba(124, 58, 237, 0.3); 
             font-size: 14px; color: #a78bfa; font-weight: 500; }}
        button {{ background: linear-gradient(135deg, #7c3aed, #a855f7); border: none; 
                  color: white; padding: 10px; margin: 5px 0; cursor: pointer; 
                  border-radius: 6px; width: 100%; font-weight: 600; }}
        button:hover {{ transform: translateY(-1px); }}
        .mesh-toggle {{
            display: flex;
            align-items: center;
            margin: 8px 0;
            padding: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            transition: all 0.2s;
            border: 1px solid transparent;
        }}
        .mesh-toggle:hover {{
            background: rgba(124, 58, 237, 0.15);
            border-color: rgba(124, 58, 237, 0.3);
        }}
        .mesh-toggle input {{
            margin-right: 12px;
            cursor: pointer;
            width: 16px;
            height: 16px;
        }}
        .mesh-toggle label {{
            cursor: pointer;
            flex: 1;
            font-size: 13px;
            user-select: none;
        }}
        .info-row {{ margin: 6px 0; font-size: 13px; }}
        .label {{ color: #a78bfa; font-weight: 500; }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info" class="panel">
        <h3>🎨 3D Model Viewer</h3>
        <div class="info-row"><span class="label">Model:</span> {Path(model_path).name}</div>
        <div class="info-row"><span class="label">Fix:</span> Position-shared normals</div>
    </div>
    <div id="controls" class="panel">
        <h4>🎨 Display</h4>
        <button onclick="toggleWireframe()">📐 Toggle Wireframe</button>
        <button onclick="resetCamera()">🎯 Reset Camera</button>
        <button onclick="toggleColors()">🎨 Change Colors</button>
        
        <h4>👁️ Mesh Visibility</h4>
        <button onclick="toggleAllMeshes(true)">✅ Show All</button>
        <button onclick="toggleAllMeshes(false)">❌ Hide All</button>
        <div id="mesh-list"></div>
    </div>
    <div id="stats" class="panel">
        <div id="fps">FPS: --</div>
        <div id="vertices">Vertices: --</div>
        <div id="triangles">Triangles: --</div>
        <div id="visible">Visible: --</div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const data = {json.dumps(meshes_data, indent=2)};
        let scene, camera, renderer, controls, meshes = [], wireframeMode = false, colorMode = 0;
        
        class OrbitControls {{
            constructor(camera, domElement) {{
                this.camera = camera; this.domElement = domElement; this.target = new THREE.Vector3();
                this.spherical = new THREE.Spherical(); this.sphericalDelta = new THREE.Spherical();
                this.scale = 1; this.panOffset = new THREE.Vector3();
                this.rotateStart = new THREE.Vector2(); this.panStart = new THREE.Vector2(); this.state = 0;
                domElement.addEventListener('contextmenu', e => e.preventDefault());
                domElement.addEventListener('mousedown', e => {{
                    if (e.button === 0) {{ this.state = 1; this.rotateStart.set(e.clientX, e.clientY); }}
                    else if (e.button === 2) {{ this.state = 2; this.panStart.set(e.clientX, e.clientY); }}
                }});
                domElement.addEventListener('mousemove', e => {{
                    if (this.state === 1) {{
                        const end = new THREE.Vector2(e.clientX, e.clientY);
                        const delta = new THREE.Vector2().subVectors(end, this.rotateStart);
                        this.sphericalDelta.theta -= 2 * Math.PI * delta.x / domElement.clientHeight;
                        this.sphericalDelta.phi -= 2 * Math.PI * delta.y / domElement.clientHeight;
                        this.rotateStart.copy(end); this.update();
                    }} else if (this.state === 2) {{
                        const end = new THREE.Vector2(e.clientX, e.clientY);
                        const delta = new THREE.Vector2().subVectors(end, this.panStart);
                        const dist = this.camera.position.distanceTo(this.target);
                        const factor = dist * Math.tan((this.camera.fov / 2) * Math.PI / 180.0);
                        const left = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 0);
                        left.multiplyScalar(-2 * delta.x * factor / domElement.clientHeight);
                        const up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 1);
                        up.multiplyScalar(2 * delta.y * factor / domElement.clientHeight);
                        this.panOffset.add(left).add(up); this.panStart.copy(end); this.update();
                    }}
                }});
                domElement.addEventListener('mouseup', () => {{ this.state = 0; }});
                domElement.addEventListener('wheel', e => {{ 
                    e.preventDefault(); 
                    this.scale *= (e.deltaY < 0) ? 0.95 : 1.05; 
                    this.update(); 
                }});
                this.update();
            }}
            update() {{
                const offset = new THREE.Vector3();
                const quat = new THREE.Quaternion().setFromUnitVectors(this.camera.up, new THREE.Vector3(0,1,0));
                offset.copy(this.camera.position).sub(this.target); offset.applyQuaternion(quat);
                this.spherical.setFromVector3(offset);
                this.spherical.theta += this.sphericalDelta.theta;
                this.spherical.phi += this.sphericalDelta.phi;
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
            let totalVerts = 0;
            let totalTriangles = 0;
            const meshListDiv = document.getElementById('mesh-list');
            
            data.forEach((m, i) => {{
                const geo = new THREE.BufferGeometry();
                geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(m.vertices), 3));
                geo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(m.normals), 3));
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
                totalVerts += m.vertices.length / 3;
                totalTriangles += m.indices.length / 3;
                
                const toggleDiv = document.createElement('div');
                toggleDiv.className = 'mesh-toggle';
                toggleDiv.innerHTML = `
                    <input type="checkbox" id="mesh-${{i}}" checked onchange="toggleMesh(${{i}})">
                    <label for="mesh-${{i}}">${{m.name}}</label>
                `;
                meshListDiv.appendChild(toggleDiv);
            }});
            
            controls = new OrbitControls(camera, renderer.domElement);
            scene.add(new THREE.GridHelper(10, 10, 0x444444, 0x222222));
            
            const box = new THREE.Box3();
            meshes.forEach(m => box.expandByObject(m));
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            const dist = Math.max(size.x, size.y, size.z) * 2;
            camera.position.set(center.x + dist*0.5, center.y + dist*0.5, center.z + dist*0.5);
            camera.lookAt(center);
            controls.target.copy(center); controls.update();
            
            document.getElementById('vertices').textContent = 'Vertices: ' + totalVerts.toLocaleString();
            document.getElementById('triangles').textContent = 'Triangles: ' + totalTriangles.toLocaleString();
            document.getElementById('visible').textContent = `Visible: ${{meshes.length}}/${{meshes.length}}`;
            
            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});
            
            animate();
        }}
        
        function toggleMesh(index) {{
            meshes[index].visible = document.getElementById(`mesh-${{index}}`).checked;
            const visibleCount = meshes.filter(m => m.visible).length;
            document.getElementById('visible').textContent = `Visible: ${{visibleCount}}/${{meshes.length}}`;
        }}
        
        function toggleAllMeshes(visible) {{
            meshes.forEach((mesh, index) => {{
                mesh.visible = visible;
                document.getElementById(`mesh-${{index}}`).checked = visible;
            }});
            document.getElementById('visible').textContent = `Visible: ${{visible ? meshes.length : 0}}/${{meshes.length}}`;
        }}
        
        function toggleColors() {{
            colorMode = (colorMode + 1) % 3;
            meshes.forEach((mesh, index) => {{
                if (colorMode === 0) {{
                    const colors = [0xE8B4B8, 0xB8D4E8, 0xC8E8B8, 0xE8D4B8, 0xD8B8E8, 0xE8E8B8];
                    mesh.material.color.setHex(colors[index % colors.length]);
                }} else if (colorMode === 1) {{
                    mesh.material.color.setHex(0xCCCCCC);
                }} else {{
                    mesh.material.color.setHSL(index / meshes.length, 0.7, 0.6);
                }}
            }});
        }}
        
        function toggleWireframe() {{
            wireframeMode = !wireframeMode;
            meshes.forEach(m => m.material.wireframe = wireframeMode);
        }}
        
        function resetCamera() {{
            const box = new THREE.Box3();
            meshes.forEach(m => box.expandByObject(m));
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
    
    output = Path(model_path) / "model_viewer.html"
    with open(output, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n[OK] Created: {output}")
    return output


def main():
    if len(sys.argv) < 2:
        print("Usage: python STANDALONE_viewer.py <model_path> [--original]")
        print("\nSTANDALONE VERSION - No mdl_viewer.py needed!")
        print("  Only necessary loading functions integrated")
        print("  HTML generation unchanged from working version")
        print("\nExample:")
        print("  python STANDALONE_viewer.py ./chr5000_c00")
        sys.exit(1)
    
    model_path = sys.argv[1]
    use_original = '--original' in sys.argv
    
    try:
        output = export_html(model_path, use_original)
        try:
            import webbrowser
            webbrowser.open(f'file://{output.absolute()}')
            print("\n[OK] Opened in browser!")
        except:
            print(f"\nPlease open manually: {output.absolute()}")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
