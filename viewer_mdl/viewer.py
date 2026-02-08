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
snorm8_pattern = re.compile(r'''^R8G8B8A8_SNORM|^R8G8_SNORM''')
unorm8_pattern = re.compile(r'''^R8G8B8A8_UNORM|^R8G8_UNORM''')
u8_pattern = re.compile(r'''^R8G8B8A8_UINT|^R8G8_UINT|^R8_UINT''')
snorm16_pattern = re.compile(r'''^R16G16B16A16_SNORM|^R16G16_SNORM''')
unorm16_pattern = re.compile(r'''^R16G16B16A16_UNORM|^R16G16_UNORM''')


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
        if f16_pattern.match(fmt):
            return lambda data: np.frombuffer(data, np.float16).astype(np.float32).tolist()
        if u32_pattern.match(fmt):
            return lambda data: np.frombuffer(data, np.uint32).tolist()
        if u16_pattern.match(fmt):
            return lambda data: np.frombuffer(data, np.uint16).tolist()
        if s32_pattern.match(fmt):
            return lambda data: np.frombuffer(data, np.int32).tolist()
        if s16_pattern.match(fmt):
            return lambda data: np.frombuffer(data, np.int16).tolist()
        if snorm8_pattern.match(fmt):
            return lambda data: (np.frombuffer(data, np.int8).astype(np.float32) / 127.0).tolist()
        if unorm8_pattern.match(fmt):
            return lambda data: (np.frombuffer(data, np.uint8).astype(np.float32) / 255.0).tolist()
        if u8_pattern.match(fmt):
            return lambda data: np.frombuffer(data, np.uint8).tolist()
        if snorm16_pattern.match(fmt):
            return lambda data: (np.frombuffer(data, np.int16).astype(np.float32) / 32767.0).tolist()
        if unorm16_pattern.match(fmt):
            return lambda data: (np.frombuffer(data, np.uint16).astype(np.float32) / 65535.0).tolist()
        
        raise ValueError(f'Unsupported format: {fmt}')
    
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
        print(f"Loading model from: {self.model_dir}")
        
        # Najdi všechny .fmt soubory
        fmt_files = sorted(self.model_dir.glob("*.fmt"))
        
        for fmt_file in fmt_files:
            base_name = fmt_file.stem  # Název bez přípony
            vb_file = self.model_dir / f"{base_name}.vb"
            ib_file = self.model_dir / f"{base_name}.ib"
            
            if not vb_file.exists():
                print(f"  Warning: Not found {vb_file.name}")
                continue
            
            print(f"  Loading mesh: {base_name}")
            
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
                print(f"  Error loading {base_name}: {e}")
                continue
        
        print(f"Loaded {len(self.meshes)} meshes")
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
            raise ValueError(f'Unsupported index format: {index_format}')
        
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
        
        print(f"Model exported to: {output_path}")


def visualize_model(model_loader, show_wireframe=True, show_points=False, elevation=20, azimuth=45):
    """Vizualizuje 3D model pomocí matplotlib"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("Installing matplotlib...")
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
    """Main function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mdl_viewer.py <directory_path> [--export-obj]")
        print("\nExample:")
        print("  python mdl_viewer.py ./chr5000_c00")
        print("  python mdl_viewer.py ./chr5000_c00 --export-obj")
        sys.exit(1)
    
    model_path = sys.argv[1]
    export_obj = '--export-obj' in sys.argv
    
    try:
        # Load model
        loader = TrailsModelLoader(model_path)
        loader.load()
        
        # Display info
        print("\n" + "="*60)
        print("MODEL INFO:")
        print("="*60)
        print(f"Mesh count: {len(loader.meshes)}")
        
        total_verts = 0
        total_tris = 0
        
        for mesh in loader.meshes:
            verts = len(mesh['vertices']) if mesh['vertices'] is not None else 0
            tris = len(mesh['indices']) // 3 if mesh['indices'] is not None else 0
            total_verts += verts
            total_tris += tris
            print(f"  - {mesh['name']}: {verts} vertices, {tris} triangles")
        
        print(f"\nTotal: {total_verts} vertices, {total_tris} triangles")
        
        bounds = loader.get_mesh_bounds()
        if bounds:
            print(f"\nBounding Box:")
            print(f"  Min: {bounds['min']}")
            print(f"  Max: {bounds['max']}")
            print(f"  Center: {bounds['center']}")
        
        print("\n" + "="*60)
        
        # Export to OBJ (optional)
        if export_obj:
            obj_file = Path(model_path) / "model_export.obj"
            loader.export_to_obj(obj_file)
        
        # Visualization
        print("\nGenerating 3D preview...")
        import matplotlib.pyplot as plt
        fig, ax = visualize_model(loader, show_wireframe=True)
        
        # Save preview
        output_file = Path(model_path) / "model_preview.png"
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Preview saved to: {output_file}")
        
        # Show interactive window
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
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

def export_html(model_path, use_original_normals=True):
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
  <title>Model Viewer - {Path(model_path).name}</title>
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
      <strong style="color: #a78bfa;">{Path(model_path).name}</strong>
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
    </div>

    <div class="section-title" style="cursor:pointer;user-select:none;" onclick="const el=document.getElementById('meshSection'); el.style.display=el.style.display==='none'?'block':'none'; this.querySelector('.arrow').textContent=el.style.display==='none'?'▶':'▼';">📦 Meshes <span class="arrow" style="font-size:10px;margin-left:4px;">▶</span></div>
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
    const CONFIG = {{
      CAMERA_ZOOM: 1.5,
      AUTO_HIDE_SHADOW: true,
      INITIAL_BACKGROUND: 0x1a1a2e
    }};

    const data = {json.dumps(meshes_data)};
    let scene, camera, renderer, controls, meshes = [];
    let wireframeMode = false, wireframeOverlayMode = false, colorMode = 0;
    let wireframeMeshes = [];

    function toggleControlsPanel() {{
      const panel = document.getElementById('controls');
      const btn = document.getElementById('controls-toggle');
      panel.classList.toggle('collapsed');
    }}

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
        this.lastX = e.clientX;
        this.lastY = e.clientY;
      }}

      onMouseUp() {{
        this.isMouseDown = false;
      }}

      onMouseMove(e) {{
        if (!this.isMouseDown) return;
        const dx = e.clientX - this.lastX;
        const dy = e.clientY - this.lastY;
        this.lastX = e.clientX;
        this.lastY = e.clientY;

        if (this.mouseButton === 0) {{
          this.sphericalDelta.theta -= 2 * Math.PI * dx / this.domElement.clientHeight * this.rotateSpeed;
          this.sphericalDelta.phi -= 2 * Math.PI * dy / this.domElement.clientHeight * this.rotateSpeed;
          this.update();
        }} else if (this.mouseButton === 2) {{
          const dist = this.camera.position.distanceTo(this.target);
          const factor = dist * Math.tan((this.camera.fov / 2) * Math.PI / 180.0);
          const left = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 0);
          left.multiplyScalar(-2 * dx * factor / this.domElement.clientHeight * this.panSpeed);
          const up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 1);
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
        const offset = new THREE.Vector3();
        const quat = new THREE.Quaternion().setFromUnitVectors(this.camera.up, new THREE.Vector3(0,1,0));
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
      const light = new THREE.DirectionalLight(0xffffff, 0.8);
      light.position.set(5, 5, 5);
      scene.add(light);

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
        mesh.userData.baseColor = colors[i % colors.length];

        // Auto-hide shadow meshes
        if (CONFIG.AUTO_HIDE_SHADOW && m.name.toLowerCase().includes('shadow')) {{
          mesh.visible = false;
        }}

        scene.add(mesh);
        meshes.push(mesh);
        totalVerts += m.vertices.length / 3;
        totalTriangles += m.indices.length / 3;

        const toggleDiv = document.createElement('div');
        toggleDiv.className = 'mesh-toggle';
        toggleDiv.innerHTML = `
          <input type="checkbox" id="mesh-${{i}}" ${{mesh.visible ? 'checked' : ''}} onchange="toggleMesh(${{i}})">
          <label for="mesh-${{i}}">${{m.name}}</label>
        `;
        meshListDiv.appendChild(toggleDiv);
      }});

      controls = new OrbitControls(camera, renderer.domElement);
      scene.add(new THREE.GridHelper(10, 10, 0x444444, 0x222222));

      const box = new THREE.Box3();
      meshes.forEach(m => {{ if (m.visible) box.expandByObject(m); }});
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const dist = Math.max(size.x, size.y, size.z) * CONFIG.CAMERA_ZOOM;
      camera.position.set(center.x + dist*0.5, center.y + dist*0.5, center.z + dist*0.5);
      camera.lookAt(center);
      controls.target.copy(center);
      controls.update();

      document.getElementById('vertices').textContent = totalVerts.toLocaleString();
      document.getElementById('triangles').textContent = totalTriangles.toLocaleString();
      const visibleCount = meshes.filter(m => m.visible).length;
      document.getElementById('visible').textContent = visibleCount + '/' + meshes.length;

      window.addEventListener('resize', () => {{
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      }});

      animate();
    }}

    function toggleMesh(index) {{
      meshes[index].visible = document.getElementById('mesh-' + index).checked;
      const visibleCount = meshes.filter(m => m.visible).length;
      document.getElementById('visible').textContent = visibleCount + '/' + meshes.length;
    }}

    function toggleAllMeshes(visible) {{
      meshes.forEach((mesh, index) => {{
        mesh.visible = visible;
        document.getElementById('mesh-' + index).checked = visible;
      }});
      document.getElementById('visible').textContent = (visible ? meshes.length : 0) + '/' + meshes.length;
    }}

    function toggleColors() {{
      colorMode = (colorMode + 1) % 3;
      const dots = document.querySelectorAll('.cycle-dot');
      dots.forEach(d => d.classList.remove('current'));
      dots[colorMode].classList.add('current');

      meshes.forEach((mesh, index) => {{
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
      meshes.forEach(m => m.material.wireframe = wireframeMode);
      // Disable overlay if wireframe-only is on
      if (wireframeMode && wireframeOverlayMode) {{
        wireframeOverlayMode = false;
        document.getElementById('swWireOver').checked = false;
        wireframeMeshes.forEach(wm => {{ scene.remove(wm); wm.geometry.dispose(); wm.material.dispose(); }});
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
          meshes.forEach(m => m.material.wireframe = false);
        }}
        meshes.forEach(m => {{
          const wGeo = m.geometry.clone();
          const wMat = new THREE.MeshBasicMaterial({{
            color: 0x000000, wireframe: true, transparent: true, opacity: 0.15
          }});
          const wMesh = new THREE.Mesh(wGeo, wMat);
          wMesh.position.copy(m.position);
          wMesh.rotation.copy(m.rotation);
          wMesh.scale.copy(m.scale);
          wMesh.visible = m.visible;
          scene.add(wMesh);
          wireframeMeshes.push(wMesh);
        }});
      }} else {{
        wireframeMeshes.forEach(wm => {{ scene.remove(wm); wm.geometry.dispose(); wm.material.dispose(); }});
        wireframeMeshes = [];
      }}
    }}

    function resetCamera() {{
      const box = new THREE.Box3();
      meshes.forEach(m => {{ if (m.visible) box.expandByObject(m); }});
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const dist = Math.max(size.x, size.y, size.z) * CONFIG.CAMERA_ZOOM;
      camera.position.set(center.x + dist*0.5, center.y + dist*0.5, center.z + dist*0.5);
      camera.lookAt(center);
      controls.target.copy(center);
      controls.update();
    }}

    function takeScreenshot() {{
      const scale = parseInt(document.getElementById('screenshotScale').value) || 2;
      const w = window.innerWidth;
      const h = window.innerHeight;
      const targetW = w * scale;
      const targetH = h * scale;

      if (scale <= 1) {{
        renderer.render(scene, camera);
        finishScreenshot(renderer.domElement.toDataURL('image/png'));
        return;
      }}

      // High-res: render to offscreen WebGLRenderTarget
      const rt = new THREE.WebGLRenderTarget(targetW, targetH, {{
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat
      }});

      renderer.setRenderTarget(rt);
      renderer.render(scene, camera);

      // Read pixels from render target
      const pixels = new Uint8Array(targetW * targetH * 4);
      renderer.readRenderTargetPixels(rt, 0, 0, targetW, targetH, pixels);
      renderer.setRenderTarget(null);
      rt.dispose();

      // Flip Y (WebGL is bottom-up) and write to canvas
      const tmpCanvas = document.createElement('canvas');
      tmpCanvas.width = targetW;
      tmpCanvas.height = targetH;
      const tmpCtx = tmpCanvas.getContext('2d');
      const imageData = tmpCtx.createImageData(targetW, targetH);
      for (let y = 0; y < targetH; y++) {{
        const srcRow = (targetH - 1 - y) * targetW * 4;
        const dstRow = y * targetW * 4;
        imageData.data.set(pixels.subarray(srcRow, srcRow + targetW * 4), dstRow);
      }}
      tmpCtx.putImageData(imageData, 0, 0);

      finishScreenshot(tmpCanvas.toDataURL('image/png'));
    }}

    function finishScreenshot(dataURL) {{
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
      const filename = '{Path(model_path).name}_' + timestamp + '.png';
      const link = document.createElement('a');
      link.download = filename;
      link.href = dataURL;
      link.click();
      document.getElementById('screenshot-filename').textContent = filename;
      document.getElementById('screenshot-modal').classList.add('show');
    }}

    function closeScreenshotModal() {{
      document.getElementById('screenshot-modal').classList.remove('show');
    }}

    let lastTime = performance.now(), frames = 0;
    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      // Sync wireframe overlay visibility
      if (wireframeOverlayMode) {{
        wireframeMeshes.forEach((wm, i) => {{ if (i < meshes.length) wm.visible = meshes[i].visible; }});
      }}
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
        print("Usage: python STANDALONE_viewer.py <model_path> [--no-original]")
        print("\nSTANDALONE VERSION - No mdl_viewer.py needed!")
        print("  Uses original normals from .vb by default")
        print("  --no-original  Use computed smooth normals instead")
        print("\nExample:")
        print("  python STANDALONE_viewer.py ./chr5000_c00")
        sys.exit(1)
    
    model_path = sys.argv[1]
    use_original = '--no-original' not in sys.argv
    
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
