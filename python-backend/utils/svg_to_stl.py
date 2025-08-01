import io
import numpy as np
import trimesh
from xml.etree import ElementTree as ET
from svg.path import parse_path
from typing import List, Tuple

try:
    from scipy.interpolate import splprep, splev
    SCI_PY_AVAILABLE = True
except ImportError:
    SCI_PY_AVAILABLE = False

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from trimesh.creation import extrude_polygon

class SVGToSTLConverter:
    def __init__(self, thickness: float = 3.0, resolution: int = 500):
        self.thickness = thickness
        self.resolution = resolution

    def parse_svg_path(self, path_string: str) -> List[Tuple[float, float]]:
        try:
            path = parse_path(path_string)
            length = path.length()
            num_samples = max(200, int(length * self.resolution))
            pts = [path.point(i / num_samples) for i in range(num_samples + 1)]
            coords = [(pt.real, pt.imag) for pt in pts]

            if SCI_PY_AVAILABLE and len(coords) > 3:
                x, y = zip(*coords)
                try:
                    tck, u = splprep([x, y], s=length * 0.05)
                    u_new = np.linspace(0, 1, num_samples + 1)
                    x_s, y_s = splev(u_new, tck)
                    coords = list(zip(x_s, y_s))
                except Exception:
                    pass
            return coords
        except Exception as e:
            print(f"Error parsing SVG path: {e}")
            return [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]

    def svg_to_stl(self, svg_content: str, output_path: str = None):
        try:
            root = ET.fromstring(svg_content)
            paths = []
            for elem in root.iter():
                if elem.tag.endswith('path'):
                    d = elem.get('d')
                    if d:
                        pts = self.parse_svg_path(d)
                        paths.append(pts)
            if not paths:
                paths = [[(25,25),(75,25),(75,75),(25,75),(25,25)]]
            mesh = self.create_3d_mesh(paths)
            if output_path:
                mesh.export(output_path)
                return output_path
            return mesh.export(file_type='stl')
        except Exception as e:
            print(f"Error converting SVG to STL: {e}")
            return self.create_fallback_stl()

    def create_3d_mesh(self, paths: List[List[Tuple[float,float]]]) -> trimesh.Trimesh:
        polygons = []
        for pts in paths:
            if len(pts) < 3:
                continue
            try:
                poly = Polygon(pts)
                if poly.is_valid and not poly.is_empty:
                    polygons.append(poly)
            except Exception:
                pass

        base_shape = None
        if polygons:
            polys_sorted = sorted(polygons, key=lambda p: p.area, reverse=True)
            outer = polys_sorted[0]
            holes = [list(p.exterior.coords) for p in polys_sorted[1:] if p.area > 1e-6]
            try:
                base_shape = Polygon(outer.exterior.coords, holes)
            except Exception:
                base_shape = unary_union(polygons)
        else:
            base_shape = Polygon([(0,0),(50,0),(50,50),(0,50)])

        try:
            mesh = extrude_polygon(
                base_shape,
                self.thickness,
                cap_ends=False,
                combine_faces=False,
                sections=max(10, int(self.resolution / 10))
            )
        except Exception as e:
            print(f"Extrusion error: {e}")
            return self.create_fallback_mesh()

        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        if not mesh.is_watertight:
            mesh.fill_holes()
        return mesh

    def create_fallback_mesh(self) -> trimesh.Trimesh:
        verts = np.array([
            [0,0,0],[50,0,0],[50,50,0],[0,50,0],
            [0,0,self.thickness],[50,0,self.thickness],[50,50,self.thickness],[0,50,self.thickness]
        ])
        faces = np.array([
            [0,1,2],[0,2,3], [4,6,5],[4,7,6],
            [0,4,5],[0,5,1], [2,6,7],[2,7,3], [0,3,7],[0,7,4], [1,5,6],[1,6,2]
        ])
        return trimesh.Trimesh(vertices=verts, faces=faces)

    def create_fallback_stl(self):
        return self.create_fallback_mesh().export(file_type='stl')


def convert_svg_to_stl(svg_content: str, thickness: float = 3.0, output_path: str = None):
    converter = SVGToSTLConverter(thickness=thickness)
    return converter.svg_to_stl(svg_content, output_path)
