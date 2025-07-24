import io
import numpy as np
import trimesh
from xml.etree import ElementTree as ET
from svg.path import parse_path
import re

class SVGToSTLConverter:
    """
    Convert SVG paths to 3D STL files for keychain manufacturing
    
    This utility extracts paths from SVG files and extrudes them
    to create 3D models suitable for printing.
    """
    
    def __init__(self, thickness=3.0, resolution=100):
        self.thickness = thickness  # mm
        self.resolution = resolution  # points per unit
        
    def parse_svg_path(self, path_string):
        """
        Parse SVG path string into 2D coordinates
        
        Args:
            path_string: SVG path data string
            
        Returns:
            List of (x, y) coordinate tuples
        """
        try:
            path = parse_path(path_string)
            points = []
            
            # Sample the path at regular intervals
            num_samples = max(50, int(path.length() * self.resolution / 100))
            
            for i in range(num_samples + 1):
                t = i / num_samples
                point = path.point(t)
                points.append((point.real, point.imag))
                
            return points
            
        except Exception as e:
            print(f"Error parsing SVG path: {e}")
            # Return a simple rectangle as fallback
            return [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
    
    def svg_to_stl(self, svg_content, output_path=None):
        """
        Convert SVG content to STL file
        
        Args:
            svg_content: SVG file content as string
            output_path: Path to save STL file (optional)
            
        Returns:
            Trimesh object or STL file bytes
        """
        try:
            # Parse SVG XML
            root = ET.fromstring(svg_content)
            
            # Find all path elements
            paths = []
            for path_elem in root.iter():
                if path_elem.tag.endswith('path'):
                    path_data = path_elem.get('d')
                    if path_data:
                        points = self.parse_svg_path(path_data)
                        paths.append(points)
            
            if not paths:
                # Create a default shape if no paths found
                paths = [[(25, 25), (75, 25), (75, 75), (25, 75), (25, 25)]]
            
            # Convert to 3D mesh
            mesh = self.create_3d_mesh(paths)
            
            if output_path:
                mesh.export(output_path)
                return output_path
            else:
                # Return STL data as bytes
                return mesh.export(file_type='stl')
                
        except Exception as e:
            print(f"Error converting SVG to STL: {e}")
            return self.create_fallback_stl()
    
    def create_3d_mesh(self, paths):
        """
        Create 3D mesh from 2D paths
        
        Args:
            paths: List of path coordinate lists
            
        Returns:
            Trimesh object
        """
        try:
            all_vertices = []
            all_faces = []
            vertex_offset = 0
            
            for path in paths:
                if len(path) < 3:
                    continue
                    
                # Create 2D polygon
                vertices_2d = np.array(path[:-1])  # Remove duplicate last point
                
                # Normalize coordinates to reasonable size
                min_coords = vertices_2d.min(axis=0)
                max_coords = vertices_2d.max(axis=0)
                size = max_coords - min_coords
                
                if size.max() > 0:
                    vertices_2d = (vertices_2d - min_coords) / size.max() * 50
                
                # Create bottom face (z=0)
                bottom_vertices = np.column_stack([vertices_2d, np.zeros(len(vertices_2d))])
                
                # Create top face (z=thickness)
                top_vertices = np.column_stack([vertices_2d, np.full(len(vertices_2d), self.thickness)])
                
                # Combine vertices
                vertices = np.vstack([bottom_vertices, top_vertices])
                all_vertices.append(vertices)
                
                # Create faces
                n = len(vertices_2d)
                
                # Bottom face (triangulated)
                for i in range(1, n - 1):
                    face = [vertex_offset, vertex_offset + i, vertex_offset + i + 1]
                    all_faces.append(face)
                
                # Top face (triangulated)
                for i in range(1, n - 1):
                    face = [vertex_offset + n, vertex_offset + n + i + 1, vertex_offset + n + i]
                    all_faces.append(face)
                
                # Side faces
                for i in range(n):
                    next_i = (i + 1) % n
                    
                    # Two triangles per side face
                    face1 = [vertex_offset + i, vertex_offset + next_i, vertex_offset + n + i]
                    face2 = [vertex_offset + next_i, vertex_offset + n + next_i, vertex_offset + n + i]
                    
                    all_faces.append(face1)
                    all_faces.append(face2)
                
                vertex_offset += 2 * n
            
            # Combine all vertices and faces
            combined_vertices = np.vstack(all_vertices)
            combined_faces = np.array(all_faces)
            
            # Create mesh
            mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces)
            
            # Clean up mesh
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
            mesh.remove_unreferenced_vertices()
            
            # Ensure manifold
            if not mesh.is_watertight:
                mesh.fill_holes()
            
            return mesh
            
        except Exception as e:
            print(f"Error creating 3D mesh: {e}")
            return self.create_fallback_mesh()
    
    def create_fallback_mesh(self):
        """Create a simple fallback mesh (cube)"""
        vertices = np.array([
            [0, 0, 0], [50, 0, 0], [50, 50, 0], [0, 50, 0],  # Bottom
            [0, 0, self.thickness], [50, 0, self.thickness], 
            [50, 50, self.thickness], [0, 50, self.thickness]  # Top
        ])
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Bottom
            [4, 6, 5], [4, 7, 6],  # Top
            [0, 4, 5], [0, 5, 1],  # Front
            [2, 6, 7], [2, 7, 3],  # Back
            [0, 3, 7], [0, 7, 4],  # Left
            [1, 5, 6], [1, 6, 2],  # Right
        ])
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def create_fallback_stl(self):
        """Create fallback STL data"""
        mesh = self.create_fallback_mesh()
        return mesh.export(file_type='stl')

def convert_svg_to_stl(svg_content, thickness=3.0, output_path=None):
    """
    Convenience function to convert SVG to STL
    
    Args:
        svg_content: SVG file content as string
        thickness: Extrusion thickness in mm
        output_path: Optional output file path
        
    Returns:
        STL file path or bytes
    """
    converter = SVGToSTLConverter(thickness=thickness)
    return converter.svg_to_stl(svg_content, output_path)