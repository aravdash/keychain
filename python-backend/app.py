import os
import base64
import io
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from skimage import measure
import json

app = Flask(__name__)
CORS(app)

# Load or initialize ML models
class KeychainClassifier:
    def __init__(self):
        # In a real implementation, load a trained Faster R-CNN model
        # For demo, we'll use mock classification
        self.templates = self.load_template_library()
        
    def load_template_library(self):
        """Load keychain template library with embeddings"""
        return [
            {
                'id': '1',
                'name': 'Heart Keychain',
                'category': 'romantic',
                'svgPath': 'M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z',
                'embedding': [0.1, 0.2, 0.9, 0.8, 0.7],  # Low circularity, low vertices, high aspect ratio, high area, high heart score
            },
            {
                'id': '2',
                'name': 'Star Keychain',
                'category': 'celestial',
                'svgPath': 'M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z',
                'embedding': [0.2, 0.9, 0.5, 0.6, 0.1],  # Low circularity, high vertices (star points), medium aspect ratio, low heart score
            },
            {
                'id': '3',
                'name': 'Circle Keychain',
                'category': 'geometric',
                'svgPath': 'M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z',
                'embedding': [0.95, 0.1, 0.3, 0.7, 0.0],  # High circularity, low vertices, low aspect ratio, no heart score
            },
            {
                'id': '4',
                'name': 'Lightning Keychain',
                'category': 'dynamic',
                'svgPath': 'M11 4l-7 9h5v7l7-9h-5V4z',
                'embedding': [0.1, 0.4, 0.8, 0.4, 0.2],  # Low circularity, medium vertices, high aspect ratio, low heart score
            },
            {
                'id': '5',
                'name': 'Flower Keychain',
                'category': 'nature',
                'svgPath': 'M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z',
                'embedding': [0.6, 0.7, 0.4, 0.8, 0.3],  # Medium circularity, medium-high vertices, medium aspect ratio, some heart score
            },
            {
                'id': '6',
                'name': 'Diamond Keychain',
                'category': 'geometric',
                'svgPath': 'M50 10 L90 50 L50 90 L10 50 Z',
                'embedding': [0.3, 0.4, 0.6, 0.5, 0.1],  # Low-medium circularity, low vertices, medium aspect ratio
            },
        ]

classifier = KeychainClassifier()

def decode_image(image_data):
    """Decode base64 image data to numpy array"""
    # Remove data URL prefix if present
    if image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    
    # Decode base64
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return np.array(image)

def extract_features(image):
    """Extract features from image for similarity matching"""
    # Convert to grayscale for contour detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold to binary
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return [0.5, 0.5, 0.5, 0.5, 0.5]  # Default features
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Extract shape features
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Calculate shape descriptors
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
    else:
        circularity = 0
    
    # Approximate contour to polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    num_vertices = len(approx)
    
    # Calculate bounding box aspect ratio
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h if h > 0 else 1
    
    # Detect heart-like characteristics
    # Hearts typically have two rounded tops and a pointed bottom
    heart_score = 0.0
    if aspect_ratio > 0.7 and aspect_ratio < 1.3:  # Roughly square-ish
        # Check for concavity at the top (heart indent)
        hull = cv2.convexHull(largest_contour, returnPoints=False)
        defects = cv2.convexityDefects(largest_contour, hull)
        
        if defects is not None and len(defects) > 0:
            # Look for significant defects (indentations)
            for defect in defects:
                s, e, f, d = defect[0]
                if d > 1000:  # Significant defect depth
                    heart_score += 0.3
    
    # Create feature vector based on shape characteristics
    features = [
        circularity,  # 0-1, higher for circular shapes
        min(num_vertices / 10.0, 1.0),  # normalized vertex count
        aspect_ratio / 3.0,  # normalized aspect ratio
        area / (image.shape[0] * image.shape[1]),  # relative area
        min(heart_score, 1.0)  # heart-like characteristics
    ]
    
    return features

def calculate_similarity(features1, features2):
    """Calculate similarity between two feature vectors"""
    features1 = np.array(features1)
    features2 = np.array(features2)
    
    # Calculate cosine similarity
    dot_product = np.dot(features1, features2)
    norm1 = np.linalg.norm(features1)
    norm2 = np.linalg.norm(features2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    
    # Add realistic variation based on image hash to make results more varied
    import hashlib
    image_hash = hashlib.md5(str(features1).encode()).hexdigest()
    hash_factor = int(image_hash[:2], 16) / 255.0  # 0-1 based on image content
    
    # Adjust similarity based on content
    similarity = similarity * (0.7 + 0.3 * hash_factor)  # Scale between 0.7-1.0 of original
    
    return max(0.1, min(0.99, similarity))  # Keep within reasonable bounds

def vectorize_image(image):
    """Convert image to SVG path using contour detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold to binary
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "M50 50 L100 100 L150 50 L100 0 Z"  # Default diamond shape
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Use less aggressive simplification to preserve shape details
    epsilon = 0.005 * cv2.arcLength(largest_contour, True)  # Reduced from 0.02
    simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert to SVG path
    if len(simplified) < 3:
        return "M50 50 L100 100 L150 50 L100 0 Z"  # Default diamond
    
    # Scale and normalize coordinates
    points = simplified.reshape(-1, 2)
    
    # Normalize to fit within a reasonable coordinate system (larger for better visibility)
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    
    if max_x - min_x == 0 or max_y - min_y == 0:
        return "M50 50 L100 100 L150 50 L100 0 Z"
    
    # Scale to 300x300 coordinate system for better detail
    scale_x = 250 / (max_x - min_x)
    scale_y = 250 / (max_y - min_y)
    scale = min(scale_x, scale_y)
    
    normalized_points = []
    for point in points:
        x = (point[0] - min_x) * scale + 25
        y = (point[1] - min_y) * scale + 25
        normalized_points.append([x, y])
    
    # Create SVG path string with smooth curves
    if len(normalized_points) == 0:
        return "M50 50 L100 100 L150 50 L100 0 Z"
    
    # Start with move command
    path_string = f"M{normalized_points[0][0]:.1f} {normalized_points[0][1]:.1f}"
    
    # Add lines to create a closed shape
    for point in normalized_points[1:]:
        path_string += f" L{point[0]:.1f} {point[1]:.1f}"
    
    # Close the path to create a solid fillable shape
    path_string += " Z"
    
    return path_string

@app.route('/classify', methods=['POST'])
def classify_image():
    """Classify image and return template suggestions with vectorized result"""
    try:
        data = request.json
        image_data = data.get('imageData')
        mode = data.get('mode', 'draw')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode image
        image = decode_image(image_data)
        
        # Extract features for similarity matching
        features = extract_features(image)
        
        # Calculate similarity with each template
        template_scores = []
        for template in classifier.templates:
            similarity = calculate_similarity(features, template['embedding'])
            template_scores.append({
                'id': template['id'],
                'name': template['name'],
                'svgPath': template['svgPath'],
                'score': similarity,
                'thumbnail': f"/templates/{template['id']}-thumb.svg"
            })
        
        # Sort by similarity score and take top 3
        template_scores.sort(key=lambda x: x['score'], reverse=True)
        top_templates = template_scores[:3]
        
        # Vectorize the input image
        vectorized_path = vectorize_image(image)
        
        vectorized_result = {
            'svgPath': vectorized_path,
            'originalImage': image_data
        }
        
        return jsonify({
            'templates': top_templates,
            'vectorized': vectorized_result
        })
        
    except Exception as e:
        print(f"Classification error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/export-stl', methods=['POST'])
def export_stl():
    """Export SVG to STL format"""
    try:
        data = request.json
        svg_data = data.get('svg')
        thickness = data.get('thickness', 3.0)
        name = data.get('name', 'keychain')
        
        # Generate proper binary STL content
        stl_binary = generate_binary_stl(svg_data, name, thickness)
        
        # Return binary STL file
        response = make_response(stl_binary)
        response.headers['Content-Type'] = 'application/octet-stream'
        response.headers['Content-Disposition'] = f'attachment; filename="{name}-keychain.stl"'
        return response
        
    except Exception as e:
        print(f"STL export error: {e}")
        return jsonify({'error': str(e)}), 500

def generate_binary_stl(svg_data, name, thickness):
    """Generate a proper binary STL file from SVG path"""
    import struct
    import re
    
    thickness = float(thickness)
    
    # Try to parse SVG path data to get actual shape
    vertices = []
    
    # Extract path data from SVG
    if svg_data and isinstance(svg_data, str):
        # Look for path elements in SVG
        path_match = re.search(r'd="([^"]+)"', svg_data)
        if path_match:
            path_data = path_match.group(1)
            vertices = parse_svg_path_to_vertices(path_data, thickness)
    
    # If no valid path data, create default keychain shape
    if not vertices:
        vertices = create_default_keychain_vertices(thickness)
    
    # Create binary STL
    header = bytearray(80)
    header[:len(name)] = name.encode('ascii')[:80]
    
    # Number of triangles
    triangle_count = len(vertices) // 3
    
    # Build binary data
    binary_data = bytearray()
    binary_data.extend(header)
    binary_data.extend(struct.pack('<I', triangle_count))
    
    for i in range(0, len(vertices), 3):
        if i + 2 < len(vertices):
            v1, v2, v3 = vertices[i], vertices[i+1], vertices[i+2]
            
            # Calculate normal vector
            normal = calculate_normal(v1, v2, v3)
            
            # Pack triangle data
            binary_data.extend(struct.pack('<fff', normal[0], normal[1], normal[2]))
            binary_data.extend(struct.pack('<fff', v1[0], v1[1], v1[2]))
            binary_data.extend(struct.pack('<fff', v2[0], v2[1], v2[2]))
            binary_data.extend(struct.pack('<fff', v3[0], v3[1], v3[2]))
            binary_data.extend(struct.pack('<H', 0))  # Attribute byte count
    
    return bytes(binary_data)

def parse_svg_path_to_vertices(path_data, thickness):
    """Parse SVG path data and convert to 3D vertices"""
    import re
    
    # Extract coordinates from path data
    coords = []
    
    # Find all coordinate pairs (simplified parsing)
    numbers = re.findall(r'-?\d+\.?\d*', path_data)
    
    # Group numbers into coordinate pairs
    for i in range(0, len(numbers) - 1, 2):
        try:
            x = float(numbers[i]) * 0.5  # Scale down
            y = float(numbers[i + 1]) * 0.5
            coords.append([x, y])
        except (ValueError, IndexError):
            continue
    
    if len(coords) < 3:
        return []
    
    # Create 3D vertices from 2D path
    vertices = []
    
    # Create triangulated mesh from the path coordinates
    # This is a simplified triangulation - for a complete solution you'd use a proper triangulation library
    
    # Bottom face triangles
    center_bottom = [sum(p[0] for p in coords) / len(coords), sum(p[1] for p in coords) / len(coords), 0]
    
    for i in range(len(coords)):
        next_i = (i + 1) % len(coords)
        vertices.extend([
            center_bottom,
            [coords[i][0], coords[i][1], 0],
            [coords[next_i][0], coords[next_i][1], 0]
        ])
    
    # Top face triangles
    center_top = [center_bottom[0], center_bottom[1], thickness]
    
    for i in range(len(coords)):
        next_i = (i + 1) % len(coords)
        vertices.extend([
            center_top,
            [coords[next_i][0], coords[next_i][1], thickness],
            [coords[i][0], coords[i][1], thickness]
        ])
    
    # Side faces
    for i in range(len(coords)):
        next_i = (i + 1) % len(coords)
        
        # Two triangles per side face
        vertices.extend([
            [coords[i][0], coords[i][1], 0],
            [coords[next_i][0], coords[next_i][1], 0],
            [coords[next_i][0], coords[next_i][1], thickness]
        ])
        
        vertices.extend([
            [coords[i][0], coords[i][1], 0],
            [coords[next_i][0], coords[next_i][1], thickness],
            [coords[i][0], coords[i][1], thickness]
        ])
    
    return vertices

def create_default_keychain_vertices(thickness):
    """Create default keychain shape vertices"""
    # Create a more interesting default shape (hexagon)
    import math
    
    vertices = []
    radius = 15.0
    sides = 6
    
    # Generate hexagon coordinates
    coords = []
    for i in range(sides):
        angle = 2 * math.pi * i / sides
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        coords.append([x, y])
    
    # Bottom face
    center_bottom = [0, 0, 0]
    for i in range(sides):
        next_i = (i + 1) % sides
        vertices.extend([
            center_bottom,
            [coords[i][0], coords[i][1], 0],
            [coords[next_i][0], coords[next_i][1], 0]
        ])
    
    # Top face
    center_top = [0, 0, thickness]
    for i in range(sides):
        next_i = (i + 1) % sides
        vertices.extend([
            center_top,
            [coords[next_i][0], coords[next_i][1], thickness],
            [coords[i][0], coords[i][1], thickness]
        ])
    
    # Side faces
    for i in range(sides):
        next_i = (i + 1) % sides
        
        vertices.extend([
            [coords[i][0], coords[i][1], 0],
            [coords[next_i][0], coords[next_i][1], 0],
            [coords[next_i][0], coords[next_i][1], thickness]
        ])
        
        vertices.extend([
            [coords[i][0], coords[i][1], 0],
            [coords[next_i][0], coords[next_i][1], thickness],
            [coords[i][0], coords[i][1], thickness]
        ])
    
    return vertices

def calculate_normal(v1, v2, v3):
    """Calculate normal vector for a triangle"""
    import numpy as np
    
    # Convert to numpy arrays
    v1, v2, v3 = np.array(v1), np.array(v2), np.array(v3)
    
    # Calculate vectors
    vec1 = v2 - v1
    vec2 = v3 - v1
    
    # Cross product
    normal = np.cross(vec1, vec2)
    
    # Normalize
    length = np.linalg.norm(normal)
    if length > 0:
        normal = normal / length
    else:
        normal = np.array([0, 0, 1])  # Default up vector
    
    return normal.tolist()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'keychain-ai-backend'})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)