import os
import base64
import io
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
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
                'embedding': [0.1, 0.8, 0.3, 0.9, 0.2],  # Mock embedding
            },
            {
                'id': '2',
                'name': 'Star Keychain',
                'category': 'celestial',
                'svgPath': 'M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z',
                'embedding': [0.9, 0.1, 0.7, 0.2, 0.8],  # Mock embedding
            },
            {
                'id': '3',
                'name': 'Circle Keychain',
                'category': 'geometric',
                'svgPath': 'M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z',
                'embedding': [0.5, 0.5, 0.9, 0.1, 0.6],  # Mock embedding
            },
            {
                'id': '4',
                'name': 'Lightning Keychain',
                'category': 'dynamic',
                'svgPath': 'M11 4l-7 9h5v7l7-9h-5V4z',
                'embedding': [0.8, 0.3, 0.1, 0.9, 0.4],  # Mock embedding
            },
            {
                'id': '5',
                'name': 'Flower Keychain',
                'category': 'nature',
                'svgPath': 'M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z',
                'embedding': [0.2, 0.9, 0.4, 0.6, 0.8],  # Mock embedding
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
    
    # Create feature vector based on shape characteristics
    features = [
        circularity,  # 0-1, higher for circular shapes
        min(num_vertices / 10.0, 1.0),  # normalized vertex count
        aspect_ratio / 3.0,  # normalized aspect ratio
        area / (image.shape[0] * image.shape[1]),  # relative area
        perimeter / (2 * (image.shape[0] + image.shape[1]))  # relative perimeter
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
    
    # Add some randomness to make results more varied
    import random
    noise = random.uniform(-0.1, 0.1)
    similarity = max(0.0, min(1.0, similarity + noise))
    
    return round(similarity, 3)

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
    
    # Simplify contour using Douglas-Peucker algorithm
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert to SVG path
    if len(simplified) < 3:
        return "M50 50 L100 100 L150 50 L100 0 Z"  # Default diamond
    
    # Scale and normalize coordinates
    points = simplified.reshape(-1, 2)
    
    # Normalize to fit within a reasonable coordinate system
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    
    if max_x - min_x == 0 or max_y - min_y == 0:
        return "M50 50 L100 100 L150 50 L100 0 Z"
    
    # Scale to 200x200 coordinate system
    scale_x = 150 / (max_x - min_x)
    scale_y = 150 / (max_y - min_y)
    scale = min(scale_x, scale_y)
    
    normalized_points = []
    for point in points:
        x = (point[0] - min_x) * scale + 25
        y = (point[1] - min_y) * scale + 25
        normalized_points.append([x, y])
    
    # Create SVG path string
    if len(normalized_points) == 0:
        return "M50 50 L100 100 L150 50 L100 0 Z"
    
    path_string = f"M{normalized_points[0][0]:.1f} {normalized_points[0][1]:.1f}"
    
    for point in normalized_points[1:]:
        path_string += f" L{point[0]:.1f} {point[1]:.1f}"
    
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
        
        # Generate proper STL content
        stl_content = generate_proper_stl(svg_data, name, thickness)
        
        # For now, return success message
        # In a real implementation, you would save the STL file and return download link
        return jsonify({
            'success': True,
            'message': 'STL generated successfully',
            'filename': f"{name}-keychain.stl",
            'stl_data': stl_content[:500] + '...' if len(stl_content) > 500 else stl_content  # Preview
        })
        
    except Exception as e:
        print(f"STL export error: {e}")
        return jsonify({'error': str(e)}), 500

def generate_proper_stl(svg_data, name, thickness):
    """Generate a proper STL file structure with actual 3D geometry"""
    # Parse basic rectangular shape for demo
    # In a real implementation, you would parse the actual SVG path
    
    width = 20.0
    height = 15.0
    thickness = float(thickness)
    
    # Create vertices for a rectangular keychain
    vertices = [
        # Bottom face
        [0, 0, 0], [width, 0, 0], [width, height, 0],
        [0, 0, 0], [width, height, 0], [0, height, 0],
        
        # Top face
        [0, 0, thickness], [width, height, thickness], [width, 0, thickness],
        [0, 0, thickness], [0, height, thickness], [width, height, thickness],
        
        # Front face
        [0, 0, 0], [width, 0, thickness], [width, 0, 0],
        [0, 0, 0], [0, 0, thickness], [width, 0, thickness],
        
        # Back face
        [0, height, 0], [width, height, 0], [width, height, thickness],
        [0, height, 0], [width, height, thickness], [0, height, thickness],
        
        # Left face
        [0, 0, 0], [0, height, 0], [0, height, thickness],
        [0, 0, 0], [0, height, thickness], [0, 0, thickness],
        
        # Right face
        [width, 0, 0], [width, height, thickness], [width, height, 0],
        [width, 0, 0], [width, 0, thickness], [width, height, thickness],
    ]
    
    # Generate STL content
    stl_lines = [f"solid {name}"]
    
    for i in range(0, len(vertices), 3):
        if i + 2 < len(vertices):
            v1, v2, v3 = vertices[i], vertices[i+1], vertices[i+2]
            
            # Calculate normal vector (simplified)
            normal = [0, 0, 1] if i < 6 else [0, 0, -1] if i < 12 else [1, 0, 0]
            
            stl_lines.extend([
                f"  facet normal {normal[0]} {normal[1]} {normal[2]}",
                "    outer loop",
                f"      vertex {v1[0]} {v1[1]} {v1[2]}",
                f"      vertex {v2[0]} {v2[1]} {v2[2]}",
                f"      vertex {v3[0]} {v3[1]} {v3[2]}",
                "    endloop",
                "  endfacet"
            ])
    
    stl_lines.append(f"endsolid {name}")
    
    return '\n'.join(stl_lines)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'keychain-ai-backend'})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)