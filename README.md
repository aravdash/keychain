# Keychain Designer

An AI-powered web application for designing custom keychains with drawing, vectorization, and 3D export capabilities.

## Features

- **Dual Input Modes**: Draw freehand or upload PNG/JPG images
- **AI Classification**: Faster R-CNN based object detection and template matching
- **Auto-Vectorization**: OpenCV contour detection with SVG path generation
- **Interactive Editor**: Fabric.js powered vector editing with anchor point manipulation
- **Template Library**: Curated collection of keychain designs with similarity search
- **Export Options**: SVG download and 3D-printable STL generation

## Tech Stack

### Frontend
- **Next.js 14** - React framework with App Router
- **Fabric.js** - Canvas drawing and vector editing
- **Tailwind CSS** - Utility-first CSS framework
- **TypeScript** - Type-safe JavaScript

### Backend
- **Python Flask** - RESTful API server
- **PyTorch** - Machine learning framework
- **OpenCV** - Computer vision and image processing
- **Trimesh** - 3D mesh processing for STL export

### Infrastructure
- **Docker** - Containerization
- **GitHub Actions** - CI/CD pipeline
- **Next.js API Routes** - Frontend API layer

## Quick Start

### Prerequisites
- Node.js 18+
- Python 3.9+
- Docker (optional)

### Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd keychain-designer
   ```

2. **Install frontend dependencies**
   ```bash
   npm install
   ```

3. **Setup Python backend**
   ```bash
   cd python-backend
   pip install -r requirements.txt
   cd ..
   ```

4. **Start development servers**
   ```bash
   # Terminal 1: Start Python backend
   npm run python-dev

   # Terminal 2: Start Next.js frontend
   npm run dev
   ```

5. **Open the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000

### Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build the Docker image manually
docker build -t keychain-designer .
docker run -p 3000:3000 -p 5000:5000 keychain-designer
```

## Usage

### Drawing Mode
1. Select "Draw Sketch" mode
2. Use the canvas to draw your keychain design
3. Adjust brush size as needed
4. Click "Generate Designs" to process

### Upload Mode
1. Select "Upload Image" mode
2. Choose a PNG or JPG file
3. Click "Generate Designs" to process

### Template Selection
1. Review AI-suggested templates with similarity scores
2. Compare with your auto-vectorized drawing
3. Select a template or your drawing to edit

### Vector Editor
1. Drag elements to reposition
2. Use corner handles to resize
3. Adjust stroke width and colors
4. Add text overlays
5. Export as SVG or 3D STL

## API Endpoints

### Classification
```http
POST /api/classify
Content-Type: application/json

{
  "imageData": "data:image/png;base64,...",
  "mode": "draw|upload"
}
```

### STL Export
```http
POST /api/export-stl
Content-Type: application/json

{
  "svg": "<svg>...</svg>",
  "thickness": 3.0,
  "name": "keychain"
}
```

## Project Structure

```
keychain-designer/
├── app/                    # Next.js app directory
│   ├── api/               # API routes
│   ├── globals.css        # Global styles
│   ├── layout.tsx         # Root layout
│   └── page.tsx          # Main page
├── components/            # React components
│   ├── DrawingCanvas.tsx  # Canvas drawing interface
│   ├── TemplateGallery.tsx # AI suggestions display
│   ├── VectorEditor.tsx   # Vector editing interface
│   └── Header.tsx         # Application header
├── python-backend/        # Python Flask backend
│   ├── app.py            # Main Flask application
│   ├── models/           # ML model implementations
│   └── requirements.txt  # Python dependencies
├── public/               # Static assets
├── .github/              # GitHub Actions workflows
├── Dockerfile           # Container configuration
├── docker-compose.yml   # Multi-service setup
└── README.md           # This file
```

## Machine Learning Pipeline

### 1. Object Detection
- **Faster R-CNN** with ResNet-50 backbone
- Pre-trained on COCO, fine-tuned on keychain dataset
- Detects and classifies keychain shapes

### 2. Feature Extraction
- Shape-based features: aspect ratio, solidity, compactness
- Contour analysis for geometric properties
- Deep features from CNN backbone

### 3. Similarity Search
- Cosine similarity between feature vectors
- Template ranking by confidence scores
- Top-3 recommendations with similarity percentages

### 4. Vectorization
- OpenCV contour detection
- Douglas-Peucker polygon simplification
- SVG path generation with normalized coordinates

### 5. 3D Export
- SVG to mesh conversion
- Configurable extrusion thickness
- STL file generation for 3D printing

## Deployment

### Production Build
```bash
npm run build
npm start
```

### Environment Variables
```env
NODE_ENV=production
PYTHON_ENV=production
MODEL_PATH=/app/models/keychain_model.pth
UPLOAD_DIR=/app/uploads
```

### Docker Production
```bash
docker-compose -f docker-compose.yml up -d
```

## Development

### Adding New Templates
1. Add SVG path to `python-backend/app.py`
2. Generate feature embeddings
3. Update template library

### Training Custom Models
1. Prepare keychain dataset with annotations
2. Use `python-backend/models/faster_rcnn_model.py`
3. Export trained model to TorchScript

### Testing
```bash
# Frontend tests
npm test

# Backend tests
cd python-backend
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Fabric.js for canvas manipulation
- OpenCV for computer vision capabilities
- PyTorch for machine learning framework
- Next.js for the excellent React framework