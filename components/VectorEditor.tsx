'use client'

import { useEffect, useRef, useState } from 'react'
import { fabric } from 'fabric'
import { ArrowLeft, Download, Type, Minus, Plus, RotateCcw, Home, Box } from 'lucide-react'
import { Template } from '@/app/page'

interface VectorEditorProps {
  template: Template
  onBack: () => void
  onStartOver: () => void
}

export default function VectorEditor({ template, onBack, onStartOver }: VectorEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fabricCanvasRef = useRef<fabric.Canvas | null>(null)
  const [strokeWidth, setStrokeWidth] = useState(2)
  const [selectedColor, setSelectedColor] = useState('#000000')
  const [isAddingText, setIsAddingText] = useState(false)
  const [textInput, setTextInput] = useState('')

  useEffect(() => {
    if (canvasRef.current && !fabricCanvasRef.current) {
      const canvas = new fabric.Canvas(canvasRef.current, {
        width: 600,
        height: 400,
        backgroundColor: 'white',
      })

      fabricCanvasRef.current = canvas

      // Load the SVG path into the canvas
      loadSVGTemplate()
    }

    return () => {
      if (fabricCanvasRef.current) {
        fabricCanvasRef.current.dispose()
        fabricCanvasRef.current = null
      }
    }
  }, [template])

  const loadSVGTemplate = () => {
    if (!fabricCanvasRef.current) return

    const canvas = fabricCanvasRef.current
    canvas.clear()
    canvas.backgroundColor = 'white'

    // Create a path from the SVG path data
    try {
      console.log('Loading template:', template.name, 'SVG:', template.svgPath)
      
      // The SVG paths from the backend are designed for smaller coordinate systems
      // We need to scale and position them properly
      const path = new fabric.Path(template.svgPath, {
        left: 0,
        top: 0,
        fill: 'transparent',
        stroke: selectedColor,
        strokeWidth: strokeWidth,
        scaleX: 15, // Increased scale for better visibility
        scaleY: 15,
        selectable: true,
        hasControls: true,
        hasBorders: true,
        strokeLineCap: 'round',
        strokeLineJoin: 'round',
      })

      canvas.add(path)
      canvas.centerObject(path)
      canvas.renderAll()
      
      console.log('Template loaded successfully')
    } catch (error) {
      console.error('Error loading SVG path:', error)
      console.log('Fallback: Creating simple shape for template:', template.name)
      
      // Fallback: create different shapes based on template name
      let fallbackShape;
      
      if (template.name.toLowerCase().includes('heart')) {
        // Create a heart-like shape
        fallbackShape = new fabric.Path('M12,21.35l-1.45-1.32C5.4,15.36,2,12.28,2,8.5 C2,5.42,4.42,3,7.5,3c1.74,0,3.41,0.81,4.5,2.09C13.09,3.81,14.76,3,16.5,3 C19.58,3,22,5.42,22,8.5c0,3.78-3.4,6.86-8.55,11.54L12,21.35z', {
          left: 200, top: 150, fill: 'transparent', stroke: selectedColor, strokeWidth: strokeWidth,
          scaleX: 8, scaleY: 8, selectable: true, hasControls: true, hasBorders: true,
        })
      } else if (template.name.toLowerCase().includes('star')) {
        // Create a star shape
        fallbackShape = new fabric.Path('M12,2l3.09,6.26L22,9.27l-5,4.87 l1.18,6.88L12,17.77l-6.18,3.25L7,14.14 L2,9.27l6.91-1.01L12,2z', {
          left: 200, top: 150, fill: 'transparent', stroke: selectedColor, strokeWidth: strokeWidth,
          scaleX: 8, scaleY: 8, selectable: true, hasControls: true, hasBorders: true,
        })
      } else if (template.name.toLowerCase().includes('circle')) {
        // Create a circle
        fallbackShape = new fabric.Circle({
          left: 250, top: 150, radius: 50, fill: 'transparent', stroke: selectedColor, strokeWidth: strokeWidth,
          selectable: true, hasControls: true, hasBorders: true,
        })
      } else {
        // Default rectangle
        fallbackShape = new fabric.Rect({
          left: 200, top: 150, width: 200, height: 100, fill: 'transparent', stroke: selectedColor, strokeWidth: strokeWidth,
          selectable: true, hasControls: true, hasBorders: true,
        })
      }
      
      canvas.add(fallbackShape)
      canvas.renderAll()
    }
  }

  const updateStrokeWidth = (width: number) => {
    setStrokeWidth(width)
    if (fabricCanvasRef.current) {
      // Update brush width if in drawing mode
      if (fabricCanvasRef.current.isDrawingMode) {
        fabricCanvasRef.current.freeDrawingBrush.width = width
      }
      
      // Update selected object stroke width
      const activeObject = fabricCanvasRef.current.getActiveObject()
      if (activeObject && (activeObject.type === 'path' || activeObject.type === 'rect' || activeObject.type === 'circle')) {
        activeObject.set('strokeWidth', width)
        fabricCanvasRef.current.renderAll()
      }
    }
  }

  const updateColor = (color: string) => {
    setSelectedColor(color)
    if (fabricCanvasRef.current) {
      // Update brush color if in drawing mode
      if (fabricCanvasRef.current.isDrawingMode) {
        fabricCanvasRef.current.freeDrawingBrush.color = color
      }
      
      // Update selected object color
      const activeObject = fabricCanvasRef.current.getActiveObject()
      if (activeObject) {
        if (activeObject.type === 'text') {
          activeObject.set('fill', color)
        } else {
          activeObject.set('stroke', color)
          if (activeObject.fill !== 'transparent') {
            activeObject.set('fill', color)
          }
        }
        fabricCanvasRef.current.renderAll()
      }
    }
  }

  const addText = () => {
    if (!fabricCanvasRef.current || !textInput.trim()) return

    const text = new fabric.Text(textInput, {
      left: 300,
      top: 200,
      fontSize: 20,
      fill: selectedColor,
      selectable: true,
      hasControls: true,
      hasBorders: true,
    })

    fabricCanvasRef.current.add(text)
    fabricCanvasRef.current.renderAll()
    setTextInput('')
    setIsAddingText(false)
  }

  const resetCanvas = () => {
    if (fabricCanvasRef.current) {
      fabricCanvasRef.current.clear()
      loadSVGTemplate()
    }
  }

  const exportSVG = () => {
    if (!fabricCanvasRef.current) return

    const svg = fabricCanvasRef.current.toSVG()
    const blob = new Blob([svg], { type: 'image/svg+xml' })
    const url = URL.createObjectURL(blob)
    
    const link = document.createElement('a')
    link.href = url
    link.download = `${template.name.toLowerCase().replace(/\s+/g, '-')}-keychain.svg`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  const exportSTL = async () => {
    if (!fabricCanvasRef.current) return

    try {
      const svg = fabricCanvasRef.current.toSVG()
      
      const response = await fetch('/api/export-stl', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          svg,
          thickness: 3, // 3mm thickness for keychain
          name: template.name,
        }),
      })

      if (response.ok) {
        const blob = await response.blob()
        const url = URL.createObjectURL(blob)
        
        const link = document.createElement('a')
        link.href = url
        link.download = `${template.name.toLowerCase().replace(/\s+/g, '-')}-keychain.stl`
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        URL.revokeObjectURL(url)
      } else {
        console.error('Failed to export STL')
        alert('STL export is not available in demo mode')
      }
    } catch (error) {
      console.error('STL export error:', error)
      alert('STL export is not available in demo mode')
    }
  }

  const enableDrawingMode = () => {
    if (fabricCanvasRef.current) {
      fabricCanvasRef.current.isDrawingMode = true
      fabricCanvasRef.current.freeDrawingBrush.width = strokeWidth
      fabricCanvasRef.current.freeDrawingBrush.color = selectedColor
      fabricCanvasRef.current.freeDrawingBrush.strokeLineCap = 'round'
      fabricCanvasRef.current.freeDrawingBrush.strokeLineJoin = 'round'
      console.log('Drawing mode enabled')
    }
  }

  const disableDrawingMode = () => {
    if (fabricCanvasRef.current) {
      fabricCanvasRef.current.isDrawingMode = false
      console.log('Drawing mode disabled')
    }
  }

  const colors = [
    '#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', 
    '#FF00FF', '#00FFFF', '#FFA500', '#800080', '#FFC0CB'
  ]

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Edit Your Design</h2>
          <p className="text-gray-600">Customize your {template.name} and export when ready</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={onBack}
            className="flex items-center gap-2 px-4 py-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back
          </button>
          <button
            onClick={onStartOver}
            className="flex items-center gap-2 px-4 py-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <Home className="w-4 h-4" />
            Start Over
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Controls Panel */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-white rounded-lg p-6 shadow-sm border">
            <h3 className="font-medium text-gray-900 mb-4">Drawing Mode</h3>
            <div className="space-y-2">
              <button
                onClick={enableDrawingMode}
                className="flex items-center gap-2 w-full px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                </svg>
                Enable Drawing
              </button>
              <button
                onClick={disableDrawingMode}
                className="flex items-center gap-2 w-full px-3 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
                </svg>
                Select Mode
              </button>
            </div>
          </div>

          <div className="bg-white rounded-lg p-6 shadow-sm border">
            <h3 className="font-medium text-gray-900 mb-4">Stroke Width</h3>
            <div className="flex items-center gap-2 mb-3">
              <button
                onClick={() => updateStrokeWidth(Math.max(1, strokeWidth - 1))}
                className="p-1 rounded hover:bg-gray-100"
              >
                <Minus className="w-4 h-4" />
              </button>
              <span className="text-sm font-medium w-8 text-center">{strokeWidth}px</span>
              <button
                onClick={() => updateStrokeWidth(strokeWidth + 1)}
                className="p-1 rounded hover:bg-gray-100"
              >
                <Plus className="w-4 h-4" />
              </button>
            </div>
            <input
              type="range"
              min="1"
              max="10"
              value={strokeWidth}
              onChange={(e) => updateStrokeWidth(Number(e.target.value))}
              className="w-full"
            />
          </div>

          <div className="bg-white rounded-lg p-6 shadow-sm border">
            <h3 className="font-medium text-gray-900 mb-4">Colors</h3>
            <div className="grid grid-cols-5 gap-2">
              {colors.map((color) => (
                <button
                  key={color}
                  onClick={() => updateColor(color)}
                  className={`w-8 h-8 rounded border-2 ${
                    selectedColor === color ? 'border-gray-800' : 'border-gray-300'
                  }`}
                  style={{ backgroundColor: color }}
                />
              ))}
            </div>
          </div>

          <div className="bg-white rounded-lg p-6 shadow-sm border">
            <h3 className="font-medium text-gray-900 mb-4">Add Text</h3>
            {!isAddingText ? (
              <button
                onClick={() => setIsAddingText(true)}
                className="flex items-center gap-2 w-full px-3 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <Type className="w-4 h-4" />
                Add Text
              </button>
            ) : (
              <div className="space-y-2">
                <input
                  type="text"
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder="Enter text..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
                  onKeyPress={(e) => e.key === 'Enter' && addText()}
                />
                <div className="flex gap-2">
                  <button
                    onClick={addText}
                    className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700"
                  >
                    Add
                  </button>
                  <button
                    onClick={() => setIsAddingText(false)}
                    className="px-3 py-1 text-gray-600 text-sm rounded hover:bg-gray-100"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>

          <div className="bg-white rounded-lg p-6 shadow-sm border">
            <h3 className="font-medium text-gray-900 mb-4">Actions</h3>
            <div className="space-y-2">
              <button
                onClick={resetCanvas}
                className="flex items-center gap-2 w-full px-3 py-2 text-gray-600 hover:bg-gray-50 rounded-lg transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                Reset
              </button>
            </div>
          </div>
        </div>

        {/* Canvas Area */}
        <div className="lg:col-span-3">
          <div className="bg-white rounded-lg p-6 shadow-sm border">
            <div className="flex justify-between items-center mb-4">
              <h3 className="font-medium text-gray-900">Design Editor</h3>
              <div className="flex gap-2">
                <button
                  onClick={exportSVG}
                  className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                >
                  <Download className="w-4 h-4" />
                  SVG
                </button>
                <button
                  onClick={exportSTL}
                  className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  <Box className="w-4 h-4" />
                  3D STL
                </button>
              </div>
            </div>

            <div className="border border-gray-300 rounded-lg bg-gray-50 p-4">
              <canvas
                ref={canvasRef}
                className="border border-gray-200 rounded bg-white"
              />
            </div>

            <div className="mt-4 text-sm text-gray-500">
              <p>
                <strong>Tips:</strong> Click and drag to move elements. Use the corner handles to resize. 
                Adjust stroke width and colors using the controls on the left.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}