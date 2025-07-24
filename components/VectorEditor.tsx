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

    // Create a path from the SVG path data
    const path = new fabric.Path(template.svgPath, {
      left: 50,
      top: 50,
      fill: selectedColor,
      stroke: selectedColor,
      strokeWidth: strokeWidth,
      scaleX: 5,
      scaleY: 5,
      selectable: true,
      hasControls: true,
      hasBorders: true,
    })

    canvas.add(path)
    canvas.centerObject(path)
    canvas.renderAll()
  }

  const updateStrokeWidth = (width: number) => {
    setStrokeWidth(width)
    if (fabricCanvasRef.current) {
      const activeObject = fabricCanvasRef.current.getActiveObject()
      if (activeObject && activeObject.type === 'path') {
        activeObject.set('strokeWidth', width)
        fabricCanvasRef.current.renderAll()
      }
    }
  }

  const updateColor = (color: string) => {
    setSelectedColor(color)
    if (fabricCanvasRef.current) {
      const activeObject = fabricCanvasRef.current.getActiveObject()
      if (activeObject) {
        activeObject.set('fill', color)
        activeObject.set('stroke', color)
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