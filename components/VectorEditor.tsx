'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import { fabric } from 'fabric'
import { ArrowLeft, Download, Type, Minus, Plus, RotateCcw, Home, Box } from 'lucide-react'

export interface Template {
  id: string
  name: string
  svgPath: string
  thumbnail?: string
  score?: number
}

interface VectorEditorProps {
  template: Template
  onBack: () => void
  onStartOver: () => void
}

export default function VectorEditor({ template, onBack, onStartOver }: VectorEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fabricCanvasRef = useRef<fabric.Canvas | null>(null)
  const [strokeWidth, setStrokeWidth] = useState<number>(2)
  const [selectedColor, setSelectedColor] = useState<string>('#000000')
  const [isAddingText, setIsAddingText] = useState<boolean>(false)
  const [textInput, setTextInput] = useState<string>('')

  const loadSVGTemplate = useCallback(() => {
    const canvas = fabricCanvasRef.current
    if (!canvas) return

    canvas.clear()
    canvas.setBackgroundColor('white', canvas.renderAll.bind(canvas))

    const svgString = `
      <svg xmlns="http://www.w3.org/2000/svg" width="${canvas.getWidth()}" height="${canvas.getHeight()}">
        <path d="${template.svgPath}"
              fill="${selectedColor}"
              stroke="${selectedColor}"
              stroke-width="${strokeWidth}" />
      </svg>
    `

    fabric.loadSVGFromString(
      svgString,
      (objects, options) => {
        const obj = fabric.util.groupSVGElements(objects, options)
        const scale = Math.min(
          canvas.getWidth() * 0.8 / obj.width!,
          canvas.getHeight() * 0.8 / obj.height!
        )
        obj.scale(scale)
        canvas.add(obj)
        canvas.centerObject(obj)
        canvas.renderAll()
      },
      (err: any) => console.error('SVG parse error:', err)
    )
  }, [template.svgPath, strokeWidth, selectedColor])

  useEffect(() => {
    if (canvasRef.current && !fabricCanvasRef.current) {
      fabricCanvasRef.current = new fabric.Canvas(canvasRef.current, {
        width: 600,
        height: 400,
        backgroundColor: 'white',
      })
    }
    return () => {
      if (fabricCanvasRef.current) {
        fabricCanvasRef.current.dispose()
        fabricCanvasRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    loadSVGTemplate()
  }, [loadSVGTemplate])

  const updateStrokeWidth = (width: number) => {
    setStrokeWidth(width)
  }

  const updateColor = (color: string) => {
    setSelectedColor(color)
  }

  const addText = () => {
    const canvas = fabricCanvasRef.current
    if (!canvas || !textInput.trim()) return
    const text = new fabric.Text(textInput, {
      left: canvas.getWidth() / 2,
      top: canvas.getHeight() / 2,
      fontSize: 20,
      fill: selectedColor,
      selectable: true,
      hasControls: true,
    })
    canvas.add(text)
    canvas.renderAll()
    setTextInput('')
    setIsAddingText(false)
  }

  const resetCanvas = () => {
    loadSVGTemplate()
  }

  const exportSVG = () => {
    const canvas = fabricCanvasRef.current
    if (!canvas) return
    const svg = canvas.toSVG()
    const blob = new Blob([svg], { type: 'image/svg+xml' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${template.name.toLowerCase().replace(/\s+/g, '-')}-keychain.svg`
    link.click()
    URL.revokeObjectURL(url)
  }

  const exportSTL = async () => {
    const canvas = fabricCanvasRef.current
    if (!canvas) return
    const svg = canvas.toSVG()
    try {
      const res = await fetch('/api/export-stl', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ svg, thickness: 3, name: template.name }),
      })
      if (!res.ok) throw new Error('STL export failed')
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `${template.name.toLowerCase().replace(/\s+/g, '-')}-keychain.stl`
      link.click()
      URL.revokeObjectURL(url)
    } catch (e) {
      console.error(e)
      alert('Error exporting STL')
    }
  }

  const colors = ['#000000','#FF0000','#00FF00','#0000FF','#FFFF00','#FF00FF','#00FFFF','#FFA500','#800080','#FFC0CB']

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <h2 className="text-3xl font-bold">Edit Your Design</h2>
        <div className="flex gap-2">
          <button onClick={onBack} className="px-4 py-2 bg-gray-200 rounded">← Back</button>
          <button onClick={onStartOver} className="px-4 py-2 bg-gray-200 rounded">🏠 Start Over</button>
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        <div className="space-y-6">
          <div className="p-4 bg-white rounded shadow">
            <h3 className="font-medium mb-2">Drawing Mode</h3>
            <button
              onClick={() => {
                const canvas = fabricCanvasRef.current
                if (canvas) canvas.isDrawingMode = true
              }}
              className="px-3 py-1 bg-blue-600 text-white rounded mr-2"
            >Enable</button>
            <button
              onClick={() => {
                const canvas = fabricCanvasRef.current
                if (canvas) canvas.isDrawingMode = false
              }}
              className="px-3 py-1 bg-gray-200 rounded"
            >Select</button>
          </div>

          <div className="p-4 bg-white rounded shadow">
            <h3 className="font-medium mb-2">Stroke Width</h3>
            <div className="flex items-center gap-2 mb-2">
              <button
                onClick={() => updateStrokeWidth(Math.max(1, strokeWidth - 1))}
                className="p-1 bg-gray-200 rounded"
              >
                <Minus />
              </button>
              <span className="text-sm font-medium">{strokeWidth}px</span>
              <button
                onClick={() => updateStrokeWidth(strokeWidth + 1)}
                className="p-1 bg-gray-200 rounded"
              >
                <Plus />
              </button>
            </div>
            <input
              type="range"
              min={1}
              max={20}
              value={strokeWidth}
              onChange={e => updateStrokeWidth(+e.target.value)}
              className="w-full"
            />
          </div>

          <div className="p-4 bg-white rounded shadow">
            <h3 className="font-medium mb-2">Colors</h3>
            <div className="grid grid-cols-5 gap-2">
              {colors.map(c => (
                <button
                  key={c}
                  onClick={() => updateColor(c)}
                  className={`w-8 h-8 rounded border-2 ${selectedColor === c ? 'border-black' : 'border-gray-200'}`}
                  style={{ backgroundColor: c }}
                />
              ))}
            </div>
          </div>

          <div className="p-4 bg-white rounded shadow">
            <h3 className="font-medium mb-2">Add Text</h3>
            {isAddingText ? (
              <div className="space-y-2">
                <input
                  type="text"
                  value={textInput}
                  onChange={e => setTextInput(e.target.value)}
                  placeholder="Enter text..."
                  className="w-full border p-2 rounded"
                  onKeyDown={e => e.key === 'Enter' && addText()}
                />
                <div className="flex gap-2">
                  <button
                    onClick={addText}
                    className="px-3 py-1 bg-blue-600 text-white rounded"
                  >Add</button>
                  <button
                    onClick={() => setIsAddingText(false)}
                    className="px-3 py-1 bg-gray-200 rounded"
                  >Cancel</button>
                </div>
              </div>
            ) : (
              <button
                onClick={() => setIsAddingText(true)}
                className="px-3 py-1 bg-gray-200 rounded"
              >+ Text</button>
            )}
          </div>

          <div className="p-4 bg-white rounded shadow">
            <h3 className="font-medium mb-2">Actions</h3>
            <button
              onClick={resetCanvas}
              className="px-3 py-1 bg-gray-200 rounded mr-2"
            >
              <RotateCcw /> Reset
            </button>
            <button
              onClick={exportSVG}
              className="px-3 py-1 bg-green-600 text-white rounded mr-2"
            >
              <Download /> SVG
            </button>
            <button
              onClick={exportSTL}
              className="px-3 py-1 bg-blue-600 text-white rounded"
            >
              <Box /> STL
            </button>
          </div>
        </div>

        <div className="lg:col-span-3 p-4 bg-white rounded shadow">
          <canvas ref={canvasRef} className="w-full h-auto border" />
          <p className="mt-2 text-sm text-gray-500">Drag objects, resize via handles, and style using the controls.</p>
        </div>
      </div>
    </div>
  )
}
