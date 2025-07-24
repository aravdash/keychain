'use client'

import { useEffect, useRef, useState } from 'react'
import { fabric } from 'fabric'
import { Upload, Palette, RotateCcw, Download, Loader2, Zap } from 'lucide-react'
import { InputMode, Template, VectorizedResult } from '@/app/page'

interface DrawingCanvasProps {
  mode: InputMode
  onResult: (result: { templates: Template[], vectorized: VectorizedResult }) => void
  onProcessingChange: (processing: boolean) => void
  isProcessing: boolean
}

export default function DrawingCanvas({ mode, onResult, onProcessingChange, isProcessing }: DrawingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fabricCanvasRef = useRef<fabric.Canvas | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [brushSize, setBrushSize] = useState(5)
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)

  useEffect(() => {
    if (canvasRef.current && !fabricCanvasRef.current) {
      const canvas = new fabric.Canvas(canvasRef.current, {
        isDrawingMode: true,
        width: 600,
        height: 400,
        backgroundColor: 'white',
      })

      canvas.freeDrawingBrush.width = brushSize
      canvas.freeDrawingBrush.color = '#000000'
      
      canvas.on('path:created', () => {
        setIsDrawing(true)
      })

      fabricCanvasRef.current = canvas
    }

    return () => {
      if (fabricCanvasRef.current) {
        fabricCanvasRef.current.dispose()
        fabricCanvasRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    if (fabricCanvasRef.current) {
      fabricCanvasRef.current.freeDrawingBrush.width = brushSize
    }
  }, [brushSize])

  useEffect(() => {
    if (fabricCanvasRef.current) {
      fabricCanvasRef.current.isDrawingMode = mode === 'draw'
    }
  }, [mode])

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && (file.type === 'image/png' || file.type === 'image/jpeg')) {
      const reader = new FileReader()
      reader.onload = (e) => {
        const imageUrl = e.target?.result as string
        setUploadedImage(imageUrl)
        
        if (fabricCanvasRef.current) {
          fabric.Image.fromURL(imageUrl, (img) => {
            if (fabricCanvasRef.current) {
              fabricCanvasRef.current.clear()
              
              // Scale image to fit canvas
              const canvas = fabricCanvasRef.current
              const canvasWidth = canvas.getWidth()
              const canvasHeight = canvas.getHeight()
              
              const scale = Math.min(
                canvasWidth / (img.width || 1),
                canvasHeight / (img.height || 1)
              ) * 0.8
              
              img.scale(scale)
              img.center()
              canvas.add(img)
              canvas.renderAll()
              setIsDrawing(true)
            }
          })
        }
      }
      reader.readAsDataURL(file)
    }
  }

  const clearCanvas = () => {
    if (fabricCanvasRef.current) {
      fabricCanvasRef.current.clear()
      fabricCanvasRef.current.backgroundColor = 'white'
      fabricCanvasRef.current.renderAll()
      setIsDrawing(false)
      setUploadedImage(null)
    }
  }

  const processImage = async () => {
    if (!fabricCanvasRef.current || !isDrawing) return

    onProcessingChange(true)

    try {
      // Convert canvas to image data
      const imageData = fabricCanvasRef.current.toDataURL('image/png')
      
      // Send to backend for classification and vectorization
      const response = await fetch('/api/classify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          imageData,
          mode,
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to process image')
      }

      const result = await response.json()
      onResult(result)
    } catch (error) {
      console.error('Error processing image:', error)
      // Mock result for demo purposes
      const mockResult = {
        templates: [
          {
            id: '1',
            name: 'Heart Keychain',
            thumbnail: '/templates/heart-thumb.svg',
            svgPath: 'M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z',
            score: 0.92,
          },
          {
            id: '2',
            name: 'Star Keychain',
            thumbnail: '/templates/star-thumb.svg',
            svgPath: 'M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z',
            score: 0.85,
          },
          {
            id: '3',
            name: 'Circle Keychain',
            thumbnail: '/templates/circle-thumb.svg',
            svgPath: 'M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z',
            score: 0.78,
          },
        ],
        vectorized: {
          svgPath: 'M50 50 L100 100 L150 50 L100 0 Z',
          originalImage: imageData,
        },
      }
      onResult(mockResult)
    }
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-semibold text-gray-800">
          {mode === 'draw' ? 'Draw Your Design' : 'Upload Your Image'}
        </h2>
        
        <div className="flex gap-2">
          {mode === 'draw' && (
            <div className="flex items-center gap-2">
              <Palette className="w-4 h-4 text-gray-500" />
              <input
                type="range"
                min="1"
                max="20"
                value={brushSize}
                onChange={(e) => setBrushSize(Number(e.target.value))}
                className="w-20"
              />
              <span className="text-sm text-gray-500">{brushSize}px</span>
            </div>
          )}
          
          <button
            onClick={clearCanvas}
            className="flex items-center gap-2 px-3 py-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Clear
          </button>
        </div>
      </div>

      <div className="flex flex-col items-center">
        {mode === 'upload' && (
          <div className="mb-4">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg"
              onChange={handleFileUpload}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Upload className="w-4 h-4" />
              Choose Image
            </button>
          </div>
        )}

        <div className={`canvas-container ${isDrawing ? 'active' : ''} p-4 bg-gray-50 rounded-lg`}>
          <canvas
            ref={canvasRef}
            className="border border-gray-300 rounded-lg bg-white"
          />
        </div>

        <div className="mt-6 flex gap-4">
          <button
            onClick={processImage}
            disabled={!isDrawing || isProcessing}
            className="flex items-center gap-2 px-8 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-medium"
          >
            {isProcessing ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Zap className="w-5 h-5" />
                Generate Designs
              </>
            )}
          </button>
        </div>

        {mode === 'draw' && (
          <p className="text-sm text-gray-500 mt-4 text-center">
            Draw your keychain design in the canvas above, then click "Generate Designs" to get AI suggestions
          </p>
        )}
        
        {mode === 'upload' && !uploadedImage && (
          <p className="text-sm text-gray-500 mt-4 text-center">
            Upload a PNG or JPG image of your design to get started
          </p>
        )}
      </div>
    </div>
  )
}