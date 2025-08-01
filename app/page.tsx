'use client'

import { useState } from 'react'
import DrawingCanvas from '@/components/DrawingCanvas'
import TemplateGallery from '@/components/TemplateGallery'
import VectorEditor from '@/components/VectorEditor'
import Header from '@/components/Header'

export type InputMode = 'draw' | 'upload'
export type DesignStep = 'input' | 'results' | 'editor'

export interface Template {
  id: string
  name: string
  thumbnail: string
  svgPath: string
  score?: number
}

export interface VectorizedResult {
  svgPath: string
  originalImage?: string
}

export default function Home() {
  const [inputMode, setInputMode] = useState<InputMode>('draw')
  const [currentStep, setCurrentStep] = useState<DesignStep>('input')
  const [templates, setTemplates] = useState<Template[]>([])
  const [vectorizedResult, setVectorizedResult] = useState<VectorizedResult | null>(null)
  const [selectedTemplate, setSelectedTemplate] = useState<Template | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)

  const handleClassificationResult = (result: { templates: Template[], vectorized: VectorizedResult }) => {
    setTemplates(result.templates)
    setVectorizedResult(result.vectorized)
    setCurrentStep('results')
    setIsProcessing(false)
  }

  const handleTemplateSelect = (template: Template) => {
    setSelectedTemplate(template)
    setCurrentStep('editor')
  }

  const handleVectorSelect = () => {
    if (vectorizedResult) {
      setSelectedTemplate({
        id: 'vectorized',
        name: 'Your Drawing',
        thumbnail: vectorizedResult.originalImage || '',
        svgPath: vectorizedResult.svgPath,
      })
      setCurrentStep('editor')
    }
  }

  const handleStartOver = () => {
    setCurrentStep('input')
    setTemplates([])
    setVectorizedResult(null)
    setSelectedTemplate(null)
    setIsProcessing(false)
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <Header />
      
      <div className="container mx-auto px-4 py-8">
        {currentStep === 'input' && (
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-8">
              <h1 className="text-4xl font-bold text-gray-900 mb-4">
                Design Your Perfect Keychain
              </h1>
              <p className="text-lg text-gray-600 mb-8">
                Draw a sketch or upload an image to get AI-powered design suggestions
              </p>
              
              <div className="flex justify-center gap-4 mb-8">
                <button
                  onClick={() => setInputMode('draw')}
                  className={`px-6 py-3 rounded-lg font-medium transition-all ${
                    inputMode === 'draw'
                      ? 'bg-blue-600 text-white shadow-lg'
                      : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  ✏️ Draw Sketch
                </button>
                <button
                  onClick={() => setInputMode('upload')}
                  className={`px-6 py-3 rounded-lg font-medium transition-all ${
                    inputMode === 'upload'
                      ? 'bg-blue-600 text-white shadow-lg'
                      : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  📁 Upload Image
                </button>
              </div>
            </div>

            <DrawingCanvas
              mode={inputMode}
              onResult={handleClassificationResult}
              onProcessingChange={setIsProcessing}
              isProcessing={isProcessing}
            />
          </div>
        )}

        {currentStep === 'results' && (
          <TemplateGallery
            templates={templates}
            vectorizedResult={vectorizedResult}
            onTemplateSelect={handleTemplateSelect}
            onVectorSelect={handleVectorSelect}
            onStartOver={handleStartOver}
          />
        )}

        {currentStep === 'editor' && selectedTemplate && (
          <VectorEditor
            template={selectedTemplate}
            onBack={() => setCurrentStep('results')}
            onStartOver={handleStartOver}
          />
        )}
      </div>
    </main>
  )
}