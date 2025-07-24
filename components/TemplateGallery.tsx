'use client'

import { ArrowLeft, Sparkles, Cpu } from 'lucide-react'
import { Template, VectorizedResult } from '@/app/page'

interface TemplateGalleryProps {
  templates: Template[]
  vectorizedResult: VectorizedResult | null
  onTemplateSelect: (template: Template) => void
  onVectorSelect: () => void
  onStartOver: () => void
}

export default function TemplateGallery({
  templates,
  vectorizedResult,
  onTemplateSelect,
  onVectorSelect,
  onStartOver,
}: TemplateGalleryProps) {
  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Design Suggestions</h2>
          <p className="text-gray-600">Choose a template or your vectorized drawing to continue editing</p>
        </div>
        <button
          onClick={onStartOver}
          className="flex items-center gap-2 px-4 py-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Start Over
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* AI Suggested Templates */}
        <div>
          <div className="flex items-center gap-2 mb-6">
            <Sparkles className="w-5 h-5 text-purple-600" />
            <h3 className="text-xl font-semibold text-gray-800">AI Suggestions</h3>
          </div>
          
          <div className="space-y-4">
            {templates.map((template) => (
              <div
                key={template.id}
                onClick={() => onTemplateSelect(template)}
                className="template-card bg-white rounded-lg p-4 border border-gray-200 cursor-pointer hover:border-purple-300"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="w-16 h-16 bg-gray-100 rounded-lg flex items-center justify-center">
                      <svg
                        width="40"
                        height="40"
                        viewBox="0 0 24 24"
                        fill="none"
                        className="text-gray-600"
                      >
                        <path
                          d={template.svgPath}
                          fill="currentColor"
                        />
                      </svg>
                    </div>
                    <div>
                      <h4 className="font-medium text-gray-900">{template.name}</h4>
                      <p className="text-sm text-gray-500">
                        {template.score ? `${Math.round(template.score * 100)}% match` : 'Template'}
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    {template.score && (
                      <div className="w-12 h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-purple-600 rounded-full"
                          style={{ width: `${template.score * 100}%` }}
                        />
                      </div>
                    )}
                    <span className="text-sm text-purple-600 font-medium">Select →</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Vectorized Result */}
        <div>
          <div className="flex items-center gap-2 mb-6">
            <Cpu className="w-5 h-5 text-blue-600" />
            <h3 className="text-xl font-semibold text-gray-800">Your Vectorized Drawing</h3>
          </div>
          
          {vectorizedResult && (
            <div
              onClick={onVectorSelect}
              className="template-card bg-white rounded-lg p-6 border border-gray-200 cursor-pointer hover:border-blue-300"
            >
              <div className="text-center">
                <div className="w-32 h-32 mx-auto bg-gray-100 rounded-lg flex items-center justify-center mb-4">
                  <svg
                    width="80"
                    height="80"
                    viewBox="0 0 200 200"
                    fill="none"
                    className="text-gray-600"
                  >
                    <path
                      d={vectorizedResult.svgPath}
                      fill="currentColor"
                    />
                  </svg>
                </div>
                
                <h4 className="font-medium text-gray-900 mb-2">Your Design</h4>
                <p className="text-sm text-gray-500 mb-4">
                  Auto-vectorized from your drawing
                </p>
                
                <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-100 text-blue-700 rounded-lg">
                  <span className="text-sm font-medium">Edit Your Design →</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="mt-12 text-center">
        <div className="bg-gray-50 rounded-lg p-6">
          <h4 className="font-medium text-gray-900 mb-2">What's Next?</h4>
          <p className="text-gray-600 text-sm">
            Select a template or your vectorized drawing to open the editor where you can:
          </p>
          <div className="flex justify-center gap-8 mt-4 text-sm text-gray-500">
            <span>• Adjust anchor points</span>
            <span>• Change stroke width</span>
            <span>• Add text</span>
            <span>• Export as SVG/STL</span>
          </div>
        </div>
      </div>
    </div>
  )
}