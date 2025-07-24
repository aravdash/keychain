import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const { imageData, mode } = await request.json()

    // Forward request to Python backend
    const backendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001'
    
    const response = await fetch(`${backendUrl}/classify`, {
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
      throw new Error(`Backend responded with status: ${response.status}`)
    }

    const result = await response.json()
    return NextResponse.json(result)

  } catch (error) {
    console.error('Classification error:', error)
    
    // Fallback to mock data if backend is unavailable
    const mockTemplates = [
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
    ]

    const mockVectorized = {
      svgPath: mode === 'draw' 
        ? 'M50 20 L80 50 L50 80 L20 50 Z' 
        : 'M30 30 L70 30 L70 70 L30 70 Z',
      originalImage: (await request.json()).imageData,
    }

    return NextResponse.json({
      templates: mockTemplates,
      vectorized: mockVectorized,
    })
  }
}