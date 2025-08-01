import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const { svg, thickness, name } = await request.json()

    const backendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001'
    
    const response = await fetch(`${backendUrl}/export-stl`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        svg,
        thickness: thickness || 3.0,
        name: name || 'keychain',
      }),
    })

    if (!response.ok) {
      throw new Error(`Backend responded with status: ${response.status}`)
    }

    const binaryData = await response.arrayBuffer()
    
    return new NextResponse(binaryData, {
      status: 200,
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Disposition': `attachment; filename="${name || 'keychain'}-keychain.stl"`,
      },
    })

  } catch (error) {
    console.error('STL export error:', error)
    return NextResponse.json(
      { error: 'Failed to export STL' },
      { status: 500 }
    )
  }
}