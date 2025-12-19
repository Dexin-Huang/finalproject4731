import { NextRequest, NextResponse } from 'next/server'
import { readFileSync, writeFileSync, existsSync } from 'fs'
import { join } from 'path'

const LABELS_PATH = join(process.cwd(), '..', 'labels', 'release_frames.json')

interface ReleaseLabel {
  release_frame: number
  confidence: 'high' | 'medium' | 'low'
  labeled_at: string
}

type ReleaseLabels = Record<string, ReleaseLabel>

function loadLabels(): ReleaseLabels {
  if (existsSync(LABELS_PATH)) {
    return JSON.parse(readFileSync(LABELS_PATH, 'utf8'))
  }
  return {}
}

function saveLabels(labels: ReleaseLabels): void {
  writeFileSync(LABELS_PATH, JSON.stringify(labels, null, 2))
}

export async function GET() {
  try {
    const labels = loadLabels()
    return NextResponse.json({
      labels,
      count: Object.keys(labels).length
    })
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to load labels', details: String(error) },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const { videoId, releaseFrame, confidence } = await request.json()

    if (!videoId || releaseFrame === undefined) {
      return NextResponse.json(
        { error: 'Missing videoId or releaseFrame' },
        { status: 400 }
      )
    }

    const labels = loadLabels()

    labels[videoId] = {
      release_frame: releaseFrame,
      confidence: confidence || 'high',
      labeled_at: new Date().toISOString(),
    }

    saveLabels(labels)

    return NextResponse.json({
      success: true,
      total: Object.keys(labels).length
    })
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to save label', details: String(error) },
      { status: 500 }
    )
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const { videoId } = await request.json()

    if (!videoId) {
      return NextResponse.json(
        { error: 'Missing videoId' },
        { status: 400 }
      )
    }

    const labels = loadLabels()
    delete labels[videoId]
    saveLabels(labels)

    return NextResponse.json({
      success: true,
      total: Object.keys(labels).length
    })
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to delete label', details: String(error) },
      { status: 500 }
    )
  }
}
