import { NextRequest, NextResponse } from 'next/server'
import { exec } from 'child_process'
import { promisify } from 'util'
import { join } from 'path'
import { existsSync, mkdirSync, readFileSync } from 'fs'

const execAsync = promisify(exec)

const DATA_DIR = join(process.cwd(), '..', 'data', 'Basketball_51 dataset')
const CACHE_DIR = join(process.cwd(), 'public', 'frame-cache')

// Ensure cache directory exists
if (!existsSync(CACHE_DIR)) {
  mkdirSync(CACHE_DIR, { recursive: true })
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const videoId = searchParams.get('videoId')
    const frameNum = searchParams.get('frame')

    if (!videoId || frameNum === null) {
      return NextResponse.json(
        { error: 'Missing videoId or frame parameter' },
        { status: 400 }
      )
    }

    const frame = parseInt(frameNum, 10)
    if (isNaN(frame) || frame < 0) {
      return NextResponse.json(
        { error: 'Invalid frame number' },
        { status: 400 }
      )
    }

    // Cache key includes frame number
    const cacheKey = `${videoId}_frame${frame}`
    const frameFile = join(CACHE_DIR, `${cacheKey}.jpg`)

    // Check cache first
    if (existsSync(frameFile)) {
      const buffer = readFileSync(frameFile)
      return new NextResponse(buffer, {
        headers: {
          'Content-Type': 'image/jpeg',
          'Cache-Control': 'public, max-age=86400',
        },
      })
    }

    // Parse video path from id: ft0_v108_002649_x264 -> ft0/ft0_v108_002649_x264.mp4
    const parts = videoId.split('_')
    const prefix = parts[0]
    const videoPath = join(DATA_DIR, prefix, `${videoId}.mp4`)

    if (!existsSync(videoPath)) {
      return NextResponse.json(
        { error: `Video not found: ${videoPath}` },
        { status: 404 }
      )
    }

    // Extract specific frame using ffmpeg
    // Use select filter to get exact frame by number
    try {
      await execAsync(
        `ffmpeg -i "${videoPath}" -vf "select=eq(n\\,${frame})" -vframes 1 -q:v 2 "${frameFile}" -y`,
        { timeout: 30000 }
      )
    } catch (ffmpegError) {
      // Fallback: use time-based seeking (assuming 30fps)
      const frameTime = (frame / 30).toFixed(3)
      try {
        await execAsync(
          `ffmpeg -ss ${frameTime} -i "${videoPath}" -vframes 1 -q:v 2 "${frameFile}" -y`,
          { timeout: 30000 }
        )
      } catch (e) {
        return NextResponse.json(
          { error: 'FFmpeg failed to extract frame', details: String(e) },
          { status: 500 }
        )
      }
    }

    if (!existsSync(frameFile)) {
      return NextResponse.json(
        { error: 'Frame extraction produced no output' },
        { status: 500 }
      )
    }

    const buffer = readFileSync(frameFile)
    return new NextResponse(buffer, {
      headers: {
        'Content-Type': 'image/jpeg',
        'Cache-Control': 'public, max-age=86400',
      },
    })
  } catch (error) {
    console.error('Error extracting frame:', error)
    return NextResponse.json(
      { error: 'Failed to extract frame', details: String(error) },
      { status: 500 }
    )
  }
}
