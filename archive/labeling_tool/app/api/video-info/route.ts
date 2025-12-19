import { NextRequest, NextResponse } from 'next/server'
import { exec } from 'child_process'
import { promisify } from 'util'
import { join } from 'path'
import { existsSync } from 'fs'

const execAsync = promisify(exec)

const DATA_DIR = join(process.cwd(), '..', 'data', 'Basketball_51 dataset')

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const videoId = searchParams.get('videoId')

    if (!videoId) {
      return NextResponse.json(
        { error: 'Missing videoId parameter' },
        { status: 400 }
      )
    }

    // Parse video path from id
    const parts = videoId.split('_')
    const prefix = parts[0]
    const videoPath = join(DATA_DIR, prefix, `${videoId}.mp4`)

    if (!existsSync(videoPath)) {
      return NextResponse.json(
        { error: `Video not found: ${videoPath}` },
        { status: 404 }
      )
    }

    // Use ffprobe to get video info
    const { stdout } = await execAsync(
      `ffprobe -v quiet -print_format json -show_streams -show_format "${videoPath}"`,
      { timeout: 10000 }
    )

    const info = JSON.parse(stdout)
    const videoStream = info.streams?.find((s: any) => s.codec_type === 'video')

    if (!videoStream) {
      return NextResponse.json(
        { error: 'No video stream found' },
        { status: 500 }
      )
    }

    // Calculate total frames
    let totalFrames = 0
    if (videoStream.nb_frames) {
      totalFrames = parseInt(videoStream.nb_frames, 10)
    } else if (info.format?.duration && videoStream.r_frame_rate) {
      // Calculate from duration and frame rate
      const [num, den] = videoStream.r_frame_rate.split('/').map(Number)
      const fps = num / den
      totalFrames = Math.floor(parseFloat(info.format.duration) * fps)
    }

    // Parse frame rate
    let fps = 30
    if (videoStream.r_frame_rate) {
      const [num, den] = videoStream.r_frame_rate.split('/').map(Number)
      fps = num / den
    }

    return NextResponse.json({
      videoId,
      totalFrames,
      fps: Math.round(fps * 100) / 100,
      width: videoStream.width,
      height: videoStream.height,
      duration: parseFloat(info.format?.duration || '0'),
    })
  } catch (error) {
    console.error('Error getting video info:', error)
    return NextResponse.json(
      { error: 'Failed to get video info', details: String(error) },
      { status: 500 }
    )
  }
}
