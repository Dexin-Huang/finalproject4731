import { NextResponse } from 'next/server'
import { readFileSync, existsSync } from 'fs'
import { join } from 'path'

export async function GET() {
  try {
    // Try multiple locations for candidates.json
    const possiblePaths = [
      join(process.cwd(), '..', 'data', 'candidates.json'),  // ../data/
      join(process.cwd(), 'public', 'candidates.json'),      // public/
      join(process.cwd(), 'data', 'candidates.json'),        // data/
    ]

    let candidatesPath = ''
    for (const p of possiblePaths) {
      if (existsSync(p)) {
        candidatesPath = p
        break
      }
    }

    if (!candidatesPath) {
      return NextResponse.json({
        error: 'candidates.json not found',
        searchedPaths: possiblePaths
      }, { status: 404 })
    }

    const candidates = JSON.parse(readFileSync(candidatesPath, 'utf8'))

    // Try multiple locations for validated.json
    const validatedPaths = [
      join(process.cwd(), '..', 'data', 'validated.json'),
      join(process.cwd(), 'public', 'validated.json'),
    ]

    let validated: Record<string, string> = {}
    for (const p of validatedPaths) {
      if (existsSync(p)) {
        validated = JSON.parse(readFileSync(p, 'utf8'))
        break
      }
    }

    return NextResponse.json({ candidates, validated })
  } catch (error) {
    return NextResponse.json({
      error: 'Failed to load candidates',
      details: String(error)
    }, { status: 500 })
  }
}
