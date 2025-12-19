'use client'

import { useState, useEffect, useRef, useCallback } from 'react'

interface Candidate {
  video_id: string
  actual_video_frame: number
  label: number
  ball_pos: [number, number]
}

interface VideoInfo {
  totalFrames: number
  fps: number
  width: number
  height: number
}

interface ReleaseLabel {
  release_frame: number
  confidence: 'high' | 'medium' | 'low'
}

export default function ReleaseLabeler() {
  const [allCandidates, setAllCandidates] = useState<Candidate[]>([])
  const [trainLabels, setTrainLabels] = useState<Record<string, ReleaseLabel>>({})
  const [testLabels, setTestLabels] = useState<Record<string, ReleaseLabel>>({})
  const [currentIndex, setCurrentIndex] = useState(0)
  const [currentFrame, setCurrentFrame] = useState(0)
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [frameLoading, setFrameLoading] = useState(false)
  const [confidence, setConfidence] = useState<'high' | 'medium' | 'low'>('high')
  const [filter, setFilter] = useState<'all' | 'made' | 'miss' | 'unlabeled'>('all')
  const [mode, setMode] = useState<'train' | 'test'>('train')
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageCache = useRef<Map<string, HTMLImageElement>>(new Map())

  // Current labels based on mode
  const labels = mode === 'train' ? trainLabels : testLabels
  const setLabels = mode === 'train' ? setTrainLabels : setTestLabels

  // Filter candidates based on selection and mode
  const candidates = allCandidates.filter(c => {
    // In test mode, exclude videos already in training set
    if (mode === 'test' && trainLabels[c.video_id]) {
      return false
    }
    if (filter === 'made') return c.label === 1
    if (filter === 'miss') return c.label === 0
    if (filter === 'unlabeled') return !labels[c.video_id]
    return true
  })

  const current = candidates[currentIndex]
  const labeledCount = Object.keys(labels).length
  const total = candidates.length

  // Count made/miss in filtered view
  const madeCount = candidates.filter(c => c.label === 1).length
  const missCount = candidates.filter(c => c.label === 0).length
  const labeledInFilter = candidates.filter(c => labels[c.video_id]).length

  // Reset index if out of bounds after filter change
  useEffect(() => {
    if (currentIndex >= candidates.length && candidates.length > 0) {
      setCurrentIndex(0)
    }
  }, [candidates.length, currentIndex])

  // Load candidates and existing labels
  useEffect(() => {
    async function load() {
      try {
        // Load candidates from main candidates API
        const candidatesRes = await fetch('/api/candidates')
        const candidatesData = await candidatesRes.json()

        // Load existing training labels
        const trainLabelsRes = await fetch('/api/release-labels')
        const trainLabelsData = await trainLabelsRes.json()

        // Load existing test labels
        const testLabelsRes = await fetch('/api/test-labels')
        const testLabelsData = await testLabelsRes.json()

        const validCandidates = (candidatesData.candidates || []).filter(
          (c: Candidate) => c.actual_video_frame !== undefined
        )

        setAllCandidates(validCandidates)
        setTrainLabels(trainLabelsData.labels || {})
        setTestLabels(testLabelsData.labels || {})

        // Find first unlabeled candidate
        const firstUnlabeled = validCandidates.findIndex(
          (c: Candidate) => !trainLabelsData.labels?.[c.video_id]
        )
        if (firstUnlabeled >= 0) {
          setCurrentIndex(firstUnlabeled)
        }

        setLoading(false)
      } catch (error) {
        console.error('Failed to load data:', error)
        setLoading(false)
      }
    }
    load()
  }, [])

  // Load video info when candidate changes
  useEffect(() => {
    if (!current) return

    async function loadVideoInfo() {
      try {
        const res = await fetch(`/api/video-info?videoId=${current.video_id}`)
        const info = await res.json()
        if (!res.ok) {
          console.error('Failed to load video info:', info.error)
          setVideoInfo(null)
          return
        }
        setVideoInfo(info)

        // Set initial frame to the estimated release frame
        const existingLabel = labels[current.video_id]
        if (existingLabel) {
          setCurrentFrame(existingLabel.release_frame)
        } else {
          setCurrentFrame(current.actual_video_frame || 0)
        }
      } catch (error) {
        console.error('Failed to load video info:', error)
        setVideoInfo(null)
      }
    }
    loadVideoInfo()
  }, [current, labels])

  // Draw frame on canvas
  const drawFrame = useCallback(async () => {
    if (!current || !canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    setFrameLoading(true)

    const cacheKey = `${current.video_id}_${currentFrame}`
    let img = imageCache.current.get(cacheKey)

    if (!img) {
      img = new Image()
      img.crossOrigin = 'anonymous'

      await new Promise<void>((resolve, reject) => {
        img!.onload = () => {
          imageCache.current.set(cacheKey, img!)
          resolve()
        }
        img!.onerror = reject
        img!.src = `/api/video-frame?videoId=${current.video_id}&frame=${currentFrame}`
      })
    }

    canvas.width = img.width
    canvas.height = img.height
    ctx.drawImage(img, 0, 0)

    // Draw ball position indicator if available
    if (current.ball_pos) {
      const [x, y] = current.ball_pos
      // Scale ball position from original resolution
      const scaleX = img.width / 320  // Original thumbnail was 320px
      const scaleY = img.height / 240
      const scaledX = x * scaleX
      const scaledY = y * scaleY

      ctx.beginPath()
      ctx.arc(scaledX, scaledY, 25, 0, Math.PI * 2)
      ctx.strokeStyle = '#fb923c'
      ctx.lineWidth = 3
      ctx.stroke()
    }

    setFrameLoading(false)
  }, [current, currentFrame])

  useEffect(() => {
    drawFrame()
  }, [drawFrame])

  // Preload adjacent frames
  useEffect(() => {
    if (!current || !videoInfo) return

    const preloadFrames = [currentFrame - 1, currentFrame + 1, currentFrame - 2, currentFrame + 2]
    preloadFrames.forEach(frame => {
      if (frame >= 0 && frame < videoInfo.totalFrames) {
        const cacheKey = `${current.video_id}_${frame}`
        if (!imageCache.current.has(cacheKey)) {
          const img = new Image()
          img.crossOrigin = 'anonymous'
          img.onload = () => imageCache.current.set(cacheKey, img)
          img.src = `/api/video-frame?videoId=${current.video_id}&frame=${frame}`
        }
      }
    })
  }, [current, currentFrame, videoInfo])

  // Save release frame label
  const saveLabel = async () => {
    if (!current) return

    const endpoint = mode === 'train' ? '/api/release-labels' : '/api/test-labels'

    try {
      await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          videoId: current.video_id,
          releaseFrame: currentFrame,
          confidence,
        }),
      })

      if (mode === 'train') {
        setTrainLabels(prev => ({
          ...prev,
          [current.video_id]: { release_frame: currentFrame, confidence },
        }))
      } else {
        setTestLabels(prev => ({
          ...prev,
          [current.video_id]: { release_frame: currentFrame, confidence },
        }))
      }

      // Move to next unlabeled
      goToNextUnlabeled()
    } catch (error) {
      console.error('Failed to save label:', error)
    }
  }

  // Skip current video
  const skipVideo = () => {
    goToNextUnlabeled()
  }

  const goToNextUnlabeled = useCallback(() => {
    let next = currentIndex + 1
    while (next < candidates.length && labels[candidates[next]?.video_id]) {
      next++
    }
    if (next < candidates.length) {
      setCurrentIndex(next)
      imageCache.current.clear() // Clear cache when changing videos
    }
  }, [currentIndex, candidates, labels])

  const goToPrevVideo = useCallback(() => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1)
      imageCache.current.clear()
    }
  }, [currentIndex])

  const goToNextVideo = useCallback(() => {
    if (currentIndex < candidates.length - 1) {
      setCurrentIndex(currentIndex + 1)
      imageCache.current.clear()
    }
  }, [currentIndex, candidates.length])

  // Frame navigation
  const goToFrame = useCallback((frame: number) => {
    if (videoInfo && frame >= 0 && frame < videoInfo.totalFrames) {
      setCurrentFrame(frame)
    }
  }, [videoInfo])

  const prevFrame = useCallback(() => goToFrame(currentFrame - 1), [currentFrame, goToFrame])
  const nextFrame = useCallback(() => goToFrame(currentFrame + 1), [currentFrame, goToFrame])
  const jumpBack = useCallback(() => goToFrame(currentFrame - 5), [currentFrame, goToFrame])
  const jumpForward = useCallback(() => goToFrame(currentFrame + 5), [currentFrame, goToFrame])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return

      switch (e.key) {
        case 'ArrowLeft':
          e.preventDefault()
          if (e.shiftKey) {
            jumpBack()
          } else {
            prevFrame()
          }
          break
        case 'ArrowRight':
          e.preventDefault()
          if (e.shiftKey) {
            jumpForward()
          } else {
            nextFrame()
          }
          break
        case ' ':
        case 'Enter':
          e.preventDefault()
          saveLabel()
          break
        case 's':
        case 'S':
          e.preventDefault()
          skipVideo()
          break
        case 'ArrowUp':
          e.preventDefault()
          goToPrevVideo()
          break
        case 'ArrowDown':
          e.preventDefault()
          goToNextVideo()
          break
        case '1':
          setConfidence('high')
          break
        case '2':
          setConfidence('medium')
          break
        case '3':
          setConfidence('low')
          break
      }
    }

    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [prevFrame, nextFrame, jumpBack, jumpForward, goToPrevVideo, goToNextVideo])

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-stone-50">
        <p className="text-xl tracking-[0.2em] text-stone-400 uppercase">Loading</p>
      </div>
    )
  }

  const currentLabel = current ? labels[current.video_id] : undefined
  const estimatedFrame = current?.actual_video_frame || 0
  const isInTrainSet = current ? !!trainLabels[current.video_id] : false

  return (
    <div className="h-screen flex flex-col bg-stone-50 overflow-hidden">
      {/* Header */}
      <header className="pt-6 pb-4 px-12 flex items-end justify-between border-b border-stone-200">
        <div className="space-y-1">
          {/* Mode Toggle */}
          <div className="flex items-center gap-2 mb-2">
            <button
              onClick={() => { setMode('train'); setCurrentIndex(0); }}
              className={`px-4 py-2 text-sm font-medium uppercase tracking-wider rounded-lg transition-colors ${
                mode === 'train'
                  ? 'bg-blue-600 text-white'
                  : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
              }`}
            >
              Training Set ({Object.keys(trainLabels).length})
            </button>
            <button
              onClick={() => { setMode('test'); setCurrentIndex(0); }}
              className={`px-4 py-2 text-sm font-medium uppercase tracking-wider rounded-lg transition-colors ${
                mode === 'test'
                  ? 'bg-purple-600 text-white'
                  : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
              }`}
            >
              Test Set ({Object.keys(testLabels).length})
            </button>
          </div>
          <p className="text-xs tracking-[0.25em] text-stone-500 uppercase font-medium">
            {mode === 'train' ? 'Training Set Labeler' : 'Test Set Labeler'}
          </p>
          <h1 className="text-5xl font-light tracking-tight text-stone-800">
            {currentIndex + 1}
            <span className="text-stone-300 mx-3">/</span>
            <span className="text-stone-400">{total}</span>
          </h1>
        </div>

        <div className="text-right space-y-3">
          {/* Filter buttons */}
          <div className="flex items-center justify-end gap-2 mb-2">
            <span className="text-xs tracking-widest text-stone-500 uppercase mr-2">Filter:</span>
            {(['all', 'made', 'miss', 'unlabeled'] as const).map((f) => (
              <button
                key={f}
                onClick={() => { setFilter(f); setCurrentIndex(0); }}
                className={`px-3 py-1 text-xs font-medium uppercase tracking-wider rounded transition-colors ${
                  filter === f
                    ? f === 'made'
                      ? 'bg-emerald-500 text-white'
                      : f === 'miss'
                      ? 'bg-rose-500 text-white'
                      : f === 'unlabeled'
                      ? 'bg-amber-500 text-white'
                      : 'bg-stone-800 text-white'
                    : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                }`}
              >
                {f}
              </button>
            ))}
          </div>
          <div className="flex items-center justify-end gap-6">
            <div>
              <p className="text-xs tracking-widest text-stone-500 uppercase mb-1">Made</p>
              <p className="text-2xl font-light text-emerald-600 tracking-tight">{madeCount}</p>
            </div>
            <div>
              <p className="text-xs tracking-widest text-stone-500 uppercase mb-1">Miss</p>
              <p className="text-2xl font-light text-rose-500 tracking-tight">{missCount}</p>
            </div>
            <div>
              <p className="text-xs tracking-widest text-stone-500 uppercase mb-1">Labeled</p>
              <p className="text-2xl font-light text-blue-600 tracking-tight">{labeledInFilter}/{total}</p>
            </div>
          </div>
        </div>
      </header>

      {/* Test mode info banner */}
      {mode === 'test' && (
        <div className="bg-purple-50 border-b border-purple-200 px-12 py-2">
          <p className="text-sm text-purple-700">
            <strong>Test Set Mode:</strong> Videos in the training set ({Object.keys(trainLabels).length}) are excluded.
            Label videos here for held-out evaluation.
          </p>
        </div>
      )}

      {/* Main Content */}
      <main className="flex-1 px-12 py-6 min-h-0 flex gap-8">
        {/* Video Canvas */}
        <div className="flex-1 relative rounded-xl overflow-hidden bg-black shadow-2xl">
          <canvas
            ref={canvasRef}
            className="w-full h-full object-contain"
            style={{ maxWidth: '100%', maxHeight: '100%' }}
          />

          {frameLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50">
              <p className="text-lg tracking-widest text-white/80 uppercase animate-pulse">
                Loading Frame
              </p>
            </div>
          )}

          {currentLabel && (
            <div className={`absolute top-6 right-6 px-4 py-2 text-white text-sm font-medium tracking-widest uppercase ${
              mode === 'train' ? 'bg-emerald-500' : 'bg-purple-500'
            }`}>
              Labeled: Frame {currentLabel.release_frame}
            </div>
          )}

          {current && (
            <div className="absolute bottom-6 left-6 space-y-1">
              <p className="text-xs tracking-wider text-white/60 font-mono">
                {current.video_id}
              </p>
              <p className="text-xs text-white/40">
                Label: {current.label === 1 ? 'Make' : 'Miss'}
                {isInTrainSet && mode === 'test' && ' (in training set)'}
              </p>
            </div>
          )}
        </div>

        {/* Controls Panel */}
        <div className="w-80 flex flex-col gap-6">
          {/* Frame Info */}
          <div className="bg-white rounded-xl p-6 shadow-sm border border-stone-200">
            <h2 className="text-xs tracking-[0.2em] text-stone-500 uppercase font-medium mb-4">
              Frame Navigation
            </h2>

            <div className="text-center mb-6">
              <p className="text-6xl font-light text-stone-800 tracking-tight">
                {currentFrame}
              </p>
              <p className="text-sm text-stone-400 mt-1">
                of {videoInfo?.totalFrames || '?'} frames
              </p>
            </div>

            {/* Frame slider */}
            {videoInfo && (
              <input
                type="range"
                min={0}
                max={videoInfo.totalFrames - 1}
                value={currentFrame}
                onChange={(e) => goToFrame(parseInt(e.target.value, 10))}
                className="w-full h-2 bg-stone-200 rounded-lg appearance-none cursor-pointer mb-4"
              />
            )}

            {/* Frame buttons */}
            <div className="grid grid-cols-4 gap-2 mb-4">
              <button
                onClick={jumpBack}
                className="py-2 text-xs font-medium bg-stone-100 hover:bg-stone-200 rounded transition-colors"
              >
                -5
              </button>
              <button
                onClick={prevFrame}
                className="py-2 text-xs font-medium bg-stone-100 hover:bg-stone-200 rounded transition-colors"
              >
                -1
              </button>
              <button
                onClick={nextFrame}
                className="py-2 text-xs font-medium bg-stone-100 hover:bg-stone-200 rounded transition-colors"
              >
                +1
              </button>
              <button
                onClick={jumpForward}
                className="py-2 text-xs font-medium bg-stone-100 hover:bg-stone-200 rounded transition-colors"
              >
                +5
              </button>
            </div>

            {/* Estimated frame marker */}
            <div className="flex items-center justify-between text-sm">
              <span className="text-stone-500">Estimated release:</span>
              <button
                onClick={() => goToFrame(estimatedFrame)}
                className="text-blue-600 hover:text-blue-800 font-medium"
              >
                Frame {estimatedFrame}
              </button>
            </div>
          </div>

          {/* Confidence selector */}
          <div className="bg-white rounded-xl p-6 shadow-sm border border-stone-200">
            <h2 className="text-xs tracking-[0.2em] text-stone-500 uppercase font-medium mb-4">
              Confidence
            </h2>
            <div className="flex gap-2">
              {(['high', 'medium', 'low'] as const).map((level) => (
                <button
                  key={level}
                  onClick={() => setConfidence(level)}
                  className={`flex-1 py-2 text-xs font-medium uppercase tracking-wider rounded transition-colors ${
                    confidence === level
                      ? level === 'high'
                        ? 'bg-emerald-500 text-white'
                        : level === 'medium'
                        ? 'bg-amber-500 text-white'
                        : 'bg-rose-500 text-white'
                      : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                  }`}
                >
                  {level}
                </button>
              ))}
            </div>
            <p className="text-xs text-stone-400 mt-3 text-center">
              Press 1, 2, 3 to set confidence
            </p>
          </div>

          {/* Action buttons */}
          <div className="space-y-3">
            <button
              onClick={saveLabel}
              className={`w-full py-4 text-sm font-medium tracking-[0.15em] uppercase
                text-white transition-colors rounded-lg ${
                  mode === 'train'
                    ? 'bg-emerald-500 hover:bg-emerald-600'
                    : 'bg-purple-500 hover:bg-purple-600'
                }`}
            >
              Save to {mode === 'train' ? 'Training' : 'Test'} Set
            </button>
            <button
              onClick={skipVideo}
              className="w-full py-3 text-sm font-medium tracking-[0.15em] uppercase
                bg-white text-stone-600 hover:bg-stone-100
                transition-colors rounded-lg border border-stone-200"
            >
              Skip Video
            </button>
          </div>

          {/* Video navigation */}
          <div className="bg-white rounded-xl p-6 shadow-sm border border-stone-200">
            <h2 className="text-xs tracking-[0.2em] text-stone-500 uppercase font-medium mb-4">
              Video Navigation
            </h2>
            <div className="flex gap-2">
              <button
                onClick={goToPrevVideo}
                disabled={currentIndex === 0}
                className="flex-1 py-2 text-xs font-medium bg-stone-100 hover:bg-stone-200
                  disabled:opacity-50 disabled:cursor-not-allowed rounded transition-colors"
              >
                Previous
              </button>
              <button
                onClick={goToNextVideo}
                disabled={currentIndex >= candidates.length - 1}
                className="flex-1 py-2 text-xs font-medium bg-stone-100 hover:bg-stone-200
                  disabled:opacity-50 disabled:cursor-not-allowed rounded transition-colors"
              >
                Next
              </button>
            </div>
          </div>
        </div>
      </main>

      {/* Footer with shortcuts */}
      <footer className="px-12 py-4 border-t border-stone-200 bg-white">
        <p className="text-center text-xs tracking-widest text-stone-400 uppercase">
          Left/Right: Frame &nbsp;&nbsp; Shift+Left/Right: Jump 5 &nbsp;&nbsp;
          Space/Enter: Save &nbsp;&nbsp; S: Skip &nbsp;&nbsp; Up/Down: Prev/Next Video
        </p>
      </footer>
    </div>
  )
}
