# Release Frame Validation Tool

A Next.js web application for manually validating automatically-detected basketball free throw release frames.

## Purpose

This tool was used to curate the training dataset by allowing human annotators to verify that auto-detected release frames met quality criteria:

1. **Court angle** - Proper camera viewpoint for pose estimation
2. **Ball visibility** - Basketball clearly visible in frame
3. **Release moment** - Frame captures the actual ball release point

## Features

- Canvas-based image viewer with ball position overlay (orange circle)
- Keyboard shortcuts for fast annotation:
  - `A` or `Enter` - Approve frame
  - `R` or `Backspace` - Reject frame
  - Arrow keys - Navigate between frames
- Real-time progress tracking (approved/rejected counts)
- Persistent validation state saved via API
- Automatic skip of already-validated samples

## Setup

```bash
cd labeling-tool
npm install
npm run dev
```

Then open http://localhost:3000

## Data Files Required

- `public/frames/` - Directory containing candidate frame images
- API endpoints expect `candidates.json` and `validated.json` in data directory

## Results

From 1,332 candidate frames:
- **Approved:** 139 (10.4%)
- **Rejected:** 648 (48.6%)
- **Skipped:** 24 (1.8%)

After subsequent pose extraction filtering: **102 final samples**

## Tech Stack

- Next.js 14.2
- React 18
- TypeScript
- Tailwind CSS
