import React from 'react'
import { BoundingBox } from '../../types'

interface AnnotationOverlayProps {
  annotations: BoundingBox[]
  imageWidth: number
  imageHeight: number
  className?: string
}

const CLASS_COLORS: Record<string, string> = {}
const PALETTE = [
  '#ef4444', '#3b82f6', '#22c55e', '#eab308', '#a855f7',
  '#ec4899', '#14b8a6', '#f97316', '#6366f1', '#84cc16',
  '#06b6d4', '#d946ef', '#f43f5e', '#0ea5e9', '#10b981',
]

function getClassColor(className: string): string {
  if (!CLASS_COLORS[className]) {
    const idx = Object.keys(CLASS_COLORS).length % PALETTE.length
    CLASS_COLORS[className] = PALETTE[idx]
  }
  return CLASS_COLORS[className]
}

function AnnotationOverlay({ annotations, imageWidth, imageHeight, className }: AnnotationOverlayProps) {
  if (!annotations || annotations.length === 0) return null

  return (
    <svg
      className={`absolute inset-0 w-full h-full pointer-events-none ${className || ''}`}
      viewBox={`0 0 ${imageWidth} ${imageHeight}`}
      preserveAspectRatio="none"
    >
      {annotations.map((ann, i) => {
        const color = getClassColor(ann.class_name)
        const x = (ann.cx - ann.w / 2) * imageWidth
        const y = (ann.cy - ann.h / 2) * imageHeight
        const w = ann.w * imageWidth
        const h = ann.h * imageHeight

        return (
          <g key={i}>
            {/* Bounding box */}
            <rect
              x={x}
              y={y}
              width={w}
              height={h}
              fill="none"
              stroke={color}
              strokeWidth={Math.max(1.5, imageWidth / 200)}
              strokeDasharray={ann.confidence != null ? 'none' : '4 2'}
            />
            {/* Semi-transparent fill */}
            <rect
              x={x}
              y={y}
              width={w}
              height={h}
              fill={color}
              fillOpacity={0.08}
            />
            {/* Label background */}
            <rect
              x={x}
              y={Math.max(0, y - 14 * (imageWidth / 400))}
              width={Math.max(w, 40 * (imageWidth / 400))}
              height={14 * (imageWidth / 400)}
              fill={color}
              rx={2}
            />
            {/* Label text */}
            <text
              x={x + 2}
              y={Math.max(10 * (imageWidth / 400), y - 3 * (imageWidth / 400))}
              fill="white"
              fontSize={Math.max(8, 10 * (imageWidth / 400))}
              fontFamily="monospace"
              fontWeight="bold"
            >
              {ann.class_name}
              {ann.confidence != null ? ` ${(ann.confidence * 100).toFixed(0)}%` : ''}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

export default AnnotationOverlay
