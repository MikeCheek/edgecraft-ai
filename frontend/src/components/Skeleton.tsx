interface SkeletonProps {
  className?: string
}

export function Skeleton({ className = '' }: SkeletonProps) {
  return (
    <div
      className={`animate-pulse rounded-lg bg-slate-700/50 ${className}`}
    />
  )
}

export function CardSkeleton({ count = 1 }: { count?: number }) {
  return (
    <div className="space-y-3">
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className="rounded-xl border border-slate-700 bg-slate-800/50 p-4 space-y-3"
        >
          <div className="flex items-center gap-3">
            <Skeleton className="h-4 w-4 rounded" />
            <Skeleton className="h-4 w-1/3" />
          </div>
          <Skeleton className="h-3 w-full" />
          <div className="flex gap-2">
            <Skeleton className="h-6 w-16 rounded-full" />
            <Skeleton className="h-6 w-20 rounded-full" />
          </div>
        </div>
      ))}
    </div>
  )
}

export function GridSkeleton({ count = 4 }: { count?: number }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className="p-4 bg-slate-900/60 rounded-lg border border-slate-700 text-center space-y-2"
        >
          <Skeleton className="h-3 w-16 mx-auto" />
          <Skeleton className="h-7 w-12 mx-auto" />
        </div>
      ))}
    </div>
  )
}

export function TreeSkeleton() {
  return (
    <div className="space-y-2 py-6">
      {Array.from({ length: 3 }).map((_, i) => (
        <div key={i} className="flex items-center gap-2 px-2">
          <Skeleton className="h-4 w-4 rounded" />
          <Skeleton className="h-4 w-1/2" />
          <Skeleton className="h-3 w-8 ml-auto" />
        </div>
      ))}
    </div>
  )
}
