import { useEffect, useRef, useState } from 'react'
import { useAPI } from './useAPI'

export function useHealthCheck () {
  const [isHealthy, setIsHealthy] = useState(false)
  const { apiClient } = useAPI()

  useEffect(() => {
    const checkHealth = async () => {
      try {
        await apiClient.health()
        setIsHealthy(true)
      } catch {
        setIsHealthy(false)
      }
    }

    checkHealth()
    const interval = setInterval(checkHealth, 30000)
    return () => clearInterval(interval)
  }, [apiClient])

  return isHealthy
}

export function usePolling<T> (
  fetchFn: () => Promise<T | null>,
  interval: number = 2000,
  enabled: boolean = true
) {
  const [data, setData] = useState<T | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const fetchFnRef = useRef(fetchFn)

  // Always keep the latest fetchFn reference
  useEffect(() => {
    fetchFnRef.current = fetchFn
  }, [fetchFn])

  useEffect(() => {
    if (!enabled) return

    let timeoutId: ReturnType<typeof setTimeout> | undefined

    const poll = async () => {
      setIsLoading(true)
      try {
        const result = await fetchFnRef.current()
        if (result) {
          setData(result)
        }
      } finally {
        setIsLoading(false)
        timeoutId = setTimeout(poll, interval)
      }
    }

    poll()
    return () => clearTimeout(timeoutId)
  }, [interval, enabled])

  return { data, isLoading }
}
