import { useEffect, useState } from 'react'
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

  useEffect(() => {
    if (!enabled) return

    let timeoutId: number | undefined

    const poll = async () => {
      setIsLoading(true)
      try {
        const result = await fetchFn()
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
  }, [fetchFn, interval, enabled])

  return { data, isLoading }
}
