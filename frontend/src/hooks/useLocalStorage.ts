import { useState } from 'react'

export function useLocalStorage<T> (key: string, initialValue: T) {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = localStorage.getItem(key)
      return item ? (JSON.parse(item) as T) : initialValue
    } catch {
      return initialValue
    }
  })

  const setValue = (value: T) => {
    setStoredValue(value)
    try {
      localStorage.setItem(key, JSON.stringify(value))
    } catch (e) {
      if (e instanceof DOMException && e.name === 'QuotaExceededError') {
        console.warn(`localStorage quota exceeded for key "${key}"`)
      } else {
        throw e
      }
    }
  }

  return [storedValue, setValue] as const
}
