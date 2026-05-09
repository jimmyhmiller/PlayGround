import { useEffect, useRef, useState } from 'react'
import { controlApi, SearchResult } from '@/lib/control-api'

export function useTaskSearch() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current)

    if (query.length < 2) {
      setResults([])
      return
    }

    setLoading(true)
    timeoutRef.current = setTimeout(() => {
      controlApi.searchTasks(query, 20)
        .then(r => {
          setResults(r)
          setLoading(false)
        })
        .catch(() => {
          setResults([])
          setLoading(false)
        })
    }, 300)

    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
    }
  }, [query])

  return { query, setQuery, results, loading }
}
