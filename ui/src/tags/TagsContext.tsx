/**
 * TagsContext — shared tag list and mutation helpers.
 *
 * Wrap the app (or any subtree) with <TagsProvider> to get:
 *   - useTags()        → { tags, loading }
 *   - useTagMutations() → { createTag, applyTag, removeTag, deleteTag }
 *
 * The tag list is fetched once on mount and updated optimistically on every
 * mutation so the picker reflects changes without a full page reload.
 */

import { createContext, useContext, useEffect, useState, useCallback, type ReactNode } from 'react'
import { api } from '../api/client'

export interface Tag {
  id: number
  name: string
  color: string
}

interface TagsState {
  tags: Tag[]
  loading: boolean
  refresh: () => void
}

interface TagMutations {
  createTag: (name: string, color: string) => Promise<Tag>
  applyTag: (backtestId: number, tagId: number) => Promise<void>
  removeTag: (backtestId: number, tagId: number) => Promise<void>
  deleteTag: (tagId: number) => Promise<void>
}

const TagsCtx = createContext<TagsState>({ tags: [], loading: false, refresh: () => {} })
const TagMutCtx = createContext<TagMutations>({} as TagMutations)

export function TagsProvider({ children }: { children: ReactNode }) {
  const [tags, setTags] = useState<Tag[]>([])
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(() => {
    setLoading(true)
    api.get<Tag[]>('/tags')
      .then(setTags)
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const createTag = useCallback(async (name: string, color: string): Promise<Tag> => {
    const created = await api.post<Tag>('/tags', { name, color })
    setTags(prev => [...prev, created].sort((a, b) => a.name.localeCompare(b.name)))
    return created
  }, [])

  const applyTag = useCallback(async (backtestId: number, tagId: number): Promise<void> => {
    await api.post(`/backtests/${backtestId}/tags/${tagId}`)
  }, [])

  const removeTag = useCallback(async (backtestId: number, tagId: number): Promise<void> => {
    await api.del(`/backtests/${backtestId}/tags/${tagId}`)
  }, [])

  const deleteTag = useCallback(async (tagId: number): Promise<void> => {
    await api.del(`/tags/${tagId}`)
    setTags(prev => prev.filter(t => t.id !== tagId))
  }, [])

  return (
    <TagsCtx.Provider value={{ tags, loading, refresh }}>
      <TagMutCtx.Provider value={{ createTag, applyTag, removeTag, deleteTag }}>
        {children}
      </TagMutCtx.Provider>
    </TagsCtx.Provider>
  )
}

export function useTags() {
  return useContext(TagsCtx)
}

export function useTagMutations() {
  return useContext(TagMutCtx)
}
