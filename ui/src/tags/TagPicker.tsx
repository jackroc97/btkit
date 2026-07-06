/**
 * TagPicker — popover for adding/removing/creating tags on a backtest.
 *
 * Usage:
 *   <TagPicker
 *     backtestId={42}
 *     currentTags={[...]}
 *     onChanged={() => refetch()}
 *   />
 */

import { useEffect, useRef, useState } from 'react'
import { useTags, useTagMutations, type Tag } from './TagsContext'
import { TagPill } from './TagPill'

const PALETTE = [
  '#3b82f6', // blue
  '#6366f1', // indigo
  '#8b5cf6', // violet
  '#ec4899', // pink
  '#f43f5e', // rose
  '#f97316', // orange
  '#f59e0b', // amber
  '#eab308', // yellow
  '#84cc16', // lime
  '#22c55e', // green
  '#14b8a6', // teal
  '#06b6d4', // cyan
  '#0ea5e9', // sky
  '#64748b', // slate
  '#737373', // neutral
  '#78716c', // stone
]

interface Props {
  backtestId: number
  currentTags: Tag[]
  onChanged: (next: Tag[]) => void
}

export function TagPicker({ backtestId, currentTags, onChanged }: Props) {
  const [open, setOpen] = useState(false)
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('')
  const [newColor, setNewColor] = useState(PALETTE[0])
  const [busy, setBusy] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  const { tags: allTags } = useTags()
  const { createTag, applyTag, removeTag } = useTagMutations()

  const currentIds = new Set(currentTags.map(t => t.id))

  // Close on outside click
  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
        setCreating(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  async function handleToggle(tag: Tag) {
    if (busy) return
    setBusy(true)
    try {
      if (currentIds.has(tag.id)) {
        await removeTag(backtestId, tag.id)
        onChanged(currentTags.filter(t => t.id !== tag.id))
      } else {
        await applyTag(backtestId, tag.id)
        onChanged([...currentTags, tag].sort((a, b) => a.name.localeCompare(b.name)))
      }
    } finally {
      setBusy(false)
    }
  }

  async function handleCreate() {
    const name = newName.trim()
    if (!name || busy) return
    setBusy(true)
    try {
      const tag = await createTag(name, newColor)
      await applyTag(backtestId, tag.id)
      onChanged([...currentTags, tag].sort((a, b) => a.name.localeCompare(b.name)))
      setNewName('')
      setNewColor(PALETTE[0])
      setCreating(false)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div ref={ref} style={{ position: 'relative', display: 'inline-block' }}>
      <button
        onClick={e => { e.stopPropagation(); setOpen(v => !v); setCreating(false) }}
        title="Add / remove tags"
        style={{
          background: 'var(--btk-surface-2)',
          border: '1px solid var(--btk-border-dk)',
          borderRadius: 6,
          color: 'var(--btk-muted-dk)',
          cursor: 'pointer',
          padding: '1px 6px',
          fontSize: '0.75rem',
          lineHeight: 1.6,
          display: 'inline-flex',
          alignItems: 'center',
          gap: 3,
        }}
      >
        <i className="bi bi-tag" />
      </button>

      {open && (
        <div
          onClick={e => e.stopPropagation()}
          style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            zIndex: 1050,
            minWidth: 200,
            background: 'var(--btk-surface)',
            border: '1px solid var(--btk-border-dk)',
            borderRadius: 8,
            boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
            padding: '8px 0',
            marginTop: 4,
          }}
        >
          {!creating ? (
            <>
              {allTags.length === 0 && (
                <div style={{ padding: '4px 12px', color: 'var(--btk-muted-dk)', fontSize: '0.8rem' }}>
                  No tags yet
                </div>
              )}
              {allTags.map(tag => {
                const active = currentIds.has(tag.id)
                return (
                  <button
                    key={tag.id}
                    onClick={() => handleToggle(tag)}
                    disabled={busy}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      width: '100%',
                      background: active ? 'var(--btk-surface-2)' : 'none',
                      border: 'none',
                      padding: '5px 12px',
                      cursor: 'pointer',
                      textAlign: 'left',
                    }}
                  >
                    <span
                      style={{
                        width: 10,
                        height: 10,
                        borderRadius: '50%',
                        background: tag.color,
                        flexShrink: 0,
                      }}
                    />
                    <span style={{ color: 'var(--bs-body-color)', fontSize: '0.85rem', flexGrow: 1 }}>
                      {tag.name}
                    </span>
                    {active && (
                      <i className="bi bi-check2" style={{ color: 'var(--btk-muted-dk)', fontSize: '0.85rem' }} />
                    )}
                  </button>
                )
              })}
              <div style={{ borderTop: '1px solid var(--btk-border-dk)', margin: '6px 0' }} />
              <button
                onClick={() => setCreating(true)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  width: '100%',
                  background: 'none',
                  border: 'none',
                  padding: '5px 12px',
                  cursor: 'pointer',
                  color: 'var(--btk-muted-dk)',
                  fontSize: '0.85rem',
                }}
              >
                <i className="bi bi-plus-circle" />
                New tag…
              </button>
            </>
          ) : (
            <div style={{ padding: '8px 12px' }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--btk-muted-dk)', marginBottom: 6 }}>
                New tag
              </div>
              <input
                autoFocus
                value={newName}
                onChange={e => setNewName(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') handleCreate() }}
                placeholder="Tag name"
                style={{
                  width: '100%',
                  background: 'var(--btk-surface-2)',
                  border: '1px solid var(--btk-border-dk)',
                  borderRadius: 5,
                  color: 'var(--bs-body-color)',
                  padding: '4px 8px',
                  fontSize: '0.85rem',
                  marginBottom: 8,
                  outline: 'none',
                  boxSizing: 'border-box',
                }}
              />
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginBottom: 10 }}>
                {PALETTE.map(c => (
                  <button
                    key={c}
                    onClick={() => setNewColor(c)}
                    style={{
                      width: 18,
                      height: 18,
                      borderRadius: '50%',
                      background: c,
                      border: newColor === c ? '2px solid #fff' : '2px solid transparent',
                      cursor: 'pointer',
                      padding: 0,
                      outline: newColor === c ? `2px solid ${c}` : 'none',
                      outlineOffset: 1,
                    }}
                  />
                ))}
              </div>
              <div style={{ display: 'flex', gap: 6 }}>
                <button
                  onClick={handleCreate}
                  disabled={!newName.trim() || busy}
                  style={{
                    flex: 1,
                    background: newColor,
                    border: 'none',
                    borderRadius: 5,
                    color: '#fff',
                    fontWeight: 600,
                    fontSize: '0.8rem',
                    padding: '4px 0',
                    cursor: 'pointer',
                    opacity: !newName.trim() || busy ? 0.5 : 1,
                  }}
                >
                  {busy ? '…' : 'Create'}
                </button>
                <button
                  onClick={() => { setCreating(false); setNewName('') }}
                  style={{
                    background: 'var(--btk-surface-2)',
                    border: '1px solid var(--btk-border-dk)',
                    borderRadius: 5,
                    color: 'var(--btk-muted-dk)',
                    fontSize: '0.8rem',
                    padding: '4px 10px',
                    cursor: 'pointer',
                  }}
                >
                  Cancel
                </button>
              </div>
              {newName.trim() && (
                <div style={{ marginTop: 8 }}>
                  <TagPill tag={{ id: 0, name: newName.trim(), color: newColor }} size="sm" />
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
