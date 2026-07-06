/**
 * TagFilterBar — horizontal row of tag chips for filtering lists.
 *
 * Usage:
 *   const [activeTags, setActiveTags] = useState<Set<number>>(new Set())
 *   <TagFilterBar tags={allTagsInView} active={activeTags} onChange={setActiveTags} />
 *
 * Returns null when there are no tags to show.
 */

import type { Tag } from './TagsContext'

interface Props {
  tags: Tag[]
  active: Set<number>
  onChange: (next: Set<number>) => void
}

export function TagFilterBar({ tags, active, onChange }: Props) {
  if (!tags.length) return null

  function toggle(id: number) {
    const next = new Set(active)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    onChange(next)
  }

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
      <span style={{ fontSize: '0.75rem', color: 'var(--btk-muted-dk)', marginRight: 2 }}>
        Filter:
      </span>
      {tags.map(tag => {
        const on = active.has(tag.id)
        return (
          <button
            key={tag.id}
            onClick={() => toggle(tag.id)}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 5,
              background: on ? tag.color : 'var(--btk-surface-2)',
              border: `1px solid ${on ? tag.color : 'var(--btk-border-dk)'}`,
              borderRadius: 99,
              color: on ? '#fff' : 'var(--btk-muted-dk)',
              cursor: 'pointer',
              fontSize: '0.75rem',
              fontWeight: 600,
              padding: '2px 10px',
              transition: 'background 0.15s, color 0.15s, border-color 0.15s',
            }}
          >
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                background: on ? '#fff' : tag.color,
                flexShrink: 0,
              }}
            />
            {tag.name}
          </button>
        )
      })}
      {active.size > 0 && (
        <button
          onClick={() => onChange(new Set())}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--btk-muted-dk)',
            cursor: 'pointer',
            fontSize: '0.75rem',
            padding: '2px 4px',
          }}
        >
          Clear
        </button>
      )}
    </div>
  )
}
