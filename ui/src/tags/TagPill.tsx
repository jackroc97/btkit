import type { Tag } from './TagsContext'

interface TagPillProps {
  tag: Tag
  onRemove?: () => void
  size?: 'sm' | 'md'
}

export function TagPill({ tag, onRemove, size = 'sm' }: TagPillProps) {
  const pad = size === 'sm' ? '1px 7px' : '3px 10px'
  const fontSize = size === 'sm' ? '0.7rem' : '0.8rem'

  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 4,
        background: tag.color,
        color: '#fff',
        borderRadius: 99,
        padding: pad,
        fontSize,
        fontWeight: 600,
        letterSpacing: '0.01em',
        lineHeight: 1.5,
        whiteSpace: 'nowrap',
      }}
    >
      {tag.name}
      {onRemove && (
        <button
          onClick={e => { e.stopPropagation(); onRemove() }}
          style={{
            background: 'none',
            border: 'none',
            color: 'rgba(255,255,255,0.8)',
            padding: 0,
            cursor: 'pointer',
            lineHeight: 1,
            fontSize: '0.75rem',
            display: 'flex',
            alignItems: 'center',
          }}
          aria-label={`Remove tag ${tag.name}`}
        >
          <i className="bi bi-x" />
        </button>
      )}
    </span>
  )
}

/** Renders up to maxVisible pills, then a +N overflow chip. */
export function TagPillList({
  tags,
  maxVisible = 3,
  onRemove,
  size = 'sm',
}: {
  tags: Tag[]
  maxVisible?: number
  onRemove?: (tagId: number) => void
  size?: 'sm' | 'md'
}) {
  if (!tags.length) return null
  const visible = tags.slice(0, maxVisible)
  const overflow = tags.length - maxVisible

  return (
    <span style={{ display: 'inline-flex', flexWrap: 'wrap', gap: 3, alignItems: 'center' }}>
      {visible.map(t => (
        <TagPill key={t.id} tag={t} size={size} onRemove={onRemove ? () => onRemove(t.id) : undefined} />
      ))}
      {overflow > 0 && (
        <span
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            background: 'var(--btk-surface-2)',
            color: 'var(--btk-muted-dk)',
            border: '1px solid var(--btk-border-dk)',
            borderRadius: 99,
            padding: '1px 7px',
            fontSize: '0.7rem',
            fontWeight: 600,
            whiteSpace: 'nowrap',
          }}
        >
          +{overflow}
        </span>
      )}
    </span>
  )
}
