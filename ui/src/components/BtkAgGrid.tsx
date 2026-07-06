import { useState, useRef, useCallback } from 'react'
import { AgGridReact } from 'ag-grid-react'
import {
  AllCommunityModule,
  ModuleRegistry,
  themeQuartz,
  type ColDef,
  type RowClickedEvent,
  type RowStyle,
  type FilterChangedEvent,
} from 'ag-grid-community'

ModuleRegistry.registerModules([AllCommunityModule])

const BTK_THEME = themeQuartz.withParams({
  backgroundColor:       '#161b27',
  foregroundColor:       '#cbd5e1',
  headerBackgroundColor: '#1e2535',
  headerTextColor:       '#64748b',
  rowHoverColor:         '#1e2535',
  borderColor:           '#2a3245',
  oddRowBackgroundColor: '#161b27',
  fontFamily:            'Inter, system-ui, sans-serif',
  fontSize:              13,
  wrapperBorderRadius:   '10px',
  spacing:               6,
})

interface Props<T> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  rowData:      T[]
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  columnDefs:   ColDef<T, any>[]
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  defaultColDef?: Partial<ColDef<T, any>>
  onRowClicked?: (e: RowClickedEvent<T>) => void
  rowStyle?:    RowStyle
  pageSize?:    number
  filterPlaceholder?: string
  initialSortColId?: string
  initialSortDir?:   'asc' | 'desc'
  exportFileName?:   string
  prefKey?:          string
}

export default function BtkAgGrid<T>({
  rowData, columnDefs, defaultColDef, onRowClicked, rowStyle,
  pageSize = 20, filterPlaceholder = 'Filter…',
  initialSortColId, initialSortDir = 'desc',
  exportFileName = 'export.csv',
  prefKey,
}: Props<T>) {
  const [filter, setFilter]               = useState('')
  const [hasColumnFilter, setHasColumnFilter] = useState(false)
  const gridRef    = useRef<AgGridReact<T>>(null)
  const filterRef  = useRef(filter)
  const saveTimer  = useRef<ReturnType<typeof setTimeout> | null>(null)

  const scheduleSave = useCallback(() => {
    if (!prefKey) return
    if (saveTimer.current) clearTimeout(saveTimer.current)
    saveTimer.current = setTimeout(() => {
      const api = gridRef.current?.api
      if (!api) return
      const prefs = {
        filterModel: api.getFilterModel(),
        sortModel: api.getColumnState()
          .filter(cs => cs.sort != null)
          .map(cs => ({ colId: cs.colId, sort: cs.sort, sortIndex: cs.sortIndex })),
        quickFilter: filterRef.current,
      }
      fetch(`/api/preferences/${prefKey}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ value: prefs }),
      }).catch(() => {})
    }, 500)
  }, [prefKey])

  const handleFirstDataRendered = useCallback(() => {
    if (!prefKey) return
    fetch(`/api/preferences/${prefKey}`)
      .then(r => r.json())
      .then(({ value }: { value: { filterModel?: object; sortModel?: { colId: string; sort: string; sortIndex?: number }[]; quickFilter?: string } | null }) => {
        if (!value) return
        const api = gridRef.current?.api
        if (!api) return
        if (value.filterModel) {
          api.setFilterModel(value.filterModel)
          setHasColumnFilter(Object.keys(value.filterModel).length > 0)
        }
        if (value.sortModel?.length) {
          api.applyColumnState({
            state: value.sortModel.map(s => ({
              colId: s.colId, sort: s.sort as 'asc' | 'desc', sortIndex: s.sortIndex ?? null,
            })),
            defaultState: { sort: null },
          })
        }
        if (value.quickFilter) {
          filterRef.current = value.quickFilter
          setFilter(value.quickFilter)
        }
      })
      .catch(() => {})
  }, [prefKey])

  const handleFilterChanged = useCallback((e: FilterChangedEvent<T>) => {
    const model = e.api.getFilterModel()
    setHasColumnFilter(model != null && Object.keys(model).length > 0)
    scheduleSave()
  }, [scheduleSave])

  const handleSortChanged = useCallback(() => {
    scheduleSave()
  }, [scheduleSave])

  const clearColumnFilters = useCallback(() => {
    gridRef.current?.api.setFilterModel(null)
  }, [])

  const exportCsv = useCallback(() => {
    gridRef.current?.api.exportDataAsCsv({ fileName: exportFileName })
  }, [exportFileName])

  return (
    <div>
      {/* Toolbar */}
      <div className="d-flex align-items-center mb-2" style={{ gap: 10 }}>
        <span className="btk-summary">
          {rowData.length} row{rowData.length !== 1 ? 's' : ''}
        </span>

        {hasColumnFilter && (
          <button
            onClick={clearColumnFilters}
            style={{
              background: 'rgba(248,113,113,0.1)',
              border: '1px solid rgba(248,113,113,0.3)',
              borderRadius: 6, color: '#f87171',
              cursor: 'pointer', fontSize: '0.72rem',
              fontWeight: 600, padding: '2px 9px', lineHeight: 1.6,
              transition: 'all 0.15s',
            }}
          >
            <i className="bi bi-x me-1" />Clear filters
          </button>
        )}

        <div className="ms-auto d-flex align-items-center" style={{ gap: 8 }}>
          {/* CSV export */}
          <button
            onClick={exportCsv}
            title="Export to CSV"
            style={{
              background: 'transparent',
              border: '1px solid #2a3245',
              borderRadius: 6, color: '#64748b',
              cursor: 'pointer', padding: '4px 9px', lineHeight: 1,
              transition: 'color 0.15s, border-color 0.15s',
            }}
            onMouseEnter={e => {
              const el = e.currentTarget as HTMLElement
              el.style.color = '#94a3b8'; el.style.borderColor = '#475569'
            }}
            onMouseLeave={e => {
              const el = e.currentTarget as HTMLElement
              el.style.color = '#64748b'; el.style.borderColor = '#2a3245'
            }}
          >
            <i className="bi bi-download" style={{ fontSize: '0.8rem' }} />
          </button>

          {/* Quick filter */}
          <div style={{ position: 'relative' }}>
            <span style={{
              position: 'absolute', left: 10, top: '50%',
              transform: 'translateY(-50%)',
              color: 'var(--btk-muted-dk)', pointerEvents: 'none',
            }}>
              <i className="bi bi-search" style={{ fontSize: '0.75rem' }} />
            </span>
            <input
              type="text"
              className="form-control btk-search"
              style={{ paddingLeft: 28, width: 220, borderRadius: 8 }}
              placeholder={filterPlaceholder}
              value={filter}
              onChange={e => {
                const v = e.target.value
                filterRef.current = v
                setFilter(v)
                scheduleSave()
              }}
            />
            {filter && (
              <button
                onClick={() => {
                  filterRef.current = ''
                  setFilter('')
                  scheduleSave()
                }}
                style={{
                  position: 'absolute', right: 8, top: '50%',
                  transform: 'translateY(-50%)',
                  background: 'none', border: 'none',
                  color: 'var(--btk-muted-dk)', cursor: 'pointer', padding: 0, lineHeight: 1,
                }}
              >
                <i className="bi bi-x" />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Grid */}
      <AgGridReact
        ref={gridRef}
        theme={BTK_THEME}
        rowData={rowData}
        columnDefs={columnDefs}
        defaultColDef={{
          sortable: true,
          filter: true,
          resizable: true,
          ...defaultColDef,
        }}
        quickFilterText={filter}
        onRowClicked={onRowClicked}
        onFilterChanged={handleFilterChanged}
        onSortChanged={handleSortChanged}
        onFirstDataRendered={handleFirstDataRendered}
        rowStyle={rowStyle}
        pagination={true}
        paginationPageSize={pageSize}
        paginationPageSizeSelector={[10, 25, 50, 100]}
        domLayout="autoHeight"
        rowHeight={36}
        headerHeight={40}
        suppressCellFocus={true}
        animateRows={true}
        initialState={initialSortColId ? {
          sort: { sortModel: [{ colId: initialSortColId, sort: initialSortDir }] },
        } : undefined}
      />
    </div>
  )
}
