import { useState } from 'react'
import { AgGridReact } from 'ag-grid-react'
import {
  AllCommunityModule,
  ModuleRegistry,
  themeQuartz,
  type ColDef,
  type RowClickedEvent,
  type RowStyle,
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
}

export default function BtkAgGrid<T>({
  rowData, columnDefs, defaultColDef, onRowClicked, rowStyle,
  pageSize = 20, filterPlaceholder = 'Filter…',
  initialSortColId, initialSortDir = 'desc',
}: Props<T>) {
  const [filter, setFilter] = useState('')

  return (
    <div>
      {/* Filter bar */}
      <div className="d-flex align-items-center mb-2" style={{ gap: 10 }}>
        <span className="btk-summary">
          {rowData.length} row{rowData.length !== 1 ? 's' : ''}
        </span>
        <div className="ms-auto" style={{ position: 'relative' }}>
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
            onChange={e => setFilter(e.target.value)}
          />
          {filter && (
            <button
              onClick={() => setFilter('')}
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

      {/* Grid */}
      <AgGridReact
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
