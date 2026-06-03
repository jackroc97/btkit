import { useState } from 'react'
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from '@tanstack/react-table'

interface Props<T extends object> {
  data:               T[]
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  columns:            ColumnDef<T, any>[]
  pageSize?:          number
  filterPlaceholder?: string
  onRowClick?:        (row: T) => void
}

export default function DataTable<T extends object>({
  data, columns, pageSize = 15, filterPlaceholder = 'Filter…', onRowClick,
}: Props<T>) {
  const [sorting, setSorting]           = useState<SortingState>([])
  const [globalFilter, setGlobalFilter] = useState('')

  const table = useReactTable({
    data,
    columns,
    state: { sorting, globalFilter },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: { pagination: { pageSize } },
    globalFilterFn: 'includesString',
  })

  const filteredCount = table.getFilteredRowModel().rows.length
  const pageIndex     = table.getState().pagination.pageIndex

  return (
    <div>
      {/* Filter bar */}
      <div className="d-flex align-items-center mb-2" style={{ gap: 10 }}>
        <span className="btk-summary">
          {filteredCount} row{filteredCount !== 1 ? 's' : ''}
          {globalFilter ? ` for "${globalFilter}"` : ''}
        </span>
        <div className="ms-auto" style={{ position: 'relative' }}>
          <span style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--btk-muted-dk)', pointerEvents: 'none' }}>
            <i className="bi bi-search" style={{ fontSize: '0.75rem' }} />
          </span>
          <input
            type="text"
            className="form-control btk-search"
            style={{ paddingLeft: 28, width: 220, borderRadius: 8 }}
            placeholder={filterPlaceholder}
            value={globalFilter}
            onChange={e => setGlobalFilter(e.target.value)}
          />
          {globalFilter && (
            <button
              onClick={() => setGlobalFilter('')}
              style={{ position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', color: 'var(--btk-muted-dk)', cursor: 'pointer', padding: 0, lineHeight: 1 }}
            >
              <i className="bi bi-x" />
            </button>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="btk-table-wrap">
        <table className="btk-table">
          <thead>
            {table.getHeaderGroups().map(hg => (
              <tr key={hg.id}>
                {hg.headers.map(h => (
                  <th
                    key={h.id}
                    onClick={h.column.getToggleSortingHandler()}
                    style={{ cursor: h.column.getCanSort() ? 'pointer' : 'default', whiteSpace: 'nowrap' }}
                  >
                    {flexRender(h.column.columnDef.header, h.getContext())}
                    {h.column.getCanSort() && (
                      <span className={`btk-sort-arrow ${h.column.getIsSorted() ? 'active' : ''}`}>
                        {h.column.getIsSorted() === 'asc' ? ' ↑' : h.column.getIsSorted() === 'desc' ? ' ↓' : ' ↕'}
                      </span>
                    )}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.length === 0 ? (
              <tr>
                <td colSpan={columns.length} style={{ textAlign: 'center', color: 'var(--btk-muted-dk)', padding: '32px 0' }}>
                  No rows{globalFilter ? ` matching "${globalFilter}"` : '.'}
                </td>
              </tr>
            ) : (
              table.getRowModel().rows.map(row => (
                <tr
                  key={row.id}
                  style={{ cursor: onRowClick ? 'pointer' : 'default' }}
                  onClick={() => onRowClick?.(row.original)}
                >
                  {row.getVisibleCells().map(cell => (
                    <td key={cell.id}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {table.getPageCount() > 1 && (
        <div className="d-flex align-items-center justify-content-between mt-2 px-1">
          <span className="btk-summary">
            {pageIndex * pageSize + 1}–{Math.min((pageIndex + 1) * pageSize, filteredCount)} of {filteredCount}
          </span>
          <div className="d-flex gap-1">
            <button
              className="btn btn-sm"
              style={{ background: 'var(--btk-surface-2)', border: '1px solid var(--btk-border-dk)', color: '#e2e8f0', padding: '3px 10px', fontSize: '0.8rem' }}
              disabled={!table.getCanPreviousPage()}
              onClick={() => table.previousPage()}
            >‹ Prev</button>
            <span style={{ padding: '3px 10px', fontSize: '0.8rem', color: 'var(--btk-muted-dk)' }}>
              {pageIndex + 1} / {table.getPageCount()}
            </span>
            <button
              className="btn btn-sm"
              style={{ background: 'var(--btk-surface-2)', border: '1px solid var(--btk-border-dk)', color: '#e2e8f0', padding: '3px 10px', fontSize: '0.8rem' }}
              disabled={!table.getCanNextPage()}
              onClick={() => table.nextPage()}
            >Next ›</button>
          </div>
        </div>
      )}
    </div>
  )
}
