"use client"

import * as React from "react"
import {
    ColumnDef,
    ColumnFiltersState,
    flexRender,
    getCoreRowModel,
    getFilteredRowModel,
    getPaginationRowModel,
    getSortedRowModel,
    SortingState,
    useReactTable,
    VisibilityState,
} from "@tanstack/react-table"
import { ArrowUpDown } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { RequirementBundle } from "@/lib/types"

interface RequirementBundleListProps {
    data: RequirementBundle[]
    onRowClick?: (bundle: RequirementBundle) => void
}

export const columns: ColumnDef<RequirementBundle>[] = [
    {
        accessorKey: "id",
        header: "Id",
        cell: ({ row }) => <div className="font-medium">{row.getValue("id")}</div>,
    },
    {
        accessorKey: "document",
        header: ({ column }) => {
            return (
                <Button
                    variant="ghost"
                    onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
                >
                    Document
                    <ArrowUpDown />
                </Button>
            )
        },
        cell: ({ row }) => <div className="font-medium">{row.getValue("document")}</div>,
    },
    {
        accessorKey: "document_type",
        header: "Type",
        cell: ({ row }) => (
            <Badge variant="secondary">
                {row.getValue("document_type")}
            </Badge>
        ),
    },
    {
        accessorKey: "business_requirements",
        header: "Business Requirements",
        cell: ({ row }) => {
            const requirements = row.getValue("business_requirements") as any[]
            return <div className="text-center">{requirements.length}</div>
        },
    },
    {
        accessorKey: "data_requirements",
        header: "Data Requirements",
        cell: ({ row }) => {
            const requirements = row.getValue("data_requirements") as any[]
            return <div className="text-center">{requirements.length}</div>
        },
    },
    {
        accessorKey: "assumptions",
        header: "Assumptions",
        cell: ({ row }) => {
            const assumptions = row.getValue("assumptions") as string[]
            return <div className="text-center">{assumptions.length}</div>
        },
    },
]

export default function RequirementBundleList({ data, onRowClick }: RequirementBundleListProps) {
    const [sorting, setSorting] = React.useState<SortingState>([])
    const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>([])
    const [columnVisibility, setColumnVisibility] = React.useState<VisibilityState>({})
    const [selectedRowId, setSelectedRowId] = React.useState<number | null>(
        data.length > 0 ? data[0].id : null
    )

    const table = useReactTable({
        data,
        columns,
        onSortingChange: setSorting,
        onColumnFiltersChange: setColumnFilters,
        getCoreRowModel: getCoreRowModel(),
        getPaginationRowModel: getPaginationRowModel(),
        getSortedRowModel: getSortedRowModel(),
        getFilteredRowModel: getFilteredRowModel(),
        onColumnVisibilityChange: setColumnVisibility,
        state: {
            sorting,
            columnFilters,
            columnVisibility,
        },
    })

    return (
        <div className="w-full space-y-4">
            <div className="overflow-hidden rounded-md border">
                <Table>
                    <TableHeader>
                        {table.getHeaderGroups().map((headerGroup) => (
                            <TableRow key={headerGroup.id}>
                                {headerGroup.headers.map((header) => {
                                    return (
                                        <TableHead key={header.id}>
                                            {header.isPlaceholder
                                                ? null
                                                : flexRender(
                                                    header.column.columnDef.header,
                                                    header.getContext()
                                                )}
                                        </TableHead>
                                    )
                                })}
                            </TableRow>
                        ))}
                    </TableHeader>
                    <TableBody>
                        {table.getRowModel().rows?.length ? (
                            table.getRowModel().rows.map((row) => (
                                <TableRow
                                    key={row.id}
                                    onClick={() => {
                                        setSelectedRowId(row.original.id)
                                        onRowClick?.(row.original)
                                    }}
                                    className={`cursor-pointer ${selectedRowId === row.original.id
                                        ? "bg-zinc-200 hover:bg-zinc-300"
                                        : ""
                                        }`}
                                >
                                    {row.getVisibleCells().map((cell) => (
                                        <TableCell key={cell.id}>
                                            {flexRender(
                                                cell.column.columnDef.cell,
                                                cell.getContext()
                                            )}
                                        </TableCell>
                                    ))}
                                </TableRow>
                            ))
                        ) : (
                            <TableRow>
                                <TableCell
                                    colSpan={columns.length}
                                    className="h-24 text-center"
                                >
                                    No bundles found.
                                </TableCell>
                            </TableRow>
                        )}
                    </TableBody>
                </Table>
            </div>
        </div>
    )
}
