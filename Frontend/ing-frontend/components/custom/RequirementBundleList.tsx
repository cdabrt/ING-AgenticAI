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
    setSelectedList: (index: number) => void
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

export default function RequirementBundleList({ data, onRowClick, setSelectedList }: RequirementBundleListProps) {
    const [sorting, setSorting] = React.useState<SortingState>([])
    const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>([])
    const [columnVisibility, setColumnVisibility] = React.useState<VisibilityState>({})
    const [selectedRowId, setSelectedRowId] = React.useState<number | null>(
        data.length > 0 ? data[0].id : null
    )
    const [expandedRowId, setExpandedRowId] = React.useState<number | null>(null)

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
                                <React.Fragment key={row.id}>
                                    <TableRow
                                        onClick={() => {
                                            setSelectedRowId(row.original.id)
                                            setExpandedRowId(
                                                expandedRowId === row.original.id ? null : row.original.id
                                            )
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
                                    <TableRow className="bg-zinc-50 border-t-0">
                                        <TableCell colSpan={columns.length} className="p-0">
                                            <div 
                                                className={`overflow-hidden transition-all duration-300 ease-in-out ${
                                                    expandedRowId === row.original.id 
                                                        ? "max-h-40 opacity-100" 
                                                        : "max-h-0 opacity-0"
                                                }`}
                                            >
                                                <div className="flex flex-col gap-2 p-3 pl-8">    
                                                    {
                                                        row.original.business_requirements?.length > 0 && 
                                                        <Button
                                                            variant="outline"
                                                            className="justify-start text-sm cursor-pointer"
                                                            onClick={() => setSelectedList(1)}
                                                        >
                                                            Business Requirements
                                                        </Button>
                                                    }
                                                    {
                                                    row.original.data_requirements?.length > 0 && 
                                                        <Button
                                                            variant="outline"
                                                            className="justify-start text-sm cursor-pointer"
                                                            onClick={() => setSelectedList(2)}
                                                        >
                                                            Data Requirements
                                                        </Button>
                                                    }
                                                </div>
                                            </div>
                                        </TableCell>
                                    </TableRow>
                                </React.Fragment>
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
