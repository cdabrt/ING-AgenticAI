"use client"

import * as React from "react"
import { RequirementItem } from "@/lib/types"

interface RequirementListProps {
    data: RequirementItem[]
    onRowClick?: (requirement: RequirementItem) => void
}

export default function RequirementList({ data, onRowClick }: RequirementListProps) {
    const [selectedRowId, setSelectedRowId] = React.useState<string | null>(
        data.length > 0 ? data[0].id : null
    )

    React.useEffect(() => {
        if (data.length > 0) {
            setSelectedRowId(data[0].id)
        } else {
            setSelectedRowId(null)
        }
    }, [data])

    return (
        <div className="space-y-3">
            {data.length === 0 && (
                <div className="rounded-xl border border-dashed border-zinc-200 p-6 text-center text-sm text-zinc-500">
                    No requirements found.
                </div>
            )}
            {data.map((item) => {
                const isSelected = selectedRowId === item.id
                return (
                    <button
                        key={item.id}
                        type="button"
                        onClick={() => {
                            setSelectedRowId(item.id)
                            onRowClick?.(item)
                        }}
                        className={`w-full rounded-xl border px-4 py-3 text-left transition ${
                            isSelected ? "border-zinc-900 bg-zinc-50" : "border-zinc-200 bg-white hover:border-zinc-400"
                        }`}
                    >
                        <div className="flex items-start justify-between gap-3">
                            <span className="rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] font-semibold text-zinc-600">
                                {item.id}
                            </span>
                            <span className="text-[11px] text-zinc-400">{item.document_sources?.length ?? 0} sources</span>
                        </div>
                        <p className="mt-2 text-sm text-zinc-800 line-clamp-3">{item.description}</p>
                    </button>
                )
            })}
        </div>
    )
}
