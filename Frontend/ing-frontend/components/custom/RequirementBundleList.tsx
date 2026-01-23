"use client"

import * as React from "react"
import { Badge } from "@/components/ui/badge"
import { RequirementBundle } from "@/lib/types"

interface RequirementBundleListProps {
    data: RequirementBundle[]
    onRowClick?: (bundle: RequirementBundle) => void
}

export default function RequirementBundleList({ data, onRowClick }: RequirementBundleListProps) {
    const [selectedRowId, setSelectedRowId] = React.useState<number | null>(
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
                    No bundles found yet.
                </div>
            )}
            {data.map((bundle) => {
                const isSelected = selectedRowId === bundle.id
                return (
                    <button
                        key={bundle.id}
                        type="button"
                        onClick={() => {
                            setSelectedRowId(bundle.id)
                            onRowClick?.(bundle)
                        }}
                        className={`w-full rounded-xl border px-4 py-3 text-left transition ${
                            isSelected ? "border-zinc-900 bg-zinc-50" : "border-zinc-200 bg-white hover:border-zinc-400"
                        }`}
                    >
                        <div className="flex items-center justify-between gap-3">
                            <div className="min-w-0">
                                <p className="text-sm font-semibold text-zinc-900 truncate">{bundle.document}</p>
                                <p className="text-xs text-zinc-500 line-clamp-2">{bundle.document_type}</p>
                            </div>
                            <Badge variant="secondary" className="text-[11px]">
                                {bundle.business_requirements.length + bundle.data_requirements.length}
                            </Badge>
                        </div>
                        <div className="mt-3 flex flex-wrap items-center gap-3 text-[11px] text-zinc-500">
                            <span>{bundle.business_requirements.length} business</span>
                            <span>{bundle.data_requirements.length} data</span>
                            <span>{bundle.assumptions.length} assumptions</span>
                            {bundle.run_completed_at && (
                                <span className="text-[10px] text-zinc-400">
                                    {new Date(bundle.run_completed_at).toLocaleString()}
                                </span>
                            )}
                        </div>
                    </button>
                )
            })}
        </div>
    )
}
