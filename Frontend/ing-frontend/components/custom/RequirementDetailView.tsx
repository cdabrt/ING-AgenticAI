import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import { FileText, Search, Sparkles } from "lucide-react";
import { Card, CardContent } from "../ui/card";
import { Separator } from "../ui/separator";
import { RequirementItem } from "@/lib/types";
import { ScrollArea } from "../ui/scroll-area";

interface HighlightResult {
    chunk_id?: string;
    source?: string | null;
    page?: number | null;
    highlights?: string[];
    score?: number;
}

function extractChunkId(source: string): string | null {
    const match = source.match(/chunk\s+([^,]+)/i);
    if (!match) {
        return null;
    }
    return match[1].trim().replace(/[.)]+$/, "");
}

function renderHighlightedText(text: string) {
    const tokens = text.split(/(\[\[H\]\]|\[\[\/H\]\])/);
    let highlighting = false;
    return tokens.map((token, index) => {
        if (token === "[[H]]") {
            highlighting = true;
            return null;
        }
        if (token === "[[/H]]") {
            highlighting = false;
            return null;
        }
        if (!token) {
            return null;
        }
        if (highlighting) {
            return (
                <mark key={`h-${index}`} className="rounded bg-amber-100 px-0.5 text-amber-950">
                    {token}
                </mark>
            );
        }
        return <span key={`t-${index}`}>{token}</span>;
    });
}

function RequirementDetailView({ requirement }: { requirement: RequirementItem | null }) {
    const [highlights, setHighlights] = useState<HighlightResult[]>([]);
    const [highlightsLoading, setHighlightsLoading] = useState(false);
    const [highlightsError, setHighlightsError] = useState<string | null>(null);

    const chunkIds = useMemo(() => {
        if (!requirement) {
            return [];
        }
        const ids = requirement.document_sources
            .map((source) => extractChunkId(source))
            .filter((value): value is string => Boolean(value));
        return Array.from(new Set(ids));
    }, [requirement]);

    const chunkIdsKey = useMemo(() => chunkIds.join("|"), [chunkIds]);

    useEffect(() => {
        if (!requirement) {
            setHighlights([]);
            setHighlightsError(null);
            return;
        }
        if (!requirement.description) {
            setHighlights([]);
            setHighlightsError(null);
            return;
        }
        setHighlightsLoading(true);
        setHighlightsError(null);
        const payload = {
            query: requirement.description,
            chunk_ids: chunkIds.length ? chunkIds : undefined,
            limit: 6,
        };
        axios.post("/api/highlights", payload)
            .then((response) => {
                const items = Array.isArray(response.data?.results) ? response.data.results : [];
                setHighlights(items);
            })
            .catch((error) => {
                const detail = error?.response?.data?.detail;
                setHighlightsError(detail || error.message);
                setHighlights([]);
            })
            .finally(() => {
                setHighlightsLoading(false);
            });
    }, [requirement?.id, requirement?.description, chunkIdsKey]);

    if (!requirement) {
        return (
            <div className="h-full w-full flex items-center justify-center p-8">
                <p className="text-muted-foreground text-center">
                    Select a requirement to view details
                </p>
            </div>
        );
    }

    return (
        <ScrollArea className="h-full w-full overflow-y-auto overflow-x-hidden min-w-0">
            <div className="p-6 space-y-6">
                {/* Header with ID */}
                <div>
                    <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-1 rounded">
                        {requirement.id}
                    </span>
                </div>

                {/* Description Section */}
                <div className="space-y-2">
                    <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                        Description
                    </h3>
                    <p className="text-base leading-relaxed break-words">
                        {requirement.description}
                    </p>
                </div>

                <Separator />

                {/* Rationale Section */}
                <div className="space-y-2">
                    <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                        Rationale
                    </h3>
                    <p className="text-base leading-relaxed text-muted-foreground break-words">
                        {requirement.rationale}
                    </p>
                </div>

                <Separator />

                {/* Document Sources Section */}
                {requirement.document_sources.length > 0 && (
                    <div className="space-y-3">
                        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                            Document Sources
                        </h3>
                        <div className="space-y-2">
                            {requirement.document_sources.map((source: string, index: number) => (
                                <Card key={index} className="border-l-4 border-l-primary">
                                    <CardContent className="px-4">
                                        <div className="flex items-start gap-2">
                                            <FileText className="h-4 w-4 mt-0.5 text-primary flex-shrink-0" />
                                            <span className="text-sm break-words">{source}</span>
                                        </div>
                                    </CardContent>
                                </Card>
                            ))}
                        </div>
                    </div>
                )}

                <Separator />

                <div className="space-y-3">
                    <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                        Evidence Highlights
                    </h3>
                    {highlightsLoading && (
                        <p className="text-sm text-muted-foreground">Scanning matching text...</p>
                    )}
                    {highlightsError && (
                        <p className="text-sm text-rose-600">Highlights unavailable: {highlightsError}</p>
                    )}
                    {!highlightsLoading && !highlightsError && highlights.length === 0 && (
                        <p className="text-sm text-muted-foreground">No highlight matches yet.</p>
                    )}
                    <div className="space-y-2">
                        {highlights.map((item, index) => (
                            <Card key={`${item.chunk_id ?? "chunk"}-${index}`} className="border-l-4 border-l-amber-500">
                                <CardContent className="px-4 space-y-2">
                                    <div className="flex items-start gap-2 text-xs text-muted-foreground">
                                        <Sparkles className="h-4 w-4 text-amber-500 flex-shrink-0" />
                                        <span className="break-words">
                                            {item.source || "Unknown source"}
                                            {item.page ? ` · page ${item.page}` : ""}
                                            {item.chunk_id ? ` · chunk ${item.chunk_id}` : ""}
                                        </span>
                                    </div>
                                    <div className="space-y-2 text-sm leading-relaxed text-muted-foreground">
                                        {(item.highlights && item.highlights.length > 0 ? item.highlights : ["No highlight fragments returned."])
                                            .map((fragment, fragIndex) => (
                                                <p key={`frag-${index}-${fragIndex}`} className="break-words">
                                                    {renderHighlightedText(fragment)}
                                                </p>
                                            ))}
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                </div>

                {/* Online Sources Section */}
                {requirement.online_sources.length > 0 && (
                    <>
                        <Separator />
                        <div className="space-y-3">
                            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                                Online Sources
                            </h3>
                            <div className="space-y-2">
                                {requirement.online_sources.map((source: string, index: number) => (
                                    <Card key={index} className="border-l-4 border-l-blue-500">
                                        <CardContent className="px-4">
                                            <a
                                                href={source}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="flex items-start gap-2 text-blue-600 hover:text-blue-800 hover:underline"
                                            >
                                                <Search className="h-4 w-4 mt-0.5 flex-shrink-0" />
                                                <span className="text-sm break-all">{source}</span>
                                            </a>
                                        </CardContent>
                                    </Card>
                                ))}
                            </div>
                        </div>
                    </>
                )}
            </div>
        </ScrollArea>
    );
}

export default RequirementDetailView;
