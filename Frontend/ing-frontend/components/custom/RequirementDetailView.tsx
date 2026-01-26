import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import { FileText, Search, Sparkles, Edit, Save, Plus, Trash2 } from "lucide-react";
import { Card, CardContent } from "../ui/card";
import { Separator } from "../ui/separator";
import { RequirementItem } from "@/lib/types";
import { ScrollArea } from "../ui/scroll-area";
import { Textarea } from "../ui/textarea";
import { Button } from "../ui/button";

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

function RequirementDetailView({ requirement, onRefresh, onDocumentSourceClick }: { requirement: RequirementItem | null; onRefresh?: () => void; onDocumentSourceClick?: (source: string) => void }) {
    const [localRequirement, setLocalRequirement] = useState<RequirementItem | null>(null);
    const [highlights, setHighlights] = useState<HighlightResult[]>([]);
    const [highlightsLoading, setHighlightsLoading] = useState(false);
    const [highlightsError, setHighlightsError] = useState<string | null>(null);
    
    const [isEditingDescription, setIsEditingDescription] = useState(false);
    const [isEditingRationale, setIsEditingRationale] = useState(false);
    const [editedDescription, setEditedDescription] = useState("");
    const [editedRationale, setEditedRationale] = useState("");
    const [isSaving, setIsSaving] = useState(false);
    
    const [newDocSource, setNewDocSource] = useState("");
    const [newOnlineSource, setNewOnlineSource] = useState("");
    const [isAddingDocSource, setIsAddingDocSource] = useState(false);
    const [isAddingOnlineSource, setIsAddingOnlineSource] = useState(false);

    const chunkIds = useMemo(() => {
        if (!localRequirement) {
            return [];
        }
        const ids = localRequirement.document_sources
            .map((source) => extractChunkId(source))
            .filter((value): value is string => Boolean(value));
        return Array.from(new Set(ids));
    }, [localRequirement]);

    const chunkIdsKey = useMemo(() => chunkIds.join("|"), [chunkIds]);

    useEffect(() => {
        setLocalRequirement(requirement);
    }, [requirement]);

    useEffect(() => {
        if (!localRequirement) {
            setHighlights([]);
            setHighlightsError(null);
            setIsEditingDescription(false);
            setIsEditingRationale(false);
            return;
        }
        setEditedDescription(localRequirement.description || "");
        setEditedRationale(localRequirement.rationale || "");
        setIsEditingDescription(false);
        setIsEditingRationale(false);
        if (!localRequirement.description) {
            setHighlights([]);
            setHighlightsError(null);
            return;
        }
        setHighlightsLoading(true);
        setHighlightsError(null);
        const payload = {
            query: localRequirement.description,
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
    }, [localRequirement?.id, localRequirement?.description, chunkIdsKey]);

    const handleSaveDescription = (): Promise<void> => {
        if (!localRequirement) return Promise.resolve();
        setIsSaving(true);
        return axios.put(`/api/requirements/${localRequirement.id}`, {
            ...localRequirement,
            description: editedDescription,
        }).then(() => {
            setLocalRequirement({ ...localRequirement, description: editedDescription });
            setIsEditingDescription(false);
            onRefresh?.();
        }).catch((error) => {
            const detail = error?.response?.data?.detail;
            console.error("Failed to save description:", error);
            alert(detail || "Failed to save description. Please try again.");
        }).finally(() => {
            setIsSaving(false);
        });
    };

    const handleSaveRationale = (): Promise<void> => {
        if (!localRequirement) return Promise.resolve();
        setIsSaving(true);
        return axios.put(`/api/requirements/${localRequirement.id}`, {
            ...localRequirement,
            rationale: editedRationale,
        }).then(() => {
            setLocalRequirement({ ...localRequirement, rationale: editedRationale });
            setIsEditingRationale(false);
            onRefresh?.();
        }).catch((error) => {
            const detail = error?.response?.data?.detail;
            console.error("Failed to save rationale:", error);
            alert(detail || "Failed to save rationale. Please try again.");
        }).finally(() => {
            setIsSaving(false);
        });
    };

    const handleAddDocSource = (): Promise<void> => {
        if (!localRequirement || !newDocSource.trim()) return Promise.resolve();
        setIsAddingDocSource(true);
        return axios.post(`/api/requirements/${localRequirement.id}/sources`, {
            source: newDocSource.trim(),
        }).then((response) => {
            setLocalRequirement({ ...localRequirement, document_sources: response.data.document_sources });
            setNewDocSource("");
            onRefresh?.();
        }).catch((error) => {
            const detail = error?.response?.data?.detail;
            console.error("Failed to add document source:", error);
            alert(detail || "Failed to add document source. Please try again.");
        }).finally(() => {
            setIsAddingDocSource(false);
        });
    };

    const handleRemoveDocSource = (index: number): Promise<void> => {
        if (!localRequirement) return Promise.resolve();
        const sourceToRemove = localRequirement.document_sources[index];
        return axios.delete(`/api/requirements/${localRequirement.id}/sources`, {
            data: { source: sourceToRemove }
        }).then(() => {
            const updatedSources = localRequirement.document_sources.filter((_, i) => i !== index);
            setLocalRequirement({ ...localRequirement, document_sources: updatedSources });
            onRefresh?.();
        }).catch((error) => {
            const detail = error?.response?.data?.detail;
            console.error("Failed to remove document source:", error);
            alert(detail || "Failed to remove document source. Please try again.");
        });
    };

    const handleAddOnlineSource = (): Promise<void> => {
        if (!localRequirement || !newOnlineSource.trim()) return Promise.resolve();
        setIsAddingOnlineSource(true);
        return axios.post(`/api/requirements/${localRequirement.id}/online-sources`, {
            source: newOnlineSource.trim(),
        }).then((response) => {
            setLocalRequirement({ ...localRequirement, online_sources: response.data.online_sources });
            setNewOnlineSource("");
            onRefresh?.();
        }).catch((error) => {
            const detail = error?.response?.data?.detail;
            console.error("Failed to add online source:", error);
            alert(detail || "Failed to add online source. Please try again.");
        }).finally(() => {
            setIsAddingOnlineSource(false);
        });
    };

    const handleRemoveOnlineSource = (index: number): Promise<void> => {
        if (!localRequirement) return Promise.resolve();
        const sourceToRemove = localRequirement.online_sources[index];
        return axios.delete(`/api/requirements/${localRequirement.id}/online-sources`, {
            data: { source: sourceToRemove }
        }).then(() => {
            const updatedSources = localRequirement.online_sources.filter((_, i) => i !== index);
            setLocalRequirement({ ...localRequirement, online_sources: updatedSources });
            onRefresh?.();
        }).catch((error) => {
            const detail = error?.response?.data?.detail;
            console.error("Failed to remove online source:", error);
            alert(detail || "Failed to remove online source. Please try again.");
        });
    };

    if (!localRequirement) {
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
                        {localRequirement.id}
                    </span>
                </div>

                {/* Description Section */}
                <div className="space-y-2">
                    <div className="flex items-center justify-between">
                        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                            Description
                        </h3>
                        {isEditingDescription ? (
                            <Button
                                size="sm"
                                variant="ghost"
                                onClick={handleSaveDescription}
                                disabled={isSaving}
                            >
                                <Save className="h-4 w-4" />
                            </Button>
                        ) : (
                            <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => setIsEditingDescription(true)}
                            >
                                <Edit className="h-4 w-4" />
                            </Button>
                        )}
                    </div>
                    {isEditingDescription ? (
                        <Textarea
                            value={editedDescription}
                            onChange={(e) => setEditedDescription(e.target.value)}
                            className="min-h-[100px] text-base resize-none w-full"
                            disabled={isSaving}
                        />
                    ) : (
                        <p className="text-base leading-relaxed break-words">
                            {localRequirement.description}
                        </p>
                    )}
                </div>

                <Separator />

                {/* Rationale Section */}
                <div className="space-y-2">
                    <div className="flex items-center justify-between">
                        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                            Rationale
                        </h3>
                        {isEditingRationale ? (
                            <Button
                                size="sm"
                                variant="ghost"
                                onClick={handleSaveRationale}
                                disabled={isSaving}
                            >
                                <Save className="h-4 w-4" />
                            </Button>
                        ) : (
                            <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => setIsEditingRationale(true)}
                            >
                                <Edit className="h-4 w-4" />
                            </Button>
                        )}
                    </div>
                    {isEditingRationale ? (
                        <Textarea
                            value={editedRationale}
                            onChange={(e) => setEditedRationale(e.target.value)}
                            className="min-h-[100px] text-base resize-none w-full"
                            disabled={isSaving}
                        />
                    ) : (
                        <p className="text-base leading-relaxed text-muted-foreground break-words">
                            {localRequirement.rationale}
                        </p>
                    )}
                </div>

                <Separator />

                {/* Document Sources Section */}
                <div className="space-y-3">
                    <div className="flex items-center justify-between">
                        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                            Document Sources
                        </h3>
                    </div>
                    <div className="space-y-2">
                        {localRequirement.document_sources.map((source: string, index: number) => (
                            <Card key={index} className="border-l-4 border-l-primary cursor-pointer" onClick={() => onDocumentSourceClick && onDocumentSourceClick(source)}>
                                <CardContent className="px-4">
                                    <div className="flex items-start gap-2 justify-between">
                                        <div className="flex items-start gap-2 flex-1 min-w-0">
                                            <FileText className="h-4 w-4 mt-0.5 text-primary flex-shrink-0" />
                                            <span className="text-sm break-words">{source}</span>
                                        </div>
                                        <Button
                                            size="sm"
                                            variant="ghost"
                                            onClick={() => handleRemoveDocSource(index)}
                                            className="h-6 w-6 p-0 flex-shrink-0"
                                        >
                                            <Trash2 className="h-3 w-3 text-destructive" />
                                        </Button>
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                        <div className="flex gap-2">
                            <Textarea
                                placeholder="Add new document source..."
                                value={newDocSource}
                                onChange={(e) => setNewDocSource(e.target.value)}
                                className="min-h-[60px] text-sm resize-none flex-1 min-w-0"
                                disabled={isAddingDocSource}
                            />
                            <Button
                                size="sm"
                                onClick={handleAddDocSource}
                                disabled={isAddingDocSource || !newDocSource.trim()}
                                className="self-start"
                            >
                                <Plus className="h-4 w-4" />
                            </Button>
                        </div>
                    </div>
                </div>

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
                <Separator />
                <div className="space-y-3">
                    <div className="flex items-center justify-between">
                        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                            Online Sources
                        </h3>
                    </div>
                    <div className="space-y-2">
                        {localRequirement.online_sources.map((source: string, index: number) => (
                            <Card key={index} className="border-l-4 border-l-blue-500">
                                <CardContent className="px-4">
                                    <div className="flex items-start gap-2 justify-between">
                                        <a
                                            href={source}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="flex items-start gap-2 text-blue-600 hover:text-blue-800 hover:underline flex-1 min-w-0"
                                        >
                                            <Search className="h-4 w-4 mt-0.5 flex-shrink-0" />
                                            <span className="text-sm break-all">{source}</span>
                                        </a>
                                        <Button
                                            size="sm"
                                            variant="ghost"
                                            onClick={() => handleRemoveOnlineSource(index)}
                                            className="h-6 w-6 p-0 flex-shrink-0"
                                        >
                                            <Trash2 className="h-3 w-3 text-destructive" />
                                        </Button>
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                        <div className="flex gap-2">
                            <Textarea
                                placeholder="Add new online source (URL)..."
                                value={newOnlineSource}
                                onChange={(e) => setNewOnlineSource(e.target.value)}
                                className="min-h-[60px] text-sm resize-none flex-1 min-w-0"
                                disabled={isAddingOnlineSource}
                            />
                            <Button
                                size="sm"
                                onClick={handleAddOnlineSource}
                                disabled={isAddingOnlineSource || !newOnlineSource.trim()}
                                className="self-start"
                            >
                                <Plus className="h-4 w-4" />
                            </Button>
                        </div>
                    </div>
                </div>
            </div>
        </ScrollArea>
    );
}

export default RequirementDetailView;
