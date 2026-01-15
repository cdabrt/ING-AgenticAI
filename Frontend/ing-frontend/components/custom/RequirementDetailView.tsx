import { FileText, Search, Edit, Save } from "lucide-react";
import { Card, CardContent } from "../ui/card";
import { Separator } from "../ui/separator";
import { RequirementItem } from "@/lib/types";
import { ScrollArea } from "../ui/scroll-area";
import { Button } from "../ui/button";
import { Textarea } from "../ui/textarea";
import { useState } from "react";
import axios from "axios";

function RequirementDetailView({ requirement }: { requirement: RequirementItem | null }) {

    const [isEditMode, setIsEditMode] = useState<boolean>(false);
    const [description, setDescription] = useState<string>("");
    const [rationale, setRationale] = useState<string>("");
    const [error, setError] = useState<string | null>(null);
    
    if (!requirement) {
        return (
            <div className="h-full w-full flex items-center justify-center p-8">
                <p className="text-muted-foreground text-center">
                    Select a requirement to view details
                </p>
            </div>
        );
    }

    // Initialize editable fields when requirement changes
    if (description === "" && requirement.description) {
        setDescription(requirement.description);
    }
    if (rationale === "" && requirement.rationale) {
        setRationale(requirement.rationale);
    }

    const handleToggleEditMode = () => {
        if (isEditMode) {
            let tmp_requirement = { ...requirement, description, rationale };

               setError(null);
                axios.post('/api/requirement', tmp_requirement).catch((error) => {
                    setError(error.message);
                }).then((response) => {
                    if (!response || !response.data) {
                        setError("No data received!");
                    }
                });
        } else {
            setDescription(requirement.description);
            setRationale(requirement.rationale);
        }
        setIsEditMode(!isEditMode);
    };

    return (
        <ScrollArea className="h-full w-full overflow-y-auto">
            <div className="p-6 space-y-6">
                {/* Header with ID and Edit Button */}
                <div className="flex items-center justify-between">
                    <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-1 rounded">
                        {requirement.id}
                    </span>
                    <Button
                        onClick={handleToggleEditMode}
                        variant={isEditMode ? "default" : "outline"}
                        size="sm"
                    >
                        {isEditMode ? (
                            <>
                                <Save className="h-4 w-4 mr-2" />
                                Save
                            </>
                        ) : (
                            <>
                                <Edit className="h-4 w-4 mr-2" />
                                Edit
                            </>
                        )}
                    </Button>
                </div>

                {/* Description Section */}
                <div className="space-y-2">
                    <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                        Description
                    </h3>
                    {isEditMode ? (
                        <Textarea
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            className="text-base leading-relaxed"
                            rows={4}
                        />
                    ) : (
                        <p className="text-base leading-relaxed">
                            {description}
                        </p>
                    )}
                </div>

                <Separator />

                {/* Rationale Section */}
                <div className="space-y-2">
                    <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                        Rationale
                    </h3>
                    {isEditMode ? (
                        <Textarea
                            value={rationale}
                            onChange={(e) => setRationale(e.target.value)}
                            className="text-base leading-relaxed text-muted-foreground"
                            rows={4}
                        />
                    ) : (
                        <p className="text-base leading-relaxed text-muted-foreground">
                            {rationale}
                        </p>
                    )}
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
                                            <span className="text-sm">{source}</span>
                                        </div>
                                    </CardContent>
                                </Card>
                            ))}
                        </div>
                    </div>
                )}

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