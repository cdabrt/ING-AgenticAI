import { FileText, Lightbulb, Briefcase, Database } from "lucide-react";
import { Badge } from "../ui/badge";
import { Card, CardContent } from "../ui/card";
import { Separator } from "../ui/separator";
import { RequirementBundle } from "@/lib/types";
import { ScrollArea } from "../ui/scroll-area";

function BundleDetailView({ bundle }: { bundle: RequirementBundle | null }) {
    if (!bundle) {
        return (
            <div className="h-full w-full flex items-center justify-center p-8">
                <p className="text-muted-foreground text-center">
                    Select a bundle to view details
                </p>
            </div>
        );
    }

    return (
        <ScrollArea className="h-full w-full overflow-x-hidden min-w-0">
            <div className="p-6 space-y-6">
                {/* Header with Document Name and Type */}
                <div className="space-y-3">
                    <div className="flex items-center gap-3">
                        <FileText className="h-6 w-6 text-primary" />
                        <h2 className="text-2xl font-semibold break-words">{bundle.document}</h2>
                    </div>
                    <div className="rounded-lg bg-muted px-3 py-2 text-xs text-muted-foreground whitespace-normal break-words">
                        {bundle.document_type}
                    </div>
                </div>

                <Separator />

                {/* Business Requirements Section */}
                <div className="space-y-3">
                    <div className="flex items-center gap-2">
                        <Briefcase className="h-5 w-5 text-primary" />
                        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                            Business Requirements
                        </h3>
                        <Badge variant="outline" className="ml-auto">
                            {bundle.business_requirements.length}
                        </Badge>
                    </div>
                    {bundle.business_requirements.length > 0 ? (
                        <div className="space-y-2">
                            {bundle.business_requirements.map((requirement, index) => (
                                <Card key={index} className="border-l-4 border-l-primary">
                                    <CardContent className="px-4">
                                        <div className="space-y-2">
                                            <div className="flex items-start justify-between gap-2">
                                                <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-1 rounded">
                                                    {requirement.id}
                                                </span>
                                            </div>
                                            <p className="text-sm leading-relaxed break-words">
                                                {requirement.description}
                                            </p>
                                        </div>
                                    </CardContent>
                                </Card>
                            ))}
                        </div>
                    ) : (
                        <p className="text-sm text-muted-foreground italic">No business requirements</p>
                    )}
                </div>

                <Separator />

                {/* Data Requirements Section */}
                <div className="space-y-3">
                    <div className="flex items-center gap-2">
                        <Database className="h-5 w-5 text-blue-500" />
                        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                            Data Requirements
                        </h3>
                        <Badge variant="outline" className="ml-auto">
                            {bundle.data_requirements.length}
                        </Badge>
                    </div>
                    {bundle.data_requirements.length > 0 ? (
                        <div className="space-y-2">
                            {bundle.data_requirements.map((requirement, index) => (
                                <Card key={index} className="border-l-4 border-l-blue-500">
                                    <CardContent className="px-4">
                                        <div className="space-y-2">
                                            <div className="flex items-start justify-between gap-2">
                                                <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-1 rounded">
                                                    {requirement.id}
                                                </span>
                                            </div>
                                            <p className="text-sm leading-relaxed break-words">
                                                {requirement.description}
                                            </p>
                                        </div>
                                    </CardContent>
                                </Card>
                            ))}
                        </div>
                    ) : (
                        <p className="text-sm text-muted-foreground italic">No data requirements</p>
                    )}
                </div>

                <Separator />

                {/* Assumptions Section */}
                <div className="space-y-3">
                    <div className="flex items-center gap-2">
                        <Lightbulb className="h-5 w-5 text-amber-500" />
                        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                            Assumptions
                        </h3>
                        <Badge variant="outline" className="ml-auto">
                            {bundle.assumptions.length}
                        </Badge>
                    </div>
                    {bundle.assumptions.length > 0 ? (
                        <div className="space-y-2">
                            {bundle.assumptions.map((assumption, index) => (
                                <Card key={index} className="border-l-4 border-l-amber-500">
                                    <CardContent className="px-4">
                                        <div className="flex items-start gap-2">
                                            <span className="text-sm font-medium text-muted-foreground min-w-6">
                                                {index + 1}.
                                            </span>
                                            <span className="text-sm leading-relaxed break-words">{assumption}</span>
                                        </div>
                                    </CardContent>
                                </Card>
                            ))}
                        </div>
                    ) : (
                        <p className="text-sm text-muted-foreground italic">No assumptions</p>
                    )}
                </div>
            </div>
        </ScrollArea>
    );
}

export default BundleDetailView;
