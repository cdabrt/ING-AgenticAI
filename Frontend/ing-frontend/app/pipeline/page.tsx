"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import { RequirementBundle } from "@/lib/types";
import { Loader2, FileText, AlertCircle } from "lucide-react";

export default function PipelinePage() {
    const [results, setResults] = useState<RequirementBundle | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleAnalyze = async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch("http://localhost:8000/api/pipeline", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({}),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || "Failed to process pipeline request");
            }

            const data: RequirementBundle = await response.json();
            setResults(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : "An error occurred");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-zinc-50 dark:bg-black py-8 px-4">
            <div className="max-w-5xl mx-auto">
                <div className="flex flex-col items-center justify-center min-h-[50vh]">
                    <Button
                        onClick={handleAnalyze}
                        disabled={loading}
                        size="lg"
                        className="mb-8"
                    >
                        {loading && <Loader2 className="mr-2 h-5 w-5 animate-spin" />}
                        {loading ? "Processing..." : "Run Pipeline Analysis"}
                    </Button>

                    {error && (
                        <Card className="mb-8 border-red-500 w-full max-w-2xl">
                            <CardContent className="pt-6">
                                <div className="flex items-center gap-2 text-red-600">
                                    <AlertCircle className="h-5 w-5" />
                                    <p>{error}</p>
                                </div>
                            </CardContent>
                        </Card>
                    )}
                </div>                {results && (
                    <div className="space-y-6">
                        <Card>
                            <CardHeader>
                                <div className="flex items-start justify-between">
                                    <div>
                                        <CardTitle className="flex items-center gap-2">
                                            <FileText className="h-5 w-5" />
                                            {results.document}
                                        </CardTitle>
                                        <CardDescription className="mt-2">
                                            <Badge variant="outline">{results.document_type}</Badge>
                                        </CardDescription>
                                    </div>
                                </div>
                            </CardHeader>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle>Business Requirements ({results.business_requirements.length})</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <Accordion type="single" collapsible className="w-full">
                                    {results.business_requirements.map((req, index) => (
                                        <AccordionItem key={req.id} value={`business-${index}`}>
                                            <AccordionTrigger>
                                                <div className="text-left">
                                                    <Badge variant="secondary" className="mr-2">
                                                        {req.id}
                                                    </Badge>
                                                    {req.description}
                                                </div>
                                            </AccordionTrigger>
                                            <AccordionContent>
                                                <div className="space-y-4 pt-2">
                                                    <div>
                                                        <h4 className="font-semibold mb-1">Rationale</h4>
                                                        <p className="text-sm text-muted-foreground">{req.rationale}</p>
                                                    </div>

                                                    {req.document_sources.length > 0 && (
                                                        <div>
                                                            <h4 className="font-semibold mb-2">Document Sources</h4>
                                                            <ul className="list-disc list-inside space-y-1">
                                                                {req.document_sources.map((source, idx) => (
                                                                    <li key={idx} className="text-sm text-muted-foreground">
                                                                        {source}
                                                                    </li>
                                                                ))}
                                                            </ul>
                                                        </div>
                                                    )}

                                                    {req.online_sources.length > 0 && (
                                                        <div>
                                                            <h4 className="font-semibold mb-2">Online Sources</h4>
                                                            <ul className="list-disc list-inside space-y-1">
                                                                {req.online_sources.map((source, idx) => (
                                                                    <li key={idx} className="text-sm">
                                                                        <a
                                                                            href={source}
                                                                            target="_blank"
                                                                            rel="noopener noreferrer"
                                                                            className="text-blue-600 hover:underline"
                                                                        >
                                                                            {source}
                                                                        </a>
                                                                    </li>
                                                                ))}
                                                            </ul>
                                                        </div>
                                                    )}
                                                </div>
                                            </AccordionContent>
                                        </AccordionItem>
                                    ))}
                                </Accordion>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle>Data Requirements ({results.data_requirements.length})</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <Accordion type="single" collapsible className="w-full">
                                    {results.data_requirements.map((req, index) => (
                                        <AccordionItem key={req.id} value={`data-${index}`}>
                                            <AccordionTrigger>
                                                <div className="text-left">
                                                    <Badge variant="secondary" className="mr-2">
                                                        {req.id}
                                                    </Badge>
                                                    {req.description}
                                                </div>
                                            </AccordionTrigger>
                                            <AccordionContent>
                                                <div className="space-y-4 pt-2">
                                                    <div>
                                                        <h4 className="font-semibold mb-1">Rationale</h4>
                                                        <p className="text-sm text-muted-foreground">{req.rationale}</p>
                                                    </div>

                                                    {req.document_sources.length > 0 && (
                                                        <div>
                                                            <h4 className="font-semibold mb-2">Document Sources</h4>
                                                            <ul className="list-disc list-inside space-y-1">
                                                                {req.document_sources.map((source, idx) => (
                                                                    <li key={idx} className="text-sm text-muted-foreground">
                                                                        {source}
                                                                    </li>
                                                                ))}
                                                            </ul>
                                                        </div>
                                                    )}

                                                    {req.online_sources.length > 0 && (
                                                        <div>
                                                            <h4 className="font-semibold mb-2">Online Sources</h4>
                                                            <ul className="list-disc list-inside space-y-1">
                                                                {req.online_sources.map((source, idx) => (
                                                                    <li key={idx} className="text-sm">
                                                                        <a
                                                                            href={source}
                                                                            target="_blank"
                                                                            rel="noopener noreferrer"
                                                                            className="text-blue-600 hover:underline"
                                                                        >
                                                                            {source}
                                                                        </a>
                                                                    </li>
                                                                ))}
                                                            </ul>
                                                        </div>
                                                    )}
                                                </div>
                                            </AccordionContent>
                                        </AccordionItem>
                                    ))}
                                </Accordion>
                            </CardContent>
                        </Card>

                        {results.assumptions.length > 0 && (
                            <Card>
                                <CardHeader>
                                    <CardTitle>Assumptions ({results.assumptions.length})</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <ul className="space-y-2">
                                        {results.assumptions.map((assumption, index) => (
                                            <li key={index} className="flex items-start gap-2">
                                                <Badge variant="outline" className="mt-0.5">
                                                    {index + 1}
                                                </Badge>
                                                <span className="text-sm">{assumption}</span>
                                            </li>
                                        ))}
                                    </ul>
                                </CardContent>
                            </Card>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
