"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Upload, X, FileText, CheckCircle2, AlertCircle } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import axios from "axios";
import { routes } from "@/config/routes";

interface FileUploadItem {
    file: File;
    progress: number;
    status: "pending" | "uploading" | "success" | "error";
    error?: string;
}

interface FileUploadDialogProps {
    onUploadComplete?: () => void;
    onClose?: () => void;
}

export default function FileUploadDialog({ onUploadComplete, onClose }: FileUploadDialogProps) {
    const [files, setFiles] = useState<FileUploadItem[]>([]);
    const [isUploading, setIsUploading] = useState(false);
    const [selectionNotice, setSelectionNotice] = useState<string | null>(null);

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            const selectedFiles = Array.from(e.target.files);
            const newFiles = selectedFiles
                .filter(file => file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf"))
                .map(file => ({
                    file,
                    progress: 0,
                    status: "pending" as const,
                }));

            setFiles(prev => [...prev, ...newFiles]);

            const nonPdfCount = selectedFiles.length - newFiles.length;
            if (nonPdfCount > 0) {
                setSelectionNotice(`${nonPdfCount} non-PDF file(s) were ignored. Only PDF files are accepted.`);
            } else if (newFiles.length > 0) {
                setSelectionNotice(null);
            } else if (selectedFiles.length > 0) {
                setSelectionNotice("No PDF files were selected.");
            }

            e.target.value = "";
        }
    };

    const removeFile = (index: number) => {
        setFiles(prev => prev.filter((_, i) => i !== index));
    };

    const uploadFiles = async () => {
        setIsUploading(true);
        setSelectionNotice(null);
        let attemptedUploads = 0;
        let successfulUploads = 0;

        for (let i = 0; i < files.length; i++) {
            if (files[i].status !== "pending") continue;
            attemptedUploads += 1;

            const formData = new FormData();
            formData.append("file", files[i].file);

            try {
                // Update status to uploading
                setFiles(prev => {
                    const updated = [...prev];
                    updated[i] = { ...updated[i], status: "uploading" };
                    return updated;
                });

                await axios.post(routes.upload_pdf, formData, {
                    headers: {
                        "Content-Type": "multipart/form-data",
                    },
                    onUploadProgress: (progressEvent) => {
                        const percentCompleted = Math.round(
                            (progressEvent.loaded * 100) / (progressEvent.total || 1)
                        );
                        setFiles(prev => {
                            const updated = [...prev];
                            updated[i] = { ...updated[i], progress: percentCompleted };
                            return updated;
                        });
                    },
                });
                successfulUploads += 1;

                // Mark as success
                setFiles(prev => {
                    const updated = [...prev];
                    updated[i] = { ...updated[i], status: "success", progress: 100 };
                    return updated;
                });
            } catch (error) {
                let message = "Upload failed";
                if (axios.isAxiosError(error)) {
                    message = error.response?.data?.detail || error.message;
                } else if (error instanceof Error) {
                    message = error.message;
                }
                // Mark as error
                setFiles(prev => {
                    const updated = [...prev];
                    updated[i] = {
                        ...updated[i],
                        status: "error",
                        error: message,
                    };
                    return updated;
                });
            }
        }

        setIsUploading(false);

        if (attemptedUploads > 0 && successfulUploads === attemptedUploads && onUploadComplete) {
            onUploadComplete();
        }
    };

    const clearAll = () => {
        setFiles([]);
    };

    const getStatusIcon = (status: FileUploadItem["status"]) => {
        switch (status) {
            case "success":
                return <CheckCircle2 className="h-5 w-5 text-green-600" />;
            case "error":
                return <AlertCircle className="h-5 w-5 text-red-600" />;
            case "uploading":
                return <Upload className="h-5 w-5 text-blue-600 animate-pulse" />;
            default:
                return <FileText className="h-5 w-5 text-gray-400" />;
        }
    };

    return (
        <Card className="w-full">
            <CardHeader>
                <div className="flex items-start justify-between">
                    <div className="flex-1 space-y-2">
                        <CardTitle>Upload PDF Files</CardTitle>
                        <CardDescription>
                            Select one or more PDF files to upload. Only PDF format is accepted.
                        </CardDescription>
                    </div>
                    {onClose && (
                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={onClose}
                            className="h-8 w-8 p-0"
                        >
                            <X className="h-4 w-4" />
                        </Button>
                    )}
                </div>
            </CardHeader>
            <CardContent className="space-y-4">
                {selectionNotice && (
                    <Alert variant="destructive">
                        <AlertDescription>{selectionNotice}</AlertDescription>
                    </Alert>
                )}
                {/* File Input */}
                <div className="flex gap-2">
                    <Button
                        variant="outline"
                        className="flex-1"
                        onClick={() => document.getElementById("file-input")?.click()}
                        disabled={isUploading}
                    >
                        <Upload className="mr-2 h-4 w-4" />
                        Select PDF Files
                    </Button>
                    <input
                        id="file-input"
                        type="file"
                        multiple
                        accept=".pdf,application/pdf"
                        className="hidden"
                        onChange={handleFileSelect}
                        disabled={isUploading}
                    />
                    {files.length > 0 && (
                        <Button
                            variant="destructive"
                            onClick={clearAll}
                            disabled={isUploading}
                        >
                            Clear All
                        </Button>
                    )}
                </div>

                {/* File List */}
                {files.length > 0 && (
                    <div className="space-y-2 max-h-96 overflow-y-auto">
                        {files.map((item, index) => (
                            <div
                                key={index}
                                className="border rounded-lg p-3 space-y-2"
                            >
                                <div className="flex items-start justify-between gap-2">
                                    <div className="flex items-start gap-2 flex-1 min-w-0">
                                        {getStatusIcon(item.status)}
                                        <div className="flex-1 min-w-0">
                                            <p className="text-sm font-medium truncate">
                                                {item.file.name}
                                            </p>
                                            <p className="text-xs text-gray-500">
                                                {(item.file.size / 1024 / 1024).toFixed(2)} MB
                                            </p>
                                        </div>
                                    </div>
                                    {item.status === "pending" && !isUploading && (
                                        <Button
                                            variant="ghost"
                                            size="sm"
                                            onClick={() => removeFile(index)}
                                        >
                                            <X className="h-4 w-4" />
                                        </Button>
                                    )}
                                </div>

                                {/* Progress Bar */}
                                {(item.status === "uploading" || item.status === "success") && (
                                    <div className="space-y-1">
                                        <p className="text-xs text-gray-500 text-right">
                                            {item.progress}%
                                        </p>
                                    </div>
                                )}

                                {/* Error Message */}
                                {item.status === "error" && item.error && (
                                    <Alert variant="destructive">
                                        <AlertDescription className="text-xs">
                                            {item.error}
                                        </AlertDescription>
                                    </Alert>
                                )}
                            </div>
                        ))}
                    </div>
                )}

                {/* Upload Button */}
                {files.length > 0 && (
                    <Button
                        className="w-full"
                        onClick={uploadFiles}
                        disabled={isUploading || files.every(f => f.status !== "pending")}
                    >
                        {isUploading ? "Uploading..." : "Upload Files"}
                    </Button>
                )}

                {/* Info Alert */}
                {files.length === 0 && (
                    <Alert>
                        <FileText className="h-4 w-4" />
                        <AlertDescription>
                            Click &quot;Select PDF Files&quot; to choose files. Only <strong>PDF</strong> files are accepted.
                        </AlertDescription>
                    </Alert>
                )}
            </CardContent>
        </Card>
    );
}
