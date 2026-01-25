'use client';

import { useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

interface PDFViewerProps {
    file: string | File;
}

export function PDFViewer({ file }: PDFViewerProps) {
    const [numPages, setNumPages] = useState<number>(0);
    const [scale, setScale] = useState<number>(1.0);

    function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
        setNumPages(numPages);
    }

    function zoomIn() {
        setScale(prev => Math.min(prev + 0.25, 3.0));
    }

    function zoomOut() {
        setScale(prev => Math.max(prev - 0.25, 0.5));
    }

    function resetZoom() {
        setScale(1.0);
    }

    return (
        <div className="flex flex-col h-full">
            <div className="flex items-center justify-center gap-2 p-3 bg-gray-50 border-b">
                <Button variant="outline" size="sm" onClick={zoomOut} disabled={scale <= 0.5}>
                    <ZoomOut className="h-4 w-4" />
                    Zoom Out
                </Button>
                <span className="text-sm font-medium min-w-[60px] text-center">{Math.round(scale * 100)}%</span>
                <Button variant="outline" size="sm" onClick={zoomIn} disabled={scale >= 3.0}>
                    <ZoomIn className="h-4 w-4" />
                    Zoom In
                </Button>
                <Button variant="outline" size="sm" onClick={resetZoom}>
                    <RotateCcw className="h-4 w-4" />
                    Reset
                </Button>
            </div>
            <div className="flex-1 overflow-auto bg-gray-100">
                <Document 
                    file={file} 
                    onLoadSuccess={onDocumentLoadSuccess}
                    className="flex flex-col items-center gap-4 p-4"
                >
                    {Array.from(new Array(numPages), (el, index) => (
                        <Page
                            key={`page_${index + 1}`}
                            pageNumber={index + 1}
                            renderTextLayer={true}
                            renderAnnotationLayer={true}
                            className="shadow-lg"
                            scale={scale}
                        />
                    ))}
                </Document>
            </div>
        </div>
    );
}
