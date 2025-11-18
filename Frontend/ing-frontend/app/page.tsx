"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import { RequirementBundle } from "@/lib/types";
import { Loader2, FileText, AlertCircle } from "lucide-react";

function RequirementAccordionItem({ item, index }: { item: RequirementBundle | null, index: number }) {
  return (
    <AccordionItem value={`bundle-${index}`}>
      <AccordionTrigger className="text-lg font-semibold">
        Requirement Bundle {index + 1}
      </AccordionTrigger>
      <AccordionContent>
        <Card>
          <CardHeader>
            <CardTitle>Requirements</CardTitle>
            <CardDescription>
              Desc
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              Test
            </div>
          </CardContent>
        </Card>
      </AccordionContent>
    </AccordionItem>
  );
}

export default function Home() {
  const [results, setResults] = useState<RequirementBundle | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const bundles: number[] = [
    1, 2, 3, 4, 5
  ];

  return (
    <div className="h-screen bg-zinc-50 dark:bg-black flex flex-col">
      {/* Header */}
      <div className="h-20 w-full shadow-md">
        <div className="h-full w-full flex flex-col justify-center items-center">
          <h1 className="text-bvblack font-bold text-3xl p-4">
            ING Agentic AI
          </h1>
        </div>
      </div>
      {/* Main Content */}
      <div className="flex-grow w-full">
        <div className="container max-w-200 mx-auto p-3">
          <div className="h-25 w-full flex p-1 justify-center items-center bg-zinc-200 rounded-md">
            <Button
              onClick={async () => {
                setLoading(true);
                await new Promise(resolve => setTimeout(resolve, 2000));
                setLoading(false);
              }}
              disabled={loading}
            >
              {loading ? <Loader2 className="animate-spin mr-2" /> : <FileText className="mr-2" />}
              Generate Requirements
            </Button>
          </div>
          <div className="pt-2">
            <div className="flex flex-col justify-center items-center bg-zinc-100 px-4 rounded-md">
              {bundles ?
                <Accordion type="single" collapsible className="w-full">
                  {bundles.map((index) =>
                    <RequirementAccordionItem key={index} item={null} index={index} />
                  )}
                </Accordion> :
                <div>
                  No requirements generated yet.
                </div>
              }
            </div>
          </div>
        </div>
      </div>
    </div >
  );
}
