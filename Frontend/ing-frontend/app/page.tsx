"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { RequirementBundle, RequirementItem } from "@/lib/types";
import { AlertCircleIcon, Search } from "lucide-react";
import axios from "axios";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { ButtonGroup, ButtonGroupSeparator } from "@/components/ui/button-group";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import RequirementDetailView from "@/components/custom/RequirementDetailView";
import RequirementListSkeleton from "@/components/custom/RequirementListSkeleton";
import RequirementList from "@/components/custom/RequirementList";
import RequirementBundleList from "@/components/custom/RequirementBundleList";
import BundleDetailView from "@/components/custom/BundleDetailView";

const buttons = ["Business Requirements", "Data Requirements"];

export default function Home() {

  // TEST MOCK DATA
  // const mockRequirementItems: RequirementItem[] = [
  //   {
  //     id: "req-1",
  //     description: "The system shall authenticate users using multi-factor authentication (MFA) to ensure secure access to sensitive financial data.",
  //     rationale: "MFA provides an additional layer of security beyond passwords, reducing the risk of unauthorized access to customer accounts and financial information.",
  //     document_sources: ["Security Policy v2.1", "Authentication Standards"],
  //     online_sources: ["https://www.nist.gov/mfa-guidelines"]
  //   },
  //   {
  //     id: "req-2",
  //     description: "The application must process payment transactions within 3 seconds under normal load conditions.",
  //     rationale: "Fast transaction processing improves user experience and reduces cart abandonment rates in e-commerce scenarios.",
  //     document_sources: ["Performance Requirements Specification"],
  //     online_sources: []
  //   },
  //   {
  //     id: "req-3",
  //     description: "All customer data must be encrypted at rest using AES-256 encryption standard.",
  //     rationale: "Encryption at rest protects sensitive customer information from unauthorized access in case of physical storage breach.",
  //     document_sources: ["Data Protection Policy", "GDPR Compliance Guidelines"],
  //     online_sources: ["https://gdpr.eu/encryption"]
  //   },
  //   {
  //     id: "req-4",
  //     description: "The system shall maintain an audit log of all user actions for a minimum of 7 years.",
  //     rationale: "Regulatory compliance requires long-term retention of audit trails for financial transactions and user activities.",
  //     document_sources: ["Audit Policy", "Financial Regulations Document"],
  //     online_sources: []
  //   },
  //   {
  //     id: "req-5",
  //     description: "The user interface must be accessible and comply with WCAG 2.1 Level AA standards.",
  //     rationale: "Accessibility ensures that users with disabilities can effectively use the application, meeting legal requirements and expanding user base.",
  //     document_sources: ["Accessibility Standards"],
  //     online_sources: ["https://www.w3.org/WAI/WCAG21/quickref"]
  //   }
  // ];

  // const mockBusinessRequirements: RequirementItem[] = [
  //   {
  //     id: "BR-1",
  //     description: "The system shall support user registration and profile management.",
  //     rationale: "Enables users to create accounts and manage their personal information.",
  //     document_sources: ["Business Requirements Document v1.0"],
  //     online_sources: []
  //   },
  //   {
  //     id: "BR-2",
  //     description: "The application must provide real-time account balance updates.",
  //     rationale: "Keeps users informed about their financial status, enhancing user experience.",
  //     document_sources: ["Business Requirements Document v1.0"],
  //     online_sources: []
  //   },
  //   {
  //     id: "BR-3",
  //     description: "The platform shall facilitate fund transfers between accounts.",
  //     rationale: "Allows users to move money easily, a core banking function.",
  //     document_sources: ["Business Requirements Document v1.0"],
  //     online_sources: []
  //   }
  // ];

  // const mockRequirementBundles: RequirementBundle[] = [
  //   {
  //     document: "Business Requirements Document v1.0",
  //     document_type: "Business Requirements",
  //     business_requirements: mockBusinessRequirements,
  //     data_requirements: mockRequirementItems,
  //     assumptions: ["The system will be used by customers aged 18 and above.", "All transactions will be conducted in compliance with local financial regulations."]
  //   }
  // ];

  // TEST
  // const [selected, setSelected] = useState<RequirementItem | null>(mockRequirementItems[0]);
  // const [selectedBundle, setSelectedBundle] = useState<RequirementBundle | null>(mockRequirementBundles[0]);
  // const [selectedList, setSelectedList] = useState<number>(0); // 0 - RequirementBundles, 1 - BusinessRequirements, 2 - DataRequirements
  // const [results, setResults] = useState<RequirementBundle[]>([...mockRequirementBundles]);

  // REAL
  const [selected, setSelected] = useState<RequirementItem | null>(null);
  const [selectedBundle, setSelectedBundle] = useState<RequirementBundle | null>(null);
  const [selectedList, setSelectedList] = useState<number>(0); // 0 - RequirementBundles, 1 - BusinessRequirements, 2 - DataRequirements
  const [results, setResults] = useState<RequirementBundle[]>([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function getRequirementList(): RequirementItem[] {
    if (selectedList == 1) {
      return selectedBundle ? selectedBundle.business_requirements : [];
    } else if (selectedList == 2) {
      return selectedBundle ? selectedBundle.data_requirements : [];
    }
    return [];
  }

  function getRequirementBundle(): Promise<void> {
    setError(null);
    return axios.post('/api/pipeline').catch((error) => {
      setError(error.message);
    }).then((response) => {
      if (response && response.data) {
        let maxid = results.length > 0 ? Math.max(...results.map(b => b.id)) : 0;
        response.data.id = maxid + 1;

        setResults([...results, response.data]);
        setSelectedBundle(response.data);
        setSelectedList(0);
      } else {
        setResults([]);
        setError("No data received!");
      }
    });
  }

  useEffect(() => {
    // Initial load can be handled here if needed
    getRequirementBundle();
  }, []);

  return (
    <div className="h-screen bg-zinc-100 dark:bg-black flex flex-col relative">
      {/* Header */}
      <div className="h-20 w-full">
        <div className="h-full w-full flex flex-row justify-start items-center px-6">
          <div className="w-80 flex-grow-1">
            <h1 className="scroll-m-20 text-center text-4xl font-extrabold tracking-tight text-balance">
              ING Agentic AI
            </h1>
          </div>
          <div className="w-2/3">
            <InputGroup className="bg-zinc-50">
              <InputGroupInput placeholder="Search..." />
              <InputGroupAddon>
                <Search />
              </InputGroupAddon>
            </InputGroup>
          </div>
        </div>
      </div>
      <div className="w-full flex-1 flex flex-row overflow-hidden">
        <ResizablePanelGroup direction={"horizontal"}>
          <ResizablePanel defaultSize={1 / 3} className="min-w-1/4">
            <div className="h-full w-full bg-white border-t border-r overflow-hidden">
              {selectedList !== 0 && <RequirementDetailView requirement={selected} />}
              {selectedList === 0 && <BundleDetailView bundle={selectedBundle} />}
            </div>
          </ResizablePanel>
          <ResizableHandle withHandle />
          <ResizablePanel defaultSize={2 / 3} className="min-w-1/4 flex flex-col">
            <div className="w-full flex flex-row justify-center items-center bg-zinc-200 p-2 space-x-4">
              <ButtonGroup>
                {buttons.map((btn, index) => (
                  <Button key={index} variant="outline" onClick={() => setSelectedList(index + 1)} className={`${selectedList === index + 1 ? "bg-zinc-400 hover:bg-zinc-500" : ""}`}>
                    {btn}
                  </Button>
                ))}
              </ButtonGroup>
              <ButtonGroup>
                <Button
                  onClick={() => setSelectedList(0)}
                >
                  Select Requirement Bundle
                </Button>
                <ButtonGroupSeparator orientation="vertical" />
                <Button
                  onClick={() => getRequirementBundle()}
                >
                  Download
                </Button>
              </ButtonGroup>
            </div>
            {/* Main Content */}
            <div className="flex-1 w-full bg-white p-4 relative flex flex-col">
              <div className="flex-grow overflow-y-auto">
                {!results && <RequirementListSkeleton loading={loading} setLoading={setLoading} getRequirementBundle={getRequirementBundle} />}
                {selectedList > 0 &&
                  <div className="flex flex-col space-y-4">
                    <h2 className="text-black">
                      Selected bundle: <span className="font-bold">{selectedBundle ? selectedBundle.document : "None"}</span>
                    </h2>
                    <RequirementList
                      data={getRequirementList()}
                      onRowClick={(requirement) => setSelected(requirement)}
                    />
                  </div>
                }
                {
                  selectedList == 0 &&
                  <RequirementBundleList
                    data={results}
                    onRowClick={(bundle) => {
                      setSelectedBundle(bundle);
                    }}
                  />
                }
              </div>
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
      {
        error &&
        <div className="absolute w-full flex flex-row justify-end items-center p-4 pointer-events-none">
          <div className="w-max-100 y-max-120 y-overflow-scroll pointer-events-auto">
            <Alert variant="destructive">
              <AlertCircleIcon />
              <AlertTitle>Unable to download requirements!</AlertTitle>
              <AlertDescription>
                {error}
              </AlertDescription>
            </Alert>
          </div>
        </div>
      }
    </div>
  );
}
