import { FileText, Loader2 } from "lucide-react";
import { AspectRatio } from "../ui/aspect-ratio";
import { Button } from "../ui/button";
import { Separator } from "../ui/separator";
import { Skeleton } from "../ui/skeleton";

function RequirementListSkeleton({ loading, setLoading, getRequirementBundle }: { loading: boolean; setLoading: (loading: boolean) => void; getRequirementBundle: () => Promise<void> }) {
    return (
        <div className="flex flex-col">
            <div className="space-y-2 px-2">
                <Skeleton className="h-4 w-[1150px]" />
                <Skeleton className="h-4 w-[900px]" />
            </div>
            <Separator className="my-4" />
            <div className="space-y-2 px-2">
                <Skeleton className="h-4 w-[1150px]" />
                <Skeleton className="h-4 w-[900px]" />
            </div>
            <Separator className="my-4" />
            <div className="space-y-2 px-2">
                <Skeleton className="h-4 w-[1150px]" />
                <Skeleton className="h-4 w-[900px]" />
            </div>
            <Separator className="my-4" />
            <div className="space-y-2 px-2">
                <Skeleton className="h-4 w-[1150px]" />
                <Skeleton className="h-4 w-[900px]" />
            </div>
            <Separator className="my-4" />
            <div className="space-y-2 px-2">
                <Skeleton className="h-4 w-[1150px]" />
                <Skeleton className="h-4 w-[900px]" />
            </div>
            <Separator className="my-4" />
            <div className="space-y-2 px-2">
                <Skeleton className="h-4 w-[1150px]" />
                <Skeleton className="h-4 w-[900px]" />
            </div>
            <Separator className="my-4" />
            <div className="space-y-2 px-2">
                <Skeleton className="h-4 w-[1150px]" />
                <Skeleton className="h-4 w-[900px]" />
            </div>
            <Separator className="my-4" />
            <div className="space-y-2 px-2">
                <Skeleton className="h-4 w-[1150px]" />
                <Skeleton className="h-4 w-[900px]" />
            </div>
            <Separator className="my-4" />
            <div className="space-y-2 px-2">
                <Skeleton className="h-4 w-[1150px]" />
                <Skeleton className="h-4 w-[900px]" />
            </div>
            <Separator className="my-4" />
            <div className="space-y-2 px-2">
                <Skeleton className="h-4 w-[1150px]" />
                <Skeleton className="h-4 w-[900px]" />
            </div>
            <Separator className="my-4" />
            <div className="space-y-2 px-2">
                <Skeleton className="h-4 w-[1150px]" />
                <Skeleton className="h-4 w-[900px]" />
            </div>
            <div className="h-full w-full flex flex-col justify-center items-center absolute top-0 left-0">
                <div className="w-full max-w-xl px-4">
                    <AspectRatio ratio={16 / 9} className="bg-white flex flex-col justify-center items-center shadow-lg rounded-lg p-4">
                        <div className="pb-12">
                            <h2 className="scroll-m-20 border-b pb-2 text-3xl font-semibold tracking-tight first:mt-0">
                                Generate Requirement Bundle!
                            </h2>
                        </div>
                        <Button
                            onClick={async () => {
                                setLoading(true);
                                await getRequirementBundle();
                                setLoading(false);
                            }}
                            disabled={loading}
                        >
                            {loading ? <Loader2 className="animate-spin mr-2" /> : <FileText className="mr-2" />}
                            Generate Requirements
                        </Button>
                    </AspectRatio>
                </div>
            </div>
        </div>
    );
}

export default RequirementListSkeleton;