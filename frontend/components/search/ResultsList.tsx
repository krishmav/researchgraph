// components/search/ResultsList.tsx
import PaperCard from "@/components/shared/PaperCard";
import type { SearchResult } from "@/lib/types";
import { EmptyState } from "@/components/shared/PaperCard";
import { Search } from "lucide-react";

interface Props {
  results: SearchResult[];
}

export default function ResultsList({ results }: Props) {
  if (results.length === 0) {
    return (
      <EmptyState
        title="No results found"
        description="Try a different query or retrieval method."
        icon={<Search className="w-10 h-10" />}
      />
    );
  }

  return (
    <div className="space-y-3">
      {results.map((r) => (
        <PaperCard
          key={r.paper.arxiv_id}
          paper={r.paper}
          score={r.score}
          explanation={r.explanation}
          rank={r.rank}
        />
      ))}
    </div>
  );
}
