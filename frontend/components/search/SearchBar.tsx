// components/search/SearchBar.tsx
"use client";
import { useState, KeyboardEvent } from "react";
import { Search } from "lucide-react";

interface Props {
  onSearch: (query: string) => void;
  loading?: boolean;
  placeholder?: string;
}

export default function SearchBar({
  onSearch,
  loading = false,
  placeholder = "Search research papers…",
}: Props) {
  const [value, setValue] = useState("");

  const handleKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && value.trim().length >= 3) {
      onSearch(value.trim());
    }
  };

  return (
    <div className="relative">
      <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
      <input
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKey}
        placeholder={placeholder}
        className="w-full bg-gray-900 border border-gray-700 hover:border-gray-600
                   focus:border-brand-500 focus:outline-none rounded-xl
                   pl-10 pr-28 py-3 text-sm text-white placeholder-gray-600
                   transition-colors"
        disabled={loading}
      />
      <button
        onClick={() => value.trim().length >= 3 && onSearch(value.trim())}
        disabled={loading || value.trim().length < 3}
        className="absolute right-2 top-1/2 -translate-y-1/2 btn-primary text-xs py-1.5 px-3"
      >
        {loading ? "Searching…" : "Search"}
      </button>
    </div>
  );
}
