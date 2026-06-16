// app/layout.tsx
import type { Metadata } from "next";
import "./globals.css";
import NavSidebar from "@/components/shared/NavSidebar";

export const metadata: Metadata = {
  title: "ResearchGraph — Semantic Research Discovery",
  description:
    "Transformer-based semantic search over 25k+ arXiv papers with knowledge graphs and topic modeling.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="flex min-h-screen bg-gray-950">
        <NavSidebar />
        <main className="flex-1 ml-56 min-h-screen overflow-y-auto">
          {children}
        </main>
      </body>
    </html>
  );
}
