// components/shared/NavSidebar.tsx
"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Search,
  Network,
  BarChart3,
  Lightbulb,
  FlaskConical,
  BookOpen,
  Activity,
} from "lucide-react";
import clsx from "clsx";

const NAV = [
  { href: "/",          label: "Search",         icon: Search },
  { href: "/topics",    label: "Topic Explorer", icon: BarChart3 },
  { href: "/graph",     label: "Knowledge Graph",icon: Network },
  { href: "/gaps",      label: "Research Gaps",  icon: Lightbulb },
  { href: "/evaluate",  label: "Evaluation",     icon: FlaskConical },
];

export default function NavSidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 w-56 h-full bg-gray-900 border-r border-gray-800
                      flex flex-col z-40">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-gray-800">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-brand-500 flex items-center justify-center">
            <BookOpen className="w-4 h-4 text-white" />
          </div>
          <span className="font-bold text-white text-sm tracking-tight">ResearchGraph</span>
        </div>
        <p className="text-xs text-gray-500 mt-1.5 leading-snug">
          Semantic discovery over 25k+ arXiv papers
        </p>
      </div>

      {/* Nav links */}
      <nav className="flex-1 px-3 py-4 space-y-0.5">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = pathname === href || (href !== "/" && pathname.startsWith(href));
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                active
                  ? "bg-brand-500/20 text-brand-300 border border-brand-500/30"
                  : "text-gray-400 hover:text-gray-100 hover:bg-gray-800"
              )}
            >
              <Icon className="w-4 h-4 shrink-0" />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* Bottom status */}
      <div className="px-4 py-4 border-t border-gray-800">
        <SystemStatus />
      </div>
    </aside>
  );
}

function SystemStatus() {
  return (
    <div className="flex items-center gap-2 text-xs text-gray-500">
      <Activity className="w-3.5 h-3.5" />
      <span>API connected</span>
      <div className="ml-auto w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
    </div>
  );
}
