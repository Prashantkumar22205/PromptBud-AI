"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useSession, signOut } from "next-auth/react";

export default function Navbar() {
  const pathname = usePathname();
  const { data: session } = useSession();

  const navLinks = [
    { href: "/", label: "Optimizer" },
    { href: "/library", label: "Library" },
  ];

  return (
    <header
      style={{
        width: "100%",
        background: "var(--color-surface-low)",
        borderBottom: "1px solid var(--color-outline)",
        position: "sticky",
        top: 0,
        zIndex: 50,
      }}
    >
      <div
        className="container-app"
        style={{ display: "flex", justifyContent: "space-between", alignItems: "center", height: 60 }}
      >
        {/* Brand */}
        <Link href="/" style={{ display: "flex", alignItems: "center", gap: 8, textDecoration: "none" }}>
          <span className="material-symbols-outlined" style={{ color: "var(--color-primary)", fontSize: 22, fontVariationSettings: "'FILL' 1" }}>
            auto_awesome
          </span>
          <span style={{ fontFamily: "var(--font-label)", fontWeight: 900, fontSize: 14, letterSpacing: "-0.02em", color: "var(--color-on-background)", textTransform: "uppercase" }}>
            PromptBud AI
          </span>
        </Link>

        {/* Nav links */}
        <nav className="top-nav" style={{ display: "flex", gap: 4 }}>
          {navLinks.map((l) => {
            const active = pathname === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                style={{
                  fontFamily: "var(--font-label)", fontSize: 13, fontWeight: 500,
                  letterSpacing: "0.05em", padding: "6px 12px", borderRadius: "var(--radius-lg)",
                  textDecoration: "none",
                  color: active ? "var(--color-primary)" : "var(--color-on-surface-variant)",
                  borderBottom: active ? "2px solid var(--color-primary)" : "2px solid transparent",
                  transition: "color 0.15s",
                }}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>

        {/* Actions */}
        <div className="nav-actions" style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {session ? (
            <>
              <Link href="/profile" title="Action Center">
                <button className="btn-ghost" style={{ padding: "6px 10px" }}>
                  <span className="material-symbols-outlined" style={{ fontSize: 18 }}>manage_accounts</span>
                </button>
              </Link>
              <button
                className="btn-ghost"
                style={{ padding: "6px 14px", fontSize: 13 }}
                onClick={() => signOut({ callbackUrl: "/login" })}
              >
                Sign Out
              </button>
            </>
          ) : (
            <>
              <Link href="/login">
                <button className="btn-ghost" style={{ padding: "6px 14px", fontSize: 13 }}>Log In</button>
              </Link>
              <Link href="/login?tab=signup">
                <button className="btn-primary" style={{ padding: "6px 14px", fontSize: 13 }}>Get Started</button>
              </Link>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
