"use client";

import { useCallback, useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import Navbar from "@/app/components/Navbar";
import Footer from "@/app/components/Footer";

type PromptItem = {
  id: string;
  title: string;
  provider: string;
  model: string;
  compressionLevel: string;
  originalTokens: number | null;
  optimizedTokens: number | null;
  reductionPercent: number | null;
  isFavorite: boolean;
  createdAt: string;
  originalText: string;
  optimizedText: string;
};

export default function LibraryPage() {
  const { status } = useSession();
  const router = useRouter();
  const [prompts, setPrompts] = useState<PromptItem[]>([]);
  const [favoritesOnly, setFavoritesOnly] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<PromptItem | null>(null);

  const loadPrompts = useCallback(async () => {
    setLoading(true);
    const response = await fetch(`/api/library?limit=50${favoritesOnly ? "&favorite=true" : ""}`);
    const data = await response.json();
    if (data.success) setPrompts(data.prompts);
    else setError(data.error?.message ?? "Could not load your library.");
    setLoading(false);
  }, [favoritesOnly]);

  useEffect(() => {
    if (status === "unauthenticated") router.replace("/login");
    if (status === "authenticated") {
      const timer = window.setTimeout(() => void loadPrompts(), 0);
      return () => window.clearTimeout(timer);
    }
  }, [status, router, loadPrompts]);

  async function updatePrompt(id: string, data: Record<string, unknown>) {
    const response = await fetch(`/api/library/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!response.ok) setError("Could not update this prompt.");
    else void loadPrompts();
  }

  async function deletePrompt(id: string) {
    if (!window.confirm("Delete this saved prompt? This cannot be undone.")) return;
    const response = await fetch(`/api/library/${id}`, { method: "DELETE" });
    if (!response.ok) setError("Could not delete this prompt.");
    else {
      setSelected(null);
      void loadPrompts();
    }
  }

  return (
    <div style={{ minHeight: "100dvh", display: "flex", flexDirection: "column" }}>
      <Navbar />
      <main className="container-app" style={{ flex: 1, paddingTop: 40, paddingBottom: 60 }}>
        <section style={{ display: "flex", justifyContent: "space-between", gap: 16, alignItems: "end", marginBottom: 28 }}>
          <div>
            <div className="chip" style={{ marginBottom: 10 }}>Private workspace</div>
            <h1 style={{ margin: 0, fontFamily: "var(--font-headline)", fontSize: "clamp(28px, 4vw, 42px)", color: "var(--color-on-background)" }}>Prompt Library</h1>
            <p style={{ margin: "8px 0 0", color: "var(--color-on-surface-variant)", fontFamily: "var(--font-body)" }}>Review, reuse, favorite, or remove your saved optimizations.</p>
          </div>
          <button className="btn-ghost" onClick={() => setFavoritesOnly((value) => !value)}>
            <span className="material-symbols-outlined" style={{ fontSize: 18 }}>{favoritesOnly ? "star" : "star_border"}</span>
            {favoritesOnly ? "Showing favorites" : "Favorites only"}
          </button>
        </section>

        {error && <p style={{ color: "var(--color-error)", marginBottom: 16 }}>{error}</p>}
        {loading || status === "loading" ? (
          <div className="glass-card" style={{ padding: 32, textAlign: "center", color: "var(--color-on-surface-variant)" }}>Loading your library…</div>
        ) : prompts.length === 0 ? (
          <div className="glass-card" style={{ padding: 40, textAlign: "center" }}>
            <span className="material-symbols-outlined" style={{ fontSize: 36, color: "var(--color-primary)" }}>auto_awesome</span>
            <h2 style={{ fontFamily: "var(--font-headline)", color: "var(--color-on-background)" }}>No saved prompts yet</h2>
            <p style={{ color: "var(--color-on-surface-variant)" }}>Run an optimization and it will appear here for future reference.</p>
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16 }}>
            {prompts.map((item) => (
              <article key={item.id} className="glass-card" style={{ padding: 20, display: "flex", flexDirection: "column", gap: 14 }}>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <div>
                    <h2 style={{ margin: 0, fontFamily: "var(--font-headline)", fontSize: 18, color: "var(--color-on-background)" }}>{item.title}</h2>
                    <p style={{ margin: "5px 0 0", color: "var(--color-on-surface-variant)", fontFamily: "var(--font-label)", fontSize: 11 }}>{item.provider} · {item.model}</p>
                  </div>
                  <button aria-label="Toggle favorite" onClick={() => void updatePrompt(item.id, { isFavorite: !item.isFavorite })} style={{ background: "none", border: 0, cursor: "pointer", color: item.isFavorite ? "var(--color-primary)" : "var(--color-on-surface-variant)" }}>
                    <span className="material-symbols-outlined">{item.isFavorite ? "star" : "star_border"}</span>
                  </button>
                </div>
                <p style={{ margin: 0, color: "var(--color-on-surface-variant)", fontSize: 13, lineHeight: 1.55, display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical", overflow: "hidden" }}>{item.optimizedText}</p>
                <div style={{ display: "flex", justifyContent: "space-between", color: "var(--color-on-surface-variant)", fontFamily: "var(--font-label)", fontSize: 11 }}>
                  <span>{item.compressionLevel} compression</span>
                  <span>{item.reductionPercent !== null ? `${item.reductionPercent}% reduced` : "—"}</span>
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <button className="btn-ghost" style={{ padding: "7px 10px", fontSize: 11 }} onClick={() => setSelected(item)}>View</button>
                  <button className="btn-ghost" style={{ padding: "7px 10px", fontSize: 11, color: "var(--color-error)" }} onClick={() => void deletePrompt(item.id)}>Delete</button>
                </div>
              </article>
            ))}
          </div>
        )}

        {selected && <div role="dialog" aria-modal="true" style={{ position: "fixed", inset: 0, zIndex: 100, background: "rgba(0,0,0,.78)", display: "grid", placeItems: "center", padding: 20 }} onClick={() => setSelected(null)}>
          <div className="glass-card" style={{ maxWidth: 820, width: "100%", maxHeight: "80vh", overflow: "auto", padding: 24 }} onClick={(event) => event.stopPropagation()}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 12, marginBottom: 18 }}><h2 style={{ margin: 0, fontFamily: "var(--font-headline)" }}>{selected.title}</h2><button className="btn-ghost" onClick={() => setSelected(null)}>Close</button></div>
            <p className="label-base">Original prompt</p><pre style={{ whiteSpace: "pre-wrap", fontFamily: "var(--font-code)", color: "var(--color-on-surface-variant)", fontSize: 13 }}>{selected.originalText}</pre>
            <p className="label-base" style={{ marginTop: 20 }}>Optimized prompt</p><pre style={{ whiteSpace: "pre-wrap", fontFamily: "var(--font-code)", color: "var(--color-on-surface)", fontSize: 13 }}>{selected.optimizedText}</pre>
          </div>
        </div>}
      </main>
      <Footer />
    </div>
  );
}
