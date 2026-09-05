"use client";
import { useState, useRef } from "react";
import Navbar from "@/app/components/Navbar";
import Footer from "@/app/components/Footer";
import { DEFAULT_MODELS } from "@/lib/llm-router";

const PROVIDERS = [
  { id: "google",    label: "Google Gemini", icon: "auto_awesome" },
  { id: "openai",    label: "OpenAI",        icon: "psychology" },
  { id: "anthropic", label: "Anthropic",     icon: "hub" },
] as const;

type Provider = "google" | "openai" | "anthropic";
type Level    = "light" | "balanced" | "aggressive";

interface OptimizeResult {
  optimizedPrompt: string;
  changes: string[];
  originalTokenEstimate: number;
  optimizedTokenEstimate: number;
  tokensReduced: number;
  reductionPercent: number;
}

const LEVEL_INFO = {
  light:      { label: "Light",      desc: "Minimal changes, preserves tone." },
  balanced:   { label: "Balanced",   desc: "Standard structure, clear sections." },
  aggressive: { label: "Aggressive", desc: "Maximum compression, 30-50%+ savings." },
};

export default function HomePage() {
  const [prompt,   setPrompt]   = useState("");
  const [provider, setProvider] = useState<Provider>("google");
  const [model,    setModel]    = useState(DEFAULT_MODELS.google[0]);
  const [level,    setLevel]    = useState<Level>("balanced");
  const [examples, setExamples] = useState(false);
  const [custom,   setCustom]   = useState("");
  const [result,   setResult]   = useState<OptimizeResult | null>(null);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState<string | null>(null);
  const [copied,   setCopied]   = useState(false);
  const resultRef = useRef<HTMLDivElement>(null);

  function handleProviderChange(p: Provider) {
    setProvider(p);
    setModel(DEFAULT_MODELS[p][0]);
  }

  async function handleOptimize() {
    if (!prompt.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch("/api/optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, provider, model, compressionLevel: level, addExamples: examples, customInstructions: custom || undefined }),
      });
      const data = await res.json();
      if (!data.success) {
        setError(data.error?.message ?? "Optimization failed.");
      } else {
        setResult(data.result);
        setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
      }
    } catch {
      setError("Network error. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  async function handleCopy() {
    if (!result) return;
    await navigator.clipboard.writeText(result.optimizedPrompt);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <div style={{ minHeight: "100dvh", display: "flex", flexDirection: "column" }}>
      <Navbar />

      <main className="container-app" style={{ flex: 1, width: "100%", maxWidth: 1200, margin: "0 auto", paddingTop: 40, paddingBottom: 60 }}>

        {/* ── Hero ── */}
        <section style={{ maxWidth: 720, marginBottom: 40 }} className="animate-fadein">
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
            <span className="chip">Tools</span>
            <span className="material-symbols-outlined" style={{ fontSize: 16, color: "var(--color-primary)" }}>chevron_right</span>
            <span className="chip">Prompt Optimizer</span>
          </div>
          <h1 style={{ fontFamily: "var(--font-headline)", fontSize: "clamp(28px, 5vw, 48px)", fontWeight: 700, letterSpacing: "-0.02em", textTransform: "uppercase", color: "var(--color-on-background)", margin: 0, lineHeight: 1.1 }}>
            Free Prompt Optimizer
          </h1>
          <p style={{ fontFamily: "var(--font-body)", fontSize: 16, color: "var(--color-on-surface-variant)", marginTop: 12, lineHeight: 1.6 }}>
            Improve your LLM prompts with AI. Reduce token costs, improve clarity, and add examples — using your own API keys, processed entirely server-side.
          </p>
        </section>

        {/* ── Main Grid ── */}
        <div className="workspace-grid" style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 340px", gap: 24, alignItems: "start", width: "100%" }}>

          {/* Left: Editor */}
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

            {/* Prompt textarea */}
            <div className="editor-zone glass-card" style={{ padding: 0, overflow: "hidden" }}>
              <div style={{ padding: "16px 20px 0", borderBottom: "1px solid var(--color-outline)" }}>
                <h2 style={{ fontFamily: "var(--font-headline)", fontSize: 22, fontWeight: 600, color: "var(--color-on-background)", margin: 0 }}>Your Prompt</h2>
              </div>
              <textarea
                id="prompt-input"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Paste or type your prompt here…"
                style={{
                  width: "100%", minHeight: 280, background: "transparent",
                  border: "none", outline: "none", resize: "vertical",
                  fontFamily: "var(--font-code)", fontSize: 14, lineHeight: 1.7,
                  color: "var(--color-on-surface)", padding: "16px 20px",
                }}
              />
              <div style={{ padding: "8px 20px 12px", display: "flex", justifyContent: "flex-end" }}>
                <span className="label-base" style={{ fontSize: 12 }}>{prompt.length} characters</span>
              </div>
            </div>

            {/* Optimize button */}
            <div>
              <button
                id="optimize-btn"
                className="btn-primary"
                onClick={handleOptimize}
                disabled={loading || !prompt.trim()}
                style={{ fontSize: 14, padding: "12px 28px", minWidth: 180 }}
              >
                {loading ? (
                  <>
                    <span className="material-symbols-outlined animate-spin-slow" style={{ fontSize: 18 }}>autorenew</span>
                    Optimizing…
                  </>
                ) : (
                  <>
                    <span className="material-symbols-outlined" style={{ fontSize: 18 }}>bolt</span>
                    Optimize Prompt
                  </>
                )}
              </button>
            </div>

            {/* Error */}
            {error && (
              <div className="animate-fadein" style={{ background: "rgba(255,180,171,0.08)", border: "1px solid var(--color-error)", borderRadius: "var(--radius-lg)", padding: "14px 18px", display: "flex", gap: 10, alignItems: "flex-start" }}>
                <span className="material-symbols-outlined" style={{ fontSize: 18, color: "var(--color-error)", flexShrink: 0 }}>error</span>
                <p style={{ fontFamily: "var(--font-body)", fontSize: 14, color: "var(--color-error)", margin: 0 }}>{error}</p>
              </div>
            )}

            {/* Result */}
            {result && (
              <div ref={resultRef} className="animate-fadein" style={{ display: "flex", flexDirection: "column", gap: 16 }}>

                {/* Stats row */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
                  {[
                    { icon: "token",       label: "Original Tokens",  value: result.originalTokenEstimate },
                    { icon: "compress",    label: "Optimized Tokens", value: result.optimizedTokenEstimate },
                    { icon: "trending_down", label: "Reduction",     value: `${result.reductionPercent}%`, green: true },
                  ].map((s) => (
                    <div key={s.label} className="glass-card" style={{ padding: "14px 16px", borderLeft: s.green ? "2px solid var(--color-primary)" : undefined }}>
                      <div className="label-base" style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
                        <span className="material-symbols-outlined" style={{ fontSize: 16 }}>{s.icon}</span>
                        {s.label}
                      </div>
                      <div style={{ fontFamily: "var(--font-headline)", fontSize: 24, fontWeight: 600, color: s.green ? "var(--color-primary)" : "var(--color-on-background)" }}>
                        {s.value}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Optimized prompt */}
                <div className="glass-card" style={{ padding: 0, overflow: "hidden" }}>
                  <div style={{ padding: "14px 20px", borderBottom: "1px solid var(--color-outline)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <h3 style={{ fontFamily: "var(--font-headline)", fontSize: 18, fontWeight: 600, color: "var(--color-primary)", margin: 0 }}>
                      ✦ Optimized Result
                    </h3>
                    <button className="btn-ghost" style={{ padding: "5px 12px", fontSize: 12 }} onClick={handleCopy}>
                      <span className="material-symbols-outlined" style={{ fontSize: 16 }}>{copied ? "check" : "content_copy"}</span>
                      {copied ? "Copied!" : "Copy"}
                    </button>
                  </div>
                  <pre style={{ margin: 0, padding: "16px 20px", fontFamily: "var(--font-code)", fontSize: 13, lineHeight: 1.7, color: "var(--color-on-surface)", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                    {result.optimizedPrompt}
                  </pre>
                </div>

                {/* Changes list */}
                {result.changes?.length > 0 && (
                  <div className="glass-card" style={{ padding: "16px 20px" }}>
                    <h4 className="label-base" style={{ marginBottom: 10 }}>Changes Made</h4>
                    <ul style={{ margin: 0, paddingLeft: 20, display: "flex", flexDirection: "column", gap: 6 }}>
                      {result.changes.map((c, i) => (
                        <li key={i} style={{ fontFamily: "var(--font-body)", fontSize: 13, color: "var(--color-on-surface-variant)" }}>{c}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Right: Settings sidebar */}
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

            {/* Provider */}
            <div className="glass-card" style={{ padding: "18px 20px" }}>
              <h3 className="label-base" style={{ marginBottom: 12 }}>AI Provider</h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {PROVIDERS.map((p) => (
                  <button
                    key={p.id}
                    onClick={() => handleProviderChange(p.id as Provider)}
                    style={{
                      background: provider === p.id ? "rgba(0,255,102,0.08)" : "transparent",
                      border: `1px solid ${provider === p.id ? "var(--color-primary)" : "var(--color-outline)"}`,
                      borderRadius: "var(--radius-lg)", padding: "10px 14px",
                      display: "flex", alignItems: "center", gap: 10, cursor: "pointer",
                      color: provider === p.id ? "var(--color-primary)" : "var(--color-on-surface-variant)",
                      fontFamily: "var(--font-label)", fontSize: 13, fontWeight: 500,
                      transition: "all 0.15s", textAlign: "left",
                    }}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: 18, fontVariationSettings: "'FILL' 1" }}>{p.icon}</span>
                    {p.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Model */}
            <div className="glass-card" style={{ padding: "18px 20px" }}>
              <label className="label-base" htmlFor="model-select" style={{ display: "block", marginBottom: 8 }}>Model</label>
              <select
                id="model-select"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="input-base"
                style={{ cursor: "pointer", appearance: "none" }}
              >
                {DEFAULT_MODELS[provider].map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>

            {/* Compression Level */}
            <div className="glass-card" style={{ padding: "18px 20px" }}>
              <h3 className="label-base" style={{ marginBottom: 12 }}>Compression Level</h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {(["light", "balanced", "aggressive"] as Level[]).map((l) => (
                  <button
                    key={l}
                    onClick={() => setLevel(l)}
                    style={{
                      background: level === l ? "rgba(0,255,102,0.06)" : "transparent",
                      border: `1px solid ${level === l ? "var(--color-primary)" : "var(--color-outline)"}`,
                      borderRadius: "var(--radius-lg)", padding: "10px 14px",
                      cursor: "pointer", textAlign: "left", transition: "all 0.15s",
                    }}
                  >
                    <div style={{ fontFamily: "var(--font-label)", fontSize: 12, fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase", color: level === l ? "var(--color-primary)" : "var(--color-on-surface)" }}>
                      {LEVEL_INFO[l].label}
                    </div>
                    <div style={{ fontFamily: "var(--font-body)", fontSize: 12, color: "var(--color-on-surface-variant)", marginTop: 2 }}>
                      {LEVEL_INFO[l].desc}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Options */}
            <div className="glass-card" style={{ padding: "18px 20px", display: "flex", flexDirection: "column", gap: 14 }}>
              <h3 className="label-base">Options</h3>

              {/* Examples toggle */}
              <label style={{ display: "flex", alignItems: "flex-start", gap: 10, cursor: "pointer" }}>
                <input type="checkbox" checked={examples} onChange={(e) => setExamples(e.target.checked)} style={{ marginTop: 2, flexShrink: 0 }} />
                <div>
                  <div style={{ fontFamily: "var(--font-label)", fontSize: 13, fontWeight: 600, color: "var(--color-on-surface)" }}>Add Few-Shot Examples</div>
                  <div style={{ fontFamily: "var(--font-body)", fontSize: 12, color: "var(--color-on-surface-variant)", marginTop: 2 }}>Include 1-2 input/output examples in the optimized prompt.</div>
                </div>
              </label>

              {/* Custom instructions */}
              <div>
                <label className="label-base" htmlFor="custom-instructions" style={{ display: "block", marginBottom: 6 }}>Custom Instructions</label>
                <textarea
                  id="custom-instructions"
                  value={custom}
                  onChange={(e) => setCustom(e.target.value)}
                  placeholder="Any additional requirements…"
                  rows={3}
                  style={{ width: "100%", background: "var(--color-surface)", border: "1px solid var(--color-outline)", borderRadius: "var(--radius-md)", padding: "10px 12px", fontFamily: "var(--font-body)", fontSize: 13, color: "var(--color-on-surface)", resize: "vertical", outline: "none" }}
                  onFocus={(e) => { e.target.style.borderColor = "var(--color-primary)"; }}
                  onBlur={(e) => { e.target.style.borderColor = "var(--color-outline)"; }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* FAQ */}
        <section style={{ marginTop: 60, borderTop: "1px solid var(--color-outline)" }}>
          {[
            { q: "What is the PromptBud Prompt Optimizer?", a: "PromptBud AI uses a master system prompt to rewrite your prompts for better clarity, fewer tokens, and improved LLM performance. You bring your own API keys — we never store them in plaintext." },
            { q: "How do I connect my API key?", a: "Go to your Action Center (profile icon) and enter your Gemini, OpenAI, or Anthropic API key. Keys are encrypted with AES-256-GCM before storage. The plaintext is never returned to your browser." },
            { q: "Is my prompt stored?", a: "Optimized prompts are saved to your private library so you can review them later. You can delete any entry at any time from the Library page." },
          ].map((item) => (
            <details key={item.q} style={{ borderBottom: "1px solid var(--color-outline)" }}>
              <summary style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "20px 4px", cursor: "pointer", listStyle: "none", fontFamily: "var(--font-headline)", fontSize: 18, fontWeight: 600, color: "var(--color-on-background)" }}>
                {item.q}
                <span className="material-symbols-outlined" style={{ color: "var(--color-on-surface-variant)" }}>expand_more</span>
              </summary>
              <p style={{ fontFamily: "var(--font-body)", fontSize: 15, color: "var(--color-on-surface-variant)", padding: "0 4px 20px", margin: 0, lineHeight: 1.7 }}>{item.a}</p>
            </details>
          ))}
        </section>
      </main>

      <Footer />
    </div>
  );
}
