"use client";
import { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import Navbar from "@/app/components/Navbar";
import Footer from "@/app/components/Footer";
import { SUPPORTED_PROVIDERS, DEFAULT_MODELS } from "@/lib/llm-router";

type Provider = "google" | "openai" | "anthropic";
type Level    = "light" | "balanced" | "aggressive";

interface Prefs {
  defaultProvider:     Provider;
  defaultModel:        string;
  defaultCompression:  Level;
  addExamples:         boolean;
  autoSaveLibrary:     boolean;
  geminiApiKeySaved:   boolean;
  openaiApiKeySaved:   boolean;
  anthropicApiKeySaved: boolean;
  geminiApiKeyMask:    string | null;
  openaiApiKeyMask:    string | null;
  anthropicApiKeyMask: string | null;
}

const PROVIDER_META = {
  google:    { label: "Google Gemini",  icon: "auto_awesome",  color: "#00ff66" },
  openai:    { label: "OpenAI",         icon: "psychology",    color: "#10a37f" },
  anthropic: { label: "Anthropic",      icon: "hub",           color: "#cc785c" },
};

export default function ProfilePage() {
  const { data: session, status } = useSession();
  const router = useRouter();

  const [prefs,   setPrefs]   = useState<Prefs | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving,  setSaving]  = useState(false);
  const [saved,   setSaved]   = useState(false);
  const [error,   setError]   = useState<string | null>(null);
  const [accountName, setAccountName] = useState("");
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [accountSaving, setAccountSaving] = useState(false);
  const [accountMessage, setAccountMessage] = useState<string | null>(null);

  // API key inputs — plaintext only while typing, cleared on load
  const [keys, setKeys] = useState({ google: "", openai: "", anthropic: "" });
  const [showKey, setShowKey] = useState({ google: false, openai: false, anthropic: false });

  useEffect(() => {
    if (status === "unauthenticated") { router.push("/login"); return; }
    if (status !== "authenticated") return;
    const accountTimer = window.setTimeout(() => setAccountName(session.user?.name ?? ""), 0);
    fetch("/api/preferences")
      .then((r) => r.json())
      .then((d) => { if (d.success) setPrefs(d.preferences); })
      .finally(() => setLoading(false));
    return () => window.clearTimeout(accountTimer);
  }, [status, router, session?.user?.name]);

  async function handleAccountSave() {
    setAccountSaving(true);
    setAccountMessage(null);
    try {
      const res = await fetch("/api/account", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: accountName, currentPassword, newPassword }),
      });
      const data = await res.json();
      if (!data.success) setAccountMessage(data.error?.message ?? "Could not update your account.");
      else {
        setAccountName(data.user.name);
        setCurrentPassword("");
        setNewPassword("");
        setAccountMessage("Account details saved.");
      }
    } catch {
      setAccountMessage("Network error. Please try again.");
    } finally {
      setAccountSaving(false);
    }
  }

  async function handleSave() {
    if (!prefs) return;
    setSaving(true); setError(null);
    const body: Record<string, unknown> = {
      defaultProvider:    prefs.defaultProvider,
      defaultModel:       prefs.defaultModel,
      defaultCompression: prefs.defaultCompression,
      addExamples:        prefs.addExamples,
      autoSaveLibrary:    prefs.autoSaveLibrary,
    };
    // Only send keys that were actually typed
    if (keys.google)    body.geminiApiKey    = keys.google;
    if (keys.openai)    body.openaiApiKey    = keys.openai;
    if (keys.anthropic) body.anthropicApiKey = keys.anthropic;

    const res = await fetch("/api/preferences", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json();
    setSaving(false);
    if (!data.success) { setError(data.error?.message ?? "Save failed."); return; }
    // Refresh prefs to get updated masks
    const fresh = await fetch("/api/preferences").then((r) => r.json());
    if (fresh.success) setPrefs(fresh.preferences);
    setKeys({ google: "", openai: "", anthropic: "" });
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  }

  async function handleDeleteKey(provider: Provider) {
    const fieldMap: Record<Provider, string> = { google: "geminiApiKey", openai: "openaiApiKey", anthropic: "anthropicApiKey" };
    await fetch("/api/preferences", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ [fieldMap[provider]]: "" }) });
    const fresh = await fetch("/api/preferences").then((r) => r.json());
    if (fresh.success) setPrefs(fresh.preferences);
  }

  if (loading || status === "loading") {
    return (
      <div style={{ minHeight: "100dvh", display: "flex", flexDirection: "column" }}>
        <Navbar />
        <main style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <span className="material-symbols-outlined animate-spin-slow" style={{ fontSize: 32, color: "var(--color-primary)" }}>autorenew</span>
        </main>
      </div>
    );
  }

  return (
    <div style={{ minHeight: "100dvh", display: "flex", flexDirection: "column" }}>
      <Navbar />
      <main className="container-app" style={{ flex: 1, paddingTop: 40, paddingBottom: 60 }}>

        {/* Header */}
        <section style={{ marginBottom: 36 }} className="animate-fadein">
          <h1 style={{ fontFamily: "var(--font-headline)", fontSize: "clamp(24px, 4vw, 40px)", fontWeight: 700, letterSpacing: "-0.02em", color: "var(--color-on-background)", margin: 0 }}>
            Action Center
          </h1>
          <p style={{ fontFamily: "var(--font-body)", fontSize: 15, color: "var(--color-on-surface-variant)", marginTop: 8 }}>
            Manage your AI provider keys, optimization defaults, and account preferences. Keys are encrypted with AES-256-GCM — never returned in plaintext.
          </p>
        </section>

        <div className="profile-grid" style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 24, alignItems: "start" }}>

          {/* Left: API Keys */}
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div className="glass-card" style={{ padding: "24px 28px" }}>
              <h2 style={{ fontFamily: "var(--font-headline)", fontSize: 18, fontWeight: 600, color: "var(--color-on-background)", margin: "0 0 6px" }}>
                API Keys
              </h2>
              <p style={{ fontFamily: "var(--font-body)", fontSize: 13, color: "var(--color-on-surface-variant)", margin: "0 0 24px" }}>
                Provide your own API keys. Each key is encrypted before storage. To add more providers in future, extend the schema and router.
              </p>

              <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                {(SUPPORTED_PROVIDERS as unknown as Provider[]).map((p) => {
                  const meta  = PROVIDER_META[p];
                  const saved_key = p === "google" ? prefs?.geminiApiKeySaved : p === "openai" ? prefs?.openaiApiKeySaved : prefs?.anthropicApiKeySaved;
                  const mask  = p === "google" ? prefs?.geminiApiKeyMask : p === "openai" ? prefs?.openaiApiKeyMask : prefs?.anthropicApiKeyMask;

                  return (
                    <div key={p} style={{ borderBottom: "1px solid var(--color-outline)", paddingBottom: 20 }}>
                      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                          <span className="material-symbols-outlined" style={{ fontSize: 20, color: meta.color, fontVariationSettings: "'FILL' 1" }}>{meta.icon}</span>
                          <span style={{ fontFamily: "var(--font-label)", fontSize: 13, fontWeight: 600, color: "var(--color-on-surface)" }}>{meta.label}</span>
                          {saved_key && <span className="chip" style={{ fontSize: 10 }}>Saved</span>}
                        </div>
                        {saved_key && (
                          <button onClick={() => handleDeleteKey(p)} className="btn-ghost" style={{ padding: "4px 10px", fontSize: 12 }}>
                            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>delete</span>
                            Remove
                          </button>
                        )}
                      </div>
                      {saved_key && mask && (
                        <p style={{ fontFamily: "var(--font-code)", fontSize: 13, color: "var(--color-on-surface-variant)", margin: "0 0 8px", letterSpacing: "0.03em" }}>{mask}</p>
                      )}
                      <div style={{ position: "relative" }}>
                        <input
                          type={showKey[p] ? "text" : "password"}
                          placeholder={saved_key ? "Enter new key to replace…" : "Paste your API key here…"}
                          value={keys[p]}
                          onChange={(e) => setKeys(k => ({ ...k, [p]: e.target.value }))}
                          className="input-base"
                          style={{ paddingRight: 42 }}
                        />
                        <button
                          type="button"
                          onClick={() => setShowKey(s => ({ ...s, [p]: !s[p] }))}
                          style={{ position: "absolute", right: 12, top: "50%", transform: "translateY(-50%)", background: "none", border: "none", cursor: "pointer", color: "var(--color-on-surface-variant)", padding: 0 }}
                        >
                          <span className="material-symbols-outlined" style={{ fontSize: 18 }}>{showKey[p] ? "visibility_off" : "visibility"}</span>
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Default Settings */}
            <div className="glass-card" style={{ padding: "24px 28px" }}>
              <h2 style={{ fontFamily: "var(--font-headline)", fontSize: 18, fontWeight: 600, color: "var(--color-on-background)", margin: "0 0 20px" }}>Optimization Defaults</h2>

              <div className="settings-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
                <div>
                  <label className="label-base" htmlFor="default-provider" style={{ display: "block", marginBottom: 6 }}>Default Provider</label>
                  <select id="default-provider" value={prefs?.defaultProvider ?? "google"} onChange={(e) => setPrefs(p => p ? { ...p, defaultProvider: e.target.value as Provider, defaultModel: DEFAULT_MODELS[e.target.value as Provider][0] } : p)} className="input-base" style={{ cursor: "pointer" }}>
                    {SUPPORTED_PROVIDERS.map((p) => <option key={p} value={p}>{PROVIDER_META[p as Provider].label}</option>)}
                  </select>
                </div>
                <div>
                  <label className="label-base" htmlFor="default-model" style={{ display: "block", marginBottom: 6 }}>Default Model</label>
                  <select id="default-model" value={prefs?.defaultModel ?? ""} onChange={(e) => setPrefs(p => p ? { ...p, defaultModel: e.target.value } : p)} className="input-base" style={{ cursor: "pointer" }}>
                    {DEFAULT_MODELS[prefs?.defaultProvider ?? "google"].map((m) => <option key={m} value={m}>{m}</option>)}
                  </select>
                </div>
              </div>

              <div>
                <label className="label-base" htmlFor="default-compression" style={{ display: "block", marginBottom: 6 }}>Default Compression</label>
                <select id="default-compression" value={prefs?.defaultCompression ?? "balanced"} onChange={(e) => setPrefs(p => p ? { ...p, defaultCompression: e.target.value as Level } : p)} className="input-base" style={{ cursor: "pointer" }}>
                  <option value="light">Light — Minimal changes</option>
                  <option value="balanced">Balanced — Standard structure</option>
                  <option value="aggressive">Aggressive — Maximum compression</option>
                </select>
              </div>
            </div>

            {/* Save button */}
            <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
              <button className="btn-primary" onClick={handleSave} disabled={saving} style={{ padding: "12px 28px" }}>
                {saving ? <span className="material-symbols-outlined animate-spin-slow" style={{ fontSize: 18 }}>autorenew</span> : <span className="material-symbols-outlined" style={{ fontSize: 18 }}>save</span>}
                {saving ? "Saving…" : "Save Changes"}
              </button>
              {saved  && <span style={{ fontFamily: "var(--font-label)", fontSize: 13, color: "var(--color-primary)" }} className="animate-fadein">✓ Saved successfully</span>}
              {error  && <span style={{ fontFamily: "var(--font-label)", fontSize: 13, color: "var(--color-error)"   }} className="animate-fadein">{error}</span>}
            </div>
          </div>

          {/* Right: Stats card */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div className="glass-card" style={{ padding: "22px 24px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
                <div style={{ width: 52, height: 52, borderRadius: "50%", background: "var(--color-surface-high)", display: "flex", alignItems: "center", justifyContent: "center", border: "1px solid var(--color-outline)" }}>
                  <span className="material-symbols-outlined" style={{ fontSize: 28, color: "var(--color-on-surface-variant)" }}>person</span>
                </div>
                <div>
                  <div style={{ fontFamily: "var(--font-headline)", fontSize: 18, fontWeight: 600, color: "var(--color-on-background)" }}>{accountName || session?.user?.name}</div>
                  <div style={{ fontFamily: "var(--font-label)", fontSize: 11, color: "var(--color-primary)", letterSpacing: "0.06em", textTransform: "uppercase" }}>Free Tier</div>
                </div>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
                {[
                  { label: "Email",   value: session?.user?.email ?? "—" },
                ].map(({ label, value }) => (
                  <div key={label} style={{ display: "flex", justifyContent: "space-between", padding: "10px 0", borderBottom: "1px solid var(--color-outline)" }}>
                    <span style={{ fontFamily: "var(--font-body)", fontSize: 13, color: "var(--color-on-surface-variant)" }}>{label}</span>
                    <span style={{ fontFamily: "var(--font-code)", fontSize: 13, color: "var(--color-on-background)" }}>{value}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="glass-card" style={{ padding: "22px 24px", display: "flex", flexDirection: "column", gap: 12 }}>
              <h3 style={{ fontFamily: "var(--font-headline)", fontSize: 16, fontWeight: 600, color: "var(--color-on-background)", margin: 0 }}>Account</h3>
              <div>
                <label className="label-base" htmlFor="account-name" style={{ display: "block", marginBottom: 6 }}>Display name</label>
                <input id="account-name" className="input-base" value={accountName} onChange={(event) => setAccountName(event.target.value)} maxLength={80} />
              </div>
              <div>
                <label className="label-base" htmlFor="current-password" style={{ display: "block", marginBottom: 6 }}>Current password</label>
                <input id="current-password" type="password" className="input-base" value={currentPassword} onChange={(event) => setCurrentPassword(event.target.value)} placeholder="Required only to change password" />
              </div>
              <div>
                <label className="label-base" htmlFor="new-password" style={{ display: "block", marginBottom: 6 }}>New password</label>
                <input id="new-password" type="password" className="input-base" value={newPassword} onChange={(event) => setNewPassword(event.target.value)} minLength={8} placeholder="At least 8 characters" />
              </div>
              <button className="btn-ghost" onClick={handleAccountSave} disabled={accountSaving} style={{ justifyContent: "center" }}>{accountSaving ? "Saving…" : "Save account"}</button>
              {accountMessage && <p style={{ margin: 0, color: accountMessage === "Account details saved." ? "var(--color-primary)" : "var(--color-error)", fontSize: 12 }}>{accountMessage}</p>}
            </div>

            {/* Preferences toggles */}
            <div className="glass-card" style={{ padding: "22px 24px", display: "flex", flexDirection: "column", gap: 14 }}>
              <h3 style={{ fontFamily: "var(--font-headline)", fontSize: 16, fontWeight: 600, color: "var(--color-on-background)", margin: 0 }}>Preferences</h3>
              {[
                { key: "addExamples",    label: "Add Examples by Default",  desc: "Include few-shot examples in all optimized prompts." },
                { key: "autoSaveLibrary", label: "Auto-Save to Library",    desc: "Automatically save all optimized prompts to your library." },
              ].map(({ key, label, desc }) => (
                <div key={key} style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, padding: "12px 14px", background: "var(--color-surface-high)", borderRadius: "var(--radius-lg)" }}>
                  <div>
                    <div style={{ fontFamily: "var(--font-label)", fontSize: 13, fontWeight: 600, color: "var(--color-on-background)" }}>{label}</div>
                    <div style={{ fontFamily: "var(--font-body)", fontSize: 12, color: "var(--color-on-surface-variant)", marginTop: 2 }}>{desc}</div>
                  </div>
                  <div
                    className="toggle-track"
                    data-checked={String(!!prefs?.[key as keyof Prefs])}
                    onClick={() => setPrefs(p => p ? { ...p, [key]: !p[key as keyof Prefs] } : p)}
                    role="switch"
                    aria-checked={!!prefs?.[key as keyof Prefs]}
                    style={{ flexShrink: 0 }}
                  >
                    <div className="toggle-thumb" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}
