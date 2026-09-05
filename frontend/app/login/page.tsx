"use client";
import { useState, Suspense } from "react";
import Link from "next/link";
import { signIn } from "next-auth/react";
import { useRouter, useSearchParams } from "next/navigation";

function AuthForm() {
  const router       = useRouter();
  const searchParams = useSearchParams();
  const initialTab   = searchParams.get("tab") === "signup" ? "signup" : "login";

  const [tab,     setTab]     = useState<"login" | "signup">(initialTab);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  const [loginForm,  setLoginForm]  = useState({ email: "", password: "" });
  const [signupForm, setSignupForm] = useState({ name: "", email: "", password: "", terms: false });

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true); setError(null);
    const res = await signIn("credentials", { redirect: false, email: loginForm.email, password: loginForm.password });
    setLoading(false);
    if (res?.error) { setError("Invalid email or password."); return; }
    router.push("/");
  }

  async function handleSignup(e: React.FormEvent) {
    e.preventDefault();
    if (!signupForm.terms) { setError("Please accept the Terms of Service."); return; }
    setLoading(true); setError(null);
    const res = await fetch("/api/auth/register", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: signupForm.name, email: signupForm.email, password: signupForm.password }),
    });
    const data = await res.json();
    if (!data.success) { setError(data.error?.message ?? "Registration failed."); setLoading(false); return; }
    // Auto sign-in after register
    await signIn("credentials", { redirect: false, email: signupForm.email, password: signupForm.password });
    setLoading(false);
    router.push("/");
  }

  const inputStyle: React.CSSProperties = {
    width: "100%", background: "var(--color-surface)", border: "1px solid var(--color-outline)",
    borderRadius: "var(--radius-md)", padding: "12px 16px 12px 42px",
    fontFamily: "var(--font-body)", fontSize: 14, color: "var(--color-on-surface)", outline: "none",
  };

  return (
    <div style={{ minHeight: "100dvh", display: "flex", flexDirection: "column", background: "var(--color-background)" }}>
      {/* Minimal Header */}
      <header style={{ display: "flex", justifyContent: "center", padding: "32px 0" }}>
        <Link href="/" style={{ fontFamily: "var(--font-label)", fontWeight: 900, fontSize: 15, letterSpacing: "-0.02em", color: "var(--color-on-background)", textDecoration: "none", textTransform: "uppercase", display: "flex", alignItems: "center", gap: 8 }}>
          <span className="material-symbols-outlined" style={{ color: "var(--color-primary)", fontSize: 22, fontVariationSettings: "'FILL' 1" }}>auto_awesome</span>
          PromptBud AI
        </Link>
      </header>

      <main style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: "0 20px 40px" }}>
        <div className="glass-card animate-fadein" style={{ width: "100%", maxWidth: 420, padding: "0", overflow: "hidden", position: "relative" }}>
          {/* Green top accent */}
          <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 1, background: "linear-gradient(to right, transparent, var(--color-primary), transparent)", opacity: 0.6 }} />

          {/* Tabs */}
          <div style={{ display: "flex", borderBottom: "1px solid var(--color-outline)" }}>
            {(["login", "signup"] as const).map((t) => (
              <button key={t} onClick={() => { setTab(t); setError(null); }}
                style={{
                  flex: 1, padding: "16px 0", fontFamily: "var(--font-label)", fontSize: 13, fontWeight: 600,
                  letterSpacing: "0.08em", textTransform: "uppercase", background: "transparent", border: "none",
                  borderBottom: `2px solid ${tab === t ? "var(--color-primary)" : "transparent"}`,
                  color: tab === t ? "var(--color-primary)" : "var(--color-on-surface-variant)",
                  cursor: "pointer", transition: "color 0.15s",
                }}
              >
                {t === "login" ? "Log In" : "Sign Up"}
              </button>
            ))}
          </div>

          <div style={{ padding: "28px 28px 32px" }}>
            {tab === "login" ? (
              <>
                <h1 style={{ fontFamily: "var(--font-headline)", fontSize: 24, fontWeight: 600, color: "var(--color-on-background)", marginBottom: 6 }}>Welcome Back</h1>
                <p style={{ fontFamily: "var(--font-body)", fontSize: 14, color: "var(--color-on-surface-variant)", marginBottom: 24 }}>Enter your credentials to access your workspace.</p>
                <form onSubmit={handleLogin} style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                  <div style={{ position: "relative" }}>
                    <span className="material-symbols-outlined" style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", fontSize: 18, color: "var(--color-on-surface-variant)" }}>mail</span>
                    <input id="login-email" type="email" required placeholder="name@company.com" value={loginForm.email} onChange={(e) => setLoginForm(f => ({ ...f, email: e.target.value }))} style={inputStyle}
                      onFocus={(e) => { e.target.style.borderColor = "var(--color-primary)"; }} onBlur={(e) => { e.target.style.borderColor = "var(--color-outline)"; }} />
                  </div>
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                      <span className="label-base">Password</span>
                      <a href="#" style={{ fontFamily: "var(--font-label)", fontSize: 12, color: "var(--color-primary)", textDecoration: "none" }}>Forgot?</a>
                    </div>
                    <div style={{ position: "relative" }}>
                      <span className="material-symbols-outlined" style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", fontSize: 18, color: "var(--color-on-surface-variant)" }}>lock</span>
                      <input id="login-password" type="password" required placeholder="••••••••" value={loginForm.password} onChange={(e) => setLoginForm(f => ({ ...f, password: e.target.value }))} style={inputStyle}
                        onFocus={(e) => { e.target.style.borderColor = "var(--color-primary)"; }} onBlur={(e) => { e.target.style.borderColor = "var(--color-outline)"; }} />
                    </div>
                  </div>
                  {error && <p style={{ fontFamily: "var(--font-body)", fontSize: 13, color: "var(--color-error)", margin: 0 }}>{error}</p>}
                  <button type="submit" className="btn-primary" disabled={loading} style={{ width: "100%", justifyContent: "center", padding: "13px 0", fontSize: 13, letterSpacing: "0.08em", textTransform: "uppercase", marginTop: 4 }}>
                    {loading ? <span className="material-symbols-outlined animate-spin-slow" style={{ fontSize: 18 }}>autorenew</span> : null}
                    {loading ? "Signing in…" : "Log In"}
                  </button>
                </form>
              </>
            ) : (
              <>
                <h1 style={{ fontFamily: "var(--font-headline)", fontSize: 24, fontWeight: 600, color: "var(--color-on-background)", marginBottom: 6 }}>Join PromptBud AI</h1>
                <p style={{ fontFamily: "var(--font-body)", fontSize: 14, color: "var(--color-on-surface-variant)", marginBottom: 24 }}>Create an account to start optimizing your prompts.</p>
                <form onSubmit={handleSignup} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                  {[
                    { id: "signup-name",     type: "text",     icon: "person",  placeholder: "Jane Doe",           field: "name",  label: "Full Name" },
                    { id: "signup-email",    type: "email",    icon: "mail",    placeholder: "name@company.com",   field: "email", label: "Email" },
                    { id: "signup-password", type: "password", icon: "lock",    placeholder: "Min. 8 characters",  field: "password", label: "Password" },
                  ].map(({ id, type, icon, placeholder, field, label }) => (
                    <div key={id}>
                      <label className="label-base" htmlFor={id} style={{ display: "block", marginBottom: 6 }}>{label}</label>
                      <div style={{ position: "relative" }}>
                        <span className="material-symbols-outlined" style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", fontSize: 18, color: "var(--color-on-surface-variant)" }}>{icon}</span>
                        <input id={id} type={type} required placeholder={placeholder}
                          value={(signupForm as Record<string, unknown>)[field] as string}
                          onChange={(e) => setSignupForm(f => ({ ...f, [field]: e.target.value }))}
                          style={inputStyle}
                          onFocus={(e) => { e.target.style.borderColor = "var(--color-primary)"; }} onBlur={(e) => { e.target.style.borderColor = "var(--color-outline)"; }} />
                      </div>
                    </div>
                  ))}
                  <label style={{ display: "flex", alignItems: "flex-start", gap: 10, cursor: "pointer" }}>
                    <input type="checkbox" id="terms" checked={signupForm.terms} onChange={(e) => setSignupForm(f => ({ ...f, terms: e.target.checked }))} style={{ marginTop: 3, flexShrink: 0 }} />
                    <span style={{ fontFamily: "var(--font-body)", fontSize: 13, color: "var(--color-on-surface-variant)", lineHeight: 1.5 }}>
                      I agree to the <a href="#" style={{ color: "var(--color-primary)", textDecoration: "none" }}>Terms of Service</a> and <a href="#" style={{ color: "var(--color-primary)", textDecoration: "none" }}>Privacy Policy</a>.
                    </span>
                  </label>
                  {error && <p style={{ fontFamily: "var(--font-body)", fontSize: 13, color: "var(--color-error)", margin: 0 }}>{error}</p>}
                  <button type="submit" className="btn-primary" disabled={loading} style={{ width: "100%", justifyContent: "center", padding: "13px 0", fontSize: 13, letterSpacing: "0.08em", textTransform: "uppercase", marginTop: 4 }}>
                    {loading ? <span className="material-symbols-outlined animate-spin-slow" style={{ fontSize: 18 }}>autorenew</span> : null}
                    {loading ? "Creating…" : "Create Account"}
                  </button>
                </form>
              </>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default function LoginPage() {
  return <Suspense><AuthForm /></Suspense>;
}
