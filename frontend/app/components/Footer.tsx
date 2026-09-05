export default function Footer() {
  return (
    <footer style={{ borderTop: "1px solid var(--color-outline)", background: "var(--color-surface-low)", marginTop: "auto" }}>
      <div
        className="container-app"
        style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 16, padding: "20px var(--spacing-margin-desktop)" }}
      >
        <span style={{ fontFamily: "var(--font-label)", fontWeight: 700, fontSize: 13, color: "var(--color-on-background)", textTransform: "uppercase" }}>
          PromptBud AI
        </span>
        <p style={{ fontFamily: "var(--font-body)", fontSize: 13, color: "var(--color-on-surface-variant)" }}>
          © 2025 PromptBud AI. Precision engineering for the AI era.
        </p>
        <nav style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
          {["Privacy Policy", "Terms of Service", "API Status", "Contact Support"].map((l) => (
            <a key={l} href="#" style={{ fontFamily: "var(--font-label)", fontSize: 12, letterSpacing: "0.04em", color: "var(--color-on-surface-variant)", textDecoration: "none", transition: "color 0.15s" }}
              onMouseEnter={(e) => (e.currentTarget.style.color = "var(--color-primary)")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "var(--color-on-surface-variant)")}
            >
              {l}
            </a>
          ))}
        </nav>
      </div>
    </footer>
  );
}
