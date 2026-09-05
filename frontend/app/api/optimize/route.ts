/**
 * POST /api/optimize
 *
 * Flow:
 *   Browser sends: { prompt, provider, model, compressionLevel, addExamples, customInstructions }
 *   Server:
 *     1. Authenticates user via session
 *     2. Fetches encrypted API key from DB → decrypts server-side only
 *     3. Builds the master system prompt
 *     4. Calls selected LLM via llm-router
 *     5. Parses JSON response from master prompt
 *     6. Optionally saves to Prompt library
 *     7. Returns structured result to browser
 *
 *   SECURITY: API key never touches the browser at any step.
 */
import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { decryptApiKey } from "@/lib/crypto";
import { callLLM, Provider } from "@/lib/llm-router";
import { buildMasterPrompt, MASTER_PROMPT_VERSION } from "@/lib/master-prompt";

function structuredError(code: string, message: string, status = 400) {
  return NextResponse.json({ success: false, error: { code, message } }, { status });
}

export async function POST(req: NextRequest) {
  const session = await getServerSession(authOptions);
  if (!session?.user?.id) return structuredError("UNAUTHORIZED", "Not authenticated.", 401);

  let body: {
    prompt: string;
    provider: Provider;
    model: string;
    compressionLevel: "light" | "balanced" | "aggressive";
    addExamples: boolean;
    customInstructions?: string;
    saveToLibrary?: boolean;
    title?: string;
  };

  try {
    body = await req.json();
  } catch {
    return structuredError("INVALID_REQUEST", "Request body must be valid JSON.");
  }

  const {
    prompt, provider, model,
    compressionLevel = "balanced",
    addExamples = false,
    customInstructions,
    saveToLibrary = true,
    title = "Untitled Prompt",
  } = body;

  if (!prompt?.trim()) return structuredError("INVALID_PROMPT", "Prompt cannot be empty.");
  if (!provider)        return structuredError("MISSING_FIELD",  "Provider is required.");
  if (!model)           return structuredError("MISSING_FIELD",  "Model is required.");

  // ── Fetch preferences & decrypt key ─────────────────────────────────────
  const prefs = await prisma.preferences.findUnique({ where: { userId: session.user.id } });
  if (!prefs) return structuredError("NO_PREFERENCES", "User preferences not found.", 404);

  const keyMap: Record<Provider, string | null> = {
    google:    prefs.geminiApiKey    ?? null,
    openai:    prefs.openaiApiKey    ?? null,
    anthropic: prefs.anthropicApiKey ?? null,
  };
  const encryptedKey = keyMap[provider];
  if (!encryptedKey) {
    return structuredError("MISSING_API_KEY", `No API key saved for provider "${provider}". Add it in your Action Center.`, 422);
  }

  let apiKey: string;
  try {
    apiKey = decryptApiKey(encryptedKey);
  } catch {
    return structuredError("KEY_DECRYPT_ERROR", "Failed to decrypt the stored API key. Please re-save it in your Action Center.", 500);
  }

  // ── Build master prompt ──────────────────────────────────────────────────
  const systemPrompt = buildMasterPrompt({ compressionLevel, addExamples, customInstructions });

  // ── Call LLM ────────────────────────────────────────────────────────────
  let rawText: string;
  try {
    const result = await callLLM({ provider, model, apiKey, systemPrompt, userPrompt: prompt });
    rawText = result.rawText;
  } catch (err: unknown) {
    const e = err as { code?: string; message?: string };
    return structuredError(e.code ?? "PROVIDER_ERROR", e.message ?? "LLM call failed.", 502);
  }

  // ── Parse JSON response ──────────────────────────────────────────────────
  let parsed: {
    optimizedPrompt: string;
    changes: string[];
    compressionLevel: string;
    originalTokenEstimate: number;
    optimizedTokenEstimate: number;
    tokensReduced: number;
    reductionPercent: number;
  };

  try {
    // Strip any accidental markdown fences just in case
    const clean = rawText.replace(/^```json\n?|```$/g, "").trim();
    parsed = JSON.parse(clean);
  } catch {
    return structuredError("PARSE_ERROR", "LLM returned an unexpected format. Please try again.", 502);
  }

  // ── Optionally save to library ───────────────────────────────────────────
  let savedPromptId: string | null = null;
  if (saveToLibrary || prefs.autoSaveLibrary) {
    const saved = await prisma.prompt.create({
      data: {
        userId:            session.user.id,
        title,
        originalText:      prompt,
        optimizedText:     parsed.optimizedPrompt,
        masterPromptUsed:  MASTER_PROMPT_VERSION,
        provider,
        model,
        compressionLevel,
        examplesEnabled:   addExamples,
        customInstructions,
        originalTokens:    parsed.originalTokenEstimate,
        optimizedTokens:   parsed.optimizedTokenEstimate,
        tokensReduced:     parsed.tokensReduced,
        reductionPercent:  parsed.reductionPercent,
      },
    });
    savedPromptId = saved.id;
  }

  return NextResponse.json({
    success: true,
    result: {
      optimizedPrompt:        parsed.optimizedPrompt,
      changes:                parsed.changes ?? [],
      compressionLevel:       parsed.compressionLevel,
      originalTokenEstimate:  parsed.originalTokenEstimate,
      optimizedTokenEstimate: parsed.optimizedTokenEstimate,
      tokensReduced:          parsed.tokensReduced,
      reductionPercent:       parsed.reductionPercent,
      savedPromptId,
    },
  });
}
