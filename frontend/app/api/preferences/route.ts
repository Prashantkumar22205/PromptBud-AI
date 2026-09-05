/**
 * GET  /api/preferences  → returns safe (masked) preferences for the auth user
 * PUT  /api/preferences  → updates settings and encrypts API keys
 * DELETE /api/preferences/keys?provider=google → removes a stored API key
 *
 * SECURITY: Plaintext API keys are NEVER returned to the client.
 */
import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { encryptApiKey, maskApiKey } from "@/lib/crypto";

// ── GET ──────────────────────────────────────────────────────────────────────
export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session?.user?.id) return NextResponse.json({ success: false, error: { code: "UNAUTHORIZED", message: "Not authenticated." } }, { status: 401 });

  const prefs = await prisma.preferences.findUnique({ where: { userId: session.user.id } });
  if (!prefs) return NextResponse.json({ success: false, error: { code: "NOT_FOUND", message: "Preferences not found." } }, { status: 404 });

  // Return masked key status — never plaintext
  return NextResponse.json({
    success: true,
    preferences: {
      defaultProvider:    prefs.defaultProvider,
      defaultModel:       prefs.defaultModel,
      defaultCompression: prefs.defaultCompression,
      addExamples:        prefs.addExamples,
      autoSaveLibrary:    prefs.autoSaveLibrary,
      // boolean flags only
      geminiApiKeySaved:   !!prefs.geminiApiKey,
      openaiApiKeySaved:   !!prefs.openaiApiKey,
      anthropicApiKeySaved: !!prefs.anthropicApiKey,
      // masked tail (safe to display in UI)
      geminiApiKeyMask:    prefs.geminiApiKey   ? maskApiKey(prefs.geminiApiKey)   : null,
      openaiApiKeyMask:    prefs.openaiApiKey   ? maskApiKey(prefs.openaiApiKey)   : null,
      anthropicApiKeyMask: prefs.anthropicApiKey ? maskApiKey(prefs.anthropicApiKey) : null,
    },
  });
}

// ── PUT ──────────────────────────────────────────────────────────────────────
export async function PUT(req: NextRequest) {
  const session = await getServerSession(authOptions);
  if (!session?.user?.id) return NextResponse.json({ success: false, error: { code: "UNAUTHORIZED", message: "Not authenticated." } }, { status: 401 });

  const body = await req.json();
  const {
    defaultProvider, defaultModel, defaultCompression,
    addExamples, autoSaveLibrary,
    geminiApiKey, openaiApiKey, anthropicApiKey,
  } = body;

  const data: Record<string, unknown> = {};

  if (defaultProvider    !== undefined) data.defaultProvider    = defaultProvider;
  if (defaultModel       !== undefined) data.defaultModel       = defaultModel;
  if (defaultCompression !== undefined) data.defaultCompression = defaultCompression;
  if (addExamples        !== undefined) data.addExamples        = addExamples;
  if (autoSaveLibrary    !== undefined) data.autoSaveLibrary    = autoSaveLibrary;

  // Encrypt API keys if provided (empty string = delete key)
  if (geminiApiKey   !== undefined) data.geminiApiKey   = geminiApiKey   ? encryptApiKey(geminiApiKey)   : null;
  if (openaiApiKey   !== undefined) data.openaiApiKey   = openaiApiKey   ? encryptApiKey(openaiApiKey)   : null;
  if (anthropicApiKey !== undefined) data.anthropicApiKey = anthropicApiKey ? encryptApiKey(anthropicApiKey) : null;

  await prisma.preferences.update({ where: { userId: session.user.id }, data });

  return NextResponse.json({ success: true });
}
