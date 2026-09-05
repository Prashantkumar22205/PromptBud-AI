/**
 * lib/llm-router.ts
 * Extensible LLM router — to add a new provider:
 *   1. Add its API key field to Prisma Preferences model.
 *   2. Add a new case to callLLM().
 *   3. Export the provider name in SUPPORTED_PROVIDERS below.
 */

import { GoogleGenAI } from "@google/genai";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

export const SUPPORTED_PROVIDERS = ["google", "openai", "anthropic"] as const;
export type Provider = (typeof SUPPORTED_PROVIDERS)[number];

// Default models per provider (kept in sync with Prisma Preferences defaults)
export const DEFAULT_MODELS: Record<Provider, string[]> = {
  google:    ["gemini-2.5-flash", "gemini-2.5-pro"],
  openai:    ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
  anthropic: ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
};

export interface LLMCallParams {
  provider: Provider;
  model: string;
  apiKey: string;
  systemPrompt: string;
  userPrompt: string;
}

export interface LLMResult {
  rawText: string;
}

/**
 * Routes the call to the correct LLM SDK based on provider.
 * Throws a structured error object to be caught by /api/optimize.
 */
export async function callLLM(params: LLMCallParams): Promise<LLMResult> {
  const { provider, model, apiKey, systemPrompt, userPrompt } = params;

  try {
    switch (provider) {
      case "google": {
        const genai = new GoogleGenAI({ apiKey });
        const response = await genai.models.generateContent({
          model,
          contents: [{ role: "user", parts: [{ text: userPrompt }] }],
          config: { systemInstruction: systemPrompt, responseMimeType: "application/json" },
        });
        return { rawText: response.text ?? "" };
      }

      case "openai": {
        const client = new OpenAI({ apiKey });
        const response = await client.chat.completions.create({
          model,
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: userPrompt },
          ],
          response_format: { type: "json_object" },
        });
        return { rawText: response.choices[0]?.message?.content ?? "" };
      }

      case "anthropic": {
        const client = new Anthropic({ apiKey });
        const response = await client.messages.create({
          model,
          max_tokens: 4096,
          system: systemPrompt,
          messages: [{ role: "user", content: userPrompt }],
        });
        const block = response.content[0];
        return { rawText: block.type === "text" ? block.text : "" };
      }

      default:
        throw { code: "UNSUPPORTED_PROVIDER", message: `Provider "${provider}" is not supported.` };
    }
  } catch (err: unknown) {
    // Re-throw structured errors as-is
    if (err && typeof err === "object" && "code" in err) throw err;

    // Translate SDK errors to structured codes
    const message = err instanceof Error ? err.message : String(err);
    if (message.toLowerCase().includes("api key") || message.toLowerCase().includes("auth")) {
      throw { code: "INVALID_API_KEY", message: "The API key for this provider is invalid or expired." };
    }
    if (message.toLowerCase().includes("rate limit") || message.toLowerCase().includes("429")) {
      throw { code: "RATE_LIMIT_EXCEEDED", message: "Rate limit exceeded for this provider. Try again later." };
    }
    throw { code: "PROVIDER_ERROR", message };
  }
}
