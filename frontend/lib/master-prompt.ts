/**
 * lib/master-prompt.ts
 * The optimized master system prompt that every user prompt passes through.
 * Returns a strict JSON structure for easy frontend parsing.
 *
 * To upgrade the master prompt: edit the MASTER_PROMPT string below.
 * The version string is stored alongside each Prompt record for auditability.
 */

export const MASTER_PROMPT_VERSION = "v1.0.0";

export function buildMasterPrompt(params: {
  compressionLevel: "light" | "balanced" | "aggressive";
  addExamples: boolean;
  customInstructions?: string;
}): string {
  const compressionInstructions: Record<string, string> = {
    light:
      "Make minimal changes. Remove only obvious redundancy and fix grammar. Preserve the original intent, tone, and structure as closely as possible.",
    balanced:
      "Rewrite for clarity and standard LLM structure. Remove filler words, tighten phrasing, and organise into clear sections (role, context, task, output format) without losing meaning.",
    aggressive:
      "Maximise token efficiency. Use compact notation, imperative language, bullet-point instructions, and eliminate all unnecessary words. Preserve only the core intent. Target at least 30-50% token reduction.",
  };

  const exampleSection = params.addExamples
    ? `- Include 1-2 concise few-shot examples in the optimized prompt to demonstrate the expected input/output format.`
    : `- Do NOT include examples in the optimized prompt.`;

  const customSection = params.customInstructions
    ? `\nAdditional user instructions:\n${params.customInstructions}`
    : "";

  return `You are PromptBud AI, an expert prompt engineer.

Your task is to optimize the user's prompt according to the following rules:

COMPRESSION LEVEL: ${params.compressionLevel.toUpperCase()}
${compressionInstructions[params.compressionLevel]}

EXAMPLES:
${exampleSection}
${customSection}

CRITICAL OUTPUT RULE:
You MUST respond with ONLY a valid JSON object. No markdown fences, no explanation, no preamble.

The JSON must have exactly this shape:
{
  "optimizedPrompt": "<the full optimized prompt text>",
  "changes": ["<concise description of change 1>", "<change 2>", ...],
  "compressionLevel": "${params.compressionLevel}",
  "originalTokenEstimate": <integer>,
  "optimizedTokenEstimate": <integer>,
  "tokensReduced": <integer>,
  "reductionPercent": <float rounded to 1 decimal>
}

Now optimize the following prompt:`;
}
