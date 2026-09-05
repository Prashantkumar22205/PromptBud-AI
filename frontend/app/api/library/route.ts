/**
 * GET  /api/library          → paginated prompt history for auth user
 * PATCH /api/library/[id]   → toggle favorite / update title
 * DELETE /api/library/[id]  → delete a prompt
 */
import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export async function GET(req: NextRequest) {
  const session = await getServerSession(authOptions);
  if (!session?.user?.id) return NextResponse.json({ success: false, error: { code: "UNAUTHORIZED", message: "Not authenticated." } }, { status: 401 });

  const { searchParams } = new URL(req.url);
  const page     = Math.max(1, parseInt(searchParams.get("page") ?? "1"));
  const limit    = Math.min(50, parseInt(searchParams.get("limit") ?? "20"));
  const favorite = searchParams.get("favorite") === "true" ? true : undefined;

  const [prompts, total] = await Promise.all([
    prisma.prompt.findMany({
      where: { userId: session.user.id, ...(favorite !== undefined ? { isFavorite: favorite } : {}) },
      orderBy: { createdAt: "desc" },
      skip: (page - 1) * limit,
      take: limit,
      select: {
        id: true, title: true, provider: true, model: true,
        compressionLevel: true, examplesEnabled: true,
        originalTokens: true, optimizedTokens: true, reductionPercent: true,
        isFavorite: true, createdAt: true,
        originalText: true, optimizedText: true,
      },
    }),
    prisma.prompt.count({ where: { userId: session.user.id, ...(favorite !== undefined ? { isFavorite: favorite } : {}) } }),
  ]);

  return NextResponse.json({ success: true, prompts, total, page, limit });
}
