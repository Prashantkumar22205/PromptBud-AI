import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import bcrypt from "bcrypt";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

function unauthorized() {
  return NextResponse.json({ success: false, error: { code: "UNAUTHORIZED", message: "Not authenticated." } }, { status: 401 });
}

export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session?.user?.id) return unauthorized();

  const user = await prisma.user.findUnique({
    where: { id: session.user.id },
    select: { id: true, name: true, email: true, tier: true, createdAt: true },
  });
  if (!user) return unauthorized();
  return NextResponse.json({ success: true, user });
}

export async function PUT(req: NextRequest) {
  const session = await getServerSession(authOptions);
  if (!session?.user?.id) return unauthorized();

  const body = await req.json();
  const name = typeof body.name === "string" ? body.name.trim() : undefined;
  const currentPassword = typeof body.currentPassword === "string" ? body.currentPassword : "";
  const newPassword = typeof body.newPassword === "string" ? body.newPassword : "";

  if (name !== undefined && (name.length < 2 || name.length > 80)) {
    return NextResponse.json({ success: false, error: { code: "INVALID_NAME", message: "Name must be between 2 and 80 characters." } }, { status: 400 });
  }
  if (newPassword && newPassword.length < 8) {
    return NextResponse.json({ success: false, error: { code: "WEAK_PASSWORD", message: "New password must be at least 8 characters." } }, { status: 400 });
  }
  if (newPassword && !currentPassword) {
    return NextResponse.json({ success: false, error: { code: "CURRENT_PASSWORD_REQUIRED", message: "Enter your current password to set a new one." } }, { status: 400 });
  }

  const user = await prisma.user.findUnique({ where: { id: session.user.id } });
  if (!user) return unauthorized();
  if (newPassword && !(await bcrypt.compare(currentPassword, user.passwordHash))) {
    return NextResponse.json({ success: false, error: { code: "INVALID_CURRENT_PASSWORD", message: "Your current password is incorrect." } }, { status: 400 });
  }

  const updated = await prisma.user.update({
    where: { id: user.id },
    data: { ...(name !== undefined ? { name } : {}), ...(newPassword ? { passwordHash: await bcrypt.hash(newPassword, 12) } : {}) },
    select: { id: true, name: true, email: true, tier: true, createdAt: true },
  });
  return NextResponse.json({ success: true, user: updated });
}
