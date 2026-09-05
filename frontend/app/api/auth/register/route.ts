/**
 * POST /api/auth/register
 * Creates a new user account with bcrypt-hashed password.
 */
import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import bcrypt from "bcrypt";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const name = typeof body.name === "string" ? body.name.trim() : "";
    const email = typeof body.email === "string" ? body.email.trim().toLowerCase() : "";
    const password = typeof body.password === "string" ? body.password : "";

    if (!name || !email || !password) {
      return NextResponse.json({ success: false, error: { code: "MISSING_FIELDS", message: "Name, email and password are required." } }, { status: 400 });
    }
    if (password.length < 8) {
      return NextResponse.json({ success: false, error: { code: "WEAK_PASSWORD", message: "Password must be at least 8 characters." } }, { status: 400 });
    }
    if (name.length < 2 || name.length > 80) {
      return NextResponse.json({ success: false, error: { code: "INVALID_NAME", message: "Name must be between 2 and 80 characters." } }, { status: 400 });
    }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      return NextResponse.json({ success: false, error: { code: "INVALID_EMAIL", message: "Enter a valid email address." } }, { status: 400 });
    }

    const existing = await prisma.user.findUnique({ where: { email } });
    if (existing) {
      return NextResponse.json({ success: false, error: { code: "EMAIL_EXISTS", message: "An account with this email already exists." } }, { status: 409 });
    }

    const passwordHash = await bcrypt.hash(password, 12);
    const user = await prisma.user.create({
      data: { name, email, passwordHash },
    });

    // Create default preferences for the new user
    await prisma.preferences.create({ data: { userId: user.id } });

    return NextResponse.json({ success: true, user: { id: user.id, email: user.email, name: user.name } }, { status: 201 });
  } catch (err) {
    console.error("[register]", err);
    return NextResponse.json({ success: false, error: { code: "SERVER_ERROR", message: "Something went wrong." } }, { status: 500 });
  }
}
