# Implementation Plan - PromptBud AI Full-Stack Application

We will build the **PromptBud AI** application as a full-stack platform using **Next.js** (Frontend + API Routes), **PostgreSQL** (Database), **Prisma** (ORM), and **Docker** (for running PostgreSQL locally).

The application design will match the **Kinetic Obsidian** design system retrieved from Stitch, featuring:
* Deep black (#000000) and charcoal (#0A0A0A) backgrounds.
* Vibrant neon green (#00FF66) primary focus states and interactive triggers.
* **Hanken Grotesk**, **Inter**, and **Geist** typography.
* Tactile, rounded elements (4px to 12px border radius) and glowing active states.

---

## Architectural & Security Updates (User Requests Added)

### 1. API Key Encryption Scheme (AES-256-GCM)
* **Secret Key:** An `ENCRYPTION_KEY` (32 bytes) will be stored in server-side environment variables.
* **Storage Structure:** API keys will be stored in PostgreSQL as JSON strings containing:
  - `ciphertext`: The encrypted key.
  - `iv`: The initialization vector/nonce (hex).
  - `tag`: The AES-256-GCM authentication tag (hex).
* **Security Constraints:** Keys are decrypted *only* server-side inside API routes. 
* **Zero Plaintext Exfiltration:** The API endpoint `/api/preferences` will **never** send decrypted key values to the client, even for the authenticated owner. Instead, it will return a boolean indicating whether the key has been saved (e.g., `{ geminiApiKeySaved: true, openaiApiKeySaved: false }`). Users can overwrite or delete keys, but never retrieve them.

### 2. Established Authentication (Auth.js / NextAuth.js)
* We will use **NextAuth.js** with the **Prisma Adapter** for session management instead of custom JWT handling.
* A standard **Credentials Provider** will handle login/signup, using **bcrypt** to hash passwords securely before database persistence.

### 3. Provider & Model Separation
* Prompt executions and user defaults will separate **provider** and **model** into discrete fields:
  - `provider`: `"google"` | `"openai"` | `"anthropic"`
  - `model`: `"gemini-2.5-flash"`, `"gpt-4o"`, `"claude-3-5-sonnet"`, etc.

### 4. Structured API Errors
* Optimization API will catch provider/network failures and return clean, structured errors:
  ```json
  {
    "success": false,
    "error": {
      "code": "PROVIDER_ERROR" | "MISSING_API_KEY" | "RATE_LIMIT_EXCEEDED" | "INVALID_PROMPT",
      "message": "User-friendly description of the error."
    }
  }
  ```

### 5. Intentional Prompt History / Library
* Prompts will be associated with an authenticated user (`userId` is non-nullable).
* The `Prompt` table will store metadata including `title`, `isFavorite`, and `updatedAt` to support a rich library management workspace.

### 6. Hybrid Local Docker Setup
* **Docker Container:** Run PostgreSQL (`postgres:16`) inside Docker, mapped to port `5432` with a persistent local volume.
* **Native Development:** Next.js will run directly on the host OS (`npm run dev`) for rapid feedback and hot reloading.

---

## Proposed Architecture & Directory Structure

```
promptbud-ai/
├── app/                  # Next.js App Router (pages & components)
│   ├── layout.tsx        # Global HTML structure and font loaders
│   ├── page.tsx          # Landing / Optimizer Workspace (Screen 1 & 2)
│   ├── login/            # Auth Page (Screen 3)
│   ├── profile/          # User Dashboard / Preferences & API Keys (Screen 4)
│   ├── library/          # Prompt History & Favorites List
│   └── api/              # Node.js API Endpoints
│       ├── optimize/     # Handles prompt optimization request
│       ├── auth/         # NextAuth endpoint handler
│       └── preferences/  # Handles user preferences CRUD and key encryption
├── prisma/               # Prisma Database Schemas & Migrations
│   └── schema.prisma
├── public/               # Static assets & icons
├── docker-compose.yml    # Runs local PostgreSQL DB
├── tailwind.config.ts    # Styled with Stitch Design DNA tokens
├── .env.example          # Environment variables template
├── .gitignore            # Excludes .env and node_modules
└── package.json
```

---

## Database Schema (PostgreSQL)

```prisma
datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

generator client {
  provider = "prisma-client-js"
}

model User {
  id           String        @id @default(uuid())
  email        String        @unique
  passwordHash String
  name         String
  tier         String        @default("FREE") // FREE or PRO
  createdAt    DateTime      @default(now())
  updatedAt    DateTime      @updatedAt
  prompts      Prompt[]
  preferences  Preferences?
}

model Preferences {
  id                    String   @id @default(uuid())
  userId                String   @unique
  user                  User     @relation(fields: [userId], references: [id], onDelete: Cascade)
  
  // API Keys (Stored as encrypted JSON strings: ciphertext, iv, tag)
  geminiApiKey          String?
  openaiApiKey          String?
  anthropicApiKey       String?
  
  // Settings Preferences
  defaultProvider       String   @default("google") // google, openai, anthropic
  defaultModel          String   @default("gemini-2.5-flash")
  defaultCompression    String   @default("balanced") // light, balanced, aggressive
  autoSaveLibrary       Boolean  @default(false)
  createdAt             DateTime @default(now())
  updatedAt             DateTime @updatedAt
}

model Prompt {
  id                 String   @id @default(uuid())
  userId             String
  user               User     @relation(fields: [userId], references: [id], onDelete: Cascade)
  title              String   @default("Untitled Prompt")
  originalText       String
  optimizedText      String
  provider           String   // google, openai, anthropic
  model              String   // gemini-2.5-flash, gpt-4o, claude-3-5-sonnet
  compressionLevel   String   // light, balanced, aggressive
  examplesEnabled    Boolean  @default(false)
  customInstructions String?
  isFavorite         Boolean  @default(false)
  createdAt          DateTime @default(now())
  updatedAt          DateTime @updatedAt
}
```

---

## Proposed Changes

### 1. Environment & Setup Config
#### [NEW] [.env.example](file:///d:/Lock%20In/PromptBud%20AI/.env.example)
* Template configuration containing `DATABASE_URL`, `AUTH_SECRET`, and `ENCRYPTION_KEY`.
#### [NEW] [.gitignore](file:///d:/Lock%20In/PromptBud%20AI/.gitignore)
* Ignores `.env`, `.env.local`, `node_modules`, `.next`, and build outputs.

### 2. Database & Docker Config
#### [NEW] [docker-compose.yml](file:///d:/Lock%20In/PromptBud%20AI/docker-compose.yml)
* PostgreSQL service running on container port `5432`.
* Exposes database connection credentials.

### 3. Next.js Boilerplate
* Initialize Next.js project natively in the root directory.
* Install dependencies: `next-auth`, `bcrypt`, `crypto` (node built-in), `@google/genai`, `openai`, `@anthropic-ai/sdk`, `prisma`, `@prisma/client`.
* Setup `tailwind.config.ts` matching the Stitch Kinetic Obsidian design guidelines.

### 4. Database ORM Layer
#### [NEW] [prisma/schema.prisma](file:///d:/Lock%20In/PromptBud%20AI/prisma/schema.prisma)
* Define schemas.
* Migration workflow:
  ```bash
  npx prisma validate
  npx prisma generate
  npx prisma migrate dev --name init
  ```

### 5. Encryption Helper Module
#### [NEW] [lib/crypto.ts](file:///d:/Lock%20In/PromptBud%20AI/lib/crypto.ts)
* Encrypt API Keys using AES-256-GCM.
* Decrypt keys dynamically, catching formatting/decryption failures.

### 6. Backend APIs (Node.js)
#### [NEW] [app/api/auth/[...nextauth]/route.ts](file:///d:/Lock%20In/PromptBud%20AI/app/api/auth/%5B...nextauth%5D/route.ts)
* Setup Auth.js with Credentials validation using bcrypt.
#### [NEW] [app/api/optimize/route.ts](file:///d:/Lock%20In/PromptBud%20AI/app/api/optimize/route.ts)
* Dynamic LLM invocation routing based on selected provider and key retrieval.
* Resolves optimization inputs (prompt, compression, examples toggle) and maps to structured API errors on failure.
#### [NEW] [app/api/preferences/route.ts](file:///d:/Lock%20In/PromptBud%20AI/app/api/preferences/route.ts)
* Read/write endpoints to manage command-center switches and encrypt/save model API keys. Ensures decrypted keys are **never** included in responses.

### 7. Frontend Pages (Stitch UI Recreations)
#### [NEW] [app/page.tsx](file:///d:/Lock%20In/PromptBud%20AI/app/page.tsx)
* Full landing page showing the optimizer tool. Includes model selector dropdown, examples toggle, and 3-level compression slider.
#### [NEW] [app/login/page.tsx](file:///d:/Lock%20In/PromptBud%20AI/app/login/page.tsx)
* Sign In / Sign Up tabs.
#### [NEW] [app/profile/page.tsx](file:///d:/Lock%20In/PromptBud%20AI/app/profile/page.tsx)
* Dashboard containing API Key fields for each model and preferences selectors.

---

## Verification Plan

### Automated Tests
* Initialize local Postgres container:
  `docker compose up -d`
* Run migrations and client generation:
  `npx prisma validate`
  `npx prisma migrate dev --name init`
* Start local server:
  `npm run dev`

### Manual Verification
1. **AES-256-GCM Integrity:** Set an API key, check Postgres to verify it exists only as encrypted JSON, and verify that `/api/preferences` returns *only* boolean flags (e.g., `geminiApiKeySaved: true`), never raw text.
2. **Auth Sessions:** Complete sign-up/sign-in using Auth.js. Verify page routing blocks unauthenticated visitors to `/profile` and `/library`.
3. **Multi-Model Optimization:** Call optimize route with Gemini, OpenAI, and Claude, capturing correct structured errors if their API keys are invalid.
