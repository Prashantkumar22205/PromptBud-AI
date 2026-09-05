# Task Checklist - PromptBud AI Implementation

- [x] Task 1: Environment & Docker Setup
  - [x] Create `.env.example`
  - [x] Create `.env` with secure keys
  - [x] Create `.gitignore`
  - [x] Create `docker-compose.yml` for PostgreSQL
  - [x] Run `docker compose up -d` to launch DB

- [x] Task 2: Project Boilerplate & Tailwind Config
  - [x] Initialize Next.js app in workspace
  - [x] Setup `tailwind.config.ts` matching Stitch Obsidian style (fonts, roundness, colors)
  - [x] Add Global CSS and install packages (`next-auth`, `bcrypt`, `@google/genai`, `openai`, `@anthropic-ai/sdk`, `prisma`, `@prisma/client`)

- [~] Task 3: Prisma Database Setup
  - [x] Create `prisma/schema.prisma`
  - [ ] Run Prisma validation, Client generation, and database migrations

- [~] Task 4: Server-side Security & Helper Modules
  - [ ] Create `lib/crypto.ts` for AES-256-GCM encryption
  - [x] Implement secure Auth.js configuration using bcrypt credentials provider

- [~] Task 5: Backend API Routes
  - [x] Implement `/api/auth/register` for signing up users
  - [ ] Implement `/api/preferences` for saving options and encrypting API keys (with strict masking)
  - [ ] Implement `/api/optimize` with master prompt orchestrator returning JSON structure from LLMs

- [x] Task 6: Frontend Views (Stitch Layouts)
  - [x] Implement `/login` page with Sign In / Sign Up tab switching
  - [x] Implement `/profile` (Action Center) page to configure LLM keys & default settings
  - [x] Implement `/` (Landing/Workspace) page with primary prompt textareas, model selectors, 3-level sliders, and optimized JSON response parsing
  - [x] Implement `/library` page for managing prompt history and favorites

- [ ] Task 7: Verification and Cleanup
  - [ ] Verify full-stack functionality (sign-up, save key, run optimizer, library saving)
  - [ ] Write `walkthrough.md` with walkthrough and code links