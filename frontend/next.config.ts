import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  turbopack: {
    // Explicitly set the workspace root to the frontend directory so Next.js
    // doesn't try to infer it from the monorepo structure and emit a warning.
    root: path.resolve(__dirname),
  },
};

export default nextConfig;
