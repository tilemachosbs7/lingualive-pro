/**
 * ESLint Configuration for LinguaLive Pro Extension
 *
 * Uses TypeScript ESLint for type-aware linting.
 * Based on recommended rules with sensible defaults.
 */
module.exports = {
  root: true,
  parser: "@typescript-eslint/parser",
  parserOptions: {
    ecmaVersion: 2022,
    sourceType: "module",
    project: "./tsconfig.json",
    tsconfigRootDir: __dirname,
  },
  plugins: ["@typescript-eslint"],
  extends: [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
  ],
  env: {
    browser: true,
    es2022: true,
  },
  rules: {
    // Allow unused vars prefixed with underscore
    "@typescript-eslint/no-unused-vars": [
      "warn",
      { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
    ],
    // Allow explicit any in some cases (can tighten later)
    "@typescript-eslint/no-explicit-any": "warn",
    // Prefer const
    "prefer-const": "warn",
    // No console in production (warn only)
    "no-console": "off",
  },
  ignorePatterns: ["dist/", "node_modules/", "*.js", "*.cjs"],
};
