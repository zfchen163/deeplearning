AGENTS.md

Overview
- This file defines how agent-like tooling should operate within this repository.
- It is intended for autonomous coding agents and human reviewers who interact with the repo.
- Follow the guidance to keep builds repeatable, tests reliable, and changes well‑documented.

Build, Lint, and Test Commands
- Build
  - Preferred: npm run build if a package.json with a build script exists in the project root or relevant package.
  - Fallback: make build if a Makefile defines a build target.
  - Fallback: python -m build or python setup.py build for Python projects.
  - If none of the above exist, describe why a build is not required for this workspace.
- Lint
  - Preferred: npm run lint or yarn lint when a JavaScript/TypeScript project uses ESLint/Prettier.
  - Fallback: eslint . --ext .js,.jsx,.ts,.tsx for plain ESLint setups.
  - Fallback: python -m flake8 . or mypy for Python typing checks, if configured.
- Tests (general)
  - Preferred: npm test or yarn test to run the test suite defined by the project.
  - If there are multiple test suites, run: npm run test:unit, npm run test:integration, etc.
  - For a single test: use the test runner feature to filter by test name.
    • Jest: npm test -- -t "Name of test" or yarn test -- -t "Name of test"
    • Mocha: npm test -- -grep "Name of test" or npm test -- --grep "Name of test"
  - Python: pytest -k "name_of_test" --maxfail=1 -q
  - Rust: cargo test -- --exact "test_name" if applicable
- Running a single test reliably
  - Identify the test framework first (Jest, Mocha, PyTest, etc.).
  - Use the framework-specific filter flag as shown above.
  - Ensure environment variables required by the test are set (e.g., CI=false locally).

Code Style and Quality Guidelines
- Imports
  - Group imports in three sections: external, internal, then side-effect/imports.
  - Use absolute imports where possible; prefer project-relative paths for internal modules only when stable.
  - Alphabetize within groups, with a clear separation between groups.
- Formatting
  - Prefer Prettier/ESLint formatting if configured; run formatters as part of every PR.
  - Enforce 2-space indentation (or project‑specific) and consistent line length (usually 80–100 chars).
  - Prefer single quotes for strings in JavaScript/TypeScript when not using a template literal.
- Types and typing
  - Use explicit types; avoid implicit any in TypeScript, especially public APIs.
  - Prefer interfaces for public shapes; use type aliases for complex unions only when helpful.
- Naming conventions
  - Variables and functions: camelCase; constants: UPPER_SNAKE_CASE; types/classes: PascalCase.
  - Method names describe behavior; avoid vague abbreviations.
- Error handling
  - Do not swallow errors; wrap with context and preserve original stack where possible.
  - Use domain-specific error types; export and reuse common error constructors.
- Documentation and comments
  - Document public APIs with JSDoc/TSDoc; include parameter and return descriptions.
  - Add comments for non-trivial algorithms; explain intent, not just what the code does.
- Tests
  - Tests should be deterministic and fast; isolate slow dependencies behind mocks.
  - Use descriptive test names; explain the scenario and expected outcome.
  - Include edge cases and failure modes; simulate error paths.
- Accessibility and UX (UI projects)
  - Include ARIA labels and keyboard navigation checks; test with keyboard first.
- Performance
  - Prefer stable benchmarks; avoid micro-optimizations without evidence.
  - Avoid unnecessary allocations in hot paths; use memoization where appropriate.
- Security and privacy
  - Do not log secrets; scrub environment values in tests; use test doubles for credentials.
- Versioning and changelog
  - Keep commits focused; update CHANGELOG only when a user-facing change occurs.

Cursor Rules (if present)
- If a Cursor rules directory exists at .cursor/rules/ or .cursorrules, ensure agents
  follow the defined patterns for memory usage, context budgeting, and command throttling.
- Respect any rate limits, avoid leaking sensitive data to the console, and cap memory footprint.
- Update rules only after reviewing potential security implications or performance trade-offs.

Copilot Rules (if present)
- If .github/copilot-instructions.md exists, follow its guidance for Copilot usage, safety, and prompts.
- Do not rely on Copilot to replace critical reasoning; verify all generated code paths manually.
- Keep Copilot suggestions bounded by the project’s coding standards and security guidelines.

Repository Hygiene and Workflows
- Do not alter unrelated files; scope changes to the intended module or feature.
- Run build, lint, and tests before creating a PR; fix failures locally when possible.
- Use conventional commit messages if the project expects them; otherwise keep messages concise and descriptive.
- When in doubt, add a short note in the PR description about the rationale and any follow-ups.

Pro-Tips for Agents
- Always prefer reproducible results; pin dependencies if the project uses a lockfile.
- Use environment configuration to switch between local/CI behavior, not code changes.
- If you generate artifacts (docs, HTML, images), place them in a dedicated output directory and add a note in the PR.

Next Steps
- If you want, I can scan the repo (once accessible) to tailor the commands to your actual package.json scripts and toolchain.
