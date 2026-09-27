#!/usr/bin/env bash
# Build the submission zip from git-tracked files only, so .env, .venv and other
# untracked files can never be included. .git is added so the commit history
# (one commit per build step) travels with the code.
set -euo pipefail
cd "$(dirname "$0")"

OUT="${1:-cheiron-take-home.zip}"
if [ -n "$(git status --porcelain)" ]; then
  echo "Uncommitted changes present; commit first so the zip matches the history." >&2
  exit 1
fi

rm -f "$OUT"
git ls-files | zip -q "$OUT" -@
zip -qr "$OUT" .git

if unzip -l "$OUT" | grep -qE '(^|/)\.env$'; then
  echo "ERROR: .env found in $OUT" >&2
  rm -f "$OUT"
  exit 1
fi
if unzip -p "$OUT" 2>/dev/null | grep -qaE 'sk-[A-Za-z0-9_-]{20,}'; then
  echo "ERROR: something that looks like an API key is in $OUT" >&2
  rm -f "$OUT"
  exit 1
fi
echo "Wrote $OUT ($(unzip -l "$OUT" | tail -1 | awk '{print $2}') files). No .env, no key-shaped strings."
