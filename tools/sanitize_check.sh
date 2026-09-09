#!/usr/bin/env bash
# Publication gate: fail if anything identifying or secret is about to be published.
# Run before committing, and in CI. Exit 0 = clean.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

fail=0
report() { printf '\n[FAIL] %s\n' "$1"; shift; printf '%s\n' "$@" | sed 's/^/    /'; fail=1; }

# Files to scan: everything tracked, minus this script (it contains the patterns itself).
if git rev-parse --git-dir >/dev/null 2>&1; then
  mapfile -d '' FILES < <(git ls-files -z ':!tools/sanitize_check.sh')
else
  mapfile -d '' FILES < <(find . -type f \
    -not -path './.git/*' -not -path './.venv/*' -not -path '*/__pycache__/*' \
    -not -path './tools/sanitize_check.sh' -print0)
fi
[ "${#FILES[@]}" -eq 0 ] && { echo "no files to scan"; exit 2; }

check() {  # check <label> <extended-regex>
  local label=$1 pattern=$2 hits
  hits=$(grep -rniE -- "$pattern" "${FILES[@]}" 2>/dev/null | grep -v '^Binary file' | head -20)
  [ -n "$hits" ] && report "$label" "$hits"
}

# --- identifying data -------------------------------------------------------
check "former project/customer identifiers"  'spacenet'
check "absolute local paths (k6 writes these into its JSON output)" '/(mnt|home|Users)/[A-Za-z0-9._-]+/'
check "domain-specific vocabulary that narrows the deployment" \
      '\b(healthcare|clinical|anamnesis|hospital ward|lab value)\b'
check "customer framing" '\b(the customer|customer-supplied|customer request|Kunde|Gesundheit)\b'
check "German prose left in an English repository" \
      '\b(nicht|werden|Zahlen|Anforderung|Messung|Ergebnis|Zusammenfassung)\b'

# --- secrets ----------------------------------------------------------------
check "OpenAI-style API key"     'sk-[A-Za-z0-9]{16,}'
check "populated licence key"    'SEMVEC_LICENSE_KEY[[:space:]]*=[[:space:]]*[A-Za-z0-9_.-]{8,}'
check "bearer token"             'Bearer[[:space:]]+[A-Za-z0-9._-]{20,}'
check "private key block"        'BEGIN (RSA|OPENSSH|EC|DSA|PGP) PRIVATE KEY'
check "AWS access key id"        'AKIA[0-9A-Z]{16}'

# --- files that must never be committed ------------------------------------
for f in .env .coverage semvec.db; do
  [ -e "$f" ] && report "file must not be committed: $f" "$f"
done
while IFS= read -r -d '' p; do
  case $p in .env|.env.example) ;; .env.*) report "dotenv variant committed" "$p" ;; esac
done < <(printf '%s\0' "${FILES[@]}")

# --- results must be valid JSON (a .json that is not JSON breaks tooling) --
if command -v python3 >/dev/null 2>&1; then
  bad=$(python3 - <<'PY'
import json, pathlib
bad = []
for p in sorted(pathlib.Path("results").glob("*.json")):
    try:
        json.loads(p.read_text())
    except Exception as e:
        bad.append(f"{p}: {e}")
print("\n".join(bad))
PY
)
  [ -n "$bad" ] && report "results/ contains invalid JSON" "$bad"
fi

if [ "$fail" -eq 0 ]; then
  echo "sanitize_check: clean (${#FILES[@]} files scanned)"
else
  printf '\nsanitize_check: FAILED — do not publish\n'
fi
exit "$fail"
