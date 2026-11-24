#!/usr/bin/env bash
# Set or change a dashboard login. Asks for the password; only the hash is stored.
set -euo pipefail

USER_NAME="${1:-}"
if [[ ! "$USER_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Usage: $0 <username>" >&2
  exit 1
fi

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
read -rsp "Password for $USER_NAME: " PW; echo
read -rsp "Repeat password: " PW2; echo
if [[ "$PW" != "$PW2" ]]; then echo "Passwords do not match." >&2; exit 1; fi
if (( ${#PW} < 12 )); then echo "Use at least 12 characters." >&2; exit 1; fi

HASH="$(printf '%s\n' "$PW" | docker run --rm -i caddy:2-alpine caddy hash-password)"
unset PW PW2
if [[ "$HASH" != \$2* ]]; then echo "Hashing failed." >&2; exit 1; fi

umask 077
mkdir -p caddy/private
FILE=caddy/private/users.caddy
TMP="$(mktemp caddy/private/.users.XXXXXX)"
{
  echo "basic_auth {"
  if [[ -f "$FILE" ]]; then
    awk -v u="$USER_NAME" 'NF == 2 && $2 ~ /^\$2/ && $1 != u { print "\t" $1 " " $2 }' "$FILE"
  fi
  printf '\t%s %s\n' "$USER_NAME" "$HASH"
  echo "}"
} > "$TMP"
mv "$TMP" "$FILE"

docker compose up -d --no-deps dashboard
docker compose up -d --no-deps --force-recreate caddy
echo "Login for $USER_NAME saved."
