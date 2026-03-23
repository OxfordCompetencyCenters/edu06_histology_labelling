#!/bin/bash
# Push changes in the correct order:
#   1. Commit & push paper submodule (if it has changes)
#   2. Update the submodule reference in the main repo and push
#
# Usage:
#   ./git_push.sh "commit message"
#   ./git_push.sh "paper msg" "main repo msg"
#
# If one message is given, it's used for both repos.
# If no message is given, you'll be prompted.
set -e

MSG_PAPER="${1}"
MSG_MAIN="${2:-$MSG_PAPER}"

if [ -z "$MSG_PAPER" ]; then
    read -rp "Commit message: " MSG_PAPER
    MSG_MAIN="$MSG_PAPER"
fi

PAPER_PUSHED=false

# 1. Handle paper submodule
if git -C paper status --porcelain | grep -q .; then
    echo "=== Paper submodule has changes ==="
    git -C paper add -A
    git -C paper status --short
    echo ""
    git -C paper commit -m "$MSG_PAPER"
    git -C paper push
    PAPER_PUSHED=true
    echo ""
fi

# 2. Handle main repo
# Stage any updated submodule ref + other changes
git add -A

if git status --porcelain | grep -q .; then
    echo "=== Main repo has changes ==="
    git status --short
    echo ""
    if [ "$PAPER_PUSHED" = true ] && [ "$MSG_MAIN" = "$MSG_PAPER" ]; then
        # Auto-message for submodule ref bump when no separate main changes
        ONLY_SUBMODULE=$(git status --porcelain | grep -v "^.. paper$" | head -1)
        if [ -z "$ONLY_SUBMODULE" ]; then
            MSG_MAIN="Update paper submodule ref"
        fi
    fi
    git commit -m "$MSG_MAIN"
    git push
else
    echo "Main repo has no changes to push."
fi

echo ""
echo "Done."
