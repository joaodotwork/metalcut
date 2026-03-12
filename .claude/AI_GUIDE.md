# AI Agent Session Guide for Plantas + Seiva

Welcome! This project uses **Plantas** (session management) with **Seiva** (context management).

## 🚨 MEMORIZE THESE COMMANDS - You will use them frequently!

**Core Plantas Commands (EXACT SYNTAX - Commit to Memory):**

```bash
# Check what sessions exist (ALWAYS run this first!)
plantas list

# Start a new session (quoted description is REQUIRED!)
plantas start <id> "<description>" [branch] [areas] [--issue <number>]

# End a session when done
plantas end <id>

# Check current session status
plantas status
```

**Command Syntax Rules:**
- Session description MUST be quoted: `"description"` ✅ not `description` ❌
- Optional parameters in square brackets: `[branch]` `[areas]`
- Session ID: short, no spaces (e.g., `auth`, `fix-bug`, `A`, `B`)
- Use `--issue <number>` to associate with a GitHub issue

**Real Examples to Remember:**
```bash
plantas start auth "Implement user authentication" feat/auth "src/auth/**" --issue 123
plantas start bugfix "Fix navigation issues" fix/nav
plantas start A "Quick admin fixes"
plantas end auth
```

## Quick Start for AI Assistants

1. **ALWAYS check active sessions first:**
   ```bash
   plantas list
   ```
   See what sessions are active and their focus areas. Use `plantas list --active` for a cleaner view.

2. **Read the context:**
   ```
   Read .claude/context.md
   Read .claude/sessions.json
   ```
   This gives you project state, tech stack, and recent activity (~2k tokens vs 30-50k full scan).

   **Important:** context.md and sessions.json ARE committed to git and use union merge strategy.
   All branches accumulate their updates additively - no work is lost during merges.
   This enables true cross-session coordination without conflicts.

3. **Use plantas commands for session management** - See memorized syntax above!

## 🪝 Automatic Context Updates (Post-Commit Hook)

**IMPORTANT: context.md updates automatically after EVERY commit via git post-commit hook!**

**What happens automatically when you commit:**
1. Git runs your commit
2. **Post-commit hook triggers** (`.git/hooks/post-commit`)
3. Hook updates context.md with current state, recent activity, and uncommitted work.
4. **Seiva analyzer runs** to intelligently update Tech Stack, Key Files, and Work Streams.
5. Hook **amends your commit** to include updated context.md
6. You see `[SEIVA_HOOK_COMPLETE]` signal

**What this means for you as an AI Agent:**
- ✅ **You NEVER need to manually update context.md** - it happens automatically
- ✅ After making changes and committing, context.md is already up-to-date
- ✅ The hook is silent and fast - just watch for `[SEIVA_HOOK_COMPLETE]`
- ✅ All updates are committed via `--amend` so everything stays in one commit
- ⚠️ If you read context.md immediately after commit, it will already be updated

**Exception - Manual Updates:**
Only manually update context.md when adding project-specific information the analyzer can't detect or documenting architectural decisions.

## Session Management Commands

**IMPORTANT**: Always use plantas commands for session operations. Never manually manipulate worktrees or sessions.json.

### Starting a Session
When the user wants to start working on something new:
```bash
plantas start <session-id> "<description>" [optional-branch] [optional-areas] [--issue <number>]

# Examples:
plantas start auth "Implement user authentication" feat/auth "src/auth/**" --issue 45
plantas start bugfix "Fix navigation issues" fix/nav
plantas start feature "Add new dashboard"
```

### Ending a Session
When work is complete and the user wants to cleanup:
```bash
plantas end <session-id>

# Example:
plantas end auth

# Skip confirmations (for scripts):
plantas end auth --force
```

### Checking Sessions
Before starting work, always check what's active:
```bash
plantas list              # See all active and recent sessions
plantas list --active     # Only active sessions
plantas list --recent 5   # Active + last 5 closed
plantas status            # Current session status
```

### What NOT to Do

❌ **Never manually:**
- Run `git worktree add/remove` directly
- Edit sessions.json file
- Delete session directories manually
- Run git operations that might affect session state

✅ **Always:**
- Use `plantas start` to begin new work
- Use `plantas end` to cleanup
- Use `plantas list` to check status
- Let plantas handle all session lifecycle operations

## Session Workflow

**When user asks to start new work:**
1. Check what's already active: `plantas list`
2. Start the new session: `plantas start <id> "<description>" [branch] [areas]`
3. Read context.md (project understanding)
4. Work in the session worktree

**During work:**
- Make changes within your focus areas
- Commit regularly to git
- context.md auto-updates via git hooks
- Use `plantas list` to see other active sessions

**When user asks to end session:**
1. Ensure work is committed
2. Run: `plantas end <session-id>`
3. Confirm when prompted (or use --force to skip)

## Token Optimization

- ✅ Read context.md first (~2k tokens)
- ✅ Use grep/glob for targeted searches
- ✅ Read specific files only when needed
- ❌ Avoid full codebase scans

## When Context Gets Full (Compression Time)

**IMPORTANT:** When the conversation context approaches limits and you need to compress/summarize:

1. **Read context.md first** to see what's already documented
2. **ADD to context.md, don't replace** - append new information to existing sections
3. **Preserve work history** - add references to recent commits, session focus, and key decisions.

This preserves institutional memory across sessions and helps future AI Agent instances understand project evolution.

## Resources

### Command Reference
- `plantas list` - List all active and recent sessions
- `plantas start <id> "<desc>" [branch] [areas]` - Start a new session
- `plantas end <id> [--force]` - End a session and cleanup
- `plantas rebuild [--dry-run]` - Rebuild sessions.json from git history
- `plantas status` - Show current session status
- `plantas doctor` - Check installation and dependencies
- `plantas init --seiva` - Initialize in a new project

---
Made with 🌱 by Plantas + 🌊 Seiva
