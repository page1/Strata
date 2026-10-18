// Helpers of plugins/empty-stop-continuer.js (kept out of plugins/: opencode runs every export there as a plugin).

export const MAX_RETRIES = 2;
const SKIP_AGENTS = new Set(["compaction", "title"]);
export const TEXT = {
  empty: "Your last turn was empty. Continue.",
  announced: "Continue.",
  list: "Continue with the next item on your todo list; ask only if a decision is needed.",
  exhausted: "Write your answer now, then continue with the next step; ask only if a decision is needed.",
};
const OURS = new Set(Object.values(TEXT));

const textOf = (parts) =>
  (parts || []).filter((p) => p.type === "text" && !p.synthetic).map((p) => p.text || "").join("").trim();
const hasTool = (parts) => (parts || []).some((p) => p.type === "tool");

// Which nudge the session needs now ("" = none).
export function nudgeFor(msgs, openTodos) {
  const last = msgs[msgs.length - 1];
  const info = last?.info ?? {};
  if (!last || info.role !== "assistant" || SKIP_AGENTS.has(info.agent) || info.error) return "";
  if (hasTool(last.parts)) return "";
  const text = textOf(last.parts);
  if (!text) return "empty";
  if (text.endsWith(":")) return "announced";
  if (openTodos > 0) {
    // autonomous work only: a tool call since the user's own last message (a direct question answered stays answered)
    for (let i = msgs.length - 2; i >= 0; i--) {
      const m = msgs[i];
      if (m.info?.role === "user" && !OURS.has(textOf(m.parts))) return "";
      if (m.info?.role === "assistant" && hasTool(m.parts)) return "list";
    }
  }
  return "";
}

// How many nudges in a row: our messages since the last progress (a tool call after one of them) or the user's text.
export function streakOf(msgs) {
  let n = 0;
  for (let i = msgs.length - 1; i >= 0; i--) {
    const m = msgs[i];
    if (m.info?.role === "assistant" && hasTool(m.parts)) break;
    if (m.info?.role === "user") {
      if (!OURS.has(textOf(m.parts))) break;
      n++;
    }
  }
  return n;
}

