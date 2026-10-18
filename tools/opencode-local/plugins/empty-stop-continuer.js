// Global: keep an agent working when a local model's turn ends where it should not.  On session.idle the last
// assistant message is checked, and one short continuation is sent when it is:
//   1. empty    - reasoning only (no text, no tool call): the model ended its turn inside its thinking;
//   2. announced - text that ends with ":" ("Running that:") and no tool call: it ended just before the call;
//   3. stopped mid-list - text and no tool call while the session's todo list still has open items, during autonomous
//      work (tool calls since the user's last message): "what next?" when the next item is known.
// At most MAX_RETRIES nudges in a row (reset by a tool call after a nudge, or by the user writing), then one request to
// write the answer and continue or ask.  Messages are short: each is visible and stays in the context.
// Measured 2026-10-13 replaying the user's real stuck agent turns on IQ3_S: every empty turn ended inside the
// thinking (Strata's "reasoning_close_retry" rescues those server-side: 15/48 empty -> 0/48); of the rest, 5/96 replies
// announced a step and stopped, and the list-stops were reports or "want me to ...?" with work left.
// (Moved here from a project's .opencode/plugins, which now keeps only its compaction hooks.)

import { MAX_RETRIES, TEXT, nudgeFor, streakOf } from "../lib/empty-stop-nudge.mjs";

export const EmptyStopContinuer = async ({ client, project }) => {
  const log = async (level, message, extra) => {
    try {
      await client?.app?.log?.({ body: { service: "empty-stop-continuer", level, message, extra } });
    } catch {
      // logging must never break the hook
    }
  };

  const prompt = async (sessionID, text) => {
    try {
      await client.session.prompt({ path: { id: sessionID }, body: { parts: [{ type: "text", text }] } });
    } catch (err) {
      await log("warn", `prompt failed: ${err?.message ?? err}`, { sessionID });
    }
  };

  const openTodos = async (sessionID) => {
    try {
      const todos = (await client.session.todo({ path: { id: sessionID } }))?.data ?? [];
      return todos.filter((t) => t.status === "pending" || t.status === "in_progress").length;
    } catch {
      return 0;   // no todo list (or an older opencode): only the empty and announced cases apply
    }
  };

  const onIdle = async (sessionID) => {
    let msgs;
    try {
      msgs = (await client.session.messages({ path: { id: sessionID } }))?.data ?? [];
    } catch (err) {
      await log("warn", `messages fetch failed: ${err?.message ?? err}`, { sessionID });
      return;
    }
    const kind = nudgeFor(msgs, await openTodos(sessionID));
    if (!kind) return;
    const n = streakOf(msgs);
    if (n > MAX_RETRIES) return;                       // exhausted: wait for progress or the user
    const text = n === MAX_RETRIES ? TEXT.exhausted : TEXT[kind];
    await log("info", `${kind}: nudge ${n + 1}/${MAX_RETRIES + 1}`, { sessionID });
    await prompt(sessionID, text);
  };

  return {
    event: async ({ event }) => {
      if (event.type !== "session.idle") return;
      const sessionID = event?.properties?.sessionID;
      if (!sessionID) return;
      if (project?.id) {
        try {
          const res = await client.session.get({ path: { id: sessionID } });
          if (res?.data?.projectID && res.data.projectID !== project.id) return;
        } catch {
          return;
        }
      }
      await onIdle(sessionID);
    },
  };
};
