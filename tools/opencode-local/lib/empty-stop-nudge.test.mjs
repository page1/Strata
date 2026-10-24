// node ~/.config/opencode/lib/empty-stop-nudge.test.mjs
import assert from "node:assert/strict";
import { MAX_RETRIES, TEXT, nudgeFor, streakOf } from "./empty-stop-nudge.mjs";

const user = (text) => ({ info: { role: "user" }, parts: [{ type: "text", text }] });
const asst = (...parts) => ({ info: { role: "assistant", agent: "build" }, parts });
const text = (t) => ({ type: "text", text: t });
const tool = () => ({ type: "tool", tool: "bash" });
const think = (t) => ({ type: "reasoning", text: t });

// 1. empty: reasoning only
assert.equal(nudgeFor([user("go"), asst(tool()), asst(think("Let me report."))], 0), "empty");
// 2. announced: ends with ":" and no call
assert.equal(nudgeFor([user("go"), asst(tool()), asst(text("Running that:"))], 0), "announced");
// 3. list: a stop with open todos during autonomous work
assert.equal(nudgeFor([user("work through the list"), asst(tool()), asst(text("Done item 2. Want me to do item 3?"))], 2), "list");
// ... but not without open todos
assert.equal(nudgeFor([user("work through the list"), asst(tool()), asst(text("All done."))], 0), "");
// ... and not when the user asked a direct question that was answered without tools
assert.equal(nudgeFor([user("go"), asst(tool()), user("what is close_lc?"), asst(text("A return ratio."))], 3), "");
// a turn with a tool call, an error or a compaction agent: none
assert.equal(nudgeFor([user("go"), asst(text("x"), tool())], 3), "");
assert.equal(nudgeFor([user("go"), { info: { role: "assistant", error: { name: "x" } }, parts: [] }], 0), "");
assert.equal(nudgeFor([user("go"), { info: { role: "assistant", agent: "compaction" }, parts: [] }], 0), "");
assert.equal(nudgeFor([user("go")], 0), "");
// the user's last message being one of ours still counts as autonomous work
assert.equal(nudgeFor([user("go"), asst(tool()), user(TEXT.announced), asst(text("Item 3 done."))], 1), "list");

// streak: our nudges since the last tool call or the user's own text
assert.equal(streakOf([user("go"), asst(think("a"))]), 0);
assert.equal(streakOf([user("go"), asst(think("a")), user(TEXT.empty), asst(think("b"))]), 1);
assert.equal(streakOf([user("go"), asst(think("a")), user(TEXT.empty), asst(think("b")), user(TEXT.empty), asst(think("c"))]), 2);
assert.equal(streakOf([user("go"), user(TEXT.empty), asst(tool()), user(TEXT.list), asst(text("ok"))]), 1);   // progress resets
assert.equal(streakOf([user(TEXT.empty), asst(think("a")), user("my own words"), asst(think("b"))]), 0);       // the user resets
assert.equal(MAX_RETRIES, 2);
console.log("empty-stop-nudge: all checks passed");
