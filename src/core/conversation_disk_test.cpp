// CPU-only tests for the disk tier of the parked conversations (include/strata/core/conversation_disk.hpp).
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_disk.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

ConversationBuffer pattern(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}

std::vector<uint8_t> bytes_of(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 7u + seed);
    return v;
}

ConversationCheckpoint checkpoint(size_t tokens, uint8_t seed, int32_t first = 1000) {
    ConversationCheckpoint c;
    for (size_t i = 0; i < tokens; ++i) c.ids.push_back(first + int32_t(i));
    c.gdn = bytes_of(4099, seed);
    c.ple = bytes_of(77, seed + 1);
    c.tails = bytes_of(301, seed + 2);
    c.dead = bytes_of(64, seed + 3);
    c.block_pos = bytes_of(8, seed + 4);
    c.used = 42 + seed;
    return c;
}

// one stage's image: `tokens` live tokens from `first`, checkpoints at a half and a quarter of them, two K/V layers
SavedConversation image(size_t tokens, uint8_t seed, int32_t first = 1000, int64_t lo = 0, int64_t hi = 48) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = lo; s.layer_hi = hi;
    s.cvec = false;
    s.live = checkpoint(tokens, seed, first);
    s.checkpoints.push_back(checkpoint(tokens / 2, uint8_t(seed + 9), first));    // the deepest first
    s.checkpoints.push_back(checkpoint(tokens / 4, uint8_t(seed + 19), first));
    for (int layer = 0; layer < 2; ++layer) {
        ConversationKv kv;
        kv.format = 2; kv.cells = 1024; kv.heads = 2; kv.head_dim = 256; kv.page_size = 64; kv.pooled_rows = 9;
        kv.idx_dim = 128;
        kv.k = pattern(layer == 0 ? (17u << 20) + 5 : 4096, uint8_t(seed + layer));   // > 16 MiB: several segments
        kv.v = pattern(4096, uint8_t(seed + 10));
        kv.k_scale = pattern(512, uint8_t(seed + 20));
        kv.v_scale = pattern(512, uint8_t(seed + 30));
        kv.pooled = pattern(9 * 128 * 4, uint8_t(seed + 40));
        s.kv.push_back(std::move(kv));
    }
    return s;
}

// A conversation laid out as the engine's K/V is: 4-token pages of cells (4 KiB of K and of V per cell, 64 B of
// each scale), one indexer row per 4 tokens plus the spare one.  Cell i's bytes depend on i and on the turn that
// wrote it (`turn(i)`), so a continued conversation keeps its earlier bytes and a rewritten one does not.
// `checks`: (tokens, seed) of its checkpoints.
SavedConversation laid_out(int64_t tokens, uint8_t (*turn)(int64_t), uint8_t live_seed,
                           const std::vector<std::pair<size_t, uint8_t>>& checks) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = 0; s.layer_hi = 48;
    s.cvec = false;
    s.live = checkpoint((size_t) tokens, live_seed);
    for (const auto& c : checks) s.checkpoints.push_back(checkpoint(c.first, c.second));
    const int64_t cells = (tokens + 3) / 4 * 4, rows = tokens / 4 + 1;
    for (int layer = 0; layer < 2; ++layer) {
        ConversationKv kv;
        kv.format = 2; kv.cells = cells; kv.heads = 8; kv.head_dim = 256; kv.page_size = 4; kv.pooled_rows = rows;
        kv.idx_dim = 128;
        auto fill = [&](ConversationBuffer& b, size_t per_cell, uint8_t salt) {
            b.resize((size_t) cells * per_cell);
            b.visit(0, b.size(), [&](uint8_t* p, size_t c, size_t at) {
                for (size_t i = 0; i < c; ++i) {
                    const int64_t cell = (int64_t) ((at + i) / per_cell);
                    p[i] = uint8_t((at + i) * 131u + salt + layer + turn(cell) * 37u);
                }
                return true;
            });
        };
        fill(kv.k, 4096, 1); fill(kv.v, 4096, 2); fill(kv.k_scale, 64, 3); fill(kv.v_scale, 64, 4);
        kv.pooled.resize((size_t) rows * 512);
        kv.pooled.visit(0, kv.pooled.size(), [&](uint8_t* p, size_t c, size_t at) {
            for (size_t i = 0; i < c; ++i) {
                const int64_t row = (int64_t) ((at + i) / 512);
                p[i] = uint8_t((at + i) * 7u + layer + (row == rows - 1 ? 0xee : turn(row * 4) * 37u));
            }
            return true;
        });
        s.kv.push_back(std::move(kv));
    }
    return s;
}

// a layer split's image: the first stage's with one later stage's
SavedConversation split_image(size_t tokens, uint8_t seed, int32_t first = 1000) {
    SavedConversation s = image(tokens, seed, first, 0, 26);
    s.stage_images.push_back(image(tokens, uint8_t(seed + 50), first, 26, 48));
    return s;
}

bool same_checkpoint(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails &&
           a.dead == b.dead && a.block_pos == b.block_pos && a.used == b.used;
}

bool same(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
        !same_checkpoint(a.live, b.live) || a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size() ||
        a.stage_images.size() != b.stage_images.size())
        return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!same_checkpoint(a.checkpoints[i], b.checkpoints[i])) return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i]; const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
            !(x.k == y.k) || !(x.v == y.v) || !(x.k_scale == y.k_scale) || !(x.v_scale == y.v_scale) ||
            !(x.pooled == y.pooled)) return false;
    }
    for (size_t i = 0; i < a.stage_images.size(); ++i)
        if (!same(a.stage_images[i], b.stage_images[i])) return false;
    return true;
}

std::vector<int32_t> prompt_of(size_t tokens, int32_t first = 1000) {
    std::vector<int32_t> p;
    for (size_t i = 0; i < tokens; ++i) p.push_back(first + int32_t(i));
    return p;
}

size_t tier_files(const fs::path& dir) {
    size_t n = 0;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().filename().string().find(ConversationDiskTier::file_prefix()) != std::string::npos) ++n;
    return n;
}

void touch(const fs::path& p) { std::ofstream(p, std::ios::binary) << "x"; }

std::vector<std::string> lines;   // the tier's diagnostic lines
std::mutex lines_m;
ConversationDiskOptions opts(const fs::path& dir, uint64_t budget) {
    ConversationDiskOptions o;
    o.dir = dir.string();
    o.budget_bytes = budget;
    o.log = [](const std::string& l) { std::lock_guard<std::mutex> g(lines_m); lines.push_back(l); };
    return o;
}
bool logged(const char* what) {
    std::lock_guard<std::mutex> g(lines_m);
    for (const auto& l : lines) if (l.find(what) != std::string::npos) return true;
    return false;
}
}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("strata_conversation_disk_test_" + std::to_string(std::rand()));
    fs::remove_all(dir);
    const uint64_t GiB = 1ull << 30;
    std::string e;

    {   // off without a folder or a budget
        ConversationDiskTier off(opts("", GiB));
        check(off.open(e) && !off.enabled() && !off.put(image(10, 1)), "no folder: off, not an error");
        ConversationDiskTier off2(opts(dir, 0));
        check(off2.open(e) && !off2.enabled(), "no budget: off");
    }

    {   // start: this tier's leftovers (and their temporary files) go, other files stay
        fs::create_directories(dir);
        touch(dir / (std::string(ConversationDiskTier::file_prefix()) + "0123-1-0.bin"));
        touch(dir / ("." + std::string(ConversationDiskTier::file_prefix()) + "0123-1-0.bin.abcd.tmp"));
        touch(dir / "keep.bin");
        ConversationDiskTier tier(opts(dir, GiB));
        check(tier.open(e) && tier.enabled(), "open");
        check(tier_files(dir) == 0 && logged("2 stale files removed"), "open removes the tier's old and temporary files");
        check(fs::exists(dir / "keep.bin"), "open keeps other files");
        fs::remove(dir / "keep.bin");
    }

    {   // background write, match, read back exactly (a layer split: two files; every checkpoint)
        ConversationDiskTier tier(opts(dir, GiB));
        check(tier.open(e), "open 2");
        const SavedConversation expect = split_image(600, 1);
        check(tier.put(split_image(600, 1)), "put a split image");
        tier.wait_idle();
        check(tier.size() == 1 && tier_files(dir) == 2 && !tier.pending(), "written: one file per stage");
        check(logged("wrote 600 tokens") && logged("in 2 files"), "the write is logged");
        const auto live = tier.best(prompt_of(700), {}, false);
        check(live.tokens == 600 && live.live && live.id != 0, "best: the live state as a prefix");
        const auto ck = tier.best([] { auto p = prompt_of(300); p.push_back(7); return p; }(), {}, false);
        check(ck.tokens == 300 && !ck.live, "best: the deepest checkpoint as a prefix");
        check(tier.best([] { auto p = prompt_of(150); p.push_back(7); return p; }(), {}, false).tokens == 150,
              "best: an earlier checkpoint is kept too");
        const auto whole = tier.best(prompt_of(600), {}, false);
        check(whole.tokens == 300 && !whole.live, "best: the whole prompt held is no prefix (its last token is read)");
        check(tier.best(prompt_of(700), {}, true).tokens == 0, "best: another control-vector state does not match");
        check(tier.best(prompt_of(700, 5), {}, false).tokens == 0, "best: other tokens do not match");
        SavedConversation back;
        check(tier.take(live.id, live.tokens, back, e), "take");
        check(same(back, expect), "read back exactly: both stages, the live state and every checkpoint");
        check(tier.size() == 0 && tier.bytes() == 0 && tier_files(dir) == 0, "take removes the entry and its files");
        const auto st = tier.stats();
        check(st.queued == 1 && st.written == 1 && st.disk_hits == 1 && st.ram_hits == 0 && st.tokens_back == 600 &&
              st.bytes_read == st.bytes_written && st.bytes_written > (34u << 20), "counters");
        check(logged("disk hit: 600 tokens") && logged("totals:"), "the hit and the totals are logged");
    }

    {   // a queued image comes back from RAM: the writer is held at its first write
        std::promise<void> release;
        std::shared_future<void> gate = release.get_future().share();
        auto o = opts(dir, GiB);
        o.before_write = [gate] { gate.wait(); };
        ConversationDiskTier tier(std::move(o));
        check(tier.open(e), "open 3");
        check(tier.put(image(400, 2)), "put a (the writer takes it and waits)");
        check(tier.put(image(400, 3, 5000)), "put b (queued behind a)");
        check(tier.pending(), "pending while the writer waits");
        const auto b = tier.best(prompt_of(450, 5000), {}, false);
        check(b.tokens == 400, "b matches in the queue");
        SavedConversation back;
        check(tier.take(b.id, b.tokens, back, e) && same(back, image(400, 3, 5000)), "b back from RAM");
        check(tier.stats().ram_hits == 1 && logged("RAM hit: 400 tokens"), "a RAM hit, logged");
        release.set_value();
        tier.wait_idle();
        check(!tier.pending() && tier.size() == 1 && tier_files(dir) == 1, "a written after the release");
        check(tier.best(prompt_of(450, 5000), {}, false).tokens == 0, "b is gone from the tier");
    }

    {   // #879: clear() takes the conversations on the disk, their files too
        ConversationDiskTier tier(opts(dir, GiB));
        check(tier.open(e) && tier.put(image(400, 40)), "clear: put a");
        tier.wait_idle();
        check(tier.size() == 1 && tier_files(dir) == 1, "clear: a on the disk");
        check(tier.clear() == 1 && tier.size() == 0 && tier_files(dir) == 0 && logged("removed all 1 conversations"),
              "clear: a and its file gone");
        check(tier.best(prompt_of(450), {}, false).tokens == 0, "clear: nothing matches");
    }

    {   // the budget: the oldest goes for a new one; an image above the budget is refused
        ConversationDiskTier tier(opts(dir, 25ull << 20));   // one ~17 MiB image, not two
        check(tier.open(e), "open 4");
        check(tier.put(image(400, 4)), "first queued");
        tier.wait_idle();
        check(tier.size() == 1, "first fits");
        check(tier.put(image(500, 5, 5000)), "second queued");
        tier.wait_idle();
        check(tier.size() == 1 && tier.stats().evicted == 1, "second evicts the first");
        check(tier.best(prompt_of(450), {}, false).tokens == 0 && tier.best(prompt_of(600, 5000), {}, false).tokens == 500,
              "the newer one stays");
        ConversationDiskTier small(opts(dir, 1ull << 20));
        check(small.open(e) && !small.put(image(400, 6)) && small.size() == 0, "above the budget: refused");
    }
    check(tier_files(dir) == 0, "the tier removes its files when it ends");

    {   // a read the RAM preflight refuses (also the live state alone): nothing restored, the files are gone anyway
        ConversationDiskTier tier(opts(dir, GiB));
        check(tier.open(e) && tier.put(split_image(400, 10)), "put for a refused read");
        tier.wait_idle();
        const auto m = tier.best(prompt_of(450), {}, false);
        SavedConversation back;
        const bool ok = tier.take(m.id, m.tokens, back, e, [](uint64_t, std::string& why) { why = "no RAM"; return false; });
        check(!ok && back.kv.empty() && back.stage_images.empty() && tier.size() == 0 && tier_files(dir) == 0,
              "refused read: empty image, entry and files gone");
        check(tier.stats().read_failed == 1, "a failed read is counted");
    }
    {   // no RAM for the checkpoints, room for the live state: it comes back without them
        ConversationDiskTier tier(opts(dir, GiB));
        const SavedConversation original = image(400, 30);
        check(tier.open(e) && tier.put(image(400, 30)), "put for a lean read");
        tier.wait_idle();
        const auto m = tier.best(prompt_of(450), {}, false);
        std::vector<uint64_t> asked;
        SavedConversation back;
        check(m.live, "lean read: a live-state match");
        const bool ok = tier.take(m.id, m.tokens, back, e, [&](uint64_t need, std::string& why) {
            asked.push_back(need);
            if (asked.size() == 1) { why = "no RAM for all of it"; return false; }
            return need < asked[0];
        }, nullptr, m.live);
        check(ok && asked.size() == 2 && asked[1] < asked[0], "lean read: asked for less the second time");
        check(ok && back.checkpoints.empty() && same_checkpoint(back.live, original.live) &&
                  back.kv.size() == original.kv.size() && back.kv[0].k == original.kv[0].k,
              "lean read: the live state and the K/V exact, no checkpoints");
        check(tier.size() == 0 && tier_files(dir) == 0, "lean read: the files are spent");
    }
    {   // a checkpoint match the RAM refuses whole: no lean read (it would drop the checkpoint the request resumes
        // from), the plain refusal - the prompt is read again
        ConversationDiskTier tier(opts(dir, GiB));
        check(tier.open(e) && tier.put(image(400, 30)), "put for a checkpoint match");
        tier.wait_idle();
        auto prompt = prompt_of(300);
        prompt.push_back(7);   // leaves the conversation after its first checkpoint (200 tokens)
        const auto m = tier.best(prompt, {}, false);
        check(m.id != 0 && !m.live && m.tokens == 200, "checkpoint match: 200 tokens, not the live state");
        size_t asked = 0;
        SavedConversation back;
        const bool ok = tier.take(m.id, m.tokens, back, e, [&](uint64_t, std::string& why) {
            ++asked;
            if (asked == 1) { why = "no RAM for all of it"; return false; }
            return true;
        }, nullptr, m.live);
        check(!ok && asked == 1 && back.kv.empty() && back.checkpoints.empty(),
              "checkpoint match: refused, no lean read tried");
        check(e.find("no RAM for all of it") != std::string::npos, "checkpoint match: the refusal says why");
        check(tier.size() == 0 && tier_files(dir) == 0, "checkpoint match: the files are spent");
    }

    {   // put_streamed: the K/V read from where it lives (here: the image's own buffers) gives the same files
        auto sources_of = [](const SavedConversation& img) {
            std::vector<SessionKvSource> out;
            for (const auto& kv : img.kv) {
                SessionKvSource s;
                s.format = kv.format; s.cells = kv.cells; s.heads = kv.heads; s.head_dim = kv.head_dim;
                s.page_size = kv.page_size; s.pooled_rows = kv.pooled_rows; s.idx_dim = kv.idx_dim;
                const ConversationBuffer* parts[5] = {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled};
                for (size_t p = 0; p < 5; ++p) s.sizes[p] = parts[p]->size();
                s.read = [parts](size_t part, size_t offset, void* dst, size_t n) {
                    return parts[part]->read(dst, offset, n);
                };
                out.push_back(std::move(s));
            }
            return out;
        };
        const SavedConversation expect = split_image(500, 30, 9000);
        std::vector<std::vector<SessionKvSource>> sources = {sources_of(expect), sources_of(expect.stage_images[0])};
        SavedConversation meta = split_image(500, 30, 9000);
        meta.kv.clear();
        meta.stage_images[0].kv.clear();
        ConversationDiskTier tier(opts(dir, GiB));
        check(tier.open(e) && tier.put_streamed(meta, sources), "put_streamed: written");
        check(meta.stage_images.size() == 1 && meta.checkpoints.size() == 2, "put_streamed: the image is given back");
        check(tier.size() == 1 && tier_files(dir) == 2 && logged("streamed from the GPU"), "put_streamed: two files, logged");
        const auto m = tier.best(prompt_of(600, 9000), {}, false);
        SavedConversation back;
        check(m.tokens == 500 && tier.take(m.id, m.tokens, back, e) && same(back, expect),
              "put_streamed: read back exactly the image the sources came from");
    }

    {   // take with K/V targets: the files' K/V streamed into them (here: CPU buffers), the rest in `out`
        using Bytes = std::vector<uint8_t>;
        auto targets_for = [](const SavedConversation& img, std::vector<std::array<Bytes, 5>>& into) {
            into.assign(img.kv.size(), {});
            std::vector<SessionKvTarget> out;
            for (size_t j = 0; j < img.kv.size(); ++j) {
                const auto& kv = img.kv[j];
                SessionKvTarget t;
                t.format = kv.format; t.cells = kv.cells; t.heads = kv.heads; t.head_dim = kv.head_dim;
                t.page_size = kv.page_size; t.pooled_rows = kv.pooled_rows; t.idx_dim = kv.idx_dim;
                const ConversationBuffer* parts[5] = {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled};
                for (size_t p = 0; p < 5; ++p) { t.sizes[p] = parts[p]->size(); into[j][p].assign(t.sizes[p], 0); }
                std::array<Bytes, 5>* layer = &into[j];
                t.write = [layer](size_t part, size_t offset, const void* src, size_t n) {
                    std::memcpy((*layer)[part].data() + offset, src, n);
                    return true;
                };
                out.push_back(std::move(t));
            }
            return out;
        };
        auto bytes_of_buffer = [](const ConversationBuffer& b) {
            Bytes v(b.size());
            b.read(v.data(), 0, v.size());
            return v;
        };
        const SavedConversation expect = split_image(500, 31, 11000);
        ConversationDiskTier tier(opts(dir, GiB));
        check(tier.open(e) && tier.put(split_image(500, 31, 11000)), "streamed take: put");
        tier.wait_idle();
        std::vector<std::array<Bytes, 5>> got0, got1;
        std::vector<std::vector<SessionKvTarget>> kv = {targets_for(expect, got0), targets_for(expect.stage_images[0], got1)};
        const auto m = tier.best(prompt_of(600, 11000), {}, false);
        SavedConversation meta;
        check(m.tokens == 500 && !m.queued && m.bytes > 0 && tier.take(m.id, m.tokens, meta, e, {}, &kv),
              "streamed take: read");
        bool same_kv = true;
        for (size_t j = 0; j < expect.kv.size(); ++j) {
            const ConversationBuffer* parts[5] = {&expect.kv[j].k, &expect.kv[j].v, &expect.kv[j].k_scale,
                                                  &expect.kv[j].v_scale, &expect.kv[j].pooled};
            for (size_t p = 0; p < 5; ++p) same_kv = same_kv && got0[j][p] == bytes_of_buffer(*parts[p]);
        }
        check(same_kv && logged("streamed into the GPU"), "streamed take: the targets hold the K/V exactly");
        check(meta.kv.size() == expect.kv.size() && meta.kv[0].k.size() == 0 &&
              same_checkpoint(meta.live, expect.live) && meta.checkpoints.size() == expect.checkpoints.size() &&
              meta.stage_images.size() == 1, "streamed take: the running state and checkpoints in RAM, no K/V");
        check(tier.put(image(400, 32, 12000)), "streamed take: put a second");
        tier.wait_idle();
        const SavedConversation other = image(400, 32, 12000);
        std::vector<std::array<Bytes, 5>> bad;
        std::vector<std::vector<SessionKvTarget>> wrong = {targets_for(other, bad)};
        wrong[0][0].sizes[0] += 1;
        const auto m2 = tier.best(prompt_of(450, 12000), {}, false);
        check(!tier.take(m2.id, m2.tokens, meta, e, {}, &wrong) && e.find("size differs") != std::string::npos,
              "streamed take: a K/V part of another size is refused");

        // a whole read the RAM refuses, with `kept`: the entry and its files stay, and a streamed take reads them
        const SavedConversation third = split_image(500, 33, 13000);
        check(tier.put(split_image(500, 33, 13000)), "kept: put");
        tier.wait_idle();
        const auto m3 = tier.best(prompt_of(600, 13000), {}, false);
        const size_t files_before = tier_files(dir);
        const uint64_t failed_before = tier.stats().read_failed;
        bool kept = false;
        SavedConversation whole;
        check(!tier.take(m3.id, m3.tokens, whole, e, [](uint64_t, std::string& why) { why = "no RAM"; return false; },
                         nullptr, m3.live, &kept) && kept && whole.kv.empty(),
              "kept: a whole read refused for RAM keeps the entry");
        check(tier.size() == 1 && tier_files(dir) == files_before && tier.stats().read_refused == 1 &&
                  tier.stats().read_failed == failed_before && logged("kept for a streamed read"),
              "kept: its files stay, counted as refused (not failed), logged");
        const auto again = tier.best(prompt_of(600, 13000), {}, false);
        check(again.id == m3.id && again.tokens == 500, "kept: best() finds it again");
        std::vector<std::array<Bytes, 5>> k0, k1;
        std::vector<std::vector<SessionKvTarget>> kv3 = {targets_for(third, k0), targets_for(third.stage_images[0], k1)};
        SavedConversation meta3;
        bool kept_streamed = true;
        check(tier.take(again.id, again.tokens, meta3, e, {}, &kv3, again.live, &kept_streamed) && !kept_streamed &&
                  same_checkpoint(meta3.live, third.live) && k0[0][0] == bytes_of_buffer(third.kv[0].k),
              "kept: the streamed take reads it, the K/V exact");
        check(tier.size() == 0 && tier_files(dir) == 0, "kept: the files are spent after the streamed read");
        // a refusal after the read began (the K/V target refuses a write) is not kept
        check(tier.put(image(400, 34, 14000)), "not kept: put");
        tier.wait_idle();
        const auto m4 = tier.best(prompt_of(450, 14000), {}, false);
        const SavedConversation fourth = image(400, 34, 14000);
        std::vector<std::array<Bytes, 5>> k4;
        std::vector<std::vector<SessionKvTarget>> kv4 = {targets_for(fourth, k4)};
        kv4[0][0].write = [](size_t, size_t, const void*, size_t) { return false; };
        bool kept4 = true;
        check(!tier.take(m4.id, m4.tokens, meta3, e, {}, &kv4, m4.live, &kept4) && !kept4 && tier.size() == 0 &&
                  tier_files(dir) == 0, "not kept: a read that failed after it began is gone");
    }
    {   // persist: a kept entry's index is written again, so the next start still has it
        auto o = opts(dir, GiB);
        o.persist = true;
        o.identity = {3, 7};
        {
            ConversationDiskTier tier(o);
            check(tier.open(e) && tier.put(image(400, 35, 15000)), "kept, persist: put");
            tier.wait_idle();
            const auto m = tier.best(prompt_of(450, 15000), {}, false);
            bool kept = false;
            SavedConversation back;
            check(!tier.take(m.id, m.tokens, back, e, [](uint64_t, std::string& why) { why = "no RAM"; return false; },
                             nullptr, m.live, &kept) && kept && tier.size() == 1, "kept, persist: kept");
        }
        ConversationDiskTier next(o);
        check(next.open(e) && next.size() == 1 && next.best(prompt_of(450, 15000), {}, false).tokens == 400,
              "kept, persist: the next start takes it in");
        const auto m = next.best(prompt_of(450, 15000), {}, false);
        SavedConversation back;
        check(next.take(m.id, m.tokens, back, e) && same(back, image(400, 35, 15000)) && tier_files(dir) == 0,
              "kept, persist: read back exactly");
    }

    // --conversation-disk-delta: a conversation read back and parked again is written as what changed since
    {
        auto o = opts(dir, GiB);
        o.delta_chain = 4;
        ConversationDiskTier tier(o);
        check(tier.open(e), "delta: open");
        static auto t0 = [](int64_t) -> uint8_t { return 0; };
        static auto t1 = [](int64_t c) -> uint8_t { return c < 3000 ? 0 : 1; };
        static auto t2 = [](int64_t c) -> uint8_t { return c < 3000 ? 0 : c < 3300 ? 1 : 2; };
        static auto td = [](int64_t c) -> uint8_t { return c < 3000 ? 0 : c < 3100 ? 1 : 4; };
        static auto te = [](int64_t c) -> uint8_t { return c >= 2500 && c < 2508 ? 9 : c < 3000 ? 0 : c < 3100 ? 1 : 4; };
        const SavedConversation a = laid_out(3000, t0, 1, {{1500, 11}});
        uint64_t chain = 0;
        SavedConversation back;
        // no parent: a whole file, as without deltas
        SavedConversation lone = laid_out(1000, t0, 1, {});
        lone.disk_parent = {0, 1000};
        check(tier.put(std::move(lone)), "delta: put a conversation without a parent");
        tier.wait_idle();
        check(tier.stats().deltas == 0 && tier_files(dir) == 1, "delta: no parent, a whole file");
        tier.clear();
        check(tier.put(laid_out(3000, t0, 1, {{1500, 11}})), "delta: put A");
        tier.wait_idle();
        const uint64_t whole_a = tier.stats().bytes_written;
        auto m = tier.best(prompt_of(3001), {}, false);
        check(tier.take(m.id, m.tokens, back, e, {}, nullptr, m.live, nullptr, &chain) && same(back, a) && chain != 0 &&
                  tier.retained() == 1 && tier_files(dir) == 1,
              "delta: A read back; its file kept as the base of A's next park");
        // B: A continued to 3300 tokens (its first 3000 unchanged), A's checkpoint kept, one more
        SavedConversation b = laid_out(3300, t1, 2, {{1500, 11}, {3200, 12}});
        b.disk_parent = {chain, 3000};
        const uint64_t before = tier.stats().bytes_written;
        check(tier.put(SavedConversation(b)), "delta: put B with its parent");
        tier.wait_idle();
        const uint64_t delta_b = tier.stats().bytes_written - before;
        check(tier.stats().deltas == 1 && delta_b < whole_a / 4 && tier.retained() == 0 && tier.size() == 1 &&
                  tier_files(dir) == 2 && logged("as delta 1 of its chain"),
              "delta: B written as a delta of A's file (a fraction of a whole file), the base continued");
        m = tier.best(prompt_of(3301), {}, false);
        check(m.tokens == 3300 && tier.take(m.id, m.tokens, back, e, {}, nullptr, m.live, nullptr, &chain) &&
                  same(back, b) && chain != 0,
              "delta: B read back exactly from its chain (base + delta)");
        // C: B continued to 3500 tokens - a chain of three
        SavedConversation c = laid_out(3500, t2, 3, {{1500, 11}, {3200, 12}, {3400, 13}});
        c.disk_parent = {chain, 3300};
        check(tier.put(SavedConversation(c)), "delta: put C");
        tier.wait_idle();
        check(tier.stats().deltas == 2 && tier_files(dir) == 3, "delta: C a second delta (three files)");
        m = tier.best(prompt_of(3501), {}, false);
        check(tier.take(m.id, m.tokens, back, e, {}, nullptr, m.live, nullptr, &chain) && same(back, c),
              "delta: C read back exactly from three files");
        // D: a rewind - C cut back to 3200 tokens, rewritten from token 3100: the prefix ends where it changed
        SavedConversation d = laid_out(3200, td, 4, {{1500, 11}});
        d.disk_parent = {chain, 3100};
        check(tier.put(SavedConversation(d)), "delta: put D (a rewind)");
        tier.wait_idle();
        m = tier.best(prompt_of(3201), {}, false);
        check(tier.stats().deltas == 3 && tier.take(m.id, m.tokens, back, e, {}, nullptr, m.live, nullptr, &chain) &&
                  same(back, d),
              "delta: D (shorter, rewritten from token 3100) read back exactly");
        // E: a wrong bound - cells 2500..2507 rewritten though 3100 tokens are claimed unchanged: the prefix check
        // finds the MiB they are in, and the delta takes from the chain only what matches
        SavedConversation ex = laid_out(3300, te, 5, {});
        ex.disk_parent = {chain, 3100};
        check(tier.put(SavedConversation(ex)), "delta: put E (a wrong bound)");
        tier.wait_idle();
        m = tier.best(prompt_of(3301), {}, false);
        check(tier.stats().deltas == 4 && tier.stats().prefix_mismatch >= 1 && logged("prefix check") &&
                  tier.take(m.id, m.tokens, back, e, {}, nullptr, m.live, nullptr, &chain) && same(back, ex),
              "delta: a wrong bound is caught by the prefix check, and the image still reads back exactly");
        // F: the chain is full (a base and 4 deltas): the next park is a whole file, a new base
        SavedConversation f = laid_out(3400, te, 6, {});
        f.disk_parent = {chain, 3300};
        check(tier.put(SavedConversation(f)), "delta: put F");
        tier.wait_idle();
        check(tier.stats().deltas == 4 && logged("deltas already: a new base") && tier_files(dir) == 1 &&
                  tier.retained() == 0,
              "delta: past the cap a whole file again; the old chain's files gone");
        m = tier.best(prompt_of(3401), {}, false);
        check(tier.take(m.id, m.tokens, back, e, {}, nullptr, m.live, nullptr, &chain) && same(back, f), "delta: F back");
        // an unknown parent: a whole file
        SavedConversation g = laid_out(3500, te, 7, {});
        g.disk_parent = {chain + 999, 3400};
        check(tier.put(SavedConversation(g)), "delta: put G");
        tier.wait_idle();
        check(tier.stats().deltas == 4, "delta: an unknown parent gives a whole file");
        tier.clear();
        check(tier_files(dir) == 0 && tier.retained() == 0 && tier.bytes() == 0, "delta: clear() leaves nothing");
    }
    {   // a streamed park (put_streamed) continues a chain too; the budget removes a base nobody continued first
        auto o = opts(dir, GiB);
        o.delta_chain = 4;
        ConversationDiskTier tier(o);
        check(tier.open(e), "delta stream: open");
        static auto t0 = [](int64_t) -> uint8_t { return 0; };
        static auto t1 = [](int64_t c) -> uint8_t { return c < 1000 ? 0 : 1; };
        check(tier.put(laid_out(1000, t0, 1, {{500, 11}})), "delta stream: put A");
        tier.wait_idle();
        uint64_t chain = 0;
        SavedConversation back;
        auto m = tier.best(prompt_of(1001), {}, false);
        check(tier.take(m.id, m.tokens, back, e, {}, nullptr, m.live, nullptr, &chain), "delta stream: A back");
        const SavedConversation b = laid_out(1600, t1, 2, {{500, 11}});
        SavedConversation meta = b;
        meta.kv.clear();
        meta.disk_parent = {chain, 1000};
        std::vector<std::vector<SessionKvSource>> sources(1);
        for (const auto& k : b.kv) {
            SessionKvSource s;
            s.format = k.format; s.cells = k.cells; s.heads = k.heads; s.head_dim = k.head_dim;
            s.page_size = k.page_size; s.pooled_rows = k.pooled_rows; s.idx_dim = k.idx_dim;
            const std::array<const ConversationBuffer*, 5> parts = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
            for (size_t p = 0; p < 5; ++p) s.sizes[p] = parts[p]->size();
            s.read = [parts](size_t part, size_t offset, void* dst, size_t n) { return parts[part]->read(dst, offset, n); };
            sources[0].push_back(std::move(s));
        }
        check(tier.put_streamed(meta, sources) && tier.stats().deltas == 1, "delta stream: a streamed park as a delta");
        m = tier.best(prompt_of(1601), {}, false);
        check(tier.take(m.id, m.tokens, back, e, {}, nullptr, m.live, nullptr, &chain) && same(back, b),
              "delta stream: read back exactly");
        const size_t retained = tier.retained();
        check(retained == 1, "delta stream: B's chain kept as a base");
        // a whole conversation bigger than what is left: the base nobody continued goes first
        auto small = opts(dir, 50ull << 20);
        small.delta_chain = 4;
        ConversationDiskTier tight(small);
        check(tight.open(e) && tight.put(laid_out(1000, t0, 1, {})), "delta budget: put");
        tight.wait_idle();
        m = tight.best(prompt_of(1001), {}, false);
        check(tight.take(m.id, m.tokens, back, e, {}, nullptr, m.live, nullptr, &chain) && tight.retained() == 1,
              "delta budget: a base kept");
        check(tight.put(laid_out(2600, t0, 3, {})), "delta budget: a bigger conversation");
        tight.wait_idle();
        check(tight.retained() == 0 && tight.size() == 1, "delta budget: the base went for it");
        tight.clear();
        tier.clear();
    }
    {   // persist: a chain is named by its index and read back after a restart
        auto o = opts(dir, GiB);
        o.delta_chain = 4;
        o.persist = true;
        o.identity = {9, 7};
        static auto t0 = [](int64_t) -> uint8_t { return 0; };
        static auto t1 = [](int64_t c) -> uint8_t { return c < 1000 ? 0 : 1; };
        const SavedConversation b = laid_out(1600, t1, 2, {{500, 11}});
        {
            ConversationDiskTier tier(o);
            check(tier.open(e) && tier.put(laid_out(1000, t0, 1, {{500, 11}})), "delta persist: put A");
            tier.wait_idle();
            uint64_t chain = 0;
            SavedConversation back;
            auto m = tier.best(prompt_of(1001), {}, false);
            check(tier.take(m.id, m.tokens, back, e, {}, nullptr, m.live, nullptr, &chain), "delta persist: A back");
            SavedConversation pb = b;
            pb.disk_parent = {chain, 1000};
            check(tier.put(std::move(pb)), "delta persist: put B");
            tier.wait_idle();
            check(tier.stats().deltas == 1, "delta persist: B a delta");
        }
        ConversationDiskTier next(o);
        SavedConversation back;
        check(next.open(e) && next.size() == 1, "delta persist: the next start takes the chain in");
        auto m = next.best(prompt_of(1601), {}, false);
        check(m.tokens == 1600 && next.take(m.id, m.tokens, back, e) && same(back, b), "delta persist: B read back exactly");
        next.clear();
        check(tier_files(dir) == 0, "delta persist: nothing left");
    }

    {   // the RAM cache's evictions go to the disk tier (set_on_evict)
        ConversationDiskTier tier(opts(dir, GiB));
        check(tier.open(e), "open 6");
        ConversationCache cache(GiB, 1);
        cache.set_on_evict([&](SavedConversation&& s) { tier.put(std::move(s)); });
        check(cache.put(image(400, 11)), "park a");
        check(cache.put(image(400, 12, 8000)), "park b: a leaves RAM");
        tier.wait_idle();
        check(cache.size() == 1 && cache.evictions() == 1 && tier.size() == 1, "a is on the disk");
        check(tier.best(prompt_of(450), {}, false).tokens == 400, "a matches on the disk");
        check(cache.best(prompt_of(450, 8000), {}, false).tokens == 400, "b stays in RAM");
    }

    {   // persist: the files outlive the tier, and the next tier with the same identity takes them in
        auto persistent = [&](uint64_t model) {
            auto o = opts(dir, GiB);
            o.persist = true;
            o.identity = {model, 7};
            return o;
        };
        const std::string prefix = ConversationDiskTier::file_prefix();
        {
            ConversationDiskTier tier(persistent(1));
            check(tier.open(e) && tier.persistent(), "persist: open");
            check(tier.put(split_image(500, 20)) && tier.put(image(400, 21, 6000)), "persist: two queued");
        }   // the end waits for the queue and keeps the files
        check(tier_files(dir) == 5 && logged("2 conversations kept on the disk"), "persist: 3 images + 2 indexes kept");
        touch(dir / ("." + prefix + "x-1-0.bin.abcd.tmp"));
        touch(dir / (prefix + "orphan-9-0.bin"));
        {
            ConversationDiskTier tier(persistent(1));
            check(tier.open(e) && tier.size() == 2 && tier_files(dir) == 5, "persist: taken in, leftovers removed");
            check(logged("2 conversations from earlier runs"), "persist: the start is logged");
            const auto m = tier.best(prompt_of(600), {}, false);
            check(m.tokens == 500 && m.live, "persist: matches after a restart");
            SavedConversation back;
            const bool read = tier.take(m.id, m.tokens, back, e);
            if (!read) std::fprintf(stderr, "take: %s\n", e.c_str());
            check(read, "persist: read back after a restart");
            check(same(back, split_image(500, 20)), "persist: the image is exactly the one written");
            check(tier_files(dir) == 2, "persist: its images and index are gone");
        }
        {
            ConversationDiskTier other(persistent(2));   // another model, engine version or settings
            check(other.open(e) && other.size() == 0 && tier_files(dir) == 0 && logged("1 refused"),
                  "persist: another identity's files are refused and removed");
        }
        {
            ConversationDiskTier tier(persistent(3));
            check(tier.open(e) && tier.put(image(300, 22)), "persist: put for a damaged index");
        }
        for (const auto& f : fs::directory_iterator(dir))
            if (f.path().extension() == ".idx") fs::resize_file(f.path(), 20);
        {
            ConversationDiskTier tier(persistent(3));
            check(tier.open(e) && tier.size() == 0 && tier_files(dir) == 0, "persist: a damaged index is refused");
        }
        {
            ConversationDiskTier tier(persistent(4));
            check(tier.open(e) && tier.put(image(400, 23)) && tier.put(image(400, 24, 6000)),
                  "persist: put a and b, the same length");
        }
        std::vector<fs::path> bins;   // a's image first: the names differ only in the id
        for (const auto& f : fs::directory_iterator(dir))
            if (f.path().extension() == ".bin") bins.push_back(f.path());
        std::sort(bins.begin(), bins.end());
        check(bins.size() == 2, "persist: two images");
        fs::copy_file(bins[0], bins[1], fs::copy_options::overwrite_existing);   // b's index now names a's image
        {
            ConversationDiskTier tier(persistent(4));
            check(tier.open(e) && tier.size() == 2, "persist: both taken in");
            const auto m = tier.best(prompt_of(450, 6000), {}, false);
            check(m.tokens == 400, "persist: b matches by its index");
            SavedConversation back;
            check(!tier.take(m.id, m.tokens, back, e) && back.kv.empty() && logged("hold another conversation"),
                  "persist: a's image under b's index is refused, never restored as b");
            check(tier.best(prompt_of(450), {}, false).tokens == 400, "persist: a is still there");
        }
        {   // max_age_s (local): by the index's write time, at the next start
            ConversationDiskTier tier(persistent(5));
            check(tier.open(e) && tier.put(image(300, 25)), "persist age: put");
        }
        {
            auto o = persistent(5);
            o.max_age_s = 3600;
            ConversationDiskTier tier(std::move(o));
            check(tier.open(e) && tier.best(prompt_of(350), {}, false).tokens == 300,
                  "persist age: younger than the limit, taken in");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2100));   // the index holds whole seconds
        {
            auto o = persistent(5);
            o.max_age_s = 1;
            ConversationDiskTier tier(std::move(o));
            check(tier.open(e) && tier.best(prompt_of(350), {}, false).tokens == 0 &&
                      logged("1 older than the age limit removed"),
                  "persist age: older than the limit, removed");
        }
    }

    fs::remove_all(dir);
    std::printf("conversation_disk_test: %d checks passed\n", checks);
    return 0;
}
