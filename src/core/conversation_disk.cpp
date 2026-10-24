// src/core/conversation_disk.cpp - the disk tier of the parked conversations.  See the header.
#include "strata/core/conversation_disk.hpp"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <set>
#include <system_error>
#include <utility>
#if defined(__linux__)
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace strata::core {
namespace {

namespace fs = std::filesystem;

// The second half of every file's identity: with this run's random number as the first, a file another run (or
// another tier) wrote is refused on read.
constexpr uint64_t kTierConfig = 0x5354524154414449ull;   // "STRATADI"

// ids and images only: the part of a checkpoint best() matches against
ConversationCheckpoint key_of(const ConversationCheckpoint& c) {
    ConversationCheckpoint k;
    k.ids = c.ids;
    k.imgs = c.imgs;
    return k;
}

// The payload is the conversation its index names: the same keys and control-vector state, in every stage.
bool holds(const SavedConversation& s, const ConversationCheckpoint& live,
           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) {
    auto same = [](const ConversationCheckpoint& x, const ConversationCheckpoint& y) {
        return x.ids == y.ids && x.imgs == y.imgs;
    };
    if (s.cvec != cvec || !same(s.live, live) || s.checkpoints.size() != checkpoints.size()) return false;
    for (size_t i = 0; i < checkpoints.size(); ++i)
        if (!same(s.checkpoints[i], checkpoints[i])) return false;
    for (const auto& stage : s.stage_images)
        if (!holds(stage, live, checkpoints, cvec)) return false;
    return true;
}

// a file of this tier (also a temporary one session_file_write left behind: ".<name>.<hex>.tmp")
bool tier_file(const std::string& name) {
    const std::string prefix = ConversationDiskTier::file_prefix();
    return name.rfind(prefix, 0) == 0 || name.rfind("." + prefix, 0) == 0;
}

void remove_file(const std::string& path) {
    std::error_code ec;
    fs::remove(fs::path(path), ec);
}

// A K/V layer of an image as a write's source (the file bytes are the same as writing the image itself).  `kv` must
// outlive the sources.
std::vector<SessionKvSource> kv_sources(const std::vector<ConversationKv>& kv) {
    std::vector<SessionKvSource> out;
    for (const auto& k : kv) {
        SessionKvSource src;
        src.format = k.format; src.cells = k.cells; src.heads = k.heads; src.head_dim = k.head_dim;
        src.page_size = k.page_size; src.pooled_rows = k.pooled_rows; src.idx_dim = k.idx_dim;
        const std::array<const ConversationBuffer*, 5> parts = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
        for (size_t p = 0; p < 5; ++p) src.sizes[p] = parts[p]->size();
        src.read = [parts](size_t part, size_t offset, void* dst, size_t n) { return parts[part]->read(dst, offset, n); };
        out.push_back(std::move(src));
    }
    return out;
}

// The bytes of K/V part `p` that hold only tokens [0, unchanged): whole pages of cells (K, V and their scales are
// stored cell after cell, page by page), and the completed indexer rows (never the moving spare row).  One page less:
// the draft layer keeps a token less, and the partial page is rewritten anyway.  The same rule as the RAM cache's
// reuse (conversation_kv_save); 0 when the layer does not have that shape.
uint64_t stable_bytes(const SessionKvSource& s, size_t p, int64_t unchanged, int64_t idx_block) {
    if (unchanged <= 0 || s.cells <= 0 || s.page_size <= 0) return 0;
    const int64_t u = std::max<int64_t>(0, unchanged - s.page_size);
    const int64_t whole = std::min<int64_t>((u / s.page_size) * s.page_size, s.cells);
    if (p < 4) {
        if (s.sizes[p] % (size_t) s.cells) return 0;
        return (uint64_t) (s.sizes[p] / (size_t) s.cells) * (uint64_t) whole;
    }
    if (s.pooled_rows <= 1 || idx_block <= 0 || s.idx_dim <= 0) return 0;
    const uint64_t row = (uint64_t) s.idx_dim * 4;
    if (s.sizes[4] != row * (uint64_t) s.pooled_rows) return 0;
    return row * (uint64_t) std::min<int64_t>(u / idx_block, s.pooled_rows - 1);
}

constexpr size_t kMaxRetained = 8;   // bases kept for a next delta: the oldest goes past this many

// The index file of a persistent entry: native-endian fields (the session files refuse a big-endian build too),
// counts before arrays.  Version 2 lists each image's chain (its files and what a delta of it needs); version 1
// (one file per image, no chain data) is still taken in: its next park is a whole file.
constexpr uint64_t kIndexMagic = 0x3158444943525453ull;    // "STRCIDX1"
constexpr uint64_t kIndexMagic2 = 0x3258444943525453ull;   // "STRCIDX2"

struct IndexWriter {
    std::string b;
    template<class T> void put(T v) { b.append(reinterpret_cast<const char*>(&v), sizeof v); }
    void str(const std::string& s) { put<uint32_t>((uint32_t) s.size()); b += s; }
    void key(const ConversationCheckpoint& c) {
        put<uint32_t>((uint32_t) c.ids.size());
        b.append(reinterpret_cast<const char*>(c.ids.data()), c.ids.size() * sizeof(int32_t));
        put<uint32_t>((uint32_t) c.imgs.size());
        for (const auto& i : c.imgs) { put<int64_t>(i.start); put<uint64_t>(i.hash); }
    }
};

struct IndexReader {
    const std::string& b;
    size_t at = 0;
    bool ok = true;
    size_t left() const { return b.size() - at; }
    template<class T> T get() {
        T v{};
        if (left() < sizeof v) { ok = false; return v; }
        std::memcpy(&v, b.data() + at, sizeof v);
        at += sizeof v;
        return v;
    }
    std::string str() {
        const uint32_t n = get<uint32_t>();
        if (!ok || left() < n) { ok = false; return {}; }
        at += n;
        return b.substr(at - n, n);
    }
    bool key(ConversationCheckpoint& c) {
        const uint32_t n = get<uint32_t>();
        if (!ok || left() / sizeof(int32_t) < n) return ok = false;
        c.ids.resize(n);
        std::memcpy(c.ids.data(), b.data() + at, n * sizeof(int32_t));
        at += n * sizeof(int32_t);
        const uint32_t m = get<uint32_t>();
        if (!ok || left() / 16 < m) return ok = false;
        for (uint32_t i = 0; i < m; ++i) c.imgs.push_back({get<int64_t>(), get<uint64_t>()});
        return ok;
    }
};

double mib(uint64_t bytes) { return double(bytes) / (1024.0 * 1024.0); }
double mbps(uint64_t bytes, double ms) { return ms > 0 ? double(bytes) / 1e6 / (ms / 1000.0) : 0.0; }
double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

#if defined(__GNUC__)
std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
#endif
std::string fmt(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

}  // namespace

ConversationDiskTier::ConversationDiskTier(ConversationDiskOptions options) : options_(std::move(options)) {
    std::random_device rd;
    nonce_ = ((uint64_t) rd() << 32) ^ rd() ^ (uint64_t) std::chrono::system_clock::now().time_since_epoch().count();
}

ConversationDiskTier::~ConversationDiskTier() {
    if (open_ && options_.persist) wait_idle();   // what is queued belongs to the next start
    size_t dropped = 0;
    {
        std::lock_guard<std::mutex> lock(m_);
        stop_ = true;
        dropped = queue_.size();
        queue_.clear();   // not written: dropped, as the RAM cache would have
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    std::lock_guard<std::mutex> lock(m_);
    if (!open_) return;
    while (!retained_.empty()) drop_retained_locked(0);   // no index names them: a next start would not take them in
    if (options_.persist) {
        log(fmt("end: %zu conversations kept on the disk for the next start (%.0f MiB)", entries_.size(), mib(bytes_)));
        return;
    }
    if (dropped || !entries_.empty())
        log(fmt("end: %zu queued dropped, %zu conversations removed from the disk", dropped, entries_.size()));
    while (!entries_.empty()) remove_locked(0, true);
}

SessionFileIdentity ConversationDiskTier::file_identity() const {
    return options_.persist ? options_.identity : SessionFileIdentity{nonce_, kTierConfig};
}

bool ConversationDiskTier::write_index(Entry& e, std::string& error) const {
    char name[96];
    std::snprintf(name, sizeof name, "%s%016llx-%llu.idx", file_prefix(), (unsigned long long) nonce_,
                  (unsigned long long) e.id);
    const fs::path path = fs::path(options_.dir) / name;
    const fs::path temp = fs::path(options_.dir) / ("." + std::string(name) + ".tmp");
    if (e.written_s <= 0)   // a kept entry's index is written again: its files are as old as before
        e.written_s = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    const SessionFileIdentity id = file_identity();
    IndexWriter out;
    out.put(kIndexMagic2);
    out.put(id.model);
    out.put(id.config);
    out.put<int64_t>(e.written_s);
    out.put<uint8_t>(e.cvec ? 1 : 0);
    out.put<uint64_t>(e.bytes);
    out.put<uint32_t>((uint32_t) e.chains.size());
    for (const auto& ch : e.chains) {
        out.put<uint32_t>((uint32_t) ch.files.size());
        for (size_t i = 0; i < ch.files.size(); ++i) {
            out.str(fs::path(ch.files[i]).filename().string());
            out.put<uint64_t>(i < ch.hash.size() ? ch.hash[i] : 0);
            out.put<uint64_t>(i < ch.payload.size() ? ch.payload[i] : 0);
            out.put<uint64_t>(i < ch.file_bytes.size() ? ch.file_bytes[i] : 0);
        }
        out.put<uint32_t>((uint32_t) ch.ck_hash.size());
        for (uint64_t h : ch.ck_hash) out.put<uint64_t>(h);
        out.put<uint32_t>((uint32_t) ch.part_len.size());
        for (const auto& len : ch.part_len) for (uint64_t v : len) out.put<uint64_t>(v);
        out.put<uint32_t>((uint32_t) ch.chunks.size());
        for (const auto& layer : ch.chunks)
            for (const auto& part : layer.part) {
                out.put<uint32_t>((uint32_t) part.size());
                for (uint64_t h : part) out.put<uint64_t>(h);
            }
    }
    out.key(e.live);
    out.put<uint32_t>((uint32_t) e.checkpoints.size());
    for (const auto& c : e.checkpoints) out.key(c);
    {
        std::ofstream f(temp, std::ios::binary | std::ios::trunc);
        f.write(out.b.data(), (std::streamsize) out.b.size());
        if (!f) {
            error = "cannot write the index " + temp.string();
            remove_file(temp.string());
            return false;
        }
    }
    std::error_code ec;
    fs::rename(temp, path, ec);   // a reader sees the whole index or none
    if (ec) {
        error = "cannot publish the index: " + ec.message();
        remove_file(temp.string());
        return false;
    }
    e.index = path.string();
    return true;
}

bool ConversationDiskTier::read_index(const std::string& path, Entry& e) const {
    std::ifstream f(fs::path(path), std::ios::binary);
    const std::string b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    IndexReader in{b};
    const SessionFileIdentity id = file_identity();
    const uint64_t magic = in.get<uint64_t>();
    if ((magic != kIndexMagic && magic != kIndexMagic2) || in.get<uint64_t>() != id.model || in.get<uint64_t>() != id.config)
        return false;
    e.written_s = in.get<int64_t>();
    e.cvec = in.get<uint8_t>() != 0;
    e.bytes = in.get<uint64_t>();
    std::error_code ec;
    auto file_at = [&](const std::string& name) { return (fs::path(options_.dir) / name).string(); };
    if (magic == kIndexMagic) {   // version 1: one file per image, no chain data
        const uint32_t nfiles = in.get<uint32_t>();
        for (uint32_t i = 0; in.ok && i < nfiles; ++i) {
            Chain ch;
            ch.files.push_back(file_at(in.str()));
            ch.file_bytes.push_back((uint64_t) fs::file_size(fs::path(ch.files.back()), ec));
            e.chains.push_back(std::move(ch));
        }
    } else {
        const uint32_t nchains = in.get<uint32_t>();
        for (uint32_t c = 0; in.ok && c < nchains && c < 64; ++c) {
            Chain ch;
            const uint32_t nfiles = in.get<uint32_t>();
            for (uint32_t i = 0; in.ok && i < nfiles && i < 256; ++i) {
                ch.files.push_back(file_at(in.str()));
                ch.hash.push_back(in.get<uint64_t>());
                ch.payload.push_back(in.get<uint64_t>());
                ch.file_bytes.push_back(in.get<uint64_t>());
            }
            const uint32_t nck = in.get<uint32_t>();
            if (!in.ok || in.left() / 8 < nck) return false;
            for (uint32_t i = 0; i < nck; ++i) ch.ck_hash.push_back(in.get<uint64_t>());
            const uint32_t nlen = in.get<uint32_t>();
            if (!in.ok || in.left() / 40 < nlen) return false;
            ch.part_len.resize(nlen);
            for (auto& len : ch.part_len) for (auto& v : len) v = in.get<uint64_t>();
            const uint32_t nch = in.get<uint32_t>();
            if (!in.ok || in.left() / 20 < nch) return false;
            ch.chunks.resize(nch);
            for (auto& layer : ch.chunks)
                for (auto& part : layer.part) {
                    const uint32_t n = in.get<uint32_t>();
                    if (!in.ok || in.left() / 8 < n) return false;
                    for (uint32_t i = 0; i < n; ++i) part.push_back(in.get<uint64_t>());
                }
            if (ch.files.empty() || ch.files.size() != nfiles) return false;
            e.chains.push_back(std::move(ch));
        }
    }
    if (!in.key(e.live)) return false;
    const uint32_t nck = in.get<uint32_t>();
    for (uint32_t i = 0; in.ok && i < nck; ++i) {
        ConversationCheckpoint c;
        if (in.key(c)) e.checkpoints.push_back(std::move(c));
    }
    if (!in.ok || in.left() != 0 || e.chains.empty()) return false;
    for (const auto& ch : e.chains)
        for (const auto& file : ch.files) {
            if (!fs::is_regular_file(fs::path(file), ec)) return false;
            e.files.push_back(file);
        }
    e.index = path;
    return true;
}

std::string ConversationDiskTier::scan_folder(size_t& stale, std::error_code& ec) {
    std::vector<Entry> found;
    std::vector<std::string> others;
    size_t refused = 0;
    for (fs::directory_iterator it(fs::path(options_.dir), ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec) || !tier_file(it->path().filename().string())) continue;
        Entry e;
        const bool index = it->path().extension() == ".idx";
        const bool taken = index && read_index(it->path().string(), e);
        if (taken) found.push_back(std::move(e));
        else others.push_back(it->path().string());
        refused += index && !taken;
    }
    // the files of the entries taken in stay; the rest goes (refused indexes and their files, interrupted writes)
    std::set<std::string> kept;
    for (const auto& e : found) kept.insert(e.files.begin(), e.files.end());
    for (const auto& p : others)
        if (!kept.count(p)) { remove_file(p); ++stale; }
    std::sort(found.begin(), found.end(), [](const Entry& a, const Entry& b) { return a.written_s < b.written_s; });
    for (auto& e : found) {
        e.id = ++next_id_;
        for (const auto& ch : e.chains)
            for (size_t i = 0; i < ch.files.size(); ++i) ref_locked(ch.files[i], i < ch.file_bytes.size() ? ch.file_bytes[i] : 0);
        entries_.push_back(std::move(e));
    }
    const size_t aged = remove_aged_locked();
    size_t over = 0;
    while (!entries_.empty() && bytes_ > options_.budget_bytes) {
        remove_locked(0, true);
        ++over;
    }
    return fmt("persistent (budget only): %zu conversations from earlier runs (%.0f MiB), %zu refused (another model, "
               "engine version or settings), %zu older than the age limit removed, %zu over the budget removed",
               entries_.size(), mib(bytes_), refused, aged, over);
}

size_t ConversationDiskTier::remove_aged_locked() {
    if (!options_.persist || options_.max_age_s <= 0) return 0;
    const int64_t now_s = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    size_t n = 0;
    while (!entries_.empty() && entries_.front().written_s > 0 &&
           now_s - entries_.front().written_s >= options_.max_age_s) {   // in write order: the oldest first
        remove_locked(0, true);
        ++n;
    }
    return n;
}

void ConversationDiskTier::log(const std::string& line) const {
    if (options_.log) options_.log(line);
    else std::fprintf(stderr, "strata serve: conversation disk: %s\n", line.c_str());
}

bool ConversationDiskTier::open(std::string& error) {
    open_ = false;
    if (options_.dir.empty() || options_.budget_bytes == 0) return true;   // off: nothing to open
    std::error_code ec;
    const fs::path dir(options_.dir);
    fs::create_directories(dir, ec);
    if (ec || !fs::is_directory(dir, ec)) {
        error = "cannot create the folder " + options_.dir + (ec ? ": " + ec.message() : std::string());
        return false;
    }
#ifndef _WIN32
    fs::permissions(dir, fs::perms::owner_all, fs::perm_options::replace, ec);   // the files hold conversations
    ec.clear();
#endif
    size_t stale = 0;
    const std::string taken_in = scan_folder(stale, ec);
    if (ec) {
        error = "cannot list the folder " + options_.dir + ": " + ec.message();
        return false;
    }
    open_ = true;
    thread_ = std::thread([this] { writer(); });
    const std::string kept = options_.persist ? taken_in : "removed at the end";
    log(fmt("on: %s, budget %.0f MiB, %s; %zu stale files removed", options_.dir.c_str(), mib(options_.budget_bytes),
            kept.c_str(), stale));
    return true;
}

void ConversationDiskTier::remove_locked(size_t index, bool delete_files) {
    Entry& e = entries_.at(index);
    if (!e.index.empty()) remove_file(e.index);   // first, always: an entry gone from the list is not taken in again
    for (const auto& f : e.files) unref_locked(f, delete_files);   // a file another entry shares stays
    entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
}

void ConversationDiskTier::drop_retained_locked(size_t index) {
    for (const auto& f : retained_.at(index).files) unref_locked(f, true);
    retained_.erase(retained_.begin() + (std::ptrdiff_t) index);
}

void ConversationDiskTier::ref_locked(const std::string& file, uint64_t size) {
    auto& slot = files_[file];
    if (slot.first++ == 0) {
        slot.second = size;
        bytes_ += size;
    }
}

void ConversationDiskTier::unref_locked(const std::string& file, bool remove) {
    const auto it = files_.find(file);
    if (it == files_.end()) return;
    if (--it->second.first == 0) {
        bytes_ -= it->second.second;
        files_.erase(it);
        if (remove) remove_file(file);
    }
}

bool ConversationDiskTier::put(SavedConversation&& image) {
    if (!open_) return false;
    Job job;
    job.image = std::move(image);
    job.entry.live = key_of(job.image.live);
    for (const auto& c : job.image.checkpoints) job.entry.checkpoints.push_back(key_of(c));
    job.entry.cvec = job.image.cvec;
    job.entry.bytes = job.image.bytes();
    const size_t tokens = job.entry.live.ids.size();
    if (job.entry.bytes > options_.budget_bytes) {
        log(fmt("dropped %zu tokens: %.0f MiB is over the disk budget", tokens, mib(job.entry.bytes)));
        std::lock_guard<std::mutex> lock(m_);
        ++stats_.write_failed;
        return false;
    }
    size_t depth = 0;
    uint64_t queued_bytes = 0;
    const uint64_t estimate = job.entry.bytes;
    {
        std::lock_guard<std::mutex> lock(m_);
        job.entry.id = ++next_id_;
        ++stats_.queued;
        queue_.push_back(std::move(job));
        depth = queue_.size();
        for (const auto& j : queue_) queued_bytes += j.entry.bytes;
    }
    cv_.notify_all();
    log(fmt("queued %zu tokens (%.0f MiB in RAM until written; queue %zu, %.0f MiB)", tokens, mib(estimate), depth,
            mib(queued_bytes)));
    return true;
}

void ConversationDiskTier::writer() {
#if defined(__linux__)
    // below the serving threads: the write is background work, decode and prompt reading come first
    setpriority(PRIO_PROCESS, (id_t) syscall(SYS_gettid), 10);
#endif
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(m_);
            cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            job = std::move(queue_.front());
            queue_.pop_front();
            writing_ = true;
            writing_entry_ = job.entry;   // best() still sees it; take() waits for its file
        }
        write(job);
        {
            std::lock_guard<std::mutex> lock(m_);
            writing_ = false;
            writing_entry_ = Entry{};
        }
        cv_.notify_all();
    }
}

void ConversationDiskTier::write(Job& job) {
    if (options_.before_write) options_.before_write();
    std::vector<SavedConversation> parts = std::move(job.image.stage_images);   // one file per image, none nested
    job.image.stage_images.clear();
    // the K/V goes out through sources (the same file bytes as the image itself), the rest as each image's meta
    std::vector<std::vector<ConversationKv>> kv;
    kv.push_back(std::move(job.image.kv));
    job.image.kv.clear();
    for (auto& part : parts) {
        kv.push_back(std::move(part.kv));
        part.kv.clear();
    }
    std::vector<const SavedConversation*> metas{&job.image};
    for (const auto& part : parts) metas.push_back(&part);
    std::vector<std::vector<SessionKvSource>> sources;
    for (const auto& layers : kv) sources.push_back(kv_sources(layers));
    store(std::move(job.entry), metas, sources, job.image.disk_parent, "");
    job.image = SavedConversation{};   // the RAM goes back now, written or not
}

bool ConversationDiskTier::put_streamed(SavedConversation& meta,
                                        const std::vector<std::vector<SessionKvSource>>& sources) {
    if (!open_ || sources.size() != meta.stage_images.size() + 1) return false;
    Entry e;
    e.live = key_of(meta.live);
    for (const auto& c : meta.checkpoints) e.checkpoints.push_back(key_of(c));
    e.cvec = meta.cvec;
    e.bytes = meta.bytes();
    for (const auto& image : sources)
        for (const auto& layer : image)
            for (size_t part : layer.sizes) e.bytes += part;
    if (e.bytes > options_.budget_bytes) {
        log(fmt("dropped %zu tokens: %.0f MiB is over the disk budget", e.live.ids.size(), mib(e.bytes)));
        std::lock_guard<std::mutex> lock(m_);
        ++stats_.write_failed;
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(m_);
        e.id = ++next_id_;
    }
    std::vector<SavedConversation> parts = std::move(meta.stage_images);
    meta.stage_images.clear();
    std::vector<const SavedConversation*> metas{&meta};
    for (const auto& part : parts) metas.push_back(&part);
    const bool ok = store(std::move(e), metas, sources, meta.disk_parent, ", streamed from the GPU");
    meta.stage_images = std::move(parts);   // given back as it came: the caller lent its checkpoints
    return ok;
}

bool ConversationDiskTier::plan_delta(const Entry& base, const std::vector<const SavedConversation*>& metas,
                                      const std::vector<std::vector<SessionKvSource>>& sources, int64_t unchanged,
                                      const std::vector<std::vector<uint64_t>>& ck_hashes,
                                      std::vector<SessionDelta>& deltas, uint64_t& delta_bytes, size_t& mismatched,
                                      std::string& why) const {
    delta_bytes = 0;
    mismatched = 0;
    if (base.chains.size() != metas.size()) { why = "another number of stage images"; return false; }
    deltas.assign(metas.size(), SessionDelta{});
    uint64_t base_bytes = 0, chain_deltas = 0;
    std::vector<uint8_t> buf;
    for (size_t k = 0; k < metas.size(); ++k) {
        const Chain& ch = base.chains[k];
        if (ch.files.empty() || ch.hash.size() != ch.files.size() || ch.part_len.size() != sources[k].size()) {
            why = "its files carry no chain data (written before deltas, or another layer count)";
            return false;
        }
        if (ch.files.size() > options_.delta_chain) {
            why = fmt("its chain holds %zu deltas already: a new base", ch.files.size() - 1);
            return false;
        }
        if (options_.verify_prefix && ch.chunks.size() != sources[k].size()) { why = "its files carry no chunk hashes"; return false; }
        base_bytes += ch.file_bytes[0];
        for (size_t i = 1; i < ch.file_bytes.size(); ++i) chain_deltas += ch.file_bytes[i];
        SessionDelta& d = deltas[k];
        d.parent_hash = ch.hash.back();
        d.parent_payload = ch.payload.back();
        // a checkpoint the chain holds already (same tokens, same state) is named, not written again
        std::vector<bool> used(ch.ck_hash.size(), false);
        delta_bytes += metas[k]->live.bytes();
        for (size_t i = 0; i < metas[k]->checkpoints.size(); ++i) {
            int64_t ref = -1;
            for (size_t j = 0; j < ch.ck_hash.size() && ref < 0; ++j)
                if (!used[j] && ch.ck_hash[j] == ck_hashes[k][i]) { ref = (int64_t) j; used[j] = true; }
            d.checkpoint_refs.push_back(ref);
            if (ref < 0) delta_bytes += metas[k]->checkpoints[i].bytes();
        }
        for (size_t l = 0; l < sources[k].size(); ++l) {
            const SessionKvSource& src = sources[k][l];
            std::array<uint64_t, 5> pre{};
            for (size_t part = 0; part < 5; ++part) {
                uint64_t keep = std::min<uint64_t>({stable_bytes(src, part, unchanged, options_.idx_block),
                                                    ch.part_len[l][part], (uint64_t) src.sizes[part]});
                keep = keep / kSessionChunkBytes * kSessionChunkBytes;
                if (options_.verify_prefix && keep) {
                    // the chain's bytes below the prefix must be this conversation's: each MiB against its hash
                    const auto& want = ch.chunks[l].part[part];
                    buf.resize((size_t) kSessionChunkBytes);
                    for (uint64_t c = 0; c < keep / kSessionChunkBytes; ++c) {
                        const bool read = src.read && src.read(part, (size_t) (c * kSessionChunkBytes), buf.data(), buf.size());
                        if (!read || c >= want.size() || session_chunk_hash(buf.data(), buf.size()) != want[c]) {
                            if (read && c < want.size()) {
                                ++mismatched;
                                log(fmt("prefix check: layer %zu part %zu differs from its chain at MiB %llu though "
                                        "%lld tokens were to be unchanged; written again from there", l, part,
                                        (unsigned long long) c, (long long) unchanged));
                            }
                            keep = c * kSessionChunkBytes;
                            break;
                        }
                    }
                }
                pre[part] = keep;
                delta_bytes += src.sizes[part] - keep;
            }
            d.prefix.push_back(pre);
        }
    }
    if (chain_deltas + delta_bytes > base_bytes) {
        why = fmt("the deltas would outgrow the base (%.0f + %.0f MiB of deltas, %.0f MiB base)", mib(chain_deltas),
                  mib(delta_bytes), mib(base_bytes));
        return false;
    }
    return true;
}

SessionWriteOptions ConversationDiskTier::write_options() const {
    SessionWriteOptions wo;
    wo.min_free_bytes = options_.min_free_bytes;
    wo.durable = false;   // a cache: a file a power loss damages fails its hash check and is read again from the prompt
    return wo;
}

bool ConversationDiskTier::store(Entry e, const std::vector<const SavedConversation*>& metas,
                                 const std::vector<std::vector<SessionKvSource>>& sources,
                                 const ConversationDiskParent& parent, const char* how) {
    const size_t tokens = e.live.ids.size(), images = metas.size();
    const auto t0 = std::chrono::steady_clock::now();
    // the chain this conversation continues, when the tier still keeps it: copied, planned without the lock
    Entry base;
    bool from_base = false;
    if (options_.delta_chain > 0 && parent.chain != 0 && parent.unchanged > 0) {
        std::lock_guard<std::mutex> lock(m_);
        for (const auto& r : retained_)
            if (r.id == parent.chain) { base = r; from_base = true; }
    }
    std::vector<std::vector<uint64_t>> ck_hashes(images);
    if (options_.delta_chain > 0)
        for (size_t k = 0; k < images; ++k)
            for (const auto& c : metas[k]->checkpoints) ck_hashes[k].push_back(session_checkpoint_hash(c));
    std::vector<SessionDelta> deltas;
    uint64_t delta_bytes = 0;
    size_t mismatched = 0;
    std::string why;
    bool delta = from_base && plan_delta(base, metas, sources, parent.unchanged, ck_hashes, deltas, delta_bytes,
                                         mismatched, why);
    if (from_base && !delta) log(fmt("%zu tokens: a whole file, not a delta (%s)", tokens, why.c_str()));
    const uint64_t whole_bytes = e.bytes;
    size_t made_room = 0, depth = 0;
    {
        std::lock_guard<std::mutex> lock(m_);
        if (delta) {   // the parent's files are this entry's from now on: the budget cannot remove them under it
            size_t at = retained_.size();
            for (size_t i = 0; i < retained_.size(); ++i)
                if (retained_[i].id == parent.chain) at = i;
            if (at == retained_.size()) {
                delta = false;   // the budget took it meanwhile
            } else {
                e.chains = retained_[at].chains;
                for (const auto& ch : e.chains)
                    for (size_t i = 0; i < ch.files.size(); ++i) ref_locked(ch.files[i], ch.file_bytes[i]);
                drop_retained_locked(at);   // continued by this entry
                depth = e.chains.empty() ? 0 : e.chains[0].files.size();
            }
        } else if (from_base) {   // a whole file instead: this conversation's base is superseded
            for (size_t i = 0; i < retained_.size(); ++i)
                if (retained_[i].id == parent.chain) { drop_retained_locked(i); break; }
        }
        stats_.prefix_mismatch += mismatched;
        const size_t aged = remove_aged_locked();   // local: the age limit while running, at every write
        if (aged) log(fmt("removed %zu conversation%s written more than %lld days ago", aged, aged == 1 ? "" : "s",
                          (long long) (options_.max_age_s / 86400)));
        const uint64_t incoming = delta ? delta_bytes : e.bytes;
        while (bytes_ + incoming > options_.budget_bytes && (!retained_.empty() || !entries_.empty())) {
            if (!retained_.empty()) drop_retained_locked(0);   // bases nobody continued go first
            else remove_locked(0, true);
            ++made_room;
        }
        stats_.evicted += made_room;
    }
    if (!delta) e.chains.assign(images, Chain{});
    const SessionFileIdentity id = file_identity();
    std::vector<std::string> new_files;
    std::vector<uint64_t> new_sizes;
    std::string error;
    bool ok = true;
    uint64_t written_bytes = 0;
    for (size_t k = 0; ok && k < images; ++k) {
        char name[96];
        std::snprintf(name, sizeof name, "%s%016llx-%llu-%zu.bin", file_prefix(), (unsigned long long) nonce_,
                      (unsigned long long) e.id, k);
        const std::string path = (fs::path(options_.dir) / name).string();
        SessionWriteOptions wo = write_options();
        uint64_t ph = 0, pl = 0;
        std::vector<SessionPartChunks> chunks;
        wo.payload_hash = &ph;
        wo.payload_bytes = &pl;
        if (options_.delta_chain > 0 && options_.verify_prefix) wo.chunks = &chunks;
        size_t written = 0;
        ok = delta ? session_delta_write(path, *metas[k], sources[k], deltas[k], id, written, error, wo)
                   : session_file_write(path, *metas[k], sources[k], id, written, error, wo);
        if (!ok) {
            remove_file(path);
            break;
        }
        new_files.push_back(path);
        new_sizes.push_back(written);
        written_bytes += written;
        Chain& ch = e.chains[k];
        if (wo.chunks) {   // the image's parts as the chain now holds them: the parent's MiBs below the prefix, then this file's
            if (!delta) {
                ch.chunks = std::move(chunks);
            } else {
                ch.chunks.resize(sources[k].size());
                for (size_t l = 0; l < sources[k].size(); ++l)
                    for (size_t part = 0; part < 5; ++part) {
                        auto& v = ch.chunks[l].part[part];
                        v.resize((size_t) std::min<uint64_t>(v.size(), deltas[k].prefix[l][part] / kSessionChunkBytes));
                        if (l < chunks.size()) v.insert(v.end(), chunks[l].part[part].begin(), chunks[l].part[part].end());
                    }
            }
            ch.chunks.resize(sources[k].size());
        } else {
            ch.chunks.clear();
        }
        ch.files.push_back(path);
        ch.hash.push_back(ph);
        ch.payload.push_back(pl);
        ch.file_bytes.push_back(written);
        ch.ck_hash = options_.delta_chain > 0 ? ck_hashes[k] : std::vector<uint64_t>{};
        ch.part_len.clear();
        for (const auto& src : sources[k]) {
            std::array<uint64_t, 5> len{};
            for (size_t part = 0; part < 5; ++part) len[part] = src.sizes[part];
            ch.part_len.push_back(len);
        }
    }
    e.files.clear();
    e.bytes = 0;
    for (const auto& ch : e.chains)
        for (size_t i = 0; i < ch.files.size(); ++i) {
            e.files.push_back(ch.files[i]);
            e.bytes += ch.file_bytes[i];
        }
    if (ok && options_.persist) ok = write_index(e, error);   // after the images: an index names complete files only
    const double ms = ms_since(t0);
    if (!ok) {
        for (const auto& f : new_files) remove_file(f);
        {
            std::lock_guard<std::mutex> lock(m_);
            if (delta)   // the parent's files this entry held
                for (const auto& f : e.files)
                    if (std::find(new_files.begin(), new_files.end(), f) == new_files.end()) unref_locked(f, true);
            ++stats_.write_failed;
        }
        log(fmt("dropped %zu tokens: the write failed after %.0f ms (%s)", tokens, ms, error.c_str()));
        return false;
    }
    const size_t nfiles = new_files.size();
    size_t on_disk = 0;
    uint64_t disk_bytes = 0;
    {
        std::lock_guard<std::mutex> lock(m_);
        for (size_t i = 0; i < new_files.size(); ++i) ref_locked(new_files[i], new_sizes[i]);
        entries_.push_back(std::move(e));
        ++stats_.written;
        stats_.bytes_written += written_bytes;
        stats_.write_ms += ms;
        if (delta) {
            ++stats_.deltas;
            stats_.bytes_saved += whole_bytes > written_bytes ? whole_bytes - written_bytes : 0;
        }
        on_disk = entries_.size();
        disk_bytes = bytes_;
    }
    const std::string kind = delta ? fmt(" as delta %zu of its chain (a whole file: ~%.0f MiB)", depth, mib(whole_bytes))
                                   : std::string();
    log(fmt("wrote %zu tokens%s%s: %.0f MiB in %zu file%s, %.0f ms (%.0f MB/s); on disk %zu, %.0f MiB; %zu removed for "
            "the budget", tokens, how, kind.c_str(), mib(written_bytes), nfiles, nfiles == 1 ? "" : "s", ms,
            mbps(written_bytes, ms), on_disk, mib(disk_bytes), made_room));
    return true;
}

bool ConversationDiskTier::take(uint64_t id, int64_t tokens, SavedConversation& out, std::string& error,
                                const std::function<bool(uint64_t, std::string&)>& admit,
                                const std::vector<std::vector<SessionKvTarget>>* kv, bool live, bool* kept,
                                uint64_t* chain) {
    out = SavedConversation{};
    if (kept) *kept = false;
    if (chain) *chain = 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(m_);
    if (!open_) { error = "the disk tier is off"; return false; }
    // still queued: back from RAM, no disk at all
    for (size_t i = 0; i < queue_.size(); ++i)
        if (queue_[i].entry.id == id) {
            if (kv) { error = "still queued: its image is in RAM"; return false; }
            out = std::move(queue_[i].image);
            const uint64_t b = queue_[i].entry.bytes;
            queue_.erase(queue_.begin() + (std::ptrdiff_t) i);
            ++stats_.ram_hits;
            stats_.tokens_back += (uint64_t) std::max<int64_t>(tokens, 0);
            lock.unlock();
            log(fmt("RAM hit: %lld tokens back from the write queue (%.0f MiB, no disk) in %.1f ms",
                    (long long) tokens, mib(b), ms_since(t0)));
            log(summary());
            return true;
        }
    // being written: its file is complete when the write ends
    double waited = 0;
    if (writing_ && writing_entry_.id == id) {
        const auto w0 = std::chrono::steady_clock::now();
        cv_.wait(lock, [&] { return !(writing_ && writing_entry_.id == id); });
        waited = ms_since(w0);
    }
    size_t index = entries_.size();
    for (size_t i = 0; i < entries_.size(); ++i)
        if (entries_[i].id == id) index = i;
    if (index == entries_.size()) {
        ++stats_.read_failed;
        error = "the entry is gone (its write failed, or the budget removed it)";
        return false;
    }
    if (kv && kv->size() != entries_[index].chains.size()) {
        error = "one set of K/V targets per image is needed";
        return false;
    }
    // the entry leaves the list now (its index too); it keeps holding its files while they are read
    Entry want = std::move(entries_[index]);
    entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
    if (!want.index.empty()) remove_file(want.index);
    want.index.clear();
    lock.unlock();
    bool chained = false;
    for (const auto& ch : want.chains) chained = chained || ch.files.size() > 1;
    const SessionFileIdentity fid = file_identity();
    SessionReadLimits limits;
    limits.admit = admit;
    uint64_t bytes = 0;
    bool ok = true;
    bool no_ram = false;
    for (size_t k = 0; ok && k < want.chains.size(); ++k) {
        SavedConversation part;
        size_t n = 0;
        SessionStatus st;
        const std::string& file = want.chains[k].files[0];
        if (chained)
            ok = session_chain_read(want.chains[k].files, fid, part, kv ? &(*kv)[k] : nullptr, n, error, limits, &st);
        else
            ok = kv ? session_file_read(file, fid, part, (*kv)[k], n, error, limits, &st)
                    : session_file_read(file, fid, part, n, error, limits, &st);
        no_ram = !ok && k == 0 && st.error == SessionError::memory;   // refused before anything was read
        if (no_ram && !chained && live && !want.checkpoints.empty() && !limits.drop_checkpoints) {
            // no room for its checkpoints: the live state alone (the conversation continues; only a later request
            // that goes back before its end has no checkpoint to start from)
            SessionReadLimits lean = limits;
            lean.drop_checkpoints = true;
            lean.checkpoint_count = want.checkpoints.size();
            const std::string full_refusal = error;
            error.clear();
            ok = kv ? session_file_read(file, fid, part, (*kv)[k], n, error, lean, &st)
                    : session_file_read(file, fid, part, n, error, lean, &st);
            if (ok) {
                limits = lean;   // the later stages' files the same way
                log(fmt("%lld tokens: no RAM for the %zu checkpoints (%s); the live state only",
                        (long long) tokens, want.checkpoints.size(), full_refusal.c_str()));
            }
            no_ram = !ok && st.error == SessionError::memory;
        }
        bytes += n;
        if (!ok) break;
        if (k == 0) out = std::move(part);
        else out.stage_images.push_back(std::move(part));
    }
    static const std::vector<ConversationCheckpoint> no_checkpoints;
    if (ok && !holds(out, want.live, limits.drop_checkpoints ? no_checkpoints : want.checkpoints, want.cvec)) {
        ok = false;
        error = "the files hold another conversation than their index";
    }
    if (!ok && no_ram && kept) {
        // refused for RAM before anything was read: the entry and its files stay, for a streamed read
        Entry back = want;
        std::string why;
        if (!options_.persist || write_index(back, why)) {
            lock.lock();
            // back where it was (still holding its files): the entries are in write order (the budget removes the
            // oldest first)
            entries_.insert(entries_.begin() + (std::ptrdiff_t) std::min(index, entries_.size()), std::move(back));
            ++stats_.read_refused;
            lock.unlock();
            *kept = true;
            log(fmt("%lld tokens: no RAM to read them whole (%s); kept for a streamed read", (long long) tokens,
                    error.c_str()));
            return false;
        }
        log(fmt("%lld tokens: not kept for a streamed read (%s)", (long long) tokens, why.c_str()));
    }
    const double ms = ms_since(t0);
    const size_t nfiles = want.files.size();
    lock.lock();
    if (!ok || options_.delta_chain == 0) {   // spent, or no next delta to keep them for
        for (const auto& f : want.files) unref_locked(f, true);
    } else {   // a base for this conversation's next park
        if (chain) *chain = want.id;
        retained_.push_back(std::move(want));
        while (retained_.size() > kMaxRetained) drop_retained_locked(0);
    }
    if (!ok) {
        ++stats_.read_failed;
        lock.unlock();
        out = SavedConversation{};
        log(fmt("read of %lld tokens failed after %.0f ms (%s)", (long long) tokens, ms, error.c_str()));
        return false;
    }
    ++stats_.disk_hits;
    stats_.bytes_read += bytes;
    stats_.read_ms += ms - waited;
    stats_.tokens_back += (uint64_t) std::max<int64_t>(tokens, 0);
    lock.unlock();
    log(fmt("disk hit: %lld tokens, %.0f MiB in %zu file%s, %.0f ms (%.0f MB/s)%s%s", (long long) tokens, mib(bytes),
            nfiles, nfiles == 1 ? "" : "s", ms - waited, mbps(bytes, ms - waited),
            kv ? ", streamed into the GPU" : "",
            waited > 0 ? fmt(", after %.0f ms waiting for its write", waited).c_str() : ""));
    log(summary());
    return true;
}

bool ConversationDiskTier::pending() const {
    std::lock_guard<std::mutex> lock(m_);
    return !queue_.empty() || writing_;
}

void ConversationDiskTier::wait_idle() {
    std::unique_lock<std::mutex> lock(m_);
    if (queue_.empty() && !writing_) return;
    const auto t0 = std::chrono::steady_clock::now();
    const size_t depth = queue_.size() + (writing_ ? 1 : 0);
    cv_.wait(lock, [&] { return queue_.empty() && !writing_; });
    const double ms = ms_since(t0);
    ++stats_.waits;
    stats_.wait_ms += ms;
    lock.unlock();
    log(fmt("waited %.0f ms for %zu queued write%s (their RAM was needed)", ms, depth, depth == 1 ? "" : "s"));
}

size_t ConversationDiskTier::size() const {
    std::lock_guard<std::mutex> lock(m_);
    return entries_.size();
}

size_t ConversationDiskTier::retained() const {
    std::lock_guard<std::mutex> lock(m_);
    return retained_.size();
}

uint64_t ConversationDiskTier::bytes() const {
    std::lock_guard<std::mutex> lock(m_);
    return bytes_;
}

ConversationDiskStats ConversationDiskTier::stats() const {
    std::lock_guard<std::mutex> lock(m_);
    return stats_;
}

size_t ConversationDiskTier::clear() {
    std::unique_lock<std::mutex> lock(m_);
    const size_t queued = queue_.size();
    queue_.clear();
    cv_.wait(lock, [&] { return !writing_; });
    const size_t n = queued + entries_.size();
    while (!entries_.empty()) remove_locked(0, true);
    while (!retained_.empty()) drop_retained_locked(0);
    lock.unlock();
    if (n) log(fmt("removed all %zu conversations (#879)", n));
    return n;
}

std::string ConversationDiskTier::summary() const {
    std::lock_guard<std::mutex> lock(m_);
    const auto& s = stats_;
    return fmt("totals: %zu queued, %zu written (%.0f MiB, %.0f MB/s), %zu RAM hits, %zu disk hits (%.0f MiB, "
               "%.0f MB/s), %llu prompt tokens given back, %zu waits for RAM (%.0f ms), "
               "%zu removed for the budget, %zu/%zu writes/reads failed; on disk %zu (%.0f MiB), queue %zu%s",
               s.queued, s.written, mib(s.bytes_written), mbps(s.bytes_written, s.write_ms), s.ram_hits, s.disk_hits,
               mib(s.bytes_read), mbps(s.bytes_read, s.read_ms), (unsigned long long) s.tokens_back, s.waits,
               s.wait_ms, s.evicted, s.write_failed, s.read_failed, entries_.size(), mib(bytes_),
               queue_.size() + (writing_ ? 1 : 0),
               options_.delta_chain == 0 ? ""
                   : fmt("; %zu deltas (%.0f MiB not rewritten, %zu prefix mismatches), %zu bases kept", s.deltas,
                         mib(s.bytes_saved), s.prefix_mismatch, retained_.size()).c_str());
}

}  // namespace strata::core
