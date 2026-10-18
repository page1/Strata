#include "strata/core/hibernate_manifest.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <system_error>

#if defined(_WIN32)
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace strata::core {
namespace {

namespace fs = std::filesystem;

fs::path utf8_path(const std::string& s) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}

bool digits(const std::string& s, size_t from, size_t to) {
    if (from >= to || to - from > 20) return false;
    for (size_t i = from; i < to; ++i) if (s[i] < '0' || s[i] > '9') return false;
    return true;
}

} // namespace

std::string hibernate_entry_name(uint64_t generation, size_t k) {
    return "hib-" + std::to_string(generation) + "-" + std::to_string(k) + ".strs";
}

bool hibernate_entry_name_ok(const std::string& name) {
    if (name.size() < 11 || name.rfind("hib-", 0) != 0 || name.size() < 5 ||
        name.compare(name.size() - 5, 5, ".strs") != 0)
        return false;
    const size_t dash = name.find('-', 4);
    return dash != std::string::npos && digits(name, 4, dash) && digits(name, dash + 1, name.size() - 5);
}

bool hibernate_manifest_write(const std::string& dir, const HibernateManifest& m, std::string& error) {
    for (const auto& f : m.files)
        if (!hibernate_entry_name_ok(f)) {
            error = "hibernate manifest: not an entry name: " + f;
            return false;
        }
    std::ostringstream text;
    text << "strata-hibernate 1\n"
         << "generation " << m.generation << "\n"
         << "engine " << m.engine << "\n"
         << "bytes " << m.bytes << "\n";
    for (const auto& f : m.files) text << "file " << f << "\n";
    text << "end\n";
    const std::string body = text.str();
    const fs::path to = utf8_path(dir) / kHibernateManifest;
    const fs::path tmp = utf8_path(dir) / (std::string(kHibernateManifest) + ".tmp");
    std::error_code ec;
#if defined(_WIN32)
    FILE* f = _wfopen(tmp.c_str(), L"wb");
#else
    FILE* f = std::fopen(tmp.c_str(), "wb");
#endif
    if (!f) {
        error = "hibernate manifest: cannot create " + tmp.string();
        return false;
    }
    bool good = std::fwrite(body.data(), 1, body.size(), f) == body.size() && std::fflush(f) == 0;
#if defined(_WIN32)
    good = good && _commit(_fileno(f)) == 0;
#else
    good = good && fsync(fileno(f)) == 0;
#endif
    if (std::fclose(f) != 0) good = false;
    if (!good) {
        fs::remove(tmp, ec);
        error = "hibernate manifest: writing " + tmp.string() + " failed";
        return false;
    }
    fs::rename(tmp, to, ec);   // replaces an older manifest (MoveFileExW REPLACE_EXISTING on Windows)
    if (ec) {
        fs::remove(tmp, ec);
        error = "hibernate manifest: " + ec.message();
        return false;
    }
#if !defined(_WIN32)
    const int fd = ::open(utf8_path(dir).c_str(), O_RDONLY);   // the new name reaches the disk too
    if (fd >= 0) { ::fsync(fd); ::close(fd); }
#endif
    return true;
}

bool hibernate_manifest_read(const std::string& dir, HibernateManifest& m, std::string& error) {
    m = {};
    std::ifstream in(utf8_path(dir) / kHibernateManifest, std::ios::binary);
    if (!in) {
        error = "hibernate manifest: cannot open " + (utf8_path(dir) / kHibernateManifest).string();
        return false;
    }
    std::string line;
    if (!std::getline(in, line) || line != "strata-hibernate 1") {
        error = "hibernate manifest: not a version 1 manifest";
        return false;
    }
    bool ended = false, have_gen = false;
    std::set<std::string> seen;
    while (std::getline(in, line)) {
        if (line == "end") { ended = true; break; }
        const size_t sp = line.find(' ');
        const std::string key = line.substr(0, sp), value = sp == std::string::npos ? "" : line.substr(sp + 1);
        try {
            if (key == "generation") { m.generation = std::stoull(value); have_gen = true; }
            else if (key == "engine") m.engine = value;
            else if (key == "bytes") m.bytes = std::stoull(value);
            else if (key == "file") {
                if (!hibernate_entry_name_ok(value) || !seen.insert(value).second) {
                    error = "hibernate manifest: bad entry name: " + value;
                    return false;
                }
                m.files.push_back(value);
            } else {
                error = "hibernate manifest: unknown line: " + line.substr(0, 80);
                return false;
            }
        } catch (const std::exception&) {
            error = "hibernate manifest: bad number in: " + line.substr(0, 80);
            return false;
        }
    }
    if (!ended || !have_gen) {
        error = "hibernate manifest: incomplete (no generation or no end line)";
        return false;
    }
    return true;
}

size_t hibernate_remove_others(const std::string& dir, const HibernateManifest& keep) {
    const std::set<std::string> names(keep.files.begin(), keep.files.end());
    size_t removed = 0;
    std::error_code ec;
    for (fs::directory_iterator it(utf8_path(dir), ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = reinterpret_cast<const char*>(it->path().filename().u8string().c_str());
        if (!hibernate_entry_name_ok(name) || names.count(name)) continue;
        std::error_code rm;
        if (fs::remove(it->path(), rm)) ++removed;
    }
    return removed;
}

} // namespace strata::core
