// Hibernate/resume (--serve): the manifest of a hibernate folder.  HIBERNATE <dir> writes every parked conversation
// as its own session file (conversation_file.hpp) and then this manifest, which names them oldest first; its presence
// is the server's latch.  A restart with --hibernate-dir reads it and re-parks the files before READY.
//
// The manifest is plain text so the server can read it too:
//     strata-hibernate 1
//     generation 1760198400123
//     engine 0.1.43
//     bytes 4404019200
//     file hib-1760198400123-0.strs
//     file hib-1760198400123-1.strs
//     end
// File names are checked against the pattern hib-<digits>-<digits>.strs, so a manifest can never name a path
// outside its folder.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

constexpr const char* kHibernateManifest = "hibernate.txt";

struct HibernateManifest {
    uint64_t generation = 0;
    std::string engine;
    uint64_t bytes = 0;
    std::vector<std::string> files;   // oldest parked conversation first
};

// "hib-<generation>-<k>.strs"
std::string hibernate_entry_name(uint64_t generation, size_t k);
// true for a name hibernate_entry_name can produce (no folder part, nothing else)
bool hibernate_entry_name_ok(const std::string& name);

// Write <dir>/hibernate.txt: a temporary file, flushed to the disk, then moved over the old one.  `dir` is UTF-8.
bool hibernate_manifest_write(const std::string& dir, const HibernateManifest& m, std::string& error);
// Read and check <dir>/hibernate.txt.  false (and `error`) for a missing, foreign or malformed manifest.
bool hibernate_manifest_read(const std::string& dir, HibernateManifest& m, std::string& error);
// Delete the hib-*.strs files in `dir` that `keep` does not name (earlier generations).  -> how many were deleted.
size_t hibernate_remove_others(const std::string& dir, const HibernateManifest& keep);

} // namespace strata::core
