#include "strata/core/hibernate_manifest.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
void touch(const fs::path& p) { std::ofstream(p) << "x"; }
void write_text(const fs::path& p, const char* text) { std::ofstream(p, std::ios::binary) << text; }
}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "strata_hibernate_manifest_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string d = dir.string();
    std::string error;

    // names
    check(hibernate_entry_name(1760198400123ull, 2) == "hib-1760198400123-2.strs", "entry name");
    check(hibernate_entry_name_ok("hib-1-0.strs") && hibernate_entry_name_ok(hibernate_entry_name(99, 12)),
          "names it makes are accepted");
    for (const char* bad : {"../hib-1-0.strs", "hib-1-0.strs.tmp", "hib-a-0.strs", "hib-1-.strs", "hib--0.strs",
                            "hib-1-0", "x/hib-1-0.strs", "hib-1-0.strsx", "hibernate.txt", ""})
        check(!hibernate_entry_name_ok(bad), "anything else is refused (no paths, no other names)");

    // round trip
    HibernateManifest m;
    m.generation = 1760198400123ull;
    m.engine = "0.1.43";
    m.bytes = 4404019200ull;
    m.files = {hibernate_entry_name(m.generation, 0), hibernate_entry_name(m.generation, 1)};
    check(hibernate_manifest_write(d, m, error), "write a manifest");
    check(!fs::exists(dir / "hibernate.txt.tmp"), "no temporary file is left");
    HibernateManifest back;
    check(hibernate_manifest_read(d, back, error), "read it back");
    check(back.generation == m.generation && back.engine == m.engine && back.bytes == m.bytes && back.files == m.files,
          "every field round-trips, files in order");

    // a newer generation replaces it, and the older generation's files are removed - nothing else is
    for (const auto& f : m.files) touch(dir / f);
    touch(dir / "notes.txt");
    HibernateManifest next = m;
    next.generation = m.generation + 1;
    next.files = {hibernate_entry_name(next.generation, 0)};
    touch(dir / next.files[0]);
    check(hibernate_manifest_write(d, next, error), "replace the manifest");
    check(hibernate_remove_others(d, next) == 2, "the earlier generation's two files are removed");
    check(fs::exists(dir / next.files[0]) && fs::exists(dir / "notes.txt") && fs::exists(dir / "hibernate.txt"),
          "the kept file, other files and the manifest stay");
    check(hibernate_manifest_read(d, back, error) && back.generation == next.generation, "the newer one is read");

    // refusals
    write_text(dir / "hibernate.txt", "strata-hibernate 2\ngeneration 1\nend\n");
    check(!hibernate_manifest_read(d, back, error), "another version is refused");
    write_text(dir / "hibernate.txt", "strata-hibernate 1\ngeneration 1\nfile ..\\..\\x.strs\nend\n");
    check(!hibernate_manifest_read(d, back, error), "a file name with a path is refused");
    write_text(dir / "hibernate.txt", "strata-hibernate 1\ngeneration 1\nfile hib-1-0.strs\n");
    check(!hibernate_manifest_read(d, back, error), "a manifest without its end line is refused (cut off)");
    write_text(dir / "hibernate.txt", "strata-hibernate 1\ngeneration 1\nfile hib-1-0.strs\nfile hib-1-0.strs\nend\n");
    check(!hibernate_manifest_read(d, back, error), "a file named twice is refused");
    write_text(dir / "hibernate.txt", "strata-hibernate 1\ngeneration x\nend\n");
    check(!hibernate_manifest_read(d, back, error), "a bad number is refused");
    fs::remove(dir / "hibernate.txt");
    check(!hibernate_manifest_read(d, back, error) && !error.empty(), "a missing manifest is an error with a reason");
    HibernateManifest bad = m;
    bad.files = {"../escape.strs"};
    check(!hibernate_manifest_write(d, bad, error), "writing a bad name is refused");

    fs::remove_all(dir);
    std::printf("hibernate_manifest_test: %d checks passed\n", checks);
    return 0;
}
