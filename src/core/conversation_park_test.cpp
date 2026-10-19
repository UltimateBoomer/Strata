// conversation_park.hpp: park files written new and updated in place, read back whole and streamed, refused when torn.
#include "strata/core/conversation_park.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <sys/stat.h>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
std::mt19937_64 rng(7);
void fill(ConversationBuffer& b, size_t n) {
    b.resize(n);
    b.visit(0, n, [](uint8_t* p, size_t c, size_t) { for (size_t i = 0; i < c; ++i) p[i] = (uint8_t) rng(); return true; });
}
std::vector<uint8_t> bytes_of(const ConversationBuffer& b) {
    std::vector<uint8_t> v(b.size());
    b.visit(0, b.size(), [&](const uint8_t* p, size_t c, size_t at) { std::memcpy(v.data() + at, p, c); return true; });
    return v;
}
ConversationCheckpoint state(std::vector<int32_t> ids, size_t gdn, uint8_t salt) {
    ConversationCheckpoint c;
    c.ids = std::move(ids);
    c.imgs = {{1, 77}};
    c.gdn.assign(gdn, salt); c.ple.assign(64, salt + 1); c.tails.assign(32, salt + 2);
    c.dead.assign(16, salt + 3); c.block_pos.assign(8, salt + 4);
    return c;
}
// a park file read back into a whole host image (its K/V in `image.kv`)
bool park_file_read_image(const std::string& path, uint64_t file, SavedConversation& image,
                          const SessionReadLimits& limits, ParkFileState& state, std::string& error) {
    SavedConversation out;
    SessionKvTarget t;
    t.begin = [&out](const SavedConversation&, uint64_t layers, std::string&) { out.kv.resize((size_t) layers); return true; };
    t.layer = [&out](size_t j, const ConversationKv& h, std::array<uint64_t, 5>& sizes, SessionKvSink& sink, std::string&) {
        ConversationKv& k = out.kv[j];
        k.format = h.format; k.cells = h.cells; k.heads = h.heads; k.head_dim = h.head_dim;
        k.page_size = h.page_size; k.pooled_rows = h.pooled_rows; k.idx_dim = h.idx_dim;
        const std::array<ConversationBuffer*, 5> parts = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
        for (size_t p = 0; p < 5; ++p) parts[p]->resize((size_t) sizes[p]);
        sink.write = [parts](size_t p, size_t at, const void* src, size_t n) {
            const auto* c = static_cast<const uint8_t*>(src);
            return parts[p]->visit(at, n, [&](uint8_t* d, size_t m, size_t off) { std::memcpy(d, c + (off - at), m); return true; });
        };
        return true;
    };
    SavedConversation meta;
    if (!park_file_read(path, file, meta, t, limits, state, error)) return false;
    meta.kv = std::move(out.kv);
    image = std::move(meta);
    return true;
}
// a conversation: running state, checkpoints and K/V layers of uneven sizes (some spanning several hash blocks)
SavedConversation image(const std::vector<std::array<size_t, 5>>& sizes) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = (int64_t) i + 3;
    s.layer_lo = 0; s.layer_hi = 4; s.cvec = true;
    s.live = state({1, 2, 3, 4, 5, 6}, 3000, 10);
    s.checkpoints = {state({1, 2}, 3000, 20), state({1, 2, 3, 4}, 3000, 30)};
    s.checkpoints[0].used = 5; s.checkpoints[1].used = 9;
    s.kv.resize(sizes.size());
    for (size_t j = 0; j < sizes.size(); ++j) {
        auto& k = s.kv[j];
        k.format = 3; k.cells = 64; k.heads = 2; k.head_dim = 128; k.page_size = 16; k.pooled_rows = 4; k.idx_dim = 64;
        fill(k.k, sizes[j][0]); fill(k.v, sizes[j][1]); fill(k.k_scale, sizes[j][2]);
        fill(k.v_scale, sizes[j][3]); fill(k.pooled, sizes[j][4]);
    }
    return s;
}
std::array<const ConversationBuffer*, 5> parts(const ConversationKv& k) { return {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled}; }
std::vector<SessionKvSource> sources(const SavedConversation& s, int* reads = nullptr) {
    std::vector<SessionKvSource> out(s.kv.size());
    for (size_t j = 0; j < s.kv.size(); ++j) {
        const ConversationKv& k = s.kv[j];
        SessionKvSource& o = out[j];
        o.format = k.format; o.cells = k.cells; o.heads = k.heads; o.head_dim = k.head_dim;
        o.page_size = k.page_size; o.pooled_rows = k.pooled_rows; o.idx_dim = k.idx_dim;
        const auto ps = parts(k);
        for (size_t p = 0; p < 5; ++p) o.sizes[p] = ps[p]->size();
        o.read = [ps, reads](size_t p, size_t at, void* dst, size_t n) {
            if (reads) ++*reads;
            auto* d = static_cast<uint8_t*>(dst);
            return ps[p]->visit(at, n, [&](const uint8_t* q, size_t c, size_t off) { std::memcpy(d + (off - at), q, c); return true; });
        };
    }
    return out;
}
SavedConversation meta_of(const SavedConversation& s) {
    SavedConversation m = s;
    m.kv.clear();
    m.checkpoints.clear();
    return m;
}
std::vector<std::array<uint64_t, 5>> caps(size_t layers, uint64_t cap) { return std::vector<std::array<uint64_t, 5>>(layers, {cap, cap, cap, cap, cap}); }
bool same_state(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails &&
           a.dead == b.dead && a.block_pos == b.block_pos;
}
bool same(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
        !same_state(a.live, b.live) || a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size())
        return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!same_state(a.checkpoints[i], b.checkpoints[i]) || a.checkpoints[i].used != b.checkpoints[i].used) return false;
    for (size_t j = 0; j < a.kv.size(); ++j) {
        const auto pa = parts(a.kv[j]), pb = parts(b.kv[j]);
        for (size_t p = 0; p < 5; ++p) if (bytes_of(*pa[p]) != bytes_of(*pb[p])) return false;
        if (a.kv[j].cells != b.kv[j].cells || a.kv[j].format != b.kv[j].format) return false;
    }
    return true;
}
uint64_t allocated(const std::string& p) { struct stat s{}; ::stat(p.c_str(), &s); return (uint64_t) s.st_blocks * 512; }
void poke(const std::string& p, uint64_t at) {
    std::fstream f(p, std::ios::in | std::ios::out | std::ios::binary);
    f.seekg((std::streamoff) at); char c = 0; f.read(&c, 1); c ^= 0x5a;
    f.seekp((std::streamoff) at); f.write(&c, 1);
}
} // namespace

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    const fs::path dir = (argc > 1 ? fs::path(argv[1]) : fs::temp_directory_path()) /
                         ("strata-park-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const uint64_t MiB = 1u << 20;
    const uint64_t cap = 8 * MiB;
    const std::vector<std::array<size_t, 5>> sz = {{size_t(3 * MiB + 1234), 2 * MiB, 4096, 777, 0},
                                                   {MiB, size_t(MiB + 1), 12345, 4096, 3000},
                                                   {size_t(2 * MiB - 1), 5000, 0, 1, 64}};
    std::string err;
    auto beats = 0;
    const std::function<void()> beat = [&] { ++beats; };
    {
        // a new file, read back whole and streamed
        const std::string p = (dir / "a.park").string();
        const SavedConversation a = image(sz);
        ParkFileState st;
        ParkWriteStats ws;
        check(park_file_write(p, 41, nullptr, meta_of(a), a.checkpoints, sources(a), caps(3, cap), {}, 0, beat, st, ws, err),
              "write a new park file");
        uint64_t kv = 0;
        for (const auto& l : sz) for (size_t n : l) kv += n;
        check(ws.kept_kv == 0 && ws.kept_checkpoints == 0 && ws.written >= kv && beats > 0, "a new file writes all of it");
        check(st.bytes() == 4096 + st.directory.length + st.live.length + st.checkpoints[0].length +
                                st.checkpoints[1].length + kv, "the budget charges what the file holds");
        // (XFS preallocates past a growing write for a while, so only the reserved regions' holes are checked)
        std::printf("  park file: %llu MiB apparent, %llu MiB allocated, %llu MiB held\n",
                    (unsigned long long) (fs::file_size(p) >> 20), (unsigned long long) (allocated(p) >> 20),
                    (unsigned long long) (st.bytes() >> 20));
        check(fs::file_size(p) > 3 * 5 * cap && allocated(p) < fs::file_size(p) / 2, "the regions are sparse");
        ParkWriteStats w2;
        ParkFileState s2;
        check(!park_file_write(p, 41, nullptr, meta_of(a), a.checkpoints, sources(a), caps(3, cap), {}, 0, beat, s2, w2, err),
              "a new file never replaces one");
        SavedConversation back;
        ParkFileState rs;
        check(park_file_read_image(p, 41, back, {}, rs, err) && same(a, back), "read back whole: the same conversation");
        check(back.checkpoints[1].park_file == 41 && back.checkpoints[1].park_record == st.checkpoints[1].offset &&
              back.live.park_file == 0, "read-back checkpoints name their file and record");
        check(rs.bytes() == st.bytes() && rs.end == st.end && rs.generation == 1, "the state read back is the state written");

        // streamed into a target that takes only K/V
        std::vector<std::vector<uint8_t>> got(15);
        SessionKvTarget t;
        bool begun = false;
        t.begin = [&](const SavedConversation& m, uint64_t layers, std::string&) { begun = layers == 3 && m.checkpoints.size() == 2; return true; };
        t.layer = [&](size_t j, const ConversationKv&, std::array<uint64_t, 5>& sizes, SessionKvSink& sink, std::string&) {
            for (size_t q = 0; q < 5; ++q) got[j * 5 + q].resize(sizes[q]);
            sink.write = [&got, j](size_t q, size_t at, const void* src, size_t n) {
                std::memcpy(got[j * 5 + q].data() + at, src, n); return true; };
            return true;
        };
        SavedConversation m;
        SessionStatus ss;
        check(park_file_read(p, 41, m, t, {}, rs, err, &ss) && begun && ss.target_written && m.kv.empty(), "streamed read");
        bool kv_same = true;
        for (size_t j = 0; j < 3; ++j) { const auto pa = parts(a.kv[j]); for (size_t q = 0; q < 5; ++q) kv_same = kv_same && got[j * 5 + q] == bytes_of(*pa[q]); }
        check(kv_same, "streamed K/V equals the conversation's");

        // an in-place update: K/V grown past an unchanged prefix, one checkpoint kept, one new, the old one dropped
        SavedConversation b = a;
        const std::vector<std::array<size_t, 5>> grown = {{size_t(3 * MiB + 1234 + 5000), size_t(2 * MiB + 3), 4096 + 100, 900, 50},
                                                          {MiB, size_t(MiB + 1), 12345, 4096, 3000},
                                                          {size_t(2 * MiB + 10), 5000, 10, 1, 64}};
        std::vector<std::array<uint64_t, 5>> keep(3);
        for (size_t j = 0; j < 3; ++j) {
            const auto pa = parts(a.kv[j]);
            const std::array<ConversationBuffer*, 5> pb = {&b.kv[j].k, &b.kv[j].v, &b.kv[j].k_scale, &b.kv[j].v_scale, &b.kv[j].pooled};
            for (size_t q = 0; q < 5; ++q) {
                // keep a prefix that is not block- or page-aligned; everything past it changes
                keep[j][q] = std::min<uint64_t>(pa[q]->size(), grown[j][q]) * 2 / 3;
                std::vector<uint8_t> v = bytes_of(*pa[q]);
                v.resize(grown[j][q]);
                for (size_t i = keep[j][q]; i < v.size(); ++i) v[i] = (uint8_t) rng();
                pb[q]->resize(v.size());
                pb[q]->visit(0, v.size(), [&](uint8_t* d, size_t c, size_t at) { std::memcpy(d, v.data() + at, c); return true; });
            }
        }
        b.live = state({1, 2, 3, 4, 5, 6, 7, 8}, 3000, 40);
        b.checkpoints = {back.checkpoints[1], state({1, 2, 3, 4, 5, 6}, 3000, 50)};   // [1] read back: kept
        b.checkpoints[0].used = 11; b.checkpoints[1].used = 12;
        const uint64_t before = allocated(p);
        ParkFileState su;
        ParkWriteStats wu;
        check(park_file_write(p, 41, &rs, meta_of(b), b.checkpoints, sources(b), caps(3, cap), keep, 0, beat, su, wu, err),
              "update the file in place");
        uint64_t keep_total = 0, new_total = 0;
        for (size_t j = 0; j < 3; ++j) for (size_t q = 0; q < 5; ++q) { keep_total += keep[j][q]; new_total += grown[j][q]; }
        check(wu.kept_kv == keep_total && wu.kept_checkpoints == 1,
              "the unchanged prefix and the kept checkpoint are not written");
        check(wu.written == new_total - keep_total + su.checkpoints[1].length + su.live.length + su.directory.length + 4096,
              "an update writes the K/V past the prefix, the new records, the directory and the superblock");
        check(su.checkpoints[0].offset == rs.checkpoints[1].offset && su.generation == 2, "the kept record stays where it is");
        SavedConversation back2;
        ParkFileState rs2;
        check(park_file_read_image(p, 41, back2, {}, rs2, err) && same(b, back2), "the updated file reads back as the new conversation");
        check(allocated(p) < before + wu.written, "the records the update dropped are punched out");

        // a rewind: a smaller prefix kept, a part that shrinks
        SavedConversation c = b;
        std::vector<std::array<uint64_t, 5>> keep2(3, {0, 0, 0, 0, 0});
        keep2[0] = {MiB / 2, 0, 0, 0, 0};
        {
            std::vector<uint8_t> v = bytes_of(c.kv[0].k);
            v.resize(MiB + 7);
            for (size_t i = MiB / 2; i < v.size(); ++i) v[i] = (uint8_t) rng();
            c.kv[0].k.resize(v.size());
            c.kv[0].k.visit(0, v.size(), [&](uint8_t* d, size_t n, size_t at) { std::memcpy(d, v.data() + at, n); return true; });
        }
        c.checkpoints = {back2.checkpoints[0]};
        ParkFileState sc;
        ParkWriteStats wc;
        check(park_file_write(p, 41, &rs2, meta_of(c), c.checkpoints, sources(c), caps(3, cap), keep2, 0, beat, sc, wc, err) &&
              wc.kept_checkpoints == 1 && wc.kept_kv == MiB / 2, "a rewound conversation keeps only its smaller prefix");
        SavedConversation back3;
        ParkFileState rs3;
        check(park_file_read_image(p, 41, back3, {}, rs3, err) && same(c, back3), "and reads back as itself");

        // a checkpoint that names another file, or a record this file no longer holds, is written again
        SavedConversation d = c;
        d.checkpoints = {back3.checkpoints[0], back2.checkpoints[1]};
        d.checkpoints[0].park_file = 99;
        ParkFileState sd;
        ParkWriteStats wd;
        check(park_file_write(p, 41, &rs3, meta_of(d), d.checkpoints, sources(d), caps(3, cap), {}, 0, beat, sd, wd, err) &&
              wd.kept_checkpoints == 0 && wd.kept_kv == 0, "records of another file or a dropped one are rewritten");
        SavedConversation back4;
        ParkFileState rs4;
        check(park_file_read_image(p, 41, back4, {}, rs4, err) && same(d, back4), "and the file reads back as itself");

        // the base must be this file's, with this layer count
        ParkFileState wrong = rs4;
        wrong.file = 40;
        check(!park_file_write(p, 41, &wrong, meta_of(d), d.checkpoints, sources(d), caps(3, cap), {}, 0, beat, sd, wd, err),
              "an update needs this file's own state");
        check(!park_file_write((dir / "x.park").string(), 42, nullptr, meta_of(d), d.checkpoints, sources(d), caps(3, MiB), {}, 0, beat, sd, wd, err),
              "a K/V part larger than its region is refused");
        check(!park_file_write((dir / "y.park").string(), 43, nullptr, meta_of(d), d.checkpoints, sources(d), caps(3, cap), {}, UINT64_MAX / 4, beat, sd, wd, err),
              "the free-space preflight refuses");
    }
    {
        // torn files: the superblock, a record, a K/V block; and the limits
        const SavedConversation a = image(sz);
        const std::string p = (dir / "b.park").string();
        ParkFileState st;
        ParkWriteStats ws;
        check(park_file_write(p, 7, nullptr, meta_of(a), a.checkpoints, sources(a), caps(3, cap), {}, 0, beat, st, ws, err), "write b");
        const std::string q = (dir / "c.park").string();
        auto copy_poke = [&](uint64_t at) { fs::copy_file(p, q, fs::copy_options::overwrite_existing); poke(q, at); };
        SavedConversation out;
        ParkFileState rs;
        SessionStatus ss;
        SessionKvTarget take_all;
        take_all.layer = [](size_t, const ConversationKv&, std::array<uint64_t, 5>&, SessionKvSink& sink, std::string&) {
            sink.write = [](size_t, size_t, const void*, size_t) { return true; }; return true; };
        copy_poke(20);
        check(!park_file_read(q, 7, out, take_all, {}, rs, err, &ss) && !ss.target_written && err.find("superblock") != std::string::npos,
              "a torn superblock is refused");
        copy_poke(st.checkpoints[0].offset + 100);
        check(!park_file_read(q, 7, out, take_all, {}, rs, err, &ss) && !ss.target_written && err.find("record hash") != std::string::npos,
              "a torn record is refused before any K/V byte");
        copy_poke(st.layers[0].parts[0].offset + 2 * MiB + 5);
        check(!park_file_read(q, 7, out, take_all, {}, rs, err, &ss) && err.find("K/V block") != std::string::npos,
              "a torn K/V block is refused");
        check(ss.target_written == false, "the first part's blocks are checked before they are handed on (one chunk)");
        copy_poke(st.layers[2].parts[0].offset + 5);
        check(!park_file_read(q, 7, out, take_all, {}, rs, err, &ss) && ss.target_written, "a later torn block: K/V was handed on");
        copy_poke(st.directory.offset + 9);
        check(!park_file_read(q, 7, out, take_all, {}, rs, err, &ss) && err.find("directory hash") != std::string::npos,
              "a torn directory is refused");
        SessionReadLimits lg;
        lg.geometry = a.geometry; (*lg.geometry)[0] += 1;
        check(!park_file_read(p, 7, out, take_all, lg, rs, err, &ss) && !ss.target_written, "another geometry is refused");
        SessionReadLimits lk;
        lk.max_kv_bytes.assign(3, {MiB, MiB, MiB, MiB, MiB});
        check(!park_file_read(p, 7, out, take_all, lk, rs, err, &ss) && !ss.target_written, "a part over this engine's limit is refused");
        SessionReadLimits lt;
        lt.max_tokens = 3;
        check(!park_file_read(p, 7, out, take_all, lt, rs, err, &ss) && !ss.target_written, "a state over the token limit is refused");
        SessionReadLimits la;
        la.admit = [](uint64_t, std::string& why) { why = "no RAM"; return false; };
        check(!park_file_read(p, 7, out, take_all, la, rs, err, &ss) && ss.error == SessionError::memory && !ss.target_written,
              "the RAM preflight refuses as memory");
        SessionKvTarget refuse = take_all;
        refuse.begin = [](const SavedConversation&, uint64_t, std::string& why) { why = "not this session's"; return false; };
        check(!park_file_read(p, 7, out, refuse, {}, rs, err, &ss) && !ss.target_written, "the target refuses at begin");
        check(!park_file_read((dir / "none.park").string(), 7, out, take_all, {}, rs, err, &ss) && ss.error == SessionError::io,
              "a missing file is an I/O failure");
        check(park_file_read(p, 7, out, take_all, {}, rs, err, &ss) && ss.error == SessionError::none, "the untouched file still reads");
    }
    std::error_code ec;
    {
        // --conversation-cache-live-tail: the live state parked as one of the checkpoints', its record shared
        const std::string p = (dir / "live-at.park").string();
        const SavedConversation a = image(sz);
        SavedConversation m = meta_of(a);
        m.live = ConversationCheckpoint{};
        m.live.ids = a.checkpoints[1].ids;
        m.live.imgs = a.checkpoints[1].imgs;
        ParkFileState st, s0;
        ParkWriteStats ws, w0;
        check(!park_file_write(p, 60, nullptr, m, a.checkpoints, sources(a), caps(3, cap), {}, 0, beat, s0, w0, err, 0),
              "a live state that is not that checkpoint's is refused");
        std::filesystem::remove(p);
        check(!park_file_write(p, 60, nullptr, m, a.checkpoints, sources(a), caps(3, cap), {}, 0, beat, s0, w0, err, 2),
              "a checkpoint index past the chain is refused");
        std::filesystem::remove(p);
        check(park_file_write(p, 60, nullptr, m, a.checkpoints, sources(a), caps(3, cap), {}, 0, beat, st, ws, err, 1),
              "park as of checkpoint 1");
        check(st.live.offset == st.checkpoints[1].offset && st.live_is_checkpoint(), "the live record is checkpoint 1's");
        uint64_t kv = 0;
        for (const auto& l : sz) for (size_t n : l) kv += n;
        check(st.bytes() == 4096 + st.directory.length + st.checkpoints[0].length + st.checkpoints[1].length + kv,
              "the shared record is charged once");
        {   // the same conversation with its own live state writes that record too
            const std::string q = (dir / "live-own.park").string();
            ParkFileState so;
            ParkWriteStats wo;
            check(park_file_write(q, 61, nullptr, meta_of(a), a.checkpoints, sources(a), caps(3, cap), {}, 0, beat, so, wo,
                                  err) && wo.written > ws.written, "with its own live state it writes more");
            check(wo.written - ws.written >= a.live.bytes(), "by the live record's size");
        }
        SavedConversation back;
        ParkFileState rs;
        check(park_file_read_image(p, 60, back, {}, rs, err), "read the file back");
        check(same_state(back.live, a.checkpoints[1]) && back.live.park_file == 0 && back.live.used == 0,
              "the live state read back is checkpoint 1's");
        check(back.checkpoints.size() == 2 && same_state(back.checkpoints[1], a.checkpoints[1]) &&
              back.checkpoints[1].park_file == 60, "and checkpoint 1 is still itself, naming its record");
        // an update in place with a live state of its own: the shared record stays (checkpoint 1 still names it)
        SavedConversation b = a;
        b.live = state({1, 2, 3, 4, 5, 6, 7}, 3000, 70);
        std::vector<ConversationCheckpoint> chain = back.checkpoints;
        chain.push_back(state({1, 2, 3, 4, 5}, 3000, 80));
        ParkFileState su;
        ParkWriteStats wu;
        check(park_file_write(p, 60, &rs, meta_of(b), chain, sources(b), caps(3, cap), {}, 0, beat, su, wu, err),
              "update in place with its own live state");
        check(wu.kept_checkpoints == 2 && !su.live_is_checkpoint() && su.checkpoints[1].offset == st.checkpoints[1].offset,
              "both checkpoints kept, the live state its own record");
        SavedConversation back2;
        ParkFileState rs2;
        check(park_file_read_image(p, 60, back2, {}, rs2, err) && same_state(back2.checkpoints[1], a.checkpoints[1]) &&
              same_state(back2.live, b.live) && same_state(back2.checkpoints[2], chain[2]),
              "the record the old live state shared was not punched out");
        // and again, as of the newest (kept) checkpoint: the live record punched, its checkpoint's shared
        SavedConversation m3 = meta_of(b);
        m3.live = ConversationCheckpoint{};
        m3.live.ids = chain[2].ids;
        m3.live.imgs = chain[2].imgs;
        ParkFileState s3;
        ParkWriteStats w3;
        check(park_file_write(p, 60, &rs2, m3, back2.checkpoints, sources(b), caps(3, cap), {}, 0, beat, s3, w3, err, 2),
              "update in place as of the kept checkpoint 2");
        check(w3.kept_checkpoints == 3 && s3.live.offset == s3.checkpoints[2].offset &&
              s3.bytes() == 4096 + s3.directory.length + s3.checkpoints[0].length + s3.checkpoints[1].length +
                                s3.checkpoints[2].length + kv, "nothing written for the live state, charged once");
        SavedConversation back3;
        ParkFileState rs3;
        check(park_file_read_image(p, 60, back3, {}, rs3, err) && same_state(back3.live, chain[2]) &&
              same_state(back3.checkpoints[2], chain[2]) && same_state(back3.checkpoints[0], a.checkpoints[0]),
              "it reads back as of checkpoint 2");
    }

    fs::remove_all(dir, ec);
    std::printf("conversation_park_test: %d checks passed\n", checks);
}
