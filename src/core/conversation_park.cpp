// Park files (conversation_park.hpp): a parked conversation's K/V in per-part regions and its running states as records,
// so the same conversation parked again writes only what changed.
#include "strata/core/conversation_park.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#if defined(__linux__)
#include <linux/falloc.h>
#endif
#endif

namespace strata::core {

uint64_t ParkFileState::bytes() const {
    uint64_t n = 4096 + directory.length + live.length;
    for (const auto& c : checkpoints) n += c.length;
    for (const auto& l : layers)
        for (const auto& p : l.parts) n += p.size;
    return n;
}

#if defined(_WIN32)

bool park_file_write(const std::string&, uint64_t, const ParkFileState*, const SavedConversation&,
                     const std::vector<ConversationCheckpoint>&, const std::vector<SessionKvSource>&,
                     const std::vector<std::array<uint64_t, 5>>&, const std::vector<std::array<uint64_t, 5>>&, uint64_t,
                     const std::function<void()>&, ParkFileState&, ParkWriteStats&, std::string& error) {
    error = "park files need a POSIX system";
    return false;
}
bool park_file_read(const std::string&, uint64_t, SavedConversation&, const SessionKvTarget&, const SessionReadLimits&,
                    ParkFileState&, std::string& error, SessionStatus* status) {
    if (status) *status = {};
    error = "park files need a POSIX system";
    return false;
}

#else

namespace {
constexpr char kMagic[8] = {'S', 'T', 'R', 'P', 'A', 'R', 'K', '\x01'};
constexpr char kDirMagic[8] = {'P', 'A', 'R', 'K', 'D', 'I', 'R', '1'};
constexpr uint32_t kVersion = 1;
constexpr uint64_t kSuper = 4096, kAlign = 4096;
constexpr size_t kChunk = 16u << 20;   // 16 hash blocks: what one read or write moves
constexpr uint64_t kMaxDirectory = 64ull << 20;
constexpr uint64_t H = kParkHashBlock;
static_assert(kChunk % H == 0, "a chunk holds whole hash blocks");

uint64_t align_up(uint64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }
uint64_t block_seed(uint64_t region, uint64_t block) { return region ^ (block * 0x9E3779B97F4A7C15ull); }
std::string sys(const char* what) { return std::string(what) + ": " + std::strerror(errno); }

struct FreeAligned { void operator()(uint8_t* p) const { std::free(p); } };
using Buffer = std::unique_ptr<uint8_t[], FreeAligned>;
Buffer chunk_buffer() { return Buffer(static_cast<uint8_t*>(std::aligned_alloc(kAlign, kChunk))); }

// Whole aligned pages move with O_DIRECT where the filesystem takes it (as session files do), so a park neither waits
// on dirty-page writeback nor pushes the model files' cached pages out of a RAM the engine has mostly locked; the
// unaligned ends of a span go through the page cache.  STRATA_SESSION_BUFFERED=1: all buffered.
bool direct_wanted() {
    const char* v = std::getenv("STRATA_SESSION_BUFFERED");
    return !(v && v[0] == '1');
}
bool aligned_with(const void* p, uint64_t at) { return ((uintptr_t) p - at) % kAlign == 0; }

class File {
public:
    ~File() {
        if (fd_ >= 0) ::close(fd_);
        if (direct_ >= 0) ::close(direct_);
    }
    bool open(const std::string& path, int flags, std::string& error) {
        fd_ = ::open(path.c_str(), flags | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd_ < 0) { error = sys(("park file " + path).c_str()); kind = SessionError::io; return false; }
#if defined(O_DIRECT)
        if (direct_wanted())   // a second descriptor of the same file (an EINVAL filesystem: none, all buffered)
            direct_ = ::open(path.c_str(), (flags & O_ACCMODE) | O_CLOEXEC | O_NOFOLLOW | O_DIRECT);
#endif
        return true;
    }
    // `n` bytes from `p` at `at`; the pages of it that are whole and aligned in both go direct
    bool write(const void* p, size_t n, uint64_t at, std::string& error) {
        if (direct_ >= 0 && n >= kAlign && aligned_with(p, at)) {
            const auto* c = static_cast<const uint8_t*>(p);
            const size_t head = (size_t) std::min<uint64_t>(n, align_up(at) - at);
            const size_t mid = (n - head) / kAlign * kAlign;
            return write_fd(fd_, c, head, at, error) && write_fd(direct_, c + head, mid, at + head, error) &&
                   write_fd(fd_, c + head + mid, n - head - mid, at + head + mid, error);
        }
        return write_fd(fd_, p, n, at, error);
    }
    // `n` bytes into `p` (an aligned buffer of `room` bytes) at an aligned `at`: one direct read of whole pages when
    // they fit, else buffered
    bool read_into(uint8_t* p, size_t n, uint64_t at, size_t room, std::string& error) {
        const size_t whole = (size_t) align_up(n);
        if (direct_ >= 0 && at % kAlign == 0 && aligned_with(p, 0) && whole <= room && n) {
            size_t got = 0;
            while (got < whole) {
                const ssize_t r = ::pread(direct_, p + got, whole - got, (off_t) (at + got));
                if (r < 0 && errno == EINTR) continue;
                if (r <= 0) break;   // a short file or a refusal: what is missing comes buffered
                got += (size_t) r;
                if (got % kAlign) break;
            }
            if (got >= n) return true;
            return read(p + got, n - got, at + got, error);
        }
        return read(p, n, at, error);
    }
private:
    bool write_fd(int fd, const void* p, size_t n, uint64_t at, std::string& error) {
        const auto* c = static_cast<const uint8_t*>(p);
        while (n) {
            const ssize_t w = ::pwrite(fd, c, n, (off_t) at);
            if (w < 0 && errno == EINTR) continue;
            if (w <= 0) {
                kind = (errno == ENOSPC || errno == EDQUOT || errno == EFBIG) ? SessionError::storage : SessionError::io;
                error = sys("park file write");
                return false;
            }
            c += w; n -= (size_t) w; at += (uint64_t) w;
        }
        return true;
    }
public:
    bool read(void* p, size_t n, uint64_t at, std::string& error) {
        auto* c = static_cast<uint8_t*>(p);
        while (n) {
            const ssize_t r = ::pread(fd_, c, n, (off_t) at);
            if (r < 0 && errno == EINTR) continue;
            if (r < 0) { kind = SessionError::io; error = sys("park file read"); return false; }
            if (r == 0) { error = "park file: ends early"; return false; }
            c += r; n -= (size_t) r; at += (uint64_t) r;
        }
        return true;
    }
    // gives the bytes' disk space back; a filesystem without holes keeps it until the file goes
    void punch(uint64_t at, uint64_t n) {
#if defined(__linux__)
        if (n) (void) ::fallocate(fd_, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, (off_t) at, (off_t) n);
#else
        (void) at; (void) n;
#endif
    }
public:
    int fd() const { return fd_; }
    // what kind of failure the last failed call was: io for a failed system call, invalid otherwise (a short file)
    SessionError kind = SessionError::invalid;
private:
    int fd_ = -1, direct_ = -1;
};

// A record's bytes appended at `offset` through the chunk buffer, hashed as they go (seed: the offset).
class RecordOut {
public:
    RecordOut(File& f, uint8_t* buf, uint64_t offset, const std::function<void()>& beat, std::string& error)
        : f_(f), buf_(buf), offset_(offset), at_(offset), hash_(offset), beat_(beat), error_(error) {}
    void raw(const void* p, size_t n) {
        const auto* c = static_cast<const uint8_t*>(p);
        hash_.update(c, n);
        length_ += n;
        while (n && ok_) {
            const size_t m = std::min(n, kChunk - fill_);
            std::memcpy(buf_ + fill_, c, m);
            fill_ += m; c += m; n -= m;
            if (fill_ == kChunk) flush();
        }
    }
    void u64(uint64_t v) { raw(&v, 8); }
    template<class T> void vec(const std::vector<T>& v) { u64(v.size()); raw(v.data(), v.size() * sizeof(T)); }
    bool finish(ParkRecord& r) {
        flush();
        r = {offset_, length_, hash_.digest()};
        return ok_;
    }
private:
    void flush() {
        if (!ok_ || !fill_) return;
        ok_ = f_.write(buf_, fill_, at_, error_);
        at_ += fill_;
        fill_ = 0;
        if (beat_) beat_();
    }
    File& f_;
    uint8_t* buf_;
    uint64_t offset_, at_, length_ = 0;
    size_t fill_ = 0;
    SessionHasher hash_;
    const std::function<void()>& beat_;
    std::string& error_;
    bool ok_ = true;
};

void put_state(RecordOut& o, const ConversationCheckpoint& c) {
    o.vec(c.ids);
    o.u64(c.imgs.size());
    for (const auto& k : c.imgs) { o.u64((uint64_t) k.start); o.u64(k.hash); }
    o.vec(c.gdn); o.vec(c.ple); o.vec(c.tails); o.vec(c.dead); o.vec(c.block_pos);
}

// A record read back through the chunk buffer, every count checked against the bytes left and the limits.
class RecordIn {
public:
    RecordIn(File& f, uint8_t* buf, const ParkRecord& r, std::string& error)
        : f_(f), buf_(buf), at_(r.offset), left_(r.length), want_(r.hash), hash_(r.offset), error_(error) {}
    bool fail(const std::string& m) { if (error_.empty()) error_ = "park file: " + m; return false; }
    bool raw(void* p, size_t n) {
        if (n > left_) return fail("record ends early");
        auto* c = static_cast<uint8_t*>(p);
        while (n) {
            if (pos_ == have_) {
                have_ = (size_t) std::min<uint64_t>(kChunk, left_);
                pos_ = 0;
                if (!f_.read_into(buf_, have_, at_, kChunk, error_)) return false;
                at_ += have_;
            }
            const size_t m = std::min(n, have_ - pos_);
            std::memcpy(c, buf_ + pos_, m);
            hash_.update(buf_ + pos_, m);
            pos_ += m; c += m; n -= m; left_ -= m;
        }
        return true;
    }
    bool u64(uint64_t& v) { return raw(&v, 8); }
    template<class T> bool vec(std::vector<T>& v, uint64_t limit) {
        uint64_t n = 0;
        if (!u64(n)) return false;
        if (n > left_ / sizeof(T) || n > limit) return fail("record count exceeds its bounds");
        v.resize((size_t) n);
        return raw(v.data(), (size_t) n * sizeof(T));
    }
    bool done() {
        if (left_ != 0 || pos_ != have_) return fail("record holds more than it says");
        if (hash_.digest() != want_) return fail("record hash mismatch");
        return true;
    }
    uint64_t left() const { return left_; }
private:
    File& f_;
    uint8_t* buf_;
    uint64_t at_, left_, want_;
    size_t have_ = 0, pos_ = 0;
    SessionHasher hash_;
    std::string& error_;
};

bool get_state(RecordIn& in, ConversationCheckpoint& c, const SessionReadLimits& l) {
    uint64_t n = 0;
    if (!in.vec(c.ids, l.max_tokens) || !in.u64(n)) return false;
    if (n > in.left() / 16 || n > l.max_tokens) return in.fail("record count exceeds its bounds");
    c.imgs.resize((size_t) n);
    for (auto& k : c.imgs) {
        uint64_t start = 0;
        if (!in.u64(start) || !in.u64(k.hash)) return false;
        k.start = (int64_t) start;
    }
    const auto& m = l.max_state_bytes;
    return in.vec(c.gdn, m[0]) && in.vec(c.ple, m[1]) && in.vec(c.tails, m[2]) && in.vec(c.dead, m[3]) &&
           in.vec(c.block_pos, m[4]) && in.done();
}

// ---- the directory: small, built and parsed in RAM
struct Bytes {
    std::vector<uint8_t> b;
    void raw(const void* p, size_t n) { const auto* c = static_cast<const uint8_t*>(p); b.insert(b.end(), c, c + n); }
    void u64(uint64_t v) { raw(&v, 8); }
    void rec(const ParkRecord& r) { u64(r.offset); u64(r.length); u64(r.hash); }
};
struct Parse {
    const std::vector<uint8_t>& b;
    size_t at = 0;
    std::string& error;
    bool fail(const std::string& m) { if (error.empty()) error = "park file: " + m; return false; }
    bool raw(void* p, size_t n) {
        if (n > b.size() - at) return fail("directory ends early");
        std::memcpy(p, b.data() + at, n);
        at += n;
        return true;
    }
    bool u64(uint64_t& v) { return raw(&v, 8); }
    bool i64(int64_t& v) { return raw(&v, 8); }
    bool rec(ParkRecord& r) { return u64(r.offset) && u64(r.length) && u64(r.hash); }
    bool count(uint64_t& n, size_t elem, uint64_t limit) {
        if (!u64(n)) return false;
        if (n > (b.size() - at) / elem || n > limit) return fail("directory count exceeds its bounds");
        return true;
    }
};

std::vector<uint8_t> directory_bytes(const ParkFileState& st, const SavedConversation& meta,
                                     const std::vector<ConversationCheckpoint>& checkpoints) {
    Bytes o;
    o.raw(kDirMagic, 8);
    for (int64_t g : meta.geometry) o.u64((uint64_t) g);
    o.u64((uint64_t) meta.layer_lo); o.u64((uint64_t) meta.layer_hi); o.u64(meta.cvec ? 1 : 0);
    o.u64(st.generation); o.u64(st.end);
    o.rec(st.live);
    o.u64(st.checkpoints.size());
    for (size_t i = 0; i < st.checkpoints.size(); ++i) { o.rec(st.checkpoints[i]); o.u64(checkpoints[i].used); }
    o.u64(st.layers.size());
    for (const auto& l : st.layers) {
        for (int64_t v : {int64_t(l.format), l.cells, l.heads, l.head_dim, l.page_size, l.pooled_rows, l.idx_dim})
            o.u64((uint64_t) v);
        for (const auto& p : l.parts) {
            o.u64(p.offset); o.u64(p.cap); o.u64(p.size); o.u64(p.hashes.size());
            o.raw(p.hashes.data(), p.hashes.size() * 8);
        }
    }
    return std::move(o.b);
}

// superblock: magic[8] version:u32 size:u32 generation:u64 directory offset:u64 length:u64 hash:u64, hash of 0..47
void super_bytes(uint8_t* s, const ParkFileState& st) {
    std::memset(s, 0, kSuper);
    std::memcpy(s, kMagic, 8);
    const uint32_t version = kVersion, size = (uint32_t) kSuper;
    std::memcpy(s + 8, &version, 4);
    std::memcpy(s + 12, &size, 4);
    std::memcpy(s + 16, &st.generation, 8);
    std::memcpy(s + 24, &st.directory.offset, 8);
    std::memcpy(s + 32, &st.directory.length, 8);
    std::memcpy(s + 40, &st.directory.hash, 8);
    const uint64_t h = session_hash64(s, 48, 0);
    std::memcpy(s + 48, &h, 8);
}

std::string folder_of(const std::string& path) {
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? "." : slash == 0 ? "/" : path.substr(0, slash);
}
} // namespace

bool park_file_write(const std::string& path, uint64_t file, const ParkFileState* base, const SavedConversation& meta,
                     const std::vector<ConversationCheckpoint>& checkpoints, const std::vector<SessionKvSource>& kv,
                     const std::vector<std::array<uint64_t, 5>>& caps, const std::vector<std::array<uint64_t, 5>>& keep,
                     uint64_t min_free_bytes, const std::function<void()>& beat, ParkFileState& state,
                     ParkWriteStats& stats, std::string& error) {
    stats = {};
    error.clear();
    auto fail = [&](const std::string& m) { if (error.empty()) error = "park file: " + m; return false; };
    if (!meta.kv.empty() || !meta.stage_images.empty() || !meta.live.stage_parts.empty())
        return fail("a park file holds one session's running state; its K/V comes from the sources");
    if (caps.size() != kv.size()) return fail("one region size per K/V layer");
    if (base && (base->file != file || base->layers.size() != kv.size()))
        return fail("the file to update does not have this conversation's layers");
    for (const auto& c : checkpoints)
        if (!c.stage_parts.empty()) return fail("a park file holds one session's checkpoints");
    File f;
    if (!f.open(path, base ? O_RDWR : (O_RDWR | O_CREAT | O_EXCL), error)) return false;

    ParkFileState st;
    st.file = file;
    st.generation = base ? base->generation + 1 : 1;
    st.layers.resize(kv.size());
    uint64_t at = kSuper, adds = 0;
    for (size_t j = 0; j < kv.size(); ++j) {
        ParkLayer& l = st.layers[j];
        l.format = kv[j].format; l.cells = kv[j].cells; l.heads = kv[j].heads; l.head_dim = kv[j].head_dim;
        l.page_size = kv[j].page_size; l.pooled_rows = kv[j].pooled_rows; l.idx_dim = kv[j].idx_dim;
        for (size_t p = 0; p < 5; ++p) {
            ParkRegion& r = l.parts[p];
            if (base) {
                r.offset = base->layers[j].parts[p].offset;
                r.cap = base->layers[j].parts[p].cap;
            } else {
                r.offset = at;
                r.cap = align_up(caps[j][p]);
                at += r.cap;
            }
            r.size = kv[j].sizes[p];
            if (r.size > r.cap) return fail("a K/V part is larger than its region");
            adds += r.size;
        }
    }
    st.end = base ? base->end : at;
    for (const auto& c : checkpoints) adds += c.bytes();
    adds += meta.live.bytes() + (4u << 20);
    if (min_free_bytes) {
        struct statvfs fs{};
        if (::statvfs(folder_of(path).c_str(), &fs) == 0 &&
            (uint64_t) fs.f_bavail * (uint64_t) fs.f_frsize < adds + min_free_bytes) {
            f.kind = SessionError::storage;
            return fail("not enough free disk space for the park file");
        }
    }
    Buffer buf = chunk_buffer();
    if (!buf) return fail("no memory for the write buffer");

    // ---- the K/V: each part's bytes past the unchanged prefix, every hash block from the first it touches
    for (size_t j = 0; j < kv.size(); ++j)
        for (size_t p = 0; p < 5; ++p) {
            ParkRegion& r = st.layers[j].parts[p];
            const ParkRegion* old = base ? &base->layers[j].parts[p] : nullptr;
            uint64_t k = 0;
            if (old && j < keep.size()) k = std::min({keep[j][p], old->size, r.size});
            const uint64_t blocks = (r.size + H - 1) / H, kb = k / H;
            r.hashes.assign((size_t) blocks, 0);
            for (uint64_t b = 0; b < kb; ++b) r.hashes[(size_t) b] = old->hashes[(size_t) b];
            for (uint64_t pos = kb * H; pos < r.size;) {
                const uint64_t n = std::min<uint64_t>(kChunk, r.size - pos);
                const uint64_t mine = k > pos ? std::min(n, k - pos) : 0;   // still the file's own bytes
                if (mine && !f.read(buf.get(), (size_t) mine, r.offset + pos, error)) return false;
                if (n > mine) {
                    const SessionKvSource& s = kv[j];
                    if (!s.read || !s.read(p, (size_t) (pos + mine), buf.get() + mine, (size_t) (n - mine)))
                        return fail("K/V source read" + (s.error ? ": " + s.error() : std::string()));
                    if (!f.write(buf.get() + mine, (size_t) (n - mine), r.offset + pos + mine, error)) return false;
                    stats.written += n - mine;
                }
                for (uint64_t o = 0; o < n; o += H)
                    r.hashes[(size_t) ((pos + o) / H)] =
                        session_hash64(buf.get() + o, (size_t) std::min(H, n - o), block_seed(r.offset, (pos + o) / H));
                pos += n;
                if (beat) beat();
            }
            stats.kept_kv += k;
            if (old && align_up(old->size) > align_up(r.size))
                f.punch(r.offset + align_up(r.size), align_up(old->size) - align_up(r.size));
        }

    // ---- the records: a checkpoint already in this file stays where it is
    auto append = [&](const ConversationCheckpoint& c, ParkRecord& rec) {
        RecordOut o(f, buf.get(), st.end, beat, error);
        put_state(o, c);
        if (!o.finish(rec)) return false;
        stats.written += rec.length;
        st.end = align_up(rec.offset + rec.length);
        return true;
    };
    st.checkpoints.resize(checkpoints.size());
    std::vector<char> kept(base ? base->checkpoints.size() : 0, 0);
    for (size_t i = 0; i < checkpoints.size(); ++i) {
        const ConversationCheckpoint& c = checkpoints[i];
        bool reused = false;
        if (base && c.park_file == file && c.park_record != 0)
            for (size_t k = 0; k < base->checkpoints.size(); ++k)
                if (!kept[k] && base->checkpoints[k].offset == c.park_record) {
                    st.checkpoints[i] = base->checkpoints[k];
                    kept[k] = 1;
                    reused = true;
                    ++stats.kept_checkpoints;
                    break;
                }
        if (!reused && !append(c, st.checkpoints[i])) return false;
    }
    if (!append(meta.live, st.live)) return false;

    // ---- the directory, then the superblock that names it; the records the old directory alone named go
    st.directory.offset = st.end;
    std::vector<uint8_t> dir = directory_bytes(st, meta, checkpoints);
    st.end = align_up(st.directory.offset + dir.size());
    {   // the directory records the end past itself: patch it in (its fixed place after magic, geometry and flags)
        const size_t end_at = 8 + 18 * 8 + 3 * 8 + 8;
        std::memcpy(dir.data() + end_at, &st.end, 8);
    }
    st.directory.length = dir.size();
    st.directory.hash = session_hash64(dir.data(), dir.size(), st.directory.offset);
    if (!f.write(dir.data(), dir.size(), st.directory.offset, error)) return false;
    std::memset(buf.get(), 0, kSuper);
    super_bytes(buf.get(), st);
    if (!f.write(buf.get(), kSuper, 0, error)) return false;
    stats.written += dir.size() + kSuper;
    if (base) {
        auto gone = [&](const ParkRecord& r) { if (r.length) f.punch(r.offset, align_up(r.length)); };
        gone(base->live);
        gone(base->directory);
        for (size_t k = 0; k < base->checkpoints.size(); ++k) if (!kept[k]) gone(base->checkpoints[k]);
    }
    state = std::move(st);
    return true;
}

bool park_file_read(const std::string& path, uint64_t file, SavedConversation& meta, const SessionKvTarget& target,
                    const SessionReadLimits& limits, ParkFileState& state, std::string& error,
                    SessionStatus* status) {
    SessionStatus local;
    SessionStatus& stt = status ? *status : local;
    stt = {};
    stt.error = SessionError::invalid;
    error.clear();
    auto fail = [&](const std::string& m) { if (error.empty()) error = "park file: " + m; return false; };
    File f;
    if (!f.open(path, O_RDONLY, error)) { stt.error = SessionError::io; return false; }
    struct stat sb{};
    if (::fstat(f.fd(), &sb) != 0 || !S_ISREG(sb.st_mode)) return fail("not a regular file");
    const uint64_t size = (uint64_t) sb.st_size;
    Buffer buf = chunk_buffer();
    if (!buf) { stt.error = SessionError::memory; return fail("no memory for the read buffer"); }
    auto io = [&](bool ok) { if (!ok) stt.error = f.kind; return ok; };

    if (size < kSuper || !io(f.read(buf.get(), kSuper, 0, error))) return fail("no superblock");
    const uint8_t* s = buf.get();
    uint32_t version = 0, ssize = 0;
    uint64_t hh = 0;
    ParkFileState st;
    st.file = file;
    std::memcpy(&version, s + 8, 4); std::memcpy(&ssize, s + 12, 4);
    std::memcpy(&st.generation, s + 16, 8);
    std::memcpy(&st.directory.offset, s + 24, 8); std::memcpy(&st.directory.length, s + 32, 8);
    std::memcpy(&st.directory.hash, s + 40, 8); std::memcpy(&hh, s + 48, 8);
    if (std::memcmp(s, kMagic, 8) != 0) return fail("not a park file");
    if (hh != session_hash64(s, 48, 0)) return fail("superblock hash mismatch");
    if (version != kVersion || ssize != kSuper) return fail("unknown park file version");
    if (st.directory.length > kMaxDirectory || st.directory.offset < kSuper ||
        st.directory.offset > size || st.directory.length > size - st.directory.offset)
        return fail("directory out of the file");
    std::vector<uint8_t> dir((size_t) st.directory.length);
    if (!io(f.read(dir.data(), dir.size(), st.directory.offset, error))) return false;
    if (session_hash64(dir.data(), dir.size(), st.directory.offset) != st.directory.hash)
        return fail("directory hash mismatch");

    Parse in{dir, 0, error};
    char magic[8];
    if (!in.raw(magic, 8) || std::memcmp(magic, kDirMagic, 8) != 0) return fail("not a park directory");
    SavedConversation m;
    for (auto& g : m.geometry) if (!in.i64(g)) return false;
    if (limits.geometry && *limits.geometry != m.geometry) return fail("saved with another model geometry");
    uint64_t cvec = 0, generation = 0;
    if (!in.i64(m.layer_lo) || !in.i64(m.layer_hi) || !in.u64(cvec) || !in.u64(generation) || !in.u64(st.end))
        return false;
    if (limits.layer_range && (limits.layer_range->first != m.layer_lo || limits.layer_range->second != m.layer_hi))
        return fail("saved with another layer range");
    if (cvec > 1 || generation != st.generation) return fail("inconsistent directory");
    m.cvec = cvec == 1;
    uint64_t n = 0;
    if (!in.rec(st.live) || !in.count(n, 32, limits.max_checkpoints)) return false;
    st.checkpoints.resize((size_t) n);
    std::vector<uint64_t> used((size_t) n);
    for (size_t i = 0; i < n; ++i) if (!in.rec(st.checkpoints[i]) || !in.u64(used[i])) return false;
    if (!in.count(n, 56 + 5 * 32, limits.max_kv_layers)) return false;
    st.layers.resize((size_t) n);
    uint64_t kv_total = 0;
    for (size_t j = 0; j < st.layers.size(); ++j) {
        ParkLayer& l = st.layers[j];
        int64_t format = 0;
        if (!in.i64(format) || !in.i64(l.cells) || !in.i64(l.heads) || !in.i64(l.head_dim) || !in.i64(l.page_size) ||
            !in.i64(l.pooled_rows) || !in.i64(l.idx_dim)) return false;
        if (format < INT32_MIN || format > INT32_MAX) return fail("invalid K/V format");
        l.format = (int) format;
        for (size_t p = 0; p < 5; ++p) {
            ParkRegion& r = l.parts[p];
            uint64_t hashes = 0;
            if (!in.u64(r.offset) || !in.u64(r.cap) || !in.u64(r.size) || !in.count(hashes, 8, UINT64_MAX)) return false;
            const uint64_t limit = j < limits.max_kv_bytes.size() ? limits.max_kv_bytes[j][p] : UINT64_MAX;
            if (r.size > limit) return fail("a K/V part exceeds this engine's limit");
            if (r.size > r.cap || r.offset < kSuper || r.offset > size || r.size > size - r.offset ||
                hashes != (r.size + H - 1) / H)
                return fail("a K/V region is out of the file");
            r.hashes.resize((size_t) hashes);
            if (!in.raw(r.hashes.data(), r.hashes.size() * 8)) return false;
            kv_total += r.size;
        }
    }
    if (in.at != dir.size()) return fail("directory holds more than it says");
    auto record_in_file = [&](const ParkRecord& r) { return r.offset >= kSuper && r.offset <= size && r.length <= size - r.offset; };
    if (!record_in_file(st.live)) return fail("a record is out of the file");
    uint64_t need = kChunk + (1u << 20) + st.live.length;
    for (const auto& r : st.checkpoints) {
        if (!record_in_file(r)) return fail("a record is out of the file");
        need += r.length;
    }
    std::string why;
    if (limits.admit && !limits.admit(need, why)) { stt.error = SessionError::memory; return fail(why); }

    // ---- the running states, before any K/V byte is handed on
    {
        RecordIn r(f, buf.get(), st.live, error);
        if (!get_state(r, m.live, limits)) { stt.error = f.kind; return false; }
    }
    m.checkpoints.resize(st.checkpoints.size());
    for (size_t i = 0; i < st.checkpoints.size(); ++i) {
        RecordIn r(f, buf.get(), st.checkpoints[i], error);
        if (!get_state(r, m.checkpoints[i], limits)) { stt.error = f.kind; return false; }
        m.checkpoints[i].used = used[i];
        m.checkpoints[i].park_file = file;
        m.checkpoints[i].park_record = st.checkpoints[i].offset;
    }
    if (target.begin && !target.begin(m, st.layers.size(), why)) return fail(why);

    // ---- the K/V, block by block, each checked before it is handed on
    uint64_t done = 0;
    for (size_t j = 0; j < st.layers.size(); ++j) {
        const ParkLayer& l = st.layers[j];
        ConversationKv header;
        header.format = l.format; header.cells = l.cells; header.heads = l.heads; header.head_dim = l.head_dim;
        header.page_size = l.page_size; header.pooled_rows = l.pooled_rows; header.idx_dim = l.idx_dim;
        std::array<uint64_t, 5> sizes{};
        for (size_t p = 0; p < 5; ++p) sizes[p] = l.parts[p].size;
        SessionKvSink sink;
        if (!target.layer || !target.layer(j, header, sizes, sink, why)) return fail(why);
        for (size_t p = 0; p < 5; ++p) {
            const ParkRegion& r = l.parts[p];
            if (sizes[p] != r.size) return fail("a K/V part's size differs from what this session holds");
            if (r.size && !sink.write) return fail("no K/V target");
            for (uint64_t pos = 0; pos < r.size;) {
                const uint64_t c = std::min<uint64_t>(kChunk, r.size - pos);
                if (!io(f.read_into(buf.get(), (size_t) c, r.offset + pos, kChunk, error))) return false;
                for (uint64_t o = 0; o < c; o += H)
                    if (session_hash64(buf.get() + o, (size_t) std::min(H, c - o), block_seed(r.offset, (pos + o) / H)) !=
                        r.hashes[(size_t) ((pos + o) / H)])
                        return fail("K/V block hash mismatch");
                stt.target_written = true;
                if (!sink.write(p, (size_t) pos, buf.get(), (size_t) c)) {
                    stt.error = SessionError::io;
                    return fail("K/V target write" + (sink.error ? ": " + sink.error() : std::string()));
                }
                pos += c;
                done += c;
                if (limits.progress) limits.progress(done, kv_total);
            }
        }
    }
    meta = std::move(m);
    state = std::move(st);
    stt.error = SessionError::none;
    return true;
}

#endif

} // namespace strata::core
