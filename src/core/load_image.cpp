// src/core/load_image.cpp - the startup's load images.  Read the header first.
#include "strata/core/load_image.hpp"

#include "strata/core/conversation_file.hpp"   // SessionHasher

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::core::load_image {

#if defined(__linux__)
namespace {
constexpr uint64_t kBlock = 4096;
constexpr uint64_t kHeader = kBlock;
constexpr char kMagic[8] = {'S', 'T', 'R', 'L', 'I', 'M', 'G', '1'};
constexpr uint64_t kRequest = 8ull << 20;    // one pread
constexpr uint64_t kChunk = 64ull << 20;     // one `fill` of a write

uint64_t round_up(uint64_t n) { return (n + kBlock - 1) / kBlock * kBlock; }

struct AlignedBuffer {
    void* p = nullptr;
    explicit AlignedBuffer(size_t n) {
        if (posix_memalign(&p, (size_t) kBlock, n) != 0) p = nullptr;
    }
    ~AlignedBuffer() { std::free(p); }
};
}  // namespace
#endif

std::string dir() {
#if defined(__linux__)
    const char* v = std::getenv("STRATA_LOAD_IMAGE");
    return v != nullptr ? std::string(v) : std::string();
#else
    return {};
#endif
}

bool write_enabled() {
    const char* v = std::getenv("STRATA_LOAD_IMAGE_WRITE");
    return v == nullptr || std::atoi(v) != 0;
}

uint64_t file_identity(int fd, uint64_t seed) {
    SessionHasher h(seed);
#if defined(__linux__)
    struct stat st{};
    if (fd >= 0 && fstat(fd, &st) == 0) {
        const uint64_t f[6] = {(uint64_t) st.st_dev, (uint64_t) st.st_ino, (uint64_t) st.st_size,
                               (uint64_t) st.st_mtim.tv_sec, (uint64_t) st.st_mtim.tv_nsec, 1};
        h.update(f, sizeof f);
    } else {
        const uint64_t none = 0;
        h.update(&none, sizeof none);
    }
#else
    (void) fd;
#endif
    return h.digest();
}

Reader::~Reader() { close(); }

void Reader::close() {
#if defined(__linux__)
    if (fd_ >= 0) ::close(fd_);
#endif
    fd_ = -1;
}

bool Reader::open(const std::string& path, uint64_t key, uint64_t bytes, std::string& why) {
    close();
#if defined(__linux__)
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    bool direct = fd >= 0;
    if (fd < 0 && errno == EINVAL) fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);   // no O_DIRECT here (tmpfs)
    if (fd < 0) {
        why = errno == ENOENT ? "none yet" : std::string("cannot open: ") + std::strerror(errno);
        return false;
    }
    AlignedBuffer head((size_t) kHeader);
    if (head.p == nullptr || pread(fd, head.p, (size_t) kHeader, 0) != (ssize_t) kHeader) {
        why = "short header";
        ::close(fd);
        return false;
    }
    uint64_t f[2];
    std::memcpy(f, (const uint8_t*) head.p + sizeof kMagic, sizeof f);
    struct stat st{};
    if (std::memcmp(head.p, kMagic, sizeof kMagic) != 0) why = "not an image";
    else if (f[0] != key) why = "made for another layout or other files";
    else if (f[1] != bytes) why = "another size";
    else if (fstat(fd, &st) != 0 || (uint64_t) st.st_size < kHeader + round_up(bytes)) why = "truncated";
    else {
        fd_ = fd;
        direct_ = direct;
        bytes_ = bytes;
        return true;
    }
    ::close(fd);
    return false;
#else
    (void) path; (void) key; (void) bytes;
    why = "Linux only";
    return false;
#endif
}

bool Reader::read(uint64_t at, void* dst, uint64_t n, int threads) const {
#if defined(__linux__)
    if (fd_ < 0 || at % kBlock != 0 || at > bytes_ || n > bytes_ - at) return false;
    if (direct_ && (uintptr_t) dst % kBlock != 0) return false;
    const uint64_t whole = direct_ ? n / kBlock * kBlock : n;   // the rest: one bounce block
    const uint64_t requests = (whole + kRequest - 1) / kRequest;
    std::atomic<uint64_t> next{0};
    std::atomic<bool> ok{true};
    auto work = [&] {
        for (uint64_t r; ok.load(std::memory_order_relaxed) && (r = next.fetch_add(1)) < requests;) {
            const uint64_t off = r * kRequest, len = std::min(kRequest, whole - off);
            for (uint64_t done = 0; done < len;) {
                const ssize_t got = pread(fd_, (uint8_t*) dst + off + done, (size_t) (len - done),
                                          (off_t) (kHeader + at + off + done));
                if (got <= 0) {
                    if (got < 0 && errno == EINTR) continue;
                    ok = false;
                    return;
                }
                done += (uint64_t) got;
            }
        }
    };
    if (threads <= 0) {   // STRATA_LOAD_IMAGE_THREADS, default 4
        const char* v = std::getenv("STRATA_LOAD_IMAGE_THREADS");
        threads = v != nullptr && std::atoi(v) > 0 ? std::atoi(v) : 4;
    }
    const int t = (int) std::max<uint64_t>(1, std::min<uint64_t>((uint64_t) threads, requests));
    std::vector<std::thread> pool;
    for (int i = 1; i < t; ++i) pool.emplace_back(work);
    work();
    for (auto& th : pool) th.join();
    if (ok && whole < n) {
        AlignedBuffer tail((size_t) kBlock);
        if (tail.p == nullptr || pread(fd_, tail.p, (size_t) kBlock, (off_t) (kHeader + at + whole)) != (ssize_t) kBlock)
            return false;
        std::memcpy((uint8_t*) dst + whole, tail.p, (size_t) (n - whole));
    }
    return ok;
#else
    (void) at; (void) dst; (void) n; (void) threads;
    return false;
#endif
}

bool write(const std::string& path, uint64_t key, uint64_t bytes,
           const std::function<bool(uint64_t at, uint64_t n, void* buf)>& fill, std::string& err) {
#if defined(__linux__)
    const std::string tmp = path + ".tmp";
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);   // an error shows at the open
    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_DIRECT, 0644);
    if (fd < 0 && errno == EINVAL) fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) {
        err = "cannot create " + tmp + ": " + std::strerror(errno);
        return false;
    }
    auto put = [&](const void* p, uint64_t n, uint64_t off) {
        for (uint64_t done = 0; done < n;) {
            const ssize_t w = pwrite(fd, (const uint8_t*) p + done, (size_t) (n - done), (off_t) (off + done));
            if (w <= 0) {
                if (w < 0 && errno == EINTR) continue;
                return false;
            }
            done += (uint64_t) w;
        }
        return true;
    };
    auto fail = [&](const std::string& m) {
        err = m;
        ::close(fd);
        (void) ::unlink(tmp.c_str());
        return false;
    };
    // two buffers: the next chunk is filled while this one is written
    AlignedBuffer head((size_t) kHeader), a((size_t) kChunk), b((size_t) kChunk);
    if (head.p == nullptr || a.p == nullptr || b.p == nullptr) return fail("no memory for the write buffers");
    std::memset(head.p, 0, (size_t) kHeader);
    std::memcpy(head.p, kMagic, sizeof kMagic);
    const uint64_t f[2] = {key, bytes};
    std::memcpy((uint8_t*) head.p + sizeof kMagic, f, sizeof f);
    void* buf[2] = {a.p, b.p};
    bool filled = fill(0, std::min(kChunk, bytes), buf[0]);
    for (uint64_t at = 0, i = 0; at < bytes; at += kChunk, ++i) {
        if (!filled) return fail("the image's bytes could not be gathered");
        const uint64_t n = std::min(kChunk, bytes - at);
        std::memset((uint8_t*) buf[i % 2] + n, 0, (size_t) (round_up(n) - n));
        std::thread writer([&, at, n, i] { filled = put(buf[i % 2], round_up(n), kHeader + at); });
        const uint64_t next = at + kChunk;
        bool next_ok = next >= bytes || fill(next, std::min(kChunk, bytes - next), buf[(i + 1) % 2]);
        writer.join();
        if (!filled) return fail(std::string("write failed: ") + std::strerror(errno));
        filled = next_ok;
    }
    // the header last: a file without it is never taken for an image
    if (!put(head.p, kHeader, 0)) return fail(std::string("write failed: ") + std::strerror(errno));
    if (fsync(fd) != 0) return fail(std::string("fsync failed: ") + std::strerror(errno));
    ::close(fd);
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        err = "cannot rename " + tmp + ": " + std::strerror(errno);
        (void) ::unlink(tmp.c_str());
        return false;
    }
    return true;
#else
    (void) path; (void) key; (void) bytes; (void) fill;
    err = "Linux only";
    return false;
#endif
}

}  // namespace strata::core::load_image
