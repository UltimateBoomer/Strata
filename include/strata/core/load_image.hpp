// A load image (STRATA_LOAD_IMAGE=<dir>, Linux): the bytes the startup assembles from scattered expert blobs - the GPU
// cache's slots filled from the profile, and the RAM copy of the experts outside it (pin_cache_complement) - kept as
// one file each, in the order they are loaded, so the next start (and a switch back to this variant) reads them
// sequentially instead of blob by blob.  Measured on pc26 (RX 7800 XT, 990 EVO Plus behind LUKS, IQ3_XXS with a
// 4300-slot cache and a 22 GiB RAM budget): the GPU cache's 7 GiB 6.7 s -> 1.3 s, the RAM copy's 22 GiB 5 s -> 3.3 s,
// the engine's start to READY 21.7 s -> 8.4 s.  Pure host code: no CUDA.
//
// File: one 4 KiB header (magic, the key, the payload's size), then the payload, padded to 4 KiB.  The key binds the
// image to the expert files (device, inode, size, mtime) and to the exact layout (which expert sits where), so a
// mismatch is a miss and the startup reads the blobs as before, then writes the image again.  Written to a temporary
// name and renamed, so a torn write is never read.  Not a checksum: an image is as trusted as the pack beside it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace strata::core::load_image {

/// The image directory (STRATA_LOAD_IMAGE), or empty: off (and always off outside Linux).
std::string dir();
/// Whether a missing or stale image is written after the blobs were read (STRATA_LOAD_IMAGE_WRITE=0: read only).
bool write_enabled();
/// The identity of an open file, for a key: device, inode, size and modification time.
uint64_t file_identity(int fd, uint64_t seed);

/// One image opened for reading: `open` is true only when the file exists with this key and payload size.
class Reader {
public:
    Reader() = default;
    ~Reader();
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;
    bool open(const std::string& path, uint64_t key, uint64_t bytes, std::string& why);
    bool is_open() const { return fd_ >= 0; }
    /// Payload bytes [at, at + n) into `dst`, on `threads` threads in 8 MiB requests (0: STRATA_LOAD_IMAGE_THREADS,
    /// default 4).  `at` must be 4 KiB-aligned; `dst`
    /// too when the file is unbuffered (a short tail goes through a bounce block).
    bool read(uint64_t at, void* dst, uint64_t n, int threads = 0) const;
    void close();
private:
    int fd_ = -1;
    bool direct_ = false;
    uint64_t bytes_ = 0;
};

/// Writes an image of `bytes` payload bytes: `fill(at, n, buf)` puts payload bytes [at, at + n) into `buf` (chunks of
/// up to 64 MiB, in order).  To `path`.tmp, then renamed over `path`.  False (and `err`) on any failure; the
/// temporary file is removed then.
bool write(const std::string& path, uint64_t key, uint64_t bytes,
           const std::function<bool(uint64_t at, uint64_t n, void* buf)>& fill, std::string& err);

}  // namespace strata::core::load_image
