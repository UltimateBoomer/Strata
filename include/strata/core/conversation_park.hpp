// --conversation-cache-dir: a parked conversation's file, updated in place when the same conversation is parked
// again.  Pure host code, no CUDA; POSIX only (the file tier is off elsewhere).
//
// A session file (conversation_file.hpp) is written whole: its K/V parts follow one another and one hash covers the
// payload, so a conversation read back and parked again a few tokens later wrote every byte once more (2.68 GB for
// 15 tokens at 121K).  A park file instead gives every K/V part its own region, reserved for the most that part can
// hold at --max-context (a sparse file: only what is written takes disk space), and keeps the running states as
// records of their own:
//
//   [0, 4096)      superblock: magic STRPARK\x01, version, generation, the directory's offset, length and hash
//   regions        layer j's part p (k, v, k_scale, v_scale, pooled) at a 4 KiB-aligned offset, a hash per MiB
//   records        the live running state and each checkpoint, appended at 4 KiB-aligned offsets, each hashed
//   directory      geometry, layer range, cvec, the records in chain order, each layer's header and regions
//
// An update writes only what changed: each part's bytes past the prefix the caller says is unchanged (`keep`, the rule
// conversation_kv_save reuses a retained image by), new checkpoint records (one already in this file is kept when the
// checkpoint names it: ConversationCheckpoint::park_file/park_record), the live state and the directory; then the
// superblock is pointed at the new directory and the records it no longer names are punched out.  Not crash-safe:
// the engine removes every park file at start, and a failed update leaves a file the caller removes.  The files are
// written by this process for itself; the hashes catch a torn or corrupted read, they do not authenticate a file.
#pragma once

#include "strata/core/conversation_file.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::core {

struct ParkRecord {
    uint64_t offset = 0, length = 0, hash = 0;
};
struct ParkRegion {
    uint64_t offset = 0, cap = 0, size = 0;
    std::vector<uint64_t> hashes;   // one per kParkHashBlock of [0, size)
};
struct ParkLayer {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    std::array<ParkRegion, 5> parts;
};
// What a park file holds, as written last or read back: what an in-place update starts from.
struct ParkFileState {
    uint64_t file = 0;          // the cache's file id, which the checkpoints read from it carry
    uint64_t generation = 0;
    uint64_t end = 0;           // where the next record goes
    ParkRecord live, directory;
    std::vector<ParkRecord> checkpoints;   // in the order of the conversation's checkpoints
    std::vector<ParkLayer> layers;
    // what the file holds that is in use: the budget charges this (regions' used bytes, records, directory)
    uint64_t bytes() const;
};
struct ParkWriteStats {
    uint64_t written = 0;        // bytes written to the file (K/V, records, directory, superblock)
    uint64_t kept_kv = 0;        // K/V bytes left as they were
    uint64_t kept_records = 0;   // checkpoint record bytes left as they were
    size_t kept_checkpoints = 0;
};
inline constexpr uint64_t kParkHashBlock = 1ull << 20;

// `base`: the file's state for an update in place (this file, written or read back by this process), nullptr for a
// new file (an existing one at `path` is refused).  `meta`: the live running state, geometry, layer range and cvec
// (its K/V empty: `kv` supplies it, one source per layer); `checkpoints`: the chain to keep.  `caps[j][p]`: the most
// layer j's part p can hold, the region a new file reserves.  `keep[j][p]`: the leading bytes of that part unchanged
// since `base` was written or read (ignored without a base).  `min_free_bytes`: the free-space preflight, as a SAVE's.
// `beat`: called after every block moved.  On success `state` describes the file; on failure the file may be torn and
// the caller removes it.
bool park_file_write(const std::string& path, uint64_t file, const ParkFileState* base, const SavedConversation& meta,
                     const std::vector<ConversationCheckpoint>& checkpoints, const std::vector<SessionKvSource>& kv,
                     const std::vector<std::array<uint64_t, 5>>& caps, const std::vector<std::array<uint64_t, 5>>& keep,
                     uint64_t min_free_bytes, const std::function<void()>& beat, ParkFileState& state,
                     ParkWriteStats& stats, std::string& error);

// Reads the file back as session_file_read_streamed does: `meta` gets the running state and the checkpoints (each
// marked with this file and its record, so an update keeps them), the K/V goes to `target`, layer by layer, each
// block checked against its hash before it is handed on.  `limits` bound every count before it is allocated and
// `limits.admit` is asked for the records' RAM.  `status->target_written` as for a session file.
bool park_file_read(const std::string& path, uint64_t file, SavedConversation& meta, const SessionKvTarget& target,
                    const SessionReadLimits& limits, ParkFileState& state, std::string& error,
                    SessionStatus* status = nullptr);
// The same into a whole host image (its K/V in `image.kv`).
bool park_file_read_image(const std::string& path, uint64_t file, SavedConversation& image,
                          const SessionReadLimits& limits, ParkFileState& state, std::string& error);

} // namespace strata::core
