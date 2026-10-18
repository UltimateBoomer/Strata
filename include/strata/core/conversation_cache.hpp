// CPU-only ownership and matching policy for --serve's parked conversations.
// Token equality, image identity, and steering mode are all required for reuse.
#pragma once

#include "strata/core/conversation_buffer.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

struct ConversationImageKey {
    int64_t start = 0;
    uint64_t hash = 0;
    bool operator==(const ConversationImageKey&) const = default;
};

struct ConversationCheckpoint {
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> imgs;
    std::vector<uint8_t> gdn, ple, tails, dead, block_pos;
    uint64_t used = 0; // upstream root-pinned/LRU checkpoint retention
    // A shared-prefix pin (the request key pin=N): this checkpoint is the read-only prefix many suffix queries branch
    // from, so retention never evicts it (conv_cache.hpp) and a parked conversation holding it stays parked.  A run-time
    // mark only: it is not in the session file, a request that pins the same prefix again sets it.
    bool pinned = false;
    // --conversation-cache-dir: the park file (its cache id) and record this checkpoint's running state was read back
    // from, so parking the conversation into that file again keeps the record instead of writing it (0: none).  Cleared
    // whenever the running state is saved anew.  Run-time only.
    uint64_t park_file = 0, park_record = 0;
    // --prefill-pipe on a layer split: the plan a prompt segment [from, to) is read in - its chunk (0: none), the loan
    // its buffers were laid out for (which experts the prompt reads in place and which it streams: that sets bits
    // too), the pipe value it was planned with, whether it is one chunk and whether the plan changed the buffers'
    // chunks.  A periodic checkpoint taken inside such a segment keeps it, so a read that goes on from there reads the
    // rest in the same chunks with the same loan.  Run-time only, like `pinned` (a layer split saves no session files).
    struct PromptPlan {
        int64_t chunk = 0, from = -1, to = -1, lend = 0;
        double pipe = 0.0;
        bool single = false, chunked = false;
    } plan;
    // Ordinary layer-split checkpoints retain each device's running state.
    // Whole-session parking is currently single-GPU and rejects these parts.
    std::vector<ConversationCheckpoint> stage_parts;

    size_t bytes() const {
        size_t n = ids.capacity() * sizeof(int32_t) + imgs.capacity() * sizeof(ConversationImageKey) +
               gdn.capacity() + ple.capacity() + tails.capacity() + dead.capacity() + block_pos.capacity() +
               stage_parts.capacity() * sizeof(ConversationCheckpoint);
        for (const auto& part : stage_parts) n += part.bytes();
        return n;
    }
};

// Identity-layout K/V pages and completed indexer rows. For streamed layers the
// source is the authoritative host pool, NOT the replaceable VRAM slots.
struct ConversationKv {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    ConversationBuffer k, v, k_scale, v_scale, pooled;
    size_t bytes() const {
        return k.bytes() + v.bytes() + k_scale.bytes() + v_scale.bytes() + pooled.bytes();
    }
};

struct ConversationKvReuse {
    std::vector<ConversationKv> kv;
    // Original image extent for validation, and the earliest subsequent rewrite.
    int64_t captured_tokens = 0, unchanged_tokens = 0;
    // With a layer split: the later stages' own retained K/V, one per stage, same extents (empty: none)
    std::vector<ConversationKvReuse> stages;
    size_t bytes() const {
        size_t n = kv.capacity() * sizeof(ConversationKv) + stages.capacity() * sizeof(ConversationKvReuse);
        for (const auto& layer : kv) n += layer.bytes();
        for (const auto& s : stages) n += s.bytes();
        return n;
    }
};

/// Layer-split parking: each running checkpoint holds the first stage's state and, in `stage_parts`, one part per
/// later stage; a parked conversation keeps them in one image per stage.  `split` MOVES the checkpoints apart (no
/// running state is copied): the complete ones (a part for each of `stages` stages) into `stage0` and `parts[k]`,
/// every part carrying the checkpoint's ids / images / LRU stamp, the others aside; `merge` puts them back in their
/// original order.  merge(split(x)) == x.
struct ConversationCheckpointSplit {
    std::vector<ConversationCheckpoint> stage0;
    std::vector<std::vector<ConversationCheckpoint>> parts;   ///< [stage][checkpoint]
    std::vector<ConversationCheckpoint> rest;                 ///< not split (no complete set of stage parts)
    std::vector<bool> complete;                               ///< per original position: in stage0/parts or in rest
};

inline ConversationCheckpointSplit conversation_checkpoints_split(std::vector<ConversationCheckpoint>&& checks,
                                                                  size_t stages) {
    ConversationCheckpointSplit out;
    out.parts.resize(stages);
    out.complete.reserve(checks.size());
    size_t whole = 0;
    for (const auto& c : checks) {
        const bool complete = c.stage_parts.size() == stages;
        out.complete.push_back(complete);
        whole += complete ? 1 : 0;
    }
    out.stage0.reserve(whole);
    out.rest.reserve(checks.size() - whole);
    for (auto& parts : out.parts) parts.reserve(whole);
    // Allocate and copy identity metadata before moving any running state.
    // If allocation fails, every input checkpoint remains usable by the caller.
    for (const auto& c : checks) {
        if (c.stage_parts.size() != stages) continue;
        for (size_t k = 0; k < stages; ++k) {
            ConversationCheckpoint part;
            part.ids = c.ids; part.imgs = c.imgs; part.used = c.used; part.pinned = c.pinned;
            out.parts[k].push_back(std::move(part));
        }
    }
    size_t w = 0;
    for (auto& c : checks) {
        if (c.stage_parts.size() != stages) { out.rest.push_back(std::move(c)); continue; }
        for (size_t k = 0; k < stages; ++k) {
            auto& dst = out.parts[k][w];
            auto& src = c.stage_parts[k];
            dst.gdn = std::move(src.gdn); dst.ple = std::move(src.ple);
            dst.tails = std::move(src.tails); dst.dead = std::move(src.dead);
            dst.block_pos = std::move(src.block_pos);
        }
        c.stage_parts.clear();
        out.stage0.push_back(std::move(c));
        ++w;
    }
    checks.clear();
    return out;
}

/// false (and `out` empty) when the parts do not line up: a stage with another number of checkpoints, or a part
/// whose ids differ from its checkpoint's.
inline bool conversation_checkpoints_merge(ConversationCheckpointSplit&& split, std::vector<ConversationCheckpoint>& out) {
    out.clear();
    for (const auto& p : split.parts)
        if (p.size() != split.stage0.size()) return false;
    size_t whole = 0, other = 0;
    for (bool c : split.complete) (c ? whole : other) += 1;
    if (!split.complete.empty() && (whole != split.stage0.size() || other != split.rest.size())) return false;
    if (split.complete.empty()) split.complete.assign(split.stage0.size(), true);   // a restored image: all whole
    for (size_t j = 0; j < split.stage0.size(); ++j)
        for (const auto& p : split.parts)
            if (p[j].ids != split.stage0[j].ids) return false;
    size_t w = 0, r = 0;
    for (bool c : split.complete) {
        if (!c) { out.push_back(std::move(split.rest[r++])); continue; }
        ConversationCheckpoint cp = std::move(split.stage0[w]);
        for (auto& p : split.parts) cp.stage_parts.push_back(std::move(p[w]));
        ++w;
        out.push_back(std::move(cp));
    }
    return true;
}

struct SavedConversation {
    // Runtime compatibility only; NOT a model/weights identity or disk schema.
    std::array<int64_t, 18> geometry{};
    // The session's layer carve the image was captured from ([0, n_layers) on one GPU); restore requires the same.
    int64_t layer_lo = 0, layer_hi = 0;
    ConversationCheckpoint live;
    std::vector<ConversationCheckpoint> checkpoints;
    std::vector<ConversationKv> kv; // main layers followed by the draft layer
    bool cvec = true;
    // with a layer split, the later stages' own images, one per stage, in stage order
    std::vector<SavedConversation> stage_images;

    /// Holds a pinned shared prefix (see ConversationCheckpoint::pinned): the parked-conversation budget keeps it.
    bool pinned() const {
        for (const auto& c : checkpoints) if (c.pinned) return true;
        return false;
    }

    size_t bytes() const {
        size_t n = live.bytes() + checkpoints.capacity() * sizeof(ConversationCheckpoint) +
                   kv.capacity() * sizeof(ConversationKv);
        for (const auto& s : stage_images) n += s.bytes();
        for (const auto& c : checkpoints) n += c.bytes();
        for (const auto& k : kv) n += k.bytes();
        return n;
    }
};

template<class Token>
int64_t conversation_prefix(const ConversationCheckpoint& c, const std::vector<Token>& prompt,
                            const std::vector<ConversationImageKey>& images) {
    const size_t n = c.ids.size();
    // The last prompt token always starts the next verify window.
    if (n == 0 || n >= prompt.size() || !std::equal(c.ids.begin(), c.ids.end(), prompt.begin())) return 0;
    size_t j = 0;
    for (const auto& image : images) {
        if (image.start >= (int64_t) n) continue;
        if (j == c.imgs.size() || !(c.imgs[j++] == image)) return 0;
    }
    if (j != c.imgs.size()) return 0;
    return (int64_t) n;
}

/// --conversation-cache-dir: what a parked conversation keeps in RAM while its image is in a file - the identity that
/// matching (best, drop_superseded, longest_tokens, pinned) reads: the token and image lists, the LRU stamps, the pins.
/// The running state and K/V are gone.
inline SavedConversation conversation_skeleton(SavedConversation&& image) {
    SavedConversation out;
    out.geometry = image.geometry;
    out.layer_lo = image.layer_lo;
    out.layer_hi = image.layer_hi;
    out.cvec = image.cvec;
    auto ident = [](ConversationCheckpoint&& c) {
        ConversationCheckpoint k;
        k.ids = std::move(c.ids); k.imgs = std::move(c.imgs); k.used = c.used; k.pinned = c.pinned;
        return k;
    };
    out.live = ident(std::move(image.live));
    out.checkpoints.reserve(image.checkpoints.size());
    for (auto& c : image.checkpoints) out.checkpoints.push_back(ident(std::move(c)));
    image = {};
    return out;
}
/// The same for an image whose checkpoints were written from where they are (meta.checkpoints empty): their
/// identities are copied from `checkpoints`, which stay as they are.
inline SavedConversation conversation_skeleton(SavedConversation&& meta, const std::vector<ConversationCheckpoint>& checkpoints) {
    SavedConversation out = conversation_skeleton(std::move(meta));
    out.checkpoints.reserve(checkpoints.size());
    for (const auto& c : checkpoints) {
        ConversationCheckpoint k;
        k.ids = c.ids; k.imgs = c.imgs; k.used = c.used; k.pinned = c.pinned;
        out.checkpoints.push_back(std::move(k));
    }
    return out;
}

/// The parked conversations' file tier: the engine writes a file itself (put_file) and supplies how one is read back
/// and removed.  `load` fills `out` from file `file` or says why not; `remove` deletes it (a missing file is fine).
struct ConversationDisk {
    std::function<bool(uint64_t file, SavedConversation& out, std::string& error)> load;
    std::function<void(uint64_t file)> remove;
};

class ConversationCache {
public:
    struct Match {
        size_t index = 0;
        int64_t tokens = 0;
        bool live = false;
        uint64_t serial = 0;   // the entry's identity; its index shifts when entries before it go
    };

    ConversationCache(size_t budget, size_t slots, int64_t min_tokens = 0)
        : budget_(budget), slots_(slots), min_tokens_(min_tokens) {}
    // The files outlive the cache: a restart finds them again (the engine reads their skeletons back at start).
    // clear() and eviction remove files; going away does not.
    ConversationCache(const ConversationCache&) = delete;
    ConversationCache& operator=(const ConversationCache&) = delete;

    /// --conversation-cache-dir: parked images may live in files (put_file); the budget then bounds the files' bytes.
    void use_disk(ConversationDisk disk) { disk_ = std::move(disk); }
    bool on_disk() const { return (bool) disk_.load; }
    /// Whether evicting a parked conversation gives RAM back.  With the file tier it gives back disk space only, so
    /// the RAM admission paths must not empty the cache for nothing.
    bool frees_ram() const { return !on_disk(); }
    /// Why the last take() of a file entry came back empty.
    const std::string& last_error() const { return last_error_; }
    bool enabled() const { return budget_ != 0 && slots_ != 0; }
    /// `--conversation-cache-min-tokens`: whether a conversation of this length is worth one of `slots`.  Every
    /// parked conversation costs a whole slot whatever its length, so a client that interleaves short side requests
    /// with one long conversation spends the cache on the side requests and evicts the long one (oldest first) -
    /// exactly the reuse it wanted.  0 parks every conversation, whatever its size.
    bool wants(int64_t tokens) const { return min_tokens_ <= 0 || tokens >= min_tokens_; }
    size_t bytes() const { return bytes_ + reuse_.bytes() + base_bytes(); }
    size_t size() const { return entries_.size(); }
    size_t evictions() const { return evictions_; }
    // the longest parked conversation, in tokens (--kv-grow keeps the K/V that long while it could be restored)
    int64_t longest_tokens() const {
        int64_t n = 0;
        for (const auto& e : entries_) n = std::max<int64_t>(n, (int64_t) e.image.live.ids.size());
        return n;
    }

    // Retain only the restored K/V buffers, not duplicate running checkpoints.
    // This optimization never evicts a parked conversation to make itself fit.
    // `stage_kv`: with a layer split, the later stages' restored K/V (one per stage), retained with the first's.
    void retain(std::vector<ConversationKv>&& kv, int64_t tokens,
                std::vector<std::vector<ConversationKv>>&& stage_kv = {}) {
        reuse_ = {};
        // the file tier keeps no K/V in RAM: the next park streams it from the device again
        if (on_disk()) return;
        ConversationKvReuse candidate{std::move(kv), tokens, tokens, {}};
        for (auto& k : stage_kv) candidate.stages.push_back(ConversationKvReuse{std::move(k), tokens, tokens, {}});
        if (enabled() && candidate.bytes() <= budget_ - bytes_) reuse_ = std::move(candidate);
    }
    void limit_reuse(int64_t first_dirty) {
        if (base_) base_->unchanged = std::min(base_->unchanged, first_dirty);
        reuse_.unchanged_tokens = std::min(reuse_.unchanged_tokens, first_dirty);
        for (auto& s : reuse_.stages) s.unchanged_tokens = std::min(s.unchanged_tokens, first_dirty);
        if (reuse_.unchanged_tokens <= 0) reuse_ = {};
    }
    ConversationKvReuse take_reuse() { return std::exchange(reuse_, {}); }
    /// --conversation-cache-dir: the file the session's conversation was read back from, kept so that parking it again
    /// updates that file in place (conversation_park.hpp) instead of writing a new one.  `unchanged`: how many leading
    /// tokens of the session's K/V are still the file's (limit_reuse bounds it at each request's first written
    /// token); `bytes`: what the file holds, charged to the budget.  The cache owns the file until take_base() hands
    /// it on; drop_base() removes it, as must the caller whenever the session takes other state in.
    struct Base {
        uint64_t file = 0;
        int64_t unchanged = 0;
        size_t bytes = 0;
    };
    void retain_file(uint64_t file, int64_t tokens, size_t bytes) {
        drop_base();
        base_ = Base{file, tokens, bytes};
    }
    const std::optional<Base>& base() const { return base_; }
    std::optional<Base> take_base() { return std::exchange(base_, std::nullopt); }
    void drop_base() {
        if (base_ && disk_.remove) disk_.remove(base_->file);
        base_.reset();
    }
    size_t retained_bytes() const { return reuse_.bytes(); }
    bool can_fit(size_t incoming, size_t held = 0) const {
        return enabled() && held <= budget_ && incoming <= budget_ - held &&
               entries_.size() < slots_ && bytes() <= budget_ - held - incoming;
    }

    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        Match best;
        // Ties prefer the most recently parked branch. The caller prefers its
        // already-active state when that offers the same prefix length.
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i].image;
            if (e.cvec != cvec) continue;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > best.tokens) best = {i, n, live, serials_[i]};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        }
        return best;
    }

    /// A parked conversation taken out of the cache but not read back yet: its skeleton and its file (0: `image` is
    /// the whole image).  The engine parks the outgoing conversation before it reads a file back, so the two images
    /// are never in RAM together.
    struct Detached {
        SavedConversation image;
        uint64_t file = 0;
    };
    Detached detach(size_t index) {
        Entry e = std::move(entries_.at(index));
        bytes_ -= e.bytes;
        erase(index);
        return {std::move(e.image), e.file};
    }
    /// The whole image of a detached entry; a file is read back and removed.  When that fails the result is empty -
    /// no K/V, which every caller's validation refuses - and last_error() says why.
    SavedConversation load(Detached&& d) {
        if (d.file == 0) return std::move(d.image);
        SavedConversation out;
        if (!load_with(std::move(d), out, disk_.load)) out = {};
        return out;
    }
    /// A detached file entry read back by `read(file, out, why)` instead of disk.load - a read that may put the K/V
    /// elsewhere (a streamed restore) - and removed, its pins carried over as load() does.  False with last_error().
    /// `keep_file`: a file read back is not removed - the caller hands it to retain_file() (a failed read removes it).
    template<class Read> bool load_with(Detached&& d, SavedConversation& out, Read&& read, bool keep_file = false) {
        last_error_.clear();
        const bool ok = read(d.file, out, last_error_);
        if (!ok) {
            if (last_error_.empty()) last_error_ = "the parked conversation's file cannot be read";
        } else if (out.checkpoints.size() == d.image.checkpoints.size()) {
            for (size_t i = 0; i < out.checkpoints.size(); ++i) out.checkpoints[i].pinned = d.image.checkpoints[i].pinned;
        }
        if (!ok || !keep_file) disk_.remove(d.file);
        return ok;
    }
    SavedConversation take(size_t index) { return load(detach(index)); }

    // A checkpoint inside a parked conversation that a NEW conversation starts with - a sibling subagent sharing the
    // system prompt and tool list, a compacted history - is borrowed: restored from the entry, which stays parked.
    // take() would move the whole conversation into the session and the new request would overwrite everything after
    // that checkpoint, so the other conversation's next turn would read its history again.  A match on the entry's
    // live state or on its deepest checkpoint is the same conversation going on (the deepest one: its client
    // re-rendered the last reply), whose tail is stale: that is still taken.  So is a match that reaches a pinned
    // shared prefix (pin=N): a borrow copies the checkpoints up to the match, and taking it instead keeps one
    // conversation holding the pin, as pin=N expects.
    bool borrows(const Match& m) const {
        if (m.tokens <= 0 || m.live || m.index >= entries_.size() || serials_[m.index] != m.serial) return false;
        size_t deepest = 0;
        for (const auto& c : entries_[m.index].image.checkpoints) {
            if (c.pinned && (int64_t) c.ids.size() <= m.tokens) return false;
            deepest = std::max(deepest, c.ids.size());
        }
        return (size_t) m.tokens < deepest;
    }
    // The entry with this identity: its index (SIZE_MAX: gone), or the entry itself (nullptr: gone; the pointer is
    // valid until the cache changes).
    size_t index_of(uint64_t serial) const {
        for (size_t i = 0; i < entries_.size(); ++i)
            if (serials_[i] == serial) return i;
        return SIZE_MAX;
    }
    /// A file entry (file_of() != 0) is only its skeleton here: borrow from it with borrow_with().
    const SavedConversation* find(uint64_t serial) const {
        const size_t i = index_of(serial);
        return i == SIZE_MAX ? nullptr : &entries_[i].image;
    }
    /// --conversation-cache-dir: the file the entry's image is in; 0 when it is in RAM or gone.
    uint64_t file_of(uint64_t serial) const {
        const size_t i = index_of(serial);
        return i == SIZE_MAX ? 0 : entries_[i].file;
    }
    /// A borrow from a file entry: `read(file, out, why)` reads the file back (a streamed read puts the K/V wherever
    /// its target does) and the entry stays parked with its file, its pins carried over as load_with() does.  False
    /// with last_error() when the entry is gone or the read failed; the entry then stays too - drop() it unless the
    /// failure says nothing about the file (no RAM to read it now).
    template<class Read> bool borrow_with(uint64_t serial, SavedConversation& out, Read&& read) {
        last_error_.clear();
        const size_t i = index_of(serial);
        if (i == SIZE_MAX || entries_[i].file == 0) {
            last_error_ = "the parked conversation is no longer in a file";
            return false;
        }
        if (!read(entries_[i].file, out, last_error_)) {
            if (last_error_.empty()) last_error_ = "the parked conversation's file cannot be read";
            return false;
        }
        const auto& skel = entries_[index_of(serial)].image;
        if (out.checkpoints.size() == skel.checkpoints.size())
            for (size_t k = 0; k < out.checkpoints.size(); ++k) out.checkpoints[k].pinned = skel.checkpoints[k].pinned;
        return true;
    }
    /// The entry with this identity goes, and its file with it (an unreadable file a borrow met).
    void drop(uint64_t serial) {
        const size_t i = index_of(serial);
        if (i == SIZE_MAX) return;
        bytes_ -= entries_[i].bytes;
        release(entries_[i]);
        erase(i);
    }
    // While a borrowed entry (the donor) is being restored, parking the outgoing conversation must not evict or drop
    // it: guard() it.  When the outgoing image fits only without it, make_room refuses before evicting anything and
    // guard_blocked() says so: the caller then takes the entry as before, so the conversation that was active a
    // moment ago keeps its place.
    void guard(uint64_t serial) { guarded_ = serial; guard_blocked_ = false; }
    void unguard() { guarded_ = 0; }
    bool guard_blocked() const { return guard_blocked_; }
    size_t borrowed() const { return borrowed_; }
    void count_borrow() { ++borrowed_; }

    // Reserve before allocating a snapshot. held is an incoming image removed
    // with take() but still alive during the exchange; count it against RAM too.
    bool make_room(size_t incoming, size_t held = 0) {
        if (!enabled() || held > budget_ || incoming > budget_ - held) return false;
        auto stays = [&](size_t i) { return entries_[i].image.pinned() || (guarded_ != 0 && serials_[i] == guarded_); };
        if (guarded_ != 0 && index_of(guarded_) != SIZE_MAX) {
            // with everything else gone, would it fit beside the entries that stay (the guarded donor and any pinned
            // shared prefix)?  If not, evict nothing
            size_t kept = 0, kept_bytes = 0;
            for (size_t i = 0; i < entries_.size(); ++i)
                if (stays(i)) { ++kept; kept_bytes += entries_[i].bytes; }
            if (kept >= slots_ || kept_bytes > budget_ - held - incoming) { guard_blocked_ = true; return false; }
        }
        if (bytes() > budget_ - held - incoming) reuse_ = {};
        while (!entries_.empty() && (entries_.size() >= slots_ || bytes_ + base_bytes() > budget_ - held - incoming)) {
            // the oldest entry that does not hold a pinned shared prefix leaves; with only pinned ones left the new
            // image does not fit (the caller skips parking it - the pinned prefix is what the queries come back to)
            if (!evict_oldest()) return false;
        }
        return true;
    }

    // The parked conversation that has gone unused the longest.  This is make_room()'s loop body, so
    // the parking path can also free RAM one entry at a time on demand (see the physical-RAM admission
    // gate in generate.cpp).  A ConversationBuffer is a list of 16 MiB segments and each segment is its
    // own allocation, far above glibc's mmap threshold, so dropping an entry returns the whole footprint
    // to the kernel at once - the next admission check reads it back from /proc/meminfo.
    // False when none can go (empty, or only entries that hold a pinned shared prefix are left).
    bool evict_oldest() {
        // the oldest entry that holds no pinned shared prefix and is not the guarded (borrowed) donor
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].image.pinned() || (guarded_ != 0 && serials_[i] == guarded_)) continue;
            bytes_ -= entries_[i].bytes;
            release(entries_[i]);
            erase(i);
            ++evictions_;
            return true;
        }
        return false;
    }

    // #879: every parked conversation (pinned ones too) and the retained K/V go - after non-finite logits any of them
    // may carry the poison.  Returns how many were parked.
    size_t clear() {
        const size_t n = entries_.size();
        for (auto& e : entries_) release(e);
        drop_base();
        entries_.clear();
        serials_.clear();
        bytes_ = 0;
        reuse_ = {};
        return n;
    }

    // The slot count, so a caller that evicts in a loop has a bound it did not invent.
    size_t slots() const { return slots_; }

    // #342: drop the parked entries an outgoing conversation (its live tokens and checkpoint chain) supersedes:
    // the same conversation a turn back, whose DEEPEST checkpoint the outgoing chain still holds, so all it adds
    // is the tail the client rewrote (the reply as it was generated, before the next request re-rendered it) and
    // checkpoints older than that one.  A subagent's successive turns parked one such copy each, and make_room's
    // oldest-first eviction then pushed the parent conversation out after `slots` turns.  An entry without
    // checkpoints, or whose deepest checkpoint the outgoing chain does not hold (another conversation that only
    // shares the system prompt's root with it), is kept.  Returns how many were dropped.
    size_t drop_superseded(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) {
        auto held = [&](const ConversationCheckpoint& c) {
            if (c.ids == ids && c.imgs == images) return true;
            for (const auto& k : checkpoints)
                if (k.ids == c.ids && k.imgs == c.imgs) return true;
            return false;
        };
        size_t dropped = 0;
        for (size_t i = 0; i < entries_.size();) {
            const auto& e = entries_[i].image;
            const ConversationCheckpoint* deepest = nullptr;
            for (const auto& c : e.checkpoints)
                if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
            if (e.cvec == cvec && deepest && !deepest->ids.empty() && held(*deepest) &&
                !(guarded_ != 0 && serials_[i] == guarded_)) {
                bytes_ -= entries_[i].bytes;
                release(entries_[i]);
                erase(i);
                ++dropped;
                continue;
            }
            ++i;
        }
        superseded_ += dropped;
        return dropped;
    }
    size_t superseded() const { return superseded_; }
    /// --conversation-cache-park-gain N: the longest parked entry this conversation's outgoing state supersedes, in
    /// tokens, 0 when it has none.  It is the entry drop_superseded() would remove, so the caller can tell whether a
    /// park now would only rewrite what a turn from now drops: the prefix such an entry holds is this conversation's
    /// prefix, and the conversation only appends, so the copy stays usable for the reuse it was made for.
    size_t parked_prefix_tokens(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                                const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) const {
        size_t longest = 0;
        for (const auto& entry : entries_) {
            const auto& e = entry.image;
            if (e.cvec != cvec) continue;
            const ConversationCheckpoint* deepest = nullptr;
            for (const auto& c : e.checkpoints)
                if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
            if (!deepest || deepest->ids.empty()) continue;
            bool holds = deepest->ids == ids && deepest->imgs == images;
            for (const auto& k : checkpoints)
                if (!holds && k.ids == deepest->ids && k.imgs == deepest->imgs) holds = true;
            if (holds) longest = std::max(longest, deepest->ids.size());
        }
        return longest;
    }

    bool put(SavedConversation&& image, size_t held = 0) {
        const size_t n = image.bytes();
        if (!enabled() || held > budget_ || n > budget_ - held) return false;   // make_room's refusal, first
        drop_superseded(image.live.ids, image.live.imgs, image.checkpoints, image.cvec);
        if (!make_room(n, held)) return false;
        entries_.push_back(Entry{std::move(image), 0, n});
        serials_.push_back(next_serial_++);
        bytes_ += n;
        return true;
    }

    /// The file tier: the engine wrote the image to file `file` (`bytes` on disk); the cache keeps `skeleton`
    /// (conversation_skeleton) and owns the file from here on, so a refusal removes it.
    bool put_file(SavedConversation&& skeleton, uint64_t file, size_t bytes, size_t held = 0) {
        if (!on_disk() || file == 0 || !enabled() || held > budget_ || bytes > budget_ - held) {
            if (on_disk() && file != 0) disk_.remove(file);
            return false;
        }
        drop_superseded(skeleton.live.ids, skeleton.live.imgs, skeleton.checkpoints, skeleton.cvec);
        if (!make_room(bytes, held)) { disk_.remove(file); return false; }
        entries_.push_back(Entry{std::move(skeleton), file, bytes});
        serials_.push_back(next_serial_++);
        bytes_ += bytes;
        return true;
    }

private:
    // file == 0: `image` is the whole parked image in RAM; otherwise its skeleton, the image is in that file.
    // `bytes`: what the entry charges the budget (the image's RAM, or its file's size)
    struct Entry {
        SavedConversation image;
        uint64_t file = 0;
        size_t bytes = 0;
    };
    void release(Entry& e) {
        if (e.file != 0 && disk_.remove) disk_.remove(e.file);
        e.file = 0;
    }

    // removes the entry only: the caller has released its file, or handed it on (detach)
    void erase(size_t index) {
        entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
        serials_.erase(serials_.begin() + (std::ptrdiff_t) index);
    }
    size_t budget_ = 0, slots_ = 0, bytes_ = 0, evictions_ = 0, superseded_ = 0, borrowed_ = 0;
    int64_t min_tokens_ = 0;   // --conversation-cache-min-tokens: 0 parks every conversation, whatever its size
    std::deque<Entry> entries_; // least recently active first
    std::deque<uint64_t> serials_;          // entries_[i]'s identity, never reused
    uint64_t next_serial_ = 1, guarded_ = 0;
    bool guard_blocked_ = false;
    size_t base_bytes() const { return base_ ? base_->bytes : 0; }
    ConversationKvReuse reuse_;
    std::optional<Base> base_;
    ConversationDisk disk_;
    std::string last_error_;
};

} // namespace strata::core
