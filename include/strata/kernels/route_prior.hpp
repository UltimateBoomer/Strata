// include/strata/kernels/route_prior.hpp - STRATA_ROUTE_PRIOR=lambda (opt-in, changes the output): cache-conditioned
// routing in verify windows.  See src/kernels/cuda/route_prior.cu.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Top-10 per token over `logits` (n_tok x 512) where a resident expert (`res_layer[e] >= 0`, the layer's residency
/// table) gets `lambda` added to its logit for the SELECTION; the weights are the router's own probabilities
/// (softmax over the 512 unbiased logits) renormalised over the 10 picked, as the native router does.
void route_prior_top10(const float* logits, const int32_t* res_layer, float lambda, int32_t* ids, float* weights,
                       int n_tok, void* stream);

/// STRATA_ROUTE_LIVE=1: the routing biases a request can set (serve: route_prior=, route_resident=,
/// route_resident_ranks=), read by the verify kernels from mapped host memory, so the captured window graphs hold the
/// kernels once and every launch reads the values of the request it runs for.  A kernel whose value is off (<= 0)
/// returns before it writes anything: the same bits as a graph without it.
struct LiveRoute {
    float prior = 0.0f;       ///< route_prior's lambda (0: off)
    float rr_margin = 0.0f;   ///< route_resident's logit margin (0: off)
    int32_t rr_lo = 6, rr_hi = 9;   ///< route_resident's ranks
};

/// The values the requests change (host side of the mapped block), or nullptr: STRATA_ROUTE_LIVE is not 1, or the
/// block could not be allocated.  The first call allocates it (outside graph capture: the Verifier's init) and fills
/// it with live_route_startup().
LiveRoute* live_route_host();
/// The device address of the same block (nullptr when live_route_host() is).
const LiveRoute* live_route_device();
/// What a request that names none of the keys runs with: STRATA_ROUTE_PRIOR, STRATA_ROUTE_RESIDENT(_RANKS).
LiveRoute& live_route_startup();

/// route_prior_top10 with lambda read from `live->prior` at run time; nothing is written when it is <= 0.
void route_prior_top10_live(const float* logits, const int32_t* res_layer, const LiveRoute* live, int32_t* ids,
                            float* weights, int n_tok, void* stream);

}  // namespace strata::kernels
