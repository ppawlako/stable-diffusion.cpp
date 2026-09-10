#include "core/vram_allocator.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <vector>

#include "core/util.h"

namespace sd {

namespace {

// Compute (activation) buffer reserved on GPUs that host no diffusion layers
// (VAE / text-encoder-only GPUs). 2 GiB covers VAE encode/decode workspaces
// (e.g. MiniMax-H3 keyframe encode needs ~1 GiB) on top of the text encoder's
// workspace sharing the GPU. Diffusion GPUs get whatever remains after all
// model placements, equalized across all of them, so every GPU that hosts
// model weights ends up with a usable compute buffer and each GPU's budgets
// (diffusion + text + vae + compute) sum exactly to its free VRAM.
constexpr size_t NON_DIFFUSION_COMPUTE_RESERVE_BYTES = 2ull * 1024 * 1024 * 1024;

bool is_diffusion_gpu(const std::vector<int>& diffusion_gpus, int gpu) {
    return std::find(diffusion_gpus.begin(), diffusion_gpus.end(), gpu) != diffusion_gpus.end();
}

bool is_text_gpu(const std::vector<int>& text_gpus, int gpu) {
    return std::find(text_gpus.begin(), text_gpus.end(), gpu) != text_gpus.end();
}

}  // namespace

VRAMBudget calculate_vram_budgets(int n_gpus,
                                  const std::vector<size_t>& vram_per_gpu,
                                  size_t diffusion_size,
                                  size_t text_size,
                                  size_t vae_size,
                                  const std::vector<int>& diffusion_gpus,
                                  const std::vector<int>& text_gpus,
                                  int vae_gpu) {
    VRAMBudget budget;
    budget.diffusion.assign(n_gpus, 0);
    budget.text.assign(n_gpus, 0);
    budget.vae.assign(n_gpus, 0);
    budget.compute.assign(n_gpus, 0);

    // Free VRAM per GPU; everything not assigned to model weights below
    // becomes the compute buffer.
    std::vector<size_t> available(n_gpus, 0);
    for (int i = 0; i < n_gpus; i++) {
        available[i] = i < (int)vram_per_gpu.size() ? vram_per_gpu[i] : 0;
    }

    // 1. VAE (video + audio VAEs share the VAE backend GPU).
    if (vae_size > 0 && vae_gpu >= 0 && vae_gpu < n_gpus) {
        budget.vae[vae_gpu] = std::min(vae_size, available[vae_gpu]);
        available[vae_gpu] -= budget.vae[vae_gpu];
        if (budget.vae[vae_gpu] < vae_size) {
            LOG_WARN("VRAM budget: VAE does not fit on GPU %d (%zu MB of %zu MB placed); check --backend assignment",
                     vae_gpu,
                     budget.vae[vae_gpu] / (1024ull * 1024ull),
                     vae_size / (1024ull * 1024ull));
        }
    }

    // 2. Text encoder.
    //    Pass 1: non-diffusion text GPUs (the VAE GPU first, so the bulk of
    //            the text model lands next to the VAE). Each such GPU keeps
    //            the 512 MB compute reserve.
    //    Pass 2: leftover text spills onto diffusion text GPUs; their compute
    //            buffer is derived in step 3, so no fixed reserve is taken.
    size_t remaining_text = text_size;
    auto place_text = [&](int gpu_idx) {
        if (remaining_text == 0 || gpu_idx < 0 || gpu_idx >= n_gpus) {
            return;
        }
        const size_t reserve = is_diffusion_gpu(diffusion_gpus, gpu_idx) ? 0 : NON_DIFFUSION_COMPUTE_RESERVE_BYTES;
        const size_t room    = available[gpu_idx] > reserve ? available[gpu_idx] - reserve : 0;
        const size_t placed  = std::min(remaining_text, room);
        budget.text[gpu_idx] += placed;
        remaining_text       -= placed;
        available[gpu_idx]   -= placed;
    };

    if (vae_gpu >= 0 && vae_gpu < n_gpus &&
        is_text_gpu(text_gpus, vae_gpu) &&
        !is_diffusion_gpu(diffusion_gpus, vae_gpu)) {
        place_text(vae_gpu);
    }
    for (int tg : text_gpus) {
        if (tg == vae_gpu || is_diffusion_gpu(diffusion_gpus, tg)) {
            continue;
        }
        place_text(tg);
    }
    for (int tg : text_gpus) {
        if (is_diffusion_gpu(diffusion_gpus, tg)) {
            place_text(tg);
        }
    }
    if (remaining_text > 0) {
        LOG_WARN("VRAM budget: text encoder does not fully fit in VRAM; %zu MB unplaced",
                 remaining_text / (1024ull * 1024ull));
    }

    // 3. Diffusion model across its GPUs, split so that the compute buffer
    //    (free VRAM left after all weights) is equal on every diffusion GPU.
    //    That is the largest equal reserve that still fits all weights:
    //    r = (sum(available on diffusion GPUs) - diffusion_size) / n.
    if (!diffusion_gpus.empty() && diffusion_size > 0) {
        const int n_diff = (int)diffusion_gpus.size();
        uint64_t headroom_sum = 0;
        for (int dg : diffusion_gpus) {
            if (dg >= 0 && dg < n_gpus) {
                headroom_sum += available[dg];
            }
        }
        const uint64_t compute_per_diffusion_gpu =
            headroom_sum >= (uint64_t)diffusion_size ? (headroom_sum - (uint64_t)diffusion_size) / (uint64_t)n_diff : 0;
        if (compute_per_diffusion_gpu == 0 && headroom_sum < (uint64_t)diffusion_size) {
            LOG_WARN("VRAM budget: diffusion model does not fit in VRAM on its GPUs; "
                     "%zu MB unplaced (diffusion=%zu MB, headroom=%" PRIu64 " MB)",
                     ((uint64_t)diffusion_size - headroom_sum) / (1024ull * 1024ull),
                     diffusion_size / (1024ull * 1024ull),
                     headroom_sum / (1024ull * 1024ull));
        }
        size_t placed_diffusion = 0;
        for (int dg : diffusion_gpus) {
            if (dg < 0 || dg >= n_gpus) {
                continue;
            }
            const size_t room = available[dg];
            const size_t load = room > compute_per_diffusion_gpu ? room - (size_t)compute_per_diffusion_gpu : 0;
            budget.diffusion[dg] = load;
            available[dg] -= load;
            placed_diffusion += load;
        }
        if (placed_diffusion < diffusion_size) {
            LOG_WARN("VRAM budget: only %zu MB of %zu MB diffusion weights fit on the diffusion GPUs",
                     placed_diffusion / (1024ull * 1024ull),
                     diffusion_size / (1024ull * 1024ull));
        }
    }

    // 4. Compute buffer = whatever remains; exact by construction, so
    //    diffusion + text + vae + compute == free VRAM on every GPU.
    for (int i = 0; i < n_gpus; i++) {
        budget.compute[i] = available[i];
    }

    return budget;
}

}  // namespace sd
