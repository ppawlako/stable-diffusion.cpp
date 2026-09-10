#ifndef __VRAM_ALLOCATOR_H__
#define __VRAM_ALLOCATOR_H__

#include <cstdint>
#include <vector>

namespace sd {

// VRAM budget per GPU for each model type
struct VRAMBudget {
    std::vector<size_t> diffusion;
    std::vector<size_t> text;
    std::vector<size_t> vae;
    std::vector<size_t> compute;  // Compute buffer budget
};

// Calculate VRAM budgets for all GPUs
// Returns budget per GPU for each model type
VRAMBudget calculate_vram_budgets(
    int n_gpus,
    const std::vector<size_t>& vram_per_gpu,
    size_t diffusion_size,
    size_t text_size,
    size_t vae_size,
    const std::vector<int>& diffusion_gpus,
    const std::vector<int>& text_gpus,
    int vae_gpu
);

}  // namespace sd

#endif  // __VRAM_ALLOCATOR_H__
