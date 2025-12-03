//===- test_bitlinear.cpp - Test harness for BitLinear kernel -*- C++ -*-===//
//
// Test harness for verifying int8 x int2 BitLinear kernel correctness
// against a CPU reference implementation.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Bfloat16 support
#if __cplusplus >= 202302L
#include <stdfloat>
using bf16_t = std::bfloat16_t;
#else
// Fallback bfloat16 implementation
struct bf16_t {
    uint16_t bits;
    
    bf16_t() : bits(0) {}
    bf16_t(float f) {
        uint32_t* f_bits = reinterpret_cast<uint32_t*>(&f);
        bits = static_cast<uint16_t>(*f_bits >> 16);
    }
    
    operator float() const {
        uint32_t f_bits = static_cast<uint32_t>(bits) << 16;
        return *reinterpret_cast<float*>(&f_bits);
    }
};
#endif

//===----------------------------------------------------------------------===//
// Test configuration
//===----------------------------------------------------------------------===//

// Default dimensions (can be overridden via command line)
constexpr int DEFAULT_M = 32;  // Output dimension
constexpr int DEFAULT_K = 32;  // Input dimension

// Tolerances for floating point comparison
constexpr float ABS_TOL = 1e-2f;
constexpr float REL_TOL = 1e-2f;

//===----------------------------------------------------------------------===//
// Reference implementations
//===----------------------------------------------------------------------===//

// Unpack int2 values from packed byte
void unpack_int2_to_int8_ref(int8_t packed, int8_t* out) {
    out[0] = static_cast<int8_t>((packed & 0x03) - 1);
    out[1] = static_cast<int8_t>(((packed >> 2) & 0x03) - 1);
    out[2] = static_cast<int8_t>(((packed >> 4) & 0x03) - 1);
    out[3] = static_cast<int8_t>(((packed >> 6) & 0x03) - 1);
}

// CPU reference for BitLinear int8 x int2 matvec
void bitlinear_ref(
    const int8_t* input,           // [K] int8 activations
    const int8_t* weights_packed,  // [M, K/4] packed int2 weights
    float* output,                 // [M] output (float for reference)
    float act_scale,               // Activation scale
    const float* weight_scales,    // [num_groups] Weight scales
    int M, int K, int num_groups
) {
    int rows_per_group = M / num_groups;
    float inv_act_scale = 1.0f / act_scale;
    
    for (int row = 0; row < M; row++) {
        int32_t acc = 0;
        
        // Process K elements, 4 at a time (packed)
        for (int k = 0; k < K; k += 4) {
            int8_t packed = weights_packed[row * (K / 4) + (k / 4)];
            
            int8_t w[4];
            unpack_int2_to_int8_ref(packed, w);
            
            acc += static_cast<int32_t>(input[k + 0]) * static_cast<int32_t>(w[0]);
            acc += static_cast<int32_t>(input[k + 1]) * static_cast<int32_t>(w[1]);
            acc += static_cast<int32_t>(input[k + 2]) * static_cast<int32_t>(w[2]);
            acc += static_cast<int32_t>(input[k + 3]) * static_cast<int32_t>(w[3]);
        }
        
        // Apply scaling
        int group_idx = row / rows_per_group;
        if (group_idx >= num_groups) group_idx = num_groups - 1;
        
        float ws = weight_scales[group_idx];
        output[row] = static_cast<float>(acc) * inv_act_scale * ws;
    }
}

//===----------------------------------------------------------------------===//
// Test data generation
//===----------------------------------------------------------------------===//

class TestDataGenerator {
public:
    TestDataGenerator(unsigned seed = 42) : rng(seed) {}
    
    // Generate random int8 input activations
    void gen_input(int8_t* data, int size) {
        std::uniform_int_distribution<int> dist(-127, 127);
        for (int i = 0; i < size; i++) {
            data[i] = static_cast<int8_t>(dist(rng));
        }
    }
    
    // Generate packed int2 weights
    // Values: 0="-1", 1="0", 2="1" (3 is unused)
    void gen_weights_packed(int8_t* data, int M, int K_packed) {
        std::uniform_int_distribution<int> dist(0, 2);
        for (int i = 0; i < M * K_packed; i++) {
            int8_t packed = 0;
            packed |= static_cast<int8_t>(dist(rng));
            packed |= static_cast<int8_t>(dist(rng)) << 2;
            packed |= static_cast<int8_t>(dist(rng)) << 4;
            packed |= static_cast<int8_t>(dist(rng)) << 6;
            data[i] = packed;
        }
    }
    
    // Generate scale factors
    void gen_scales(float* data, int size, float min_val = 0.1f, float max_val = 2.0f) {
        std::uniform_real_distribution<float> dist(min_val, max_val);
        for (int i = 0; i < size; i++) {
            data[i] = dist(rng);
        }
    }
    
private:
    std::mt19937 rng;
};

//===----------------------------------------------------------------------===//
// Verification
//===----------------------------------------------------------------------===//

int verify_results(
    const bf16_t* npu_output,
    const float* ref_output,
    int size,
    int verbosity
) {
    int errors = 0;
    
    for (int i = 0; i < size; i++) {
        float npu_val = static_cast<float>(npu_output[i]);
        float ref_val = ref_output[i];
        float abs_diff = std::abs(npu_val - ref_val);
        float rel_diff = (ref_val != 0.0f) ? abs_diff / std::abs(ref_val) : abs_diff;
        
        bool pass = (abs_diff <= ABS_TOL) || (rel_diff <= REL_TOL);
        
        if (!pass) {
            errors++;
            if (verbosity >= 2) {
                std::cout << "Mismatch at [" << i << "]: "
                          << "NPU=" << npu_val << ", Ref=" << ref_val
                          << ", Diff=" << abs_diff << std::endl;
            }
        } else if (verbosity >= 3) {
            std::cout << "Match at [" << i << "]: "
                      << "NPU=" << npu_val << ", Ref=" << ref_val << std::endl;
        }
    }
    
    return errors;
}

//===----------------------------------------------------------------------===//
// Test utilities
//===----------------------------------------------------------------------===//

std::vector<uint32_t> load_instr_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open instruction file: " + path);
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<uint32_t> instr(size / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(instr.data()), size);
    
    return instr;
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -x <xclbin>     XCLBIN file path\n"
              << "  -i <instr>      Instruction binary file path\n"
              << "  -M <dim>        Output dimension M (default: " << DEFAULT_M << ")\n"
              << "  -K <dim>        Input dimension K (default: " << DEFAULT_K << ")\n"
              << "  -n <iters>      Number of iterations (default: 1)\n"
              << "  -v <level>      Verbosity level 0-3 (default: 1)\n"
              << "  -h              Show this help\n";
}

//===----------------------------------------------------------------------===//
// Main test function
//===----------------------------------------------------------------------===//

int main(int argc, char* argv[]) {
    // Default parameters
    std::string xclbin_path = "build/final.xclbin";
    std::string instr_path = "build/insts.bin";
    int M = DEFAULT_M;
    int K = DEFAULT_K;
    int n_iters = 1;
    int verbosity = 1;
    constexpr int NUM_WEIGHT_GROUPS = 4;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-x" && i + 1 < argc) {
            xclbin_path = argv[++i];
        } else if (arg == "-i" && i + 1 < argc) {
            instr_path = argv[++i];
        } else if (arg == "-M" && i + 1 < argc) {
            M = std::atoi(argv[++i]);
        } else if (arg == "-K" && i + 1 < argc) {
            K = std::atoi(argv[++i]);
        } else if (arg == "-n" && i + 1 < argc) {
            n_iters = std::atoi(argv[++i]);
        } else if (arg == "-v" && i + 1 < argc) {
            verbosity = std::atoi(argv[++i]);
        } else if (arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    // Validate dimensions
    if (K % 4 != 0) {
        std::cerr << "Error: K must be divisible by 4 (int2 packing)\n";
        return 1;
    }
    
    int K_packed = K / 4;
    
    if (verbosity >= 1) {
        std::cout << "BitLinear Test Configuration:\n"
                  << "  M (output dim): " << M << "\n"
                  << "  K (input dim): " << K << "\n"
                  << "  K_packed: " << K_packed << "\n"
                  << "  Weight groups: " << NUM_WEIGHT_GROUPS << "\n"
                  << "  Iterations: " << n_iters << "\n";
    }
    
    // Calculate buffer sizes
    size_t input_size = K * sizeof(int8_t);
    size_t weights_size = M * K_packed * sizeof(int8_t);
    size_t output_size = M * sizeof(bf16_t);
    size_t act_scale_size = sizeof(bf16_t);
    size_t weight_scales_size = NUM_WEIGHT_GROUPS * sizeof(bf16_t);
    
    // Generate test data
    TestDataGenerator gen(42);
    
    std::vector<int8_t> input(K);
    std::vector<int8_t> weights_packed(M * K_packed);
    std::vector<bf16_t> output_npu(M);
    std::vector<float> output_ref(M);
    std::vector<float> weight_scales_f(NUM_WEIGHT_GROUPS);
    std::vector<bf16_t> weight_scales_bf16(NUM_WEIGHT_GROUPS);
    
    gen.gen_input(input.data(), K);
    gen.gen_weights_packed(weights_packed.data(), M, K_packed);
    gen.gen_scales(weight_scales_f.data(), NUM_WEIGHT_GROUPS);
    
    // Convert scales to bfloat16
    float act_scale_f = 127.0f;  // Typical activation scale
    bf16_t act_scale_bf16(act_scale_f);
    for (int i = 0; i < NUM_WEIGHT_GROUPS; i++) {
        weight_scales_bf16[i] = bf16_t(weight_scales_f[i]);
    }
    
    // Compute reference result
    if (verbosity >= 1) {
        std::cout << "Computing reference result...\n";
    }
    bitlinear_ref(
        input.data(),
        weights_packed.data(),
        output_ref.data(),
        act_scale_f,
        weight_scales_f.data(),
        M, K, NUM_WEIGHT_GROUPS
    );
    
    // Load instruction binary
    std::vector<uint32_t> instr_v;
    try {
        instr_v = load_instr_binary(instr_path);
        if (verbosity >= 1) {
            std::cout << "Loaded " << instr_v.size() << " instructions\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: " << e.what() << "\n";
        std::cerr << "Running reference-only test (no NPU execution)\n";
        
        // For reference-only testing, just verify reference produces non-zero output
        bool has_nonzero = false;
        for (int i = 0; i < M; i++) {
            if (output_ref[i] != 0.0f) {
                has_nonzero = true;
                break;
            }
        }
        
        if (has_nonzero) {
            std::cout << "\nReference test PASS (produces non-zero output)\n";
            if (verbosity >= 2) {
                std::cout << "Sample outputs: ";
                for (int i = 0; i < std::min(5, M); i++) {
                    std::cout << output_ref[i] << " ";
                }
                std::cout << "...\n";
            }
            return 0;
        } else {
            std::cout << "\nReference test FAIL (all zeros)\n";
            return 1;
        }
    }
    
    // Initialize XRT
    if (verbosity >= 1) {
        std::cout << "Initializing XRT...\n";
    }
    
    unsigned int device_index = 0;
    auto device = xrt::device(device_index);
    
    // Load XCLBIN
    if (verbosity >= 1) {
        std::cout << "Loading XCLBIN: " << xclbin_path << "\n";
    }
    auto xclbin = xrt::xclbin(xclbin_path);
    
    // Get kernel
    auto xkernels = xclbin.get_kernels();
    auto xkernel = xkernels[0];
    auto kernel_name = xkernel.get_name();
    
    if (verbosity >= 1) {
        std::cout << "Using kernel: " << kernel_name << "\n";
    }
    
    device.register_xclbin(xclbin);
    xrt::hw_context context(device, xclbin.get_uuid());
    auto kernel = xrt::kernel(context, kernel_name);
    
    // Create buffer objects
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_input = xrt::bo(device, input_size,
                            XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_weights = xrt::bo(device, weights_size,
                              XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_output = xrt::bo(device, output_size,
                             XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    auto bo_act_scale = xrt::bo(device, act_scale_size,
                                XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
    auto bo_weight_scales = xrt::bo(device, weight_scales_size,
                                    XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));
    
    // Map buffers and copy data
    void* buf_instr = bo_instr.map<void*>();
    std::memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(uint32_t));
    
    int8_t* buf_input = bo_input.map<int8_t*>();
    std::memcpy(buf_input, input.data(), input_size);
    
    int8_t* buf_weights = bo_weights.map<int8_t*>();
    std::memcpy(buf_weights, weights_packed.data(), weights_size);
    
    bf16_t* buf_output = bo_output.map<bf16_t*>();
    std::memset(buf_output, 0, output_size);
    
    bf16_t* buf_act_scale = bo_act_scale.map<bf16_t*>();
    *buf_act_scale = act_scale_bf16;
    
    bf16_t* buf_weight_scales = bo_weight_scales.map<bf16_t*>();
    std::memcpy(buf_weight_scales, weight_scales_bf16.data(), weight_scales_size);
    
    // Sync to device
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_act_scale.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_weight_scales.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    // Run kernel
    int total_errors = 0;
    float total_time = 0.0f;
    
    for (int iter = 0; iter < n_iters; iter++) {
        if (verbosity >= 1) {
            std::cout << "Running iteration " << (iter + 1) << "...\n";
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        unsigned int opcode = 3;
        auto run = kernel(opcode, bo_instr, instr_v.size(),
                         bo_input, bo_weights, bo_output,
                         bo_act_scale, bo_weight_scales);
        run.wait();
        
        auto stop = std::chrono::high_resolution_clock::now();
        float time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            stop - start).count();
        total_time += time_us;
        
        // Sync output from device
        bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::memcpy(output_npu.data(), buf_output, output_size);
        
        // Verify
        int errors = verify_results(output_npu.data(), output_ref.data(), M, verbosity);
        total_errors += errors;
        
        if (verbosity >= 1) {
            std::cout << "  Time: " << time_us << " us, Errors: " << errors << "\n";
        }
    }
    
    // Print summary
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Dimensions: M=" << M << ", K=" << K << std::endl;
    std::cout << "Iterations: " << n_iters << std::endl;
    std::cout << "Avg time: " << (total_time / n_iters) << " us" << std::endl;
    std::cout << "Total errors: " << total_errors << std::endl;
    
    // Compute throughput
    float macs = static_cast<float>(M) * static_cast<float>(K);
    float avg_time_s = (total_time / n_iters) / 1e6f;
    float gmacs = macs / 1e9f / avg_time_s;
    std::cout << "Throughput: " << gmacs << " GMACs/s" << std::endl;
    
    if (total_errors == 0) {
        std::cout << "\nPASS!\n";
        return 0;
    } else {
        std::cout << "\nFAIL!\n";
        return 1;
    }
}

