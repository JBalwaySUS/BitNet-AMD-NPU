//===- bitlinear_int8xint2.cc -----------------------------------*- C++ -*-===//
//
// BitNet int8 x int2 matrix-vector multiplication kernel for AMD AIE
//
// This kernel performs quantized matrix-vector multiplication where:
// - Activations (input vector) are int8 (quantized from bfloat16)
// - Weights (matrix) are int2 packed (4 values per byte)
// - Output is bfloat16 after scaling
//
// Copyright (C) 2025
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

// Default tile dimensions - can be overridden at compile time
#ifndef DIM_M
#define DIM_M 32  // Output dimension (rows of weight matrix)
#endif

#ifndef DIM_K
#define DIM_K 32  // Input dimension (columns of weight matrix)
#endif

// Number of weight groups for per-group scaling
#ifndef NUM_WEIGHT_GROUPS
#define NUM_WEIGHT_GROUPS 4
#endif

//===----------------------------------------------------------------------===//
// Helper: Unpack int2 values from packed byte
//===----------------------------------------------------------------------===//

// Unpack 4 int2 values from a single byte
// int2 values are packed as: byte = (v3 << 6) | (v2 << 4) | (v1 << 2) | v0
// Values are in range [-1, 0, 1] stored as [0, 1, 2, 3] -> subtract 1 to get [-1, 0, 1]
// Actually BitNet uses ternary: {-1, 0, 1} stored as 2-bit: 00=0, 01=1, 10=-1, 11=unused
// Or simpler: values 0,1,2,3 map to -1,0,1 (subtract 1, but 3 is unused)

inline void unpack_int2_to_int8(int8_t packed, int8_t *out) {
  // Extract 4 int2 values and convert to int8
  // Packed format: bits [1:0] = v0, [3:2] = v1, [5:4] = v2, [7:6] = v3
  out[0] = (int8_t)((packed & 0x03) - 1);       // bits [1:0] - 1
  out[1] = (int8_t)(((packed >> 2) & 0x03) - 1); // bits [3:2] - 1  
  out[2] = (int8_t)(((packed >> 4) & 0x03) - 1); // bits [5:4] - 1
  out[3] = (int8_t)(((packed >> 6) & 0x03) - 1); // bits [7:6] - 1
}

//===----------------------------------------------------------------------===//
// Zero output buffer
//===----------------------------------------------------------------------===//

template <typename T, int M>
void zero_output(T *__restrict c) {
  for (int i = 0; i < M; i++) {
    c[i] = 0;
  }
}

template <int M>
void zero_output_i32(int32_t *__restrict c) {
  constexpr int vec_size = 8; // 256-bit / 32-bit = 8 elements
  static_assert(M % vec_size == 0, "M must be divisible by vector size");
  
  const aie::vector<int32_t, vec_size> zeros = aie::zeros<int32_t, vec_size>();
  
  for (int i = 0; i < M; i += vec_size) {
    aie::store_v(c + i, zeros);
  }
}

//===----------------------------------------------------------------------===//
// Scalar BitLinear kernel (for verification)
//===----------------------------------------------------------------------===//

// Scalar implementation for correctness verification
// input: int8 activation vector of size K
// weights_packed: int2 packed weight matrix of size M x (K/4)
// output_acc: int32 accumulator of size M
// After accumulation, scaling is applied separately
template <int M, int K>
void bitlinear_scalar(int8_t *__restrict input,
                      int8_t *__restrict weights_packed,
                      int32_t *__restrict output_acc) {
  event0();
  
  // For each output row
  for (int row = 0; row < M; row++) {
    int32_t acc = 0;
    
    // Process K input elements, 4 at a time (since 4 int2 per byte)
    for (int k = 0; k < K; k += 4) {
      // Get packed weight byte for this position
      int8_t packed = weights_packed[row * (K / 4) + (k / 4)];
      
      // Unpack to 4 int8 values
      int8_t w[4];
      unpack_int2_to_int8(packed, w);
      
      // Multiply-accumulate with input
      acc += (int32_t)input[k + 0] * (int32_t)w[0];
      acc += (int32_t)input[k + 1] * (int32_t)w[1];
      acc += (int32_t)input[k + 2] * (int32_t)w[2];
      acc += (int32_t)input[k + 3] * (int32_t)w[3];
    }
    
    output_acc[row] += acc;
  }
  
  event1();
}

//===----------------------------------------------------------------------===//
// Vectorized BitLinear kernel
//===----------------------------------------------------------------------===//

// Vectorized implementation using AIE vector operations
// Processes multiple elements in parallel
template <int M, int K, int VEC_SIZE>
void bitlinear_vectorized(int8_t *__restrict input,
                          int8_t *__restrict weights_packed,
                          int32_t *__restrict output_acc) {
  static_assert(K % 4 == 0, "K must be divisible by 4 (int2 packing)");
  static_assert(M % VEC_SIZE == 0, "M must be divisible by vector size");
  
  event0();
  
  // Process VEC_SIZE output rows at a time
  for (int row_base = 0; row_base < M; row_base += VEC_SIZE) {
    aie::accum<acc32, VEC_SIZE> acc;
    acc.from_vector(aie::load_v<VEC_SIZE>(output_acc + row_base));
    
    // Process K input elements
    for (int k = 0; k < K; k += 4) {
      // Load 4 input values (to multiply with unpacked weights)
      int8_t in_vals[4];
      in_vals[0] = input[k + 0];
      in_vals[1] = input[k + 1];
      in_vals[2] = input[k + 2];
      in_vals[3] = input[k + 3];
      
      // For each row in current vector batch
      for (int v = 0; v < VEC_SIZE; v++) {
        int row = row_base + v;
        
        // Get packed weight byte
        int8_t packed = weights_packed[row * (K / 4) + (k / 4)];
        
        // Unpack weights
        int8_t w[4];
        unpack_int2_to_int8(packed, w);
        
        // Accumulate dot product
        int32_t partial = 0;
        partial += (int32_t)in_vals[0] * (int32_t)w[0];
        partial += (int32_t)in_vals[1] * (int32_t)w[1];
        partial += (int32_t)in_vals[2] * (int32_t)w[2];
        partial += (int32_t)in_vals[3] * (int32_t)w[3];
        
        // Add to accumulator element v
        // Note: This is a simplified version; full vectorization would use
        // AIE multiply-accumulate intrinsics
        auto acc_vec = acc.to_vector<int32_t>();
        acc_vec[v] += partial;
        acc.from_vector(acc_vec);
      }
    }
    
    // Store accumulated results
    aie::store_v(output_acc + row_base, acc.to_vector<int32_t>());
  }
  
  event1();
}

//===----------------------------------------------------------------------===//
// Apply scaling to convert int32 accumulator to bfloat16 output
//===----------------------------------------------------------------------===//

// Apply activation scale and weight scale to convert accumulated int32 to bfloat16
// output = (acc / act_scale) * weight_scale
// act_scale: per-tensor activation scale (scalar)
// weight_scales: per-group weight scales
template <int M>
void apply_scales_scalar(int32_t *__restrict acc,
                         bfloat16 *__restrict output,
                         bfloat16 act_scale,
                         bfloat16 *__restrict weight_scales,
                         int num_weight_groups) {
  event0();
  
  int rows_per_group = M / num_weight_groups;
  float inv_act_scale = 1.0f / (float)act_scale;
  
  for (int i = 0; i < M; i++) {
    int group_idx = i / rows_per_group;
    if (group_idx >= num_weight_groups) group_idx = num_weight_groups - 1;
    
    float ws = (float)weight_scales[group_idx];
    float result = (float)acc[i] * inv_act_scale * ws;
    output[i] = (bfloat16)result;
  }
  
  event1();
}

//===----------------------------------------------------------------------===//
// Combined BitLinear kernel (accumulate + scale in one pass)
//===----------------------------------------------------------------------===//

// Full BitLinear operation: matvec with int8 x int2 -> bfloat16
// This version computes everything in one kernel call
template <int M, int K>
void bitlinear_full_scalar(int8_t *__restrict input,
                           int8_t *__restrict weights_packed,
                           bfloat16 *__restrict output,
                           bfloat16 act_scale,
                           bfloat16 *__restrict weight_scales,
                           int num_weight_groups) {
  event0();
  
  int rows_per_group = M / num_weight_groups;
  float inv_act_scale = 1.0f / (float)act_scale;
  
  // For each output row
  for (int row = 0; row < M; row++) {
    int32_t acc = 0;
    
    // Process K input elements, 4 at a time
    for (int k = 0; k < K; k += 4) {
      int8_t packed = weights_packed[row * (K / 4) + (k / 4)];
      
      int8_t w[4];
      unpack_int2_to_int8(packed, w);
      
      acc += (int32_t)input[k + 0] * (int32_t)w[0];
      acc += (int32_t)input[k + 1] * (int32_t)w[1];
      acc += (int32_t)input[k + 2] * (int32_t)w[2];
      acc += (int32_t)input[k + 3] * (int32_t)w[3];
    }
    
    // Apply scaling
    int group_idx = row / rows_per_group;
    if (group_idx >= num_weight_groups) group_idx = num_weight_groups - 1;
    
    float ws = (float)weight_scales[group_idx];
    float result = (float)acc * inv_act_scale * ws;
    output[row] = (bfloat16)result;
  }
  
  event1();
}

//===----------------------------------------------------------------------===//
// C-linkage function exports
//===----------------------------------------------------------------------===//

extern "C" {

// Zero output buffer (int32)
void zero_i32(int32_t *c) {
  zero_output<int32_t, DIM_M>(c);
}

// Zero output buffer (bfloat16)
void zero_bf16(bfloat16 *c) {
  zero_output<bfloat16, DIM_M>(c);
}

// Scalar BitLinear accumulation (no scaling)
// Used when accumulating across multiple tiles
void bitlinear_acc_scalar(int8_t *input, int8_t *weights_packed, 
                          int32_t *output_acc) {
  bitlinear_scalar<DIM_M, DIM_K>(input, weights_packed, output_acc);
}

// Apply scales to accumulated results
void bitlinear_scale(int32_t *acc, bfloat16 *output,
                     bfloat16 *act_scale, bfloat16 *weight_scales) {
  apply_scales_scalar<DIM_M>(acc, output, *act_scale, weight_scales, 
                              NUM_WEIGHT_GROUPS);
}

// Full BitLinear operation (accumulate + scale)
void bitlinear_full(int8_t *input, int8_t *weights_packed,
                    bfloat16 *output, bfloat16 *act_scale,
                    bfloat16 *weight_scales) {
  bitlinear_full_scalar<DIM_M, DIM_K>(input, weights_packed, output,
                                       *act_scale, weight_scales,
                                       NUM_WEIGHT_GROUPS);
}

// Parameterized versions with runtime dimensions
void bitlinear_acc_scalar_param(int8_t *input, int8_t *weights_packed,
                                int32_t *output_acc, int32_t M, int32_t K) {
  // Use default tile sizes, caller responsible for tiling
  bitlinear_scalar<DIM_M, DIM_K>(input, weights_packed, output_acc);
}

} // extern "C"

