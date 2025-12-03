//===- bitlinear.cc -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025
//
// BitLinear int8 x int2 matrix-vector multiplication kernel for AMD AIE
// Based on mlir-aie/aie_kernels/aie2/mv.cc
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

#include "zero.cc"

//===----------------------------------------------------------------------===//
// Helper: Unpack int2 values from packed byte
//===----------------------------------------------------------------------===//

// int2 values packed as: byte = (v3 << 6) | (v2 << 4) | (v1 << 2) | v0
// Values {0,1,2} map to {-1,0,1} (subtract 1)
inline void unpack_int2_to_int8(int8_t packed, int8_t *out) {
  out[0] = (int8_t)((packed & 0x03) - 1);
  out[1] = (int8_t)(((packed >> 2) & 0x03) - 1);
  out[2] = (int8_t)(((packed >> 4) & 0x03) - 1);
  out[3] = (int8_t)(((packed >> 6) & 0x03) - 1);
}

//===----------------------------------------------------------------------===//
// Scalar BitLinear kernel
//===----------------------------------------------------------------------===//

// Scalar implementation: int8 activations x int2 packed weights -> int32 acc
// input: int8 vector [K]
// weights_packed: int2 packed matrix [M, K/4] (4 int2 per byte)
// output_acc: int32 accumulator [M]
template <int M, int K>
void bitlinear_scalar(int8_t *__restrict input,
                      int8_t *__restrict weights_packed,
                      int32_t *__restrict output_acc) {
  event0();

  for (int row = 0; row < M; row++) {
    int32_t acc = 0;

    // Process K elements, 4 at a time (int2 packing)
    for (int col = 0; col < K; col += 4) {
      int8_t packed = weights_packed[row * (K / 4) + (col / 4)];

      int8_t w[4];
      unpack_int2_to_int8(packed, w);

      acc += (int32_t)input[col + 0] * (int32_t)w[0];
      acc += (int32_t)input[col + 1] * (int32_t)w[1];
      acc += (int32_t)input[col + 2] * (int32_t)w[2];
      acc += (int32_t)input[col + 3] * (int32_t)w[3];
    }

    output_acc[row] += acc;
  }

  event1();
}

//===----------------------------------------------------------------------===//
// Vectorized BitLinear kernel (WIP - uses scalar for now)
//===----------------------------------------------------------------------===//

template <int M, int K, int r>
void bitlinear_vectorized(int8_t *__restrict input,
                          int8_t *__restrict weights_packed,
                          int32_t *__restrict output_acc) {
  // TODO: Implement vectorized version using AIE intrinsics
  // For now, fall back to scalar
  bitlinear_scalar<M, K>(input, weights_packed, output_acc);
}

//===----------------------------------------------------------------------===//
// Scale application: int32 accumulator -> bfloat16 output
//===----------------------------------------------------------------------===//

// Apply activation and weight scales to convert int32 to bfloat16
// output = (acc / act_scale) * weight_scale
template <int M, int num_groups>
void apply_scales_scalar(int32_t *__restrict acc,
                         bfloat16 *__restrict output,
                         bfloat16 act_scale,
                         bfloat16 *__restrict weight_scales) {
  event0();

  int rows_per_group = M / num_groups;
  float inv_act_scale = 1.0f / (float)act_scale;

  for (int i = 0; i < M; i++) {
    int group_idx = i / rows_per_group;
    if (group_idx >= num_groups) group_idx = num_groups - 1;

    float ws = (float)weight_scales[group_idx];
    float result = (float)acc[i] * inv_act_scale * ws;
    output[i] = (bfloat16)result;
  }

  event1();
}

//===----------------------------------------------------------------------===//
// Combined BitLinear: matvec + scale in one pass
//===----------------------------------------------------------------------===//

template <int M, int K, int num_groups>
void bitlinear_full_scalar(int8_t *__restrict input,
                           int8_t *__restrict weights_packed,
                           bfloat16 *__restrict output,
                           bfloat16 act_scale,
                           bfloat16 *__restrict weight_scales) {
  event0();

  int rows_per_group = M / num_groups;
  float inv_act_scale = 1.0f / (float)act_scale;

  for (int row = 0; row < M; row++) {
    int32_t acc = 0;

    // Compute dot product
    for (int col = 0; col < K; col += 4) {
      int8_t packed = weights_packed[row * (K / 4) + (col / 4)];

      int8_t w[4];
      unpack_int2_to_int8(packed, w);

      acc += (int32_t)input[col + 0] * (int32_t)w[0];
      acc += (int32_t)input[col + 1] * (int32_t)w[1];
      acc += (int32_t)input[col + 2] * (int32_t)w[2];
      acc += (int32_t)input[col + 3] * (int32_t)w[3];
    }

    // Apply scaling
    int group_idx = row / rows_per_group;
    if (group_idx >= num_groups) group_idx = num_groups - 1;

    float ws = (float)weight_scales[group_idx];
    float result = (float)acc * inv_act_scale * ws;
    output[row] = (bfloat16)result;
  }

  event1();
}

//===----------------------------------------------------------------------===//
// C-linkage exports
//===----------------------------------------------------------------------===//

extern "C" {

// Tile dimensions - define at compile time with -DDIM_M=32 etc.
#ifndef DIM_M
#define DIM_M 32
#endif

#ifndef DIM_K
#define DIM_K 32
#endif

#ifndef NUM_WEIGHT_GROUPS
#define NUM_WEIGHT_GROUPS 4
#endif

// Zero functions (scalar and vectorized)
void zero_scalar_i32(int32_t *c) {
  zero_scalar<int32_t, DIM_M, 1>(c);
}

void zero_vectorized_i32(int32_t *c) {
  zero_vectorized<int32_t, DIM_M, 1>(c);
}

void zero_scalar_bf16(bfloat16 *c) {
  zero_scalar<bfloat16, DIM_M, 1>(c);
}

// BitLinear accumulation (no scaling) - scalar version
void bitlinear_scalar_i8_i2_i32(int8_t *a_in, int8_t *b_in, int32_t *c_out) {
  bitlinear_scalar<DIM_M, DIM_K>(a_in, b_in, c_out);
}

// BitLinear accumulation - vectorized version
void bitlinear_vectorized_i8_i2_i32(int8_t *a_in, int8_t *b_in, int32_t *c_out) {
  bitlinear_vectorized<DIM_M, DIM_K, 16>(a_in, b_in, c_out);
}

// Apply scales: int32 -> bfloat16
void bitlinear_scale_i32_bf16(int32_t *acc, bfloat16 *output,
                               bfloat16 *act_scale, bfloat16 *weight_scales) {
  apply_scales_scalar<DIM_M, NUM_WEIGHT_GROUPS>(acc, output, *act_scale, weight_scales);
}

// Full BitLinear: int8 x int2 -> bfloat16 (scalar)
void bitlinear_full_scalar_i8_i2_bf16(int8_t *input, int8_t *weights,
                                       bfloat16 *output, bfloat16 *act_scale,
                                       bfloat16 *weight_scales) {
  bitlinear_full_scalar<DIM_M, DIM_K, NUM_WEIGHT_GROUPS>(
      input, weights, output, *act_scale, weight_scales);
}

// Full BitLinear: int8 x int2 -> bfloat16 (vectorized - uses scalar for now)
void bitlinear_full_vectorized_i8_i2_bf16(int8_t *input, int8_t *weights,
                                           bfloat16 *output, bfloat16 *act_scale,
                                           bfloat16 *weight_scales) {
  bitlinear_full_scalar<DIM_M, DIM_K, NUM_WEIGHT_GROUPS>(
      input, weights, output, *act_scale, weight_scales);
}

} // extern "C"

