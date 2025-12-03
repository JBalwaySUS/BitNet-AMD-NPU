//===- zero.cc ----------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025
//
// Zero buffer kernels for AIE
// Based on mlir-aie/aie_kernels/aie2/zero.cc
//
//===----------------------------------------------------------------------===//

#ifndef ZERO_CC
#define ZERO_CC

#include <aie_api/aie.hpp>

// Scalar zero implementation
template <typename T, unsigned m, unsigned n>
void zero_scalar(T *__restrict c) {
  for (unsigned i = 0; i < m * n; i++) {
    c[i] = 0;
  }
}

// Vectorized zero implementation
template <typename T, unsigned m, unsigned n>
void zero_vectorized(T *__restrict c) {
  constexpr unsigned vec_size = 32 / sizeof(T);  // 256-bit vector
  const aie::vector<T, vec_size> zeros = aie::zeros<T, vec_size>();

  for (unsigned i = 0; i < m * n; i += vec_size) {
    aie::store_v(c + i, zeros);
  }
}

#endif // ZERO_CC

