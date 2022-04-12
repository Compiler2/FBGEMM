/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <random>

#ifndef GEMM_M
#define GEMM_M 64
#endif

#ifndef GEMM_N
#define GEMM_N 800
#endif

#ifndef GEMM_K
#define GEMM_K 320
#endif

#include "fbgemm/Fbgemm.h"
#include "common.h"

void performance_test() {
  std::cout << "start" << std::endl;
  // clang-format off
  static const vector<vector<int>> shapes = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
    // m, n, k
    {GEMM_M, GEMM_N, GEMM_K},
  };
  // clang-format on
  bool flush = true;
  std::vector<char> llc;

  if (flush) {
    llc.resize(128 * 1024 * 1024, 1.0);
  }

  constexpr int NWARMUP = 2;
  constexpr int NITER = 10;

  chrono::time_point<chrono::high_resolution_clock> start, end;
  for (const auto& shape : shapes) {
    int m = shape[0];
    int n = shape[1];
    int k = shape[2];

    aligned_vector<uint8_t> Aint8(m * k);

    aligned_vector<int8_t> Bint8(k * n);

    aligned_vector<float> Cfp32_mkl(m * n);
    aligned_vector<int32_t> Cint32_mkl(Cfp32_mkl.size());
    aligned_vector<int32_t> Cint32_ref(Cfp32_mkl.size());
    aligned_vector<int32_t> Cint32_fb_acc32(Cfp32_mkl.size());
    aligned_vector<int32_t> Cint32_fb_acc16(Cfp32_mkl.size());

    // A matrix
    randFill<uint8_t>(Aint8, 0, 5);
    aligned_vector<float> Afp32(Aint8.begin(), Aint8.end());

    randFill<int8_t>(Bint8, -4, 4);
    avoidOverflow(m, n, k, Aint8.data(), Bint8.data());

    aligned_vector<float> Bfp32(Bint8.begin(), Bint8.end());

    double nops = 2.0 * m * n * k;
    double ttot = 0.0;
    string runType;

    vector<int32_t> row_offsets(m);

    PackBMatrix<int8_t> packedB_int32(
        matrix_op_t::NoTranspose, k, n, Bint8.data(), n, nullptr, 1);

    ttot = 0.0;
    runType = "FBGEMM_i8_acc32";
    cout << setw(6) << m << ", " << setw(6) << n << ", " << setw(6) << k << ", "
         << setw(16) << runType;

    for (auto i = 0; i < NWARMUP + NITER; ++i) {
      llc_flush(llc);
      start = chrono::high_resolution_clock::now();

      {
        PackAMatrix<uint8_t> packA_int32(
            matrix_op_t::NoTranspose, m, k, Aint8.data(), k, nullptr, 1);

        DoNothing<int32_t, int32_t> doNothing32BitObj;
        memCopy<> memcopyObj(doNothing32BitObj);
        // printf ( "tid: %d, num_threads: %d\n", tid, num_threads );
        fbgemmPacked(
            packA_int32,
            packedB_int32,
            Cint32_fb_acc32.data(),
            Cint32_fb_acc32.data(),
            n,
            memcopyObj,
            0,
            1);
      }

      end = chrono::high_resolution_clock::now();

      if (i >= NWARMUP) {
        auto dur = chrono::duration_cast<chrono::nanoseconds>(end - start);
        ttot += dur.count();
      }
    }
    std::cout << ", " << setw(6) << ttot << std::endl;

    if (flush) {
      ((volatile char*)(llc.data()))[0] += 1;
    }
  }
  std::cout << "end" << std::endl;
}

int main(int /* unused */, char** /* unused */) {
  performance_test();
  return 0;
}
