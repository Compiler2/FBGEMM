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

void llc_flush(std::vector<char>& llc) {
  volatile char* data = llc.data();
  for (size_t i = 0; i < llc.size(); i++) {
    data[i]++;
  }
}


/**
 * Allocator for aligned data.
 *
 * Modified from the Mallocator from Stephan T. Lavavej.
 * <http://blogs.msdn.com/b/vcblog/archive/2008/08/28/the-mallocator.aspx>
 *
 */
template <typename T, std::size_t Alignment>
class aligned_allocator {
 public:
  // The following will be the same for virtually all allocators.
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef T value_type;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  T* address(T& r) const {
    return &r;
  }

  const T* address(const T& s) const {
    return &s;
  }

  std::size_t max_size() const {
    // The following has been carefully written to be independent of
    // the definition of size_t and to avoid signed/unsigned warnings.
    return (static_cast<std::size_t>(0) - static_cast<std::size_t>(1)) /
        sizeof(T);
  }

  // The following must be the same for all allocators.
  template <typename U>
  struct rebind {
    typedef aligned_allocator<U, Alignment> other;
  };

  bool operator!=(const aligned_allocator& other) const {
    return !(*this == other);
  }

  void construct(T* const p, const T& t) const {
    void* const pv = static_cast<void*>(p);

    new (pv) T(t);
  }

  void destroy(T* const p) const {
    p->~T();
  }

  // Returns true if and only if storage allocated from *this
  // can be deallocated from other, and vice versa.
  // Always returns true for stateless allocators.
  bool operator==(const aligned_allocator& /*other*/) const {
    return true;
  }

  // Default constructor, copy constructor, rebinding constructor, and
  // destructor. Empty for stateless allocators.
  aligned_allocator() {}

  aligned_allocator(const aligned_allocator&) {}

  template <typename U>
  aligned_allocator(const aligned_allocator<U, Alignment>&) {}

  ~aligned_allocator() {}

  // The following will be different for each allocator.
  T* allocate(const std::size_t n) const {
    // The return value of allocate(0) is unspecified.
    // Mallocator returns NULL in order to avoid depending
    // on malloc(0)'s implementation-defined behavior
    // (the implementation can define malloc(0) to return NULL,
    // in which case the bad_alloc check below would fire).
    // All allocators can return NULL in this case.
    if (n == 0) {
      return nullptr;
    }

    // All allocators should contain an integer overflow check.
    // The Standardization Committee recommends that std::length_error
    // be thrown in the case of integer overflow.
    if (n > max_size()) {
      throw std::length_error(
          "aligned_allocator<T>::allocate() - Integer overflow.");
    }

    // Mallocator wraps malloc().
    void* pv = nullptr;
    int ret;
#ifdef _MSC_VER
    pv = _aligned_malloc(n * sizeof(T), Alignment);
    ret = 0;
#else
    ret = posix_memalign(&pv, Alignment, n * sizeof(T));
#endif
    // pv = aligned_alloc(Alignment, n * sizeof(T));

    // Allocators should throw std::bad_alloc in the case of memory allocation
    // failure.
    if (ret || pv == nullptr) {
      throw std::bad_alloc();
    }

    return static_cast<T*>(pv);
  }

  void deallocate(T* const p, const std::size_t /*n*/) const {
#ifdef _MSC_VER
    _aligned_free(p);
#else
    free(p);
#endif
  }

  // The following will be the same for all allocators that ignore hints.
  template <typename U>
  T* allocate(const std::size_t n, const U* /* const hint */) const {
    return allocate(n);
  }

  // Allocators are not required to be assignable, so
  // all allocators should have a private unimplemented
  // assignment operator. Note that this will trigger the
  // off-by-default (enabled under /Wall) warning C4626
  // "assignment operator could not be generated because a
  // base class assignment operator is inaccessible" within
  // the STL headers, but that warning is useless.
 private:
  aligned_allocator& operator=(const aligned_allocator&) {
    assert(0);
  }
};

template <typename T>
using aligned_vector = std::vector<T, aligned_allocator<T, 64>>;

using namespace std;
using namespace fbgemm;

namespace fbgemm {

std::default_random_engine eng;

template <typename T>
void randFill(aligned_vector<T>& vec, T low, T high, std::true_type) {
  std::uniform_int_distribution<int> dis(low, high);
  std::generate(vec.begin(), vec.end(), [&] { return dis(eng); });
}

template <typename T>
void randFill(aligned_vector<T>& vec, T low, T high, std::false_type) {
  std::uniform_real_distribution<T> dis(low, high);
  std::generate(vec.begin(), vec.end(), [&] { return dis(eng); });
}

template <typename T>
void randFill(aligned_vector<T>& vec, T low, T high) {
  randFill(vec, low, high, std::is_integral<T>());
}

template void
randFill<float>(aligned_vector<float>& vec, float low, float high);
template void
randFill<uint8_t>(aligned_vector<uint8_t>& vec, uint8_t low, uint8_t high);
template void
randFill<int8_t>(aligned_vector<int8_t>& vec, int8_t low, int8_t high);
template void randFill<int>(aligned_vector<int>& vec, int low, int high);
// template void
// randFill<int64_t>(aligned_vector<int64_t>& vec, int64_t low, int64_t high);
template <>
void randFill(aligned_vector<int64_t>& vec, int64_t low, int64_t high) {
  std::uniform_int_distribution<int64_t> dis(low, high);
  std::generate(vec.begin(), vec.end(), [&] { return dis(eng); });
}



/*
 * @brief Make sure we won't have overflows from vpmaddubsw instruction.
 */
template <typename T>
void avoidOverflow(
    int m,
    int n,
    int k,
    const uint8_t* Aint8,
    int lda,
    T* B,
    int ldb) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int kk = 0; kk < k / 2 * 2; kk += 2) {
        int a0 = Aint8[i * lda + kk], a1 = Aint8[i * lda + kk + 1];
        int b0 = B[kk * ldb + j], b1 = B[(kk + 1) * ldb + j];
        int sum_pair = a0 * b0 + a1 * b1;
        if (sum_pair < numeric_limits<int16_t>::lowest()) {
          int b1_adjusted =
              ceil((numeric_limits<int16_t>::lowest() - a0 * b0) / a1);
          b1_adjusted = std::min(std::max(b1_adjusted, -128), 127);

          int new_sum_pair = a0 * b0 + a1 * b1_adjusted;
          (void)new_sum_pair; // Suppress unused variable warning
          assert(
              new_sum_pair >= numeric_limits<int16_t>::lowest() &&
              new_sum_pair <= numeric_limits<int16_t>::max());
          B[(kk + 1) * n + j] = b1_adjusted;
        } else if (sum_pair > numeric_limits<int16_t>::max()) {
          int b1_adjusted =
              floor((numeric_limits<int16_t>::max() - a0 * b0) / a1);
          b1_adjusted = std::min(std::max(b1_adjusted, -128), 127);

          int new_sum_pair = a0 * b0 + a1 * b1_adjusted;
          (void)new_sum_pair; // Suppress unused variable warning
          assert(
              new_sum_pair >= numeric_limits<int16_t>::lowest() &&
              new_sum_pair <= numeric_limits<int16_t>::max());
          B[(kk + 1) * ldb + j] = b1_adjusted;
        }
      }
    } // for each j
  } // for each i
}

template <typename T>
void avoidOverflow(int m, int n, int k, const uint8_t* Aint8, T* B) {
  return avoidOverflow(m, n, k, Aint8, k, B, n);
}

template void
avoidOverflow(int m, int n, int k, const uint8_t* Aint8, int8_t* B);
template void
avoidOverflow(int m, int n, int k, const uint8_t* Aint8, float* B);
}

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

  constexpr int NWARMUP = 4;
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
