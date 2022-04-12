/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fbgemm/Fbgemm.h"
#include "common.h"

using namespace std;
using namespace fbgemm;

#define CASE 1

// 2D conv shapes
vector<conv_param_t<2>> shapes_2d = {
  // MB, IC, OC, IH, IW, G, KH, KW, stride_h, stride_w,
  // pad_h_top, pad_w_left, pad_h_bottom, pad_w_right,
  // (dilation_h, dilation_w, output_padding_h, output_padding_w, tranpose)
  // 2D convolutions
  // regular
  conv_param_t<>(1, 128, 128, {56, 56}, 1, {3, 3},
      {1, 1}, {1, 1, 1, 1}),
  conv_param_t<>(1, 128, 128, {56, 56}, 1, {3, 3},
      {1, 1}, {1, 1, 1, 1}, {1, 1}, {0}, true),
  conv_param_t<>(1, 128, 128, {56, 56}, 1, {3, 3},
      {1, 1}, {1, 1, 1, 1}, {1, 1}, {1}, true),
  // regular with dilation
  conv_param_t<>(1, 128, 128, {56, 56}, 1, {3, 3},
      {1, 1}, {1, 1, 1, 1}, {2, 2}),
  conv_param_t<>(1, 128, 128, {56, 56}, 1, {3, 3},
      {1, 1}, {1, 1, 1, 1}, {2, 2}, {0}, true),
  conv_param_t<>(1, 128, 128, {56, 56}, 1, {3, 3},
      {1, 1}, {1, 1, 1, 1}, {2, 2}, {1}, true),
  // groupwise
  conv_param_t<>(1, 128, 128, {56, 56}, 32, {3, 3},
      {1, 1}, {1, 1, 1, 1}),
  conv_param_t<>(1, 128, 128, {56, 56}, 32, {3, 3},
      {1, 1}, {1, 1, 1, 1},  {1, 1}, {0}, true),
  conv_param_t<>(1, 128, 128, {56, 56}, 32, {3, 3},
      {1, 1}, {1, 1, 1, 1},  {1, 1}, {0}, true),
  // DW
  conv_param_t<>(1, 272, 272, {47, 125}, 272, {3, 3},
      {1, 1}, {1, 1, 1, 1}),
  conv_param_t<>(1, 128, 256, {32, 100}, 128, {3, 3},
      {1, 1}, {1, 1, 1, 1}),
  // Pointwise
  conv_param_t<>(1, 128, 128, {56, 56}, 1, {1, 1},
      {1, 1}, {0, 0, 0, 0})
};

vector<conv_param_t<2>> shapes_2d_resnext_101 = {
  // ResNext-101 (unique shapes only)
  // conv_param_t<>(N, C, M, H, W, groups, /* kern */ {KH, KW}, /* stride */
  //   {stride_h, stride_w}, /* padding pad_l = pad_h */ {pad_l, pad_l, pad_l, pad_l}, /* dialation */
  //   {1, 1}, /* otpt_pad */ {0, 0}, /* trans */transpose)

  conv_param_t<>(1, 3,    64,   {224, 224}, 1,  {7, 7},
     {2, 2}, {3, 3, 3, 3}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 64,   128,  {56,  56},  1,  {1, 1},
     {1, 1}, {0, 0, 0, 0}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 128,  128,  {56,  56},  32, {3, 3},
     {1, 1}, {1, 1, 1, 1}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 128,  256,  {56,  56},  1,  {1, 1},
     {1, 1}, {0, 0, 0, 0}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 64,   256,  {56,  56},  1,  {1, 1},
     {1, 1}, {0, 0, 0, 0}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 256,  128,  {56,  56},  1,  {1, 1},
     {1, 1}, {1, 1, 1, 1}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 256,  128,  {56,  56},  1,  {1, 1},
     {1, 1}, {0, 0, 0, 0}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 256,  256,  {56,  56},  1,  {1, 1},
     {1, 1}, {0, 0, 0, 0}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 256,  256,  {56,  56},  32, {3, 3},
     {2, 2}, {1, 1, 1, 1}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 256,  512,  {28,  28},  1,  {1, 1},
     {1, 1}, {0, 0, 0, 0}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 256,  512,  {56,  56},  1,  {1, 1},
     {2, 2}, {0, 0, 0, 0}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 512,  256,  {28,  28},  1,  {1, 1},
     {1, 1}, {0, 0, 0, 0}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 256,  256,  {28,  28},  32, {3, 3},
     {1, 1}, {1, 1, 1, 1}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 512,  512,  {28,  28},  1,  {1, 1},
     {1, 1}, {0, 0, 0, 0}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 512,  512,  {28,  28},  32, {3, 3},
     {2, 2}, {1, 1, 1, 1}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 512,  1024, {14,  14},  1,  {1, 1},
     {1, 1}, {0, 0, 0, 0}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 512,  1024, {28,  28},  1,  {1, 1},
     {2, 2}, {0, 0, 0, 0}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 1024, 512,  {14,  14},  1,  {1, 1},
     {1, 1}, {0, 0, 0, 0}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 512,  512,  {14,  14},  32, {3, 3},
     {1, 1}, {1, 1, 1, 1}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 1024, 1024, {14,  14},  1,  {1, 1},
     {1, 1}, {0, 0, 0, 0}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 1024, 1024, {14,  14},  32, {3, 3},
     {2, 2}, {1, 1, 1, 1}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 1024, 2048, {7,   7},   1,  {1, 1},
     {1, 1}, {0, 0, 0, 0}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 1024, 2048, {14,  14},  1,  {1, 1},
     {2, 2}, {0, 0, 0, 0}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 2048, 1024, {7,   7},   1,  {1, 1},
     {1, 1}, {0, 0, 0, 0}, {1, 1}, {0, 0}, false),
  conv_param_t<>(1, 1024, 1024, {7,   7},   32, {3, 3},
     {1, 1}, {1, 1, 1, 1}, {1, 1}, {0, 0}, false)
};

// 3D conv shapes
vector<conv_param_t<3>> shapes_3d = {
  // MB, IC, OC, {IT, IH, IW}, G, {KT, KH, KW}, {stride_t, stride_h,
  // stride_w},
  // {pad_prev, pad_h_top, pad_w_left, pad_next, pad_h_bottom, pad_w_right},
  // ({dilation_t, dilation_h, dilation_w},
  // {output_padding_t, output_padding_h, output_padding_w}, tranpose)
  // Regular
  conv_param_t<3>(1, 64, 64, {8, 14, 14}, 1, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(1, 64, 64, {8, 14, 14}, 1, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}, {1, 1, 1}, {0, 0, 0}, true),
  conv_param_t<3>(1, 64, 64, {8, 14, 14}, 1, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}, {1, 1, 1}, {1, 1, 1}, true),
  //With dilations
  conv_param_t<3>(1, 64, 64, {8, 14, 14}, 1, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}, {2, 2, 2}),
  conv_param_t<3>(1, 64, 64, {8, 14, 14}, 1, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}, {2, 2, 2}, {0, 0, 0}, true),
  conv_param_t<3>(1, 64, 64, {8, 14, 14}, 1, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}, {2, 2, 2}, {1, 1, 1}, true),

  // Groupwise
  conv_param_t<3>(32, 192, 192, {2, 28, 28}, 96, {3, 3, 3},
      {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 192, 192, {1, 14, 14}, 96, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 384, 384, {1, 14, 14}, 192, {3, 3, 3},
      {1, 2, 2}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 384, 384, {1, 7, 7}, 192, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}),

  conv_param_t<3>(32, 16, 16, {4, 56, 56}, 8, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 16, 16, {2, 28, 28}, 8, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 32, 32, {4, 56, 56}, 16, {3, 3, 3},
      {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 32, 32, {2, 28, 28}, 16, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 32, 32, {2, 28, 28}, 16, {3, 3, 3},
      {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 32, 32, {1, 14, 14}, 16, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 128, 128, {2, 28, 28}, 32, {3, 3, 3},
      {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 128, 128, {1, 14, 14}, 32, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 256, 256, {1, 14, 14}, 64, {3, 3, 3},
      {1, 2, 2}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 256, 256, {1, 7, 7}, 64, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 128, 128, {2, 28, 28}, 32, {3, 3, 3},
      {2, 2, 2}, {1, 1, 1, 1, 1, 1}, {1, 1, 1}, {0, 0, 0}, true),
  conv_param_t<3>(32, 128, 128, {1, 14, 14}, 32, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}, {1, 1, 1}, {0, 0, 0}, true),
  conv_param_t<3>(32, 128, 128, {1, 14, 14}, 32, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}, {1, 1, 1}, {1, 1, 1}, true),

  // Depthwise
  conv_param_t<3>(1, 64, 64, {8, 14, 14}, 64, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(1, 144, 144, {4, 28, 28}, 144, {3, 5, 5},
      {1, 2, 2}, {1, 2, 2, 1, 2, 2}),
  conv_param_t<3>(1, 288, 288, {4, 14, 14}, 288, {3, 5, 5},
      {1, 1, 1}, {1, 2, 2, 1, 2, 2}),

  // Pointwise
  conv_param_t<3>(1, 128, 128, {8, 14, 14}, 1, {1, 1, 1},
      {1, 1, 1}, {0, 0, 0, 0})
};
// clang-format on

template <int SPATIAL_DIM, typename Acc_t>
void performance_test() {
  std::vector<char> llc;

  llc.resize(128 * 1024 * 1024, 1.0);

  constexpr int NWARMUP = 2;
  const int NITER = 10;

  string header = "MB, IC, OC, ";
  if (SPATIAL_DIM == 3) {
    header += "IT, ";
  }
  if (SPATIAL_DIM > 1) {
    header += "IH, ";
  }
  header += "IW, G, ";
  if (SPATIAL_DIM == 3) {
    header += "KT, ";
  }
  if (SPATIAL_DIM > 1) {
    header += "KH, ";
  }
  header += "KW, ";
  if (SPATIAL_DIM == 3) {
    header += "stride_t, ";
  }
  if (SPATIAL_DIM > 1) {
    header += "stride_h, ";
  }
  header += "stride_w, ";
  if (SPATIAL_DIM == 3) {
    header += "pad_t, ";
  }
  if (SPATIAL_DIM > 1) {
    header += "pad_h, ";
  }
  header += "pad_w, ";
  if (SPATIAL_DIM == 3) {
    header += "dilation_t, ";
  }
  if (SPATIAL_DIM > 1) {
    header += "dilation_h, ";
  }
  header += "dilation_w, ";
  if (SPATIAL_DIM == 3) {
    header += "output_padding_t, ";
  }
  if (SPATIAL_DIM > 1) {
    header += "output_padding_h, ";
  }
  header += "output_padding_w, ";
  header += "transposed, ";

  header += "Type, M, N, K, ";

  header += "#_ops, ";

  cout << setw(6) << header << setw(5) << "GOPS" << endl;

  chrono::time_point<chrono::high_resolution_clock> begin, end;

  auto conv_p = shapes_2d[CASE];
  if (conv_p.IC % conv_p.G != 0 || conv_p.OC % conv_p.G != 0) {
    // invalid shapes
    return;
  }
  int im_in_dim = accumulate(
      conv_p.IN_DIM.begin(), conv_p.IN_DIM.end(), 1, multiplies<int>());
  aligned_vector<uint8_t> Aint8(conv_p.MB * im_in_dim * conv_p.IC);

  int kernel_dim =
      accumulate(conv_p.K.begin(), conv_p.K.end(), 1, multiplies<int>());
  aligned_vector<int8_t> Bint8(
      kernel_dim * conv_p.IC * (conv_p.OC / conv_p.G));

  aligned_vector<int8_t> Bint8_tr(
      kernel_dim * conv_p.IC * (conv_p.OC / conv_p.G));

  int im_out_dim = accumulate(
      conv_p.OUT_DIM.begin(), conv_p.OUT_DIM.end(), 1, multiplies<int>());
  aligned_vector<int32_t> Cint32_ref(conv_p.MB * im_out_dim * conv_p.OC);
  aligned_vector<uint8_t> Cint8_ref(Cint32_ref.size(), 0);
  aligned_vector<int32_t> Cint32_fb(Cint32_ref.size());
  aligned_vector<uint8_t> Cint8_fb(Cint32_ref.size(), 0);
  aligned_vector<uint8_t> Cint8_fb2(Cint32_ref.size(), 0);
  aligned_vector<int32_t> Cint32_fb2(Cint32_ref.size());

  // A matrix (input activations)
  randFill<uint8_t>(Aint8, 0, 5);
  int32_t Aint8_zero_point = 4;

  // B matrix (weights)
  randFill<int8_t>(Bint8, -4, 4);
  aligned_vector<int32_t> Bint8_zero_point(1);
  randFill(Bint8_zero_point, -3, -1);

  aligned_vector<float> C_multiplier(Bint8_zero_point.size());
  randFill(C_multiplier, 0.1234f / 2, 0.1234f * 3 / 2);
  int32_t C_zero_point = 5;

  // matrix dimensions after im2col
  int MDim = conv_p.MB * im_out_dim;
  int NDim = conv_p.OC / conv_p.G;
  int KDim = kernel_dim * conv_p.IC;
  int KDimPerGroup = KDim / conv_p.G;

  int OC_per_G = conv_p.OC / conv_p.G;

  double nops = 2.0 * static_cast<double>(NITER) * MDim * NDim * KDim;
  double ttot = 0.0;
  string runType;

  PackWeightsForConv<SPATIAL_DIM> packedB(conv_p, Bint8.data());

  runType = "UniConv";
  ttot = 0;
  for (auto i = 0; i < NWARMUP + NITER; ++i) {
    llc_flush(llc);
    begin = chrono::high_resolution_clock::now();
    {
      int num_threads = 1;
      int tid = 0;
      // no-op output process objects
      DoNothing<> doNothingObj{};
      vector<int32_t> col_offsets(conv_p.OC);
      ReQuantizeOutput<false, QuantizationGranularity::TENSOR> outputProcObj(
          doNothingObj,
          C_multiplier.data(),
          C_zero_point,
          Aint8_zero_point,
          Bint8_zero_point.data(),
          nullptr, // row offsets
          col_offsets.data(),
          nullptr, // bias
          conv_p.OC,
          conv_p.G);

      fbgemmConv(
          conv_p,
          Aint8.data(),
          packedB,
          Cint8_fb.data(),
          Cint32_fb.data(),
          outputProcObj,
          tid,
          num_threads);
    }
    end = chrono::high_resolution_clock::now();

    if (i >= NWARMUP) {
      auto dur = chrono::duration_cast<chrono::nanoseconds>(end - begin);
      ttot += dur.count();
    }
  }

  cout << conv_p.MB << ", " << conv_p.IC << ", " << conv_p.OC << ", ";
  for (int i = 0; i < SPATIAL_DIM; ++i) {
    cout << conv_p.IN_DIM[i] << ", ";
  }
  cout << conv_p.G << ", ";
  for (int i = 0; i < SPATIAL_DIM; ++i) {
    cout << conv_p.K[i] << ", ";
  }
  for (int i = 0; i < SPATIAL_DIM; ++i) {
    cout << conv_p.stride[i] << ", ";
  }
  for (int i = 0; i < SPATIAL_DIM; ++i) {
    cout << conv_p.pad[i] << ", ";
  }
  for (int i = 0; i < SPATIAL_DIM; ++i) {
    cout << conv_p.dilation[i] << ", ";
  }
  for (int i = 0; i < SPATIAL_DIM; ++i) {
    cout << conv_p.output_pad[i] << ", ";
  }
  cout << conv_p.transposed;
  cout << setw(13) << ", " << runType << ", " << setw(5) << fixed << setw(5)
       << setw(6) << MDim << ", " << setw(6) << NDim << ", " << setw(6)
       << KDim << ", " << nops << ", ";
  cout << setprecision(2) << nops / ttot << endl;
}


int main(int argc, const char* argv[]) {
  performance_test<2, int32_t>();
  return 0;
}
