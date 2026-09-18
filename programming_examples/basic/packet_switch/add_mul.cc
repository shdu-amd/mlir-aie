//===- add_mul.cc -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>



#if defined(__chess__)

#define AIE_BANK_A chess_storage(DM_bankA)
#define AIE_BANK_B chess_storage(DM_bankB)
#define AIE_BANK_C chess_storage(DM_bankC)
#define AIE_BANK_D chess_storage(DM_bankD)

#else

#define AIE_BANK_A __attribute__((section(".aie.bank0")))
#define AIE_BANK_B __attribute__((section(".aie.bank1")))
#define AIE_BANK_C __attribute__((section(".aie.bank2")))
#define AIE_BANK_D __attribute__((section(".aie.bank3")))

#endif

AIE_BANK_A alignas(aie::vector_decl_align) float sigmoid_lut_ab[128] = {
    0.00000000f, 0.00000000f, 0.02531559f, 0.11778884f,
    0.03202982f, 0.14126447f, 0.04036180f, 0.16831453f,
    0.00000000f, 0.00000000f, 0.02531559f, 0.11778884f,
    0.03202982f, 0.14126447f, 0.04036180f, 0.16831453f,
    0.05060361f, 0.19900633f, 0.06304193f, 0.23317364f,
    0.07791794f, 0.27032241f, 0.09536672f, 0.30954009f,
    0.05060361f, 0.19900633f, 0.06304193f, 0.23317364f,
    0.07791794f, 0.27032241f, 0.09536672f, 0.30954009f,
    0.11533491f, 0.34943751f, 0.13748258f, 0.38816571f,
    0.16108698f, 0.42355781f, 0.18498056f, 0.45343356f,
    0.11533491f, 0.34943751f, 0.13748258f, 0.38816571f,
    0.16108698f, 0.42355781f, 0.18498056f, 0.45343356f,
    0.20756739f, 0.47605875f, 0.22695885f, 0.49067320f,
    0.24124110f, 0.49791516f, 0.24883294f, 0.49993517f,
    0.20756739f, 0.47605875f, 0.22695885f, 0.49067320f,
    0.24124110f, 0.49791516f, 0.24883294f, 0.49993517f,
    0.24883488f, 0.50006461f, 0.24124668f, 0.50208276f,
    0.22696742f, 0.50932146f, 0.20757805f, 0.52393193f,
    0.24883488f, 0.50006461f, 0.24124668f, 0.50208276f,
    0.22696742f, 0.50932146f, 0.20757805f, 0.52393193f,
    0.18499233f, 0.54655320f, 0.16109897f, 0.57642570f,
    0.13749410f, 0.61181558f, 0.11534548f, 0.65054266f,
    0.18499233f, 0.54655320f, 0.16109897f, 0.57642570f,
    0.13749410f, 0.61181558f, 0.11534548f, 0.65054266f,
    0.09537609f, 0.69043999f, 0.07792602f, 0.72965841f,
    0.06304874f, 0.76680847f, 0.05060926f, 0.80097743f,
    0.09537609f, 0.69043999f, 0.07792602f, 0.72965841f,
    0.06304874f, 0.76680847f, 0.05060926f, 0.80097743f,
    0.04036642f, 0.83167103f, 0.03203356f, 0.85872291f,
    0.02531859f, 0.88220029f, 0.00000000f, 1.00000000f,
    0.04036642f, 0.83167103f, 0.03203356f, 0.85872291f,
    0.02531859f, 0.88220029f, 0.00000000f, 1.00000000f,
};

AIE_BANK_B alignas(aie::vector_decl_align) float sigmoid_lut_cd[128] = {
    0.00000000f, 0.00000000f, 0.02531559f, 0.11778884f,
    0.03202982f, 0.14126447f, 0.04036180f, 0.16831453f,
    0.00000000f, 0.00000000f, 0.02531559f, 0.11778884f,
    0.03202982f, 0.14126447f, 0.04036180f, 0.16831453f,
    0.05060361f, 0.19900633f, 0.06304193f, 0.23317364f,
    0.07791794f, 0.27032241f, 0.09536672f, 0.30954009f,
    0.05060361f, 0.19900633f, 0.06304193f, 0.23317364f,
    0.07791794f, 0.27032241f, 0.09536672f, 0.30954009f,
    0.11533491f, 0.34943751f, 0.13748258f, 0.38816571f,
    0.16108698f, 0.42355781f, 0.18498056f, 0.45343356f,
    0.11533491f, 0.34943751f, 0.13748258f, 0.38816571f,
    0.16108698f, 0.42355781f, 0.18498056f, 0.45343356f,
    0.20756739f, 0.47605875f, 0.22695885f, 0.49067320f,
    0.24124110f, 0.49791516f, 0.24883294f, 0.49993517f,
    0.20756739f, 0.47605875f, 0.22695885f, 0.49067320f,
    0.24124110f, 0.49791516f, 0.24883294f, 0.49993517f,
    0.24883488f, 0.50006461f, 0.24124668f, 0.50208276f,
    0.22696742f, 0.50932146f, 0.20757805f, 0.52393193f,
    0.24883488f, 0.50006461f, 0.24124668f, 0.50208276f,
    0.22696742f, 0.50932146f, 0.20757805f, 0.52393193f,
    0.18499233f, 0.54655320f, 0.16109897f, 0.57642570f,
    0.13749410f, 0.61181558f, 0.11534548f, 0.65054266f,
    0.18499233f, 0.54655320f, 0.16109897f, 0.57642570f,
    0.13749410f, 0.61181558f, 0.11534548f, 0.65054266f,
    0.09537609f, 0.69043999f, 0.07792602f, 0.72965841f,
    0.06304874f, 0.76680847f, 0.05060926f, 0.80097743f,
    0.09537609f, 0.69043999f, 0.07792602f, 0.72965841f,
    0.06304874f, 0.76680847f, 0.05060926f, 0.80097743f,
    0.04036642f, 0.83167103f, 0.03203356f, 0.85872291f,
    0.02531859f, 0.88220029f, 0.00000000f, 1.00000000f,
    0.04036642f, 0.83167103f, 0.03203356f, 0.85872291f,
    0.02531859f, 0.88220029f, 0.00000000f, 1.00000000f,
};

__attribute__((always_inline)) v16bfloat16 getsigmoidBf16(v16bfloat16 vInput) {
  aie::vector<bfloat16, 16> input = vInput;

  int step_bits = -2;
  int bias = 16;
  int data_size = 16;
  int LUT_elems = 32;
  int shift_offset = 0; // unused

  using lut_type = aie::lut<4, float, bfloat16>;

  lut_type test_lut(LUT_elems, (bfloat16 *)sigmoid_lut_ab, (bfloat16 *)sigmoid_lut_cd);

  aie::linear_approx<bfloat16, lut_type> lin_aprox(test_lut, step_bits, bias, shift_offset);

  aie::vector<bfloat16, 16> output = lin_aprox.compute(input).to_vector<bfloat16>();

  return (v16bfloat16)output;
}


template <typename T, int W = 128>
void mul_aie(T *restrict in0, T *restrict out) {
  event0();
  const int vec_factor = 16;

  aie::vector<T, vec_factor> In0;
  aie::accum<acc32, vec_factor> Out;

  const int F = W / vec_factor;
  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(6, ) {
      In0 = aie::load_v<vec_factor>(in0);
      Out = aie::mul(In0, (T)2);
      aie::store_v(out, Out.template to_vector<T>());
      in0 += vec_factor;
      out += vec_factor;
    }
  event1();
}

template <typename T, int W = 128>
void add_aie(T *restrict in0, T *restrict out) {
  event0();
  const int vec_factor = 16;

  aie::vector<T, vec_factor> In0;
  aie::vector<T, vec_factor> Out;

  const int F = W / vec_factor;
  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(6, ) {
      In0 = aie::load_v<vec_factor>(in0);
      Out = aie::add(In0, (T)2);
      aie::store_v(out, Out);
      in0 += vec_factor;
      out += vec_factor;
    }
  event1();
}

extern "C" {

#ifndef DIM_S
#define DIM_S 256
#endif

void add(int8 *restrict In0, int8 *restrict y) { add_aie<int8, DIM_S>(In0, y); }

void mul(int8 *restrict In0, int8 *restrict y) { mul_aie<int8, DIM_S>(In0, y); }
} // extern "C"
