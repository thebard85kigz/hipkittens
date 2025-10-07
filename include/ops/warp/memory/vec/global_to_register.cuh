/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/**
 * @brief Load data into a register vector from a source array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the source array.
 * @param[out] dst The destination register vector to load data into.
 * @param[in] src The source array in global memory to load data from.
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void load(RV &dst, const GL &src, const COORD &idx) {
    using T2 = RV::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    U *src_ptr = (U*)&src[(idx.template unit_coord<-1, 3>())];
    int laneid = ::kittens::laneid();
    
    // TODO: this uses no inter-thread communication and is therefore not optimal.
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int idx = w*RV::reductions + RV::packed_stride*(laneid/RV::aligned_threads);
            // this should be a maximally coalesced load.
            #pragma unroll
            for(int i = 0; i < RV::strides_per_tile; i++) {
                #pragma unroll
                for(int j = 0; j < RV::packed_per_stride; j++) {
                    dst[w][i * RV::packed_per_stride + j] = 
                        base_types::convertor<T2, U2>::convert(*(U2*)&src_ptr[idx + i * RV::elements_per_stride_group + j * RV::packing]);
                }
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(auto w = 0; w < (dst.outer_dim+1)/2; w++) {
            int idx = w*kittens::WARP_THREADS + laneid;
            int o_dim = w*2;
            // this should be a maximally coalesced load.
            if(idx < dst.length) {
                dst[o_dim][0] = base_types::convertor<T, U>::convert(src_ptr[idx]);
            }
        }


        #pragma unroll
        for(auto w = 0; w < dst.outer_dim/2; w++) {
            const int o_dim = w*2;
            const int other_o_dim = o_dim + 1;
            if constexpr (std::is_same_v<T, float>) {
                uint2_t res = __builtin_amdgcn_permlane32_swap(__float_as_uint(dst[o_dim][0]), __float_as_uint(dst[o_dim][0]), false, true);
                dst[o_dim][0] = __uint_as_float(res.x);
                dst[other_o_dim][0] = __uint_as_float(res.y);
            }
            else if constexpr (std::is_same_v<T, bf16>) {
                uint2_t res = __builtin_amdgcn_permlane32_swap(__bfloat16_as_ushort(dst[o_dim][0]), __bfloat16_as_ushort(dst[o_dim][0]), false, true);
                dst[o_dim][0] = __ushort_as_bfloat16(res.x);
                dst[other_o_dim][0] = __ushort_as_bfloat16(res.y);
            }
            else if constexpr (std::is_same_v<T, half>) {
                uint2_t res = __builtin_amdgcn_permlane32_swap(__half_as_ushort(dst[o_dim][0]), __half_as_ushort(dst[o_dim][0]), false, true);
                dst[o_dim][0] = __ushort_as_half(res.x);
                dst[other_o_dim][0] = __ushort_as_half(res.y);
            } else {
                static_assert(false, "Unsupported type");
            }
        }

        if constexpr (RV::outer_dim % 2 == 1) {
            const int o_dim = dst.outer_dim - 1;
            if constexpr (std::is_same_v<T, float>) {
                uint2_t res = __builtin_amdgcn_permlane32_swap(__float_as_uint(dst[o_dim][0]), __float_as_uint(dst[o_dim][0]), false, true);
                dst[o_dim][0] = __uint_as_float(res.x);
            }
            else if constexpr (std::is_same_v<T, bf16>) {
                uint2_t res = __builtin_amdgcn_permlane32_swap(__bfloat16_as_ushort(dst[o_dim][0]), __bfloat16_as_ushort(dst[o_dim][0]), false, true);
                dst[o_dim][0] = __ushort_as_bfloat16(res.x);
            }
            else if constexpr (std::is_same_v<T, half>) {
                uint2_t res = __builtin_amdgcn_permlane32_swap(__half_as_ushort(dst[o_dim][0]), __half_as_ushort(dst[o_dim][0]), false, true);
                dst[o_dim][0] = __ushort_as_half(res.x);
            } else {
                static_assert(false, "Unsupported type");
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int idx = w*kittens::WARP_THREADS + laneid;
            if(idx < dst.length) {
                dst[w][0] = base_types::convertor<T, U>::convert(src_ptr[idx]);
            }
        }
    }
}

/**
 * @brief Store data from a register vector to a destination array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register vector to store data from.
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void store(const GL &dst, const RV &src, const COORD &idx) {
    using T2 = RV::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;
    
    U *dst_ptr = (U*)&dst[(idx.template unit_coord<-1, 3>())];
    int laneid = ::kittens::laneid();
    
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+3)/4; w++) {
            int idx = w*2*kittens::WARP_THREADS + 16*((laneid%32)/4) + 8*(laneid/32) + 2*(laneid%4);
            int o_dim = w*4 + ((laneid%32)/8);
            int i_dim = (laneid%8);
            // this should be a maximally coalesced store. I hope!
            if(idx < src.length)
                *(U2*)&dst_ptr[idx] = base_types::convertor<U2, T2>::convert(src[o_dim][i_dim]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+1)/2; w++) {
            int idx = w*kittens::WARP_THREADS + laneid;
            int o_dim = w*2 + laneid/32;
            // this should be a maximally coalesced load.
            if(idx < src.length) {
                dst_ptr[idx] = base_types::convertor<U, T>::convert(src[o_dim][0]);
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(auto w = 0; w < src.outer_dim; w++) {
            int idx = w*64 + laneid;
            if(idx < src.length) {
                dst_ptr[idx] = base_types::convertor<U, T>::convert(src[w][0]);
            }
        }
    }
}

} // namespace kittens