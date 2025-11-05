/**
 * @file
 * @brief Conversions between data layouts and types for register tiles.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/* ----------  LAYOUT SWAPS  ---------- */

/**
 * @brief Swaps the layout of a register tile.
 *
 * This function swaps the layout of a register tile by iterating over its height and width
 * and performing layout swaps on each of its base elements.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the register tile.
 * @tparam _width The width of the register tile.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the destination register tile where the result will be stored.
 * @param src[in] Reference to the source register tile to be swapped.
 */
template<ducks::rt_layout::all layout1, ducks::rt_shape::all shape1, typename T, int _height, int _width, ducks::rt_layout::all layout2, ducks::rt_shape::all shape2>
__device__ static inline void swap_layout(rt<T, _height, _width, layout1, shape1> &dst, const rt<T, _height, _width, layout2, shape2> &src) {

    if constexpr (std::is_same_v<T, bf16>) {
        if constexpr (std::is_same_v<layout1, typename ducks::rt_layout::col> && std::is_same_v<layout2, typename ducks::rt_layout::col>) {
            // src consists of 16x16 tiles while dst consists of 16x32 tiles.
            // the reduction dimension (rows) stays the same, while the column dimension (cols) is doubled.
            // For every two 16x16 tiles in src along the (width) axis, we fill one 16x32 tile in dst along the (width) axis.
            // To do this for bf16, we issue 4 v_permlane16_swap instructions.
            if constexpr (std::is_same_v<shape1, typename ducks::rt_shape::rt_16x32> && std::is_same_v<shape2, typename ducks::rt_shape::rt_16x16>) {
                #pragma unroll
                for (int i = 0; i < dst.height; i++) {
                    #pragma unroll
                    for (int j = 0; j < dst.width; j++) {

                        // now we are at the granularity of a single 16x32 tile in dst.
                        // V_PERMLANE16_SWAP_B32:
                        // Swap data between two vector registers. Odd rows of the first operand are swapped with even rows of the
                        // second operand (one row is 16 lanes).
                        #pragma unroll
                        for (int k = 0; k < 2; k++) {
                            uint2_t res = __builtin_amdgcn_permlane16_swap(*reinterpret_cast<const uint32_t *>(&src.tiles[i][j * 2].data[k]), *reinterpret_cast<const uint32_t *>(&src.tiles[i][j * 2 + 1].data[k]), false, true);
                            *reinterpret_cast<uint32_t *>(&dst.tiles[i][j].data[k]) = res.x;
                            *reinterpret_cast<uint32_t *>(&dst.tiles[i][j].data[k + 2]) = res.y;
                        }
                    }
                }
            } else if constexpr (std::is_same_v<shape1, typename ducks::rt_shape::rt_16x32_4> && std::is_same_v<shape2, typename ducks::rt_shape::rt_32x32>) {
                #pragma unroll
                for (int i = 0; i < dst.height; i++) {
                    #pragma unroll
                    for (int j = 0; j < dst.width; j++) {
                        #pragma unroll
                        for (int k = 0; k < dst.packed_per_base_tile; k++) {
                            dst.tiles[i][j].data[k] = src.tiles[i / 2][j].data[(i % 2) * dst.packed_per_base_tile + k];
                        }
                    }
                }
            } else {
                static_assert(false, "Unsupported shape swap");
            }
        } else {
            static_assert(false, "Unsupported layout swap");
        }
    } else {
        static_assert(false, "Unsupported dtype");
    }
}

/**
 * @brief Swaps the layout of a register tile in place.
 *
 * This function swaps the layout of a register tile in place by iterating over its height and width
 * and performing in-place layout swaps on each of its base elements.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the register tile.
 * @tparam _width The width of the register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile to be swapped in place.
 * @return A reference to the swapped register tile.
 */
template<ducks::rt_layout::all layout1, ducks::rt_shape::all shape1, typename T, int _rows, int _cols, ducks::rt_layout::all layout2, ducks::rt_shape::all shape2>
__device__ static inline rt<T, _rows, _cols, layout1, shape1>& swap_layout_inplace(rt<T, _rows, _cols, layout2, shape2> &tile) {
    rt<T, _rows, _cols, layout1, shape1> &dst = *(rt<T, _rows, _cols, layout1, shape1>*)(&tile);
    if constexpr (std::is_same_v<T, bf16>) {
        if constexpr (std::is_same_v<layout1, typename ducks::rt_layout::col> && std::is_same_v<layout2, typename ducks::rt_layout::col>) {
            if constexpr (std::is_same_v<shape1, typename ducks::rt_shape::rt_16x32> && std::is_same_v<shape2, typename ducks::rt_shape::rt_16x16>) {
                swap_layout(dst, tile);
            } else {
                static_assert(false, "Unsupported shape swap");
            }
        } else {
            static_assert(false, "Unsupported layout swap");
        }
    } else {
        static_assert(false, "Unsupported dtype");
    }
    return dst;   
}

/* ----------  TRANSPOSE  ---------- */
template<typename T2, int _rows, int _cols, ducks::rt_layout::all layout, ducks::rt_shape::all shape>
__device__ static inline void transpose(rt<T2, _cols, _rows, typename ducks::rt_layout::transpose<layout>::type, typename ducks::rt_shape::transpose<shape>::type> &result, const rt<T2, _rows, _cols, layout, shape> &tile) {
    #pragma unroll
    for (int i = 0; i < tile.height; i++) {
        #pragma unroll
        for (int j = 0; j < tile.width; j++) {
            #pragma unroll
            for (int k = 0; k < tile.packed_per_base_tile; k++) {
                // result.tiles[j][i].data[k] = tile.tiles[i][j].data[k];

                // This generates fewer v_bfi_b32 under AMD beta docker.
                __builtin_memcpy(&result.tiles[j][i].data[k],
                    &tile.tiles[i][j].data[k],
                    sizeof(tile.tiles[i][j].data[k]));
            }
        }
    }
}

/* ----------  TYPE SWAPS  ---------- */

/**
 * @brief Copies a register base tile, converting the underlying type if necessary.
 *
 * @tparam T2 The data type of the destination register elements.
 * @tparam U2 The data type of the source register elements.
 * @tparam layout The current layout of the register base tile.
 * @param[out] dst A reference to the destination register base tile.
 * @param[in] src A reference to the source register base tile.
 */
template<typename T, typename U, ducks::rt_layout::all layout, ducks::rt_shape::all shape>
__device__ static inline void copy(rt_base<T, layout, shape> &dst, const rt_base<U, layout, shape> &src) {
    using T2 = typename base_types::packing<T>::packed_type;
    using U2 = typename base_types::packing<U>::packed_type;
    #pragma unroll
    for(int k = 0; k < dst.packed_per_thread; k++) {
        dst.data[k] = base_types::convertor<T2, U2>::convert(src.data[k]);
    }
}

/**
 * @brief Copies a register tile, converting the underlying type if necessary.
 *
 * @tparam T2 The data type of the destination register elements.
 * @tparam U2 The data type of the source register elements.
 * @tparam _height The height (in units of 16) of the register tiles.
 * @tparam _width The width (in units of 16) of the register tiles.
 * @tparam layout The current layout of the register tile.
 * @param[out] dst A reference to the destination register tile.
 * @param[in] src A reference to the source register tile.
 */
template<typename T2, typename U2, int _height, int _width, ducks::rt_layout::all layout, ducks::rt_shape::all shape>
__device__ static inline void copy(rt<T2, _height, _width, layout, shape> &dst, const rt<U2, _height, _width, layout, shape> &src) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            copy(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
} 

/* ----------  CAUSAL  ---------- */

/**
 * @brief Makes a square register tile causal by zeroing elements above the main diagonal.
 *
 * This function modifies a square register tile in-place to make it causal. All elements
 * above the main diagonal are set to zero, while elements on or below the main diagonal
 * are left unchanged.
 *
 * @tparam T The data type of the register tile elements.
 * @tparam _size The size (height and width) of the square register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile to be made causal.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void make_causal(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    int lane = laneid();
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if(j < i) { // below the diagonal, copy
                #pragma unroll
                for(int k = 0; k < dst.packed_per_base_tile; k++) {
                    dst.tiles[i][j].data[k] = src.tiles[i][j].data[k];
                }
            }
            else if(j > i) { // above the diagonal, zero
                #pragma unroll
                for(int k = 0; k < dst.packed_per_base_tile; k++) {
                    dst.tiles[i][j].data[k] = packed_val;
                }
            }
            else { // on the diagonal, interesting!
                constexpr uint64_t MASKS[16] = {0xFFFFFF00FFFFFFFF, 0xFFFFFE00FFFFFFFE,
                                                0xFFFFFC00FFFFFFFC, 0xFFFFF800FFFFFFF8,
                                                0xFFFFF000FFFFFFF0, 0xFFFFE000FFFFFFE0,
                                                0xFFFFC000FFFFFFC0, 0xFFFF8000FFFFFF80,
                                                0xFF000000FFFF0000, 0xFE000000FFFE0000,
                                                0xFC000000FFFC0000, 0xF8000000FFF80000,
                                                0xF0000000FFF00000, 0xE0000000FFE00000,
                                                0xC0000000FFC00000, 0x80000000FF800000};

                #pragma unroll
                for(int k = 0; k < dst.packed_per_base_tile; k++) {
                    if ((MASKS[k * 2] >> lane) & 1) {
                        dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x;
                    }
                    else {
                        dst.tiles[i][j].data[k].x = val;
                    }
                    if ((MASKS[k * 2 + 1] >> laneid()) & 1) {
                        dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y;
                    }
                    else {
                        dst.tiles[i][j].data[k].y = val;
                    }
                }
            }
            // __syncwarp();
        }
    }
}

/**
 * @brief Makes a square register tile anti-causal by zeroing elements below the main diagonal.
 *
 * This function modifies a square register tile in-place to make it anti-causal. All elements
 * below the main diagonal are set to zero, while elements on or above the main diagonal
 * are left unchanged.
 *
 * @tparam T The data type of the register tile elements.
 * @tparam _size The size (height and width) of the square register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile to be made causal.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void make_causal_t(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if(j > i) { // above the diagonal, copy
                #pragma unroll
                for(int k = 0; k < dst.packed_per_base_tile; k++) {
                    dst.tiles[i][j].data[k] = src.tiles[i][j].data[k];
                }
            }
            else if(j < i) { // below the diagonal, zero
                #pragma unroll
                for(int k = 0; k < dst.packed_per_base_tile; k++) {
                    dst.tiles[i][j].data[k] = packed_val;
                }
            }
            else { // on the diagonal, interesting!
                constexpr uint64_t MASKS[16] = {
                    0x000001FF00000001,  // ~0xFFFFFF00FFFFFFFF
                    0x000003FF00000003,  // ~0xFFFFFE00FFFFFFFE
                    0x000007FF00000007,  // ~0xFFFFFC00FFFFFFFC
                    0x00000FFF0000000F,  // ~0xFFFFF800FFFFFFF8
                    0x00001FFF0000001F,  // ~0xFFFFF000FFFFFFF0
                    0x00003FFF0000003F,  // ~0xFFFFE000FFFFFFE0
                    0x00007FFF0000007F,  // ~0xFFFFC000FFFFFFC0
                    0x0000FFFF000000FF,  // ~0xFFFF8000FFFFFF80
                    0x01FFFFFF0001FFFF,  // ~0xFF000000FFFF0000
                    0x03FFFFFF0003FFFF,  // ~0xFE000000FFFE0000
                    0x07FFFFFF0007FFFF,  // ~0xFC000000FFFC0000
                    0x0FFFFFFF000FFFFF,  // ~0xF8000000FFF80000
                    0x1FFFFFFF001FFFFF,  // ~0xF0000000FFF00000
                    0x3FFFFFFF003FFFFF,  // ~0xE0000000FFE00000
                    0x7FFFFFFF007FFFFF,  // ~0xC0000000FFC00000
                    0xFFFFFFFF00FFFFFF   // ~0x80000000FF800000
                };

                #pragma unroll
                for(int k = 0; k < dst.packed_per_base_tile; k++) {
                    if ((MASKS[k * 2] >> laneid()) & 1) {
                        dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x;
                    }
                    else {
                        dst.tiles[i][j].data[k].x = val;
                    }
                    if ((MASKS[k * 2 + 1] >> laneid()) & 1) {
                        dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y;
                    }
                    else {
                        dst.tiles[i][j].data[k].y = val;
                    }
                }
                
            }
            // __syncwarp();
        }
    }
}

/* ----------  TRIANGULAR FILLS  ---------- */

/**
 * @brief Makes a register tile triangular by zeroing elements above the row index
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to triangularize from.
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void tril(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    const int lane = laneid();
    const int row = lane % 32;
    const int col = 8 * (lane / 32);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_base_tile; k++) {
                const int global_row_idx   = (i * dst.tile_size_row) + row;
                const int global_col_idx_x = (j * dst.tile_size_col) + col + (16 * (k / (dst.packed_per_base_tile / 2))) + 2 * (k % (dst.packed_per_base_tile / 2));
                const int global_col_idx_y = (j * dst.tile_size_col) + col + (16 * (k / (dst.packed_per_base_tile / 2))) + 2 * (k % (dst.packed_per_base_tile / 2)) + 1;

                if (global_row_idx < row_idx) { dst.tiles[i][j].data[k] = packed_val; }
                else {
                    if (global_col_idx_x <= global_row_idx - row_idx) { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                    else                                              { dst.tiles[i][j].data[k].x = val; }

                    if (global_col_idx_y <= global_row_idx - row_idx) { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                    else                                              { dst.tiles[i][j].data[k].y = val; }
                }
            }
        }
        // __syncwarp();
    }
}

template<ducks::rt::col_layout RT>
__device__ static inline void tril(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    
    const int lane = laneid();
    const int row = 8 * (lane / 32);
    const int col = lane % 32;
    
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_base_tile; k++) {
                const int global_row_idx_x = (i * dst.tile_size_row) + row + (16 * (k / (dst.packed_per_base_tile / 2))) + 2 * (k % (dst.packed_per_base_tile / 2));
                const int global_row_idx_y = (i * dst.tile_size_row) + row + (16 * (k / (dst.packed_per_base_tile / 2))) + 2 * (k % (dst.packed_per_base_tile / 2)) + 1;
                const int global_col_idx   = (j * dst.tile_size_col) + col;

                if (global_row_idx_x < row_idx) { dst.tiles[i][j].data[k].x = val; }
                else { 
                    if (global_col_idx <= global_row_idx_x - row_idx) { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                    else                                              { dst.tiles[i][j].data[k].x = val; }
                }

                if (global_row_idx_y < row_idx) { dst.tiles[i][j].data[k].y = val; }
                else { 
                    if (global_col_idx <= global_row_idx_y - row_idx) { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                    else                                              { dst.tiles[i][j].data[k].y = val; }
                }
            }
        }
        // __syncwarp();
    }
}

/**
 * @brief Makes a register tile triangular by zeroing elements below the row index
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to triangularize from.
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void triu(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    const int lane = laneid();
    const int row = lane % 32;
    const int col = 8 * (lane / 32);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_base_tile; k++) {
                const int global_row_idx   = (i * dst.tile_size_row) + row;
                const int global_col_idx_x = (j * dst.tile_size_col) + col + (16 * (k / (dst.packed_per_base_tile / 2))) + 2 * (k % (dst.packed_per_base_tile / 2));
                const int global_col_idx_y = (j * dst.tile_size_col) + col + (16 * (k / (dst.packed_per_base_tile / 2))) + 2 * (k % (dst.packed_per_base_tile / 2)) + 1;

                if (global_row_idx < row_idx) { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
                else {
                    if (global_col_idx_x < global_row_idx - row_idx) { dst.tiles[i][j].data[k].x = val; }
                    else                                             { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }

                    if (global_col_idx_y < global_row_idx - row_idx) { dst.tiles[i][j].data[k].y = val; }
                    else                                             { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                }
            }
        }
        // __syncwarp();
    }
}

template<ducks::rt::col_layout RT>
__device__ static inline void triu(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    
    const int lane = laneid();
    const int row = 8 * (lane / 32);
    const int col = lane % 32;

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_base_tile; k++) {
                const int global_row_idx_x = (i * dst.tile_size_row) + row + (16 * (k / (dst.packed_per_base_tile / 2))) + 2 * (k % (dst.packed_per_base_tile / 2));
                const int global_row_idx_y = (i * dst.tile_size_row) + row + (16 * (k / (dst.packed_per_base_tile / 2))) + 2 * (k % (dst.packed_per_base_tile / 2)) + 1;
                const int global_col_idx   = (j * dst.tile_size_col) + col;

                if (global_row_idx_x < row_idx) { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                else                            { 
                    if (global_col_idx < global_row_idx_x - row_idx) { dst.tiles[i][j].data[k].x = val; }
                    else                                             { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                }

                if (global_row_idx_y < row_idx) { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                else                            { 
                    if (global_col_idx < global_row_idx_y - row_idx) { dst.tiles[i][j].data[k].y = val; }
                    else                                             { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                }
            }
        }
        // __syncwarp();
    }
}

/* ----------  RECTANGULAR FILLS  ---------- */

/**
 * @brief Makes a register tile right filled with a given value.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param col_idx[in] The column index to fill from and onwards to the right.
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void right_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(col_idx >= dst.cols) return;

    const int col = 8 * (laneid() / 32);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_base_tile; k++) {
                const int col_idx_x = (j * dst.tile_size_col) + col + (16 * (k / (dst.packed_per_base_tile / 2))) + 2 * (k % (dst.packed_per_base_tile / 2));
                const int col_idx_y = (j * dst.tile_size_col) + col + (16 * (k / (dst.packed_per_base_tile / 2))) + 2 * (k % (dst.packed_per_base_tile / 2)) + 1;
                if (col_idx_x >= col_idx)  { dst.tiles[i][j].data[k].x = val; }
                else                       { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                if (col_idx_y >= col_idx)  { dst.tiles[i][j].data[k].y = val; }
                else                       { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
    }
}

template<ducks::rt::col_layout RT>
__device__ static inline void right_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    
    const int col = laneid() % 32;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_base_tile; k++) {
                const int t_col_idx = (j * dst.tile_size_col) + col; 
                if (t_col_idx >= col_idx)  { dst.tiles[i][j].data[k] = packed_val; }
                else                       { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
        // __syncwarp();
    }
}

/**
 * @brief Makes a register tile left filled with a given value.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param col_idx[in] The column index to fill to the left (exclusive).
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void left_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(col_idx <= 0) return;

    const int col = 8 * (laneid() / 32);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_base_tile; k++) {
                const int col_idx_x = (j * dst.tile_size_col) + col + (16 * (k / (dst.packed_per_base_tile / 2))) + 2 * (k % (dst.packed_per_base_tile / 2));
                const int col_idx_y = (j * dst.tile_size_col) + col + (16 * (k / (dst.packed_per_base_tile / 2))) + 2 * (k % (dst.packed_per_base_tile / 2)) + 1;
                if (col_idx_x < col_idx)  { dst.tiles[i][j].data[k].x = val; }
                else                       { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                if (col_idx_y < col_idx)  { dst.tiles[i][j].data[k].y = val; }
                else                       { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
    }
}

template<ducks::rt::col_layout RT>
__device__ static inline void left_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    const int col = laneid() % 32;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_base_tile; k++) {
                const int thread_col = (j * dst.tile_size_col) + col;
                if (thread_col < col_idx)  { dst.tiles[i][j].data[k] = packed_val; }
                else                       { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
        // __syncwarp();
    }
}

/**
 * @brief Makes a register tile upper filled with a given value.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to fill to, from the top (exclusive).
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void upper_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(row_idx <= 0) return;
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    const int row = laneid() % 32;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_base_tile; k++) {
                const int thread_row = (i * dst.tile_size_row) + row;
                if (thread_row < row_idx)  { dst.tiles[i][j].data[k] = packed_val; }
                else                       { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
    }
}

template<ducks::rt::col_layout RT>
__device__ static inline void upper_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const int row = 8 * (laneid() / 32);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_base_tile; k++) {
                const int row_idx_x = (i * dst.tile_size_row) + row + (16 * (k / (dst.packed_per_base_tile / 2))) + 2 * (k % (dst.packed_per_base_tile / 2));
                const int row_idx_y = (i * dst.tile_size_row) + row + (16 * (k / (dst.packed_per_base_tile / 2))) + 2 * (k % (dst.packed_per_base_tile / 2)) + 1;
                if (row_idx_x < row_idx)  { dst.tiles[i][j].data[k].x = val; }
                else                      { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                if (row_idx_y < row_idx)  { dst.tiles[i][j].data[k].y = val; }
                else                      { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
    }
}

/**
 * @brief Makes a register tile lower filled with a given value.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to fill from and onwards to the bottom of the tile (inclusive).
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void lower_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(row_idx >= dst.rows) return;
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    const int row = laneid() % 32;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_base_tile; k++) {
                const int thread_row = (i * dst.tile_size_row) + row;
                if (thread_row >= row_idx)  { dst.tiles[i][j].data[k] = packed_val; }
                else                        { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
    }
}


template<ducks::rt::col_layout RT>
__device__ static inline void lower_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const int row = 8 * (laneid() / 32);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_base_tile; k++) {
                const int row_idx_x = (i * dst.tile_size_row) + row + (16 * (k / (dst.packed_per_base_tile / 2))) + 2 * (k % (dst.packed_per_base_tile / 2));
                const int row_idx_y = (i * dst.tile_size_row) + row + (16 * (k / (dst.packed_per_base_tile / 2))) + 2 * (k % (dst.packed_per_base_tile / 2)) + 1;
                if (row_idx_x >= row_idx)  { dst.tiles[i][j].data[k].x = val; }
                else                       { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                if (row_idx_y >= row_idx)  { dst.tiles[i][j].data[k].y = val; }
                else                       { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
    }
}

/* ----------  SUBTILE  ---------- */

/**
* @brief Returns a reference to a subtile of the given tile.
*
* @tparam subtile_height The height of the subtile.
* @tparam RT The type of the input tile, which must satisfy the ducks::rt::all concept.
* @param src The input tile.
* @param idx The coord of the subtile.
* @return A reference to the subtile.
*
* @note The subtile height must evenly divide the tile height.
*/
template<int subtile_rows, ducks::rt::all RT>
__device__ inline rt<typename RT::T, subtile_rows, RT::cols, typename RT::layout, typename RT::shape> &subtile_inplace(RT & src, int idx) {
    using T = typename RT::T;
    static_assert(RT::rows % (subtile_rows / RT::base_tile_rows) == 0, "subtile height should evenly divide tile height.");
    return reinterpret_cast<rt<typename RT::T, subtile_rows, RT::cols, typename RT::layout, typename RT::shape>&>(
        src.tiles[idx*(subtile_rows / RT::base_tile_rows)]
    );
}

}
