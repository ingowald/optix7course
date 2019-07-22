// // ======================================================================================
// //  Copyright (c) 2019 NVIDIA Corporation.  All rights reserved.
// //
// //  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// //  rights in and to this software, related documentation and any modifications thereto.
// //  Any use, reproduction, disclosure or distribution of this software and related
// //  documentation without an express license agreement from NVIDIA Corporation is strictly
// //  prohibited.
// //
// //  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
// //  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
// //  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// //  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
// //  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
// //  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
// //  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
// //  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
// //  SUCH DAMAGES
// // ======================================================================================

// #pragma once

// #include "GLFWindow.h"

// /*! \namespace osc - Optix Siggraph Course */
// namespace osc {
//   using namespace gdt;
  
//   struct CUDAFrameBuffer
//   {
//     void resize(const vec2i &size)
//     {
//       if (d_colorBuffer) CUDA_CHECK(Free(d_colorBuffer));
//       if (h_colorBuffer) delete[] h_colorBuffer;

//       CUDA_CHECK(Malloc(&d_colorBuffer,size.x*size.y*sizeof(uint32_t)));
//       h_colorBuffer = new uint32_t[size.x*size.y];
//     }
//     ~CUDAFrameBuffer()
//     {
//       if (d_colorBuffer) CUDA_CHECK_NOEXCEPT(Free(d_colorBuffer));
//       if (h_colorBuffer) delete[] h_colorBuffer;
//     }

//     vec2i size              { 0,0 };
//     uint32_t *h_colorBuffer { nullptr };
//     uint32_t *d_colorBuffer { nullptr };
//   };
  
// } // ::osc
