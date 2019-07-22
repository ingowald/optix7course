// ======================================================================================
//  Copyright (c) 2019 NVIDIA Corporation.  All rights reserved.
//
//  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from NVIDIA Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
// ======================================================================================

#define OPTIX_COMPATIBILITY 7
#include <optix_device.h>

#include "LaunchParams.h"

namespace osc {
  
  extern "C" __global__ void __anyhit__radiance()
  {
    /*! for this simple example, this will remain empty */
  }
  
  extern "C" __global__ void __closesthit__radiance()
  {
    /*! for this simple example, this will remain empty */
  }
  
  /*! miss program. we don't actually use miss programs in this
      renderer, so this is empty */
  extern "C" __global__ void __miss__radiance()
  {
    /*! for this simple example, this will remain empty */
  }

  /*! launch parameters in constant memory, filled in by optix upon optixLaunch */
  extern "C" __constant__ LaunchParams optixLaunchParams;

  //------------------------------------------------------------------------------
  //
  // ray gen program - the actual rendering happens in here
  //
  //------------------------------------------------------------------------------
  extern "C" __global__ void __raygen__renderFrame()
  {
    if (optixLaunchParams.fbSize.x == 100000)
      optixLaunchParams.fbSize.x = 0;
    if (optixGetLaunchIndex().x == 0 &&
        optixGetLaunchIndex().y == 0) 
      printf("Hello world from OptiX 7 (%i)!\n",optixLaunchParams.fbSize.x);
  }
  
}
