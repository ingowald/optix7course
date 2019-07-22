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

#pragma once

// our own classes, partly shared between host and device
#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "gdt/math/AffineSpace.h"

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  struct Camera {
    /*! camera position - *from* where we are looking */
    vec3f from;
    /*! which point we are looking *at* */
    vec3f at;
    /*! general up-vector */
    vec3f up;
  };
  
  /*! a simple indexed triangle mesh that our sample renderer will
      render */
  struct TriangleMesh {
    /*! add a unit cube (subject to given xfm matrix) to the current
        triangleMesh */
    void addUnitCube(const affine3f &xfm);
    
    //! add aligned cube aith front-lower-left corner and size
    void addCube(const vec3f &center, const vec3f &size);
    
    std::vector<vec3f> vertex;
    std::vector<vec3i> index;
  };
  
  /*! a sample OptiX-7 renderer that demonstrates how to set up
      context, module, programs, pipeline, SBT, etc, and perform a
      valid launch that renders some pixel (using a simple test
      pattern, in this case */
  class SampleRenderer
  {
    // ------------------------------------------------------------------
    // publicly accessible interface
    // ------------------------------------------------------------------
  public:
    /*! constructor - performs asll setup, inlucuding initializing
      optix, creates module, pipeline, programs, SBT, etc. */
    SampleRenderer(const TriangleMesh &model);

    /*! render one frame */
    void render();

    /*! resize frame buffer to given resolution */
    void resize(const vec2i &newSize);

    /*! download the rendered color buffer */
    void downloadPixels(uint32_t h_pixels[]);

    /*! set camera to render with */
    void setCamera(const Camera &camera);
  protected:
    // ------------------------------------------------------------------
    // internal helper functions
    // ------------------------------------------------------------------

    /*! helper function that initializes optix, and checks for errors */
    void initOptix();
  
    /*! creates and configures a optix device context (in this simple
      example, only for the primary GPU device) */
    void createContext();

    /*! creates the module that contains all the programs we are going
      to use. in this simple example, we use a single module from a
      single .cu file, using a single embedded ptx string */
    void createModule();
    
    /*! does all setup for the raygen program(s) we are going to use */
    void createRaygenPrograms();
    
    /*! does all setup for the miss program(s) we are going to use */
    void createMissPrograms();
    
    /*! does all setup for the hitgroup program(s) we are going to use */
    void createHitgroupPrograms();

    /*! assembles the full pipeline of all programs */
    void createPipeline();

    /*! constructs the shader binding table */
    void buildSBT();

    /*! build an acceleration structure for the given triangle mesh */
    OptixTraversableHandle buildAccel(const TriangleMesh &model);

  protected:
    /*! @{ CUDA device context and stream that optix pipeline will run
        on, as well as device properties for this device */
    CUcontext          cudaContext;
    CUstream           stream;
    cudaDeviceProp     deviceProps;
    /*! @} */

    //! the optix context that our pipeline will run in.
    OptixDeviceContext optixContext;

    /*! @{ the pipeline we're building */
    OptixPipeline               pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions;
    OptixPipelineLinkOptions    pipelineLinkOptions;
    /*! @} */

    /*! @{ the module that contains out device programs */
    OptixModule                 module;
    OptixModuleCompileOptions   moduleCompileOptions;
    /* @} */

    /*! vector of all our program(group)s, and the SBT built around
        them */
    std::vector<OptixProgramGroup> raygenPGs;
    CUDABuffer raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    CUDABuffer missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    CUDABuffer hitgroupRecordsBuffer;
    OptixShaderBindingTable sbt = {};

    /*! @{ our launch parameters, on the host, and the buffer to store
        them on the device */
    LaunchParams launchParams;
    CUDABuffer   launchParamsBuffer;
    /*! @} */

    CUDABuffer colorBuffer;

    /*! the camera we are to render with. */
    Camera lastSetCamera;
    
    /*! the model we are going to trace rays against */
    const TriangleMesh model;
    CUDABuffer vertexBuffer;
    CUDABuffer indexBuffer;
    //! buffer that keeps the (final, compacted) accel structure
    CUDABuffer asBuffer;
  };

} // ::osc
