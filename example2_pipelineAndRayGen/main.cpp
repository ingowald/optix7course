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

#include "optix7.h"
// our helper library for window handling
#include "glfWindow/CUDAFrameBuffer.h"
#include "glfWindow/GLFWindow.h"
// common std stuff
#include <vector>

extern "C" char external_ptx_code[];
  
/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  struct LaunchParams
  {
    uint32_t colorBuffer;
    vec2i    fbSize;
  };
    
  struct CUDABuffer {
    inline CUdeviceptr d_pointer() const
    { return (CUdeviceptr)d_ptr; }

    void alloc(size_t size)
    {
      assert(d_ptr == nullptr);
      this->sizeInBytes = size;
      CUDA_CHECK(Malloc( (void**)&d_ptr, sizeInBytes));
      assert(valid());
    }

    void free()
    {
      CUDA_CHECK(Free(d_ptr));
      d_ptr = nullptr;
      sizeInBytes = 0;
    }
    
    template<typename T>
    void upload(const T *t, size_t count)
    {
      assert(d_ptr != nullptr);
      assert(sizeInBytes == count*sizeof(T));
      CUDA_CHECK(Memcpy(d_ptr, (void *)&t,
                        count*sizeof(T), cudaMemcpyHostToDevice));
    }
    
    size_t sizeInBytes { 0 };
    void  *d_ptr { nullptr };
  };

  struct SampleRenderer
  {
    SampleRenderer()
    {
      std::cout << "#osc: creating optix context ..." << std::endl;
      createContext();
      
      std::cout << "#osc: setting up optix pipeline ..." << std::endl;
      createPipeline();
    }
      
    void render()
    {
      launchParamsBuffer.upload(&launchParams,1);
      OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                              pipeline,stream,
                              /*! parameters and SBT */
                              launchParamsBuffer.d_pointer(),
                              launchParamsBuffer.sizeInBytes,
                              &sbt,
                              /*! dimensions of the launch: */
                              launchParams.fbSize.x,
                              launchParams.fbSize.y,
                              1
                              ));
    }

    void resize(const vec2i &newSize)
    {
      fb.resize(newSize);
    }

    static void context_log_cb(unsigned int level,
                               const char *tag,
                               const char *message,
                               void *)
    {
      fprintf( stderr, "[%2d][%12s]: %s\n", level, tag, message );
    }
  
    void createContext()
    {
      // for this sample, do everything on one device
      const int deviceID = 0;
      CUDA_CHECK(SetDevice(deviceID));
      CUDA_CHECK(StreamCreate(&stream));

      cudaGetDeviceProperties(&deviceProps, deviceID);
      std::cout << "#osc: running on device device: " << deviceProps.name << std::endl;
      
      CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
      if( cuRes != CUDA_SUCCESS ) 
        fprintf( stderr, "Error querying current context: error code %d\n", cuRes );
      
      OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
      OPTIX_CHECK(optixDeviceContextSetLogCallback
                  (optixContext,context_log_cb,nullptr,4));
    }

    void createModule()
    {
      const std::string ptxCode = external_ptx_code;
      OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                           &moduleCompileOptions,
                                           &pipelineCompileOptions,
                                           ptxCode.c_str(),
                                           ptxCode.size(),
                                           nullptr,      // Log string
                                           0,            // Log string sizse
                                           &module
                                           ));
    }
    
    void createPipeline()
    {
      pipelineLinkOptions.overrideUsesMotionBlur = false;
      pipelineLinkOptions.maxTraceDepth          = 2;
      
      std::vector<OptixProgramGroup> programGroups;
      
      char log[2048];
      size_t sizeof_log = sizeof( log );
      OPTIX_CHECK(optixPipelineCreate(optixContext,
                                      &pipelineCompileOptions,
                                      &pipelineLinkOptions,
                                      programGroups.data(),
                                      programGroups.size(),
                                      log,&sizeof_log,
                                      &pipeline
                                      ));
      OPTIX_CHECK(optixPipelineSetStackSize
                  (/* [in] The pipeline to configure the stack size for */
                   pipeline, 
                   /* [in] The direct stack size requirement for direct
                      callables invoked from IS or AH. */
                   2*1024,
                   /* [in] The direct stack size requirement for direct
                      callables invoked from RG, MS, or CH.  */                 
                   2*1024,
                   /* [in] The continuation stack requirement. */
                   2*1024,
                   /* [in] The maximum depth of a traversable graph
                      passed to trace. */
                   3));
    }
    
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
    
    OptixShaderBindingTable sbt = {};

    /*! @{ our launch parameters, on the host, and the buffer to store
        them on the device */
    LaunchParams launchParams;
    CUDABuffer   launchParamsBuffer;
    /*! @} */
    
    /*! our internal frame buffer */
    CUDAFrameBuffer fb;
  };
  
  struct SampleWindow : public GLFWindow
  {
    SampleWindow(const std::string &title)
      : GLFWindow(title)
    {}
    
    virtual void render() override
    {
      sample.render();
    }
    
    virtual void draw() override
    {
    }
    
    virtual void resize(const vec2i &newSize) 
    {
      sample.resize(newSize);
    }

    SampleRenderer sample;
  };
  
  
  /*! helper function that initializes optix, and checks for errors */
  void initOptix()
  {
    std::cout << "#osc: initializing optix..." << std::endl;
      
    // -------------------------------------------------------
    // check for available optix7 capable devices
    // -------------------------------------------------------
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
      throw std::runtime_error("#osc: no CUDA capable devices found!");
    std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;

    // -------------------------------------------------------
    // initialize optix
    // -------------------------------------------------------
    OPTIX_CHECK( optixInit() );
    std::cout << GDT_TERMINAL_GREEN
              << "#osc: successfully initialized optix... yay!"
              << GDT_TERMINAL_DEFAULT << std::endl;
  }
  
  /*! main entry point to this example - initialy optix, print hello
    world, then exit */
  extern "C" int main(int ac, char **av)
  {
    try {
      initOptix();
      
      SampleWindow *window = new SampleWindow("Optix 7 Course Example");
      window->run();
      
    } catch (std::runtime_error e) {
      std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
                << GDT_TERMINAL_DEFAULT << std::endl;
      exit(1);
    }
    return 0;
  }
  
} // ::osc
