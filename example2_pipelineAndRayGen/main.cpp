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
// our own classes, partly shared between host and device
#include "LaunchParams.h"
// common std stuff
#include <vector>

extern "C" char embedded_ptx_code[];

/*! SBT record for a raygen program */
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
{
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  // just a dummy value - later examples will use more interesting
  // data here
  void *data;
};

/*! SBT record for a miss program */
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
{
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  // just a dummy value - later examples will use more interesting
  // data here
  void *data;
};

/*! SBT record for a hitgroup program */
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
{
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  // just a dummy value - later examples will use more interesting
  // data here
  int objectID;
};


/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  struct CUDABuffer {
    inline CUdeviceptr d_pointer() const
    { return (CUdeviceptr)d_ptr; }

    void alloc(size_t size)
    {
      assert(d_ptr == nullptr);
      this->sizeInBytes = size;
      CUDA_CHECK(Malloc( (void**)&d_ptr, sizeInBytes));
    }

    void free()
    {
      CUDA_CHECK(Free(d_ptr));
      d_ptr = nullptr;
      sizeInBytes = 0;
    }

    template<typename T>
    void alloc_and_upload(const std::vector<T> &vt)
    {
      alloc(vt.size()*sizeof(T));
      upload((const T*)vt.data(),vt.size());
    }
    
    template<typename T>
    void upload(const T *t, size_t count)
    {
      assert(d_ptr != nullptr);
      assert(sizeInBytes == count*sizeof(T));
      CUDA_CHECK(Memcpy(d_ptr, (void *)t,
                        count*sizeof(T), cudaMemcpyHostToDevice));
    }
    
    size_t sizeInBytes { 0 };
    void  *d_ptr { nullptr };
  };

  struct SampleRenderer
  {
    SampleRenderer()
    {
      initOptix();
      
      std::cout << "#osc: creating optix context ..." << std::endl;
      createContext();
      
      std::cout << "#osc: setting up module ..." << std::endl;
      createModule();

      std::cout << "#osc: creating raygen programs ..." << std::endl;
      createRaygenPrograms();
      std::cout << "#osc: creating miss programs ..." << std::endl;
      createMissPrograms();
      std::cout << "#osc: creating hitgroup programs ..." << std::endl;
      createHitgroupPrograms();

      std::cout << "#osc: setting up optix pipeline ..." << std::endl;
      createPipeline();

      std::cout << "#osc: building SBT ..." << std::endl;
      buildSBT();

      launchParamsBuffer.alloc(sizeof(launchParams));
      std::cout << "#osc: context, module, pipline, etc, all set up ..." << std::endl;

      std::cout << GDT_TERMINAL_GREEN;
      std::cout << "#osc: Optix 7 Sample fully set up" << std::endl;
      std::cout << GDT_TERMINAL_DEFAULT;
    }

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
  
    void buildSBT()
    {
      // ------------------------------------------------------------------
      // build raygen records
      // ------------------------------------------------------------------
      std::vector<RaygenRecord> raygenRecords;
      for (int i=0;i<raygenPGs.size();i++) {
        RaygenRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i],&rec));
        rec.data = nullptr; /* for now ... */
        raygenRecords.push_back(rec);
      }
      raygenRecordsBuffer.alloc_and_upload(raygenRecords);
      sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

      // ------------------------------------------------------------------
      // build miss records
      // ------------------------------------------------------------------
      std::vector<MissRecord> missRecords;
      for (int i=0;i<missPGs.size();i++) {
        MissRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i],&rec));
        rec.data = nullptr; /* for now ... */
        missRecords.push_back(rec);
      }
      missRecordsBuffer.alloc_and_upload(missRecords);
      sbt.missRecordBase          = missRecordsBuffer.d_pointer();
      sbt.missRecordStrideInBytes = sizeof(MissRecord);
      sbt.missRecordCount         = missRecords.size();

      // ------------------------------------------------------------------
      // build hitgroup records
      // ------------------------------------------------------------------

      // we don't actually have any objects in this example, but let's
      // create a dummy one so the SBT doesn't have any null pointers
      // (which the sanity checks in compilation would compain about)
      int numObjects = 1;
      std::vector<HitgroupRecord> hitgroupRecords;
      for (int i=0;i<numObjects;i++) {
        int objectType = 0;
        HitgroupRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[objectType],&rec));
        rec.objectID = i;
        hitgroupRecords.push_back(rec);
      }
      hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
      sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
      sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
      sbt.hitgroupRecordCount         = hitgroupRecords.size();
    }
    
    void createRaygenPrograms()
    {
      // we do a single ray gen program in this example:
      raygenPGs.resize(1);
      
      OptixProgramGroupOptions pgOptions = {};
      OptixProgramGroupDesc pgDesc    = {};
      pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
      pgDesc.raygen.module            = module;           
      pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

      // OptixProgramGroup raypg;
      char log[2048];
      size_t sizeof_log = sizeof( log );
      OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                          &pgDesc,
                                          1,
                                          &pgOptions,
                                          log,&sizeof_log,
                                          &raygenPGs[0]
                                          ));
      if (sizeof_log > 1) PRINT(log);
    }
    
    void createMissPrograms()
    {
      // we do a single ray gen program in this example:
      missPGs.resize(1);
      
      OptixProgramGroupOptions pgOptions = {};
      OptixProgramGroupDesc pgDesc    = {};
      pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
      pgDesc.miss.module            = module;           
      pgDesc.miss.entryFunctionName = "__miss__radiance";

      // OptixProgramGroup raypg;
      char log[2048];
      size_t sizeof_log = sizeof( log );
      OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                          &pgDesc,
                                          1,
                                          &pgOptions,
                                          log,&sizeof_log,
                                          &missPGs[0]
                                          ));
      if (sizeof_log > 1) PRINT(log);
    }
    
    void createHitgroupPrograms()
    {
      // for this simple example, we set up a single hit group
      hitgroupPGs.resize(1);
      
      OptixProgramGroupOptions pgOptions = {};
      OptixProgramGroupDesc pgDesc    = {};
      pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
      pgDesc.hitgroup.moduleCH            = module;           
      pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
      pgDesc.hitgroup.moduleAH            = module;           
      pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

      char log[2048];
      size_t sizeof_log = sizeof( log );
      OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                          &pgDesc,
                                          1,
                                          &pgOptions,
                                          log,&sizeof_log,
                                          &hitgroupPGs[0]
                                          ));
      if (sizeof_log > 1) PRINT(log);
    }
    
    void render()
    {
      // sanity check: make sure we launch only after first resize is
      // already done:
      if (launchParams.fbSize.x == 0) return;
      
      launchParamsBuffer.upload(&launchParams,1);
      launchParams.frameID++;
      
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
      // sync - make sure the frame is rendered before we download and
      // display (obviously, for a high-performance application you
      // want to use streams and double-buffering, but for this simple
      // example, this will have to do)
      CUDA_SYNC_CHECK();
    }

    void resize(const vec2i &newSize)
    {
      // resize our cuda frame buffer
      fb.resize(newSize);

      // update the launch paramters that we'll pass to the optix
      // launch:
      launchParams.fbSize      = newSize;
      launchParams.colorBuffer = fb.d_colorBuffer;
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
      moduleCompileOptions.maxRegisterCount  = 100;
      moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
      moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

      pipelineCompileOptions = {};
      pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
      pipelineCompileOptions.usesMotionBlur     = false;
      pipelineCompileOptions.numPayloadValues   = 2;
      pipelineCompileOptions.numAttributeValues = 2;
      pipelineCompileOptions.exceptionFlags     = OPTIX_EXCEPTION_FLAG_NONE;
      pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
      
      pipelineLinkOptions.overrideUsesMotionBlur = false;
      pipelineLinkOptions.maxTraceDepth          = 2;
      
      const std::string ptxCode = embedded_ptx_code;
      
      char log[2048];
      size_t sizeof_log = sizeof( log );
      OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                           &moduleCompileOptions,
                                           &pipelineCompileOptions,
                                           ptxCode.c_str(),
                                           ptxCode.size(),
                                           log,&sizeof_log,
                                           &module
                                           ));
      if (sizeof_log > 1) PRINT(log);
    }
    
    void createPipeline()
    {
      std::vector<OptixProgramGroup> programGroups;
      for (auto pg : raygenPGs)
        programGroups.push_back(pg);
      for (auto pg : missPGs)
        programGroups.push_back(pg);
      for (auto pg : hitgroupPGs)
        programGroups.push_back(pg);
      
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
      if (sizeof_log > 1) PRINT(log);

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
      if (sizeof_log > 1) PRINT(log);
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
  
  
  /*! main entry point to this example - initialy optix, print hello
    world, then exit */
  extern "C" int main(int ac, char **av)
  {
    try {
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
