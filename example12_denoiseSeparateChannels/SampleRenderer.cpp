// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "SampleRenderer.h"
#include "LaunchParams.h"
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

#define MAX_CUBE_NUM 16

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

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
    TriangleMeshSBTData data;
  };


  /*! constructor - performs all setup, including initializing
    optix, creates module, pipeline, programs, SBT, etc. */
  SampleRenderer::SampleRenderer(const Model *model, const QuadLight &light)
    : model(model)
  {
    initOptix();

    launchParams.light.origin = light.origin;
    launchParams.light.du     = light.du;
    launchParams.light.dv     = light.dv;
    launchParams.light.power  = light.power;

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

    launchParams.traversable = buildAccel();
    
    std::cout << "#osc: setting up optix pipeline ..." << std::endl;
    createPipeline();

    createTextures();

    
    std::cout << "#osc: building SBT ..." << std::endl;
    buildSBT();

    launchParamsBuffer.alloc(sizeof(launchParams));
    std::cout << "#osc: context, module, pipeline, etc, all set up ..." << std::endl;

    std::cout << GDT_TERMINAL_GREEN;
    std::cout << "#osc: Optix 7 Sample fully set up" << std::endl;
    std::cout << GDT_TERMINAL_DEFAULT;
  }

  void SampleRenderer::createTextures()
  {
    int numTextures = (int)model->textures.size();

    textureArrays.resize(numTextures);
    textureObjects.resize(numTextures);
    
    for (int textureID=0;textureID<numTextures;textureID++) {
      auto texture = model->textures[textureID];
      
      cudaResourceDesc res_desc = {};
      
      cudaChannelFormatDesc channel_desc;
      int32_t width  = texture->resolution.x;
      int32_t height = texture->resolution.y;
      int32_t numComponents = 4;
      int32_t pitch  = width*numComponents*sizeof(uint8_t);
      channel_desc = cudaCreateChannelDesc<uchar4>();
      
      cudaArray_t   &pixelArray = textureArrays[textureID];
      CUDA_CHECK(MallocArray(&pixelArray,
                             &channel_desc,
                             width,height));
      
      CUDA_CHECK(Memcpy2DToArray(pixelArray,
                                 /* offset */0,0,
                                 texture->pixel,
                                 pitch,pitch,height,
                                 cudaMemcpyHostToDevice));
      
      res_desc.resType          = cudaResourceTypeArray;
      res_desc.res.array.array  = pixelArray;
      
      cudaTextureDesc tex_desc     = {};
      tex_desc.addressMode[0]      = cudaAddressModeWrap;
      tex_desc.addressMode[1]      = cudaAddressModeWrap;
      tex_desc.filterMode          = cudaFilterModeLinear;
      tex_desc.readMode            = cudaReadModeNormalizedFloat;
      tex_desc.normalizedCoords    = 1;
      tex_desc.maxAnisotropy       = 1;
      tex_desc.maxMipmapLevelClamp = 99;
      tex_desc.minMipmapLevelClamp = 0;
      tex_desc.mipmapFilterMode    = cudaFilterModePoint;
      tex_desc.borderColor[0]      = 1.0f;
      tex_desc.sRGB                = 0;
      
      // Create texture object
      cudaTextureObject_t cuda_tex = 0;
      CUDA_CHECK(CreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
      textureObjects[textureID] = cuda_tex;
    }
  }
  
  OptixTraversableHandle SampleRenderer::buildAccel()
  {
    const int numMeshes = (int)model->meshes.size();
    /*���������*/
    const int numCubes = MAX_CUBE_NUM;
    vertexBuffer.resize(numMeshes + numCubes);
    normalBuffer.resize(numMeshes + numCubes);
    texcoordBuffer.resize(numMeshes + numCubes);
    indexBuffer.resize(numMeshes + numCubes);
    
    OptixTraversableHandle asHandle { 0 };
    
    // ==================================================================
    // triangle inputs
    // ==================================================================
    /* ��ӷ����buildinput */
    triangleInput.resize(numMeshes + numCubes);
    d_vertices.resize(numMeshes + numCubes);
    d_indices.resize(numMeshes + numCubes);
    triangleInputFlags.resize(numMeshes + numCubes);

    for (int meshID = 0;meshID < numMeshes + numCubes;meshID++) {
        TriangleMesh mesh;
        if (meshID < numMeshes) {
            // upload the model to the device: the builder
            mesh = *model->meshes[meshID];
        }
        /* ��������� */
        else {
            mesh.vertex = std::vector<vec3f>{
                vec3f(0.0f, 0.0f, 0.0f),
                vec3f(0.0f, 0.0f, 0.0f),
                vec3f(0.0f, 0.0f, 0.0f),
                vec3f(0.0f, 0.0f, 0.0f),
                vec3f(0.0f, 0.0f, 0.0f),
                vec3f(0.0f, 0.0f, 0.0f),
                vec3f(0.0f, 0.0f, 0.0f),
                vec3f(0.0f, 0.0f, 0.0f)
            };
            mesh.index = std::vector<vec3i>{
                vec3i(0, 1, 2),
                vec3i(1, 2, 3),
                vec3i(4, 5, 6),
                vec3i(5, 6, 7),
                vec3i(0, 2, 4),
                vec3i(4, 6, 2),
                vec3i(1, 3, 5),
                vec3i(5, 7, 3),
                vec3i(2, 3, 6),
                vec3i(3, 6, 7),
                vec3i(0, 1, 4),
                vec3i(1, 4, 5)
            };
            mesh.normal = std::vector<vec3f>{};
            mesh.texcoord = std::vector<vec2f>{};
        }
        vertexBuffer[meshID].alloc_and_upload(mesh.vertex);
        indexBuffer[meshID].alloc_and_upload(mesh.index);
        if (!mesh.normal.empty())
            normalBuffer[meshID].alloc_and_upload(mesh.normal);
        if (!mesh.texcoord.empty())
            texcoordBuffer[meshID].alloc_and_upload(mesh.texcoord);

        triangleInput[meshID] = {};
        triangleInput[meshID].type
            = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        // create local variables, because we need a *pointer* to the
        // device pointers
        d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
        d_indices[meshID] = indexBuffer[meshID].d_pointer();

        triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(vec3f);
        triangleInput[meshID].triangleArray.numVertices = (int)mesh.vertex.size();
        triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

        triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(vec3i);
        triangleInput[meshID].triangleArray.numIndexTriplets = (int)mesh.index.size();
        triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];

        triangleInputFlags[meshID] = 0;

        // in this example we have one SBT entry, and no per-primitive
        // materials:
        triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
        triangleInput[meshID].triangleArray.numSbtRecords = 1;
        triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
        triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
        triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
    }
    
    // ==================================================================
    // BLAS setup
    // ==================================================================
    
    /* ���flag����update��������ʶ������� */
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION
      | OPTIX_BUILD_FLAG_ALLOW_UPDATE
      | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS
      ;
    accelOptions.motionOptions.numKeys  = 1;
    accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
                (optixContext,
                 &accelOptions,
                 triangleInput.data(),
                 (int)numMeshes + numCubes,  // num_build_inputs
                 &blasBufferSizes
                 ));
    
    // ==================================================================
    // prepare compaction
    // ==================================================================
    
    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));
    
    OptixAccelEmitDesc emitDesc;
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();
    
    // ==================================================================
    // execute build (main stage)
    // ==================================================================
    
    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
    
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);
      
    OPTIX_CHECK(optixAccelBuild(optixContext,
                                /* stream */0,
                                &accelOptions,
                                triangleInput.data(),
                                (int)numMeshes + numCubes,
                                tempBuffer.d_pointer(),
                                tempBuffer.sizeInBytes,
                                
                                outputBuffer.d_pointer(),
                                outputBuffer.sizeInBytes,
                                
                                &asHandle,
                                
                                &emitDesc,1
                                ));
    CUDA_SYNC_CHECK();

    /*-------------------------------------*/
    
    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize,1);
    
    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(optixContext,
                                  /*stream:*/0,
                                  asHandle,
                                  asBuffer.d_pointer(),
                                  asBuffer.sizeInBytes,
                                  &asHandle));
    CUDA_SYNC_CHECK();
    
    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    // outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();
    
    return asHandle;
  }

  /* ��ӷ��� */
  OptixTraversableHandle SampleRenderer::updateAccel()
  {
      vec3f center;
      float edgeLength;
      printf("Please enter the center coordinates and edge length of the cube:\n");
      scanf("%f%f%f%f", &center.x, &center.y, &center.z, &edgeLength);
      const int numMeshes = (int)model->meshes.size();
      const int numCubes = MAX_CUBE_NUM;
      /* ��������� */
      cubeNums++;
      printf("Current cube number: %d\n", cubeNums);
      if (cubeNums > MAX_CUBE_NUM) {
          printf("The number of cubes has reached the upper limit.\n");
          return 0;
      }
      OptixTraversableHandle asHandle{ 0 };
      // ==================================================================
      // triangle inputs
      // ==================================================================
      TriangleMesh mesh;
      mesh.vertex = std::vector<vec3f>{
                  center + vec3f(-1.0, -1.0, -1.0) * edgeLength / 2,
                  center + vec3f(1.0, -1.0, -1.0) * edgeLength / 2,
                  center + vec3f(-1.0, 1.0, -1.0) * edgeLength / 2,
                  center + vec3f(1.0, 1.0, -1.0) * edgeLength / 2,
                  center + vec3f(-1.0, -1.0, 1.0) * edgeLength / 2,
                  center + vec3f(1.0, -1.0, 1.0) * edgeLength / 2,
                  center + vec3f(-1.0, 1.0, 1.0) * edgeLength / 2,
                  center + vec3f(1.0, 1.0, 1.0) * edgeLength / 2,
      };
      vertexBuffer[numMeshes + cubeNums - 1].free();
      vertexBuffer[numMeshes + cubeNums - 1].alloc_and_upload(mesh.vertex);
      d_vertices[numMeshes + cubeNums - 1] = vertexBuffer[numMeshes + cubeNums - 1].d_pointer();
      triangleInput[numMeshes + cubeNums - 1].triangleArray.vertexBuffers = &d_vertices[numMeshes + cubeNums - 1];

      // ==================================================================
      // BLAS setup
      // ==================================================================

      OptixAccelBuildOptions accelOptions = {};
      accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
          | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
          | OPTIX_BUILD_FLAG_ALLOW_UPDATE
          ;
      accelOptions.motionOptions.numKeys = 1;
      accelOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;

      OptixAccelBufferSizes blasBufferSizes;
      OPTIX_CHECK(optixAccelComputeMemoryUsage
      (optixContext,
          &accelOptions,
          triangleInput.data(),
          (int)numMeshes + numCubes,  // num_build_inputs
          &blasBufferSizes
      ));

      // ==================================================================
      // prepare compaction
      // ==================================================================

      CUDABuffer compactedSizeBuffer;
      compactedSizeBuffer.alloc(sizeof(uint64_t));

      OptixAccelEmitDesc emitDesc;
      emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
      emitDesc.result = compactedSizeBuffer.d_pointer();

      // ==================================================================
      // execute build (main stage)
      // ==================================================================

      CUDABuffer tempBuffer;
      tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

      OPTIX_CHECK(optixAccelBuild(optixContext,
          /* stream */0,
          &accelOptions,
          triangleInput.data(),
          (int)numMeshes + numCubes,
          tempBuffer.d_pointer(),
          tempBuffer.sizeInBytes,

          outputBuffer.d_pointer(),
          outputBuffer.sizeInBytes,

          &asHandle,

          &emitDesc, 1
      ));
      CUDA_SYNC_CHECK();

      /*-------------------------------------*/

      // ==================================================================
      // perform compaction
      // ==================================================================
      uint64_t compactedSize;
      compactedSizeBuffer.download(&compactedSize, 1);

      asBuffer.free();
      asBuffer.alloc(compactedSize);
      OPTIX_CHECK(optixAccelCompact(optixContext,
          /*stream:*/0,
          asHandle,
          asBuffer.d_pointer(),
          asBuffer.sizeInBytes,
          &asHandle));
      CUDA_SYNC_CHECK();

      // ==================================================================
      // aaaaaand .... clean up
      // ==================================================================
      // outputBuffer.free(); // << the UNcompacted, temporary output buffer
      tempBuffer.free();
      compactedSizeBuffer.free();

      return asHandle;
  }

  
  /*! helper function that initializes optix and checks for errors */
  void SampleRenderer::initOptix()
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

  static void context_log_cb(unsigned int level,
                             const char *tag,
                             const char *message,
                             void *)
  {
    fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
  }

  /*! creates and configures a optix device context (in this simple
    example, only for the primary GPU device) */
  void SampleRenderer::createContext()
  {
    // for this sample, do everything on one device
    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));
    CUDA_CHECK(StreamCreate(&stream));
      
    cudaGetDeviceProperties(&deviceProps, deviceID);
    std::cout << "#osc: running on device: " << deviceProps.name << std::endl;
      
    CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
    if( cuRes != CUDA_SUCCESS ) 
      fprintf( stderr, "Error querying current context: error code %d\n", cuRes );
      
    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback
                (optixContext,context_log_cb,nullptr,4));
  }



  /*! creates the module that contains all the programs we are going
    to use. in this simple example, we use a single module from a
    single .cu file, using a single embedded ptx string */
  void SampleRenderer::createModule()
  {
    moduleCompileOptions.maxRegisterCount  = 50;
    moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur     = false;
    pipelineCompileOptions.numPayloadValues   = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags     = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
    pipelineCompileOptions.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE
        ;
      
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
    


  /*! does all setup for the raygen program(s) we are going to use */
  void SampleRenderer::createRaygenPrograms()
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
    
  /*! does all setup for the miss program(s) we are going to use */
  void SampleRenderer::createMissPrograms()
  {
    // we do a single ray gen program in this example:
    missPGs.resize(RAY_TYPE_COUNT);

    char log[2048];
    size_t sizeof_log = sizeof( log );

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module              = module ;           

    // ------------------------------------------------------------------
    // radiance rays
    // ------------------------------------------------------------------
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &missPGs[RADIANCE_RAY_TYPE]
                                        ));
    if (sizeof_log > 1) PRINT(log);

    // ------------------------------------------------------------------
    // shadow rays
    // ------------------------------------------------------------------
    pgDesc.miss.entryFunctionName = "__miss__shadow";

    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &missPGs[SHADOW_RAY_TYPE]
                                        ));
    if (sizeof_log > 1) PRINT(log);
  }
    
  /*! does all setup for the hitgroup program(s) we are going to use */
  void SampleRenderer::createHitgroupPrograms()
  {
    // for this simple example, we set up a single hit group
    hitgroupPGs.resize(RAY_TYPE_COUNT);

    char log[2048];
    size_t sizeof_log = sizeof( log );
      
    OptixProgramGroupOptions pgOptions  = {};
    OptixProgramGroupDesc    pgDesc     = {};
    pgDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH            = module;           
    pgDesc.hitgroup.moduleAH            = module;        

    // -------------------------------------------------------
    // radiance rays
    // -------------------------------------------------------
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &hitgroupPGs[RADIANCE_RAY_TYPE]
                                        ));
    if (sizeof_log > 1) PRINT(log);

    // -------------------------------------------------------
    // shadow rays: technically we don't need this hit group,
    // since we just use the miss shader to check if we were not
    // in shadow
    // -------------------------------------------------------
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";

    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &hitgroupPGs[SHADOW_RAY_TYPE]
                                        ));
    if (sizeof_log > 1) PRINT(log);
  }
    

  /*! assembles the full pipeline of all programs */
  void SampleRenderer::createPipeline()
  {
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs)
      programGroups.push_back(pg);
    for (auto pg : hitgroupPGs)
      programGroups.push_back(pg);
    for (auto pg : missPGs)
      programGroups.push_back(pg);
      
    char log[2048];
    size_t sizeof_log = sizeof( log );
    PING;
    PRINT(programGroups.size());
    OPTIX_CHECK(optixPipelineCreate(optixContext,
                                    &pipelineCompileOptions,
                                    &pipelineLinkOptions,
                                    programGroups.data(),
                                    (int)programGroups.size(),
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
                 1));
    if (sizeof_log > 1) PRINT(log);
  }


  /*! constructs the shader binding table */
  void SampleRenderer::buildSBT()
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
    sbt.missRecordCount         = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------
    int numObjects = (int)model->meshes.size();
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int meshID=0;meshID<numObjects;meshID++) {
      for (int rayID=0;rayID<RAY_TYPE_COUNT;rayID++) {
        auto mesh = model->meshes[meshID];
      
        HitgroupRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[rayID],&rec));
        rec.data.color   = mesh->diffuse;
        if (mesh->diffuseTextureID >= 0 && mesh->diffuseTextureID < textureObjects.size()) {
          rec.data.hasTexture = true;
          rec.data.texture    = textureObjects[mesh->diffuseTextureID];
        } else {
          rec.data.hasTexture = false;
        }
        rec.data.index    = (vec3i*)indexBuffer[meshID].d_pointer();
        rec.data.vertex   = (vec3f*)vertexBuffer[meshID].d_pointer();
        rec.data.normal   = (vec3f*)normalBuffer[meshID].d_pointer();
        rec.data.texcoord = (vec2f*)texcoordBuffer[meshID].d_pointer();
        hitgroupRecords.push_back(rec);
      }
    }
    /* �����SBT record */
    const int numCubes = MAX_CUBE_NUM;
    for (int meshID = numObjects;meshID < numObjects + numCubes;meshID++) {
        for (int rayID = 0;rayID < RAY_TYPE_COUNT;rayID++) {
            HitgroupRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[rayID], &rec));
            rec.data.color = vec3f(1.0f, 0.0f, 0.0f);
            rec.data.hasTexture = false;
            rec.data.index = (vec3i*)indexBuffer[meshID].d_pointer();
            rec.data.vertex = (vec3f*)vertexBuffer[meshID].d_pointer();
            rec.data.normal = (vec3f*)normalBuffer[meshID].d_pointer();
            rec.data.texcoord = (vec2f*)texcoordBuffer[meshID].d_pointer();
            hitgroupRecords.push_back(rec);
        }
    }

    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();
  }



  /*! render one frame */
  void SampleRenderer::render()
  {
    // sanity check: make sure we launch only after first resize is
    // already done:
    if (launchParams.frame.size.x == 0) return;

    if (!accumulate)
      launchParams.frame.frameID = 0;
    launchParamsBuffer.upload(&launchParams,1);
    launchParams.frame.frameID++;
    
    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                            pipeline,stream,
                            /*! parameters and SBT */
                            launchParamsBuffer.d_pointer(),
                            launchParamsBuffer.sizeInBytes,
                            &sbt,
                            /*! dimensions of the launch: */
                            launchParams.frame.size.x,
                            launchParams.frame.size.y,
                            1
                            ));

    denoiserIntensity.resize(sizeof(float));

    OptixDenoiserParams denoiserParams;
#if OPTIX_VERSION > 70500
    denoiserParams.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
#endif
#if OPTIX_VERSION >= 70300
    if (denoiserIntensity.sizeInBytes != sizeof(float))
        denoiserIntensity.alloc(sizeof(float));
#endif
    denoiserParams.hdrIntensity = denoiserIntensity.d_pointer();
    if(accumulate)
        denoiserParams.blendFactor  = 1.f/(launchParams.frame.frameID);
    else
        denoiserParams.blendFactor = 0.0f;

    // -------------------------------------------------------
    OptixImage2D inputLayer[3];
    inputLayer[0].data = fbColor.d_pointer();
    /// Width of the image (in pixels)
    inputLayer[0].width = launchParams.frame.size.x;
    /// Height of the image (in pixels)
    inputLayer[0].height = launchParams.frame.size.y;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer[0].rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer[0].pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    inputLayer[0].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // ..................................................................
    inputLayer[2].data = fbNormal.d_pointer();
    /// Width of the image (in pixels)
    inputLayer[2].width = launchParams.frame.size.x;
    /// Height of the image (in pixels)
    inputLayer[2].height = launchParams.frame.size.y;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer[2].rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer[2].pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    inputLayer[2].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // ..................................................................
    inputLayer[1].data = fbAlbedo.d_pointer();
    /// Width of the image (in pixels)
    inputLayer[1].width = launchParams.frame.size.x;
    /// Height of the image (in pixels)
    inputLayer[1].height = launchParams.frame.size.y;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer[1].rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer[1].pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    inputLayer[1].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // -------------------------------------------------------
    OptixImage2D outputLayer;
    outputLayer.data = denoisedBuffer.d_pointer();
    /// Width of the image (in pixels)
    outputLayer.width = launchParams.frame.size.x;
    /// Height of the image (in pixels)
    outputLayer.height = launchParams.frame.size.y;
    /// Stride between subsequent rows of the image (in bytes).
    outputLayer.rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    outputLayer.pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // -------------------------------------------------------
    if (denoiserOn) {
      OPTIX_CHECK(optixDenoiserComputeIntensity
                  (denoiser,
                   /*stream*/0,
                   &inputLayer[0],
                   (CUdeviceptr)denoiserIntensity.d_pointer(),
                   (CUdeviceptr)denoiserScratch.d_pointer(),
                   denoiserScratch.size()));
      
#if OPTIX_VERSION >= 70300
    OptixDenoiserGuideLayer denoiserGuideLayer = {};
    denoiserGuideLayer.albedo = inputLayer[1];
    denoiserGuideLayer.normal = inputLayer[2];

    OptixDenoiserLayer denoiserLayer = {};
    denoiserLayer.input = inputLayer[0];
    denoiserLayer.output = outputLayer;

      OPTIX_CHECK(optixDenoiserInvoke(denoiser,
                                      /*stream*/0,
                                      &denoiserParams,
                                      denoiserState.d_pointer(),
                                      denoiserState.size(),
                                      &denoiserGuideLayer,
                                      &denoiserLayer,1,
                                      /*inputOffsetX*/0,
                                      /*inputOffsetY*/0,
                                      denoiserScratch.d_pointer(),
                                      denoiserScratch.size()));
#else
      OPTIX_CHECK(optixDenoiserInvoke(denoiser,
                                      /*stream*/0,
                                      &denoiserParams,
                                      denoiserState.d_pointer(),
                                      denoiserState.size(),
                                      &inputLayer[0],2,
                                      /*inputOffsetX*/0,
                                      /*inputOffsetY*/0,
                                      &outputLayer,
                                      denoiserScratch.d_pointer(),
                                      denoiserScratch.size()));
#endif
    } else {
      cudaMemcpy((void*)outputLayer.data,(void*)inputLayer[0].data,
                 outputLayer.width*outputLayer.height*sizeof(float4),
                 cudaMemcpyDeviceToDevice);
    }
    computeFinalPixelColors();
    
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();
  }

  /*! set camera to render with */
  void SampleRenderer::setCamera(const Camera &camera)
  {
    lastSetCamera = camera;
    // reset accumulation
    launchParams.frame.frameID = 0;
    launchParams.camera.position  = camera.from;
    launchParams.camera.direction = normalize(camera.at-camera.from);
    const float cosFovy = 0.66f;
    const float aspect
      = float(launchParams.frame.size.x)
      / float(launchParams.frame.size.y);
    launchParams.camera.horizontal
      = cosFovy * aspect * normalize(cross(launchParams.camera.direction,
                                           camera.up));
    launchParams.camera.vertical
      = cosFovy * normalize(cross(launchParams.camera.horizontal,
                                  launchParams.camera.direction));
  }
  
  /*! resize frame buffer to given resolution */
  void SampleRenderer::resize(const vec2i &newSize)
  {
    if (denoiser) {
      OPTIX_CHECK(optixDenoiserDestroy(denoiser));
    };


    // ------------------------------------------------------------------
    // create the denoiser:
    OptixDenoiserOptions denoiserOptions = {};
#if OPTIX_VERSION >= 70300
    OPTIX_CHECK(optixDenoiserCreate(optixContext,OPTIX_DENOISER_MODEL_KIND_LDR,&denoiserOptions,&denoiser));
#else
    denoiserOptions.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO;
#if OPTIX_VERSION < 70100
    // these only exist in 7.0, not 7.1
    denoiserOptions.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
#endif

    OPTIX_CHECK(optixDenoiserCreate(optixContext,&denoiserOptions,&denoiser));
    OPTIX_CHECK(optixDenoiserSetModel(denoiser,OPTIX_DENOISER_MODEL_KIND_LDR,NULL,0));
#endif

    // .. then compute and allocate memory resources for the denoiser
    OptixDenoiserSizes denoiserReturnSizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser,newSize.x,newSize.y,
                                                    &denoiserReturnSizes));

#if OPTIX_VERSION < 70100
    denoiserScratch.resize(denoiserReturnSizes.recommendedScratchSizeInBytes);
#else
    denoiserScratch.resize(std::max(denoiserReturnSizes.withOverlapScratchSizeInBytes,
                                    denoiserReturnSizes.withoutOverlapScratchSizeInBytes));
#endif
    denoiserState.resize(denoiserReturnSizes.stateSizeInBytes);
    
    // ------------------------------------------------------------------
    // resize our cuda frame buffer
    denoisedBuffer.resize(newSize.x*newSize.y*sizeof(float4));
    fbColor.resize(newSize.x*newSize.y*sizeof(float4));
    fbNormal.resize(newSize.x*newSize.y*sizeof(float4));
    fbAlbedo.resize(newSize.x*newSize.y*sizeof(float4));
    finalColorBuffer.resize(newSize.x*newSize.y*sizeof(uint32_t));
    
    // update the launch parameters that we'll pass to the optix
    // launch:
    launchParams.frame.size          = newSize;
    launchParams.frame.colorBuffer   = (float4*)fbColor.d_pointer();
    launchParams.frame.normalBuffer  = (float4*)fbNormal.d_pointer();
    launchParams.frame.albedoBuffer  = (float4*)fbAlbedo.d_pointer();

    // and re-set the camera, since aspect may have changed
    setCamera(lastSetCamera);

    // ------------------------------------------------------------------
    OPTIX_CHECK(optixDenoiserSetup(denoiser,0,
                                   newSize.x,newSize.y,
                                   denoiserState.d_pointer(),
                                   denoiserState.size(),
                                   denoiserScratch.d_pointer(),
                                   denoiserScratch.size()));
  }
  
  /*! download the rendered color buffer */
  void SampleRenderer::downloadPixels(uint32_t h_pixels[])
  {
    finalColorBuffer.download(h_pixels,
                              launchParams.frame.size.x*launchParams.frame.size.y);
  }
  
} // ::osc
