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

// common gdt helper tools
#include "gdt/math/vec.h"
#include "optix7.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

/*! \namespace osc - Optix Siggraph Course */
namespace osc {
  using namespace gdt;
  
  struct FrameBuffer
  {
    void resize(const vec2i &size)
    {
      if (d_colorBuffer) CUDA_CHECK(Free(d_colorBuffer));
      if (h_colorBuffer) delete[] h_colorBuffer;

      CUDA_CHECK(Malloc(&d_colorBuffer,size.x*size.y*sizeof(uint32_t)));
      h_colorBuffer = new uint32_t[size.x*size.y];
    }
    ~FrameBuffer()
    {
      if (d_colorBuffer) CUDA_CHECK_NOEXCEPT(Free(d_colorBuffer));
      if (h_colorBuffer) delete[] h_colorBuffer;
    }

    vec2i size              { 0,0 };
    uint32_t *h_colorBuffer { nullptr };
    uint32_t *d_colorBuffer { nullptr };
  };

  static void glfw_error_callback(int error, const char* description)
  {
    fprintf(stderr, "Error: %s\n", description);
  }
  
  struct GLFWindow {
    GLFWindow(const std::string &title)
    {
      if (current)
        throw std::runtime_error("GLFWindow: can only have one active window ...");
      current = this;
      
      glfwSetErrorCallback(glfw_error_callback);
      // glfwInitHint(GLFW_COCOA_MENUBAR, GLFW_FALSE);
      
      if (!glfwInit())
        exit(EXIT_FAILURE);
      
      glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
      glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
      glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
      
      handle = glfwCreateWindow(640, 480, title.c_str(), NULL, NULL);
      if (!handle) {
        glfwTerminate();
        exit(EXIT_FAILURE);
      }
      
      glfwMakeContextCurrent(handle);
      glfwSwapInterval( 1 );
    }

    virtual void draw()
    {
    }

    static void reshape(GLFWwindow* window, int width, int height )
    {
      assert(current);
      current->resize(vec2i(width,height));
    }
    
    void run()
    {
      int width, height;
      glfwGetFramebufferSize(handle, &width, &height);
      resize(vec2i(width,height));

      glfwSetFramebufferSizeCallback(handle, reshape);
      
      while (!glfwWindowShouldClose(handle)) {
        // Draw gears
        draw();
        
        // Swap buffers
        glfwSwapBuffers(handle);
        glfwPollEvents();
      }
    }

    
    ~GLFWindow()
    {
      glfwDestroyWindow(handle);
      glfwTerminate();
    }
    
    virtual void resize(const vec2i &newSize) 
    {
      fb.resize(newSize);
    }
    
    virtual void render()
    {
    }

    static GLFWindow *current;
    /*! the glfw window handle */
    GLFWwindow *handle;
    
    /*! our internal frame buffer */
    FrameBuffer fb;
  };

  GLFWindow *GLFWindow::current = nullptr;
  
  struct SampleWindow : public GLFWindow
  {
    SampleWindow(const std::string &title)
      : GLFWindow(title)
    {}
    
    virtual void render() override
    {
      PING;
    }
    
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
  
}
