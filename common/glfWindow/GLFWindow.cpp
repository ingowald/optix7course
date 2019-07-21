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

#include "GLFWindow.h"

/*! \namespace osc - Optix Siggraph Course */
namespace osc {
  using namespace gdt;
  
  static void glfw_error_callback(int error, const char* description)
  {
    fprintf(stderr, "Error: %s\n", description);
  }
  
  GLFWindow::~GLFWindow()
  {
    glfwDestroyWindow(handle);
    glfwTerminate();
  }

  GLFWindow::GLFWindow(const std::string &title)
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

  /*! opens the actual window, and runs the window's events to
    completion. This function will only return once the window
    gets closed */
  static void glfw_reshape(GLFWwindow* window, int width, int height )
  {
    assert(GLFWindow::current);
    GLFWindow::current->resize(vec2i(width,height));
  }
    
  void GLFWindow::run()
  {
    int width, height;
    glfwGetFramebufferSize(handle, &width, &height);
    resize(vec2i(width,height));

    glfwSetFramebufferSizeCallback(handle, glfw_reshape);
    
    while (!glfwWindowShouldClose(handle)) {
      draw();
        
      glfwSwapBuffers(handle);
      glfwPollEvents();
    }
  }

  GLFWindow *GLFWindow::current = nullptr;
  
} // ::osc
