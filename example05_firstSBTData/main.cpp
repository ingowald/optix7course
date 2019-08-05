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

#include "SampleRenderer.h"

// our helper library for window handling
#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  struct SampleWindow : public GLFCameraWindow
  {
    SampleWindow(const std::string &title,
                 const TriangleMesh &model,
                 const Camera &camera,
                 const float worldScale)
      : GLFCameraWindow(title,camera.from,camera.at,camera.up,worldScale),
        sample(model)
    {
      sample.setCamera(camera);
    }
    
    virtual void render() override
    {
      if (cameraFrame.modified) {
        sample.setCamera(Camera{ cameraFrame.get_from(),
                                 cameraFrame.get_at(),
                                 cameraFrame.get_up() });
        cameraFrame.modified = false;
      }
      sample.render();
    }
    
    virtual void draw() override
    {
      sample.downloadPixels(pixels.data());
      if (fbTexture == 0)
        glGenTextures(1, &fbTexture);
      
      glBindTexture(GL_TEXTURE_2D, fbTexture);
      GLenum texFormat = GL_RGBA;
      GLenum texelType = GL_UNSIGNED_BYTE;
      glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
                   texelType, pixels.data());

      glDisable(GL_LIGHTING);
      glColor3f(1, 1, 1);

      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, fbTexture);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      
      glDisable(GL_DEPTH_TEST);

      glViewport(0, 0, fbSize.x, fbSize.y);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

      glBegin(GL_QUADS);
      {
        glTexCoord2f(0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);
      
        glTexCoord2f(0.f, 1.f);
        glVertex3f(0.f, (float)fbSize.y, 0.f);
      
        glTexCoord2f(1.f, 1.f);
        glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);
      
        glTexCoord2f(1.f, 0.f);
        glVertex3f((float)fbSize.x, 0.f, 0.f);
      }
      glEnd();
    }
    
    virtual void resize(const vec2i &newSize) 
    {
      fbSize = newSize;
      sample.resize(newSize);
      pixels.resize(newSize.x*newSize.y);
    }

    GLuint                fbTexture {0};
    vec2i                 fbSize;
    SampleRenderer        sample;
    std::vector<uint32_t> pixels;
  };
  
  
  /*! main entry point to this example - initialy optix, print hello
    world, then exit */
  extern "C" int main(int ac, char **av)
  {
    try {
      TriangleMesh model;
      // 100x100 thin ground plane
      model.addCube(vec3f(0.f,-1.5f,0.f),vec3f(10.f,.1f,10.f));
      // a unit cube centered on top of that
      model.addCube(vec3f(0.f,0.f,0.f),vec3f(2.f,2.f,2.f));
      model.color = vec3f(.2f,.8f,.2f);
      
      Camera camera = { /*from*/vec3f(-10.f,2.f,-12.f),
                        /* at */vec3f(0.f,0.f,0.f),
                        /* up */vec3f(0.f,1.f,0.f) };
      // something approximating the scale of the world, so the
      // camera knows how much to move for any given user interaction:
      const float worldScale = 10.f;

      SampleWindow *window = new SampleWindow("Optix 7 Course Example",
                                              model,camera,worldScale);
      window->run();
      
    } catch (std::runtime_error e) {
      std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
                << GDT_TERMINAL_DEFAULT << std::endl;
      exit(1);
    }
    return 0;
  }
  
} // ::osc
