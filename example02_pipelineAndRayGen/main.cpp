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

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "3rdParty/stb_image_write.h"

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  /*! main entry point to this example - initialy optix, print hello
    world, then exit */
  extern "C" int main(int ac, char **av)
  {
    try {
      SampleRenderer sample;

      const vec2i fbSize(vec2i(1200,1024));
      sample.resize(fbSize);
      sample.render();

      std::vector<uint32_t> pixels(fbSize.x*fbSize.y);
      sample.downloadPixels(pixels.data());

      const std::string fileName = "osc_example2.png";
      stbi_write_png(fileName.c_str(),fbSize.x,fbSize.y,4,
                     pixels.data(),fbSize.x*sizeof(uint32_t));
      std::cout << GDT_TERMINAL_GREEN
                << std::endl
                << "Image rendered, and saved to " << fileName << " ... done." << std::endl
                << GDT_TERMINAL_DEFAULT
                << std::endl;
    } catch (std::runtime_error e) {
      std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
                << GDT_TERMINAL_DEFAULT << std::endl;
      exit(1);
    }
    return 0;
  }
  
} // ::osc
