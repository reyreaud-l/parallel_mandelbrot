#include "render.hpp"
#include "spdlog/spdlog.h"
#include <SDL.h>
#include <SDL_surface.h>
#include <SDL_video.h>
#include <iostream>

// Usage: ./view resolution

int main(int argc, char** argv)
{
  auto console = spdlog::stdout_color_mt("console");
  if (argc > 3)
  {
    std::cerr << "Usage: " << argv[0] << " [yres = 1080 [output.bmp]]\n";
    return 1;
  }

  if (SDL_Init(SDL_INIT_VIDEO) != 0)
  {
    console->error("{}:{} Unable to init SDL ({})", __FILE__, __LINE__,
                   SDL_GetError());
    return 1;
  }
  int height = (argc >= 2) ? std::clamp(std::atoi(argv[1]), 360, 2160) : 1080;
  int width = height * 16 / 9;

  // Rendering
  auto img =
    SDL_CreateRGBSurfaceWithFormat(0, width, height, 24, SDL_PIXELFORMAT_RGB24);
  SDL_LockSurface(img);
  render_mt(static_cast<std::byte*>(img->pixels), width, height, img->pitch);
  SDL_UnlockSurface(img);

  bool tofile_display = argc >= 3;
  if (tofile_display)
  {
    SDL_SaveBMP(img, argv[2]);
  }
  else
  {
    // Create window
    auto window =
      SDL_CreateWindow("Render", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                       width, height, SDL_WINDOW_SHOWN | SDL_WINDOW_BORDERLESS);
    if (window == nullptr)
    {
      console->error("{}:{} Unable to create window ({})", __FILE__, __LINE__,
                     SDL_GetError());
      return 1;
    }
    auto surface = SDL_GetWindowSurface(window);

    // Update the surface
    if (SDL_BlitSurface(img, nullptr, surface, nullptr) != 0)
    {
      console->error("{}:{} Cannot blit surface.", __FILE__, __LINE__);
    }
    SDL_UpdateWindowSurface(window);
    SDL_Delay(5000);
    SDL_DestroyWindow(window);
  }

  // Quit
  SDL_Quit();
}
