#include <algorithm>
#include <cassert>
#include <cstdint>
#include <future>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

#include "render.hpp"
#include "tbb/parallel_for.h"

struct rgb8_t
{
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
};

struct params
{
  int width_min;
  int height_min;
  int width;
  int height;
  int width_real;
  int height_real;
};

rgb8_t heat_lut(double x)
{
  assert(0 <= x <= 1);
  constexpr float x0 = 1.f / 4.f;
  constexpr float x1 = 2.f / 4.f;
  constexpr float x2 = 3.f / 4.f;

  if (x < x0)
  {
    auto g = static_cast<std::uint8_t>(x / x0 * 255);
    return rgb8_t{0, g, 255};
  }
  else if (x < x1)
  {
    auto b = static_cast<std::uint8_t>((x1 - x) / x0 * 255);
    return rgb8_t{0, 255, b};
  }
  else if (x < x2)
  {
    auto r = static_cast<std::uint8_t>((x - x1) / x0 * 255);
    return rgb8_t{r, 255, 0};
  }
  else
  {
    auto b = static_cast<std::uint8_t>((1.f - x) / x0 * 255);
    return rgb8_t{255, b, 0};
  }
}

void calc_screen(std::vector<std::vector<unsigned>>& iter,
                 std::vector<unsigned>& histo, int n_iterations, params p)
{
  float scx = 3.5 / (p.width_real - 1);
  float scy = 2.0 / (p.height_real - 1);

  for (int py = 0; py < p.height; py++)
  {
    __m256 y0 = _mm256_set1_ps(py * scy - 1);
    for (int px = 0; px < p.width; px += 8)
    {
      // Instantiate 8 floats to parallelize
      __m256 x0 = _mm256_set_ps((px + 7) * scx - 2.5, (px + 6) * scx - 2.5,
                                (px + 5) * scx - 2.5, (px + 4) * scx - 2.5,
                                (px + 3) * scx - 2.5, (px + 2) * scx - 2.5,
                                (px + 1) * scx - 2.5, (px)*scx - 2.5);

      __m256 x = _mm256_setzero_ps();
      __m256 y = _mm256_setzero_ps();
      __m256 iterations = _mm256_setzero_ps();

      int iteration = 0;
      for (; iteration != n_iterations; iteration++)
      {
        // float xtemp = x * x - y * y + x0;
        // y = 2 * x * y + y0;
        // x = xtemp;
        __m256 xsquare = x * x;
        __m256 ysquare = y * y;
        __m256 xy = x * y;

        y = xy + xy + y0;
        x = xsquare - ysquare + x0;

        // x * x + y * y < 4
        __m256 cmp = (xsquare + ysquare) < _mm256_set1_ps(4);

        iterations += _mm256_and_ps(cmp, _mm256_set1_ps(1));
        if (_mm256_testz_ps(cmp, _mm256_set1_ps(-1)))
          break;
      }

      // Store as integer
      _mm256_storeu_si256((__m256i*)(&iter[py][px]),
                          _mm256_cvtps_epi32(iterations));
    }
  };
  for (size_t py = 0; py < p.height; py++)
    for (size_t px = 0; px < p.width; px++)
      histo[iter[py][px]] += 1;
}

void render(std::byte* start, int width, int height, std::ptrdiff_t stride,
            int n_iterations)
{
  std::vector<std::vector<unsigned>> iter(height,
                                          std::vector<unsigned>(width + 8));
  std::vector<unsigned> histo(n_iterations + 1, 0);

  auto p = params{
    0, 0, width, height % 2 == 0 ? height / 2 : height / 2 + 1, width, height};
  std::byte* end = start + (p.height_real - 1) * stride;

  calc_screen(iter, histo, n_iterations, p);

  // Compute total
  double total = 0;
  for (size_t i = 0; i < n_iterations; i++)
    total += histo[i];

  for (size_t py = 0; py < p.height; ++py)
  {
    rgb8_t* top = reinterpret_cast<rgb8_t*>(start);
    rgb8_t* bot = reinterpret_cast<rgb8_t*>(end);

    for (size_t px = 0; px < p.width; ++px)
    {
      rgb8_t pix;
      if (iter[py][px] == n_iterations)
        pix = rgb8_t{0, 0, 0};
      else
      {
        unsigned limit = iter[py][px];
        double hue = 0;
        for (size_t i = 0; i <= limit; ++i)
          hue += (double)histo[i] / (double)total;
        pix = heat_lut(hue);
      }
      top[px] = pix;
      bot[px] = pix;
    }

    end -= stride;
    start += stride;
  }
}

void calc_screen_mt(std::vector<std::vector<unsigned>>& iter,
                    std::vector<unsigned>& histo, int n_iterations, params p)
{
  float scx = 3.5 / (p.width_real - 1);
  float scy = 2.0 / (p.height_real - 1);

  tbb::parallel_for(
    tbb::blocked_range<int>(0, p.height), [&](tbb::blocked_range<int>& r) {
      for (int py = r.begin(); py < r.end(); py++)
      {
        __m256 y0 = _mm256_set1_ps(py * scy - 1);
        for (int px = 0; px < p.width; px += 8)
        {
          // Instantiate 8 floats to parallelize
          __m256 x0 = _mm256_set_ps((px + 7) * scx - 2.5, (px + 6) * scx - 2.5,
                                    (px + 5) * scx - 2.5, (px + 4) * scx - 2.5,
                                    (px + 3) * scx - 2.5, (px + 2) * scx - 2.5,
                                    (px + 1) * scx - 2.5, (px)*scx - 2.5);

          __m256 x = _mm256_setzero_ps();
          __m256 y = _mm256_setzero_ps();
          __m256 iterations = _mm256_setzero_ps();

          int iteration = 0;
          for (; iteration != n_iterations; iteration++)
          {
            // float xtemp = x * x - y * y + x0;
            // y = 2 * x * y + y0;
            // x = xtemp;
            __m256 xsquare = x * x;
            __m256 ysquare = y * y;
            __m256 xy = x * y;

            y = xy + xy + y0;
            x = xsquare - ysquare + x0;

            // x * x + y * y < 4
            __m256 cmp = (xsquare + ysquare) < _mm256_set1_ps(4);

            iterations += _mm256_and_ps(cmp, _mm256_set1_ps(1));
            if (_mm256_testz_ps(cmp, _mm256_set1_ps(-1)))
              break;
          }

          // Store as integer
          _mm256_storeu_si256((__m256i*)(&iter[py][px]),
                              _mm256_cvtps_epi32(iterations));
        }
      };
    });
  tbb::parallel_for(tbb::blocked_range<int>(0, p.height),
                    [&](tbb::blocked_range<int>& r) {
                      for (int py = r.begin(); py < r.end(); py++)
                        for (int px = 0; px < p.width; px++)
                          histo[iter[py][px]] += 1;
                    });
}

void render_mt(std::byte* start, int width, int height, std::ptrdiff_t stride,
               int n_iterations)
{
  std::vector<std::vector<unsigned>> iter(height,
                                          std::vector<unsigned>(width + 8));
  std::vector<unsigned> histo(n_iterations + 1, 0);

  auto p = params{
    0, 0, width, height % 2 == 0 ? height / 2 : height / 2 + 1, width, height};
  std::byte* end = start + (p.height_real - 1) * stride;

  calc_screen_mt(iter, histo, n_iterations, p);

  // Compute total
  unsigned total = 0;
  for (size_t i = 0; i < n_iterations; ++i)
    total += histo[i];

  tbb::parallel_for(
    tbb::blocked_range<int>(0, p.height), [&](tbb::blocked_range<int>& r) {
      for (int py = r.begin(); py < r.end(); py++)
      {
        rgb8_t* top = reinterpret_cast<rgb8_t*>(start + py * stride);
        rgb8_t* bot = reinterpret_cast<rgb8_t*>(end - py * stride);

        for (size_t px = 0; px < width; ++px)
        {
          rgb8_t pix;
          if (iter[py][px] == n_iterations)
            pix = rgb8_t{0, 0, 0};
          else
          {
            unsigned limit = iter[py][px];
            double hue = 0;
            for (size_t i = 0; i <= limit; ++i)
              hue += (double)histo[i] / (double)total;
            pix = heat_lut(hue);
          }
          top[px] = pix;
          bot[px] = pix;
        }
      }
    });
}
