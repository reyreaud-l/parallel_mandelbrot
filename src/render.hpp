#pragma once
#include <cstddef>

/// \param buffer The RGB24 image buffer
/// \param width Image width
/// \param height Image height
/// \param stride Number of bytes between two lines
/// \param n_iterations Number of iterations maximal to decide if a point
///                     belongs to the mandelbrot set.
void render(std::byte* buffer,
            int width,
            int height,
            std::ptrdiff_t stride,
            int n_iterations = 100);

void render_mt(std::byte* buffer,
               int width,
               int height,
               std::ptrdiff_t stride,
               int n_iterations = 100);
