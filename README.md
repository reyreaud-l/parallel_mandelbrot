# C++ Parallel mandelbrot implementation

This repository contains a mandelbrot fractal implementation in C++, optimized
with AVX2 instructions to speed up pixel calculations, and TBB parallel for to
speed up rendering with multiple threads.

The mandelbrot is implemented in `render.cpp` in the function `render` for the
not multi threaded version, and in `render_mt` for the multi threaded version.

When building the project, two binaries are built:
- view
- bench

# View
View takes a size in parameter and render the image on the screen. If given a
file name after the size, the image will be written in it. 

```
view 720 [filename]
```

# Bench
The bench executable is a benchmark which uses google benchmark lib. It give an
overview of the performance of the implementation.
