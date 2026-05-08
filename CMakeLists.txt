cmake_minimum_required(VERSION 3.20)
project(grizzlars LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

# ── DataFrame helper library ─────────────────────────────────────
# The hmdf library is header-only except for DateTime.cc.
add_library(hmdf_datetime STATIC DataFrame/src/Utils/DateTime.cc)
target_include_directories(hmdf_datetime PUBLIC DataFrame/include)
target_compile_features(hmdf_datetime PUBLIC cxx_std_23)
target_link_libraries(hmdf_datetime PUBLIC Threads::Threads)

# PORTABLE_BUILD=ON strips -march=native so wheels run on any CPU of that arch.
# CI sets this; local dev leaves it OFF for maximum speed.
option(PORTABLE_BUILD "Build for portability rather than native CPU" OFF)

if(MSVC)
    target_compile_options(hmdf_datetime PRIVATE
        /bigobj /wd4251 /wd5030 /wd5222 /wd4996)
    target_compile_definitions(hmdf_datetime PUBLIC
        _CRT_SECURE_NO_WARNINGS NOMINMAX _USE_MATH_DEFINES)
else()
    if(PORTABLE_BUILD)
        target_compile_options(hmdf_datetime PRIVATE -O3 -fno-math-errno)
    else()
        target_compile_options(hmdf_datetime PRIVATE -O3 -march=native -fno-math-errno)
    endif()
endif()

# ── pybind11 extension module ─────────────────────────────────────
pybind11_add_module(_grizzlars src/grizzlars_bindings.cpp)
target_link_libraries(_grizzlars PRIVATE hmdf_datetime)

if(MSVC)
    # /O2        full speed optimisation (MSVC has no /O3)
    # /Oi        replace eligible function calls with intrinsics
    # /GL        whole-program optimisation (link-time inlining across TUs)
    # /fp:precise keep IEEE 754 NaN/Inf behaviour — do NOT use /fp:fast
    # /arch:AVX2 generate AVX2 vector instructions (SIMD filter compress, etc.)
    target_compile_options(_grizzlars PRIVATE
        /O2 /Oi /GL /fp:precise /arch:AVX2
        /bigobj /wd4251 /wd5030 /wd5222 /wd4996)
    target_compile_definitions(_grizzlars PRIVATE
        _CRT_SECURE_NO_WARNINGS NOMINMAX)
    # Parallel STL on MSVC uses the built-in concurrency runtime — no TBB needed
else()
    if(PORTABLE_BUILD)
        target_compile_options(_grizzlars PRIVATE -O3 -fno-math-errno)
    else()
        target_compile_options(_grizzlars PRIVATE -O3 -march=native -fno-math-errno)
    endif()
    # GCC/Clang parallel algorithms need TBB
    find_package(TBB QUIET)
    if(TBB_FOUND)
        target_link_libraries(_grizzlars PRIVATE TBB::tbb)
    endif()
endif()

install(TARGETS _grizzlars DESTINATION grizzlars)
