cmake_minimum_required(VERSION 3.20)
project(grizzlars LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Whole-program / link-time optimization, applied globally so it also covers
# the hmdf DataFrame::DataFrame target below (set before add_subdirectory,
# without touching cpp_lib/DataFrame's own CMakeLists.txt). CMake's own
# INTERPROCEDURAL_OPTIMIZATION abstraction is used instead of hand-rolling
# per-compiler flags (/GL) because it correctly pairs /GL with /LTCG on the
# *link* step for MSVC — without an explicit /LTCG, link.exe was silently
# restarting itself every build ("module compiled with /GL found;
# restarting link with /LTCG"), doing the same optimization but slower to
# build. On GCC/Clang it emits -flto, which was previously not enabled at
# all outside MSVC.
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
if(IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
else()
    message(STATUS "IPO/LTO not supported by this toolchain: ${IPO_ERROR}")
endif()

# nanobind ships its CMake config inside the pip package itself — locate it
# via the same interpreter scikit-build-core is using to build the wheel.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# ── hmdf DataFrame — unedited upstream, consumed as a git submodule ───────
# Defines the DataFrame::DataFrame target (DateTime.cc + header-only rest),
# including the MSVC/GNU compile options its own headers require (PUBLIC,
# so they propagate automatically to any target that links against it).
set(HMDF_TESTING OFF CACHE BOOL "" FORCE)
set(HMDF_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HMDF_BENCHMARKS OFF CACHE BOOL "" FORCE)
add_subdirectory(cpp_lib/DataFrame)

# ── litgen codegen: regenerate nanobind glue from the shim header ────────
# Always run at build time so the glue can never drift from grizzlars_shim.h.
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})
set(GENERATED_GLUE ${GENERATED_DIR}/grizzlars_glue.inc)

add_custom_command(
    OUTPUT ${GENERATED_GLUE}
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_bindings.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/grizzlars_shim.h"
            "${GENERATED_GLUE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/grizzlars/_grizzlars.pyi"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/grizzlars_shim.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_bindings.py"
    COMMENT "Generating nanobind glue from grizzlars_shim.h via litgen"
    VERBATIM
)
add_custom_target(generate_grizzlars_glue DEPENDS ${GENERATED_GLUE})

# PORTABLE_BUILD=ON strips -march=native so wheels run on any CPU of that arch.
# CI sets this; local dev leaves it OFF for maximum speed.
option(PORTABLE_BUILD "Build for portability rather than native CPU" OFF)

# ── the nanobind extension module ─────────────────────────────────────────
nanobind_add_module(_grizzlars src/grizzlars_module.cpp src/grizzlars_shim.cpp)
add_dependencies(_grizzlars generate_grizzlars_glue)
target_include_directories(_grizzlars PRIVATE ${GENERATED_DIR})
target_link_libraries(_grizzlars PRIVATE DataFrame::DataFrame)

if(MSVC)
    # /O2        full speed optimisation (MSVC has no /O3)
    # /Oi        replace eligible function calls with intrinsics
    # /fp:precise keep IEEE 754 NaN/Inf behaviour — do NOT use /fp:fast
    # (whole-program/link-time optimisation is /GL+/LTCG, applied globally
    # above via CMAKE_INTERPROCEDURAL_OPTIMIZATION instead of by hand here)
    target_compile_options(_grizzlars PRIVATE /O2 /Oi /fp:precise /bigobj)
    target_compile_definitions(_grizzlars PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    if(PORTABLE_BUILD)
        target_compile_options(_grizzlars PRIVATE -O3 -fno-math-errno)
    else()
        target_compile_options(_grizzlars PRIVATE -O3 -march=native -fno-math-errno)
    endif()
endif()

install(TARGETS _grizzlars DESTINATION grizzlars)
