cmake_minimum_required(VERSION 3.18)
project(tetwrap LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- pybind11 (use installed if available, else fetch) ----
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.12.0
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# ============================================================
# 1)Locate TetGen sources in ../tetgen
#   Structure:
#   TetGen/
#     ├─ tetgen/        <-- contains tetgen.cxx, tetgen.h, (optional) predicates.cxx
#     └─ tetwrap/       <-- this CMakeLists.txt lives here
# ============================================================
set(TETGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tetgen")

if(NOT EXISTS "${TETGEN_DIR}/tetgen.cxx")
  message(FATAL_ERROR "tetgen.cxx not found at ${TETGEN_DIR}. Expected ../tetgen relative to tetwrap/")
endif()
if(NOT EXISTS "${TETGEN_DIR}/tetgen.h")
  message(FATAL_ERROR "tetgen.h not found at ${TETGEN_DIR}.")
endif()

set(TETGEN_SOURCES "${TETGEN_DIR}/tetgen.cxx")

# predicates.cxx is strongly recommended (insphere/orient3d)
if(EXISTS "${TETGEN_DIR}/predicates.cxx")
  list(APPEND TETGEN_SOURCES "${TETGEN_DIR}/predicates.cxx")
else()
  message(WARNING "predicates.cxx not found in ${TETGEN_DIR}. You may hit undefined symbols (e.g., insphere) on load.")
endif()

# ============================================================
# 2) Build TetGen as a static library: libtet.a  (as per docs)
#    NOTE: You must list ALL TetGen sources that your copy provides.
#          Typically: tetgen.cxx + predicates.cxx
# ============================================================
add_library(tet STATIC ${TETGEN_SOURCES})
target_compile_definitions(tet PUBLIC TETLIBRARY)
target_include_directories(tet PUBLIC "${TETGEN_DIR}")

# Optional: catch unresolved symbols on macOS at link time
if(APPLE)
  target_link_options(tet PRIVATE "-Wl,-undefined,error")
endif()

# ============================================================
# 3) Build your pybind11 module and link against libtet.a
#    This mirrors: g++ -o test tetcall.cxx -L./ -ltet
#    but builds a Python extension instead of an exe.
#    Replace 'tetwrap.cpp' with your binding file name (entry point).
# ============================================================
pybind11_add_module(_tetwrap
  tetwrap.cpp
)
target_link_libraries(_tetwrap PRIVATE tet)

# Drop the built extension (.so/.pyd) directly into the tetwrap/ directory
set_target_properties(_tetwrap PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"  # mostly for Windows/static variants
)

# Warnings
if(MSVC)
  target_compile_options(tet PRIVATE /W4)
  target_compile_options(_tetwrap PRIVATE /W4)
else()
  target_compile_options(tet PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(_tetwrap PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ------------------------------------------------------------
# (Optional) If you ALSO want a plain executable like the docs:
#   g++ -o test tetcall.cxx -L./ -ltet
# uncomment below and set the correct source file.
# add_executable(test_tet tetcall.cxx)
# target_link_libraries(test_tet PRIVATE tet)
# set_target_properties(test_tet PROPERTIES
#   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/../../"
# )
# ------------------------------------------------------------
