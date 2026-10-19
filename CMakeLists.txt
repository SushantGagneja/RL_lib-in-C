cmake_minimum_required(VERSION 3.20)

# The version lives in include/version.h - the same single source the Makefile,
# the pkg-config file and the shared-library soname read - so take it from there
# rather than repeating it.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/version.h" _rl_ver_line
     REGEX "#define +RL_LIB_VERSION_STRING")
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" RL_LIB_VERSION "${_rl_ver_line}")
if(NOT RL_LIB_VERSION)
  message(FATAL_ERROR "could not read RL_LIB_VERSION_STRING from include/version.h")
endif()

project(RL_lib-in-C VERSION ${RL_LIB_VERSION} LANGUAGES C)

option(RL_LIB_USE_DOUBLE "Build the f64 configuration (RL_LIB_USE_DOUBLE)" OFF)
option(RL_LIB_NO_THREADS "Build without POSIX threads (serial VecEnv)" OFF)
option(RL_LIB_BUILD_SHARED "Also build the shared library" ON)
option(RL_LIB_BUILD_TESTS "Build the test suites" ON)
option(RL_LIB_BUILD_EXAMPLES "Build the examples and the benchmark" ON)
set(RL_SIMD_BACKEND "" CACHE STRING
    "Force a SIMD backend: 0 scalar, 1 neon, 2 avx2 (empty = auto)")

set(RL_LIB_SOURCES
    src/memory.c src/random.c src/simd.c src/linalg.c src/graph.c
    src/layers.c src/optim.c src/checkpoint.c src/rl.c src/dqn.c src/vecenv.c
    envs/snake.c envs/cartpole.c envs/pendulum.c)

# Everything the static library, the shared library, the tests and the examples
# share: the two include roots the Makefile uses (the repository root is what
# makes "envs/cartpole.h" resolve) and the configuration switches.
add_library(rl_lib_options INTERFACE)
target_include_directories(rl_lib_options INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/rl_lib>)
target_compile_definitions(rl_lib_options INTERFACE
  $<$<BOOL:${RL_LIB_USE_DOUBLE}>:RL_LIB_USE_DOUBLE>
  $<$<BOOL:${RL_LIB_NO_THREADS}>:RL_LIB_NO_THREADS>
  $<$<NOT:$<STREQUAL:${RL_SIMD_BACKEND},>>:RL_SIMD_BACKEND=${RL_SIMD_BACKEND}>)
if(NOT MSVC)
  target_link_libraries(rl_lib_options INTERFACE m)
endif()
set(RL_LIB_PC_THREADS 0)
if(NOT RL_LIB_NO_THREADS AND NOT WIN32)
  find_package(Threads REQUIRED)
  set(RL_LIB_PC_THREADS 1)
endif()

if(MSVC)
  set(RL_LIB_C_STANDARD 11) # MSVC implements C99 through its C11 mode
else()
  set(RL_LIB_C_STANDARD 99)
endif()

set(RL_LIB_LIB_TARGETS rl_lib)

add_library(rl_lib STATIC ${RL_LIB_SOURCES})
add_library(RL_lib-in-C::rl_lib ALIAS rl_lib)

if(RL_LIB_BUILD_SHARED)
  add_library(rl_lib_shared SHARED ${RL_LIB_SOURCES})
  add_library(RL_lib-in-C::rl_lib_shared ALIAS rl_lib_shared)
  set_target_properties(rl_lib_shared PROPERTIES
    OUTPUT_NAME rl_lib
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
  if(WIN32)
    # No __declspec annotations in the sources: let CMake generate the .def.
    set_target_properties(rl_lib_shared PROPERTIES
      WINDOWS_EXPORT_ALL_SYMBOLS ON)
    if(MSVC)
      # MSVC names the static archive and an import library alike (rl_lib.lib),
      # so in the build tree and in the prefix one would quietly overwrite the
      # other, and a consumer that linked the survivor would need rl_lib.dll
      # without being told. Keep the import library distinct; the DLL itself
      # keeps its name.
      set_target_properties(rl_lib_shared PROPERTIES
        ARCHIVE_OUTPUT_NAME rl_lib_shared)
    endif()
  endif()
  list(APPEND RL_LIB_LIB_TARGETS rl_lib_shared)
endif()

foreach(tgt IN LISTS RL_LIB_LIB_TARGETS)
  target_link_libraries(${tgt} PUBLIC rl_lib_options)
  if(RL_LIB_PC_THREADS)
    target_link_libraries(${tgt} PUBLIC Threads::Threads)
  endif()
  set_target_properties(${tgt} PROPERTIES
    C_STANDARD ${RL_LIB_C_STANDARD}
    C_STANDARD_REQUIRED ON)
  if(MSVC)
    target_compile_options(${tgt} PRIVATE /W3)
  else()
    target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wshadow -Wconversion
                                           -Wmissing-prototypes)
  endif()
endforeach()

set(RL_LIB_EXAMPLES
    train_snake train_cartpole train_a2c train_ppo train_ppo_continuous
    train_ppo_snake train_dqn hello_rl)

set(RL_LIB_TESTS test_mat test_gradcheck test_envs test_rl test_checkpoint
                 test_dqn)

if(RL_LIB_BUILD_EXAMPLES)
  foreach(ex ${RL_LIB_EXAMPLES} bench)
    add_executable(${ex} examples/${ex}.c)
    target_link_libraries(${ex} PRIVATE rl_lib)
  endforeach()
endif()

if(RL_LIB_BUILD_TESTS)
  enable_testing()
  foreach(t ${RL_LIB_TESTS})
    add_executable(${t} tests/${t}.c)
    target_link_libraries(${t} PRIVATE rl_lib)
    add_test(NAME ${t} COMMAND ${t})
  endforeach()
endif()

# ── Install ─────────────────────────────────────────────────────────
include(GNUInstallDirs)

install(TARGETS rl_lib
        EXPORT RL_lib-in-CTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(RL_LIB_BUILD_SHARED)
  install(TARGETS rl_lib_shared
          EXPORT RL_lib-in-CTargets
          # ARCHIVE is the import library on Windows (rl_lib_shared.lib with
          # MSVC, librl_lib.dll.a with MinGW); without a destination for it the
          # exported target has no IMPORTED_IMPLIB and consumers cannot link.
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(EXPORT RL_lib-in-CTargets
        NAMESPACE RL_lib-in-C::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RL_lib-in-C)
install(TARGETS rl_lib_options EXPORT RL_lib-in-CTargets)

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/RL_lib-in-CConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/RL_lib-in-CConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RL_lib-in-C)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/RL_lib-in-CConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/RL_lib-in-CConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/RL_lib-in-CConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RL_lib-in-C)

# Headers go flat into <prefix>/include/rl_lib with an envs/ subdirectory,
# exactly like `make install`.
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rl_lib
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY envs/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rl_lib/envs
        FILES_MATCHING PATTERN "*.h")

# pkg-config. This stamps the same template the Makefile does (rl_lib.pc.in at
# the repository root), so the two builds cannot describe the library
# differently; only the variable names differ. Relocatable prefix: the .pc lives
# in <libdir>/pkgconfig, so the prefix is that many directory levels up - which
# keeps the file correct when the tree is moved or installed with
# `cmake --install --prefix ...` after configuring. An absolute libdir has no
# such relation (that is already outside the prefix).
if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
  set(PREFIX "${CMAKE_INSTALL_PREFIX}")
else()
  string(REPLACE "/" ";" _rl_libdir_parts "${CMAKE_INSTALL_LIBDIR}")
  list(LENGTH _rl_libdir_parts _rl_libdir_depth)
  math(EXPR _rl_lib_pc_ups "${_rl_libdir_depth} + 1")
  set(PREFIX "\${pcfiledir}")
  foreach(_i RANGE 1 ${_rl_lib_pc_ups})
    string(APPEND PREFIX "/..")
  endforeach()
endif()
set(LIBDIR "\${prefix}/${CMAKE_INSTALL_LIBDIR}")
set(INCLUDEDIR "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
set(VERSION "${PROJECT_VERSION}")
set(LIBS_EXTRA "")
if(NOT WIN32)
  set(LIBS_EXTRA "-lm")
  if(NOT RL_LIB_NO_THREADS)
    string(APPEND LIBS_EXTRA " -pthread")
  endif()
endif()
configure_file(rl_lib.pc.in ${CMAKE_CURRENT_BINARY_DIR}/rl_lib.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/rl_lib.pc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
