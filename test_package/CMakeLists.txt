cmake_minimum_required(VERSION 3.16)
project(rl_lib_conan_test C)

# What a consumer writes: find the installed package and link its target. The
# include directory ("<prefix>/include/rl_lib") travels with the target, so the
# sources below include headers by their installed names.
find_package(RL_lib-in-C CONFIG REQUIRED)

add_executable(hello_conan src/hello_conan.c)
target_link_libraries(hello_conan PRIVATE RL_lib-in-C::rl_lib)
