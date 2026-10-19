# Reference vcpkg port for RL_lib-in-C. vcpkg's own ports/rl-lib-in-c/ carries
# these two files; this copy keeps them reviewable next to the sources. The
# port is deliberately thin: the CMake project already installs a complete
# package (headers, both libraries, the CMake config and the pkg-config file),
# so this only configures, builds and fixes up what upstream installs.
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO SushantGagneja/RL_lib-in-C
    REF "v${VERSION}"
    SHA512 8c56855aa167c660ba7066c16fd490809bb14af8c46b70c4047a517c670bc66fe76b76462b1eb489facfbc84e2246ced651794676fb5a9ba6c3455c2bd75c8da
    HEAD_REF main
)

if(VCPKG_LIBRARY_LINKAGE STREQUAL "dynamic")
    set(RL_LIB_SHARED ON)
else()
    set(RL_LIB_SHARED OFF)
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DRL_LIB_BUILD_TESTS=OFF
        -DRL_LIB_BUILD_EXAMPLES=OFF
        -DRL_LIB_BUILD_SHARED=${RL_LIB_SHARED}
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(
    PACKAGE_NAME RL_lib-in-C
    CONFIG_PATH lib/cmake/RL_lib-in-C
)
vcpkg_copy_pdbs()

file(REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/debug/include"
    "${CURRENT_PACKAGES_DIR}/debug/share"
)

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
