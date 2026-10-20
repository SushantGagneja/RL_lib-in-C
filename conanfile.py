"""Conan 2 recipe for RL_lib-in-C.

`conan create .` exports the checkout, configures the CMake project, builds it,
runs the ctest suites (skip them with `-c tools.build:skip_test=True`) and
packages the same install tree `make install` produces: headers under
include/rl_lib, the static and (by default) shared libraries, the CMake package
config and the pkg-config file. `test_package/` then consumes the package the
way a downstream project would, through `find_package(RL_lib-in-C)`.

The version attribute must match include/version.h; the `conan` CI job builds
the recipe from the release tag, so a drift would fail there.
"""

from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import copy
import os


class RlLibInCRecipe(ConanFile):
    name = "rl_lib_in_c"
    version = "1.1.0"
    package_type = "library"
    description = ("A dependency-free C reinforcement-learning and autograd "
                   "library: arena-based memory, SIMD matmul, reverse-mode AD, "
                   "REINFORCE/A2C/PPO/DQN/SAC and CartPole/Snake/Pendulum")
    license = "MIT"
    url = "https://github.com/SushantGagneja/RL_lib-in-C"
    homepage = "https://github.com/SushantGagneja/RL_lib-in-C"
    topics = ("reinforcement-learning", "autograd", "neural-network", "c")
    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "use_double": [True, False],
        "no_threads": [True, False],
        "with_blas": [True, False],
    }
    default_options = {
        "shared": True,
        "fPIC": True,
        "use_double": False,
        "no_threads": False,
        "with_blas": False,
    }
    exports_sources = (
        "CMakeLists.txt",
        "LICENSE",
        "rl_lib.pc.in",
        "include/*",
        "src/*",
        "envs/*",
        "tests/*",
        "cmake/*",
    )

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["RL_LIB_BUILD_SHARED"] = self.options.shared
        # Build the suites only when they are going to run; with
        # tools.build:skip_test=True the binaries would just be dead weight.
        tc.variables["RL_LIB_BUILD_TESTS"] = not self.conf.get(
            "tools.build:skip_test", default=False)
        tc.variables["RL_LIB_BUILD_EXAMPLES"] = False
        tc.variables["RL_LIB_USE_DOUBLE"] = self.options.use_double
        tc.variables["RL_LIB_NO_THREADS"] = self.options.no_threads
        tc.variables["RL_LIB_USE_BLAS"] = self.options.with_blas
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
        if not self.conf.get("tools.build:skip_test", default=False):
            cmake.test()

    def package(self):
        copy(self, "LICENSE", src=self.source_folder,
             dst=os.path.join(self.package_folder, "licenses"))
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["rl_lib"]
        # Headers are installed flat under include/rl_lib, the same layout the
        # Makefile install and the CMake project produce (and what rl_lib.pc's
        # Cflags point at), so advertise that as the include directory; the
        # default would be the bare include/, one level too high.
        self.cpp_info.includedirs = ["include/rl_lib"]
        # The names a downstream CMake or pkg-config consumer should use.
        self.cpp_info.set_property("cmake_file_name", "RL_lib-in-C")
        self.cpp_info.set_property("cmake_target_name", "RL_lib-in-C::rl_lib")
        self.cpp_info.set_property("pkg_config_name", "rl_lib")
