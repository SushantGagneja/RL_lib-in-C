"""Conan test package: consume RL_lib-in-C the way a downstream project does.

`conan create .` builds this against the freshly packaged library through
find_package(RL_lib-in-C) and runs it, so the recipe cannot succeed with a
package that is missing headers, libraries or the CMake config.
"""

from conan import ConanFile
from conan.tools.build import can_run
from conan.tools.cmake import CMake, cmake_layout
import os


class RlLibInCTestPackage(ConanFile):
    settings = "os", "arch", "compiler", "build_type"
    generators = "CMakeToolchain", "CMakeDeps"

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        self.requires(self.tested_reference_str)

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def test(self):
        if can_run(self):
            self.run(os.path.join(self.cpp.build.bindirs[0], "hello_conan"),
                     env="conanrun")
