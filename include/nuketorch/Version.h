#pragma once

/// Library version. Keep in sync with `project(nuketorch VERSION ...)` in CMakeLists.txt.
#define NUKETORCH_VERSION_MAJOR 0
#define NUKETORCH_VERSION_MINOR 2
#define NUKETORCH_VERSION_PATCH 0

#define NUKETORCH_VERSION_STRING "0.2.0"

namespace nuketorch {

/// Runtime access to the compiled-in library version.
inline const char* versionString() { return NUKETORCH_VERSION_STRING; }

}  // namespace nuketorch
