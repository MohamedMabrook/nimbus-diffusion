cmake_minimum_required(VERSION 3.20)
project(NimbusDiffusor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# build a universal binary that works on both Intel and Apple Silicon
if (APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0")
endif()

# ---------------------------------------------------------------------------
# Download the OpenFX headers only — do NOT build the OpenFX project itself
# ---------------------------------------------------------------------------
include(FetchContent)

# CMP0169: suppress deprecation of FetchContent_Populate on CMake 3.30+
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

FetchContent_Declare(openfx
    GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openfx.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(openfx)
if(NOT openfx_POPULATED)
    FetchContent_Populate(openfx)   # download only — skips OpenFX's CMakeLists.txt
endif()

# ---------------------------------------------------------------------------
# Build the .ofx shared library
# ---------------------------------------------------------------------------
add_library(NimbusDiffusor SHARED
    src/NimbusDiffusor.cpp
)

target_include_directories(NimbusDiffusor PRIVATE
    ${openfx_SOURCE_DIR}/include
)

# No runtime dependency on the C++ standard library (keeps the .ofx portable)
if (MSVC)
    target_compile_options(NimbusDiffusor PRIVATE /O2 /W3)
    # Static-link the CRT so the .ofx doesn't require a VC++ redistributable
    set_property(TARGET NimbusDiffusor PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    target_compile_options(NimbusDiffusor PRIVATE -O3 -Wall)
endif()

# OFX plugins are DLLs/dylibs/sos named *.ofx
set_target_properties(NimbusDiffusor PROPERTIES
    PREFIX  ""
    SUFFIX  ".ofx"
    OUTPUT_NAME "NimbusDiffusor"
)

# ---------------------------------------------------------------------------
# Assemble the .ofx.bundle folder after the build
# ---------------------------------------------------------------------------
if (WIN32)
    set(BUNDLE_ARCH "Win64")
elseif (APPLE)
    set(BUNDLE_ARCH "MacOS")
else()
    set(BUNDLE_ARCH "Linux-x86-64")
endif()

set(BUNDLE_DIR      "${CMAKE_BINARY_DIR}/NimbusDiffusor.ofx.bundle/Contents/${BUNDLE_ARCH}")
set(BUNDLE_RSCDIR   "${CMAKE_BINARY_DIR}/NimbusDiffusor.ofx.bundle/Contents/Resources")

add_custom_command(TARGET NimbusDiffusor POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy
        "$<TARGET_FILE:NimbusDiffusor>"
        "${BUNDLE_DIR}/NimbusDiffusor.ofx"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_RSCDIR}"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/NimbusDiffusion.png"
        "${BUNDLE_RSCDIR}/NimbusDiffusion.png"
    COMMENT "Assembling NimbusDiffusor.ofx.bundle"
)

# ---------------------------------------------------------------------------
# Install target: copy bundle to the system OFX plugin folder
# ---------------------------------------------------------------------------
if (WIN32)
    set(OFX_INSTALL_DIR "C:/Program Files/Common Files/OFX/Plugins")
elseif (APPLE)
    set(OFX_INSTALL_DIR "/Library/OFX/Plugins")
else()
    set(OFX_INSTALL_DIR "/usr/OFX/Plugins")
endif()

install(
    DIRECTORY   "${CMAKE_BINARY_DIR}/NimbusDiffusor.ofx.bundle"
    DESTINATION "${OFX_INSTALL_DIR}"
)
