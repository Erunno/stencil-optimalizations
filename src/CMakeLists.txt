enable_language(CUDA)

# Gather CUDA source files
file(GLOB_RECURSE CudaFiles
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

# Gather regular C++ source files
file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_executable(stencils
    ${SOURCE_FILES}
    ${CudaFiles}
)
