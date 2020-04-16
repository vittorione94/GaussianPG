cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
project(example-app)

set(CMAKE_PREFIX_PATH "./libtorch")

find_package(Torch REQUIRED)
#find_package(OpenMP REQUIRED)
# Find OpenMP
if(APPLE AND USE_OpenMP)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C "${CMAKE_C_COMPILER}")
        set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
        set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
        set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
        set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
        set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
        set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
        set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
        set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    endif()
endif()

include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

message("${CMAKE_SOURCE_DIR}/mujocoLibs")
include_directories(${CMAKE_SOURCE_DIR}/mujocoLibs)
include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB Protobuf_LIBRARIES "/usr/local/lib/libprotobuf.dylib")
file(GLOB Mujoco "mujocoLibs/*.dylib")
file(GLOB uitoolsMujoco "include/*.c")

set(LIBRARIES ${TORCH_LIBRARIES} ${Mujoco} ${uitoolsMujoco})
message("LIBRARIES = ${LIBRARIES}")

add_executable(example-app  ./src/example-app.cpp ${SRCS} ${HDRS})

if(OpenMP_CXX_FOUND)
	message("OpenMP_CXX_FOUND")	
    target_link_libraries(example-app PUBLIC OpenMP::OpenMP_CXX "${LIBRARIES}")
else()
	target_link_libraries(example-app "${LIBRARIES}")
endif()

set_property(TARGET example-app PROPERTY CXX_STANDARD 17)
