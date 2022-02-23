cmake_minimum_required(VERSION 3.10)
project(controlBind)
set(CMAKE_CXX_STANDARD 17)

####################
### dependencies ###
####################
set(Dependecnies)

set(RAISIM_OS linux)

find_package(pybind11 2.3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(raisim CONFIG REQUIRED)
find_package(raisimOgre CONFIG REQUIRED)
find_package(k4a REQUIRED)
find_package(k4abt REQUIRED)
find_package(k4arecord REQUIRED)
find_package(OpenCV REQUIRED)
add_compile_options(-DHAVE_CSTDDEF)
link_directories(lib)
set(EXTRA_LIBS -pthread libunitree_legged_sdk_amd64.so lcm)
set(CMAKE_CXX_FLAGS "-O3")

set(ENV_INCLUDE_DIR /home/sonic/Libraries/unitree_legged_sdk-3.2/include)

pybind11_add_module(controlBind cb/env/bind_gym.cpp)
target_link_libraries(controlBind PRIVATE ${Dependencies} ${EXTRA_LIBS} raisim::raisimOgre k4a k4arecord k4abt ${OpenCV_LIBS})
target_include_directories(controlBind PUBLIC ${EIGEN3_INCLUDE_DIRS} ${ENV_INCLUDE_DIR})
target_compile_options(controlBind PRIVATE -mtune=native -fPIC -O3 -g -mno-avx2)