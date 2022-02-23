cmake_minimum_required(VERSION 3.10)

project(bvh_test)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-fPIC)
add_compile_options(-Wdeprecated-declarations)

find_package(pybind11 2.3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(raisim CONFIG REQUIRED)
find_package(raisimOgre CONFIG REQUIRED)
find_package(OGRE 1.12.1 CONFIG REQUIRED COMPONENTS Bites RTShaderSystem)
find_package(k4a REQUIRED)
find_package(k4abt REQUIRED)
find_package(k4arecord REQUIRED)
find_package(OpenCV REQUIRED)

function(create_executable app_name file_name)
  # add_executable(${app_name} file_name)
  target_link_libraries(${app_name} PUBLIC raisim::raisimOgre
  	k4a
    k4arecord
    k4abt
    ${OpenCV_LIBS}
    /usr/lib/libipopt.so
   )
  target_include_directories(${app_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIRS}
  ./sample_helper_includes /usr/include/coin)
  target_compile_definitions(${app_name} PRIVATE "-DEXAMPLE_ROBOT_RESOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/rsc/")
endfunction()

add_compile_options(-DHAVE_CSTDDEF)

file(GLOB srcs "*.cpp" "include/*.h" "include/*.cpp")
message(${srcs})

add_executable(alien ${srcs})
create_executable(alien ${srcs})
