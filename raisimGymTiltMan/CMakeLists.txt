cmake_minimum_required(VERSION 3.10)
project(_raisim_gym_tiltman)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/raisim_gym_tiltman/env/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/raisim_gym_tiltman/env/bin)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/raisim_gym_tiltman/env/bin)

######################
### help functions ###
######################
MACRO(SUBDIRLIST result curdir)
    FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
    SET(dirlist "")
    FOREACH(child ${children})
        IF(IS_DIRECTORY ${curdir}/${child})
            LIST(APPEND dirlist ${child})
        ENDIF()
    ENDFOREACH()
    SET(${result} ${dirlist})
ENDMACRO()

####################
### dependencies ###
####################
set(Dependencies)

if (UNIX AND NOT APPLE AND NOT ANDROID AND NOT WEBGL AND NOT WIN32)
    set(RAISIM_OS linux)
    list(APPEND CMAKE_PREFIX_PATH ../raisim/${RAISIM_OS})
    MESSAGE (${CMAKE_PREFIX_PATH})
elseif(APPLE)
    set(RAISIM_OS mac)
    list(APPEND CMAKE_PREFIX_PATH ../raisim/${RAISIM_OS})
elseif(WIN32)
    set(RAISIM_OS win32)
    list(APPEND CMAKE_PREFIX_PATH ../raisim/${RAISIM_OS}/mt_release)
    list(APPEND CMAKE_PREFIX_PATH ../raisim/${RAISIM_OS}/mt_debug)
    list(APPEND CMAKE_PREFIX_PATH ../thirdParty)
    list(APPEND Dependencies Ws2_32)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/raisim_gym_tiltman/env/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/raisim_gym_tiltman/env/bin)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/raisim_gym_tiltmans/env/bin)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../thirdParty/pybind11 pybind11)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(raisim CONFIG REQUIRED)
find_package(raisimOgre 0.6.0 CONFIG REQUIRED)
find_package(k4a REQUIRED)
find_package(k4abt REQUIRED)
find_package(k4arecord REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(iir REQUIRED)
# find_package(glfw3 REQUIRED)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

#######################
### src and linking ###
#######################

# So this part is for building  ==> append??? 

# message(${CMAKE_CURRENT_SOURCE_DIR})
# SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR}/raisimGymPulse/env/envs)
# set(RAISIMGYM_ENV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/raisimGymPulse/env/envs)

SUBDIRLIST(SUBDIRS /home/sonic/Project/Alien_Gesture/env/A1_TiltMan)
set(RAISIMGYM_ENV_DIR /home/sonic/Project/Alien_Gesture/env/A1_TiltMan)
set(RAISIMGYM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/raisim_gym_tiltman/env)
set(ENV_INCLUDE_DIR /home/sonic/Project/Alien_Gesture/env/include)
MESSAGE (${RAISIMGYM_INCLUDE_DIR})

FOREACH(subdir ${SUBDIRS})
    message(subdir is ${subdir})
    pybind11_add_module(${subdir} raisim_gym_tiltman/env/raisim_gym.cpp raisim_gym_tiltman/env/Yaml/Yaml.cpp)
    target_link_libraries(${subdir} PRIVATE raisim::raisim ${Dependencies} raisim::raisimOgre 
        k4a k4arecord k4abt ${OPENGL_LIBRARIES} ${OpenCV_LIBS} iir::iir)
    target_include_directories(${subdir} PUBLIC ${EIGEN3_INCLUDE_DIRS} ${ENV_INCLUDE_DIR} ${RAISIMGYM_INCLUDE_DIR} ${RAISIMGYM_ENV_DIR}/${subdir})
    target_compile_options(${subdir} PRIVATE -mtune=native -fPIC -O3 -g -mno-avx2)
    target_compile_definitions(${subdir} PRIVATE "-DRAISIMGYM_ENV_NAME=${subdir}")

    if (CMAKE_BUILD_TYPE STREQUAL "DEBUG" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("[RAISIM_GYM] BUILDING THE DEBUG APP for ${subdir}")
        add_executable(${subdir}_debug_app raisim_gym_tiltman/env/debug_app.cpp raisim_gym_tiltman/env/Yaml.cpp)
        target_link_libraries(${subdir}_debug_app PRIVATE raisim::raisim raisim::raisimOgre k4a k4arecord k4abt ${OPENGL_LIBRARIES} ${OpenCV_LIBS})
        target_include_directories(${subdir}_debug_app PUBLIC ${ENV_INCLUDE_DIR} ${RAISIMGYM_INCLUDE_DIR} ${RAISIMGYM_ENV_DIR}/${subdir} ${EIGEN3_INCLUDE_DIRS}  ${OPENGL_INCLUDE_DIRS})
        target_compile_options(${subdir}_debug_app PRIVATE -mtune=native -fPIC -g -O0 -mno-avx2)
    endif()
ENDFOREACH()
