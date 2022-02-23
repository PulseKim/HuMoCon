#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "Environment.hpp"
#include "VectorizedEnvironment.hpp"

namespace py = pybind11;
using namespace raisim;

#ifndef ENVIRONMENT_NAME
  #define ENVIRONMENT_NAME RaisimGymEnv
#endif

PYBIND11_MODULE(env, env) {
  py::class_<VectorizedEnvironment<ENVIRONMENT>>(env, RSG_MAKE_STR(ENVIRONMENT_NAME))
    .def(py::init<std::string, std::string>())
    .def("init", &VectorizedEnvironment<ENVIRONMENT>::init)
    .def("getExtraInfoNames", &VectorizedEnvironment<ENVIRONMENT>::getExtraInfoNames)
    .def("reset", &VectorizedEnvironment<ENVIRONMENT>::reset)
    .def("observe", &VectorizedEnvironment<ENVIRONMENT>::observe)
    .def("step", &VectorizedEnvironment<ENVIRONMENT>::step)
    .def("setSeed", &VectorizedEnvironment<ENVIRONMENT>::setSeed)
    .def("step", &VectorizedEnvironment<ENVIRONMENT>::step)
    .def("testStep", &VectorizedEnvironment<ENVIRONMENT>::testStep)
    .def("stepRef", &VectorizedEnvironment<ENVIRONMENT>::stepRef)
    .def("close", &VectorizedEnvironment<ENVIRONMENT>::close)
    .def("isTerminalState", &VectorizedEnvironment<ENVIRONMENT>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<ENVIRONMENT>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<ENVIRONMENT>::setControlTimeStep)
    .def("getObDim", &VectorizedEnvironment<ENVIRONMENT>::getObDim)
    .def("getActionDim", &VectorizedEnvironment<ENVIRONMENT>::getActionDim)
    .def("getExtraInfoDim", &VectorizedEnvironment<ENVIRONMENT>::getExtraInfoDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<ENVIRONMENT>::getNumOfEnvs)
    .def("startRecordingVideo", &VectorizedEnvironment<ENVIRONMENT>::startRecordingVideo)
    .def("stopRecordingVideo", &VectorizedEnvironment<ENVIRONMENT>::stopRecordingVideo)
    .def("showWindow", &VectorizedEnvironment<ENVIRONMENT>::showWindow)
    .def("hideWindow", &VectorizedEnvironment<ENVIRONMENT>::hideWindow)
    .def("curriculumUpdate", &VectorizedEnvironment<ENVIRONMENT>::curriculumUpdate)

    // Implemented afterwards
    .def("kinect_init", &VectorizedEnvironment<ENVIRONMENT>::kinect_init)
    .def("stepTest", &VectorizedEnvironment<ENVIRONMENT>::stepTest)
    .def("getContactStep", &VectorizedEnvironment<ENVIRONMENT>::getContactStep)
    .def("getNumFrames", &VectorizedEnvironment<ENVIRONMENT>::getNumFrames)
    .def("get_robot_key", &VectorizedEnvironment<ENVIRONMENT>::get_robot_key)
    .def("set_alien_pose", &VectorizedEnvironment<ENVIRONMENT>::set_alien_pose)
    .def("get_alien_pose", &VectorizedEnvironment<ENVIRONMENT>::get_alien_pose)
    .def("set_bvh_time", &VectorizedEnvironment<ENVIRONMENT>::set_bvh_time)
    .def("get_animal_key", &VectorizedEnvironment<ENVIRONMENT>::get_animal_key)
    .def("get_target_pose", &VectorizedEnvironment<ENVIRONMENT>::get_target_pose)
    .def("get_current_q", &VectorizedEnvironment<ENVIRONMENT>::get_current_q)
    .def("get_jacobian", &VectorizedEnvironment<ENVIRONMENT>::get_jacobian)
    .def("get_current_torque", &VectorizedEnvironment<ENVIRONMENT>::get_current_torque);

}
