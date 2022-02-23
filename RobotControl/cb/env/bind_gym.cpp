#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "VectorizedEnvironment.hpp"

namespace py = pybind11;
#define __RSG_MAKE_STR(x) #x
#define _RSG_MAKE_STR(x) __RSG_MAKE_STR(x)
#define RSG_MAKE_STR(x) _RSG_MAKE_STR(x)

#ifndef ENVIRONMENT_NAME
  #define ENVIRONMENT_NAME ControlEnv
#endif

PYBIND11_MODULE(controlBind, m) {
  py::class_<VectorizedEnvironment>(m, RSG_MAKE_STR(ENVIRONMENT_NAME))
    .def(py::init<std::string, std::string>())
    .def("init", &VectorizedEnvironment::init)
    .def("check_delay", &VectorizedEnvironment::check_delay)
    .def("kinectInit", &VectorizedEnvironment::kinect_init)
    .def("readSeq", &VectorizedEnvironment::readSeq)
    .def("simInit", &VectorizedEnvironment::settingSimulation)
    .def("visInit", &VectorizedEnvironment::initVis)
    .def("reset", &VectorizedEnvironment::reset)
    .def("resetSim", &VectorizedEnvironment::resetSim)
    .def("start", &VectorizedEnvironment::startMotion)
    .def("startPD", &VectorizedEnvironment::startMotionPD)
    .def("modeID", &VectorizedEnvironment::decideModeTransition)
    .def("step", &VectorizedEnvironment::step)
    .def("stepSim", &VectorizedEnvironment::stepSimulation)
    .def("stepSim2", &VectorizedEnvironment::stepSimulation2)
    .def("stepTest", &VectorizedEnvironment::stepTest)
    .def("control", &VectorizedEnvironment::control)
    .def("doSit", &VectorizedEnvironment::doSit)
    .def("doStand", &VectorizedEnvironment::doStand)
    .def("isMoving", &VectorizedEnvironment::isMoving)
    .def("setState", &VectorizedEnvironment::setState)
    .def("setNeutral", &VectorizedEnvironment::setNeutral)
    .def("isNeutral", &VectorizedEnvironment::isNeutral)
    .def("setDynamic", &VectorizedEnvironment::setDynamic)
    .def("getObDim", &VectorizedEnvironment::getObDim)
    .def("getActionDim", &VectorizedEnvironment::getActionDim)
    .def("getHumanSkeleton", &VectorizedEnvironment::getHumanSkeleton)
    .def("getCurrentRobot", &VectorizedEnvironment::getCurrentRobot)
    .def("getCurrentReference", &VectorizedEnvironment::getCurrentReference)
    .def("getHumanData", &VectorizedEnvironment::getHumanData)
    .def("getContactStep", &VectorizedEnvironment::getContactStep)
    .def("getStandStep", &VectorizedEnvironment::getStandStep)
    .def("getWalkStep", &VectorizedEnvironment::getWalkStep)
    .def("getSitStep", &VectorizedEnvironment::getSitStep)
    .def("getParamLocoDel", &VectorizedEnvironment::getParamLocoDel)
    .def("getParamLoco", &VectorizedEnvironment::getParamLoco)
    .def("readRecordedHumanParam", &VectorizedEnvironment::readRecordedHumanParam)
    .def("readRecordedHumanSkel", &VectorizedEnvironment::readRecordedHumanSkel)
    .def("readHumanData", &VectorizedEnvironment::readHumanData)
    .def("generateRandomSeqLoco", &VectorizedEnvironment::generateRandomSeqLoco)
    .def("generateRandomSeqMani", &VectorizedEnvironment::generateRandomSeqMani)
    .def("generateRandomSeqSit", &VectorizedEnvironment::generateRandomSeqSit)
    .def("generateSitPD", &VectorizedEnvironment::generateSitPD)
    .def("generateStandPD", &VectorizedEnvironment::generateStandPD)
    .def("updateSimObs", &VectorizedEnvironment::updateSimObs)
    .def("updateRecorded", &VectorizedEnvironment::updateRecorded)
    .def("kinectStop", &VectorizedEnvironment::KinectEmergencyStop)
    .def("closeDevice", &VectorizedEnvironment::CloseDevice)
    .def("endSim", &VectorizedEnvironment::finishSim)
    .def("end", &VectorizedEnvironment::finish);
}
