//
// Created by jemin on 4/11/19.
// MIT License
//
// Copyright (c) 2019-2019 Robotic Systems Lab, ETH Zurich
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef RAISIMOGREVISUALIZER_RAISIMKEYBOARDCALLBACK_HPP
#define RAISIMOGREVISUALIZER_RAISIMKEYBOARDCALLBACK_HPP

#include "raisim/OgreVis.hpp"
#include "raisimKeyboardCallback.hpp"
#include "guiState.hpp"

bool mStopFlag = false;
bool mMotionFlag = false;

bool raisimKeyboardCallback(const OgreBites::KeyboardEvent &evt) {
  auto &key = evt.keysym.sym;
  // termination gets the highest priority
  switch (key) {
    case '1':
      raisim::gui::showBodies = !raisim::gui::showBodies;
      std::cout << "1 is pressed" << std::endl; 
      break;
    case '2':
      raisim::gui::showCollision = !raisim::gui::showCollision;
      break;
    case '3':
      raisim::gui::showContacts = !raisim::gui::showContacts;
      break;
    case '4':
      raisim::gui::showForces = !raisim::gui::showForces;
      break;
    case ']':
      // std::cout << "stop" << std::endl;
      mStopFlag = !mStopFlag;
      break;
    case 's':
      mMotionFlag = !mMotionFlag;
      break;
    default:
      break;
  }
  return false;
}

#endif //RAISIMOGREVISUALIZER_RAISIMKEYBOARDCALLBACK_HPP
