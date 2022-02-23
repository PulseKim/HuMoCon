//
// Created by jemin on 11/12/19.
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

#include "Environment.hpp"
#include "VectorizedEnvironment.hpp"

using namespace raisim;

int main(int argc, char *argv[]) {
  // RSFATAL_IF(argc != 3, "got "<<argc<<" arguments. "<<"This executable takes three arguments: 1. resource directory, 2. configuration file")

  std::string resourceDir(argv[1]), cfgFile(argv[2]);
  std::ifstream myfile (cfgFile);
  std::string config_str, line;
  bool escape = false;

  while (std::getline(myfile, line)) {
    if(line == "environment:") {
      escape = true;
      while (std::getline(myfile, line)) {
        if(line.substr(0, 2) == "  ")
          config_str += line.substr(2) + "\n";
        else if (line[0] == '#')
          continue;
        else
          break;
      }
    }
    if(escape)
      break;
  }
  config_str.pop_back();
  VectorizedEnvironment<ENVIRONMENT> vecEnv(resourceDir, config_str);
  vecEnv.init();

  Yaml::Node config;
  Yaml::Parse(config, config_str);

  EigenRowMajorMat observation(config["num_envs"].template As<int>(), vecEnv.getObDim());
  EigenRowMajorMat action(config["num_envs"].template As<int>(), vecEnv.getActionDim());
  EigenVec reward(config["num_envs"].template As<int>(), 1);
  EigenBoolVec dones(config["num_envs"].template As<int>(), 1);
  EigenRowMajorMat extra_info(config["environment"]["num_envs"].template As<int>(), vecEnv.getExtraInfoDim());
  action.setZero();

  Eigen::Ref<EigenRowMajorMat> ob_ref(observation), action_ref(action), extra_info_ref(extra_info);
  Eigen::Ref<EigenVec> reward_ref(reward);
  Eigen::Ref<EigenBoolVec> dones_ref(dones);

  vecEnv.reset(ob_ref);
  vecEnv.step(action_ref, ob_ref, reward_ref, dones_ref, extra_info_ref);

  return 0;
}