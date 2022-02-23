#ifndef  _ROOT_PREDICTOR_HPP_
#define  _ROOT_PREDICTOR_HPP_

#include <stdio.h>
#include <iostream>
#include <Eigen/Core>
#include "raisim/World.hpp"
#include "math.h"
#include "RobotInfo.hpp"

class RootPredictor
{
public:
  RootPredictor(){}
  RootPredictor(raisim::ArticulatedSystem* robot):
  mRobot(robot)
  {
    gcDim_ = mRobot->getGeneralizedCoordinateDim();
    prev_pose.setZero(gcDim_);
    current_pose.setZero(gcDim_);
    current_pose[0] = 0;current_pose[1] = 0;current_pose[2] = 0.4;
    current_pose[3] = 1;current_pose[4] = 0;current_pose[5] = 0;current_pose[6] = 0; 
    startFlag = true;
  }

  void reset()
  {
    startFlag = true;
  }

  void setPrevPose(Eigen::VectorXd pose)
  {
    current_pose = pose;
    startFlag = false;
  }

  void predict(Eigen::VectorXd gc_joints)
  {
    // Set the current joint pose and prev joint pose
    if(startFlag)
    {
      current_pose.tail(12) = gc_joints;
      prev_pose = current_pose;
      startFlag = false;
    }
    else{
      prev_pose = current_pose;
      current_pose.tail(12) = gc_joints;
    }
    // std::cout << "initialized" << std::endl;
    // FK twice and save the position and min foot location
    std::vector<Eigen::Vector3d> prev_feet;
    std::vector<Eigen::Vector3d> current_feet;
    float prev_min=1000, current_min=1000;
    mRobot->setGeneralizedCoordinate(prev_pose);
    for(int i =0; i < 4; ++i)
    {
      raisim::Vec<3> foot;
      mRobot->getFramePosition(FR_FOOT + 3*i, foot);
      if(foot[2] < prev_min)
        prev_min = foot[2];
      prev_feet.push_back(foot.e());
    }
    mRobot->setGeneralizedCoordinate(current_pose);
    for(int i =0; i < 4; ++i)
    {
      raisim::Vec<3> foot;
      mRobot->getFramePosition(FR_FOOT + 3*i, foot);
      if(foot[2] < current_min)
        current_min = foot[2];
      current_feet.push_back(foot.e());
    }
    current_height = current_pose[2]-current_min + 0.01;
    // std::cout << "fks" << std::endl;
    // Calculate the pinned foot 
    Eigen::Vector4d prev_pinned = Eigen::Vector4d::Zero();
    Eigen::Vector4d current_pinned = Eigen::Vector4d::Zero();
    float confident = 0.005;
    for(int i =0; i < 4; ++i)
    {
      if(prev_feet[i][2] < prev_min + confident)
        prev_pinned[i] = 1;
      if(current_feet[i][2] < current_min + confident)
        current_pinned[i] = 1;
    }
    Eigen::Vector4d pinned = prev_pinned.cwiseProduct(current_pinned);
    // std::cout << "pinned" << std::endl;
    // Case 2) Some feet are fixed
    // Case 1) No foot is fixed
    float delX = 0, delZ = 0;
    if(pinned == Eigen::Vector4d::Zero())
    {
      // Have to implement here 
    }
    else
    {
      int cnt = 0;
      for(int i = 0; i <4; ++i)
      {
        int curr = int(pinned[i]);
        cnt += curr;
        delX += curr * (prev_feet[i][0] - current_feet[i][0]);
        delZ += curr * (prev_feet[i][2] - current_feet[i][2]);
      }
      delX /= cnt;
      delZ /= cnt;
    }
    dX = delX;
    dZ = delZ;
    // Clear the memory
    prev_feet.clear();
    prev_feet.shrink_to_fit();
    current_feet.clear();
    current_feet.shrink_to_fit();
  }

  void predictFull(Eigen::VectorXd gc_joints)
  {
    // Set the current joint pose and prev joint pose
    if(startFlag)
    {
      current_pose.tail(16) = gc_joints;
      prev_pose = current_pose;
      startFlag = false;
    }
    else{
      prev_pose = current_pose;
      current_pose.tail(16) = gc_joints;
    }
    // std::cout << "initialized" << std::endl;
    // FK twice and save the position and min foot location
    std::vector<Eigen::Vector3d> prev_feet;
    std::vector<Eigen::Vector3d> current_feet;
    float prev_min=1000, current_min=1000;
    mRobot->setGeneralizedCoordinate(prev_pose);
    for(int i =0; i < 4; ++i)
    {
      raisim::Vec<3> foot;
      mRobot->getFramePosition(FR_FOOT + 3*i, foot);
      if(foot[2] < prev_min)
        prev_min = foot[2];
      prev_feet.push_back(foot.e());
    }
    mRobot->setGeneralizedCoordinate(current_pose);
    for(int i =0; i < 4; ++i)
    {
      raisim::Vec<3> foot;
      mRobot->getFramePosition(FR_FOOT + 3*i, foot);
      if(foot[2] < current_min)
        current_min = foot[2];
      current_feet.push_back(foot.e());
    }
    current_height = current_pose[2]-current_min + 0.01;
    // std::cout << "fks" << std::endl;
    // Calculate the pinned foot 
    Eigen::Vector4d prev_pinned = Eigen::Vector4d::Zero();
    Eigen::Vector4d current_pinned = Eigen::Vector4d::Zero();
    float confident = 0.005;
    for(int i =0; i < 4; ++i)
    {
      if(prev_feet[i][2] < prev_min + confident)
        prev_pinned[i] = 1;
      if(current_feet[i][2] < current_min + confident)
        current_pinned[i] = 1;
    }
    Eigen::Vector4d pinned = prev_pinned.cwiseProduct(current_pinned);
    // std::cout << "pinned" << std::endl;
    // Case 2) Some feet are fixed
    // Case 1) No foot is fixed
    float delX = 0, delZ = 0;
    if(pinned == Eigen::Vector4d::Zero())
    {
      // Have to implement here 
    }
    else
    {
      int cnt = 0;
      for(int i = 0; i <4; ++i)
      {
        int curr = int(pinned[i]);
        cnt += curr;
        delX += curr * (prev_feet[i][0] - current_feet[i][0]);
        delZ += curr * (prev_feet[i][2] - current_feet[i][2]);
      }
      delX /= cnt;
      delZ /= cnt;
    }
    dX = delX;
    dZ = delZ;
    // Clear the memory
    prev_feet.clear();
    prev_feet.shrink_to_fit();
    current_feet.clear();
    current_feet.shrink_to_fit();
  }

  float getDX(){return dX;}
  float getDY(){return dY;}
  float getDZ(){return dZ;}
  float getHeight(){return current_height;}

private:
  raisim::ArticulatedSystem* mRobot;
  Eigen::VectorXd prev_pose, current_pose;
  int gcDim_;
  bool startFlag;
  float dX, dY, dZ;
  float current_height;
};

#endif