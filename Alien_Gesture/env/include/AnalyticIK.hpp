#ifndef _ANAYLYTIC_IK_H
#define _ANAYLYTIC_IK_H

#include <stdlib.h>
#include <iostream>
#include <Eigen/Core>
#include "raisim/World.hpp"
#include "math.h"
// #include "EnvMath.hpp"
#include "RobotInfo.hpp"

class AnalyticRIK
{
public:
  AnalyticRIK(raisim::ArticulatedSystem* robot_, int front_back): 
  mRobot(robot_){
    int gcDim_ = mRobot->getGeneralizedCoordinateDim();
    int gvDim_ = mRobot->getDOF();
    gc_.setZero(gcDim_);
    gv_.setZero(gvDim_); 
    mInitialPositions = get_current_q();
    mSolution = Eigen::Vector3d(0,0,0);
    jointLimits = mRobot->getJointLimits();
    // 0 when front /1 when rear
    mHipIdx = front_back;
  }
  void setHipIdx(int front_back){mHipIdx = front_back;}

  void setCurrentPose(Eigen::VectorXd pose)
  {
    mInitialPositions = pose;
    mRobot->setGeneralizedCoordinate(pose);
  }

  void setReady()
  {
    // mInitialPositions = get_current_q();
    mTargetGlobal.setZero();
    mTarget.setZero();
  }

  void setManualJointLimit(int idx, float x_l, float x_u)
  {
    jointLimits[idx][0] = x_l;
    jointLimits[idx][1] = x_u;
  }

  void resetJointLimit()
  {
    jointLimits = mRobot->getJointLimits();
  }

  const Eigen::Vector3d& getTargets(){return mTarget;}
  void clearTarget(){
    mTargetGlobal.setZero();
    mTarget.setZero();
  }
  void setTarget(Eigen::Vector3d target_){
    mTargetGlobal = target_;
  }

  const Eigen::Vector3d& getSolution(){
    return mSolution;
  }
  void setSolution(const Eigen::Vector3d& sol){
    mSolution = sol;
  }

  ~AnalyticRIK(){}

  void reset()
  {
    mRobot->setState(mInitialPositions, gv_);
  }

  void solveIK()
  {
    setRefZeroPose();
    make_local();
    setRefZeroRot();
    solve_calf();
    solve_thigh();
    solve_hip();
    reset();
  }

  void make_local()
  {
    // R_inv_robot * (global_target - global_hip_location)
    raisim::Mat<3,3> orientation_r;
    mRobot->getBaseOrientation(orientation_r);
    Eigen::Matrix3d rot = orientation_r.e();
    raisim::Vec<3> hip_location;
    mRobot->getFramePosition(FR_HIP + 2 * mHipIdx, hip_location);
    mTarget = rot.inverse() * (mTargetGlobal - hip_location.e());
  }

  void setRefZeroPose()
  {
    // Set Reference robot to zero position to align 
    mRobot->getState(gc_, gv_);
    Eigen::VectorXd zero_pose = gc_;
    // zero_pose[3] = 1;
    // zero_pose[4] = 0;
    // zero_pose[5] = 0;
    // zero_pose[6] = 0;
    for(int i = 0; i < 4; ++i)
    {
      zero_pose[7 + 3* i] = 0;
      zero_pose[7 + 3* i + 1] = 0;
      zero_pose[7 + 3* i + 2] = -1.9;
    }
    set_robot_pose(zero_pose);
  }
  void setRefZeroRot()
  {
    // Set Reference robot to zero position to align 
    mRobot->getState(gc_, gv_);
    Eigen::VectorXd zero_pose = gc_;
    zero_pose[3] = 1;
    zero_pose[4] = 0;
    zero_pose[5] = 0;
    zero_pose[6] = 0;
    set_robot_pose(zero_pose);
  }


  void solve_calf()
  {
    // l2 = norm of local pose
    // Compensate with Perpendicular line
    // cos theta_c_comp = a^2 + b^2 -c^2 / 2ab
    // thetac = -pi + theta_c_comp
    // if thetac out of range, clip
    raisim::Vec<3> thigh_location;
    mRobot->getFramePosition(FR_THIGH + 6 * mHipIdx, thigh_location);
    raisim::Vec<3> calf_location;
    mRobot->getFramePosition(FR_CALF + 6 * mHipIdx, calf_location);
    raisim::Vec<3> foot_location;
    mRobot->getFramePosition(FR_FOOT + 6 * mHipIdx, foot_location);

    len_tc = (thigh_location.e() - calf_location.e()).norm();
    len_cf = (calf_location.e() - foot_location.e()).norm();
    Eigen::Vector3d reduced_target = mTarget;
    // Get perpendicular solution
    float det = pow(len_ht,4) * pow(mTarget[2],2) - ((pow(mTarget[1],2) + pow(mTarget[2],2)) * ((pow(len_ht,4) - pow(mTarget[1],2)*pow(len_ht,2))));
    float y1, y2;
    if(det >= 0.0){
      y1 = (pow(len_ht,2) * mTarget[2] - sqrt(det)) 
          /(pow(mTarget[1],2)+ pow(mTarget[2],2));
      y2 = (pow(len_ht,2) * mTarget[2] + sqrt(det)) 
          /(pow(mTarget[1],2)+ pow(mTarget[2],2));      
    }
    else if(det > -1E-5){
      y1 = (pow(len_ht,2) * mTarget[2]) 
          /(pow(mTarget[1],2)+ pow(mTarget[2],2));
      y2 = (pow(len_ht,2) * mTarget[2]) 
            /(pow(mTarget[1],2)+ pow(mTarget[2],2));  
    }
    else{
      y1 = -1E3;
      y2 = -1E3;
    }
    if(y1 > len_ht * sin(jointLimits[7 + 6 *mHipIdx - 1][0]) 
      && y1 < len_ht * sin(jointLimits[7 + 6 *mHipIdx - 1][1]))
    {
      float x1 = (pow(len_ht,2) - mTarget[2] * y1) / mTarget[1];
      if(x1 > len_ht * cos(jointLimits[7 + 6 *mHipIdx - 1][1]))
      {
        reduced_target[1] -= x1;
        reduced_target[2] -= y1;
      }
      else if(y2 > len_ht * sin(jointLimits[7 + 6 *mHipIdx - 1][0]) 
        && y2 < len_ht * sin(jointLimits[7 + 6 *mHipIdx - 1][1]))
      {
        float x2 = (pow(len_ht,2) - mTarget[2] * y2) / mTarget[1];
        if(x2 > len_ht * cos(jointLimits[7 + 6 *mHipIdx - 1][1]))
        {
          reduced_target[1] -= x2;
          reduced_target[2] -= y2;
        }
      }
    }
    else if(y2 > len_ht * sin(jointLimits[7 + 6 *mHipIdx - 1][0]) 
      && y2 < len_ht * sin(jointLimits[7 + 6 *mHipIdx - 1][1]))
    {
      float x2 = (pow(len_ht,2) - mTarget[2] * y2) / mTarget[1];
      if(x2 > len_ht * cos(jointLimits[7 + 6 *mHipIdx - 1][1]))
      {
        reduced_target[1] -= x2;
        reduced_target[2] -= y2;
      }
      else
      {
        Eigen::Vector3d prj_Target(0,mTarget[1], mTarget[2]);
        Eigen::Vector3d one(0, len_ht * cos(jointLimits[7 + 6 *mHipIdx - 1][0]),len_ht * sin(jointLimits[7 + 6 *mHipIdx - 1][0]));
        Eigen::Vector3d two(0, len_ht * cos(jointLimits[7 + 6 *mHipIdx - 1][1]),len_ht * sin(jointLimits[7 + 6 *mHipIdx - 1][1]));
        float norm_one = (prj_Target - one).norm();
        float norm_two = (prj_Target - two).norm();
        if(norm_one < norm_two) reduced_target = reduced_target - one;
        else reduced_target = reduced_target - two;
      }
    }
    else
    {
      Eigen::Vector3d prj_Target(0,mTarget[1], mTarget[2]);
      Eigen::Vector3d one(0, len_ht * cos(jointLimits[7 + 6 *mHipIdx - 1][0]),len_ht * sin(jointLimits[7 + 6 *mHipIdx - 1][0]));
      Eigen::Vector3d two(0, len_ht * cos(jointLimits[7 + 6 *mHipIdx - 1][1]),len_ht * sin(jointLimits[7 + 6 *mHipIdx - 1][1]));
      float norm_one = (prj_Target - one).norm();
      float norm_two = (prj_Target - two).norm();
      if(norm_one < norm_two) reduced_target = reduced_target - one;
      else reduced_target = reduced_target - two;

    }
    len_tf = reduced_target.norm();
    float val = (len_tc * len_tc + len_cf * len_cf - len_tf * len_tf) / (2 * len_tc * len_cf);
    if(val >1) val = 1;
    if(val < -1) val = -1;
    float theta_c_comp = acos(val);
    float thetac = -M_PI + theta_c_comp;
    if(jointLimits[7 + 6 *mHipIdx + 1][0] > thetac)thetac = jointLimits[7 + 6 *mHipIdx + 1][0];
    else if(jointLimits[7 + 6 *mHipIdx + 1][1] < thetac) thetac = jointLimits[7 + 6 *mHipIdx + 1][1];
    mSolution[2] = thetac;
    set_robot_part(7 + 6 *mHipIdx+2, thetac);
    mRobot->getFramePosition(FR_THIGH + 6 * mHipIdx, thigh_location);
    mRobot->getFramePosition(FR_FOOT + 6 * mHipIdx, foot_location);
    len_tf = (thigh_location.e() - foot_location.e()).norm();
  }

  void solve_thigh()
  {
    // zero_comp = acos(l2/ |pose_foot.z|)
    // theta_th = thetha_th_ + zero_comp
    // l2 sin theta2_half = xp
    // theta2_half = asin xp/l2
    // if coordinate z < 0 : thetha_th_ = theta_half / else: PI-theta_half or -PI-theta_half
    // if theta_th > pi : theta_th -= 2PI
    // zero_comp = acos(len_tf / std::abs());
    raisim::Vec<3> thigh_location;
    raisim::Vec<3> foot_location;
    mRobot->getFramePosition(FR_THIGH + 6 * mHipIdx, thigh_location);
    mRobot->getFramePosition(FR_FOOT + 6 * mHipIdx, foot_location);
    float val = abs(foot_location[2] - thigh_location[2])/len_tf;
    if(val > 1) val = 1;
    if(val < -1) val =-1;
    zero_comp = acos(val);
    val = mTarget[0] / len_tf;
    if(val > 1) val = 1;
    if(val < -1) val = -1;
    float theta_th_half = -asin(val);
    if(mTarget[2] < 0) theta_th_prime = theta_th_half;
    else{
      if(theta_th_half > 0) theta_th_prime = M_PI - theta_th_half;
      else theta_th_prime = -M_PI - theta_th_half;
    }
    float theta_th = theta_th_prime + zero_comp;

    if(theta_th < jointLimits[10 + 6 *mHipIdx][0]) {
      if(theta_th + 2 * M_PI < jointLimits[10 + 6 *mHipIdx][1])
        theta_th = theta_th + 2 * M_PI;
      else
        theta_th = jointLimits[10 + 6 *mHipIdx][0];
    }    if(theta_th > jointLimits[7 + 6 *mHipIdx][1]) theta_th = jointLimits[7 + 6 *mHipIdx][1];
    mSolution[1] = theta_th;
  }

  void solve_hip()
  {
    // Here, c2 = cos theta_th_
    // Solve sin theta_h = (z * l1 + y *l2c2) / (l1^2 + (l2c2)^2)
    float val = -(mTarget[2] * len_ht - mTarget[1] * len_tf * cos(theta_th_prime)) 
          / (len_ht*len_ht + len_tf * cos(theta_th_prime)*len_tf * cos(theta_th_prime));
    if(val > 1) val = 1;
    if(val < -1) val =-1;     
    float theta_hip = asin(val);
    if(jointLimits[7 + 6 *mHipIdx - 1][0] > theta_hip) theta_hip = jointLimits[7 + 6 *mHipIdx - 1][0];
    else if(jointLimits[7 + 6 *mHipIdx - 1][1] < theta_hip) theta_hip = jointLimits[7 + 6 *mHipIdx - 1][1];
    mSolution[0] = theta_hip;
  }

  Eigen::VectorXd get_current_q()
  {
    mRobot->getState(gc_, gv_);
    return gc_;
  }


  void set_robot_pose(const Eigen::VectorXd pose_)
  {
    mRobot->setGeneralizedCoordinate(pose_);
  }

  void set_robot_part(int idx, float angle)
  {
    mRobot->getState(gc_, gv_);
    Eigen::VectorXd pose = gc_;
    pose[idx] = angle;
    mRobot->setGeneralizedCoordinate(pose);
  }

protected:
  AnalyticRIK();
  raisim::ArticulatedSystem* mRobot;
  Eigen::VectorXd mSavePositions;
  Eigen::VectorXd mInitialPositions;
  Eigen::Vector3d mTarget;
  Eigen::Vector3d mTargetGlobal;
  Eigen::Vector3d mSolution;
  Eigen::VectorXd gc_, gv_;
  std::vector<raisim::Vec<2>> jointLimits;
  int mHipIdx;
  float theta_th_prime, zero_comp;
  float len_tc, len_cf, len_tf;
  float len_ht = 0.083;
};

class AnalyticLIK
{
public:
  AnalyticLIK(raisim::ArticulatedSystem* robot_, int front_back): 
  mRobot(robot_){
    int gcDim_ = mRobot->getGeneralizedCoordinateDim();
    int gvDim_ = mRobot->getDOF();
    gc_.setZero(gcDim_);
    gv_.setZero(gvDim_); 
    mInitialPositions = get_current_q();
    mSolution = Eigen::Vector3d(0,0,0);
    jointLimits = mRobot->getJointLimits();
    // 0 when front /1 when rear
    mHipIdx = front_back;
  }

  void setHipIdx(int front_back){mHipIdx = front_back;}

  void setCurrentPose(Eigen::VectorXd pose)
  {
    mInitialPositions = pose;
    mRobot->setGeneralizedCoordinate(pose);
  }


  void setReady()
  {
    // mInitialPositions = get_current_q();
    mTargetGlobal.setZero();
    mTarget.setZero();
  }

  void setManualJointLimit(int idx, float x_l, float x_u)
  {
    jointLimits[idx][0] = x_l;
    jointLimits[idx][1] = x_u;
  }

  void resetJointLimit()
  {
    jointLimits = mRobot->getJointLimits();
  }

  const Eigen::Vector3d& getTargets(){return mTarget;}
  void clearTarget(){
    mTargetGlobal.setZero();
    mTarget.setZero();
  }
  void setTarget(Eigen::Vector3d target_){
    mTargetGlobal = target_;
  }

  const Eigen::Vector3d& getSolution(){
    return mSolution;
  }
  void setSolution(const Eigen::Vector3d& sol){
    mSolution = sol;
  }

  ~AnalyticLIK(){}

  void reset()
  {
    mRobot->setState(mInitialPositions, gv_);
  }

  void solveIK()
  {
    setRefZeroPose();
    make_local();
    setRefZeroRot();
    solve_calf();
    solve_thigh();
    solve_hip();
    reset();
  }

  void make_local()
  {
    // R_inv_robot * (global_target - global_hip_location)
    raisim::Mat<3,3> orientation_r;
    mRobot->getBaseOrientation(orientation_r);
    Eigen::Matrix3d rot = orientation_r.e();
    raisim::Vec<3> hip_location;
    mRobot->getFramePosition(FL_HIP + 2 * mHipIdx, hip_location);
    mTarget = rot.inverse() * (mTargetGlobal - hip_location.e());
  }

  void setRefZeroPose()
  {
    // Set Reference robot to zero position to align 
    mRobot->getState(gc_, gv_);
    Eigen::VectorXd zero_pose = gc_;
    // zero_pose[3] = 1;
    // zero_pose[4] = 0;
    // zero_pose[5] = 0;
    // zero_pose[6] = 0;
    for(int i = 0; i < 4; ++i)
    {
      zero_pose[7 + 3* i] = 0;
      zero_pose[7 + 3* i + 1] = 0;
      zero_pose[7 + 3* i + 2] = -1.9;
    }
    set_robot_pose(zero_pose);
  }
  void setRefZeroRot()
  {
    // Set Reference robot to zero position to align 
    mRobot->getState(gc_, gv_);
    Eigen::VectorXd zero_pose = gc_;
    zero_pose[3] = 1;
    zero_pose[4] = 0;
    zero_pose[5] = 0;
    zero_pose[6] = 0;
    set_robot_pose(zero_pose);
  }


  void solve_calf()
  {
    // l2 = norm of local pose
    // Compensate with Perpendicular line
    // cos theta_c_comp = a^2 + b^2 -c^2 / 2ab
    // thetac = -pi + theta_c_comp
    // if thetac out of range, clip
    raisim::Vec<3> thigh_location;
    mRobot->getFramePosition(FL_THIGH + 6 * mHipIdx, thigh_location);
    raisim::Vec<3> calf_location;
    mRobot->getFramePosition(FL_CALF + 6 * mHipIdx, calf_location);
    raisim::Vec<3> foot_location;
    mRobot->getFramePosition(FL_FOOT + 6 * mHipIdx, foot_location);

    len_tc = (thigh_location.e() - calf_location.e()).norm();
    len_cf = (calf_location.e() - foot_location.e()).norm();
    Eigen::Vector3d reduced_target = mTarget;
    // Get perpendicular solution
    float det = pow(len_ht,4) * pow(mTarget[2],2) - ((pow(mTarget[1],2) + pow(mTarget[2],2)) * ((pow(len_ht,4) - pow(mTarget[1],2)*pow(len_ht,2))));
    float y1, y2;
    if(det >= 0.0){
      y1 = (pow(len_ht,2) * mTarget[2] - sqrt(det)) 
          /(pow(mTarget[1],2)+ pow(mTarget[2],2));
      y2 = (pow(len_ht,2) * mTarget[2] + sqrt(det)) 
          /(pow(mTarget[1],2)+ pow(mTarget[2],2));      
    }
    else if(det > -1E-5){
      y1 = (pow(len_ht,2) * mTarget[2]) 
          /(pow(mTarget[1],2)+ pow(mTarget[2],2));
      y2 = (pow(len_ht,2) * mTarget[2]) 
            /(pow(mTarget[1],2)+ pow(mTarget[2],2));  
    }
    else{
      y1 = -1E3;
      y2 = -1E3;
    }
    if(y1 > len_ht * sin(jointLimits[10 + 6 *mHipIdx - 1][0]) 
      && y1 < len_ht * sin(jointLimits[10 + 6 *mHipIdx - 1][1]))
    {
      float x1 = (pow(len_ht,2) - mTarget[2] * y1) / mTarget[1];
      if(x1 > len_ht * cos(jointLimits[10 + 6 *mHipIdx - 1][1]))
      {
        reduced_target[1] -= x1;
        reduced_target[2] -= y1;
      }
      else if(y2 > len_ht * sin(jointLimits[10 + 6 *mHipIdx - 1][0]) 
        && y2 < len_ht * sin(jointLimits[10 + 6 *mHipIdx - 1][1]))
      {
        float x2 = (pow(len_ht,2) - mTarget[2] * y2) / mTarget[1];
        if(x2 > len_ht * cos(jointLimits[10 + 6 *mHipIdx - 1][1]))
        {
          reduced_target[1] -= x2;
          reduced_target[2] -= y2;
        }
      }
    }
    else if(y2 > len_ht * sin(jointLimits[10 + 6 *mHipIdx - 1][0]) 
      && y2 < len_ht * sin(jointLimits[10 + 6 *mHipIdx - 1][1]))
    {
      float x2 = (pow(len_ht,2) - mTarget[2] * y2) / mTarget[1];
      if(x2 > len_ht * cos(jointLimits[10 + 6 *mHipIdx - 1][1]))
      {
        reduced_target[1] -= x2;
        reduced_target[2] -= y2;
      }
      else
      {
        Eigen::Vector3d prj_Target(0,mTarget[1], mTarget[2]);
        Eigen::Vector3d one(0, len_ht * cos(jointLimits[10 + 6 *mHipIdx - 1][0]),len_ht * sin(jointLimits[10 + 6 *mHipIdx - 1][0]));
        Eigen::Vector3d two(0, len_ht * cos(jointLimits[10 + 6 *mHipIdx - 1][1]),len_ht * sin(jointLimits[10 + 6 *mHipIdx - 1][1]));
        float norm_one = (prj_Target - one).norm();
        float norm_two = (prj_Target - two).norm();
        if(norm_one < norm_two) reduced_target = reduced_target - one;
        else reduced_target = reduced_target - two;
      }
    }
    else
    {
      Eigen::Vector3d prj_Target(0,mTarget[1], mTarget[2]);
      Eigen::Vector3d one(0, len_ht * cos(jointLimits[10 + 6 *mHipIdx - 1][0]),len_ht * sin(jointLimits[10 + 6 *mHipIdx - 1][0]));
      Eigen::Vector3d two(0, len_ht * cos(jointLimits[10 + 6 *mHipIdx - 1][1]),len_ht * sin(jointLimits[10 + 6 *mHipIdx - 1][1]));
      float norm_one = (prj_Target - one).norm();
      float norm_two = (prj_Target - two).norm();
      if(norm_one < norm_two) reduced_target = reduced_target - one;
      else reduced_target = reduced_target - two;

    }
    len_tf = reduced_target.norm();
    float val = (len_tc * len_tc + len_cf * len_cf - len_tf * len_tf) / (2 * len_tc * len_cf);
    if(val >1) val = 1;
    if(val < -1) val = -1;
    float theta_c_comp = acos(val);
    float thetac = -M_PI + theta_c_comp;
    if(jointLimits[10 + 6 *mHipIdx + 1][0] > thetac)thetac = jointLimits[10 + 6 *mHipIdx + 1][0];
    else if(jointLimits[10 + 6 *mHipIdx + 1][1] < thetac) thetac = jointLimits[10 + 6 *mHipIdx + 1][1];
    mSolution[2] = thetac;
    set_robot_part(10 + 6 *mHipIdx+2, thetac);
    mRobot->getFramePosition(FL_THIGH + 6 * mHipIdx, thigh_location);
    mRobot->getFramePosition(FL_FOOT + 6 * mHipIdx, foot_location);
    len_tf = (thigh_location.e() - foot_location.e()).norm();
  }

  void solve_thigh()
  {
    // zero_comp = acos(l2/ |pose_foot.z|)
    // theta_th = thetha_th_ + zero_comp
    // l2 sin theta2_half = xp
    // theta2_half = asin xp/l2
    // if coordinate z < 0 : thetha_th_ = theta_half / else: PI-theta_half or -PI-theta_half
    // if theta_th > pi : theta_th -= 2PI
    // zero_comp = acos(len_tf / std::abs());
    raisim::Vec<3> thigh_location;
    raisim::Vec<3> foot_location;
    mRobot->getFramePosition(FL_THIGH + 6 * mHipIdx, thigh_location);
    mRobot->getFramePosition(FL_FOOT + 6 * mHipIdx, foot_location);
    float val = abs(foot_location[2] - thigh_location[2])/len_tf;
    if(val > 1) val = 1;
    if(val < -1) val =-1;
    zero_comp = acos(val);
    val = -mTarget[0] / len_tf;
    if(val > 1) val = 1;
    if(val < -1) val = -1;
    float theta_th_half = asin(val);
    if(mTarget[2] < 0) theta_th_prime = theta_th_half;
    else{
      if(theta_th_half > 0) theta_th_prime = M_PI - theta_th_half;
      else theta_th_prime = -M_PI - theta_th_half;
    }
    float theta_th = theta_th_prime + zero_comp;
    if(theta_th < jointLimits[10 + 6 *mHipIdx][0]) {
      if(theta_th + 2 * M_PI < jointLimits[10 + 6 *mHipIdx][1])
        theta_th = theta_th + 2 * M_PI;
      else
        theta_th = jointLimits[10 + 6 *mHipIdx][0];
    }
    if(theta_th > jointLimits[10 + 6 *mHipIdx][1]) theta_th = jointLimits[10 + 6 *mHipIdx][1];
    mSolution[1] = theta_th;
  }

  void solve_hip()
  {
    // Here, c2 = cos theta_th_
    // Solve sin theta_h = (z * l1 + y *l2c2) / (l1^2 + (l2c2)^2)
    float val = (mTarget[2] * len_ht + mTarget[1] * len_tf * cos(theta_th_prime)) 
          / (len_ht*len_ht + len_tf * cos(theta_th_prime)*len_tf * cos(theta_th_prime));
    if(val > 1) val = 1;
    if(val < -1) val =-1;     
    float theta_hip = asin(val);
    if(jointLimits[10 + 6 *mHipIdx - 1][0] > theta_hip) theta_hip = jointLimits[10 + 6 *mHipIdx - 1][0];
    else if(jointLimits[10 + 6 *mHipIdx - 1][1] < theta_hip) theta_hip = jointLimits[10 + 6 *mHipIdx - 1][1];
    mSolution[0] = theta_hip;
  }

  Eigen::VectorXd get_current_q()
  {
    mRobot->getState(gc_, gv_);
    return gc_;
  }


  void set_robot_pose(const Eigen::VectorXd pose_)
  {
    mRobot->setGeneralizedCoordinate(pose_);
  }

  void set_robot_part(int idx, float angle)
  {
    mRobot->getState(gc_, gv_);
    Eigen::VectorXd pose = gc_;
    pose[idx] = angle;
    mRobot->setGeneralizedCoordinate(pose);
  }

protected:
  AnalyticLIK();
  raisim::ArticulatedSystem* mRobot;
  Eigen::VectorXd mSavePositions;
  Eigen::VectorXd mInitialPositions;
  Eigen::Vector3d mTarget;
  Eigen::Vector3d mTargetGlobal;
  Eigen::Vector3d mSolution;
  Eigen::VectorXd gc_, gv_;
  std::vector<raisim::Vec<2>> jointLimits;
  int mHipIdx;
  float theta_th_prime, zero_comp;
  float len_tc, len_cf, len_tf;
  float len_ht = 0.083;
};

class AnalyticFullIK
{
public:
  AnalyticFullIK(raisim::ArticulatedSystem* robot_) : 
  mRobot(robot_){
    FRIK = new AnalyticRIK(mRobot,0);
    RRIK = new AnalyticRIK(mRobot,1);
    FLIK = new AnalyticLIK(mRobot,0);
    RLIK = new AnalyticLIK(mRobot,1);
  }
  ~AnalyticFullIK(){}

  void setReady()
  {
    FRIK->setReady(); RRIK->setReady(); FLIK->setReady(); RLIK->setReady();
    FRIK->resetJointLimit(); RRIK->resetJointLimit(); FLIK->resetJointLimit(); RLIK->resetJointLimit();
  }

  void setFullTargets(std::vector<Eigen::Vector3d> targets)
  { 
    global_targets = targets;
  }

  void appendTargets(Eigen::Vector3d target)
  { 
    global_targets.push_back(target);
  }

  void clearTargets()
  {
    global_targets.clear();
    global_targets.shrink_to_fit();
    solutions.clear();
    solutions.shrink_to_fit();
    FRIK->clearTarget();
    FLIK->clearTarget();
    RRIK->clearTarget();
    RLIK->clearTarget();
  }

  Eigen::VectorXd getSolution()
  {
    Eigen::VectorXd sol(12);
    for(int i =0; i< 4 ; ++i)
    {
      Eigen::Vector3d current = solutions[i];
      for(int j = 0; j < 3; ++j){
        sol[3*i + j] = current[j];
      }
    }
    return sol;
  }

  void setCurrentPose(Eigen::VectorXd pose)
  {
    FRIK->setCurrentPose(pose);
    FLIK->setCurrentPose(pose);
    RRIK->setCurrentPose(pose);
    RLIK->setCurrentPose(pose);
  }

  void solve(Eigen::VectorXd pose)
  {
    FRIK->setCurrentPose(pose);
    FRIK->setTarget(global_targets[0]);
    FRIK->solveIK();
    solutions.push_back(FRIK->getSolution());
    FLIK->setCurrentPose(pose);
    FLIK->setTarget(global_targets[1]);
    FLIK->solveIK();
    solutions.push_back(FLIK->getSolution());
    RRIK->setCurrentPose(pose);
    RRIK->setTarget(global_targets[2]);
    RRIK->solveIK();
    solutions.push_back(RRIK->getSolution());
    RLIK->setCurrentPose(pose);
    RLIK->setTarget(global_targets[3]);
    RLIK->solveIK();
    solutions.push_back(RLIK->getSolution());
  }

protected:
  AnalyticFullIK();
  raisim::ArticulatedSystem* mRobot;
  AnalyticRIK* FRIK;  AnalyticLIK* FLIK;
  AnalyticRIK* RRIK;  AnalyticLIK* RLIK;
  std::vector<Eigen::Vector3d> global_targets;
  std::vector<Eigen::Vector3d> solutions;
};



#endif