#ifndef _IKSOLVER_H
#define _IKSOLVER_H

#include <iostream>
#include <Eigen/Core>
#include "raisim/World.hpp"



class IKSolver
{
public:
  IKSolver(raisim::ArticulatedSystem* robot_, std::vector<int> frames, std::vector<Eigen::Vector3d> ends) :
    mRobot(robot_), target_frames(frames), Ends(ends)
  {
    int gcDim_ = mRobot->getGeneralizedCoordinateDim();
    int gvDim_ = mRobot->getDOF();
    gc_.setZero(gcDim_); 
    gv_.setZero(gvDim_);
  }

  void ClearFrame(){
    target_frames.clear();
  }

  void SetFrame(std::vector<int> frames){
    target_frames = frames;
  }

  void ClearTarget(){
    Ends.clear();
  }
  void SetTarget(std::vector<Eigen::Vector3d> targets_){
    Ends = targets_;
  }

  Eigen::VectorXd solve(int total_iter)
  {
    mRobot->getState(gc_, gv_);
    initpose = gc_;
    Eigen::MatrixXd J_stack = Eigen::MatrixXd::Zero(3 * Ends.size(),mRobot->getDOF());
    Eigen::VectorXd dev_stack;
    dev_stack = Eigen::VectorXd::Zero(Ends.size()*3); 
    Eigen::VectorXd newPose;
    double epsilon = 0.001;

    for(int current_iter = 0; current_iter < total_iter; current_iter++)
    {
      Eigen::VectorXd currentPose = get_current_q();
      double total_dev = 0;
      for(int i = 0 ; i < Ends.size() ; i ++){
        Eigen::Vector3d currentPart;
        currentPart = get_robot_pose(target_frames[i]);
        Eigen::Vector3d deviation = (Ends[i] - currentPart);
        total_dev += deviation.norm();
        Eigen::MatrixXd J = get_jacobian(target_frames[i]);
        J_stack.block(J.rows()*i,0,J.rows(),J.cols()) = J;
        dev_stack(3*i) = deviation[0];
        dev_stack(3*i+1) = deviation[1];
        dev_stack(3*i+2) = deviation[2];
      }
      if(total_dev < epsilon) break;
      Eigen::MatrixXd JJT = J_stack*J_stack.transpose();
      Eigen::MatrixXd pseudoJ = J_stack.transpose() * (JJT+ 0.0025 * Eigen::MatrixXd::Identity(JJT.rows(), JJT.cols())).inverse();
      Eigen::VectorXd tempPose = pseudoJ * dev_stack * 0.05;
      set_alien_pose(currentPose+tempPose);
    }   
    mRobot->getState(gc_, gv_);
    newPose = gc_;
    mRobot->setGeneralizedCoordinate(initpose);
    return newPose;
  }

  Eigen::VectorXd solve(std::vector<bool> linear, int total_iter)
  {
    mRobot->getState(gc_, gv_);
    initpose = gc_;
    Eigen::MatrixXd J_stack = Eigen::MatrixXd::Zero(3 * Ends.size(),mRobot->getDOF());
    Eigen::VectorXd dev_stack;
    dev_stack = Eigen::VectorXd::Zero(Ends.size()*3); 
    Eigen::VectorXd newPose;
    double epsilon = 0.001;

    for(int current_iter = 0; current_iter < total_iter; current_iter++)
    {
      Eigen::VectorXd currentPose = get_current_q();
      double total_dev = 0;
      for(int i = 0 ; i < Ends.size() ; i ++){
        if(linear[i]){
          Eigen::Vector3d currentPart;
          currentPart = get_robot_pose(target_frames[i]);
          Eigen::Vector3d deviation = (Ends[i] - currentPart);
          total_dev += deviation.norm();
          Eigen::MatrixXd J = get_all_fixed_jacobian(target_frames[i]);
          J_stack.block(J.rows()*i,0,J.rows(),J.cols()) = J;
          dev_stack(3*i) = deviation[0];
          dev_stack(3*i+1) = deviation[1];
          dev_stack(3*i+2) = deviation[2];
        }
        else{
          Eigen::Vector3d currentPart;
          currentPart = get_robot_rotation(target_frames[i]);
          Eigen::Vector3d deviation = (Ends[i] - currentPart);
          total_dev += deviation.norm();
          Eigen::MatrixXd J = get_all_fixed_rot_jacobian(target_frames[i]);
          J_stack.block(J.rows()*i,0,J.rows(),J.cols()) = J;
          dev_stack(3*i) = deviation[0];
          dev_stack(3*i+1) = deviation[1];
          dev_stack(3*i+2) = deviation[2];
        }
      }
      if(total_dev < epsilon) break;
      Eigen::MatrixXd JJT = J_stack*J_stack.transpose();
      Eigen::MatrixXd pseudoJ = J_stack.transpose() * (JJT+ 0.0025 * Eigen::MatrixXd::Identity(JJT.rows(), JJT.cols())).inverse();
      Eigen::VectorXd tempPose = pseudoJ * dev_stack * 0.07;
      set_alien_pose(currentPose+tempPose);
    }   
    mRobot->getState(gc_, gv_);
    newPose = gc_;
    mRobot->setGeneralizedCoordinate(initpose);
    return newPose;
  }

  void set_alien_pose(const Eigen::VectorXd pose_)
  {
    Eigen::VectorXd pose_quat(pose_.size()+1);
    for(int i = 0 ; i < 3; i++)
    {
      pose_quat[i] = pose_[i];
    }
    raisim::Vec<4> quat;
    raisim::Vec<3> axis;
    axis[0] = pose_[3]; axis[1] = pose_[4]; axis[2] = pose_[5];
    raisim::eulerVecToQuat(axis, quat);
    for(int i = 0 ; i < 4; i++)
    {
      pose_quat[i+3] = quat[i];
    }
    for(int i = 6 ; i < pose_.size(); i++)
    {
      pose_quat[i+1] = pose_[i];
    }
    mRobot->setGeneralizedCoordinate(pose_quat);
  }

  Eigen::Vector3d get_robot_pose(int frame_name)
  {
    Eigen::Vector3d pose;
    raisim::Vec<3> tempPosition;
    mRobot->getFramePosition(frame_name, tempPosition);
    pose = tempPosition.e();
    return pose;
  }

  Eigen::Vector3d get_robot_rotation(int frame_name)
  {
    raisim::Mat<3, 3> orientation_Current;
    mRobot->getFrameOrientation(frame_name, orientation_Current);
    Eigen::AngleAxisd rot(orientation_Current.e());
    Eigen::Vector3d aa = rot.angle() * rot.axis();
    return aa;
  }

  Eigen::VectorXd get_current_q(){
    Eigen::VectorXd pose;
    mRobot->getState(gc_, gv_);
    pose.setZero(gv_.size());
    for(int i = 0 ; i < 3; i++)
    {
      pose[i] = gc_[i];
    }
    Eigen::Quaterniond quat(gc_[3], gc_[4], gc_[5], gc_[6]);
    Eigen::AngleAxisd aa(quat);
    Eigen::Vector3d rotation = aa.angle()*aa.axis();

    for(int i = 0 ; i < 3; i++)
    {
      pose[i+3] = rotation[i];
    }
    for(int i = 7 ; i < gc_.size(); i++)
    {
      pose[i-1] = gc_[i];
    }
    return pose;
  }

  Eigen::MatrixXd get_jacobian(const int joint_name)
  {
    mRobot->getState(gc_, gv_);
    Eigen::MatrixXd Jacobian(3, mRobot->getDOF());
    Jacobian.setZero();
    mRobot->getDenseFrameJacobian(joint_name, Jacobian);
    return Jacobian;
  }

  Eigen::MatrixXd get_height_fixed_jacobian(const int joint_name)
  {
    mRobot->getState(gc_, gv_);
    Eigen::MatrixXd Jacobian(3, mRobot->getDOF());
    Jacobian.setZero();

    mRobot->getDenseFrameJacobian(joint_name, Jacobian);
    Jacobian.block(0,0,2,3) *= 0.01;
    Jacobian.block(2,0,1,3) *= 0.001;
    return Jacobian;
  }

  Eigen::MatrixXd get_root_fixed_jacobian(const int joint_name)
  {
    mRobot->getState(gc_, gv_);
    Eigen::MatrixXd Jacobian(3, mRobot->getDOF());
    Jacobian.setZero();

    mRobot->getDenseFrameJacobian(joint_name, Jacobian);
    Jacobian.block(0,0,2,3) *= 0.0;
    Jacobian.block(2,0,1,3) *= 0.0;
    return Jacobian;
  }

  Eigen::MatrixXd get_all_fixed_jacobian(const int joint_name)
  {
    mRobot->getState(gc_, gv_);
    Eigen::MatrixXd Jacobian(3, mRobot->getDOF());
    Jacobian.setZero();

    mRobot->getDenseFrameJacobian(joint_name, Jacobian);
    Jacobian.block(0,0,3,6) *= 0.0;
    return Jacobian;
  }

  Eigen::MatrixXd get_rot_jacobian(const int joint_name)
  {
    mRobot->getState(gc_, gv_);
    Eigen::MatrixXd Jacobian(3, mRobot->getDOF());
    Jacobian.setZero();

    auto& frame = mRobot->getFrameByIdx(joint_name);
    mRobot->getDenseRotationalJacobian(frame.parentId, Jacobian);
    return Jacobian;
  }

  Eigen::MatrixXd get_all_fixed_rot_jacobian(const int joint_name)
  {
    mRobot->getState(gc_, gv_);
    Eigen::MatrixXd Jacobian(3, mRobot->getDOF());
    Jacobian.setZero();

    auto& frame = mRobot->getFrameByIdx(joint_name);
    mRobot->getDenseRotationalJacobian(frame.parentId, Jacobian);
    Jacobian.block(0,0,3,6) *= 0.0;
    return Jacobian;
  }

private:
  raisim::ArticulatedSystem* mRobot;
  std::vector<Eigen::Vector3d> Ends;
  std::vector<int> target_frames;
  Eigen::VectorXd gc_, gv_;
  Eigen::VectorXd initpose;
};
#endif