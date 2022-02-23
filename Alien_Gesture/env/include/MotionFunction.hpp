#ifndef _MOTION_FUNCTION_H
#define _MOTION_FUNCTION_H

#include <stdlib.h>
#include <iostream>
#include <Eigen/Core>
#include "raisim/World.hpp"
#include "math.h"
#include "AnalyticIK.hpp"
#include "RobotInfo.hpp"
#include "EnvMath.hpp"

class MotionReshaper
{
public:
  MotionReshaper(raisim::ArticulatedSystem* robot_, float fps_ = 30) : 
  mRobot(robot_), fps(fps_){
    gcDim_ = mRobot->getGeneralizedCoordinateDim();
    prev_pose.setZero(gcDim_);
    raw_pose.setZero(gcDim_);
    reshaped_pose.setZero(gcDim_);
    prev_kin_pose.setZero(gcDim_);
    dyn_pose.setZero(gcDim_);
    dyn_prev_pose.setZero(gcDim_);
    mAIK = new AnalyticFullIK(mRobot);
    JointVelocityLimit = 21.;
    threshold = 0.5;
    twoside = -1;
    // 0 : Over / 1: Contact
    contactCFlag.setZero();
    startFlag = true;
  }

  bool isInside(Eigen::Vector3d point, Eigen::VectorXd pose)
  {
    point[2] = 0;
    mRobot->setGeneralizedCoordinate(pose);
    std::vector<int> contact_feet;
    if(contactCFlag[0] == 1) contact_feet.push_back(0);
    if(contactCFlag[1] == 1) contact_feet.push_back(1);
    if(contactCFlag[3] == 1) contact_feet.push_back(3);
    if(contactCFlag[2] == 1) contact_feet.push_back(2);
  
    float foot_radius = 0.001;
    Eigen::Vector3d supportCentre = Eigen::Vector3d::Zero();

    std::vector<Eigen::Vector3d> poly;
    if (nContact == 0) return false;
    else if(nContact == 1)
    {
      raisim::Vec<3> foot1;
      foot1[2] = 0;
      mRobot->getFramePosition(FR_FOOT + 3 * contact_feet[0], foot1);
      float dist = (point - foot1.e()).norm();
      return dist <= foot_radius;
    }
    else if(nContact == 2)
    { 
      raisim::Vec<3> foot1;
      mRobot->getFramePosition(FR_FOOT + 3 * contact_feet[0], foot1);
      raisim::Vec<3> foot2;
      mRobot->getFramePosition(FR_FOOT + 3 * contact_feet[1], foot2);
      foot1[2] = 0; foot2[2] = 0;
      supportCentre = (foot1.e() + foot2.e())/2;
      float dist = distToSegment(point, foot1.e(), foot2.e());
      return dist <= foot_radius;
    }
    else
    {
      supportCentre.setZero();
      for(int i = 0; i < contact_feet.size(); ++i)
      {
        raisim::Vec<3> foot;
        mRobot->getFramePosition(FR_FOOT + 3 * contact_feet[i], foot);
        foot[2] = 0;
        supportCentre += foot.e();
      }
      supportCentre /= contact_feet.size();
      for(int i = 0; i < contact_feet.size(); ++i)
      {
        raisim::Vec<3> foot;
        mRobot->getFramePosition(FR_FOOT + 3 * contact_feet[i], foot);
        Eigen::Vector3d current = foot.e();
        poly.push_back(current);
      }
    }
    return isInPoly(point, poly);
  }

  bool isInPoly(Eigen::Vector3d point, std::vector<Eigen::Vector3d> poly)
  {
    int crosses = 0;
    for(int i = 0 ; i < poly.size() ; i++){
      int j = (i+1)%poly.size();
      if((poly[i][1] > point[1]) != (poly[j][1] > point[1]) ){
        double atX = (poly[j][0]- poly[i][0])*(point[1]-poly[i][1])/(poly[j][1]-poly[i][1])+poly[i][0];
        if(point[0] < atX)
          crosses++;
      }
    }
    poly.clear();
    poly.shrink_to_fit();
    return crosses % 2 > 0;
  }

  void contactInformation(Eigen::Vector4i contactCFlag_)
  {
    contactCFlag = contactCFlag_;
    nContact = contactCFlag.squaredNorm();
    if(nContact == 2)
    {
      if(contactCFlag[0] == 1 && contactCFlag[2] == 1) twoside = 0;
      if(contactCFlag[0] == 1 && contactCFlag[3] == 1) twoside = 1;
      if(contactCFlag[1] == 1 && contactCFlag[2] == 1) twoside = 1;
      if(contactCFlag[1] == 1 && contactCFlag[3] == 1) twoside = 0;
    }
  }

  void getCurrentRawPose(Eigen::VectorXd pose_)
  {
    raw_pose = pose_;
  }

  void getPreviousRealPose(Eigen::VectorXd pose_)
  {
    prev_pose = pose_;
    if(startFlag){
      prev_kin_pose = prev_pose;
      startFlag = false;
    }
  }

  void getDynStates(Eigen::VectorXd pose0, Eigen::VectorXd pose1)
  {
    dyn_pose = pose0;
    dyn_prev_pose = pose1;
  }

  void getAngularVelocity(Eigen::Vector3d rot)
  {
    rotVel = rot;
  }

  void dynContactAdjustment()
  {
    if(nContact == 2 && twoside == 1) rotationalContactAdjustment();
    comContactAdjustment();
  }

  void rotationalContactAdjustment()
  {
    // Assume that we only check this condition on the two diagonal case : twoside == 1
    float velThresh = deg2rad(15);
    raisim::Vec<3> fr;
    raisim::Vec<3> rr;
    raisim::Vec<3> fh;
    raisim::Vec<3> rh;
    mRobot->setGeneralizedCoordinate(dyn_pose);

    if(contactCFlag[0] == 1){
      mRobot->getFramePosition(FR_HIP, fh);
      mRobot->getFramePosition(FL_HIP, fr);
      mRobot->getFramePosition(RR_HIP, rr);
      mRobot->getFramePosition(RL_HIP, rh);
    }
    else{
      mRobot->getFramePosition(FR_HIP, fr);
      mRobot->getFramePosition(FL_HIP, fh);
      mRobot->getFramePosition(RR_HIP, rh);
      mRobot->getFramePosition(RL_HIP, rr);
    }
    Eigen::Vector3d axis = (fh.e() - rh.e()).normalized();
    float theta = (axis[0] * rotVel[0] + axis[1] * rotVel[1] + axis[2] * rotVel[2]);
    float gap = 0.08;
    if(contactCFlag[0] == 1) {
      if(fr[2] - rr[2] > gap * 1.2) contactCFlag[2] = 1;
      if(rr[2] - fr[2] > gap * 1.2) contactCFlag[1] = 1;
    }
    else
    {
      if(rr[2] - fr[2] > gap* 1.2) contactCFlag[0] = 1;
      if(fr[2] - rr[2] > gap* 1.2) contactCFlag[3] = 1;
    }
    if(abs(theta) < velThresh) return;
    if(contactCFlag[0] == 1){
      if(theta > velThresh &&  fr[2] - rr[2] > gap) contactCFlag[2] = 1;
      if(theta < -velThresh && rr[2] - fr[2] > gap) contactCFlag[1] = 1;
    }
    else
    {
      if(theta > velThresh && rr[2] - fr[2] > gap) contactCFlag[0] = 1;
      if(theta < -velThresh && fr[2] - rr[2] > gap) contactCFlag[3] = 1;
    }
  }

  void comContactAdjustment()
  {
    // Have to implement more stable contact data
    if(isInside(COM_solver(dyn_pose), dyn_pose)) return;
    mRobot->setGeneralizedCoordinate(dyn_prev_pose);
    // if(isInside(dynCOM, dyn_pose)) return;
    if(nContact == 4) return;
    else if(nContact ==3 )
    {
      // Determine whether to get the rest foot down or not
      if(isInside(COM_solver(dyn_pose), dyn_pose))
      {
        contactCFlag = Eigen::Vector4i(1,1,1,1);
        return;
      }
      float dist_prev = 10000;
      float dist_cur = 10000;
      std::vector<int> contact_feet;
      if(contactCFlag[0] == 1) contact_feet.push_back(0);
      if(contactCFlag[1] == 1) contact_feet.push_back(1);
      if(contactCFlag[3] == 1) contact_feet.push_back(3);
      if(contactCFlag[2] == 1) contact_feet.push_back(2);
      Eigen::Vector3d prevCOM = COM_solver(dyn_prev_pose);
      for(int i = 0; i < 3; ++i)
      {
        raisim::Vec<3> foot1;
        mRobot->getFramePosition(FR_FOOT + 3 * contact_feet[i], foot1);
        raisim::Vec<3> foot2;
        mRobot->getFramePosition(FR_FOOT + 3 * contact_feet[(i+1)%3], foot2);
        prevCOM[2] = 0; foot1[2] = 0; foot2[2] = 0;
        float cur = distToSegment(prevCOM, foot1.e(), foot2.e());
        if(dist_prev > cur) dist_prev = cur;
      }
      Eigen::Vector3d currentCOM = COM_solver(dyn_pose);
      currentCOM[2] = 0;
      for(int i = 0; i < 3; ++i)
      {
        raisim::Vec<3> foot1;
        mRobot->getFramePosition(FR_FOOT + 3 * contact_feet[i], foot1);
        raisim::Vec<3> foot2;
        mRobot->getFramePosition(FR_FOOT + 3 * contact_feet[(i+1)%3], foot2);
        currentCOM[2] = 0; foot1[2] = 0 ; foot2[2] = 0;
        float cur = distToSegment(currentCOM, foot1.e(), foot2.e());
        if(dist_cur > cur) dist_cur = cur;
      }
      if(dist_cur > dist_prev)
      {
        contactCFlag = Eigen::Vector4i(1,1,1,1);
        return;
      }
      else return;
    }
    else if(nContact == 2)
    {
      if(twoside == 0)
      {
        std::vector<int> contact_feet;
        std::vector<int> free_feet;
        if(contactCFlag[0] == 1) contact_feet.push_back(0);
        else free_feet.push_back(0);
        if(contactCFlag[1] == 1) contact_feet.push_back(1);
        else free_feet.push_back(1);
        if(contactCFlag[3] == 1) contact_feet.push_back(3);
        else free_feet.push_back(3);
        if(contactCFlag[2] == 1) contact_feet.push_back(2);
        else free_feet.push_back(2);
        if(isInside(COM_solver(dyn_prev_pose), dyn_pose))
        {
          Eigen::Vector3d currentCOM = COM_solver(dyn_pose);
          raisim::Vec<3> side1;
          raisim::Vec<3> side2;
          mRobot->getFramePosition(FR_HIP + free_feet[0], side1);
          mRobot->getFramePosition(FR_HIP + free_feet[1], side2);
          currentCOM[2] = 0; side1[2] = 0; side2[2] = 0;
          float first = (currentCOM - side1.e()).norm();
          float second = (currentCOM - side2.e()).norm();
          if(first < second) contact_feet.push_back(free_feet[0]);
          else contact_feet.push_back(free_feet[1]);
          for(int i = 0 ; i < contact_feet.size(); ++i)
          {
            contactCFlag[contact_feet[i]] = 1;
          }
          return;
        }
        // Here COM violation
        Eigen::Vector3d prevCOM = COM_solver(dyn_prev_pose);
        Eigen::Vector3d currentCOM = COM_solver(dyn_pose);
        raisim::Vec<3> foot1;
        raisim::Vec<3> foot2;
        prevCOM[2] = 0; currentCOM[2] = 0; foot1[2] = 0; foot2[2] = 0;
        mRobot->getFramePosition(FR_FOOT + 3 * contact_feet[0], foot1);
        mRobot->getFramePosition(FR_FOOT + 3 * contact_feet[1], foot2);
        float dist_prev = distToSegment(prevCOM, foot1.e(), foot2.e());
        mRobot->getFramePosition(FR_FOOT + 3 * contact_feet[0], foot1);
        mRobot->getFramePosition(FR_FOOT + 3 * contact_feet[1], foot2);
        float dist_cur = distToSegment(currentCOM, foot1.e(), foot2.e());
        if(dist_cur > dist_prev)
        {
          raisim::Vec<3> side1;
          raisim::Vec<3> side2;
          mRobot->getFramePosition(FR_HIP + free_feet[0], side1);
          mRobot->getFramePosition(FR_HIP + free_feet[1], side2);
          currentCOM[2] = 0; side1[2] = 0; side2[2] = 0;
          float first = (currentCOM - side1.e()).norm();
          float second = (currentCOM - side2.e()).norm();
          if(first < second) contact_feet.push_back(free_feet[0]);
          else contact_feet.push_back(free_feet[1]);
          for(int i = 0 ; i < contact_feet.size(); ++i)
          {
            contactCFlag[contact_feet[i]] = 1;
          }
          return;
        }
      }
      else return;
    }
  } 

   Eigen::VectorXd contactConsistency(Eigen::VectorXd pose_)
  {
    Eigen::VectorXd sol(gcDim_);
    sol = pose_;
    std::vector<Eigen::Vector3d> feetPose;
    raisim::Vec<3> tempPositionA;
    mRobot->setGeneralizedCoordinate(prev_pose);

    float lowest = 100.0;
    for(int i = 0 ; i < 4; ++i){
      mRobot->getFramePosition(FR_FOOT + 3 * i, tempPositionA);
      if(lowest >tempPositionA[2]) lowest = tempPositionA[2];
    }
    for(int i = 0 ; i < 4; ++i){
      mRobot->getFramePosition(FR_FOOT + 3 * i, tempPositionA);
      tempPositionA[2] = lowest;
      feetPose.push_back(tempPositionA.e());
    }

    for(int i = 0; i < 4; ++i)
    {
      if(contactCFlag[i] == 0) continue;
      if((i % 2) == 0)
      {
        AnalyticRIK* rik = new AnalyticRIK(mRobot, int(i/2));
        rik->setReady();
        rik->setCurrentPose(pose_);
        rik->setTarget(feetPose[i]);
        rik->solveIK();
        sol.segment(7 + 6 * (i/2),3) = rik->getSolution();
        rik->reset();
        delete rik;
      }
      else
      {
        AnalyticLIK* lik = new AnalyticLIK(mRobot, int(i/2));
        lik->setReady();
        lik->setCurrentPose(pose_);
        lik->setTarget(feetPose[i]);
        lik->solveIK();
        sol.segment(10 + 6 * (i/2),3) =lik->getSolution();
        lik->reset();
        delete lik;
      }
    }
    return sol;
  }

  Eigen::VectorXd velocity_adjustment(Eigen::VectorXd src)
  {
    Eigen::VectorXd temp = src;
    float max_dev = 0.95 * JointVelocityLimit / fps;
    for(int i = 7; i < temp.size(); ++i)
    {
      float dev = temp[i] - prev_kin_pose[i];
      if(dev > max_dev) temp[i] = prev_kin_pose[i] + max_dev;
      else if(-max_dev > dev) temp[i] = prev_kin_pose[i] - max_dev;
    }
    return temp;
  }

  Eigen::VectorXd getReshapedMotion()
  {
    return reshaped_pose;
  }

  Eigen::VectorXd solveAnalyticIK()
  {
    Eigen::VectorXd sol(gcDim_);
    sol.head(7) = raw_pose.head(7);
    mAIK->setCurrentPose(raw_pose);
    mAIK->setReady();  
    raisim::Vec<3> tempPositionA;
    mRobot->setGeneralizedCoordinate(prev_pose);
    for(int i = 0 ; i < 4; ++i){
      mRobot->getFramePosition(FR_FOOT + 3 * i, tempPositionA);
      mAIK->appendTargets(tempPositionA.e());
    }
    mAIK->solve(raw_pose);
    sol.tail(12) = mAIK->getSolution();
    mAIK->clearTargets();
    return sol;
  }


  Eigen::Vector3d COM_solver(Eigen::VectorXd pose)
  {
    Eigen::Vector3d com = Eigen::Vector3d::Zero();
    // mRobot->printOutBodyNamesInOrder();
    mRobot->setGeneralizedCoordinate(pose);
    for(int i = 0; i < 13; ++i){
      raisim::Vec<3> bodypose;
      mRobot->getPosition(i, bodypose);
      Eigen::Vector3d curr = mRobot->getMass(i) * bodypose.e();
      com += curr;
    }
    com /= mRobot->getTotalMass();
    // com = pose.head(3);
    return com;
  }

  // Attract com doesn't work instead, we should attract COP
  Eigen::VectorXd adjustCenter(Eigen::VectorXd pose)
  {
    mRobot->setGeneralizedCoordinate(pose);

  }

  Eigen::Vector3d getDCM(Eigen::Vector3d COM, Eigen::Vector3d prev)
  {

  }

  Eigen::VectorXd attractCOM(Eigen::VectorXd pose)
  {
    mRobot->setGeneralizedCoordinate(pose);
    Eigen::Vector3d mCOM = COM_solver(pose);
    mCOM[2] = 0;
    if(isInside(mCOM, prev_pose)) return pose;
    else
    {
      float tractor = 0.001;
      Eigen::Vector3d centre;
      int cnt = 0;
      for(int i = 0; i < 4; ++i)
      {
        if(contactCFlag[i] == 1)
        {
          cnt++;
          raisim::Vec<3> foot;
          mRobot->getFramePosition(FR_FOOT + 3 * i, foot);
          centre += foot.e();
        }
      }
      centre /= cnt;
      centre[2] = 0;
      Eigen::VectorXd solPose = pose;

      for(int i = 0 ; i < 100; ++i)
      {
        Eigen::Vector3d dir = solPose.head(3) + tractor * (centre - mCOM).normalized();
        solPose.head(3) = dir;
        mRobot->setGeneralizedCoordinate(solPose);
        mCOM = COM_solver(solPose);
        mCOM[2] = 0;
        if(isInside(mCOM, prev_pose)) break;
      }
      return solPose;
    }
  }

  void getKinematicFixedPose()
  {
    Eigen::VectorXd mid_result = raw_pose;
    dynContactAdjustment();
    // mid_result = attractCOM(mid_result);
    mid_result = contactConsistency(mid_result);
    mid_result = velocity_adjustment(mid_result);
    twoside = -1;
    // Finalize the solution
    reshaped_pose = mid_result;
    prev_kin_pose = reshaped_pose;
  }

  void getKinematicFixedPoseTest()
  {
    Eigen::VectorXd mid_result = raw_pose;
    // dynContactAdjustment();
    // mid_result = attractCOM(mid_result);
    mid_result = contactConsistency(mid_result);
    mid_result = velocity_adjustment(mid_result);
    twoside = -1;
    // Finalize the solution
    reshaped_pose = mid_result;
    prev_kin_pose = reshaped_pose;
  }


  void reshapeMotion()
  {
    Eigen::VectorXd mid_result = raw_pose;
    // 1) Using Foot fixing constraint

    // 2) Velocity constraint ==>  Reshape
    mid_result = velocity_adjustment(mid_result);


    // Finalize the solution
    reshaped_pose = mid_result;
  }



private:
  float threshold;
  raisim::ArticulatedSystem* mRobot;
  Eigen::VectorXd prev_kin_pose;
  Eigen::VectorXd prev_pose, dyn_pose, dyn_prev_pose, raw_pose, reshaped_pose;
  float JointVelocityLimit;
  bool startFlag;
  int gcDim_;
  float fps;
  std::vector<int> r_frames; 
  std::vector<Eigen::Vector3d> r_targets;
  AnalyticFullIK* mAIK;
  Eigen::Vector4i contactCFlag;
  Eigen::Vector3d rotVel;
  int nContact;
  // Cross = 0, oneside = 1, else = -1
  int twoside;
};

#endif