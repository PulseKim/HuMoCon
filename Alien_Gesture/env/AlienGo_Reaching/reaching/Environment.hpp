#include <stdlib.h>
#include <cstdint>
#include <set>
#include <chrono>
#include <random>
#include <fstream>
#include <deque>
#include <raisim/OgreVis.hpp>
#include "math.h"
#include "RaisimGymEnv.hpp"
#include "../include/visSetupCallback2.hpp"
#include "../include/Utilities.h" 
#include "../include/RobotInfo.hpp"
#include "../include/AnalyticIK.hpp"
#include "../include/EnvMath.hpp"
#include "../include/TrajectoryGenerator.hpp"
#include "../include/GaitMapper.hpp"

#include <k4arecord/playback.h>
#include <k4a/k4a.h>
#include <k4abt.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"

#include "../visualizer_alien/raisimKeyboardCallback.hpp"
#include "../visualizer_alien/helper.hpp"
#include "../visualizer_alien/guiState.hpp"
#include "../visualizer_alien/raisimBasicImguiPanel.hpp"
#include "../visualizer_alien/BodyTrakingHelper.hpp"


// This code is for learning PD-gain with a task
// All joint's maximum torque is limited to 44.4N m
// Curriculum learning is used
// Need to automate curriculum learning


namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:
  struct InputSettings
  {
    k4a_depth_mode_t DepthCameraMode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    bool CpuOnlyMode = false;
    bool Offline = false;
    std::string FileName;
  };


  explicit ENVIRONMENT(const std::string& resourceDir, const YAML::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) 
  {
    /// add objects
    alien_ = world_->addArticulatedSystem(resourceDir_+"/urdf/aliengo.urdf");
    ref_dummy = new raisim::ArticulatedSystem(resourceDir_+"/urdf/aliengo.urdf");
    alien_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    auto ground = world_->addGround();
    world_->setERP(0,0);
    /// get robot data
    gcDim_ = alien_->getGeneralizedCoordinateDim();
    gvDim_ = alien_->getDOF();
    nJoints_ = 12;

    READ_YAML(double, control_dt_, cfg["control_dt"])

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    torque_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);
    
    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.360573, 1.0, 0.0, 0.0, 0.0, 0.000356295, 0.765076, -1.5303, 5.69796e-05, 0.765017, -1.53019, 0.000385732, 0.765574, -1.53102, 8.34913e-06, 0.765522, -1.53092;
    // gc_init_ << 0, 0, 0.43, 1.0, 0.0, 0.0, 0.0, 0.045377, 0.667921, -1.23225,0.045377, 0.667921, -1.23225, 0.045377, 0.667921, -1.23225, 0.045377, 0.667921, -1.23225;
    alien_->setState(gc_init_, gv_init_);
    ref_dummy->setState(gc_init_, gv_init_);

    /// set pd gains
    jointPgain.setZero(gvDim_); jointPgain.tail(nJoints_).setConstant(init_pgain);
    jointDgain.setZero(gvDim_); jointDgain.tail(nJoints_).setConstant(1.4 * std::sqrt(init_pgain));
    alien_->setPdGains(jointPgain, jointDgain);
    alien_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    Eigen::VectorXd torque_upperlimit = alien_->getUpperLimitForce().e();
    Eigen::VectorXd torque_lowerlimit = alien_->getLowerLimitForce().e();
    torque_upperlimit.tail(nJoints_).setConstant(effort_limit);
    torque_lowerlimit.tail(nJoints_).setConstant(-effort_limit);
    alien_->setActuationLimits(torque_upperlimit, torque_lowerlimit);

    // des angleR, current angleR, des angleL, current angleL ,N joints 
    obDim_ = 97;
    actionDim_ = nJoints_ * 2;
    actionMean_.setZero(nJoints_); actionStd_.setZero(nJoints_);
    obMean_.setZero(obDim_); obStd_.setZero(obDim_);

    /// action & observation scaling
    actionMean_ = gc_init_.tail(nJoints_);
    // actionStd_.setConstant(0.6);
    actionStd_ << 0.2, 0.8, 0.6, 0.6, 0.8, 0.6, 0.2, 0.8, 0.6, 0.2, 0.8, 0.6;

    freq_w = 0.15;    
    stable_w = 0.1;
    main_w = 1.0 - freq_w - stable_w;

    freq_scale = 1e-4;
    angle_scale = 10;
    end_scale = 100;
    root_scale = 5;
    support_scale = 400;
    torque_scale = 1e-4;

    current_count = 0;
    change_count = 30;
    force_count = 60;
    scaler = -1.0 / 2000;

    end_names.push_back("FR_foot_fixed");
    // end_names.push_back("FL_foot_fixed");
    end_names.push_back("RR_foot_fixed");
    end_names.push_back("RL_foot_fixed");

    // gui::rewardLogger.init({"AngleReward", "EndReward", "SupportReward", "RotReward", "TorqueReward"});
    gui::rewardLogger.init({"AngleReward", "EndReward", "SupportReward", "RotReward", "TorqueReward", "FreqReward"});

    /// indices of links that should not make contact with ground
    footIndices_.insert(alien_->getBodyIdx("FL_calf"));
    footIndices_.insert(alien_->getBodyIdx("FR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RL_calf"));

    desired_footIndices_.push_back(alien_->getBodyIdx("FR_calf"));
    desired_footIndices_.push_back(alien_->getBodyIdx("RR_calf"));
    desired_footIndices_.push_back(alien_->getBodyIdx("RL_calf"));
    
    pp.setZero();

    for(int i = 0; i < end_names.size(); ++i)
    {
      raisim::Vec<3> tempPosition;
      alien_->getFramePosition(alien_->getFrameByName(end_names[i]), tempPosition);
      feetPoses.push_back(tempPosition.e());
    }

    force[0] = 0; force[1]=0; force[2] = 0;
    mIK = new AnalyticLIK(ref_dummy, 0);

    hip_distribution = std::uniform_real_distribution<double> (0.1, 0.1);
    thigh_distribution = std::uniform_real_distribution<double> (deg2rad(80), deg2rad(80));
    calf_distribution = std::uniform_real_distribution<double>(-1.9, -1.9);
    force_distribution = std::uniform_real_distribution<double> (0.0, 1.0);
    xaxis_distribution = std::uniform_real_distribution<double> (-1.0, 1.0);
    yaxis_distribution = std::uniform_real_distribution<double> (-0.4, 0.4);
    zaxis_distribution = std::uniform_real_distribution<double> (-0.2, 0.2);
    init_pose_flag = true;

    for(int i = 0; i < nJoints_; ++i)
    {
      std::deque<double> zero_(32, 0.0);
      torque_sequence.push_back(zero_);
      zero_.shrink_to_fit();
    }

    /// visualize if it is the first environment
    if (visualizable_) {
      auto vis = raisim::OgreVis::get();

      /// these method must be called before initApp
      vis->setWorld(world_.get());
      vis->setWindowSize(1800, 1200);
      vis->setImguiSetupCallback(imguiSetupCallback);
      vis->setImguiRenderCallback(imguiRenderCallBack);
      vis->setKeyboardCallback(raisimKeyboardCallback);
      vis->setSetUpCallback(setupCallback);
      vis->setAntiAliasing(2);
      /// starts visualizer thread
      vis->initApp();

      anymalVisual_ = vis->createGraphicalObject(alien_, "AlienGo");
      // vis->createGraphicalObject(ref_dummy, "dummy");
      vis->createGraphicalObject(ground, 10, "floor", "checkerboard_green");
      vis->addVisualObject("sphere_indicator", "sphereMesh", "green", {0.03, 0.03, 0.03}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("com", "sphereMesh", "blueEmit", {0.02, 0.02, 0.02}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("supportPoly0", "cylinderMesh", "red", {0.01, 0.01, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("supportPoly1", "cylinderMesh", "red", {0.01, 0.01, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("supportPoly2", "cylinderMesh", "red", {0.01, 0.01, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      // vis->addVisualObject("force", "arrowMesh", "red", {0.1, 0.1, 0.2}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);

      auto& list = raisim::OgreVis::get()->getVisualObjectList();
      for(int i = 0; i < 3; ++i){
        int j;
        if(i==2) j = 0;
        else j = i+1;
        Eigen::Vector3d half = (feetPoses[j] + feetPoses[i]) / 2 ;
        float len = (feetPoses[j] - feetPoses[i]).norm();
        raisim::Mat<3,3> rot;
        Eigen::Vector3d way = (feetPoses[j] - feetPoses[i]).normalized();
        raisim::Vec<3> direction = {way[0], way[1], way[2]};
        raisim::zaxisToRotMat(direction, rot);

        list["supportPoly"+ std::to_string(i)].setPosition(half);
        list["supportPoly"+ std::to_string(i)].setScale(Eigen::Vector3d(0.01, 0.01, len));
        list["supportPoly"+ std::to_string(i)].setOrientation(rot);
      }
      desired_fps_ = 1.0/ control_dt_;
      vis->setDesiredFPS(desired_fps_);
    }
  }

  ~ENVIRONMENT() final = default;

  void init() final { }

  void kinect_init() final
  {
    InputSettings inputSettings;
    SetFromDevice(inputSettings);

    auto vis = raisim::OgreVis::get();
    for(int i =0 ; i < 6;++i){ 
      vis->addVisualObject("sphere" + std::to_string(i), "sphereMesh", "orange", {0.02, 0.02, 0.02}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
    }
    for(int i =0 ; i < 5; ++i){
      vis->addVisualObject("bone" + std::to_string(i), "cylinderMesh", "red", {0.015, 0.015, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
    }
    vis->addVisualObject("sphere_indicator_pr", "sphereMesh", "orange", {0.03, 0.03, 0.03}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);

    connection.push_back(3);
    connection.push_back(4);
    connection.push_back(5);
    connection.push_back(0);
    connection.push_back(1);
    connection.push_back(2);
    raisim::Vec<3> tempPosition;
    ref_dummy->getFramePosition(ref_dummy->getFrameByName("FL_foot_fixed"), tempPosition);
    pp = tempPosition.e();
    image = cv::Mat::ones(colorHeight,colorWidth,CV_8UC3);
    namedWindow("Human_Motion", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("Human_Motion", 1800, 1200);
    cv::Size frame_size(colorWidth, colorHeight);
    vod = cv::VideoWriter("/home/sonic/Project/Alien_Gesture/Human.avi", 
    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), desired_fps_, frame_size, true);
    prev_foot =  pp;
    startFlag = true;
    deviceFlag = false;
    for(int i = 0; i < 6; ++i)
      points.push_back(Eigen::Vector3d(0,0,0));
    put_boxes();

  }

  void put_boxes()
  {
    std::vector<raisim::Box*> cubes;
    float mass = 0.1;
    auto vis = raisim::OgreVis::get();

    raisim::SingleBodyObject *ob = nullptr;
    cubes.push_back(world_->addBox(0.1, 0.5, 0.25, 1000 * mass));
    vis->createGraphicalObject(cubes.back(), "cubes1", "blue");
    ob = cubes.back();
    Eigen::Vector3d pose;
    pose[0] = 0.550;
    pose[1] = 0.0;
    pose[2] = 0.23/2;
    ob->setPosition(pose);
    cubes.push_back(world_->addBox(0.08, 0.08, 0.08, mass));
    vis->createGraphicalObject(cubes.back(), "cubes2", "red");
    ob = cubes.back();
    pose[1] = 0.1;
    pose[2] = 0.23 + 0.041;
    ob->setPosition(pose);
    cubes.push_back(world_->addBox(0.08, 0.08, 0.08, mass));
    vis->createGraphicalObject(cubes.back(), "cubes3", "orange");
    ob = cubes.back();
    pose[1] = -0.1;
    pose[2] = 0.23 + 0.041;
    ob->setPosition(pose);
  }


  void reset() final
  {
    alien_->setState(gc_init_, gv_init_);
    ref_dummy ->setState(gc_init_, gv_init_);
    
    current_count = 0;
    curriculum_cnt = 0;
    desired_angleH = gc_init_[LEFT_HIP];
    desired_angleT = gc_init_[LEFT_THIGH];
    desired_angleC = gc_init_[LEFT_CALF];

    force[0] = 0; force[1]=0; force[2] = 0;
    pTarget_.setZero();
    pTarget_.tail(nJoints_) = gc_init_.tail(12);
    alien_->setPdTarget(pTarget_, vTarget_);
    mCOP = alien_->getCompositeCOM().e();
    gc_prev1 = gc_init_;
    jointPgain.tail(nJoints_).setConstant(init_pgain);
    jointDgain.tail(nJoints_).setConstant(1.4 * std::sqrt(init_pgain));
    updateObservation();

    if(visualizable_)
      gui::rewardLogger.clean();
  }

  float stepRef() final
  {
    return 0;
  }

  float step(const Eigen::Ref<EigenVec>& action) final {

    /// action scaling
    Eigen::VectorXd currentAction_PD = action.cast<double>();
    Eigen::VectorXd currentAction(nJoints_);
    currentAction = currentAction_PD.head(nJoints_);
    Eigen::VectorXd rawGain = currentAction_PD.tail(nJoints_) * pGainScaler;
    Eigen::VectorXd meanShift(nJoints_);
    meanShift.setConstant(pGainShift);
    rawGain = rawGain + meanShift;
    for(int i = 0; i < rawGain.size();++i)
      if(rawGain[i] < 0.0) rawGain[i] = 0.0;
    jointPgain.tail(nJoints_) = rawGain;
    jointDgain.tail(nJoints_) = jointPgain.tail(nJoints_).cwiseSqrt();
    jointDgain = jointDgain * 1.4;

    alien_->setPdGains(jointPgain, jointDgain);
    pTarget12_ = currentAction;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += 0.3 * actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;
    pTarget_[LEFT_HIP] += desired_angleH;
    pTarget_[LEFT_THIGH] += desired_angleT;
    pTarget_[LEFT_CALF] += desired_angleC;
    alien_->setPdTarget(pTarget_, vTarget_);

    auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
    auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);    
    int p = current_count%force_count;

    for(int i=0; i<loopCount; i++) {
      world_->integrate();  
      alien_->getState(gc_, gv_);
      CenterOfPressure();
      if(p < 30){
        raisim::Vec<3> leg_temp;
        alien_->getFramePosition(alien_->getFrameByName("FL_foot_fixed"), leg_temp);                
        alien_->setExternalForce(6, alien_->Frame::WORLD_FRAME, force, alien_->Frame::WORLD_FRAME, leg_temp);
      }
      Eigen::VectorXd ref_pose = gc_;
      ref_pose[LEFT_HIP] = desired_angleH;
      ref_pose[LEFT_THIGH] = desired_angleT;
      ref_pose[LEFT_CALF] = desired_angleC;
      ref_dummy->setGeneralizedCoordinate(ref_pose);

      for(int i = 0; i < end_names.size(); ++i)
      {
        raisim::Vec<3> tempPosition2;
        alien_->getFramePosition(alien_->getFrameByName(end_names[i]), tempPosition2);
        feetPoses[i] = tempPosition2.e();
      }

      if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
      {
        auto vis = raisim::OgreVis::get();
        auto& list = raisim::OgreVis::get()->getVisualObjectList();

        raisim::Vec<3> tempPosition;
        ref_dummy->getFramePosition(ref_dummy->getFrameByName("FL_foot_fixed"), tempPosition);
        list["sphere_indicator"].setPosition(tempPosition);
        // Eigen::Vector3d com = alien_->getCompositeCOM().e();
        // com[2] = 0;
        // list["com"].setPosition(com);
        list["com"].setPosition(mCOP);

        for(int i = 0; i < 3; ++i){
          int j;
          if(i==2) j = 0;
          else j = i+1;
          Eigen::Vector3d half = (feetPoses[j] + feetPoses[i]) / 2 ;
          float len = (feetPoses[j] - feetPoses[i]).norm();
          raisim::Mat<3,3> rot;
          Eigen::Vector3d way = (feetPoses[j] - feetPoses[i]).normalized();
          raisim::Vec<3> direction = {way[0], way[1], way[2]};
          raisim::zaxisToRotMat(direction, rot);

          list["supportPoly"+ std::to_string(i)].setPosition(half);
          list["supportPoly"+ std::to_string(i)].setScale(Eigen::Vector3d(0.01, 0.01, len));
          list["supportPoly"+ std::to_string(i)].setOrientation(rot);
        }
        vis->renderOneFrame();
      }      
      visualizationCounter_++;
    }

    updateObservation();

    float totalReward = CalcReward();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("AngleReward", angle_reward);
      gui::rewardLogger.log("EndReward",  end_reward);
      gui::rewardLogger.log("SupportReward",  support_reward);
      gui::rewardLogger.log("RotReward",  root_reward);
      gui::rewardLogger.log("TorqueReward",  torque_reward);
      gui::rewardLogger.log("FreqReward",  freq_reward);

      /// reset camera
      auto vis = raisim::OgreVis::get();

      vis->select(anymalVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }
    
    if(current_count%change_count == 0){
      update_random_leaning_angle();
      desired_angleHC = desired_angleH;
      desired_angleTC = desired_angleT;
      desired_angleCC = desired_angleC;
      // current_count = 0;
    }
    interpolate_desired();
    current_count++;

    if(p == 0){
      exert_random_force();  
    }

    if(!mStopFlag) 
      return totalReward;
    else
      return -1E10;
  }

  float stepTest(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    Eigen::VectorXd currentAction_PD = action.cast<double>();
    Eigen::VectorXd currentAction(nJoints_);
    currentAction = currentAction_PD.head(nJoints_);
    Eigen::VectorXd rawGain = currentAction_PD.tail(nJoints_) * pGainScaler;
    Eigen::VectorXd meanShift(nJoints_);
    meanShift.setConstant(pGainShift);
    rawGain = rawGain + meanShift;
    // for(int i = 0; i < rawGain.size();++i)
    //   if(rawGain[i] < 0.0) rawGain[i] = 0.0;
    jointPgain.tail(nJoints_) = rawGain;
    jointDgain.tail(nJoints_) = jointPgain.tail(nJoints_).cwiseSqrt();
    jointDgain = jointDgain * 1.4;

    alien_->setPdGains(jointPgain, jointDgain);
    pTarget12_ = currentAction;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += 0.3 * actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;
    pTarget_[LEFT_HIP] += desired_angleH;
    pTarget_[LEFT_THIGH] += desired_angleT;
    pTarget_[LEFT_CALF] += desired_angleC;
    alien_->setPdTarget(pTarget_, vTarget_);

    auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
    auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);    
    int p = current_count%force_count;

    for(int i=0; i<loopCount; i++) {
      world_->integrate();  
      alien_->getState(gc_, gv_);
      CenterOfPressure();
      GetFromDevice(true);
      if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
      {

        Eigen::VectorXd ref_pose = gc_;
        ref_pose[LEFT_HIP] = desired_angleH;
        ref_pose[LEFT_THIGH] = desired_angleT;
        ref_pose[LEFT_CALF] = desired_angleC;
        ref_dummy->setGeneralizedCoordinate(ref_pose);
        auto vis = raisim::OgreVis::get();
        auto& list = raisim::OgreVis::get()->getVisualObjectList();
        raisim::Vec<3> tempPosition;
        ref_dummy->getFramePosition(ref_dummy->getFrameByName("FL_foot_fixed"), tempPosition);
        list["sphere_indicator"].setPosition(tempPosition);
        list["sphere_indicator_pr"].setPosition(pp);
        for(int i =0 ; i < points.size();++i){
          list["sphere" + std::to_string(i)].setPosition(points[i]);
        }
        for(int i =0 ; i < connection.size()-1;++i){
          Eigen::Vector3d half = (points[connection[i+1]] + points[connection[i]]) / 2 ;
          float len = (points[connection[i+1]] - points[connection[i]]).norm();
          raisim::Mat<3,3> rot;
          Eigen::Vector3d way = (points[connection[i+1]] - points[connection[i]]).normalized();
          raisim::Vec<3> direction = {way[0], way[1], way[2]};
          raisim::zaxisToRotMat(direction, rot);

          list["bone" + std::to_string(i)].setPosition(half);
          list["bone" + std::to_string(i)].setScale(Eigen::Vector3d(0.015, 0.015, len));
          list["bone" + std::to_string(i)].setOrientation(rot);
        }
        // Eigen::Vector3d com = alien_->getCompositeCOM().e();
        // com[2] = 0;
        // list["com"].setPosition(com);
        list["com"].setPosition(mCOP);
        for(int i = 0; i < end_names.size(); ++i)
        {
          raisim::Vec<3> tempPosition2;
          alien_->getFramePosition(alien_->getFrameByName(end_names[i]), tempPosition2);
          feetPoses[i] = tempPosition2.e();
        }
        for(int i = 0; i < 3; ++i){
          int j;
          if(i==2) j = 0;
          else j = i+1;
          Eigen::Vector3d half = (feetPoses[j] + feetPoses[i]) / 2 ;
          float len = (feetPoses[j] - feetPoses[i]).norm();
          raisim::Mat<3,3> rot;
          Eigen::Vector3d way = (feetPoses[j] - feetPoses[i]).normalized();
          raisim::Vec<3> direction = {way[0], way[1], way[2]};
          raisim::zaxisToRotMat(direction, rot);

          list["supportPoly"+ std::to_string(i)].setPosition(half);
          list["supportPoly"+ std::to_string(i)].setScale(Eigen::Vector3d(0.01, 0.01, len));
          list["supportPoly"+ std::to_string(i)].setOrientation(rot);
        }
        vis->renderOneFrame();
        if(startFlag){
        vis->select(anymalVisual_->at(0), false);
        vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.1), 3, true);
        }
      }      
      visualizationCounter_++;  
    }
    bgra2Mat();

    vod.write(image);
    imshow("Human_Motion", image);
    cv::waitKey(1);
    // std::chrono::system_clock::time_point start1 = std::chrono::system_clock::now(); 
    calcTarget();
    // std::chrono::duration<double> sec1 = std::chrono::system_clock::now() - start1;
    // std::cout << "target " << sec1.count() << " seconds" << std::endl;
    
    updateObservation();

    float totalReward = CalcReward();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("AngleReward", angle_reward);
      gui::rewardLogger.log("EndReward",  end_reward);
      gui::rewardLogger.log("SupportReward",  support_reward);
      gui::rewardLogger.log("RotReward",  root_reward);
      gui::rewardLogger.log("TorqueReward",  torque_reward);
      gui::rewardLogger.log("FreqReward",  freq_reward);

      /// reset camera
      // auto vis = raisim::OgreVis::get();

      // vis->select(anymalVisual_->at(0), false);
      // vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.1), 3, true);
    }
    // if(current_count%change_count == 0){
    //   update_random_leaning_angle();
    //   desired_angleHC = desired_angleH;
    //   desired_angleTC = desired_angleT;
    //   desired_angleCC = desired_angleC;
    //   // current_count = 0;
    // }
    // interpolate_desired();
    // current_count++;
    if(!mStopFlag) 
      return totalReward;
    else{
      CloseDevice();
      return -1E10;
    }
  }
  void curriculumUpdate(int update) final {
    int curr1 = 500;
    int curr2 = 1500;
    int curr3 = 3000;
    int curr4 = 4500;
    if(update > curr1 && update < curr2)
    {
      init_pose_flag = false;
      curriculum_distribution(1, hip_distribution, thigh_distribution, calf_distribution);
      curriculum_force_distribution(1, force_distribution);
    }
    else if(update > curr2 && update < curr3)
    {
      init_pose_flag = false;
      curriculum_distribution(2, hip_distribution, thigh_distribution, calf_distribution);
      curriculum_force_distribution(2, force_distribution);
    }
    else if(update > curr3 && update < curr4)
    {
      init_pose_flag = false;
      curriculum_distribution(3, hip_distribution, thigh_distribution, calf_distribution);
      curriculum_force_distribution(3, force_distribution);
    }
    else if(update > curr4)
    {
      init_pose_flag = false;
      curriculum_distribution(4, hip_distribution, thigh_distribution, calf_distribution);
      curriculum_force_distribution(4, force_distribution);
    }
  }

  void interpolate_desired()
  {
    int interpol = current_count%change_count;
    float percentile = (float) interpol / (float) change_count;
    if(init_pose_flag) percentile = 1.0;
    desired_angleH = desired_angleHC * (1.0 - percentile) + desired_angleHF * percentile;
    desired_angleT = desired_angleTC * (1.0 - percentile) + desired_angleTF * percentile;
    desired_angleC = desired_angleCC * (1.0 - percentile) + desired_angleCF * percentile;
  }

  bool isInside()
  {
    float crosser = 1.0;
    for(int i = 0; i < end_names.size(); ++i)
    {
      int j;
      if(i==2) j = 0;
      else j = i+1;
      Eigen::Vector3d dir = feetPoses[j] - feetPoses[i];
      crosser *= (com[0]*dir[1] - com[1]*dir[0]);
    }
    if(crosser > 0) return true;
    else  return false;
  }

  bool isInside(std::vector<Eigen::Vector3d> safe)
  {
    float crosser = 1.0;
    for(int i = 0; i < end_names.size(); ++i)
    {
      int j;
      if(i==2) j = 0;
      else j = i+1;
      Eigen::Vector3d dir = safe[j] - safe[i];
      crosser *= (com[0]*dir[1] - com[1]*dir[0]);
    }
    if(crosser > 0) return true;
    else  return false;
  }

  std::vector<Eigen::Vector3d> SafeSP(float scale)
  {
    std::vector<Eigen::Vector3d> safers;
    Eigen::Vector3d centre;
    centre.setZero();
    for(int i = 0 ; i < 3; ++i)
      centre += feetPoses[i];
    centre /= 3;
    for(int i = 0; i < 3; ++i){
      Eigen::Vector3d current;
      current = scale * feetPoses[i] + (1 - scale) * centre;
      safers.push_back(current);
    }
    return safers;
  }

  float CalcReward()
  {   
    float reward = 0;
    double angle_err = 0;
    double end_err = 0;
    double root_err = 0;
    double support_err = 0;
    double torque_err = 0;
    double freq_err = 0;

    alien_->getState(gc_, gv_);
    if(isnan(gc_[0]))
    {
      return 0;
    }

    angle_err = CalcAngleErr();
    end_err = CalcEndErr();
    root_err = CalcRootErr();
    support_err = CalcSupportErr(true);
    torque_err = CalcTorqueErr();
    freq_err = CalcFreqErr();

    angle_reward = exp(-angle_scale * angle_err);
    end_reward = exp(-end_scale * end_err);
    root_reward = exp(-root_scale * root_err);
    support_reward = exp(-support_scale * support_err);
    torque_reward = exp(-torque_scale * torque_err);
    freq_reward = exp(-freq_scale * freq_err);

    reward = main_w * angle_reward * end_reward * support_reward * root_reward + 
            stable_w * torque_reward + freq_w * freq_reward;

    return reward;
  }

  float CalcAngleErr()
  {
    float diff = 0.0;
    diff += pow((desired_angleH - gc_[LEFT_HIP]),2);
    diff += pow((desired_angleT - gc_[LEFT_THIGH]),2);
    diff += pow((desired_angleC - gc_[LEFT_CALF]),2);
    return diff;
  }

  float CalcEndErr()
  {
    float diff = 0.0;
    auto contacts = alien_->getContacts();

    for(int i = 0; i < end_names.size(); ++i)
    {
      Eigen::Vector3d projected_diff;
      projected_diff.setZero();
      raisim::Vec<3> hip_c;
      alien_->getFramePosition(alien_->getFrameByName(end_names[i].substr(0,2) + "_thigh_joint"), hip_c);
      projected_diff[0] = hip_c[0] - feetPoses[i][0];
      projected_diff[1] = hip_c[1] - feetPoses[i][1];
      diff += projected_diff.squaredNorm() * 0.01;
    }

    for(int i =0 ; i < desired_footIndices_.size() ; ++i)
    {
      int j;
      for(j =0 ; j < contacts.size(); ++j)
        if(desired_footIndices_[i] == contacts[j].getlocalBodyIndex())
          break;
      if(j == contacts.size()){
        diff += feetPoses[i][2] * feetPoses[i][2];
      }

    }
    return diff;
  }

  float CalcRootRotErr()
  {
    Eigen::Vector3d xvec(1.0, 0.0, 0.0);
    float rot_err = 0;
    raisim::Vec<4> quat;
    raisim::Mat<3, 3> R;
    for(int j = 0; j<4; ++j)
      quat[j] = gc_[j+3];
    quatToRotMat(quat, R);
    Eigen::Matrix3d R_curr = R.e();
    Eigen::Vector3d current = R_curr * xvec;
    current[2] = 0.0;

    float angle;
    if(current.norm() < 1e-10)
      angle = 0;
    else
      angle = acos(current.dot(xvec)/current.norm());
    rot_err = angle * angle;
    return rot_err;
  }

  float CalcRootErr()
  {
    Eigen::Vector3d xvec(1.0, 0.0, 0.0);
    float rot_err = 0;
    raisim::Vec<4> quat;
    raisim::Mat<3, 3> R;
    for(int j = 0; j<4; ++j)
      quat[j] = gc_[j+3];
    quatToRotMat(quat, R);
    Eigen::Matrix3d R_curr = R.e();
    Eigen::Vector3d current = R_curr * xvec;
    current[2] = 0.0;

    float angle;
    if(current.norm() < 1e-10)
      angle = 0;
    else
      angle = acos(current.dot(xvec)/current.norm());
    rot_err = angle * angle;

    float bound_err = 0;
    float thold = 0.2;
    float dist = gc_[0] * gc_[0] + gc_[1] * gc_[1];
    if(dist < thold) bound_err = 0.0;
    else bound_err = dist - thold;

    return rot_err + bound_err;
  }


  float CalcSupportErr(bool safe)
  {
    Eigen::Vector3d com = alien_->getCompositeCOM().e();
    com[2];
    float min_d = 100000;
    float safe_scale = 0.3;

    if(safe){
      auto safeZone = SafeSP(safe_scale);
      if(isInside(safeZone)) return 0;
      for(int i = 0; i < end_names.size(); ++i)
      {
        int j;
        if(i==2) j = 0;
        else j = i+1;
        float current = distToSegment(com, safeZone[i], safeZone[j]);
        if(min_d > current) min_d = current;
      }
    }

    else{    
      if(isInside()) return 0;
      for(int i = 0; i < end_names.size(); ++i)
      {
        int j;
        if(i==2) j = 0;
        else j = i+1;
        float current = distToSegment(com, feetPoses[i], feetPoses[j]);
        if(min_d > current) min_d = current;
      }
    }
    return min_d * min_d;
  }
  float CalcTorqueErr()
  {
    float total_torque = alien_->getGeneralizedForce().squaredNorm();
    return total_torque;
  }


  void CenterOfPressure(){
    Eigen::Vector3d COP;
    COP.setZero();
    float tot_mag = 0.0;
    for (auto &contact: alien_->getContacts()) {
      for(int i = 0 ; i < desired_footIndices_.size(); ++i){
        if(desired_footIndices_[i] == contact.getlocalBodyIndex())
        {
          auto current = contact.getPosition();
          auto zaxis = *contact.getImpulse();
          float mag = abs(zaxis[2]);
          tot_mag += mag;
          COP[0] += current[0] * mag;
          COP[1] += current[1] * mag;
        }
      }
    }
    if(tot_mag == 0.0){
      Eigen::Vector3d com = alien_->getCompositeCOM().e();
      com[2] = 0;
      mCOP = com;
      return;
    }
    COP /= tot_mag;
    mCOP = COP;
  }

  float CalcFreqErr()
  {
    // Use FFT method 
    // err = magnitude * frequncy 
    // Low frequency is recommended
    float err = 0.0;
    std::vector<double> mag;
    Eigen::VectorXd torque = alien_->getGeneralizedForce().e().tail(nJoints_);
    for(int i = 0; i < nJoints_; ++i ){
      torque_sequence[i].push_back(torque[i]);
      torque_sequence[i].pop_front();
      double *x = new double[32]; 
      double *y = new double[32];
      for(int j = 0; j < 32; ++j)
      {
        x[j] = torque_sequence[i][j];
        y[j] = 0.0;
      }
      mag = FFT_Mag(1, 5, x, y);
      for(int j = 0; j < mag.size() ; ++j)
      {
        // Actaul frequency = j * Fs / N but we can use j instead since Fs/N is fixed
        // for all joints (Sampled equally)
        err += mag[j] * j;
      }
      mag.shrink_to_fit();
      delete [] x;
      delete [] y;
    }   
    return err;
  }

  void solverIKTarget(Eigen::Vector3d task){
    alien_->getState(gc_, gv_);
    Eigen::VectorXd ref_pose = gc_;
    ref_dummy->setGeneralizedCoordinate(ref_pose);
    mIK->setReady();
    mIK->setTarget(task);
    mIK->solveIK();
    auto sol = mIK->getSolution();    
    desired_angleH = sol[0];
    desired_angleT = sol[1];
    desired_angleC = sol[2];
    // std::cout << sol.transpose() << std::endl;
  }

  void updateExtraInfo() final {
    extraInfo_["base height"] = gc_[2];
  }

  void updateObservation() {
    alien_->getState(gc_, gv_);
    obScaled_.setZero(obDim_);

    /// update observations
    obScaled_[0] = desired_angleH;
    obScaled_[1] = desired_angleT;
    obScaled_[2] = desired_angleC;
    obScaled_.segment(3, 12) = gc_.tail(12);
    Eigen::Quaterniond quat(gc_[3], gc_[4], gc_[5], gc_[6]);
    Eigen::AngleAxisd aa(quat);
    Eigen::Vector3d rotation = aa.angle()*aa.axis();
    obScaled_.segment(15, 3) = rotation;
    obScaled_.segment(18, 3) = gv_.segment(0,3);
    obScaled_.segment(21, 12) = gc_prev1.tail(12);
    auto frameNames = RootOutFrames();
    for(int i = 0; i < frameNames.size(); ++i){
      raisim::Vec<3> tempPose;
      alien_->getFramePosition(alien_->getFrameByName(frameNames[i]), tempPose);
      obScaled_.segment(33 + 3*i, 3) = tempPose.e();
    }
    obScaled_[84] = gc_[2];
    obScaled_.segment(85, nJoints_) = jointPgain.tail(nJoints_);
    gc_prev1 = gc_;
  }


  void SetFromDevice(InputSettings inputSettings) {
    VERIFY(k4a_device_open(0, &device), "Open K4A Device failed!");

    // Start camera. Make sure depth camera is enabled.
    k4a_device_configuration_t deviceConfig = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    deviceConfig.depth_mode = inputSettings.DepthCameraMode;
    deviceConfig.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    deviceConfig.color_resolution = K4A_COLOR_RESOLUTION_720P;
    VERIFY(k4a_device_start_cameras(device, &deviceConfig), "Start K4A cameras failed!");

    // Get calibration information
    k4a_calibration_t sensorCalibration;
    VERIFY(k4a_device_get_calibration(device, deviceConfig.depth_mode, deviceConfig.color_resolution, &sensorCalibration),
        "Get depth camera calibration failed!");
    depthWidth = sensorCalibration.depth_camera_calibration.resolution_width;
    depthHeight = sensorCalibration.depth_camera_calibration.resolution_height;
    colorWidth = sensorCalibration.color_camera_calibration.resolution_width;
    colorHeight = sensorCalibration.color_camera_calibration.resolution_height;

    // Create Body Tracker
    k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
    tracker_config.processing_mode = inputSettings.CpuOnlyMode ? K4ABT_TRACKER_PROCESSING_MODE_CPU : K4ABT_TRACKER_PROCESSING_MODE_GPU;
    VERIFY(k4abt_tracker_create(&sensorCalibration, tracker_config, &tracker), "Body tracker initialization failed!");
    // window3d.Create("3D Visualization", sensorCalibration);
  }


  void CloseDevice(){
    std::cout << "Finished body tracking processing!" << std::endl;

    k4abt_tracker_shutdown(tracker);
    k4abt_tracker_destroy(tracker);

    k4a_device_stop_cameras(device);
    k4a_device_close(device);
  }


  void GetFromDevice(bool vis_flag) {
    k4a_capture_t sensorCapture = nullptr;
    k4a_wait_result_t getCaptureResult = k4a_device_get_capture(device, &sensorCapture, 0); // timeout_in_ms is set to 0

    if (getCaptureResult == K4A_WAIT_RESULT_SUCCEEDED)
    {
        // timeout_in_ms is set to 0. Return immediately no matter whether the sensorCapture is successfully added
        // to the queue or not.
        k4a_wait_result_t queueCaptureResult = k4abt_tracker_enqueue_capture(tracker, sensorCapture, 0);
        if(vis_flag){
          k4a_image_t colorImage = k4a_capture_get_color_image(sensorCapture);
          img_bgra32 = k4a_image_get_buffer(colorImage);
        }

        // Release the sensor capture once it is no longer needed.
        k4a_capture_release(sensorCapture);

        if (queueCaptureResult == K4A_WAIT_RESULT_FAILED)
        {
            std::cout << "Error! Add capture to tracker process queue failed!" << std::endl;
            CloseDevice();
        }
    }
    else if (getCaptureResult != K4A_WAIT_RESULT_TIMEOUT)
    {
        std::cout << "Get depth capture returned error: " << getCaptureResult << std::endl;
        CloseDevice();
    }

    // Pop Result from Body Tracker
    k4abt_frame_t bodyFrame = nullptr;
    k4a_wait_result_t popFrameResult = k4abt_tracker_pop_result(tracker, &bodyFrame, 0); // timeout_in_ms is set to 0
    
    if (popFrameResult == K4A_WAIT_RESULT_SUCCEEDED)
    {
        /************* Successfully get a body tracking result, process the result here ***************/
        // VisualizeResult(bodyFrame, window3d, depthWidth, depthHeight);
        // uint32_t numBodies = k4abt_frame_get_num_bodies(bodyFrame);
        uint32_t numBodies = 1;
        for (uint32_t i = 0; i < numBodies; i++)
        {
          k4abt_body_t body;
          VERIFY(k4abt_frame_get_body_skeleton(bodyFrame, i, &body.skeleton), "Get skeleton from body frame failed!");
          body.id = k4abt_frame_get_body_id(bodyFrame, 0);

          k4a_float3_t leftWristJoint = body.skeleton.joints[K4ABT_JOINT_WRIST_LEFT].position;
          k4a_float3_t leftElbowJoint = body.skeleton.joints[K4ABT_JOINT_ELBOW_LEFT].position;
          k4a_float3_t leftShoulderJoint = body.skeleton.joints[K4ABT_JOINT_SHOULDER_LEFT].position;
          k4a_float3_t leftClavicleJoint = body.skeleton.joints[K4ABT_JOINT_CLAVICLE_LEFT].position;
          k4a_float3_t rightClavicleJoint = body.skeleton.joints[K4ABT_JOINT_CLAVICLE_RIGHT].position;
          k4a_float3_t NeckJoint = body.skeleton.joints[K4ABT_JOINT_NECK].position;
          NeckOrientation = body.skeleton.joints[K4ABT_JOINT_NECK].orientation;
          
          points[0] = scaler * Eigen::Vector3d((leftShoulderJoint.xyz.x - NeckJoint.xyz.x), (leftShoulderJoint.xyz.z - NeckJoint.xyz.z), (leftShoulderJoint.xyz.y - NeckJoint.xyz.y));
          points[1] = scaler * Eigen::Vector3d((leftElbowJoint.xyz.x - NeckJoint.xyz.x), (leftElbowJoint.xyz.z - NeckJoint.xyz.z), (leftElbowJoint.xyz.y - NeckJoint.xyz.y));
          points[2] = scaler * Eigen::Vector3d((leftWristJoint.xyz.x - NeckJoint.xyz.x), (leftWristJoint.xyz.z - NeckJoint.xyz.z), (leftWristJoint.xyz.y - NeckJoint.xyz.y));
          points[3] = scaler * Eigen::Vector3d((rightClavicleJoint.xyz.x - NeckJoint.xyz.x), (rightClavicleJoint.xyz.z - NeckJoint.xyz.z), (rightClavicleJoint.xyz.y - NeckJoint.xyz.y));
          points[4] = scaler * Eigen::Vector3d((NeckJoint.xyz.x - NeckJoint.xyz.x), (NeckJoint.xyz.z - NeckJoint.xyz.z), (NeckJoint.xyz.y - NeckJoint.xyz.y));
          points[5] = scaler * Eigen::Vector3d((leftClavicleJoint.xyz.x - NeckJoint.xyz.x), (leftClavicleJoint.xyz.z - NeckJoint.xyz.z), (leftClavicleJoint.xyz.y - NeckJoint.xyz.y));
          for(int i =0 ; i < points.size();++i){
            points[i][2] +=z_offset;
          }
          deviceFlag = true;
        }
        //Release the bodyFrame    
        k4abt_frame_release(bodyFrame);
    }
  }

  void bgra2Mat()
  {
    int stride = colorWidth * 4;
    for(int y = 0; y < colorHeight; ++y)
    {
      for(int x = 0; x < colorWidth; ++x)
      {
        image.at<cv::Vec3b>(y, x)[0] = img_bgra32[4 * x + y * stride];
        image.at<cv::Vec3b>(y, x)[1] = img_bgra32[4 * x + y * stride+ 1];
        image.at<cv::Vec3b>(y, x)[2] = img_bgra32[4 * x + y * stride+ 2];
      }
    }
  }

  void calcTarget(){
    if(!deviceFlag) return;
    
    if(startFlag)
    {
      alien_->getState(gc_, gv_);
      raisim::Vec<3> tempPositionh;
      ref_dummy->getFramePosition(ref_dummy->getFrameByName("FL_hip_joint"), tempPositionh);
      raisim::Vec<3> tempPositionc;
      ref_dummy->getFramePosition(ref_dummy->getFrameByName("FL_calf_joint"), tempPositionc);
      raisim::Vec<3> tempPositionf;
      ref_dummy->getFramePosition(ref_dummy->getFrameByName("FL_foot_fixed"), tempPositionf);
      float robot_arm = (tempPositionf.e() - tempPositionc.e()).norm() + (tempPositionc.e() - tempPositionh.e()).norm();
      float human_arm = (points[0] - points[1]).norm() + (points[1] - points[2]).norm();

      // float robot_arm = (tempPositionf.e() - tempPositionh.e()).norm();
      // float human_arm = (points[0] - points[2]).norm();
      hum2rob = robot_arm / human_arm;
    }

    Eigen::Matrix3d Rcurrent;
    // Rcurrent = Eigen::Quaterniond(NeckOrientation.v[0], NeckOrientation.v[1], NeckOrientation.v[2], NeckOrientation.v[3]);
    Eigen::Vector3d mid_clav = (points[3] + points[5]) / 2;
    Eigen::Vector3d x_vec = points[4] - mid_clav;
    x_vec.normalize();
    auto eq = EquationPlane(points[3], points[4], points[5]);
    Eigen::Vector3d z_vec = eq.segment(0,3);
    z_vec.normalize();
    Eigen::Vector3d y_vec = z_vec.cross(x_vec);
    y_vec.normalize();
    Rcurrent.col(0) = x_vec;
    Rcurrent.col(1) = y_vec;
    Rcurrent.col(2) = z_vec;
    Eigen::Vector3d prelhuman = Rcurrent.inverse() * (points[2] - points[0]);
    Eigen::Vector3d prelrobotraw = prelhuman * hum2rob;

    if(startFlag)
    {
      raisim::Vec<3> tempPositionh;
      alien_->getFramePosition(alien_->getFrameByName("FL_hip_joint"), tempPositionh);
      raisim::Vec<3> tempPositionf;
      alien_->getFramePosition(alien_->getFrameByName("FL_foot_fixed"), tempPositionf);
      z_standard = tempPositionh[2];
      raisim::Mat<3,3> orientation_r;
      ref_dummy->getBaseOrientation(orientation_r);
      Eigen::Matrix3d rot = orientation_r.e();
      float theta_z = atan2(rot(2,1), rot(1,1));
      Eigen::Vector3d prelreal = Eigen::AngleAxisd(theta_z, Eigen::Vector3d(0,0,1)).inverse() * (tempPositionf.e() - tempPositionh.e());
      Eigen::Vector3d axis = prelrobotraw.cross(prelreal);
      float angle = angle_finder(prelrobotraw, Eigen::Vector3d(0,0,0), prelreal);
      Rcomp = Eigen::AngleAxisd(angle,axis.normalized());
      startFlag = false;
      return;
    }
    raisim::Vec<3> tempPositionh;
    alien_->getFramePosition(ref_dummy->getFrameByName("FL_hip_joint"), tempPositionh);
    // tempPositionh[2] += 0.05;
    raisim::Mat<3,3> orientation_r;
    ref_dummy->getBaseOrientation(orientation_r);
    Eigen::Matrix3d rot = orientation_r.e();
    float theta_z = atan2(rot(2,1), rot(1,1));    
    Eigen::Matrix3d body_rot;
    body_rot= Eigen::AngleAxisd(theta_z, Eigen::Vector3d(0,0,1));
    Eigen::Vector3d prelrobot = Rcomp * prelrobotraw;
    tempPositionh[2] = z_standard;
    Eigen::Vector3d task = body_rot * prelrobot + tempPositionh.e();
    // solverIKTarget(prelrobot);
    solverIKTarget(task);
    pp = task;
  }

  Eigen::Vector3d speed_limit_getter()
  {
    Eigen::MatrixXd fullJacobian(3,gvDim_);
    Eigen::Matrix3d leftJacobian;
    leftJacobian.setZero();
    fullJacobian.setZero();
    auto& frame = alien_->getFrameByName("FL_foot_fixed");
    raisim::Vec<3> position_W;
    alien_->getFramePosition(alien_->getFrameByName("FL_foot_fixed"), position_W);
    alien_->getDenseJacobian(frame.parentId, position_W, fullJacobian);
    leftJacobian = fullJacobian.block(0,9,3,3);
    Eigen::Vector3d vel = leftJacobian * Eigen::Vector3d(40, 40, 40);
    return vel.cwiseAbs();
  }



  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obScaled_.cast<float>();
  }

  void update_random_leaning_angle()
  {
    unsigned seed_temp = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed_temp);
    desired_angleHF = hip_distribution(generator);
    desired_angleTF = thigh_distribution(generator);
    desired_angleCF = calf_distribution(generator);
  }

  void exert_random_force()
  {
    unsigned seed_temp = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed_temp);
    Eigen::Vector3d axis(0,0,0);
    axis[0] = xaxis_distribution(generator);
    axis[1] = yaxis_distribution(generator);
    axis[2] = zaxis_distribution(generator);
    axis.normalize();
    float mag = force_distribution(generator);
    for(int i =0 ; i < 3; ++i)
      force[i] = mag * axis[i];
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = 0.f;

    /// if the contact body is not feet
    for(auto& contact: alien_->getContacts())
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end()) {
        return true;
      }

    terminalReward = 0.f;
    return false;
  }

  void setSeed(int seed) final {
    std::srand(seed);
  }

  void close() final {
  }

  void get_current_torque(Eigen::Ref<EigenVec>& tau) {
    Eigen::VectorXd torque = alien_->getGeneralizedForce().e().tail(nJoints_);
    for(int i = 0; i < nJoints_; ++i)
      tau[i] = torque[i];
  }

  Eigen::MatrixXd get_jacobian(const std::string joint_name){
    return Eigen::MatrixXd::Constant(1,1, 0.0); 
  }
  int getNumFrames(){return 0;}


 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* alien_;
  raisim::ArticulatedSystem* ref_dummy;
  std::vector<GraphicObject> * anymalVisual_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, torque_, gc_dummy_init;
  Eigen::VectorXd gc_prev1, gc_prev2;
  double desired_fps_ = 60.;
  int visualizationCounter_=0;
  float effort_limit = 44.4;
  Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_;
  Eigen::VectorXd obDouble_, obScaled_;
  std::set<size_t> footIndices_;
  std::vector<int> desired_footIndices_;
  std::vector<std::deque<double>> torque_sequence;

  double desired_angleH, desired_angleT, desired_angleC;
  double desired_angleHC, desired_angleTC, desired_angleCC;
  double desired_angleHF, desired_angleTF, desired_angleCF;
  bool startFlag, deviceFlag;
  double base_angleH, base_angleT, base_angleC;

  int change_count, current_count, force_count;
  int curriculum_cnt;

  // Reward Shapers
  double main_w, stable_w, freq_w;
  double angle_scale, end_scale, root_scale, support_scale, torque_scale, freq_scale;
  double angle_reward, end_reward, root_reward, support_reward, torque_reward, freq_reward;

  // Kp Settings
  double init_pgain = 30;
  Eigen::VectorXd jointPgain, jointDgain;
  float pGainScaler = 100.0;
  float pGainShift = 100.0;
  float pGainOffSet = 30.0;

  // Kinect Settings
  float z_offset = 0.85;
  k4a_device_t device = nullptr;
  k4abt_tracker_t tracker = nullptr;
  std::vector<Eigen::Vector3d> points;
  std::vector<int> connection;
  k4a_float3_t startOrigin;
  float scaler;
  k4a_quaternion_t NeckOrientation;
  float z_standard;
  int depthWidth, depthHeight, colorWidth, colorHeight;
  cv::Mat image;
  uint8_t* img_bgra32;
  cv::VideoWriter vod;

  AnalyticLIK* mIK;

  std::vector<CoordinateFrame> body_frames;
  std::vector<std::string> end_names;
  std::vector<Eigen::Vector3d> feet;
  std::vector<Eigen::Vector3d> feetPoses;
  Eigen::Vector3d mCOP;
  Eigen::Matrix3d Rinit, Rcomp;
  float hum2rob;
  Eigen::Vector3d prev_foot;
  Eigen::Vector3d pp;
  raisim::Vec<3> force; 
  std::uniform_real_distribution<double> hip_distribution;
  std::uniform_real_distribution<double> thigh_distribution;
  std::uniform_real_distribution<double> calf_distribution;
  std::uniform_real_distribution<double> force_distribution;
  std::uniform_real_distribution<double> xaxis_distribution;
  std::uniform_real_distribution<double> yaxis_distribution;
  std::uniform_real_distribution<double> zaxis_distribution;
  bool init_pose_flag;
  //IK solvers
};

}