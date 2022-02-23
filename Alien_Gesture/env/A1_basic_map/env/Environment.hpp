
#include <stdlib.h>
#include <cstdint>
#include <set>
#include <chrono>
#include <random>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <raisim/OgreVis.hpp>
#include "RaisimGymEnv.hpp"
#include "../include/visSetupCallback2.hpp"
#include "../include/Utilities.h" 
#include "../include/RobotInfo.hpp"
#include "../include/EnvMath.hpp"
#include "../include/MotionFunction.hpp"

#include "../visualizer_alien/raisimKeyboardCallback.hpp"
#include "../visualizer_alien/helper.hpp"
#include "../visualizer_alien/guiState.hpp"
#include "../visualizer_alien/raisimBasicImguiPanel.hpp"
#include "../visualizer_alien/BodyTrakingHelper.hpp"


namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:
  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) 
  {
    /// add objects
    alien_ = world_->addArticulatedSystem(resourceDir_+"/a1/a1/a1.urdf");
    vis_ref = new raisim::ArticulatedSystem(resourceDir_+"/a1/a1/a1.urdf");       

    alien_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    auto ground = world_->addGround(0, "floor");
    world_->setERP(0,0);
    /// get robot data 
    gcDim_ = alien_->getGeneralizedCoordinateDim();
    gvDim_ = alien_->getDOF();
    nJoints_ = 12;

    control_dt_ = 0.03333333;
    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    torque_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);
    gc_target.setZero(4 + nJoints_); gc_prev.setZero(gcDim_);

    /// this is nominal configuration of anymal
    gc_init_ <<  0, 0, 0.27, 1.0, 0.0, 0.0, 0.0, 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99);
    alien_->setState(gc_init_, gv_init_);
    tracking_ball->setPosition(0, 0.5, 0.3);

    /// set pd gains
    jointPgain.setZero(gvDim_); jointPgain.tail(nJoints_).setConstant(init_pgain);
    jointDgain.setZero(gvDim_); jointDgain.tail(nJoints_).setConstant(init_dgain);
    alien_->setPdGains(jointPgain, jointDgain);
    alien_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    // Set actuation limits
    Eigen::VectorXd torque_upperlimit = alien_->getUpperLimitForce().e();
    Eigen::VectorXd torque_lowerlimit = alien_->getLowerLimitForce().e();
    torque_upperlimit.tail(nJoints_).setConstant(effort_limit);
    torque_lowerlimit.tail(nJoints_).setConstant(-effort_limit);
    alien_->setActuationLimits(torque_upperlimit, torque_lowerlimit);

    obDim_ = 51;
    actionDim_ = nJoints_;

    actionStd_.setZero(nJoints_);  actionMean_.setZero(nJoints_);
    obMean_.setZero(obDim_); obStd_.setZero(obDim_);
    prev_vel.setZero(nJoints_); joint_acc.setZero(nJoints_);

    float ah = deg2rad(30), at = deg2rad(30), ac = deg2rad(30);
    float mh = 0, mt = M_PI /2, mc = -1.80;
    actionStd_ << ah, at, ac, ah, at, ac, ah, at, ac, ah, at, ac;
    // actionMean_ << mh, mt, mc, mh, mt, mc, mh, mt, mc, mh, mt, mc;

    freq_w = 0.2;
    main_w = 1.0 - freq_w;

    joint_imit_scale = 10.0;
    ori_imit_scale = 1.0;
    end_imit_scale = 1.0;
    freq_scale = 1e-4;

    gui::rewardLogger.init({"Angle_reward", "Ori_reward", "End_reward", "Freq_reward"});

    /// indices of links that should not make contact with ground
    footIndices_.insert(alien_->getBodyIdx("FL_calf"));
    footIndices_.insert(alien_->getBodyIdx("FR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RL_calf"));

    /// visualize if it is the first environment
    if (visualizable_) {
      auto vis = raisim::OgreVis::get();
      vis->setWorld(world_.get());
      vis->setWindowSize(1800, 1200);
      vis->setImguiSetupCallback(imguiSetupCallback);
      vis->setImguiRenderCallback(imguiRenderCallBack);
      vis->setKeyboardCallback(raisimKeyboardCallback);
      vis->setSetUpCallback(setupCallback);
      vis->setAntiAliasing(2);
      vis->initApp();

      anymalVisual_ = vis->createGraphicalObject(alien_, "AlienGo");
      vis->createGraphicalObject(vis_ref, "visualization");
      vis->createGraphicalObject(ground, 10, "floor", "checkerboard_green");
      tracking_graphics = vis->createGraphicalObject(tracking_ball, "Tracker", "None");

      desired_fps_ = 1.0/ control_dt_;
      vis->setDesiredFPS(desired_fps_);
      vis->select(anymalVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }
    mrl = new MotionReshaper(ik_dummy);
    // clearKeys();
    // setKeyPoseIndices();
    // motionGenerator();
  }

  ~ENVIRONMENT() final = default;

  void init() final { }

  void kinect_init() final{}

  void reset() final
  {
    alien_->setState(gc_init_, gv_init_);

    gc_reference = gc_init_;
    gc_reference[1] += 0.7;

    prev_vel.setZero();
    joint_acc.setZero();
    gc_prev = gc_init_;
    gc_target = gc_init_.tail(nJoints_ + 4);

    joint_imit_reward = 0.0;
    ori_imit_reward = 0.0;
    end_imit_reward = 0.0;
    freq_reward = 0.0;

    pTarget_.setZero();
    pTarget_.tail(nJoints_) = gc_init_.tail(nJoints_);
    alien_->setPdTarget(pTarget_, vTarget_);

    updateObservation();

    if(visualizable_)
      gui::rewardLogger.clean();
  }

  void setVisRef()
  {
    gc_reference.tail(16) = gc_target.tail(16);
    vis_ref->setGeneralizedCoordinate(gc_reference);
    float min_foot = 1000;
    for(int i =0 ; i < 4; ++i)
    {
      raisim::Vec<3> tempPosition;
      vis_ref->getFramePosition(FR_FOOT + 3 * i, tempPosition);
      float height = (tempPosition[2]-0.015);
      if(min_foot > height) min_foot = height;
    }
    gc_reference[2] -= min_foot;
    vis_ref->setGeneralizedCoordinate(gc_reference);

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

    pTarget12_ = currentAction;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += gc_target.tail(12);
    pTarget_.tail(nJoints_) = pTarget12_;
    setVisRef();
    alien_->setPdTarget(pTarget_, vTarget_);
    auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
    auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);    

    for(int i=0; i<loopCount; i++) {
      world_->integrate();  

      if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
      {
        auto vis = raisim::OgreVis::get();
        vis->renderOneFrame();
      }      
      visualizationCounter_++;
    }

    updateObservation();

    float totalReward = CalcReward();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("Angle_reward", joint_imit_reward);
      gui::rewardLogger.log("Ori_reward", ori_imit_reward);
      gui::rewardLogger.log("End_reward", end_imit_reward);
      gui::rewardLogger.log("Freq_reward", freq_reward);
      auto vis = raisim::OgreVis::get();

      vis->select(anymalVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
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
    pTarget12_ = currentAction;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += gc_target.tail(12);
    pTarget_.tail(nJoints_) = pTarget12_;
    setVisRef();

    alien_->setPdTarget(pTarget_, vTarget_);
    auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
    auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);    

    for(int i=0; i<loopCount; i++) {
      world_->integrate();  
      alien_->getState(gc_, gv_);
      // GetFromDevice();
      if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
      {
        auto vis = raisim::OgreVis::get();
        auto& list = raisim::OgreVis::get()->getVisualObjectList();
        vis->renderOneFrame();
      }      
      visualizationCounter_++;
    }

    updateObservation();

    float totalReward = CalcReward();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("Angle_reward", joint_imit_reward);
      gui::rewardLogger.log("Ori_reward", ori_imit_reward);
      gui::rewardLogger.log("End_reward", end_imit_reward);
      gui::rewardLogger.log("Freq_reward", freq_reward);
      /// reset camera
      // auto vis = raisim::OgreVis::get();      
    }

    if(!mStopFlag) 
      return totalReward;
    else
      return -1E10;
  }

  void curriculumUpdate(int update) final {
    int curr1 = 800;
    int curr2 = 1500;
    int curr3 = 3000;
    int curr4 = 4000;
  } 

  float CalcReward()
  {   
    float reward = 0;
    double joint_imit_err = 0;
    double ori_imit_err = 0;
    double end_imit_err = 0;
    double freq_err = 0;

    alien_->getState(gc_, gv_);
    if(isnan(gc_[0]) || isnan(gv_[8]))
    {
      return 0;
    }

    joint_acc = (gv_.tail(nJoints_) - prev_vel) / control_dt_;
    prev_vel = gv_.tail(nJoints_);

    joint_imit_err = jointImitationErr();
    ori_imit_err = orientationImitationErr();
    end_imit_err = endEffectorImitationErr();
    freq_err = CalcAccErr();

    joint_imit_reward = exp(-joint_imit_scale * joint_imit_err);
    ori_imit_reward = exp(-ori_imit_scale * ori_imit_err);
    end_imit_reward = exp(-end_imit_scale * end_imit_err);
    freq_reward = exp(- freq_scale * freq_err);

    reward = main_w * joint_imit_reward * ori_imit_reward * end_imit_reward + freq_w * freq_reward;
    return reward;
  }

  float jointImitationErr()
  {
    float err = 0.;
    for(int i = 0; i < nJoints_; ++i)
    {
      err += pow((gc_target[4 + i] - gc_[7 + i]),2);
    }
    return err;
  }

  float orientationImitationErr()
  {
    float err = 0.;
    // Use slerp
    Eigen::Quaterniond target(gc_target[0],gc_target[1],gc_target[2],gc_target[3]);
    Eigen::Quaterniond current(gc_[3],gc_[4],gc_[5],gc_[6]);
    err = pow(current.angularDistance(target), 2);
    return err;
  }

  float endEffectorImitationErr()
  {
    // Change HERE!!!!! To make gc_init foot reward ==> Precalculation
    // Much Faster learning is expected + We don't need ref_dummy! 
    float err = 0.;
    Eigen::VectorXd gc_extended = gc_;
    gc_extended.tail(4 + nJoints_) = gc_target;
    ref_dummy->setGeneralizedCoordinate(gc_extended);
    // Use endeffector position
    for(int i = 0; i < 4; ++i)
    {
      raisim::Vec<3> footPositionA;
      raisim::Vec<3> footPositionR;
      alien_->getFramePosition(FR_FOOT+3 * i, footPositionA);
      ref_dummy->getFramePosition(FR_FOOT+3 * i, footPositionR);
      err += (footPositionR.e() - footPositionA.e()).norm();
    }
    return err;
  }

  float CalcAccErr()
  {
    return joint_acc.squaredNorm();
  }

  void updateExtraInfo() final {
    extraInfo_["base height"] = gc_[2];
  }

  void updateObservation() {
    alien_->getState(gc_, gv_);
    obScaled_.setZero(obDim_);

    /// update observations
    // Joint angle, root orientation, root height
    obScaled_.segment(0, 12) = gc_.tail(12);
    Eigen::Quaterniond quat1(gc_[3], gc_[4], gc_[5], gc_[6]);
    Eigen::AngleAxisd aa1(quat1);
    Eigen::Vector3d rotation1 = aa1.angle()*aa1.axis();
    obScaled_.segment(12, 3) = rotation1;
    obScaled_[15] = gc_[2];

    // Previous pose
    obScaled_.segment(16, 12) = gc_prev.tail(12);
    Eigen::Quaterniond quat2(gc_prev[3], gc_prev[4], gc_prev[5], gc_prev[6]);
    Eigen::AngleAxisd aa2(quat2);
    Eigen::Vector3d rotation2 = aa2.angle()*aa2.axis();
    obScaled_.segment(28, 3) = rotation2;
    obScaled_[31] = gc_prev[2];

    // Target orientation and target joint angles
    obScaled_.segment(32, 12) = gc_target.tail(12);
    Eigen::Quaterniond quat3(gc_target[0], gc_target[1], gc_target[2], gc_target[3]);
    Eigen::AngleAxisd aa3(quat3);
    Eigen::Vector3d rotation3 = aa3.angle()*aa3.axis();
    obScaled_.segment(44, 3) = rotation3;

    for(int i = 0 ; i < 4; ++ i){
      raisim::Vec<3> footHeight;
      alien_->getFramePosition(FR_FOOT + 3 * i, footHeight);
      obScaled_[47 + i] = footHeight[2] - 0.015;
    }
    gc_prev = gc_;
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obScaled_.cast<float>();
  }

  void get_current_torque(Eigen::Ref<EigenVec>& tau) {
    Eigen::VectorXd torque = alien_->getGeneralizedForce().e().tail(nJoints_);
    for(int i = 0; i < nJoints_; ++i)
      tau[i] = torque[i];
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

  Eigen::MatrixXd get_jacobian(const std::string joint_name){
    return Eigen::MatrixXd::Constant(1,1, 0.0); 
  }
  int getNumFrames(){return 0;}


 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* alien_;
  raisim::ArticulatedSystem* vis_ref;
  raisim::Sphere* tracking_ball;
  std::vector<GraphicObject> * anymalVisual_;
  std::vector<GraphicObject>* tracking_graphics;

  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, torque_, gc_dummy_init;
  Eigen::VectorXd gc_prev, gc_target, gc_reference;
  double desired_fps_ = 30.;
  int visualizationCounter_=0;
  float effort_limit = 33.5 * 0.90;

  Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_;
  Eigen::VectorXd obDouble_, obScaled_;
  std::set<size_t> footIndices_;
  std::set<size_t> foot_hipIndices_;
  Eigen::VectorXd prev_vel, joint_acc;
  std::vector<std::deque<double>> torque_sequence;

  // Reward Shapers
  double main_w, freq_w;
  double joint_imit_scale, ori_imit_scale, end_imit_scale, freq_scale;
  double joint_imit_reward, ori_imit_reward, end_imit_reward, freq_reward;

  // Kp Settings
  double init_pgain = 200;
  double init_dgain = 10;
  Eigen::VectorXd jointPgain, jointDgain;
};

} 