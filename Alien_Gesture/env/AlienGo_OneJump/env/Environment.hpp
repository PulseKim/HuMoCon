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
#include "../include/TrajectoryGenerator.hpp"
#include "../include/RootPredictor.hpp"

#include "../visualizer_alien/raisimKeyboardCallback.hpp"
#include "../visualizer_alien/helper.hpp"
#include "../visualizer_alien/guiState.hpp"
#include "../visualizer_alien/raisimLocomotionImguiPanel.hpp"
#include "../visualizer_alien/BodyTrakingHelper.hpp"

#include "Iir.h"

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:
  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) 
  {
    /// add objects
    alien_ = world_->addArticulatedSystem(resourceDir_+"/a1/a1/a1.urdf");
    ref_dummy = new raisim::ArticulatedSystem(resourceDir_+"/a1/a1/a1.urdf");    
    vis_ref = new raisim::ArticulatedSystem(resourceDir_+"/a1/a1/a1.urdf");    
    tracking_ball = world_->addSphere(0.0001, 0.0001, "None", raisim::COLLISION(100),raisim::COLLISION(100));
    // Set world
    alien_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    auto ground = world_->addGround(0, "floor");
    world_->setERP(0,0);
    /// get robot data
    gcDim_ = alien_->getGeneralizedCoordinateDim();
    gvDim_ = alien_->getDOF();
    nJoints_ = 12;

    control_dt_ = 0.03333333;
    // READ_YAML(double, control_dt_, cfg["control_dt"])

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gc_bal.setZero(gcDim_); gc_crouch.setZero(gcDim_); gc_jump.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    torque_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);
    gc_noise_target.setZero(4 + nJoints_); gc_target.setZero(4 + nJoints_); gc_prev.setZero(gcDim_);
    gc_reference.setZero(gcDim_);
    /// this is nominal configuration of anymal
    gc_init_ <<  0, 0, 0.27, 1.0, 0.0, 0.0, 0.0, 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99);
    alien_->setState(gc_init_, gv_init_);
    ref_dummy->setState(gc_init_, gv_init_);
    tracking_ball->setPosition(0, 0.5, 0.3);

    /// set pd gains
    jointPgain.setZero(gvDim_); jointPgain.tail(nJoints_).setConstant(init_pgain);
    jointDgain.setZero(gvDim_); jointDgain.tail(nJoints_).setConstant(init_dgain);
    alien_->setPdGains(jointPgain, jointDgain);
    alien_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    Eigen::VectorXd torque_upperlimit = alien_->getUpperLimitForce().e();
    Eigen::VectorXd torque_lowerlimit = alien_->getLowerLimitForce().e();
    torque_upperlimit.tail(nJoints_).setConstant(effort_limit);
    torque_lowerlimit.tail(nJoints_).setConstant(-effort_limit);
    alien_->setActuationLimits(torque_upperlimit, torque_lowerlimit);

    // Have to adjust the observation dimension
    // Have to adjust the action std and mean
    // Have to........ 
    nRefClips = 12;
    obDim_ = 51;
    actionDim_ = nJoints_;

    actionStd_.setZero(nJoints_);  actionMean_.setZero(nJoints_);
    obMean_.setZero(obDim_); obStd_.setZero(obDim_);
    prev_vel.setZero(nJoints_); joint_acc.setZero(nJoints_);

    // getKeyPoses("/home/sonic/Project/Alien_Gesture/env/rsc/new_motion.txt");

    float ah = deg2rad(30), at = deg2rad(30), ac = deg2rad(30);
    float mh = 0, mt = M_PI /2, mc = -1.80;
    actionStd_ << ah, at, ac, ah, at, ac, ah, at, ac, ah, at, ac;
    // actionMean_ << mh, mt, mc, mh, mt, mc, mh, mt, mc, mh, mt, mc;

    freq_w = 0.2;
    main_w = 1.0 - freq_w;

    root_imit_scale = 20.0;
    joint_imit_scale = 10.0;
    ori_imit_scale = 5.0;
    end_imit_scale = 0.5;
    freq_scale = 1e-4;

    gui::rewardLogger.init({"Root_reward", "Angle_reward", "Ori_reward", "End_reward", "Freq_reward"});
  
    /// indices of links that should not make contact with ground
    footIndices_.insert(alien_->getBodyIdx("FL_calf"));
    footIndices_.insert(alien_->getBodyIdx("FR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RL_calf"));

    raisim::Vec<3> thigh_location;
    alien_->getFramePosition(alien_->getFrameByName("FL_thigh_joint"), thigh_location);
    raisim::Vec<3> calf_location;
    alien_->getFramePosition(alien_->getFrameByName("FL_calf_joint"), calf_location);
    raisim::Vec<3> foot_location;
    alien_->getFramePosition(alien_->getFrameByName("FL_foot_fixed"), foot_location);
    float ltc = (thigh_location.e() - calf_location.e()).norm();
    float lcf = (calf_location.e() - foot_location.e()).norm();
    float dt = 0.033333;
    mTG = TrajectoryGenerator(WALKING_TROT, ltc, lcf, dt);
    mTG.set_Ae(0.11);
    Ae_distribution =  std::uniform_real_distribution<double> (0.09, 0.11);
    CoS_distribution =  std::uniform_real_distribution<double> (-deg2rad(1), deg2rad(1));
    height_distribution =  std::uniform_real_distribution<double> (0.27, 0.27);
    crouch_distribution =  std::uniform_real_distribution<double> (0.16, 0.16);
    swing_distribution =  std::uniform_real_distribution<double> (deg2rad(0), deg2rad(0));
    noise_distribution =  std::uniform_real_distribution<double> (-deg2rad(0.), deg2rad(0.));
    dir_distribution = std::uniform_int_distribution<int>(0, 1);

    mrl = new MotionReshaper(ref_dummy);

    if (visualizable_) {
      auto vis = raisim::OgreVis::get();

      /// these method must be called before initApp
      vis->setWorld(world_.get());
      vis->setWindowSize(1800, 1200);
      vis->setImguiSetupCallback(imguiSetupCallback);
      vis->setImguiRenderCallback(imguiRenderCallBack_L);
      vis->setKeyboardCallback(raisimKeyboardCallback);
      vis->setSetUpCallback(setupCallback);
      vis->setAntiAliasing(2);
      /// starts visualizer thread
      vis->initApp();

      anymalVisual_ = vis->createGraphicalObject(alien_, "AlienGo");
      auto ref_graphics = vis->createGraphicalObject(vis_ref, "visualization");
      tracking_graphics = vis->createGraphicalObject(tracking_ball, "Tracker", "None");

      // vis->createGraphicalObject(ref_dummy, "dummy");
      vis->createGraphicalObject(ground, 20, "floor", "checkerboard_green");

      desired_fps_ = 1.0/ control_dt_;
      vis->setDesiredFPS(desired_fps_);
      vis->select(tracking_graphics->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(-2.1), Ogre::Radian(-1.1), 4, true);
      vis->addVisualObject("supportPoly0", "cylinderMesh", "red", {0.005, 0.005, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("supportPoly1", "cylinderMesh", "red", {0.005, 0.005, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("supportPoly2", "cylinderMesh", "red", {0.005, 0.005, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("supportPoly3", "cylinderMesh", "red", {0.005, 0.005, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
    }
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
    ref_dummy->setState(gc_init_, gv_init_);
    current_count = 0;
    gc_reference = gc_init_;
    gc_reference[1] += 0.7;
    change_cnt = 30;

    prev_vel.setZero();
    joint_acc.setZero();

    gc_prev = gc_init_;
    gc_noise_target = gc_init_.tail(nJoints_ + 4);
    gc_target = gc_init_.tail(nJoints_ + 4);
    gc_bal = gc_init_; gc_crouch = gc_init_;  gc_jump = gc_init_; 
    root_imit_reward = 0.0;
    joint_imit_reward = 0.0;
    ori_imit_reward = 0.0;
    end_imit_reward = 0.0;
    freq_reward = 0.0;

    phi = M_PI;
    noise.setZero(12);

    pTarget_.setZero();
    pTarget_.tail(nJoints_) = gc_init_.tail(nJoints_);
    alien_->setPTarget(pTarget_);

    std::random_device rd;
    std::mt19937 generator(rd());
    side = dir_distribution(generator);
    randomBalancePose();    
    randomCrouchPose();
    determJumpPose();

    updateObservation();

    if(visualizable_)
      gui::rewardLogger.clean();
  }

  void setVisRef()
  {
    vis_ref->setGeneralizedCoordinate(gc_reference);
  }

  void visualizeSupportPolygon()
  {
    auto& list = raisim::OgreVis::get()->getVisualObjectList();
    int cnt_feet = 0;
    float contact_thresh = 0.025;
    int foot_order[4] = {0,1,3,2};
    std::vector<int> contact_feet;

    auto contacts = alien_->getContacts();
    for(int i = 0; i < 4; ++i)
    {
      int j;
      for(j =0 ; j < contacts.size(); ++j){
        if(alien_->getBodyIdx("FR_calf") + 3 * foot_order[i] == contacts[j].getlocalBodyIndex())
        {
          cnt_feet++;
          contact_feet.push_back(foot_order[i]);
        }
      }
    }

    for(int i = 0; i < cnt_feet; ++i){
      int j = (i+1) % cnt_feet;
      raisim::Vec<3> foot1;
      alien_->getFramePosition(FR_FOOT + 3 * contact_feet[i], foot1);
      raisim::Vec<3> foot2;
      alien_->getFramePosition(FR_FOOT + 3 * contact_feet[j], foot2);

      Eigen::Vector3d half = (foot1.e() + foot2.e()) / 2 ;
      float len = (foot1.e() - foot2.e()).norm();
      raisim::Mat<3,3> rot;
      Eigen::Vector3d way = (foot1.e() - foot2.e()).normalized();

      raisim::Vec<3> direction = {way[0], way[1], way[2]};
      raisim::zaxisToRotMat(direction, rot);
      list["supportPoly"+ std::to_string(i)].setPosition(half);
      list["supportPoly"+ std::to_string(i)].setScale(Eigen::Vector3d(0.005, 0.005, len));
      list["supportPoly"+ std::to_string(i)].setOrientation(rot);
    }
    for(int i = cnt_feet; i < 4; ++i)
    {
      list["supportPoly"+ std::to_string(i)].setScale(Eigen::Vector3d(0.0, 0.0, 0.0));
    }
    contact_feet.clear();
    contact_feet.shrink_to_fit();
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
    pTarget12_ += gc_noise_target.tail(12);
    pTarget_.tail(nJoints_) = pTarget12_;
    setVisRef();

    alien_->setPTarget(pTarget_);
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

    stages();
    updateObservation();

    float totalReward = CalcReward();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("Root_reward", root_imit_reward);
      gui::rewardLogger.log("Angle_reward", joint_imit_reward);
      gui::rewardLogger.log("Ori_reward", ori_imit_reward);
      gui::rewardLogger.log("End_reward", end_imit_reward);
      gui::rewardLogger.log("Freq_reward", freq_reward);
      auto vis = raisim::OgreVis::get();

      vis->select(anymalVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }
    current_count++;
    if(!mStopFlag) 
      return totalReward;
    else
      return -1E10;
  }

  float stepTest(const Eigen::Ref<EigenVec>& action) final {

    curriculumUpdate(12300);

    // action scaling
    Eigen::VectorXd currentAction_PD = action.cast<double>();
    Eigen::VectorXd currentAction(nJoints_);
    currentAction = currentAction_PD.head(nJoints_);
    pTarget12_ = currentAction;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += gc_noise_target.tail(12);
    pTarget_.tail(nJoints_) = pTarget12_;

    setVisRef();
    alien_->setPTarget(pTarget_);
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
        visualizeSupportPolygon();
        vis->renderOneFrame();
      }      
      visualizationCounter_++;
    }
    stages();
    updateObservation();

    tracking_ball->setPosition(gc_[0], 0.5, 0.3);

    float totalReward = CalcReward();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("Root_reward", root_imit_reward);
      gui::rewardLogger.log("Angle_reward", joint_imit_reward);
      gui::rewardLogger.log("Ori_reward", ori_imit_reward);
      gui::rewardLogger.log("End_reward", end_imit_reward);
      gui::rewardLogger.log("Freq_reward", freq_reward);
    }
    current_count++;

    if(!mStopFlag) 
      return totalReward;
    else
      return -1E10;
  }


  void randomBalancePose()
  {
    std::random_device rd;
    std::mt19937 generator(rd());

    h_tg  = height_distribution(generator);
    float alpha_tg = swing_distribution(generator);
    Ae = Ae_distribution(generator);
    Ae = std::min(Ae, float(h_tg - 0.08));
    float Cs = CoS_distribution(generator);
    for(int i =0; i<12; ++i)
    {
      unsigned seed_temp = std::chrono::system_clock::now().time_since_epoch().count();
      std::default_random_engine gen(seed_temp);
      noise[i] = noise_distribution(gen);
    }
    mTG.change_Cs(Cs);
    mTG.set_Ae(Ae);
    mTG.get_tg_parameters(1.0, alpha_tg, h_tg);
    float phi;
    if(side == 0) phi = deg2rad(105);
    else phi = deg2rad(285);
    mTG.manual_timing_update(phi);
    gc_bal[2] = h_tg;
    gc_bal.tail(12) = mTG.get_u();
  }

  void randomCrouchPose()
  {
    std::random_device rd;
    std::mt19937 generator(rd());

    float crouch_height = crouch_distribution(generator);
    crouch_height = std::max(float(crouch_height), float(0.27 - Ae));
    mTG.change_Cs(0.0);
    mTG.get_tg_parameters(1.0, 0.0, crouch_height);
    mTG.manual_timing_update(0.0);
    gc_crouch[2] = crouch_height;
    gc_crouch.tail(12) = gc_bal.tail(12);

    if(side == 0)
    {
      gc_crouch.segment(7, 3) = mTG.get_u().head(3);
      gc_crouch.segment(16, 3) = mTG.get_u().tail(3);
    }
    else
    {
      gc_crouch.segment(10, 6) = mTG.get_u().segment(3, 6);
    }
  }

  void determJumpPose()
  {
    mTG.change_Cs(0.0);
    mTG.get_tg_parameters(1.0, 0.0, 0.40);
    mTG.manual_timing_update(0.0);
    gc_jump[2] = 0.40;
    gc_jump.tail(12) = gc_jump.tail(12);

    if(side == 0)
    {
      gc_jump.segment(7, 3) = mTG.get_u().head(3);
      gc_jump.segment(16, 3) = mTG.get_u().tail(3);
    }
    else
    {
      gc_jump.segment(10, 6) = mTG.get_u().segment(3, 6);
    }
  }

  void stages()
  {
    if(current_count <= 30)
    {
      float interpol = (current_count) / 30.;
      get_current_target(gc_init_, gc_bal, interpol);
    }
    else if(current_count <= 90)
    {
      gc_target = gc_bal.tail(16);
    }
    else if(current_count <= 120)
    {
      float interpol = (current_count - 90) / 30.;
      get_current_target(gc_bal, gc_crouch, interpol);
    }
    else if(current_count <= 123)
    {
      float interpol = (current_count - 120) / 3.;
      get_current_target(gc_crouch, gc_jump, interpol);
      if(current_count == 123) randomBalancePose();      
    }
    else if(current_count <= 133)
    {
      float interpol = (current_count - 123) / 10.;
      get_current_target(gc_jump, gc_bal, interpol);
    }
    else if(current_count <= 240)
    {
      gc_target = gc_bal.tail(16);
      if(current_count == 240) randomCrouchPose();
    }
    else if(current_count <= 270)
    {
      float interpol = (current_count - 240) / 30.;
      get_current_target(gc_bal, gc_crouch, interpol);
      if(current_count == 270) randomBalancePose();
    }
    else if(current_count <= 300)
    {
      float interpol = (current_count - 270) / 30.;
      get_current_target(gc_crouch, gc_bal, interpol);
    }
    else
    {
      gc_target = gc_bal.tail(16);
    }
  }

  void get_current_target(Eigen::VectorXd prev, Eigen::VectorXd future, float interpol)
  {
    gc_ = alien_->getGeneralizedCoordinate().e();
    gc_target.tail(12) = (1. - interpol) *  prev.tail(12) + interpol * future.tail(12);
    
    Eigen::Vector4i currentContact;
    currentContact.setZero();
    if(side == 0){     
      currentContact[0] = 1;
      currentContact[3] = 1;
    }
    else{
      currentContact[1] = 1;
      currentContact[2] = 1;
    }
    Eigen::VectorXd gc_raw_target = gc_init_;
    gc_raw_target[2] = (1. - interpol) *  prev[2] + interpol * future[2];
    gc_raw_target.tail(16) = gc_target;
    mrl->contactInformation(currentContact);
    mrl->getDynStates(gc_, gc_prev);
    mrl->getAngularVelocity(gv_.segment(3,3));
    mrl->getPreviousRealPose(gc_init_);
    mrl->getCurrentRawPose(gc_raw_target);
    mrl->getKinematicFixedPose();
    Eigen::VectorXd gc_reshaped = mrl->getReshapedMotion();
    gc_target = gc_reshaped.tail(16);
    gc_reference = gc_reshaped;
    gc_noise_target = gc_target;
    gc_noise_target.tail(12) += noise;
    gc_reference[1] = 0.5;
    gc_reference.tail(16) = gc_noise_target;
  }


  void curriculumUpdate(int update) final {
    int curr1 = 2000;
    int curr2 = 4000;
    int curr3 = 6000;
    int curr4 = 8000;
    int curr5 = 12000;
    if(update >= curr1 && update <curr2) 
    {
      Ae_distribution =  std::uniform_real_distribution<double> (0.08, 0.12);
      CoS_distribution =  std::uniform_real_distribution<double> (-deg2rad(3), deg2rad(3));
      height_distribution =  std::uniform_real_distribution<double> (0.26, 0.28);
      crouch_distribution =  std::uniform_real_distribution<double> (0.15, 0.18);
      swing_distribution =  std::uniform_real_distribution<double> (deg2rad(0), deg2rad(10));
      noise_distribution =  std::uniform_real_distribution<double> (-deg2rad(0.2), deg2rad(0.2));
    }
    else if(update >= curr2 && update <curr3) 
    {
      Ae_distribution =  std::uniform_real_distribution<double> (0.15, 0.18);
      CoS_distribution =  std::uniform_real_distribution<double> (-deg2rad(5), deg2rad(5));
      height_distribution =  std::uniform_real_distribution<double> (0.25, 0.29);
      crouch_distribution =  std::uniform_real_distribution<double> (0.13, 0.14);
      swing_distribution =  std::uniform_real_distribution<double> (deg2rad(0), deg2rad(20));
      noise_distribution =  std::uniform_real_distribution<double> (-deg2rad(0.5), deg2rad(0.5));
    }
    else if(update >= curr3 && update <curr4) 
    {
      Ae_distribution =  std::uniform_real_distribution<double> (0.06, 0.16);
      CoS_distribution =  std::uniform_real_distribution<double> (-deg2rad(7), deg2rad(7));
      height_distribution =  std::uniform_real_distribution<double> (0.24, 0.31);
      crouch_distribution =  std::uniform_real_distribution<double> (0.12, 0.14);
      swing_distribution =  std::uniform_real_distribution<double> (deg2rad(0), deg2rad(30));
      noise_distribution =  std::uniform_real_distribution<double> (-deg2rad(1.), deg2rad(1.));
    }
    else if(update >= curr4&& update <curr5) 
    {
      Ae_distribution =  std::uniform_real_distribution<double> (0.05, 0.18);
      CoS_distribution =  std::uniform_real_distribution<double> (-deg2rad(9), deg2rad(9));
      height_distribution =  std::uniform_real_distribution<double> (0.23, 0.32);
      crouch_distribution =  std::uniform_real_distribution<double> (0.11, 0.14);
      swing_distribution =  std::uniform_real_distribution<double> (deg2rad(0), deg2rad(40));
      noise_distribution =  std::uniform_real_distribution<double> (-deg2rad(1.2), deg2rad(1.2));
    }
    else if(update >= curr5)
    {
      Ae_distribution =  std::uniform_real_distribution<double> (0.19, 0.20);
      CoS_distribution =  std::uniform_real_distribution<double> (-deg2rad(2), deg2rad(2));
      height_distribution =  std::uniform_real_distribution<double> (0.25, 0.31);
      crouch_distribution =  std::uniform_real_distribution<double> (0.10, 0.12);
      swing_distribution =  std::uniform_real_distribution<double> (deg2rad(0), deg2rad(0));
      noise_distribution =  std::uniform_real_distribution<double> (-deg2rad(1.2), deg2rad(1.2));
    }
  } 

  float CalcReward()
  {   
    float reward = 0;
    double root_imit_err = 0;
    double joint_imit_err = 0;
    double ori_imit_err = 0;
    double end_imit_err = 0;
    double freq_err = 0;

    alien_->getState(gc_, gv_);
    for(int i = 0 ; i < gc_.size();++i){
      if(isnan(gc_[i]))
      {
        return 0;
      }
    }

    joint_acc = (gv_.tail(nJoints_) - prev_vel) / control_dt_;
    prev_vel = gv_.tail(nJoints_);

    root_imit_err = rootImitationErr();
    joint_imit_err = jointImitationErr();
    ori_imit_err = orientationImitationErr();
    end_imit_err = endEffectorImitationErr();
    freq_err = CalcAccErr();

    root_imit_reward = exp(-root_imit_scale * root_imit_err);
    joint_imit_reward = exp(-joint_imit_scale * joint_imit_err);
    ori_imit_reward = exp(-ori_imit_scale * ori_imit_err);
    end_imit_reward = exp(-end_imit_scale * end_imit_err);
    freq_reward = exp(- freq_scale * freq_err);

    reward = main_w * root_imit_reward * joint_imit_reward * ori_imit_reward * end_imit_reward 
      + freq_w * freq_reward;
    return reward;
  }

  float rootImitationErr()
  {
    float err = 0.;
    // mRP.predict(gc_noise_target.tail(12));
    err += pow((gc_init_[0] - gc_[0]),2);
    err += pow((gc_init_[1] - gc_[1]),2);
    // err += 0.1 * pow((gc_reference[2] - gc_[2]),2);
    if(120 < current_count && current_count <= 125)  err += pow((0.8 - gc_[2]) , 2);
    return err;
  }

  float jointImitationErr()
  {
    float err = 0.;
    for(int i = 0; i < nJoints_; ++i)
    {
      if(i%3 == 0)
        err += 0.05 * pow((gc_noise_target[4+i] - gc_[7 + i]) , 2);
      else
        err += pow((gc_noise_target[4 + i] - gc_[7 + i]),2);
    }
    return err;
  }

  float orientationImitationErr()
  {
    float err = 0.;
    // float epsilon = 1e-6;
    // Eigen::Matrix3d rot_m1;
    // Eigen::Matrix3d rot_m2;
    // rot_m1  = Eigen::Quaterniond(gc_[3],gc_[4],gc_[5],gc_[6]);
    // rot_m2  = Eigen::Quaterniond(gc_noise_target[0],gc_noise_target[1],gc_noise_target[2],gc_noise_target[3]);
    // Eigen::Vector3d current_dir = rot_m1 * Eigen::Vector3d::UnitX();
    // Eigen::Vector3d next_dir = rot_m2 * Eigen::Vector3d::UnitX();
    // double value = (current_dir[0] * next_dir[0] + current_dir[1] * next_dir[1]) / 
    //   (std::sqrt(pow(current_dir[0], 2) + pow(current_dir[1], 2)) * std::sqrt(pow(next_dir[0], 2) + pow(next_dir[1], 2)));
    // value = std::min(1.0 - epsilon, value);
    // value = std::max(-1.0 + epsilon, value);
    // float angle = acos(value);
    // err = angle * angle;
    // // Use slerp
    // // Eigen::Quaterniond target(gc_target[0],gc_target[1],gc_target[2],gc_target[3]);
    // // Eigen::Quaterniond current(gc_[3],gc_[4],gc_[5],gc_[6]);
    // // err = pow(current.angularDistance(target), 2);
    // Eigen::Quaterniond target(gc_target[0], gc_target[1], gc_target[2], gc_target[3]);
    // Eigen::Quaterniond current(gc_[3],gc_[4],gc_[5],gc_[6]);
    // err = pow(current.angularDistance(target), 2);
    return err;
  }

  float endEffectorImitationErr()
  {
    float err = 0.;
    Eigen::VectorXd gc_extended = gc_init_;
    if(current_count > 120 && current_count < 128)
      gc_extended = gc_;
    gc_extended.tail(4 + nJoints_) = gc_noise_target;
    gc_extended[2] = h_tg;
    ref_dummy->setGeneralizedCoordinate(gc_extended);
    // Use endeffector position except for 120~123
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
    obScaled_.segment(32, 12) = gc_noise_target.tail(12);
    Eigen::Quaterniond quat3(gc_noise_target[0], gc_noise_target[1], gc_noise_target[2], gc_noise_target[3]);
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

    for(int i =0 ; i < gc_.size();++i)
    {
      if(isnan(gc_[i])) return true;
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
  raisim::ArticulatedSystem* ref_dummy;
  raisim::ArticulatedSystem* vis_ref;
  raisim::Sphere* tracking_ball;
  MotionReshaper *mrl;


  std::vector<GraphicObject>* anymalVisual_;
  std::vector<GraphicObject>* tracking_graphics;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, torque_, gc_dummy_init;
  Eigen::VectorXd gc_prev, gc_target, gc_noise_target, gc_reference;
  Eigen::VectorXd gc_bal, gc_crouch, gc_jump;
  Eigen::VectorXd noise;
  double desired_fps_ = 30.;
  int visualizationCounter_=0;
  float effort_limit = 33.5 * 0.90;
  Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_;
  Eigen::VectorXd obDouble_, obScaled_;
  std::set<size_t> footIndices_;

  int current_count;
  float h_tg, Ae;
  Eigen::VectorXd prev_vel, joint_acc;
  std::vector<std::deque<double>> torque_sequence;

  // Reward Shapers
  double main_w, freq_w;
  double root_imit_scale, joint_imit_scale, ori_imit_scale, end_imit_scale, freq_scale;
  double root_imit_reward, joint_imit_reward, ori_imit_reward, end_imit_reward, freq_reward;

  // Kp Settings
  double init_pgain = 200;
  double init_dgain = 10;
  Eigen::VectorXd jointPgain, jointDgain;

  int change_cnt = 30;
  int stop_cnt = 15;
  double phi, phase_speed;
  int side;
  // Imitation Learning
  int nRefClips;

  std::uniform_real_distribution<double> Ae_distribution;
  std::uniform_real_distribution<double> height_distribution;
  std::uniform_real_distribution<double> crouch_distribution;
  std::uniform_real_distribution<double> swing_distribution;
  std::uniform_real_distribution<double> CoS_distribution;
  std::uniform_real_distribution<double> noise_distribution;
  std::uniform_int_distribution<int> dir_distribution;
  TrajectoryGenerator mTG;
};

}