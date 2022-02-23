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
    ref_dummy = new raisim::ArticulatedSystem(resourceDir_+"/a1/a1/a1.urdf");
    ik_dummy = new raisim::ArticulatedSystem(resourceDir_+"/a1/a1/a1.urdf");
    vis_ref = new raisim::ArticulatedSystem(resourceDir_+"/a1_ref/a1/a1.urdf");       
    alien_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    auto ground = world_->addGround(0, "floor");
    world_->setERP(0,0);
    /// get robot data
    gcDim_ = alien_->getGeneralizedCoordinateDim();
    gvDim_ = alien_->getDOF();
    nJoints_ = 12;

    control_dt_ = 0.0333333;
    // READ_YAML(double, control_dt_, cfg["control_dt"])

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    torque_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);
    gc_target.setZero(4 + nJoints_); gc_prev.setZero(gcDim_);

    /// this is nominal configuration of anymal
    sittingRoot[0] = -(0.1805 + 0.05);
    sittingRoot[1] = 0.0;
    sittingRoot[2] = 0.1805 + 0.1* sqrt(3);
    Eigen::Matrix3d m;
    m = Eigen::AngleAxisd(deg2rad(-90), Eigen::Vector3d::UnitY());
    Eigen::Quaterniond current_quat(m);
    // gc_init_ <<  0, 0, 0.27, 1.0, 0.0, 0.0, 0.0, 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99);
    gc_init_ <<  sittingRoot[0], sittingRoot[1] ,sittingRoot[2], current_quat.w(), current_quat.x(), current_quat.y(), current_quat.z(), 0.27, deg2rad(50), -1.05, -0.27, deg2rad(50), -1.05, 0., deg2rad(120), deg2rad(-120), 0., deg2rad(120), deg2rad(-120);
    alien_->setState(gc_init_, gv_init_);
    ref_dummy->setState(gc_init_, gv_init_);
    vis_ref->setState(gc_init_, gv_init_);
    ik_dummy->setState(gc_init_, gv_init_);

    for(int i = 0; i < 4; ++i)
    {
      raisim::Vec<3> footPositionA;
      alien_->getFramePosition(FR_FOOT+3 * i, footPositionA);
      init_feet_poses.push_back(footPositionA.e());
    }

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

    obDim_ = 164;
    actionDim_ = nJoints_;

    actionStd_.setZero(nJoints_);  actionMean_.setZero(nJoints_);
    obMean_.setZero(obDim_); obStd_.setZero(obDim_);
    prev_vel.setZero(nJoints_); joint_acc.setZero(nJoints_);

    float ah = 0.8028, at = 2.6179, ac = 0.8901;
    float mh = 0, mt = M_PI /2, mc = -1.806;
    actionStd_ << ah, at, ac, ah, at, ac, ah, at, ac, ah, at, ac;
    actionMean_ << mh, mt, mc, mh, mt, mc, mh, mt, mc, mh, mt, mc;

    freq_w = 0.1;
    main_w = 1.0 - freq_w;

    joint_imit_scale = 0.1;
    root_imit_scale = 2.0;
    ori_imit_scale = 200.0;
    end_imit_scale = 10.0;
    freq_scale = 1e-4;

    default_friction_coeff = 0.7;
    raisim::MaterialManager materials_;
    materials_.setMaterialPairProp("floor", "robot", default_friction_coeff, 0.0, 0.0);
    world_->updateMaterialProp(materials_);
    alien_->getCollisionBody("FR_foot/0").setMaterial("robot");
    alien_->getCollisionBody("FL_foot/0").setMaterial("robot");
    alien_->getCollisionBody("RL_foot/0").setMaterial("robot");
    alien_->getCollisionBody("RR_foot/0").setMaterial("robot");

    x_distribution = std::uniform_real_distribution<double> (deg2rad(-7), deg2rad(7));
    y_distribution = std::uniform_real_distribution<double> (deg2rad(-7), deg2rad(7));
    z_distribution = std::uniform_real_distribution<double> (deg2rad(-7), deg2rad(7));
    ref_noise_distribution = std::uniform_real_distribution<double> (deg2rad(-1), deg2rad(1));
    duration_distribution = std::uniform_int_distribution<int> (20,40);
    noise_distribution = std::uniform_real_distribution<double> (deg2rad(-10), deg2rad(10));

    f_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(10), deg2rad(10));
    f_thigh_distribution = std::uniform_real_distribution<double> (deg2rad(50), deg2rad(65));
    f_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(110), -deg2rad(85));

    mass_distribution = std::uniform_real_distribution<double> (0.95, 1.05);
    pgain_distribution = std::uniform_real_distribution<double> (0.95, 1.05);
    dgain_distribution = std::uniform_real_distribution<double> (0.95, 1.05);
    friction_distribution = std::uniform_real_distribution<double> (0.9, 1.1);
    latency_distribution = std::uniform_real_distribution<double> (1.0, 1.1);
    original_mass  = alien_->getMass();

    gui::rewardLogger.init({"Angle_reward","Root_reward", "Ori_reward", "End_reward", "Freq_reward"});

    /// indices of links that should not make contact with ground
    footIndices_.insert(alien_->getBodyIdx("FL_calf"));
    footIndices_.insert(alien_->getBodyIdx("FR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RL_calf"));
    footIndices_.insert(alien_->getBodyIdx("FL_thigh"));
    footIndices_.insert(alien_->getBodyIdx("FR_thigh"));
    footIndices_.insert(alien_->getBodyIdx("RR_thigh"));
    footIndices_.insert(alien_->getBodyIdx("RL_thigh"));

    mAIK = new AnalyticFullIK(ik_dummy);

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
      vis->initApp();

      anymalVisual_ = vis->createGraphicalObject(alien_, "AlienGo");
      auto ref_graphics = vis->createGraphicalObject(vis_ref, "visualization");

      // vis->createGraphicalObject(ref_dummy, "dummy");
      vis->createGraphicalObject(ground, 10, "floor", "checkerboard_green");

      desired_fps_ = 1.0/ control_dt_;
      vis->setDesiredFPS(desired_fps_);
      vis->select(anymalVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }
      }

  ~ENVIRONMENT() final = default;

  void init() final { }

  void kinect_init() final{
    // Here, it is only used in the test step
    // Do the initialization of test step like initialize the human 
    TGcurriculum(3000);
    Armcurriculum(7200);
    max_cnt = 2000;
    // std::string file_name = "/home/sonic/Project/Alien_Gesture/tilting.txt";
    // writeActions.open(file_name);
    // testStep = true;
    // DRcurriculum(1400);
    // From here, do it
    // readHumanPoints("/home/sonic/Project/Alien_Gesture/KinectRaisimtest/build/rsc/points_leanlong.txt");
    // init_bones();
    // human_cnt = 0;
    // currentContact = Eigen::Vector4i(1,1,1,1);
    // prevContact = Eigen::Vector4i(1,1,1,1);
    // kin_gc_prev = gc_init_;
    // init_foot = init_feet_poses[0][2];
  }


  void estimateContact(const Eigen::Ref<EigenVec>& contact)
  {
    Eigen::VectorXd contact_vec = contact.cast<double>();
    for(int i = 0; i <4; ++i)
    {
      if(contact_vec[i] > 0.5)
        currentContact[i] = 0;
      else
        currentContact[i] = 1;
    }
  }

  void getReferencePoses(const Eigen::Ref<EigenVec>& pose)
  {
    Eigen::VectorXd currentAction = pose.cast<double>();
    // if(currentAction[0] > 0)
    //   gc_target.head(4) = currentAction.head(4);
    // else
    //   gc_target.head(4) = -currentAction.head(4);
    // gc_target.tail(12) = currentAction.tail(12);
    setReshapedMotion(currentAction);
  }

  void myReshapedMotion(Eigen::VectorXd currentAction)
  {
    Eigen::VectorXd raw_pose(gcDim_);
    raw_pose.head(3) = gc_init_.head(3);
    raw_pose.tail(16) = currentAction.tail(16);
    raw_pose.tail(12) = gc_init_.tail(12);
    // ik_dummy->setGeneralizedCoordinate(raw_pose);
    // float current_foot = 10000;
    // for(int i= 0; i < 4; ++i)
    // {
    //   raisim::Vec<3> temp;
    //   ik_dummy->getFramePosition(FR_FOOT + 3 * i, temp);
    //   if(temp[2] < current_foot) current_foot = temp[2];
    // }
    // raw_pose[2] += (init_foot - current_foot);
    // raw_pose[2] = 0.24;
    mAIK->setCurrentPose(raw_pose);
    mAIK->setReady();  
    for(int i = 0 ; i < 4; ++i){
      mAIK->appendTargets(init_feet_poses[i]);
    }
    mAIK->solve(raw_pose);
    Eigen::VectorXd current_pose = mAIK->getSolution();
    mAIK->clearTargets();
    if(currentAction[0] > 0)
      gc_target.head(4) = currentAction.head(4);
    else
      gc_target.head(4) = -currentAction.head(4);
    gc_target.tail(12)= current_pose;
  }

  void setReshapedMotion(Eigen::VectorXd currentAction)
  {
    for(int i = 0; i  < 4 ; ++i)
    {
      if(currentContact[i] == 0)
      {
        kin_gc_prev.segment(7 + 3 * i, 3) = currentAction.segment(4 + 3 * i, 3);
      }
      else if(currentContact[i] - prevContact[i] == 1)
      {
        kin_gc_prev.segment(7 + 3 * i, 3) = currentAction.segment(4 + 3 * i, 3);
      }
    }

    Eigen::VectorXd gc_raw = gc_init_;
    gc_raw.tail(16) = currentAction;
    ik_dummy->setGeneralizedCoordinate(gc_raw);
    float current_foot = 10000;
    for(int i= 0; i < 4; ++i)
    {
      raisim::Vec<3> temp;
      ik_dummy->getFramePosition(FR_FOOT + 3 * i, temp);
      if(temp[2] < current_foot) current_foot = temp[2];
    }
    gc_raw[2] += (init_foot - current_foot);
    MotionReshaper *mrl = new MotionReshaper(ik_dummy);
    // mrl->getDynStates(gc_, gc_prev);
    mrl->contactInformation(currentContact);
    mrl->getPreviousRealPose(kin_gc_prev);
    mrl->getCurrentRawPose(gc_raw);
    mrl->getKinematicFixedPoseTest();
    Eigen::VectorXd gc_reshaped = mrl->getReshapedMotion();
    gc_reference[2] =gc_reshaped[2];
    gc_target = gc_reshaped.tail(16);
    if(currentAction[0] > 0)
      gc_target.head(4) = gc_target.head(4);
    else
      gc_target.head(4) = -gc_target.head(4);
    delete mrl;
    if(human_cnt < 1) gc_target = gc_init_.tail(16);
    prevContact = currentContact;
  }

  void reset() final
  {
    TGcurriculum(12000);
    Armcurriculum(8500);
    DRcurriculum(11000);
    alien_->setState(gc_init_, gv_init_);
    vis_ref->setState(gc_init_, gv_init_);
    ik_dummy->setState(gc_init_, gv_init_);
    ref_dummy->setState(gc_init_, gv_init_);
    current_count = 0;

    gc_reference = gc_init_;
    gc_reference[1] += 0.7;

    prev_vel.setZero();
    joint_acc.setZero();
    gc_prev = gc_init_;
    gc_target = gc_init_.tail(nJoints_ + 4);

    trajectories.clear();
    trajectories.shrink_to_fit();
    rootTraj.clear();
    rootTraj.shrink_to_fit();

    reference_history.clear();
    reference_history.shrink_to_fit();
    state_history.clear();
    action_history.clear();
    state_history.shrink_to_fit();
    action_history.shrink_to_fit();
    state_history.push_back(gc_init_.tail(16));
    reference_history.push_back(gc_init_.tail(16));
    for(int i = 0; i < 3; ++i)
    {
      reference_history.push_back(gc_init_.tail(16));
      state_history.push_back(gc_init_.tail(16));
      action_history.push_back(gc_init_.tail(12));
    }

    joint_imit_reward = 0.0;
    root_imit_reward = 0.0;
    ori_imit_reward = 0.0;
    end_imit_reward = 0.0;
    freq_reward = 0.0;

    pTarget_.setZero();
    pTarget_.tail(nJoints_) = gc_init_.tail(nJoints_);
    alien_->setPdTarget(pTarget_, vTarget_);

    generateRandomSitting();
    // generateSit();
    // generateRandomSeq();
    randomizeDomain();
    updateObservation();
    

    if(visualizable_)
      gui::rewardLogger.clean();
  }

  void setVisRef()
  {
    gc_reference.tail(16) = gc_target;
    gc_reference.head(3) = rootTraj[current_count];
    gc_reference[1] += 0.7;    
    vis_ref->setGeneralizedCoordinate(gc_reference);
  }

  void readHumanPoints(std::string human_points)
  { 
    // Reader ==> Save as vector
    std::ifstream human_input(human_points);
    if(!human_input){
      std::cout << "Expected valid input file name" << std::endl;
    }

    int triple_bone = 3 * K4ABT_JOINT_COUNT;
    while(!human_input.eof()){
      Eigen::VectorXd humanoid(triple_bone);
      for(int i = 0 ; i < triple_bone; ++i)
      {
        float ang;
        human_input >> ang;
        humanoid[i] = ang;
      }
      human_point_list.push_back(humanoid);
    }
    // human_point_list.pop_back();
    human_input.close();
  }


  void generateRandomSitting()
  {
    // start
    int cnt = 0; 
    std::random_device rd;
    std::mt19937 generator(rd());
    // Initialize for some duration
    int inbetween = duration_distribution(generator) / 10;
    Eigen::Quaterniond prev_quat(gc_init_[3],gc_init_[4],gc_init_[5],gc_init_[6]);
    Eigen::VectorXd prev_pose = gc_init_.tail(12);
    motionBlender(prev_quat, prev_quat, prev_pose, prev_pose, inbetween);
    cnt+= inbetween;
    for(int i =0 ; i< inbetween; ++i)
    {
      rootTraj.push_back(sittingRoot);
    }
    Eigen::VectorXd sitpose(12);
    sitpose << 0.27, deg2rad(50), -1.05, -0.27, deg2rad(50), -1.05, 0., deg2rad(120), deg2rad(-120), 0., deg2rad(120), deg2rad(-120);
    // Now tilt + manipulation
    Eigen::Vector3d prev_root = sittingRoot;
    prev_pose = sitpose;
    while(cnt < max_cnt)
    {
      std::random_device rd;
      std::mt19937 generator(rd());
      int inbetween = duration_distribution(generator);
      float x_angle = x_distribution(generator);
      float y_angle = y_distribution(generator);
      float z_angle = z_distribution(generator);
      Eigen::Matrix3d m;
      m = Eigen::AngleAxisd(x_angle, Eigen::Vector3d::UnitX())
        * Eigen::AngleAxisd(y_angle + deg2rad(-90),  Eigen::Vector3d::UnitY())
        * Eigen::AngleAxisd(z_angle, Eigen::Vector3d::UnitZ());

      Eigen::Matrix3d root_comp;
      root_comp = Eigen::AngleAxisd(x_angle, Eigen::Vector3d::UnitX())
        * Eigen::AngleAxisd(y_angle,  Eigen::Vector3d::UnitY())
        * Eigen::AngleAxisd(z_angle, Eigen::Vector3d::UnitZ());
      Eigen::Quaterniond current_quat(m);
      Eigen::VectorXd current_pose(12);
      current_pose.tail(6) = sitpose.tail(6);
      for(int idx = 0; idx <2; ++idx)
      {
        current_pose[3 * idx + 0] = f_hip_distribution(generator);
        current_pose[3 * idx + 1] = f_thigh_distribution(generator);
        current_pose[3 * idx + 2] = f_calf_distribution(generator);
      }

      // motionBlender(prev_quat, current_quat, sitpose, sitpose, inbetween);
      motionBlender(prev_quat, current_quat, prev_pose, current_pose, inbetween);
      prev_quat = current_quat;
      prev_pose = current_pose;

      Eigen::Vector3d adder = root_comp * Eigen::Vector3d(0,0,0.1805);
      Eigen::Vector3d current_root;
      current_root[0] = sittingRoot[0] + adder[0];
      current_root[1] = adder[1];
      current_root[2] = 0.1* sqrt(3) + adder[2];
      for(int i = 0; i < inbetween; ++i)
      {
        Eigen::Vector3d temp;
        temp = (current_root * i + prev_root * (inbetween-i))/ inbetween;
        rootTraj.push_back(temp);
      }
      prev_root = current_root;
      cnt += inbetween;

      int inbetween2 = duration_distribution(generator) / 4;
      motionBlender(current_quat, current_quat, current_pose, current_pose, inbetween2);
      cnt+= inbetween2;
      for(int i = 0; i < inbetween2; ++i)
        rootTraj.push_back(current_root);
    }
  }

  // void generateSit()
  // {
  //   // start
  //   Eigen::Quaterniond prev_quat(gc_init_[3],gc_init_[4],gc_init_[5],gc_init_[6]);
  //   Eigen::VectorXd prev_pose = gc_init_.tail(12);
  //   std::vector<Eigen::Vector3d> targets;
  //   Eigen::VectorXd gc_adj;
  //   gc_adj.setZero(19);
  //   gc_adj.head(3)= Eigen::Vector3d(0,0,0.27);
  //   gc_adj.tail(16) = gc_init_.tail(16);;
  //   gc_adj[2] = 0.12;
  //   for(int i = 0; i < 4; ++i)
  //   {
  //     raisim::Vec<3> current;
  //     alien_->getFramePosition(FR_FOOT+ 3 * i, current);
  //     targets.push_back(current.e());
  //   }
  //   mAIK->setReady();
  //   mAIK->setFullTargets(targets);
  //   mAIK->solve(gc_adj);
  //   auto sol = mAIK->getSolution();
  //   mAIK->clearTargets();
  //   motionBlender(prev_quat, prev_quat, prev_pose, sol, 6, 4);
  //   prev_pose = sol;

  //   Eigen::Matrix3d m;
  //   m = Eigen::AngleAxisd(deg2rad(-90), Eigen::Vector3d::UnitY());
  //   Eigen::Quaterniond current_quat(m);
  //   Eigen::VectorXd sitpose(12);
  //   sitpose << 0.27, deg2rad(50), -1.05, -0.27, deg2rad(50), -1.05, 0., deg2rad(120), deg2rad(-120), 0., deg2rad(120), deg2rad(-120);
  //   motionBlender(prev_quat, current_quat, prev_pose, sitpose, 12, 4);
  //   int stay = 30;
  //   motionBlender(current_quat, current_quat, sitpose, sitpose, 30, 4);
  //   motionBlender(current_quat, prev_quat, sitpose, prev_pose , 12, 4);
  //   motionBlender(prev_quat, prev_quat, prev_pose, gc_init_.tail(12) , 6, 4);
  //   motionBlender(prev_quat, prev_quat, gc_init_.tail(12), gc_init_.tail(12) , 1, 4);
  // }

  void generateRandomSeq()
  {
    // start
    int cnt = 0; 
    std::random_device rd;
    std::mt19937 generator(rd());
    int inbetween = duration_distribution(generator);
    Eigen::Quaterniond prev_quat(gc_init_[3],gc_init_[4],gc_init_[5],gc_init_[6]);
    Eigen::VectorXd prev_pose = gc_init_.tail(12);
    motionBlender(prev_quat, prev_quat, prev_pose, prev_pose, inbetween);
    cnt+= inbetween;
    for(int i =0 ; i< inbetween; ++i)
    {
      rootTraj.push_back(gc_init_.head(3));
    }
    std::vector<Eigen::Vector3d> targets;
    Eigen::VectorXd gc_adj = gc_init_;
    gc_adj[2] = 0.12;
    for(int i = 0; i < 4; ++i)
    {
      raisim::Vec<3> current;
      alien_->getFramePosition(FR_FOOT+ 3 * i, current);
      targets.push_back(current.e());
    }
    mAIK->setReady();
    mAIK->setFullTargets(targets);
    mAIK->solve(gc_adj);
    auto sol = mAIK->getSolution();
    mAIK->clearTargets();
    motionBlender(prev_quat, prev_quat, prev_pose, sol, 6);
    // motionBlender(prev_quat, prev_quat, sol, sol, 15);
    for(int i = 0; i < 6; ++i)
    {
      Eigen::Vector3d temp = gc_init_.head(3);
      temp[2] = (gc_adj[2] * i + gc_init_[2] * (6-i))/ 6.;
      rootTraj.push_back(temp);
    }
    for(int i = 0; i < 15; ++i)
    {
      rootTraj.push_back(gc_adj.head(3));
    }
    prev_pose = sol;

    Eigen::Matrix3d m;
    m = Eigen::AngleAxisd(deg2rad(-90), Eigen::Vector3d::UnitY());
    Eigen::Quaterniond current_quat(m);
    Eigen::VectorXd sitpose(12);
    sitpose << 0.27, deg2rad(50), -1.05, -0.27, deg2rad(50), -1.05, 0., deg2rad(120), deg2rad(-120), 0., deg2rad(120), deg2rad(-120);
    motionBlender(prev_quat, current_quat, prev_pose, sitpose, 12);
    for(int i = 0; i < 12; ++i)
    {
      Eigen::Vector3d temp = gc_adj.head(3);
      temp = (sittingRoot * i + gc_adj * (12-i))/ 12.;
      rootTraj.push_back(temp);
    }
    int stay = 5* duration_distribution(generator);
    // for(int i = 0; i < max_cnt - cnt; ++i)
    // {
    //   rootTraj.push_back(sittingRoot);
    // }
    // motionBlender(current_quat, current_quat, sitpose, sitpose, max_cnt - cnt);

    motionBlender(current_quat, current_quat, sitpose, sitpose, stay);
    for(int i = 0; i < stay; ++i)
    {
      rootTraj.push_back(sittingRoot);
    }
    cnt += stay;
    // prev_quat = current_quat;
    // prev_pose = sitpose;

    motionBlender(current_quat, prev_quat, sitpose, prev_pose , 12);
    for(int i = 0; i < 12; ++i)
    {
      Eigen::Vector3d temp = gc_adj.head(3);
      temp = (sittingRoot* (12-i)  + gc_adj * i)/ 12.;
      rootTraj.push_back(temp);
    }
    motionBlender(prev_quat, prev_quat, prev_pose, gc_init_.tail(12) , 6);
    for(int i = 0; i < 6; ++i)
    {
      Eigen::Vector3d temp = gc_init_.head(3);
      temp[2] = (gc_adj[2] * (6-i) + gc_init_[2] * i)/ 6.;
      rootTraj.push_back(temp);
    }
    motionBlender(prev_quat, prev_quat, gc_init_.tail(12), gc_init_.tail(12) , max_cnt);
    for(int i = 0; i < max_cnt; ++i)
    {
      Eigen::Vector3d temp = gc_init_.head(3);
      rootTraj.push_back(temp);
    }
  //   while(cnt < max_cnt)
  //   {
  //     std::random_device rd;
  //     std::mt19937 generator(rd());
  //     int inbetween = duration_distribution(generator);
  //     float x_angle = x_distribution(generator);
  //     float y_angle = y_distribution(generator);
  //     float z_angle = z_distribution(generator);
  //     Eigen::Matrix3d m;
  //     m = Eigen::AngleAxisd(x_angle, Eigen::Vector3d::UnitX())
  //       * Eigen::AngleAxisd(y_angle + deg2rad(-90),  Eigen::Vector3d::UnitY())
  //       * Eigen::AngleAxisd(z_angle, Eigen::Vector3d::UnitZ());

  //     Eigen::Quaterniond current_quat2(m);
  //     motionBlender(prev_quat, current_quat2, sitpose, sitpose, inbetween);
  //     prev_quat = current_quat2;
  //     for(int i = 0; i < inbetween; ++i)
  //     {
  //       rootTraj.push_back(sittingRoot);
  //     }
  //     cnt += inbetween;
  //   }
  }

  void motionBlender(Eigen::Quaterniond q1, Eigen::Quaterniond q2, 
    Eigen::VectorXd now, Eigen::VectorXd next, int inbetween)
  {
    for(int j = 0 ; j < inbetween ; ++j)
    {
      std::random_device rd;
      std::mt19937 generator(rd());
      Eigen::Quaterniond slerped = q1.slerp(float(j) / float(inbetween), q2);
      Eigen::VectorXd interpol = (now *(inbetween - j) / inbetween) + (next *j / inbetween);
      Eigen::VectorXd total(4 + nJoints_);
      slerped.normalize();
      total[0] = slerped.w();
      total[1] = slerped.x();
      total[2] = slerped.y();
      total[3] = slerped.z();
      total.tail(nJoints_) = interpol;
      for(int i = 0; i < 6; ++i)
      {
        float noise = ref_noise_distribution(generator);
        total[4 + i] += noise;
      }
      for(int i = 6; i < 12; ++i)
      {
        float noise = noise_distribution(generator);
        total[4 + i] += noise;
      }
      trajectories.push_back(total);
    }
  }

  void randomizeDomain()
  {
    // What to randomize? 
    // 1) PD gain
    // 2) Mass of the body segment - use setMass() and updateMassInfo()
    // 3) Sensor delay -- comunication delay
    // 4) Damping and stiffness
    std::random_device rd;
    std::mt19937 generator(rd());

    for(int i = 0 ; i < 13; ++i)
    {
      float randomized_mass = mass_distribution(generator) * original_mass[i];
      alien_->setMass(i, randomized_mass);
    }

    Eigen::VectorXd rand_pgain(gvDim_);
    Eigen::VectorXd rand_dgain(gvDim_);

    for(int i = 0; i < gvDim_; ++i)
    {
      float ithpgain = pgain_distribution(generator) * jointPgain[i];
      rand_pgain[i] = ithpgain;
      float ithdgain = dgain_distribution(generator) * jointDgain[i];
      rand_dgain[i] = ithdgain;
    }
    float new_coeff = default_friction_coeff *  friction_distribution(generator);
    raisim::MaterialManager materials_;
    materials_.setMaterialPairProp("floor", "robot", new_coeff, 0.0, 0.0);
    world_->updateMaterialProp(materials_);

    alien_->setPdGains(rand_pgain, rand_dgain);
    alien_->updateMassInfo();
  }

  void init_bones()
  {
    auto vis = raisim::OgreVis::get();
    for(int i =0 ; i < 31; ++i){
      vis->addVisualObject("bone" + std::to_string(i), "cylinderMesh", "red", {0.015, 0.015, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
    }
    for(int i = 0 ; i < K4ABT_JOINT_COUNT; ++i)
    {
      vis->addVisualObject("joint" + std::to_string(i), "sphereMesh", "blue", {0.03, 0.03, 0.03}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
    }
  }

  void visualizeBones()
  {
    float scaler = 2.1;
    float y_shift = 0.2;
    auto& list = raisim::OgreVis::get()->getVisualObjectList();    

    Eigen::VectorXd current_keypose = scaler * human_point_list[human_cnt];
    float foot_z = current_keypose[3*K4ABT_JOINT_FOOT_LEFT + 2] > current_keypose[3*K4ABT_JOINT_FOOT_RIGHT + 2] ? current_keypose[3*K4ABT_JOINT_FOOT_RIGHT + 2]: current_keypose[3*K4ABT_JOINT_FOOT_LEFT + 2];

    for(size_t boneIdx = 0; boneIdx < K4ABT_JOINT_COUNT; boneIdx++)    
    {
      Eigen::Vector3d point(current_keypose[3*boneIdx], current_keypose[3*boneIdx+1]- y_shift, current_keypose[3*boneIdx+2] - foot_z);
      list["joint" + std::to_string(boneIdx)].setPosition(point);
    }

    for(size_t boneIdx = 0; boneIdx < g_boneList.size(); boneIdx++)
    {
      k4abt_joint_id_t joint1 = g_boneList[boneIdx].first;
      k4abt_joint_id_t joint2 = g_boneList[boneIdx].second;
      // if(joint1 > K4ABT_JOINT_HIP_LEFT || joint2 > K4ABT_JOINT_HIP_LEFT) continue;
      Eigen::Vector3d point1(current_keypose[3*joint1], current_keypose[3*joint1+1], current_keypose[3*joint1+2]- foot_z);
      Eigen::Vector3d point2(current_keypose[3*joint2], current_keypose[3*joint2+1], current_keypose[3*joint2+2]- foot_z);
      Eigen::Vector3d half = (point1 + point2) / 2 ;
      half[1] -= y_shift ;
      float len = (point2 - point1).norm();
      raisim::Mat<3,3> rot;
      Eigen::Vector3d way = (point2 - point1).normalized();
      raisim::Vec<3> direction = {way[0], way[1], way[2]};
      raisim::zaxisToRotMat(direction, rot);
      list["bone" + std::to_string(boneIdx)].setPosition(half);
      list["bone" + std::to_string(boneIdx)].setScale(Eigen::Vector3d(0.015, 0.015, len));
      list["bone" + std::to_string(boneIdx)].setOrientation(rot);
    }
  }

  float stepRef() final
  {
    return 0;
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    gc_target = trajectories[current_count];
    /// action scaling
    Eigen::VectorXd currentAction = action.cast<double>();
    pTarget12_ = currentAction;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(12) = pTarget12_;
    setVisRef();
    alien_->setPTarget(pTarget_);
    std::random_device rd;
    std::mt19937 generator(rd());
    float new_c_dt = control_dt_ * latency_distribution(generator);
    auto loopCount = int(new_c_dt / simulation_dt_ + 1e-10);
    auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);    
    action_history.erase(action_history.begin());
    action_history.push_back(pTarget12_);

    for(int i=0; i<loopCount; i++) {
      world_->integrate();  
      if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
      {
        auto vis = raisim::OgreVis::get();
        vis->renderOneFrame();
      }      
      visualizationCounter_++;
    }
    float totalReward = CalcReward();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("Angle_reward", joint_imit_reward);
      gui::rewardLogger.log("Root_reward", root_imit_reward);
      gui::rewardLogger.log("Ori_reward", ori_imit_reward);
      gui::rewardLogger.log("End_reward", end_imit_reward);
      gui::rewardLogger.log("Freq_reward", freq_reward);
      auto vis = raisim::OgreVis::get();

      // vis->select(anymalVisual_->at(0), false);
      // vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }
    current_count++;
    updateObservation();
    if(!mStopFlag) 
      return totalReward;
    else
      return -1E10;
  }

  float stepTest(const Eigen::Ref<EigenVec>& action) final {
    gc_target = trajectories[current_count];
    /// action scaling
    Eigen::VectorXd currentAction = action.cast<double>();
    pTarget12_ = currentAction;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(12) = pTarget12_;
    setVisRef();
    alien_->setPTarget(pTarget_);
    // float new_c_dt = control_dt_ * latency_distribution(generator);
    auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
    auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);    
    action_history.erase(action_history.begin());
    action_history.push_back(pTarget12_);

    for(int i=0; i<loopCount; i++) {
      world_->integrate();  
      // visualizeBones();
      if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
      {
        auto vis = raisim::OgreVis::get();
        vis->renderOneFrame();
      }      
      visualizationCounter_++;
    }
    float totalReward = CalcReward();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("Angle_reward", joint_imit_reward);
      gui::rewardLogger.log("Root_reward", root_imit_reward);
      gui::rewardLogger.log("Ori_reward", ori_imit_reward);
      gui::rewardLogger.log("End_reward", end_imit_reward);
      gui::rewardLogger.log("Freq_reward", freq_reward);
      auto vis = raisim::OgreVis::get();

      // vis->select(anymalVisual_->at(0), false);
      // vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }
    human_cnt++;
    current_count++;
    updateObservation();

    if(!mStopFlag) 
      return totalReward;
    else{
      writeActions.close();
      return -1E10;
    }
  }

  void curriculumUpdate(int update) final {
    TGcurriculum(12000);
    Armcurriculum(8500);
    DRcurriculum(update);
    // if(update > 1000)ori_imit_scale = 200;
    // DRcurriculum(1400);
  } 

  void TGcurriculum(int update)
  {
    int curr1 = 2000;
    int curr2 = 3000;
    int curr3 = 4000;
    int curr4 = 7000;
    int curr5 = 10000;

    if(update > curr1 && update <=curr2)
    {
      // x_distribution = std::uniform_real_distribution<double> (deg2rad(-10), deg2rad(10));
      y_distribution = std::uniform_real_distribution<double> (deg2rad(-10), deg2rad(10));
      // z_distribution = std::uniform_real_distribution<double> (deg2rad(-10), deg2rad(10));
      duration_distribution = std::uniform_int_distribution<int> (20,40);
    }
    else if(update > curr2 && update <=curr3)
    {
      x_distribution = std::uniform_real_distribution<double> (deg2rad(-10), deg2rad(10));
      y_distribution = std::uniform_real_distribution<double> (deg2rad(-15), deg2rad(20));
      z_distribution = std::uniform_real_distribution<double> (deg2rad(-10), deg2rad(10));
      duration_distribution = std::uniform_int_distribution<int> (10,30);
    }
    else if(update > curr3 && update <=curr4)
    {
      // x_distribution = std::uniform_real_distribution<double> (deg2rad(-20), deg2rad(20));
      y_distribution = std::uniform_real_distribution<double> (deg2rad(-15), deg2rad(25));
      // z_distribution = std::uniform_real_distribution<double> (deg2rad(-20), deg2rad(20));
      duration_distribution = std::uniform_int_distribution<int> (20,40);
    }
    else if(update > curr4&& update <=curr5)
    {
      x_distribution = std::uniform_real_distribution<double> (deg2rad(-10), deg2rad(10));
      y_distribution = std::uniform_real_distribution<double> (deg2rad(-15), deg2rad(30));
      z_distribution = std::uniform_real_distribution<double> (deg2rad(-10), deg2rad(10));
      duration_distribution = std::uniform_int_distribution<int> (20,40);
    }
    else if(update > curr5)
    {
      x_distribution = std::uniform_real_distribution<double> (deg2rad(-10), deg2rad(10));
      y_distribution = std::uniform_real_distribution<double> (deg2rad(-15), deg2rad(35));
      z_distribution = std::uniform_real_distribution<double> (deg2rad(-10), deg2rad(10));
      duration_distribution = std::uniform_int_distribution<int> (10,30);
      // duration_distribution = std::uniform_int_distribution<int> (,8);
    }   
  }

  void Armcurriculum(int update)
  {
    int curr1 = 500;
    int curr2 = 700;
    int curr3 = 2500;
    int curr4 = 4500;
    int curr5 = 6500;
    int curr6 = 8500; 

    if(update >= curr1 && update <curr2) 
    {
      end_imit_scale = 50.0;
      f_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(10), deg2rad(10));
      f_thigh_distribution = std::uniform_real_distribution<double> (deg2rad(0), deg2rad(85));
      f_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(110), -deg2rad(85));
    }
    else if(update >= curr2 && update <curr3) 
    {
      f_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(15), deg2rad(15));
      f_thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(5), deg2rad(100));
      f_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(120), -deg2rad(75));
    }
    else if(update >= curr3 && update <curr4) 
    {
      f_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(20), deg2rad(20));
      f_thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(15), deg2rad(150));
      f_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(125), -deg2rad(70));
    }
    else if(update >= curr4 && update <curr5) 
    {
      f_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(30), deg2rad(30));
      f_thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(30), deg2rad(150));
      f_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(130), -deg2rad(60));
    }
    else if(update >= curr5 && update <curr6) 
    {
      f_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(40), deg2rad(40));
      f_thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(45), deg2rad(150));
      f_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(140), -deg2rad(55));
    }
    else if(update >= curr6) 
    {
      f_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(45), deg2rad(45));
      f_thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(55), deg2rad(153));
      f_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(150), -deg2rad(50));
    }
  }

  void DRcurriculum(int update)
  {
    int curr1 = 1000;
    int curr2 = 2000;
    int curr3 = 5000;
    int curr4 = 7000;
    int curr5 = 10000;

    // x_distribution = std::uniform_real_distribution<double> (deg2rad(-40), deg2rad(40));
    // y_distribution = std::uniform_real_distribution<double> (deg2rad(-40), deg2rad(40));
    // z_distribution = std::uniform_real_distribution<double> (deg2rad(-40), deg2rad(40));
    // h_distribution = std::uniform_real_distribution<double> (0.22, 0.31);
    // duration_distribution = std::uniform_int_distribution<int> (18,45);

    if(update > curr1 && update <=curr2)
    {
      mass_distribution = std::uniform_real_distribution<double> (0.85, 1.10);
      pgain_distribution = std::uniform_real_distribution<double> (0.85, 1.10);
      dgain_distribution = std::uniform_real_distribution<double> (0.90, 1.10);
      friction_distribution = std::uniform_real_distribution<double> (0.7, 1.3);
      latency_distribution = std::uniform_real_distribution<double> (1.0, 1.2);
    }
    else if(update > curr2 && update <=curr3)
    {
      mass_distribution = std::uniform_real_distribution<double> (0.80, 1.15);
      pgain_distribution = std::uniform_real_distribution<double> (0.80, 1.20);
      dgain_distribution = std::uniform_real_distribution<double> (0.80, 1.20);
      friction_distribution = std::uniform_real_distribution<double> (0.6, 1.4);
      latency_distribution = std::uniform_real_distribution<double> (0.95, 1.3);
    }
    else if(update > curr3 && update <=curr4)
    {
      mass_distribution = std::uniform_real_distribution<double> (0.75, 1.15);
      pgain_distribution = std::uniform_real_distribution<double> (0.75, 1.25);
      dgain_distribution = std::uniform_real_distribution<double> (0.75, 1.25);
      friction_distribution = std::uniform_real_distribution<double> (0.5, 1.5);
      latency_distribution = std::uniform_real_distribution<double> (0.95, 1.4);
    }
    else if(update > curr4&& update <=curr5)
    {
      mass_distribution = std::uniform_real_distribution<double> (0.75, 1.2);
      pgain_distribution = std::uniform_real_distribution<double> (0.75, 1.25);
      dgain_distribution = std::uniform_real_distribution<double> (0.75, 1.25);
      friction_distribution = std::uniform_real_distribution<double> (0.5, 1.5);
      latency_distribution = std::uniform_real_distribution<double> (0.95, 1.5);
    }
    else if(update > curr5)
    {
      mass_distribution = std::uniform_real_distribution<double> (0.75, 1.2);
      pgain_distribution = std::uniform_real_distribution<double> (0.7, 1.3);
      dgain_distribution = std::uniform_real_distribution<double> (0.7, 1.3);
      friction_distribution = std::uniform_real_distribution<double> (0.5, 1.5);
      latency_distribution = std::uniform_real_distribution<double> (0.95, 1.6);
    }
  }

  float CalcReward()
  {   
    float reward = 0;
    double joint_imit_err = 0;
    double root_imit_err = 0;
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
    root_imit_err = rootImitationErr();
    ori_imit_err = orientationImitationErr();
    end_imit_err = endEffectorImitationErr();
    freq_err = CalcAccErr();

    joint_imit_reward = exp(-joint_imit_scale * joint_imit_err);
    root_imit_reward = exp(-root_imit_scale * root_imit_err);
    ori_imit_reward = exp(-ori_imit_scale * ori_imit_err);
    end_imit_reward = exp(-end_imit_scale * end_imit_err);
    freq_reward = exp(- freq_scale * freq_err);

    reward = main_w * joint_imit_reward * ori_imit_reward * root_imit_reward * end_imit_reward + freq_w * freq_reward;
    return reward;
  }

  float jointImitationErr()
  {
    float err = 0.;
    for(int i = 0; i < nJoints_; ++i)
    {
      if(current_count > 100 && i > 12)
        err+= 0.05 * pow((gc_target[4 + i] - gc_[7 + i]),2);
      else
        err += pow((gc_target[4 + i] - gc_[7 + i]),2);
    }
    return err;
  }

  float rootImitationErr()
  {
    float err = 0.;
    err += (rootTraj[current_count] - gc_.head(3)).squaredNorm();
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
    Eigen::VectorXd fixed_pose = gc_;
    fixed_pose.head(3) = rootTraj[current_count];
    fixed_pose.tail(16) = trajectories[current_count];
    ik_dummy->setGeneralizedCoordinate(fixed_pose);
    Eigen::VectorXd follow_pose = gc_;
    follow_pose.tail(4+ nJoints_) = gc_target;
    ref_dummy->setGeneralizedCoordinate(follow_pose);
    for(int i = 0; i < 2; ++i)
    {
      raisim::Vec<3> footPositionA;
      raisim::Vec<3> footPositionP;
      alien_->getFramePosition(FR_FOOT+3 * i, footPositionA);
      ref_dummy->getFramePosition(FR_FOOT+3 * i, footPositionP);
      err += (footPositionP.e() - footPositionA.e()).squaredNorm();
    }
    for(int i = 2; i < 4; ++i)
    {
      raisim::Vec<3> footPositionA;
      raisim::Vec<3> footPositionP;
      alien_->getFramePosition(FR_FOOT+3 * i, footPositionA);
      ik_dummy->getFramePosition(FR_FOOT+3 * i, footPositionP);
      err += (footPositionP.e() - footPositionA.e()).squaredNorm();
    }

    if(isnan(err)) return 1000000;
    return err;
  }

  float CalcAccErr()
  {
    return 20 * joint_acc.squaredNorm();
  }

  void updateExtraInfo() final {
    extraInfo_["base height"] = gc_[2];
  }

  void updateObservation() {
    alien_->getState(gc_, gv_);
    obScaled_.setZero(obDim_);

    if(testStep)
    {
      for(int i = 7; i < gc_.size(); ++i)
      {
        writeActions << std::setprecision(3) << std::fixed << gc_[i] << " ";
      }
      writeActions << std::endl;
    }   

    state_history.erase(state_history.begin());
    state_history.push_back(gc_.tail(16));
    reference_history.erase(reference_history.begin());
    reference_history.push_back(gc_target);

    for(int i = 0; i < 4; ++i)
    {
      obScaled_.segment(16 * i, 16) = reference_history[i];
    }
    for(int i = 0; i < 4; ++i)
    {
      obScaled_.segment(64 + 16 * i, 16) = state_history[i];
    }
    for(int i = 0; i < 3; ++i)
    {
      obScaled_.segment(128 + 12 * i, 12) = action_history[i];
    }
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

    for(auto& contact: alien_->getContacts()){
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end()) {
        return true;
      }
    }

    if(current_count > 70)
    {
      Eigen::Quaterniond target(gc_target[0],gc_target[1],gc_target[2],gc_target[3]);
      Eigen::Quaterniond current(gc_[3],gc_[4],gc_[5],gc_[6]);
      if(current.angularDistance(target) > deg2rad(30)){
        terminalReward = -5;
        return true;
      }
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
  raisim::ArticulatedSystem* ik_dummy;
  raisim::ArticulatedSystem* ref_dummy;
  raisim::ArticulatedSystem* vis_ref;
  std::vector<GraphicObject> * anymalVisual_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, torque_, gc_dummy_init;
  Eigen::VectorXd gc_prev, gc_target, gc_reference;;
  double desired_fps_ = 30.;
  int visualizationCounter_=0;
  float effort_limit = 33.5 * 0.90;
  Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_;
  Eigen::VectorXd obDouble_, obScaled_;
  std::set<size_t> footIndices_;
  std::set<size_t> foot_hipIndices_;
  int max_cnt = 303;

  std::vector<Eigen::VectorXd> action_history;
  std::vector<Eigen::VectorXd> state_history;
  std::vector<Eigen::VectorXd> reference_history;
  std::vector<Eigen::VectorXd> trajectories;
  std::vector<Eigen::Vector3d> rootTraj;

  std::vector<Eigen::VectorXd> init_feet_poses;
  std::ofstream writeActions;

  AnalyticFullIK* mAIK;
  int current_count;
  Eigen::VectorXd prev_vel, joint_acc;
  std::vector<std::deque<double>> torque_sequence;

  // Reward Shapers
  double main_w, freq_w;
  double joint_imit_scale, root_imit_scale, ori_imit_scale, end_imit_scale, freq_scale;
  double joint_imit_reward, root_imit_reward, ori_imit_reward, end_imit_reward, freq_reward;

  Eigen::Vector3d sittingRoot;

  // Kp Settings
  double init_pgain = 100;
  double init_dgain = 2;
  // double init_pgain = 200;
  // double init_dgain = 10;
  Eigen::VectorXd jointPgain, jointDgain;
  bool testStep = false;

  float default_friction_coeff;
  std::vector<double> original_mass;
  // Parameters for learning
  std::uniform_real_distribution<double> x_distribution;
  std::uniform_real_distribution<double> y_distribution;
  std::uniform_real_distribution<double> z_distribution;
  std::uniform_real_distribution<double> ref_noise_distribution;
  std::uniform_real_distribution<double> noise_distribution;
  std::uniform_int_distribution<int> duration_distribution;

  std::uniform_real_distribution<double> f_hip_distribution;
  std::uniform_real_distribution<double> f_thigh_distribution;
  std::uniform_real_distribution<double> f_calf_distribution;

  // Parameters for domain randomization
  std::uniform_real_distribution<double> mass_distribution;
  std::uniform_real_distribution<double> pgain_distribution;
  std::uniform_real_distribution<double> dgain_distribution;
  std::uniform_real_distribution<double> friction_distribution; 
  std::uniform_real_distribution<double> latency_distribution; 

  // For human data
  std::vector<Eigen::VectorXd> human_point_list;
  std::vector<Eigen::VectorXd> raw_trj;
  int human_cnt;
  Eigen::Vector4i currentContact;
  Eigen::Vector4i prevContact;
  Eigen::VectorXd kin_gc_prev;
  float init_foot;

};

} 