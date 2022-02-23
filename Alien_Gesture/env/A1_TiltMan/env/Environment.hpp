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
    // for(int i = 0; i < 81; ++ i)  heightVec.push_back(0.0);
    for(int i = 0; i < 8100; ++ i)  heightVec.push_back(0.0);
    // heightMap = world_->addHeightMap(9, 9, 8.0, 8.0, 0., 0., heightVec, "floor");
    heightMap = world_->addHeightMap(90, 90, 80.0, 80.0, 0., 0., heightVec, "floor");

    world_->setERP(0,0);
    /// get robot data
    gcDim_ = alien_->getGeneralizedCoordinateDim();
    gvDim_ = alien_->getDOF();
    nJoints_ = 12;

    control_dt_ = 0.03333333;
    // READ_YAML(double, control_dt_, cfg["control_dt"])

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    torque_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);
    gc_target.setZero(4 + nJoints_); gc_prev.setZero(gcDim_);

    /// this is nominal configuration of anymal
    gc_init_ <<  0, 0, 0.27, 1.0, 0.0, 0.0, 0.0, 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99);
    alien_->setState(gc_init_, gv_init_);
    ref_dummy->setState(gc_init_, gv_init_);
    vis_ref->setState(gc_init_, gv_init_);
    ik_dummy->setState(gc_init_, gv_init_);

    for(int i = 0; i < 4; ++i)
    {
      raisim::Vec<3> footPositionA;
      alien_->getFramePosition(FR_FOOT+3 * i, footPositionA);
      init_feet_poses.push_back(footPositionA.e());
      // std::cout << footPositionA.e().transpose() << std::endl;
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

    // Have to adjust the observation dimension
    // Have to adjust the action std and mean
    // Have to........ 
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

    joint_imit_scale = 1.0;
    ori_imit_scale = 20.0;
    end_imit_scale = 40.0;
    support_scale = 10.0;
    freq_scale = 1e-4;

    default_friction_coeff = 0.7;
    raisim::MaterialManager materials_;
    materials_.setMaterialPairProp("floor", "robot", default_friction_coeff, 0.0, 0.0);
    world_->updateMaterialProp(materials_);
    alien_->getCollisionBody("FR_foot/0").setMaterial("robot");
    alien_->getCollisionBody("FL_foot/0").setMaterial("robot");
    alien_->getCollisionBody("RL_foot/0").setMaterial("robot");
    alien_->getCollisionBody("RR_foot/0").setMaterial("robot");

    x_distribution = std::uniform_real_distribution<double> (deg2rad(-10), deg2rad(10));
    y_distribution = std::uniform_real_distribution<double> (deg2rad(-10), deg2rad(10));
    z_distribution = std::uniform_real_distribution<double> (deg2rad(-10), deg2rad(10));
    h_distribution = std::uniform_real_distribution<double> (0.26, 0.28);
    duration_distribution = std::uniform_int_distribution<int> (25,35);
    // noise_distribution = std::uniform_real_distribution<double> (deg2rad(-1), deg2rad(1));

    f_hip_distribution = std::uniform_real_distribution<double> (deg2rad(-5),deg2rad(5));
    f_thigh_distribution = std::uniform_real_distribution<double> (deg2rad(80), deg2rad(85));
    f_calf_distribution = std::uniform_real_distribution<double>(deg2rad(-140), deg2rad(-130));
    r_hip_distribution = std::uniform_real_distribution<double> (deg2rad(-5),deg2rad(5));
    r_thigh_distribution = std::uniform_real_distribution<double> (deg2rad(80), deg2rad(85));
    r_calf_distribution = std::uniform_real_distribution<double>(deg2rad(-140), deg2rad(-130));
    // noise_distribution = std::uniform_real_distribution<double> (deg2rad(-1), deg2rad(1));
    contact_distribution = std::uniform_real_distribution<double> (deg2rad(-15), deg2rad(15));
    force_distribution = std::uniform_real_distribution<double> (2, 8);
    axis_distribution = std::uniform_real_distribution<double> (0,1);


    rad_distribution = std::uniform_real_distribution<double>(0.0, 0.0);
    phi_distribution = std::uniform_real_distribution<double>(-M_PI, M_PI);

    change_distribution = std::uniform_int_distribution<int>(30, 30);
    leg_distribution = std::uniform_int_distribution<int>(0, 3);

    mass_distribution = std::uniform_real_distribution<double> (0.95, 1.05);
    pgain_distribution = std::uniform_real_distribution<double> (0.95, 1.05);
    dgain_distribution = std::uniform_real_distribution<double> (0.95, 1.05);
    friction_distribution = std::uniform_real_distribution<double> (0.9, 1.1);
    terrain_distribution = std::uniform_real_distribution<double> (-0.02, 0.02);
    latency_distribution = std::uniform_real_distribution<double> (1.0, 1.0);
    original_mass  = alien_->getMass();

    gui::rewardLogger.init({"Angle_reward", "Ori_reward", "Support_reward", "End_reward", "Freq_reward"});

    /// indices of links that should not make contact with ground
    footIndices_.insert(alien_->getBodyIdx("FL_calf"));
    footIndices_.insert(alien_->getBodyIdx("FR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RL_calf"));

    mAIK = new AnalyticFullIK(ik_dummy);
    rik = new AnalyticRIK(ref_dummy, 0);
    lik = new AnalyticLIK(ref_dummy, 0);
    /// visualize if it is the first environment
    if (visualizable_) {
      auto vis = raisim::OgreVis::get();

      /// these method must be called before initApp
      vis->setWorld(world_.get());
      vis->setWindowSize(1920, 1080);
      vis->setImguiSetupCallback(imguiSetupCallback);
      vis->setImguiRenderCallback(imguiRenderCallBack);
      vis->setKeyboardCallback(raisimKeyboardCallback);
      vis->setSetUpCallback(setupCallback);
      vis->setAntiAliasing(8);
      vis->initApp();

      anymalVisual_ = vis->createGraphicalObject(alien_, "AlienGo");
      auto ref_graphics = vis->createGraphicalObject(vis_ref, "visualization");
      // vis->createGraphicalObject(heightMap,  "floor", "gravel_Height");
      vis->createGraphicalObject(heightMap,  "floor", "checkerboard_mine");;

      desired_fps_ = 1.0/ control_dt_;
      vis->setDesiredFPS(desired_fps_);
      vis->select(anymalVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
      vis->addVisualObject("cop", "sphereMesh", "aqua_marine", {0.02, 0.02, 0.02}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("com", "sphereMesh", "medium_purple", {0.02, 0.02, 0.02}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("dcm", "sphereMesh", "dark_magenta", {0.02, 0.02, 0.02}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      initSupportPolygon();
    }
    std::string filePath1 = "/home/sonic/Project/Alien_Gesture/robot.txt";
    std::string filePath2 = "/home/sonic/Project/Alien_Gesture/ref.txt";
    writeRobot.open(filePath1);
    writeReference.open(filePath2);
  }

  ~ENVIRONMENT() final = default;

  void init() final { }

  void kinect_init() final{
    // Lean_TGcurriculum(3000);
    // Man_TGcurriculum(10000);
    // f_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(30), deg2rad(30));
    // f_thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(60), deg2rad(60));
    // f_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(60), -deg2rad(52));
    // leg_distribution = std::uniform_int_distribution<int>(0, 1);
    // Here, it is only used in the test step
    // Do the initialization of test step like initialize the human 
    // TGcurriculum(5000);
    max_cnt = 450;
    // std::string file_name = "/home/sonic/Project/Alien_Gesture/tilting.txt";
    // writeActions.open(file_name);
    // testStep = true;
    // DRcurriculum(14000);
    // From here, do it
    // readHumanPoints("/home/sonic/Project/Alien_Gesture/KinectRaisimtest/build/rsc/points_leanlong.txt");
    // readHumanPoints("/home/sonic/Project/Alien_Gesture/rsc/map_reach/points_reacht.txt");
    // init_bones();
    // human_cnt = 0;
    currentContact = Eigen::Vector4i(1,1,1,1);
    prevContact = Eigen::Vector4i(1,1,1,1);
    kin_gc_prev = gc_init_;
    init_foot = init_feet_poses[0][2];
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
    Lean_TGcurriculum(16000);
    Man_TGcurriculum(10000);
    leg_distribution = std::uniform_int_distribution<int>(0, 1);
    // DRcurriculum(12000);
    alien_->setState(gc_init_, gv_init_);
    ref_dummy->setState(gc_init_, gv_init_);
    vis_ref->setState(gc_init_, gv_init_);
    ik_dummy->setState(gc_init_, gv_init_);
    current_count = 0;
    test_mode = true;
    gc_reference = gc_init_;
    gc_reference[1] += 0.7;

    prev_vel.setZero();
    joint_acc.setZero();
    gc_prev = gc_init_;
    gc_target = gc_init_.tail(nJoints_ + 4);

    force_exerted.clear();
    force_exerted.shrink_to_fit();
    trajectories.clear();
    trajectories.shrink_to_fit();
    leg_history.clear();
    leg_history.shrink_to_fit();
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
    ori_imit_reward = 0.0;
    end_imit_reward = 0.0;
    support_reward = 0.0;
    freq_reward = 0.0;

    mCOM = Eigen::Vector3d(0, 0, 0.27);
    height1 = mCOM[2];
    height2 = mCOM[2];
    mCOM[2] = 0;
    mCOP = mCOM;
    mDCM = mCOM;

    pTarget_.setZero();
    pTarget_.tail(nJoints_) = gc_init_.tail(nJoints_);
    alien_->setPdTarget(pTarget_, vTarget_);
    // readSeq();
    generateRandomSeq();
    // readReference("/home/sonic/Project/Alien_Gesture/abl_robot.txt");

    randomizeDomain();
    updatedSupportPolygon();
    updateObservation();

    if(visualizable_){
      gui::rewardLogger.clean();
      auto& list = raisim::OgreVis::get()->getVisualObjectList();
      list["com"].setPosition(mCOM);
      list["dcm"].setPosition(mDCM);
      list["cop"].setPosition(mCOP);
      resetPolygon();
    }
  }

  void readSeq()
  {
    std::string ref_poses = "/home/sonic/Project/Alien_Gesture/ref_motion.txt";
    std::ifstream human_input(ref_poses);
    if(!human_input){
      std::cout << "Expected valid input file name" << std::endl;
    }

    int triple_bone = 16;
    while(!human_input.eof()){
      Eigen::VectorXd humanoid(triple_bone);
      for(int i = 0 ; i < triple_bone; ++i)
      {
        float ang;
        human_input >> ang;
        humanoid[i] = ang;
      }
      trajectories.push_back(humanoid);
    }
    trajectories.pop_back();
    human_input.close();
    test_mode = true;
  }
  void updatedSupportPolygon()
  {
    contact_feet.clear();
    contact_feet.shrink_to_fit();
    float contact_thresh = 0.025;
    if(test_mode){
      for(int i = 0; i < 4; ++ i)
      {
        raisim::Vec<3> foot;
        alien_->getFramePosition(FR_FOOT + 3 * foot_order[i], foot);
        if(foot[2] < contact_thresh) 
        {
          contact_feet.push_back(foot_order[i]);
        }
      }
    }
    else{
      for(int i = 0; i < 4; ++ i)
      {
        if(leg_history[current_count] != foot_order[i]) 
        {
          raisim::Vec<3> foot;
          alien_->getFramePosition(FR_FOOT + 3 * foot_order[i], foot);
          if(foot[2] < contact_thresh) 
          {
            contact_feet.push_back(foot_order[i]);
          }
        }
      }
    }    
  }

  void initSupportPolygon()
  {
    auto vis = raisim::OgreVis::get();      
    vis->addVisualObject("supportPoly0", "cylinderMesh", "red", {0.005, 0.005, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
    vis->addVisualObject("supportPoly1", "cylinderMesh", "red", {0.005, 0.005, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
    vis->addVisualObject("supportPoly2", "cylinderMesh", "red", {0.005, 0.005, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
    vis->addVisualObject("supportPoly3", "cylinderMesh", "red", {0.005, 0.005, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);    
  }

  void visualizeSupportPolygon()
  {
    auto& list = raisim::OgreVis::get()->getVisualObjectList();
    int cnt_feet = contact_feet.size();
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
  }

  void resetPolygon()
  {
    auto& list = raisim::OgreVis::get()->getVisualObjectList();
    raisim::Mat<3,3> rot;
    rot.setIdentity();
    for(int i = 0; i < 4; ++i){
      list["supportPoly"+ std::to_string(i)].setPosition(Eigen::Vector3d(0,0,0));
      list["supportPoly"+ std::to_string(i)].setScale(Eigen::Vector3d(0.005, 0.005, 0.005));
      list["supportPoly"+ std::to_string(i)].setOrientation(rot);
    }
  }

  // COM DCM is updated in control frequency step domain
  void updateCOMDCM()
  {
    auto& list = raisim::OgreVis::get()->getVisualObjectList();
    Eigen::Vector3d prevCOM = mCOM;
    mCOM = alien_->getCompositeCOM().e();
    mCOM[2] = 0;
    float current_height = alien_->getCompositeCOM()[2];
    float z_acc = (current_height - 2 * height1 + height2) / control_dt_ * control_dt_;
    float natural_freq = std::sqrt((9.8 + z_acc) / current_height);
    mDCM = mCOM + ((mCOM - prevCOM) / control_dt_) / natural_freq;
    height1 = current_height;
    height2 = height1;
    list["com"].setPosition(mCOM);
    list["dcm"].setPosition(mDCM);
  }

  void updateCOP(){
    auto& list = raisim::OgreVis::get()->getVisualObjectList();
    Eigen::Vector3d COP;
    COP.setZero();
    float tot_mag = 0.0;

    for (auto &contact: alien_->getContacts()){
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
        continue;
      if(!test_mode){
        if(contact.getlocalBodyIndex() == 3 * (leg_history[current_count] + 1))
          continue;
      }
      auto current = contact.getPosition();
      auto zaxis = *contact.getImpulse();
      tot_mag += zaxis[2];
      COP[0] += current[0] * zaxis[2] - current[2] * zaxis[1];
      COP[1] += current[1] * zaxis[2] - current[2] * zaxis[1];
    }
    if(tot_mag == 0.0){
      mCOP = mCOM;
      return;
    }
    COP /= tot_mag;
    mCOP = COP;
    list["cop"].setPosition(mCOP);
  }
  void setVisRef()
  {
    gc_reference.tail(16) = gc_target;
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

  void generateRandomSeq()
  {
    // 
    int cnt = 0; 
    Eigen::Quaterniond prev_quat(gc_init_[3],gc_init_[4],gc_init_[5],gc_init_[6]);
    Eigen::VectorXd prev_pose = gc_init_.tail(12);
    std::random_device rd;
    std::mt19937 generator(rd());
    int inbetween = duration_distribution(generator);
    motionBlender(prev_quat, prev_quat, prev_pose, prev_pose, inbetween);
    cnt+= inbetween;
    while(cnt < max_cnt)
    {
      inbetween = duration_distribution(generator);
      float x_angle = x_distribution(generator);
      float y_angle = y_distribution(generator);
      float z_angle = z_distribution(generator);
      float target_h = h_distribution(generator);
      Eigen::Matrix3d m;
      m = Eigen::AngleAxisd(x_angle, Eigen::Vector3d::UnitX())
        * Eigen::AngleAxisd(y_angle,  Eigen::Vector3d::UnitY())
        * Eigen::AngleAxisd(z_angle, Eigen::Vector3d::UnitZ());

      Eigen::Quaterniond current_quat(m);
      Eigen::VectorXd raw_pose(gcDim_);
      raw_pose.head(2) = gc_init_.head(2);
      raw_pose[2] = target_h;
      raw_pose[3] = current_quat.w(); raw_pose[4] = current_quat.x();
      raw_pose[5] = current_quat.y(); raw_pose[6] = current_quat.z();
      raw_pose.tail(12) = gc_init_.tail(12);
      mAIK->setCurrentPose(raw_pose);
      mAIK->setReady();  
      for(int i = 0 ; i < 4; ++i){
        mAIK->appendTargets(init_feet_poses[i]);
      }
      mAIK->solve(raw_pose);
      Eigen::VectorXd current_pose = mAIK->getSolution();
      mAIK->clearTargets();
      motionBlender(prev_quat, current_quat, prev_pose, current_pose, inbetween);
      prev_quat = current_quat;
      prev_pose = current_pose;
      cnt += inbetween;
    } 
    inbetween = duration_distribution(generator);
    cnt = inbetween;
    int prev_leg = -1;
    for(int i = 0; i < inbetween + 1; ++i)
      leg_history.push_back(prev_leg);

    std::vector<int> idx;
    while(cnt < max_cnt)
    {
      int current_leg = leg_distribution(generator);
      Eigen::VectorXd prev_fin = gc_init_;
      
      if(prev_leg != current_leg && prev_leg != -1)
      {
        // Land the prev leg for the next movement and then increase the count
        int land_between = change_distribution(generator);
        int end = std::min(cnt + land_between -1, int(trajectories.size())-1);
        prev_fin.tail(16) = trajectories[end];
        Eigen::Vector3d now(trajectories[cnt][3 * prev_leg + 4], trajectories[cnt][3 * prev_leg + 5], trajectories[cnt][3 * prev_leg + 6]);
        // Define the target
        float phi = phi_distribution(generator);
        float rad = rad_distribution(generator);
        ref_dummy->setGeneralizedCoordinate(gc_init_);
        raisim::Vec<3> original_landing;
        ref_dummy->getFramePosition(FR_FOOT+ 3 * prev_leg, original_landing);
        Eigen::Vector3d target_land = original_landing.e() + Eigen::Vector3d(rad * cos(phi), rad * sin(phi),0);
        Eigen::Vector3d sol;
        if(prev_leg % 2 == 0)
        {
          rik->setHipIdx(prev_leg/2);
          rik->setReady();
          rik->setCurrentPose(prev_fin);
          rik->setTarget(target_land);
          rik->solveIK();
          sol = rik->getSolution();
          rik->reset();
        }
        else
        {
          lik->setHipIdx(prev_leg/2);
          lik->setReady();
          lik->setCurrentPose(prev_fin);
          lik->setTarget(target_land);
          lik->solveIK();
          sol = lik->getSolution();
          lik->reset();
        }
        idx.push_back(3 * prev_leg + 0);
        idx.push_back(3 * prev_leg + 1);
        idx.push_back(3 * prev_leg + 2);
        partial_blender(idx, now, sol, cnt, cnt+land_between);
        cnt += land_between;
        for(int i = 0; i < land_between; ++i)
          leg_history.push_back(prev_leg);
        // Here for stability
        partial_blender(idx, sol, sol, cnt, cnt+10);
        cnt+= 10;
        for(int i = 0; i < 10; ++i)
          leg_history.push_back(-1);
        idx.clear();
        idx.shrink_to_fit();
      }
      if(cnt >= trajectories.size()-1) break;
      int inbetween = change_distribution(generator);
      Eigen::Vector3d now(trajectories[cnt][3 * current_leg + 4], trajectories[cnt][3 * current_leg + 5], trajectories[cnt][3 * current_leg + 6]);
     
      // Here distinguish rear and front leg
      Eigen::Vector3d pose;
      if(current_leg < 2)
      {
        pose[0] = f_hip_distribution(generator);
        pose[1] = f_thigh_distribution(generator);
        pose[2] = f_calf_distribution(generator);
      }
      else
      {
        pose[0] = r_hip_distribution(generator);
        pose[1] = r_thigh_distribution(generator);
        pose[2] = r_calf_distribution(generator);
      }
      idx.push_back(3 * current_leg + 0);
      idx.push_back(3 * current_leg + 1);
      idx.push_back(3 * current_leg + 2);
      partial_blender(idx, now, pose, cnt, cnt+inbetween);
      cnt += inbetween;
      prev_leg = current_leg;
      for(int i = 0; i < inbetween; ++i)
        leg_history.push_back(current_leg);

      int inbetween2 = change_distribution(generator) / 4;
      partial_blender(idx, pose, pose, cnt, cnt+inbetween2);
      cnt+= inbetween2;
      for(int i = 0; i < inbetween2; ++i)
        leg_history.push_back(current_leg);
      idx.clear();
      idx.shrink_to_fit();
    }
    // cnt = 0;
    // while(cnt < max_cnt)
    // {
    //   int duration = change_distribution(generator);
    //   float force = force_distribution(generator);
    //   int exert_flag = leg_distribution(generator)%2;

    //   Eigen::Vector3d way = force * exert_flag * 
    //     Eigen::Vector3d(axis_distribution(generator),axis_distribution(generator),axis_distribution(generator)).normalized();
    //   raisim::Vec<3> direction = {way[0], way[1], way[2]};
    //   for(int i = 0; i < duration; ++i){
    //     force_exerted.push_back(direction);
    //   }
    //   cnt += duration;
    // }
  }

  void motionBlender(Eigen::Quaterniond q1, Eigen::Quaterniond q2, 
    Eigen::VectorXd now, Eigen::VectorXd next, int inbetween)
  {
    for(int j = 0 ; j < inbetween ; ++j)
    {
      std::random_device rd;
      std::mt19937 generator(rd());
      Eigen::VectorXd noise(12);
      // for(int k =0; k < 12; ++k) noise[k] = contact_distribution(generator);
      Eigen::Quaterniond slerped = q1.slerp(float(j) / float(inbetween), q2);
      Eigen::VectorXd interpol = (now *(inbetween - j) / inbetween) + (next *j / inbetween);
      Eigen::VectorXd total(4 + nJoints_);
      slerped.normalize();
      total[0] = slerped.w();
      total[1] = slerped.x();
      total[2] = slerped.y();
      total[3] = slerped.z();

      total.tail(nJoints_) = interpol;
      trajectories.push_back(total);
    }
  }

  void partial_blender(std::vector<int> idx, Eigen::Vector3d now, Eigen::Vector3d next, int start, int end)
  {
    int inbetween = end - start + 1;
    for(int i = 1 ; i < inbetween + 1; ++i)
    {
      if(start + i >= trajectories.size()-1) break;
      Eigen::VectorXd current = trajectories[start + i];
      for(int j = 0; j < idx.size(); ++j)
      {
        std::random_device rd;
        std::mt19937 generator(rd());
        float noise = 0;
        current[idx[j] + 4] =  (now[j] *(inbetween - i) / inbetween) + (next[j] *i / inbetween) + noise;
      }
      trajectories[start + i] = current;
    }
  }

  void readReference(std::string human_points)
  { 
    // Reader ==> Save as vector
    std::ifstream human_input(human_points);
    if(!human_input){
      std::cout << "Expected valid input file name" << std::endl;
    }

    int triple_bone = 19;
    while(!human_input.eof()){
      Eigen::VectorXd humanoid(19);
      for(int i = 0 ; i < 19; ++i)
      {
        float ang;
        human_input >> ang;
        humanoid[i] = ang;
      }
      trajectories.push_back(humanoid.tail(16));
    }
    std::cout <<"Read" << std::endl;
    std::cout << trajectories[1].transpose()<< std::endl;
    // human_point_list.pop_back();
    human_input.close();
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
    // double h1 = terrain_distribution(generator);
    // double h2 = terrain_distribution(generator);
    // double h3 = terrain_distribution(generator);
    // double h4 = terrain_distribution(generator);
    // heightVec[30] = h1;
    // heightVec[31] = (h1 + h2)/2.;
    // heightVec[32] = h2;
    // heightVec[39] = (h1 + h3)/2.;
    // heightVec[41] = (h2 + h4)/2.;
    // heightVec[48] = h3;
    // heightVec[49] = (h3 + h4)/2.;
    // heightVec[50] = h4;
    
    // if(visualizable_)
    // {
    //   std::cout << "start remove" << std::endl;
    //   raisim::OgreVis::get()->remove(heightMap);
    //   std::cout << "end remove" << std::endl;
    // }
    // world_->removeObject(heightMap);
    // heightMap = world_->addHeightMap(3, 3, 8.0, 8.0, 0., 0., heightVec, "floor");
    // if(visualizable_)
    // {
    //   auto vis = raisim::OgreVis::get(); 
    //   vis->createGraphicalObject(heightMap, "floor", "gravel_Height");
    // }

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
    // Eigen::VectorXd dampingCoefficient;
    // damping_distribution(generator);
    // alien_-> setJointDamping
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
    float y_shift = 0.15;
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
      alien_->setExternalForce(0, raisim::Vec<3>{0,0,0}, force_exerted[current_count]);
      world_->integrate();  
      if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
      {
        auto vis = raisim::OgreVis::get();
        vis->renderOneFrame();
      }      
      visualizationCounter_++;
    }
    alien_->getState(gc_, gv_);
    updatedSupportPolygon();
    updateCOMDCM();
    updateCOP();
    updateObservation();
    float totalReward = CalcReward();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("Angle_reward", joint_imit_reward);
      gui::rewardLogger.log("Ori_reward", ori_imit_reward);
      gui::rewardLogger.log("Support_reward", support_reward);
      gui::rewardLogger.log("End_reward", end_imit_reward);
      gui::rewardLogger.log("Freq_reward", freq_reward);
      auto vis = raisim::OgreVis::get();
      visualizeSupportPolygon();
      vis->select(anymalVisual_->at(0), false);
      // vis->getCameraMan()->setYawPitchDist(Ogre::Radian(-0.40), Ogre::Radian(-0.6), 4, true);
    }
    current_count++;
    return totalReward;
  }

  float stepTest(const Eigen::Ref<EigenVec>& action) final {

    // initSupportPolygon();

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
      // visualizeBones();
      if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
      {
        auto vis = raisim::OgreVis::get();
        vis->renderOneFrame();
      }      
      visualizationCounter_++;
    }
    alien_->getState(gc_, gv_);
    for(int i = 0; i < gc_.size()-1;++i)
    {
      writeRobot << std::setprecision(3) << std::fixed << gc_[i] << " ";
      writeReference << std::setprecision(3) << std::fixed << gc_reference[i] << " ";
    }
    writeRobot << std::setprecision(3) << std::fixed << gc_[gc_.size()-1];
    writeReference << std::setprecision(3) << std::fixed << gc_reference[gc_.size()-1];
    writeRobot << std::endl;
    writeReference << std::endl;
    updatedSupportPolygon();
    updateCOMDCM();
    updateCOP();
    visualizeSupportPolygon();
    updateObservation();
    // float totalReward = CalcReward();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("Angle_reward", joint_imit_reward);
      gui::rewardLogger.log("Ori_reward", ori_imit_reward);
      gui::rewardLogger.log("Support_reward", support_reward);
      gui::rewardLogger.log("End_reward", end_imit_reward);
      gui::rewardLogger.log("Freq_reward", freq_reward);
      // auto vis = raisim::OgreVis::get();
      // vis->select(anymalVisual_->at(0), false);
      // vis->getCameraMan()->setYawPitchDist(Ogre::Radian(-0.40), Ogre::Radian(-0.6), 4, true);
    }
    current_count++;
    human_cnt++;

    if(!mStopFlag) 
      return 0;
    else{
      // if (visualizable_) resetPolygon();
      return -1E10;
    }
  }

  void curriculumUpdate(int update) final {
    Lean_TGcurriculum(10000);
    Man_TGcurriculum(10000);
    // f_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(30), deg2rad(30));
    // f_thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(60), deg2rad(0));
    // f_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(60), -deg2rad(52));
    leg_distribution = std::uniform_int_distribution<int>(0, 1);
    DRcurriculum(update);
  } 

  void Lean_TGcurriculum(int update)
  {
    int curr1 = 2000;
    int curr2 = 3000;
    int curr3 = 4000;
    int curr4 = 7000;
    int curr5 = 10000;

    if(update > curr1 && update <=curr2)
    {
      x_distribution = std::uniform_real_distribution<double> (deg2rad(-15), deg2rad(15));
      y_distribution = std::uniform_real_distribution<double> (deg2rad(-15), deg2rad(15));
      z_distribution = std::uniform_real_distribution<double> (deg2rad(-15), deg2rad(15));
      h_distribution = std::uniform_real_distribution<double> (0.255, 0.285);
      duration_distribution = std::uniform_int_distribution<int> (25,35);
    }
    else if(update > curr2 && update <=curr3)
    {
      x_distribution = std::uniform_real_distribution<double> (deg2rad(-25), deg2rad(25));
      y_distribution = std::uniform_real_distribution<double> (deg2rad(-25), deg2rad(25));
      z_distribution = std::uniform_real_distribution<double> (deg2rad(-25), deg2rad(25));
      h_distribution = std::uniform_real_distribution<double> (0.25, 0.29);
      duration_distribution = std::uniform_int_distribution<int> (20,40);
    }
    else if(update > curr3 && update <=curr4)
    {
      x_distribution = std::uniform_real_distribution<double> (deg2rad(-30), deg2rad(30));
      y_distribution = std::uniform_real_distribution<double> (deg2rad(-30), deg2rad(30));
      z_distribution = std::uniform_real_distribution<double> (deg2rad(-30), deg2rad(30));
      h_distribution = std::uniform_real_distribution<double> (0.23, 0.30);
      duration_distribution = std::uniform_int_distribution<int> (20,40);
    }
    else if(update > curr4&& update <=curr5)
    {
      x_distribution = std::uniform_real_distribution<double> (deg2rad(-40), deg2rad(40));
      y_distribution = std::uniform_real_distribution<double> (deg2rad(-40), deg2rad(40));
      z_distribution = std::uniform_real_distribution<double> (deg2rad(-40), deg2rad(40));
      h_distribution = std::uniform_real_distribution<double> (0.23, 0.31);
      duration_distribution = std::uniform_int_distribution<int> (20,40);
    }
    else if(update > curr5)
    {
      x_distribution = std::uniform_real_distribution<double> (deg2rad(-40), deg2rad(40));
      y_distribution = std::uniform_real_distribution<double> (deg2rad(-40), deg2rad(40));
      z_distribution = std::uniform_real_distribution<double> (deg2rad(-40), deg2rad(40));
      h_distribution = std::uniform_real_distribution<double> (0.22, 0.31);
      duration_distribution = std::uniform_int_distribution<int> (20,45);
    }   
  }

  void Man_TGcurriculum(int update)
  {
    int curr1 = 500;
    int curr2 = 700;
    int curr3 = 2500;
    int curr4 = 4500;
    int curr5 = 6500;
    int curr6 = 8500; 

    if(update >= curr1 && update <curr2) 
    {
      f_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(10), deg2rad(10));
      f_thigh_distribution = std::uniform_real_distribution<double> (deg2rad(0), deg2rad(85));
      f_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(110), -deg2rad(85));
      r_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(5), deg2rad(5));
      r_thigh_distribution = std::uniform_real_distribution<double> (deg2rad(30), deg2rad(60));
      r_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(110), -deg2rad(100));
      noise_distribution = std::uniform_real_distribution<double> (deg2rad(-1), deg2rad(1));
      rad_distribution = std::uniform_real_distribution<double>(0.0, 0.0);
    }
    else if(update >= curr2 && update <curr3) 
    {
      f_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(15), deg2rad(15));
      f_thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(5), deg2rad(100));
      f_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(120), -deg2rad(75));
      r_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(10), deg2rad(10));
      r_thigh_distribution = std::uniform_real_distribution<double> (deg2rad(10), deg2rad(70));
      r_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(115), -deg2rad(100));
      noise_distribution = std::uniform_real_distribution<double> (deg2rad(-1), deg2rad(1));
      change_distribution = std::uniform_int_distribution<int>(20, 30);
      rad_distribution = std::uniform_real_distribution<double>(0.0, 0.005);
    }
    else if(update >= curr3 && update <curr4) 
    {
      f_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(20), deg2rad(20));
      f_thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(15), deg2rad(150));
      f_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(125), -deg2rad(70));
      r_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(12), deg2rad(12));
      r_thigh_distribution = std::uniform_real_distribution<double> (deg2rad(-5), deg2rad(80));
      r_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(120), -deg2rad(100));
      change_distribution = std::uniform_int_distribution<int>(20, 40);
      noise_distribution = std::uniform_real_distribution<double> (deg2rad(-1.5), deg2rad(1.5));
      rad_distribution = std::uniform_real_distribution<double>(0.0, 0.01);
    }
    else if(update >= curr4 && update <curr5) 
    {
      f_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(30), deg2rad(30));
      f_thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(30), deg2rad(150));
      f_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(130), -deg2rad(60));
      r_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(15), deg2rad(15));
      r_thigh_distribution = std::uniform_real_distribution<double> (deg2rad(-10), deg2rad(90));
      r_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(125), -deg2rad(100));
      noise_distribution = std::uniform_real_distribution<double> (deg2rad(-2.), deg2rad(2.));
      change_distribution = std::uniform_int_distribution<int>(15, 45);
      rad_distribution = std::uniform_real_distribution<double>(0.0, 0.015);
      support_scale = 100.0;
      
    }
    else if(update >= curr5 && update <curr6) 
    {
      f_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(40), deg2rad(40));
      f_thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(45), deg2rad(150));
      f_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(140), -deg2rad(55));
      r_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(18), deg2rad(18));
      r_thigh_distribution = std::uniform_real_distribution<double> (deg2rad(-15), deg2rad(100));
      r_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(130), -deg2rad(100));
      // noise_distribution = std::uniform_real_distribution<double> (deg2rad(-2.5), deg2rad(2.5));
      rad_distribution = std::uniform_real_distribution<double>(0.0, 0.02);
      
      support_scale = 100.0;
      end_imit_scale = 40.0;
    }
    else if(update >= curr6) 
    {
      f_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(45), deg2rad(45));
      f_thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(60), deg2rad(30));
      f_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(150), -deg2rad(50));
      r_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(30), deg2rad(30));
      r_thigh_distribution = std::uniform_real_distribution<double> (deg2rad(-15), deg2rad(120));
      r_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(130), -deg2rad(100));
      // noise_distribution = std::uniform_real_distribution<double> (deg2rad(-3), deg2rad(3));
      change_distribution = std::uniform_int_distribution<int>(20, 40);
      rad_distribution = std::uniform_real_distribution<double>(0.0, 0.02);
      
      support_scale = 100.0;
      end_imit_scale = 40.0;
    }
  }

  void DRcurriculum(int update)
  {
    int curr1 = 1000;
    int curr2 = 2000;
    int curr3 = 5000;
    int curr4 = 7000;
    int curr5 = 20500;

    x_distribution = std::uniform_real_distribution<double> (deg2rad(-40), deg2rad(40));
    y_distribution = std::uniform_real_distribution<double> (deg2rad(-40), deg2rad(40));
    z_distribution = std::uniform_real_distribution<double> (deg2rad(-40), deg2rad(40));
    h_distribution = std::uniform_real_distribution<double> (0.22, 0.31);
    duration_distribution = std::uniform_int_distribution<int> (18,45);

    if(update > curr1 && update <=curr2)
    {
      mass_distribution = std::uniform_real_distribution<double> (0.85, 1.10);
      pgain_distribution = std::uniform_real_distribution<double> (0.85, 1.10);
      dgain_distribution = std::uniform_real_distribution<double> (0.90, 1.10);
      friction_distribution = std::uniform_real_distribution<double> (0.7, 1.3);

    }
    else if(update > curr2 && update <=curr3)
    {
      mass_distribution = std::uniform_real_distribution<double> (0.80, 1.15);
      pgain_distribution = std::uniform_real_distribution<double> (0.80, 1.20);
      dgain_distribution = std::uniform_real_distribution<double> (0.80, 1.20);
      friction_distribution = std::uniform_real_distribution<double> (0.6, 1.4);

    }
    else if(update > curr3 && update <=curr4)
    {
      mass_distribution = std::uniform_real_distribution<double> (0.75, 1.15);
      pgain_distribution = std::uniform_real_distribution<double> (0.75, 1.25);
      dgain_distribution = std::uniform_real_distribution<double> (0.75, 1.25);
      friction_distribution = std::uniform_real_distribution<double> (0.5, 1.5);
    }
    else if(update > curr4&& update <=curr5)
    {
      mass_distribution = std::uniform_real_distribution<double> (0.75, 1.2);
      pgain_distribution = std::uniform_real_distribution<double> (0.75, 1.25);
      dgain_distribution = std::uniform_real_distribution<double> (0.75, 1.25);
    }
    else if(update > curr5)
    {
      mass_distribution = std::uniform_real_distribution<double> (0.75, 1.2);
      pgain_distribution = std::uniform_real_distribution<double> (0.7, 1.3);
      dgain_distribution = std::uniform_real_distribution<double> (0.7, 1.3);
    }
  }

  float CalcReward()
  {   
    float reward = 0;
    double joint_imit_err = 0;
    double ori_imit_err = 0;
    double end_imit_err = 0;
    double support_err = 0;
    double freq_err = 0;

    alien_->getState(gc_, gv_);
    if(isnan(gc_[0]) || isnan(gv_[8]))
    {
      return 0;
    }

    joint_acc = (gv_.tail(nJoints_) - prev_vel) / control_dt_;
    prev_vel = gv_.tail(nJoints_);

    joint_imit_err = jointImitationErr();
    ori_imit_err = rootErr();
    end_imit_err = endEffectorImitationErr();
    support_err = CalcSupportErr();
    freq_err = CalcAccErr();

    joint_imit_reward = exp(-joint_imit_scale * joint_imit_err);
    ori_imit_reward = exp(-ori_imit_scale * ori_imit_err);
    end_imit_reward = exp(-end_imit_scale * end_imit_err);
    support_reward = exp(-support_scale * support_err);
    freq_reward = exp(- freq_scale * freq_err);

    reward = main_w * joint_imit_reward * ori_imit_reward * end_imit_reward * support_reward  
      + freq_w * freq_reward;
    if(isnan(reward)) return 0.0;
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

  float rootErr()
  {
    float err = 0;
    err += 10 * orientationImitationErr();
    err += 1 * pow((gc_reference[0] - gc_[0]),2);
    err += 1 * pow((gc_reference[1] - 0.7 - gc_[1]),2);
    err += 0.4 * pow((gc_reference[2] - gc_[2]),2);
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
    float err = 0.;
    Eigen::VectorXd gc_extended = gc_;
    gc_extended.tail(4 + nJoints_) = gc_target;
    ref_dummy->setGeneralizedCoordinate(gc_extended);
    Eigen::VectorXd fixed_pose = gc_init_;
    fixed_pose.tail(4+ nJoints_) = gc_target;
    ik_dummy->setGeneralizedCoordinate(fixed_pose);
    for(int i = 0; i < 4; ++i)
    {
      raisim::Vec<3> footPositionA;
      raisim::Vec<3> footPositionR;
      raisim::Vec<3> footPositionP;

      alien_->getFramePosition(FR_FOOT+3 * i, footPositionA);
      ref_dummy->getFramePosition(FR_FOOT+3 * i, footPositionR);
      ik_dummy->getFramePosition(FR_FOOT+3 * i, footPositionP);
      
      if(i == leg_history[current_count])
      {
        err += (footPositionR.e() - footPositionA.e()).squaredNorm();
      }
      else
      {
        err += pow((footPositionP[2] - footPositionA[2]) ,2);
      }
    }
    if(isnan(err)) return 1000000;
    return err;
  }

  float CalcSupportErr()
  {
    float min_d = 100000;
    float error = 0.0;

    if(contact_feet.size() == 0) return 100000;
    if(contact_feet.size() == 1) 
    {
      if(isInside(mCOM)) error += 0.;
      else{
        raisim::Vec<3> foot1;
        alien_->getFramePosition(FR_FOOT + 3 * contact_feet[0], foot1);
        foot1[2] = mCOM[2];
        error += (foot1.e() - mCOM).norm();
      }
      if(isInside(mDCM)) error += 0.;
      else{
        raisim::Vec<3> foot1;
        alien_->getFramePosition(FR_FOOT + 3 * contact_feet[0], foot1);
        foot1[2] = mDCM[2];
        error += (foot1.e() - mDCM).norm();
      }
      if(isInside(mCOP)) error += 0.;
      else{
        raisim::Vec<3> foot1;
        alien_->getFramePosition(FR_FOOT + 3 * contact_feet[0], foot1);
        foot1[2] = mCOP[2];
        error += (foot1.e() - mCOP).norm();
      }
    }

    min_d = 100000;
    // COM 
    if(isInside(mCOM)) error += 0.;
    else{
      for(int i = 0; i < contact_feet.size()-1; ++i)
      {
        int j = (i+1) % contact_feet.size();
        raisim::Vec<3> foot1;
        alien_->getFramePosition(FR_FOOT + 3 * contact_feet[i], foot1);
        raisim::Vec<3> foot2;
        alien_->getFramePosition(FR_FOOT + 3 * contact_feet[j], foot2);
        Eigen::Vector3d current1 = poly_scale * foot1.e() + (1 - poly_scale) * supportCentre;
        Eigen::Vector3d current2 = poly_scale * foot2.e() + (1 - poly_scale) * supportCentre;
        float current = distToSegment(mCOM, current1, current2);
        float temp = distToSegment(mCOM, foot1.e(), foot2.e());
        if(min_d > current) min_d = current;
      }
      error += min_d * min_d;
    } 
    min_d = 100000;
    if(isInside(mDCM)) error += 0.;
    else{
      for(int i = 0; i < contact_feet.size(); ++i)
      {
        int j = (i+1) % contact_feet.size();
        raisim::Vec<3> foot1;
        alien_->getFramePosition(FR_FOOT + 3 * contact_feet[i], foot1);
        raisim::Vec<3> foot2;
        alien_->getFramePosition(FR_FOOT + 3 * contact_feet[j], foot2);
        Eigen::Vector3d current1 = poly_scale * foot1.e() + (1 - poly_scale) * supportCentre;
        Eigen::Vector3d current2 = poly_scale * foot2.e() + (1 - poly_scale) * supportCentre;
        float current = distToSegment(mCOM, current1, current2);
        if(min_d > current) min_d = current;
      }
      error += min_d * min_d;
    } 
    min_d = 100000;
    if(isInside(mCOP)) error += 0.;
    else{
      for(int i = 0; i < contact_feet.size(); ++i)
      {
        int j = (i+1) % contact_feet.size();
        raisim::Vec<3> foot1;
        alien_->getFramePosition(FR_FOOT + 3 * contact_feet[i], foot1);
        raisim::Vec<3> foot2;
        alien_->getFramePosition(FR_FOOT + 3 * contact_feet[j], foot2);
        Eigen::Vector3d current1 = poly_scale * foot1.e() + (1 - poly_scale) * supportCentre;
        Eigen::Vector3d current2 = poly_scale * foot2.e() + (1 - poly_scale) * supportCentre;
        float current = distToSegment(mCOM, current1, current2);
        if(min_d > current) min_d = current;
      }
      error += min_d * min_d;
    }  
    if(isnan(error)) return 100000;
    if(contact_feet.size() == 2) 
      error *= 20;
    return error;
  }

  float CalcSupportHardErr()
  {
    float min_d = 100000;
    float error = 0.0;
    if(contact_feet.size() == 0) return 100000;
    Eigen::Vector3d centre = Eigen::Vector3d::Zero();
    int contact_cnt = contact_feet.size();
    for(int i = 0; i < contact_cnt ; ++i)
    {
      raisim::Vec<3> foot;
      alien_->getFramePosition(FR_FOOT + 3 * contact_feet[i], foot);
      centre += foot.e();
    }
    centre /= contact_cnt;
    error += (mCOM - centre).norm();
    error += (mDCM - centre).norm();
    error += (mCOP - centre).norm();

    if(isnan(error)) return 100000;
    return error;
  }

  bool isInside(Eigen::Vector3d point)
  {
    int nContact = contact_feet.size();
    float foot_radius = 0.01;

    std::vector<Eigen::Vector3d> poly;
    if (nContact == 0) return false;
    else if(nContact == 1)
    {
      raisim::Vec<3> foot1;
      alien_->getFramePosition(FR_FOOT + 3 * contact_feet[0], foot1);
      float dist = (point - foot1.e()).norm();
      return dist <= foot_radius;
    }
    else if(nContact == 2)
    { 
      raisim::Vec<3> foot1;
      alien_->getFramePosition(FR_FOOT + 3 * contact_feet[0], foot1);
      raisim::Vec<3> foot2;
      alien_->getFramePosition(FR_FOOT + 3 * contact_feet[1], foot2);
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
        alien_->getFramePosition(FR_FOOT + 3 * contact_feet[i], foot);
        supportCentre += foot.e();
      }
      supportCentre /= contact_feet.size();
      for(int i = 0; i < contact_feet.size(); ++i)
      {
        raisim::Vec<3> foot;
        alien_->getFramePosition(FR_FOOT + 3 * contact_feet[i], foot);
        Eigen::Vector3d current = poly_scale * foot.e() + (1 - poly_scale) * supportCentre;
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

    for(auto& contact: alien_->getContacts())
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end()) {
        // if(visualizable_){
        //   std::cout << "almost ended" << std::endl;
        //   return false;
        // }
        return true;
      }
    for(int i =0 ; i < gc_.size();++i)
    {
      if(isnan(gc_[i])) {
        if(visualizable_){
          return false;
        }
        return true;
      }
    }
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
  raisim::HeightMap* heightMap;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, torque_, gc_dummy_init;
  Eigen::VectorXd gc_prev, gc_target, gc_reference;
  double desired_fps_ = 30.;
  int visualizationCounter_=0;
  float effort_limit = 33.5 * 0.90;
  Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_;
  Eigen::VectorXd obDouble_, obScaled_;
  std::set<size_t> footIndices_;
  std::set<size_t> foot_hipIndices_;
  Eigen::Vector3d mCOP, mCOM, mDCM;

  std::vector<Eigen::VectorXd> action_history;
  std::vector<Eigen::VectorXd> state_history;
  std::vector<Eigen::VectorXd> reference_history;
  std::vector<Eigen::VectorXd> trajectories;

  std::vector<Eigen::VectorXd> init_feet_poses;
  std::ofstream writeActions;

  AnalyticFullIK* mAIK;
  AnalyticRIK* rik;
  AnalyticLIK* lik;
  
  Eigen::VectorXd prev_vel, joint_acc;
  std::vector<std::deque<double>> torque_sequence;

  // Reward Shapers
  double main_w, freq_w;
  double joint_imit_scale, ori_imit_scale, end_imit_scale,support_scale, freq_scale;
  double joint_imit_reward, ori_imit_reward, end_imit_reward,support_reward, freq_reward;

  // Kp Settings
  double init_pgain = 100;
  double init_dgain = 2;
  Eigen::VectorXd jointPgain, jointDgain;
  bool testStep = false;

  float default_friction_coeff;
  std::vector<double> original_mass;

  float height1, height2;
  std::vector<int> contact_feet;
  int foot_order[4] = {0,1,3,2};
  int current_count;

  Eigen::Vector3d supportCentre;
  float poly_scale = 0.45;  
  std::vector<double> heightVec;
  int max_cnt = 452;
  std::vector<int> leg_history;
  std::vector<raisim::Vec<3>> force_exerted;

  bool test_mode = false;

  // Front leg
  std::uniform_real_distribution<double> f_hip_distribution;
  std::uniform_real_distribution<double> f_thigh_distribution;
  std::uniform_real_distribution<double> f_calf_distribution;
  // Rear leg
  std::uniform_real_distribution<double> r_hip_distribution;
  std::uniform_real_distribution<double> r_thigh_distribution;
  std::uniform_real_distribution<double> r_calf_distribution;
  // Landing distribution
  std::uniform_real_distribution<double> rad_distribution;
  std::uniform_real_distribution<double> phi_distribution;
  std::uniform_int_distribution<int> leg_distribution;
  std::uniform_int_distribution<int> change_distribution;
  std::uniform_real_distribution<double> force_distribution;
  std::uniform_real_distribution<double> axis_distribution;

  // Parameters for leaning
  std::uniform_real_distribution<double> x_distribution;
  std::uniform_real_distribution<double> y_distribution;
  std::uniform_real_distribution<double> z_distribution;
  std::uniform_real_distribution<double> h_distribution;
  std::uniform_real_distribution<double> noise_distribution;
  std::uniform_real_distribution<double> contact_distribution;
  std::uniform_int_distribution<int> duration_distribution;

  // Parameters for domain randomization
  std::uniform_real_distribution<double> mass_distribution;
  std::uniform_real_distribution<double> pgain_distribution;
  std::uniform_real_distribution<double> dgain_distribution;
  std::uniform_real_distribution<double> friction_distribution; 
  std::uniform_real_distribution<double> terrain_distribution; 
  std::uniform_real_distribution<double> latency_distribution; 

  // For human data
  std::vector<Eigen::VectorXd> human_point_list;
  std::vector<Eigen::VectorXd> raw_trj;
  int human_cnt;
  Eigen::Vector4i currentContact;
  Eigen::Vector4i prevContact;
  Eigen::VectorXd kin_gc_prev;
  float init_foot;
  std::ofstream writeRobot;
  std::ofstream writeReference;

};

} 