#include <stdlib.h>
#include <cstdint>
#include <set>
#include <chrono>
#include <random>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <raisim/OgreVis.hpp>
#include <dirent.h>
#include "RaisimGymEnv.hpp"
#include "../include/visSetupCallback2.hpp"
#include "../include/Utilities.h" 
#include "../include/RobotInfo.hpp"
#include "../include/EnvMath.hpp"
#include "../include/MotionFunction.hpp"
#include "../include/TrajectoryGenerator.hpp"
#include "../include/RootPredictor.hpp"
#include "../include/Controller.hpp"

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
    ik_dummy = new raisim::ArticulatedSystem(resourceDir_+"/a1/a1/a1.urdf");
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
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    torque_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);
    gc_target.setZero(4 + nJoints_);
    gc_reference.setZero(gcDim_);
    /// this is nominal configuration of anymal
    gc_init_ <<  0, 0, 0.27, 1.0, 0.0, 0.0, 0.0, 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99);
    alien_->setState(gc_init_, gv_init_);
    ref_dummy->setState(gc_init_, gv_init_);
    ik_dummy->setState(gc_init_, gv_init_);
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

    human0.setZero(32); human1.setZero(32); human2.setZero(32);
    obDim_ = 196;
    actionDim_ = nJoints_;

    kin_gc_prev.setZero(gcDim_);
    actionStd_.setZero(nJoints_);  actionMean_.setZero(nJoints_);
    obMean_.setZero(obDim_); obStd_.setZero(obDim_);
    prev_vel.setZero(nJoints_); joint_acc.setZero(nJoints_);
    mController = new Controller(alien_, world_->getTimeStep());
    mController->jointControlSetter(init_pgain, init_dgain);

    // getKeyPoses("/home/sonic/Project/Alien_Gesture/env/rsc/new_motion.txt");

    float ah = 0.8028, at = 2.6179, ac = 0.8901;
    float mh = 0, mt = M_PI /2, mc = -1.806;
    actionStd_ << ah, at, ac, ah, at, ac, ah, at, ac, ah, at, ac;
    actionMean_ << mh, mt, mc, mh, mt, mc, mh, mt, mc, mh, mt, mc;

    freq_w = 0.1;
    main_w = 1.0 - freq_w;

    root_imit_scale = 20.0;
    joint_imit_scale = 1.0;
    ori_imit_scale = 5.0;
    end_imit_scale = 20.0;
    support_scale = 100.0;
    freq_scale = 1e-4;
    // readHumanPoints("/home/sonic/Project/Warping_Test/temp_rsc/mk2/points_reach11.txt");
    readHumanPoints("/home/sonic/Project/Warping_Test/temp_rsc/shorts/points/points_rs12.txt");
    // readHumanParams("/home/sonic/Project/Warping_Test/temp_rsc/mk2/params_reach11.txt"); 
    readHumanParams("/home/sonic/Project/Warping_Test/temp_rsc/shorts/params/params_rs12.txt"); 
    // readHumanPoints("/home/sonic/Project/Warping_Test/temp_rsc/mk2/points_reach12.txt");
    // readHumanParams("/home/sonic/Project/Warping_Test/temp_rsc/mk2/params_reach12.txt");

    gui::rewardLogger.init({"Root_reward", "Angle_reward", "Ori_reward", "End_reward", "Freq_reward"});
  
    /// indices of links that should not make contact with ground
    footIndices_.insert(alien_->getBodyIdx("FL_calf"));
    footIndices_.insert(alien_->getBodyIdx("FR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RL_calf"));

    default_friction_coeff = 0.7;
    raisim::MaterialManager materials_;
    materials_.setMaterialPairProp("floor", "robot", default_friction_coeff, 0.0, 0.0);
    world_->updateMaterialProp(materials_);
    alien_->getCollisionBody("FR_foot/0").setMaterial("robot");
    alien_->getCollisionBody("FL_foot/0").setMaterial("robot");
    alien_->getCollisionBody("RL_foot/0").setMaterial("robot");
    alien_->getCollisionBody("RR_foot/0").setMaterial("robot");

    mRP = RootPredictor(ref_dummy);
    mrl = new MotionReshaper(ref_dummy);
    raisim::Vec<3> foot_location;
    alien_->getFramePosition(alien_->getFrameByName("FL_foot_fixed"), foot_location);
    init_foot = foot_location[2];

    DIR *dir; struct dirent *diread;
    std::vector<std::string> files;
    char * motionDir = "/home/sonic/Project/Warping_Test/temp_rsc/shorts/params";
    std::string stDir(motionDir);
    if ((dir = opendir(motionDir)) != nullptr) {
      while ((diread = readdir(dir)) != nullptr) {
        if(strcmp(diread->d_name, ".") != 0 && strcmp(diread->d_name, "..") != 0 )
          files.push_back(std::string(diread->d_name));
      }
      closedir (dir);
    }
    for (auto file : files) {
      std::string paramtemp = stDir + '/' + file;
      std::string pointtemp = paramtemp;
      pointtemp.replace(pointtemp.find("params"), 6, "points");
      pointtemp.replace(pointtemp.find("params"), 6, "points");
      returnHumanParams(paramtemp);
      returnHumanPoints(pointtemp);
    }

    motion_picker = std::uniform_int_distribution<int>(0, files.size()-1);
    max_inter = 10;
    interpolated.setZero(3 * K4ABT_JOINT_COUNT); lastpoint.setZero(3 * K4ABT_JOINT_COUNT);
    lastclip.setZero(32);

    mass_distribution = std::uniform_real_distribution<double> (0.95, 1.05);
    pgain_distribution = std::uniform_real_distribution<double> (0.95, 1.05);
    dgain_distribution = std::uniform_real_distribution<double> (0.95, 1.05);
    damping_distribution = std::uniform_real_distribution<double> (0.95, 1.05);
    // stiffness_distribution = std::uniform_real_distribution<double> (0.95, 1.05);
    delay_distribution = std::uniform_real_distribution<double> (0.0, 0.005);
    friction_distribution = std::uniform_real_distribution<double> (0.9, 1.1);

    // Initialize the original dynamic parameters
    original_mass  = alien_->getMass();

    /// visualize if it is the first environment
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

      vis->createGraphicalObject(ground, 20, "floor", "checkerboard_green");

      desired_fps_ = 1.0/ control_dt_;
      vis->setDesiredFPS(desired_fps_);
      vis->select(tracking_graphics->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(-2.1), Ogre::Radian(-1.1), 4, true);
      vis->addVisualObject("supportPoly0", "cylinderMesh", "red", {0.005, 0.005, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("supportPoly1", "cylinderMesh", "red", {0.005, 0.005, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("supportPoly2", "cylinderMesh", "red", {0.005, 0.005, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("supportPoly3", "cylinderMesh", "red", {0.005, 0.005, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("cop", "sphereMesh", "aqua_marine", {0.02, 0.02, 0.02}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("com", "sphereMesh", "medium_purple", {0.02, 0.02, 0.02}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("dcm", "sphereMesh", "dark_magenta", {0.02, 0.02, 0.02}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      init_bones();
    }
  }

  ~ENVIRONMENT() final = default;

  void init() final { }

  void kinect_init() final{}

  void reset() final
  {
    alien_->setState(gc_init_, gv_init_);
    ref_dummy->setState(gc_init_, gv_init_);
    ik_dummy->setState(gc_init_, gv_init_);
    human_cnt = 0;
    len_shortclip = 0;
    gc_reference = gc_init_;
    kin_gc_prev = gc_init_;
    gc_reference[1] += 0.7;
    startFlag = true;

    prev_vel.setZero();
    joint_acc.setZero();

    state_history.clear();
    action_history.clear();
    state_history.shrink_to_fit();
    action_history.shrink_to_fit();
    state_history.push_back(gc_init_.tail(16));
    currentShortClip.clear();
    currentShortClip.shrink_to_fit();
    currentShortPoint.clear();
    currentShortPoint.shrink_to_fit();
    
    for(int i = 0; i < 3; ++i)
    {
      state_history.push_back(gc_init_.tail(16));
      action_history.push_back(gc_init_.tail(12));
    }
    gc_target = gc_init_.tail(nJoints_ + 4);

    root_imit_reward = 0.0;
    joint_imit_reward = 0.0;
    ori_imit_reward = 0.0;
    end_imit_reward = 0.0;
    support_reward = 0.0;
    freq_reward = 0.0;

    pTarget_.setZero();
    pTarget_.tail(nJoints_) = gc_init_.tail(nJoints_);
    alien_->setPTarget(pTarget_);

    mRP.reset();
    Bfilters.clear();
    Bfilters.shrink_to_fit();
    currentContact = Eigen::Vector4i(1,1,1,1);
    prevContact = Eigen::Vector4i(1,1,1,1);

    double cutoff_frequency = 6;
    for(int i = 0; i < nJoints_; ++i){
      Iir::Butterworth::LowPass<12> Bfilter;
      Bfilter.setup(desired_fps_, cutoff_frequency);
      Bfilters.push_back(Bfilter);
    }

    randomizeDomain();
    // updateHumanPose();
    updateHumanPoseShort();

    if(visualizable_){
      gui::rewardLogger.clean();
      auto& list = raisim::OgreVis::get()->getVisualObjectList();
      list["com"].setPosition(mCOM);
      list["dcm"].setPosition(mDCM);
      list["cop"].setPosition(mCOP);
    }
  }

  void setVisRef()
  {
    gc_reference.tail(16) = gc_target;
    mRP.predict(gc_target.tail(12));
    gc_reference[0] += mRP.getDX();
    gc_reference[2] = mRP.getHeight();
    vis_ref->setGeneralizedCoordinate(gc_reference);
  }

  void updateCOMDCM()
  {
    Eigen::Vector3d prevCOM = mCOM;
    mCOM = alien_->getCompositeCOM().e();
    mCOM[2] = 0;
    float current_height = alien_->getCompositeCOM()[2];
    float z_acc = (current_height - 2 * height1 + height2) / control_dt_ * control_dt_;
    float natural_freq = std::sqrt((9.8 + z_acc) / current_height);
    mDCM = mCOM + ((mCOM - prevCOM) / control_dt_) / natural_freq;
    height1 = current_height;
    height2 = height1;
    if (visualizable_) {
      auto& list = raisim::OgreVis::get()->getVisualObjectList();
      list["com"].setPosition(mCOM);
      list["dcm"].setPosition(mDCM);
    }
  }

  void updateCOP(){
    Eigen::Vector3d COP;
    COP.setZero();
    float tot_mag = 0.0;
    for (auto &contact: alien_->getContacts()){
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
        continue;
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
    if (visualizable_) {
      auto& list = raisim::OgreVis::get()->getVisualObjectList();
      list["cop"].setPosition(mCOP);
    }
  }

  void updatedSupportPolygon()
  {
    cnt_feet = 0;
    float contact_thresh = 0.025;
    for(int i = 0; i < 4; ++ i)
    {
      raisim::Vec<3> foot;
      alien_->getFramePosition(FR_FOOT + 3 * foot_order[i], foot);
      if(foot[2] < contact_thresh) 
      {
        cnt_feet++;
        contact_feet.push_back(foot_order[i]);
      }
    }
  }

  void visualizeSupportPolygon()
  {
    auto& list = raisim::OgreVis::get()->getVisualObjectList();
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

  void returnHumanPoints(std::string human_params)
  {
    std::ifstream human_input(human_params);
    if(!human_input){
      std::cout << "Expected valid input file name" << std::endl;
      return;
    }
    int triple_bone = 3 * K4ABT_JOINT_COUNT;
    std::vector<Eigen::VectorXd> tempHuman;
    while(!human_input.eof()){
      Eigen::VectorXd humanoid(triple_bone);
      for(int i = 0 ; i < triple_bone; ++i)
      {
        float ang;
        human_input >> ang;
        humanoid[i] = ang;
      }
      tempHuman.push_back(humanoid);
    }
    tempHuman.pop_back();
    shortpoints.push_back(tempHuman);
    tempHuman.clear();
    tempHuman.shrink_to_fit();
    human_input.close();
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

  void returnHumanParams(std::string human_params)
  {
    std::ifstream human_input(human_params);
    if(!human_input){
      std::cout << "Expected valid input file name" << std::endl;
      return;
    }
    std::vector<Eigen::VectorXd> tempHuman;
    int num_params = 32;
    while(!human_input.eof()){
      Eigen::VectorXd humanoid(num_params);
      for(int i = 0 ; i < num_params; ++i)
      {
        float ang;
        human_input >> ang;
        humanoid[i] = ang;
      }
      tempHuman.push_back(humanoid);
    }
    tempHuman.pop_back();
    shortclips.push_back(tempHuman);
    tempHuman.clear();
    tempHuman.shrink_to_fit();
    human_input.close();
  }

  void readHumanParams(std::string human_params)
  {
    std::ifstream human_input(human_params);
    if(!human_input){
      std::cout << "Expected valid input file name" << std::endl;
    }

    int num_params = 32;
    while(!human_input.eof()){
      Eigen::VectorXd humanoid(num_params);
      for(int i = 0 ; i < num_params; ++i)
      {
        float ang;
        human_input >> ang;
        humanoid[i] = ang;
      }
      human_param_list.push_back(humanoid);
    }
    human_param_list.pop_back();
    human_input.close();
  }

  void updateHumanPose() 
  {
    human0 = human_param_list[human_cnt];
    if(startFlag) 
    {
      human1 = human0;
      human2 = human0;
      startFlag = false;
    }
    if(visualizable_) visualizeBones();
    human_cnt++;
    if(human_cnt >= human_point_list.size()-1) human_cnt = 0;
    updateObservation();
  }

  void updateHumanPoseShort() 
  {
    if(human_cnt == len_shortclip)
    { 
      if(!startFlag){
        lastclip = currentShortClip[human_cnt -2];
        lastpoint = currentShortPoint[human_cnt -2];
      }
      currentShortClip.clear();
      currentShortClip.shrink_to_fit();
      currentShortPoint.clear();
      currentShortPoint.shrink_to_fit();
      std::random_device rd;
      std::mt19937 generator(rd());
      int idx = motion_picker(generator);
      currentShortClip = shortclips[idx];
      currentShortPoint = shortpoints[idx];
      len_shortclip = currentShortClip.size();
      intermediate_cnt = 0;
      human_cnt = 0;
    }
    if(startFlag) 
    {
      human0 = currentShortClip[0];
      human1 = human0;
      human2 = human0;
      intermediate_cnt = max_inter;
      startFlag = false;
    }

    if(intermediate_cnt < max_inter)
    {
      interpolateBones();
    }
    else
    {
      human0 = currentShortClip[human_cnt];
      human_cnt++;
    }
    if(visualizable_) visualizeBonesShorts();
    intermediate_cnt++;
    updateObservation();
  }

  void interpolateBones()
  {
    float percentile = (float)intermediate_cnt / (float)max_inter;
    auto curclip = currentShortClip[0];
    auto curpoint = currentShortPoint[0];
    Eigen::VectorXd interclip(32);
    Eigen::Quaterniond rootc(curclip[0], curclip[1], curclip[2], curclip[3]);
    Eigen::Quaterniond rootl(lastclip[0], lastclip[1], lastclip[2], lastclip[3]);
    Eigen::Quaterniond chestc(curclip[4], curclip[5], curclip[6], curclip[7]);
    Eigen::Quaterniond chestl(lastclip[4], lastclip[5], lastclip[6], lastclip[7]);
    Eigen::Quaterniond rootn = rootc.slerp(percentile, rootl);
    Eigen::Quaterniond chestn = chestc.slerp(percentile, chestl);
    interclip[0] = rootn.w(); interclip[1] = rootn.x(); interclip[2] = rootn.y(); interclip[3] = rootn.z();
    interclip[4] = chestn.w(); interclip[5] = chestn.x(); interclip[6] = chestn.y(); interclip[7] = chestn.z(); 
    for(int i = 8; i < curclip.size(); ++i)
    {
      interclip[i] = lastclip[i] * (1.0 - percentile) + curclip[i] * percentile;
    }
    human0 = interclip;
    interpolated = lastpoint * (1.0 - percentile) + curpoint * percentile;
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

  void visualizeBonesShorts()
  {
    float scaler = 2.1;
    float y_shift = 0.2;
    auto& list = raisim::OgreVis::get()->getVisualObjectList();    

    Eigen::VectorXd current_keypose(3 * K4ABT_JOINT_COUNT);
    if(intermediate_cnt < max_inter){
      current_keypose = scaler * interpolated;
    }
    else
    {
      current_keypose = scaler * currentShortPoint[human_cnt-1];
    }
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
    /// action scaling
    Eigen::VectorXd currentAction = action.cast<double>();
    pTarget12_ = currentAction;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;
    setVisRef();
    alien_->setPTarget(pTarget_);
    auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
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

    updatedSupportPolygon();
    updateCOMDCM();
    updateCOP();
    if(visualizable_) visualizeSupportPolygon();

    float totalReward = CalcReward();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("Support_reward", support_reward);
      gui::rewardLogger.log("Root_reward", root_imit_reward);
      gui::rewardLogger.log("Angle_reward", joint_imit_reward);
      gui::rewardLogger.log("Ori_reward", ori_imit_reward);
      gui::rewardLogger.log("End_reward", end_imit_reward);
      gui::rewardLogger.log("Freq_reward", freq_reward);
      auto vis = raisim::OgreVis::get();

      vis->select(anymalVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }
    updateHumanPoseShort();
    return totalReward;
  }

  float stepTest(const Eigen::Ref<EigenVec>& action) final {
    Eigen::VectorXd currentAction = action.cast<double>();
    pTarget12_ = currentAction;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;
    // std::cout << pTarget12_.transpose() << std::endl;
    setVisRef();
    alien_->setPTarget(pTarget_);
    auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
    auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);    
    action_history.erase(action_history.begin());
    action_history.push_back(pTarget12_);
    mController->setTargetPosition(pTarget_);

    for(int i=0; i<loopCount; i++) {
      // mController->clearForces();
      // mController->addSPDForces();
      world_->integrate();
      // std::cout <<"imp" << alien_->getGeneralizedForce().e().transpose() << std::endl;
      // std::cout <<"exp" << mController->getForces().transpose() << std::endl;
      if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
      {
        auto vis = raisim::OgreVis::get();
        vis->renderOneFrame();
      }      
      visualizationCounter_++;
    }
    tracking_ball->setPosition(gc_reference[0], 0.5, 0.3);

    updatedSupportPolygon();
    updateCOMDCM();
    updateCOP();
    visualizeSupportPolygon();
    float totalReward = CalcReward();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("Support_reward", support_reward);
      gui::rewardLogger.log("Angle_reward", joint_imit_reward);
      gui::rewardLogger.log("Ori_reward", ori_imit_reward);
      gui::rewardLogger.log("End_reward", end_imit_reward);
      gui::rewardLogger.log("Freq_reward", freq_reward);
      auto vis = raisim::OgreVis::get();

      // vis->select(anymalVisual_->at(0), false);
      // vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }
    // updateHumanPose();
    updateHumanPoseShort();
    if(!mStopFlag) 
      return totalReward;
    else
      return -1E10;
  }

  void getMappedTarget(const Eigen::Ref<EigenVec>& target, const Eigen::Ref<EigenVec>& contact_info)
  {
    // CurrentContact = 1 : In contact, 0: On air
    gc_target = target.cast<double>();
    Eigen::VectorXd contact_vec = contact_info.cast<double>();
    for(int i = 0; i <4; ++i)
    {
      if(contact_vec[i] > 0.5)
        currentContact[i] = 0;
      else
        currentContact[i] = 1;
    }
    setReshapedMotion2();
  }

  void setReshapedMotion()
  {
    for(int i = 0; i  < 4 ; ++i)
    {
      if(currentContact[i] == 0)
      {
        kin_gc_prev.segment(7 + 3 * i, 3) = gc_target.segment(4 + 3 * i, 3);
      }
      else if(currentContact[i] - prevContact[i] == 1)
      {
        kin_gc_prev.segment(7 + 3 * i, 3) = gc_target.segment(4 + 3 * i, 3);
      }
    }

    Eigen::VectorXd gc_raw = gc_init_;
    gc_raw.tail(16) = gc_target;
    ref_dummy->setGeneralizedCoordinate(gc_raw);
    float current_foot = 10000;
    for(int i= 0; i < 4; ++i)
    {
      raisim::Vec<3> temp;
      ref_dummy->getFramePosition(FR_FOOT + 3 * i, temp);
      if(temp[2] < current_foot) current_foot = temp[2];
    }
    gc_raw[2] += (init_foot - current_foot);
    mrl->getDynStates(state_history[0], state_history[1]);
    mrl->getAngularVelocity(gv_.segment(3,3));
    mrl->contactInformation(currentContact);
    mrl->getPreviousRealPose(kin_gc_prev);
    mrl->getCurrentRawPose(gc_raw);
    mrl->getKinematicFixedPose();
    Eigen::VectorXd gc_reshaped = mrl->getReshapedMotion();
    gc_reference[2] =gc_reshaped[2];
    gc_target = gc_reshaped.tail(16);
    prevContact = currentContact;
  }

  void setReshapedMotion2()
  {
    kin_gc_prev = gc_init_;
    kin_gc_prev.tail(16) = gc_target;
    Eigen::VectorXd gc_raw = gc_init_;
    gc_raw.tail(16) = gc_target;
    ref_dummy->setGeneralizedCoordinate(gc_raw);
    float current_foot = 10000;
    for(int i= 0; i < 4; ++i)
    {
      raisim::Vec<3> temp;
      ref_dummy->getFramePosition(FR_FOOT + 3 * i, temp);
      if(temp[2] < current_foot) current_foot = temp[2];
    }
    gc_raw[2] += (init_foot - current_foot);
    mrl->getDynStates(state_history[0], state_history[1]);
    mrl->getAngularVelocity(gv_.segment(3,3));
    mrl->contactInformation(currentContact);
    mrl->getPreviousRealPose(kin_gc_prev);
    mrl->getCurrentRawPose(gc_raw);
    mrl->getKinematicFixedPoseTest();
    Eigen::VectorXd gc_reshaped = mrl->getReshapedMotion();
    gc_reference[2] = gc_reshaped[2];
    gc_target = gc_reshaped.tail(16);
  }

  void curriculumUpdate(int update) final 
  {
    int curr1 = 1000;
    int curr2 = 2000;
    int curr3 = 3000;
    int curr4 = 5000;
    int curr5 = 20500;
    int curr6 = 23000;
    int curr7 = 25000;

    if(update > curr1 && update <=curr2)
    {
      mass_distribution = std::uniform_real_distribution<double> (0.85, 1.10);
      pgain_distribution = std::uniform_real_distribution<double> (0.85, 1.10);
      dgain_distribution = std::uniform_real_distribution<double> (0.90, 1.10);
    }
    else if(update > curr2 && update <=curr3)
    {
      mass_distribution = std::uniform_real_distribution<double> (0.80, 1.15);
      pgain_distribution = std::uniform_real_distribution<double> (0.80, 1.20);
      dgain_distribution = std::uniform_real_distribution<double> (0.80, 1.20);
    }
    else if(update > curr3 && update <=curr4)
    {
      mass_distribution = std::uniform_real_distribution<double> (0.75, 1.15);
      pgain_distribution = std::uniform_real_distribution<double> (0.75, 1.25);
      dgain_distribution = std::uniform_real_distribution<double> (0.75, 1.25);
    }
    else if(update > curr4&& update <=curr5)
    {
      mass_distribution = std::uniform_real_distribution<double> (0.75, 1.2);
      pgain_distribution = std::uniform_real_distribution<double> (0.75, 1.25);
      dgain_distribution = std::uniform_real_distribution<double> (0.75, 1.25);
    }
    else if(update > curr5&& update <=curr6)
    {
      mass_distribution = std::uniform_real_distribution<double> (0.75, 1.2);
      pgain_distribution = std::uniform_real_distribution<double> (0.7, 1.3);
      dgain_distribution = std::uniform_real_distribution<double> (0.7, 1.3);
      friction_distribution = std::uniform_real_distribution<double> (0.7, 1.3);
    }
    else if(update > curr6&& update <=curr7)
    {
      friction_distribution = std::uniform_real_distribution<double> (0.6, 1.4);
    }
    else if(update > curr7)
    {
      friction_distribution = std::uniform_real_distribution<double> (0.5, 1.5);
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
    // Eigen::VectorXd dampingCoefficient;
    // damping_distribution(generator);
    // alien_-> setJointDamping
    alien_->updateMassInfo();
  }

  float CalcReward()
  {   
    float reward = 0;
    double root_imit_err = 0;
    double joint_imit_err = 0;
    double ori_imit_err = 0;
    double end_imit_err = 0;
    double support_err = 0;
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
    joint_imit_err = jointImitationErr2();
    ori_imit_err = orientationImitationErr();
    // end_imit_err = endEffectorImitationErr2();
    end_imit_err = endEffectorImitationErrLoose();
    support_err = CalcSupportErr();
    freq_err = CalcAccErr();

    root_imit_reward = exp(-root_imit_scale * root_imit_err);
    joint_imit_reward = exp(-joint_imit_scale * joint_imit_err);
    ori_imit_reward = exp(-ori_imit_scale * ori_imit_err);
    end_imit_reward = exp(-end_imit_scale * end_imit_err);
    support_reward = exp(-support_scale * support_err);
    freq_reward = exp(- freq_scale * freq_err);

    reward = main_w * support_reward * joint_imit_reward * ori_imit_reward * end_imit_reward *root_imit_reward
      + freq_w * freq_reward;
    return reward;
  }

  float rootImitationErr()
  {
    float err = 0.;
    err += pow((gc_reference[0] - gc_[0]),2);
    err += pow((gc_reference[1] - 0.7 - gc_[1]),2);
    err += 0.1 * pow((gc_reference[2] - gc_[2]),2);

    return err;
  }

  float jointImitationErr()
  {
    float err = 0.;
    for(int i = 0; i < nJoints_; ++i)
    {
      if(i%3 == 0)
        err += 0.05 * pow((gc_target[4+i] - gc_[7 + i]) , 2);
      else
        err += pow((gc_target[4 + i] - gc_[7 + i]),2);
    }
    return err;
  }

  float jointImitationErr2()
  {
    float err = 0.;
    for(int i = 0; i < nJoints_; ++i)
    {
      if(currentContact[int(i/3)] == 1)
        err += 0.1 * pow((gc_target[4 + i] - gc_[7 + i]) , 2);
      else
        err += pow((gc_target[4 + i] - gc_[7 + i]),2);
    }
    return err;
  }

  float orientationImitationErr()
  {
    float err = 0.;
    float epsilon = 1e-6;
    Eigen::Matrix3d rot_m1;
    Eigen::Matrix3d rot_m2;
    rot_m1  = Eigen::Quaterniond(gc_[3],gc_[4],gc_[5],gc_[6]);
    rot_m2  = Eigen::Quaterniond(gc_target[0],gc_target[1],gc_target[2],gc_target[3]);
    Eigen::Vector3d current_dir = rot_m1 * Eigen::Vector3d::UnitX();
    Eigen::Vector3d next_dir = rot_m2 * Eigen::Vector3d::UnitX();
    double value = (current_dir[0] * next_dir[0] + current_dir[1] * next_dir[1]) / 
      (std::sqrt(pow(current_dir[0], 2) + pow(current_dir[1], 2)) * std::sqrt(pow(next_dir[0], 2) + pow(next_dir[1], 2)));
    value = std::min(1.0 - epsilon, value);
    value = std::max(-1.0 + epsilon, value);
    float angle = acos(value);
    err = angle * angle;
    // Use slerp
    // Eigen::Quaterniond target(gc_target[0],gc_target[1],gc_target[2],gc_target[3]);
    // Eigen::Quaterniond current(gc_[3],gc_[4],gc_[5],gc_[6]);
    // err = pow(current.angularDistance(target), 2);

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
    Eigen::VectorXd fixed_pose = gc_init_;
    fixed_pose.tail(4+ nJoints_) = gc_target;
    ik_dummy->setGeneralizedCoordinate(fixed_pose);
    // Use endeffector position
    for(int i = 0; i < 4; ++i)
    {
      raisim::Vec<3> footPositionA;
      raisim::Vec<3> footPositionR;
      raisim::Vec<3> footPositionP;

      alien_->getFramePosition(FR_FOOT+3 * i, footPositionA);
      ref_dummy->getFramePosition(FR_FOOT+3 * i, footPositionR);
      ik_dummy->getFramePosition(FR_FOOT+3 * i, footPositionP);
      if(currentContact[i] == 1)
      {
        err += 100 * pow((footPositionP[2] - footPositionA[2]) ,2);
      }
      else
      {
        err += (footPositionR.e() - footPositionA.e()).squaredNorm();
      }
    }
    return err;
  }
  
  float endEffectorImitationErr2()
  {
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
      if(currentContact[i] == 1)
        err+= 10 * pow((footPositionR[2] - footPositionA[2]), 2);
      else
        err += (footPositionR.e() - footPositionA.e()).squaredNorm();
    }
    return err;
  }

  float endEffectorImitationErrLoose()
  {
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
      if(currentContact[i] == 1)
        err+= 0.1 * pow((footPositionR[2] - footPositionA[2]), 2);
      else
        err += (footPositionR.e() - footPositionA.e()).squaredNorm();
    }
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
    contact_feet.clear();
    contact_feet.shrink_to_fit();
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
    // Here, time horizon and ... 

    alien_->getState(gc_, gv_);
    obScaled_.setZero(obDim_);

    state_history.erase(state_history.begin());
    state_history.push_back(gc_.tail(16));

    // Human pose triplets
    obScaled_.segment(0, 32) = human2;
    obScaled_.segment(32, 32) = human1;
    obScaled_.segment(64, 32) = human0;
    for(int i = 0; i < 4; ++i)
    {
      obScaled_.segment(96 + 16 * i, 16) = state_history[i];
    }
    for(int i = 0; i < 3; ++i)
    {
      obScaled_.segment(160 + 12 * i, 12) = action_history[i];
    }
    human2 = human1;
    human1 = human0;
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
  raisim::ArticulatedSystem* ik_dummy;
  raisim::ArticulatedSystem* ref_dummy;
  raisim::ArticulatedSystem* vis_ref;
  raisim::Sphere* tracking_ball;

  std::vector<std::vector<Eigen::VectorXd>> shortclips;
  std::vector<std::vector<Eigen::VectorXd>> shortpoints;
  std::vector<GraphicObject>* anymalVisual_;
  std::vector<GraphicObject>* tracking_graphics;
  std::vector<Eigen::VectorXd> action_history;
  std::vector<Eigen::VectorXd> state_history;

  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, torque_, gc_dummy_init;
  Eigen::VectorXd gc_target, gc_reference, kin_gc_prev;
  Eigen::VectorXd noise;
  double desired_fps_ = 30.;
  int visualizationCounter_=0;
  float effort_limit = 33.5 * 0.90;
  Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_;
  Eigen::VectorXd obDouble_, obScaled_;
  std::set<size_t> footIndices_;
  Eigen::VectorXd human0, human1, human2;
  std::vector<Eigen::VectorXd> human_point_list;
  std::vector<Eigen::VectorXd> human_param_list;
  bool startFlag;
  Eigen::Vector4i currentContact;
  Eigen::Vector4i prevContact;
  float init_foot;
  int human_cnt;
  int cnt_feet;
  Eigen::VectorXd prev_vel, joint_acc;
  std::vector<std::deque<double>> torque_sequence;
  std::vector<Iir::Butterworth::LowPass<12>> Bfilters;
  RootPredictor mRP;
  MotionReshaper *mrl;
  Controller *mController;
  // Reward Shapers
  double main_w, freq_w;
  double root_imit_scale, joint_imit_scale, ori_imit_scale, end_imit_scale,support_scale, freq_scale;
  double root_imit_reward, joint_imit_reward, ori_imit_reward, end_imit_reward, support_reward, freq_reward;
  float poly_scale = 0.1;
  Eigen::Vector3d mCOP, mCOM, mDCM;
  std::vector<int> contact_feet;
  float height1, height2;
  int foot_order[4] = {0,1,3,2};

  std::uniform_int_distribution<int> motion_picker;
  int intermediate_cnt, max_inter, len_shortclip;
  std::vector<Eigen::VectorXd> currentShortClip;
  std::vector<Eigen::VectorXd> currentShortPoint;
  Eigen::VectorXd interpolated, lastclip, lastpoint;

  // What else?
  std::uniform_real_distribution<double> mass_distribution;
  std::uniform_real_distribution<double> pgain_distribution;
  std::uniform_real_distribution<double> dgain_distribution;
  std::uniform_real_distribution<double> damping_distribution;
  // std::uniform_real_distribution<double> stiffness_distribution;
  std::uniform_real_distribution<double> delay_distribution;
  std::uniform_real_distribution<double> friction_distribution;

  std::vector<double> original_mass;

  // PD Settings
  double init_pgain = 200;
  double init_dgain = 10;
  Eigen::VectorXd jointPgain, jointDgain;
  Eigen::Vector3d supportCentre;

  float default_friction_coeff;
  // Imitation Learning

};

}