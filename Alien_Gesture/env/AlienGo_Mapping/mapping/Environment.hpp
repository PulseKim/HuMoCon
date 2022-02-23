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
    ik_dummy = new raisim::ArticulatedSystem(resourceDir_+"/a1/a1/a1.urdf");
    vis_ref = new raisim::ArticulatedSystem(resourceDir_+"/a1/a1/a1.urdf");    
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
    gc_target.setZero(4 + nJoints_); gc_prev.setZero(gcDim_);
    gc_reference.setZero(gcDim_);
    /// this is nominal configuration of anymal
    gc_init_ <<  0, 0, 0.27, 1.0, 0.0, 0.0, 0.0, 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99);
    alien_->setState(gc_init_, gv_init_);
    ref_dummy->setState(gc_init_, gv_init_);
    ik_dummy->setState(gc_init_, gv_init_);

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

    root_imit_scale = 20.0;
    joint_imit_scale = 3.0;
    ori_imit_scale = 5.0;
    end_imit_scale = 10.0;
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
    // mTG = TrajectoryGenerator(CRAB_WALK, ltc, lcf, dt);
    mTG = TrajectoryGenerator(WALKING_TROT, ltc, lcf, dt);
    // mTG = TrajectoryGenerator(JUST_WALK, ltc, lcf, dt);
    mTG.set_Ae(0.11);

    Ae_distribution =  std::uniform_real_distribution<double> (0.11, 0.11);
    height_distribution =  std::uniform_real_distribution<double> (0.25, 0.28);
    swing_distribution =  std::uniform_real_distribution<double> (deg2rad(10), deg2rad(15));
    phase_speed_distribution =  std::uniform_real_distribution<double> (0.8, 1.2);
    noise_distribution =  std::uniform_real_distribution<double> (-deg2rad(0.), deg2rad(0.));
    anglularvel_distribution =  std::uniform_real_distribution<double> (-deg2rad(0.), deg2rad(0.));
    initial_distribution =  std::uniform_int_distribution<int> (0,3);
    // change_distribution = std::uniform_int_distribution<int>(30,30);
    change_distribution = std::uniform_int_distribution<int>(60,60);


    mass_distribution = std::uniform_real_distribution<double> (0.95, 1.05);
    pgain_distribution = std::uniform_real_distribution<double> (0.95, 1.05);
    dgain_distribution = std::uniform_real_distribution<double> (0.95, 1.05);
    friction_distribution = std::uniform_real_distribution<double> (0.9, 1.1);
    latency_distribution = std::uniform_real_distribution<double> (1.0, 1.1);

    original_mass  = alien_->getMass();

    mRP = RootPredictor(ref_dummy);
    // readReference("/home/sonic/Project/Alien_Gesture/ref_motion.txt");

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
      ref_graphics = vis->createGraphicalObject(vis_ref, "visualization");

      // vis->createGraphicalObject(ref_dummy, "dummy");
      vis->createGraphicalObject(ground, 20, "floor", "checkerboard_green");

      desired_fps_ = 1.0/ control_dt_;
      vis->setDesiredFPS(desired_fps_);
      vis->select(ref_graphics->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(-2.1), Ogre::Radian(-1.1), 4, true);
      initSupportPolygon();
    }
    std::string filePath1 = "/home/sonic/Project/Alien_Gesture/deep_mimic.txt";
    writeRobot.open(filePath1);
  }

  ~ENVIRONMENT() final = default;

  void init() final { }

  void kinect_init() final{
    // int max_cnt = 500;
    TGcurriculum(11000);
    DRcurriculum(11000);

    // human_cnt = 0;
    // readHumanPoints("/home/sonic/Project/Alien_Gesture/rsc/map_loco/points_loco11.txt");
    // init_bones();
    // currentContact = Eigen::Vector4i(1,1,1,1);
    // prevContact = Eigen::Vector4i(1,1,1,1);
    // kin_gc_prev = gc_init_;
    // raisim::Vec<3> footPositionA;
    // alien_->getFramePosition(FR_FOOT, footPositionA);
    // init_foot = footPositionA[2];
  }

  void reset() final
  {
    TGcurriculum(20000);
    DRcurriculum(11000);
    alien_->setState(gc_init_, gv_init_);
    ref_dummy->setState(gc_init_, gv_init_);
    ik_dummy->setState(gc_init_, gv_init_);
    current_count = 0;
    gc_reference = gc_init_;
    gc_reference[1] += 0.7;

    prev_vel.setZero();
    joint_acc.setZero();

    gc_prev = gc_init_;
    gc_target = gc_init_.tail(nJoints_ + 4);

    trajectories.clear();
    trajectories.shrink_to_fit();
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

    root_imit_reward = 0.0;
    joint_imit_reward = 0.0;
    ori_imit_reward = 0.0;
    end_imit_reward = 0.0;
    freq_reward = 0.0;

    std::random_device rd;
    std::mt19937 generator(rd());
    float rand = initial_distribution(generator);
    phi = rand * M_PI / 2;
    ang_vel = 0.;
    angle_cummul = 0.;

    pTarget_.setZero();
    pTarget_.tail(nJoints_) = gc_init_.tail(nJoints_);
    alien_->setPTarget(pTarget_);

    mRP.reset();
    mTG.set_Ae(0.11);
    mTG.change_Cs(0.0);
    mTG.get_tg_parameters(1.0, 0.0, 0.27);

    generateRandomSeq();
    randomizeDomain();
    updatedSupportPolygon();
    updateObservation();

    if(visualizable_)
      gui::rewardLogger.clean();
  }

  void estimateContact(const Eigen::Ref<EigenVec>& contact)
  {
    Eigen::VectorXd contact_vec = contact.cast<double>();
    for(int i = 0; i <4; ++i)
    {
      if(contact_vec[i] > 0.0)
        currentContact[i] = 0;
      else
        currentContact[i] = 1;
    }
    
  }

  void getReferencePoses(const Eigen::Ref<EigenVec>& pose)
  {
    Eigen::VectorXd currentAction = pose.cast<double>();
    setReshapedMotion(currentAction);
  }

  void setReshapedMotion(Eigen::VectorXd currentAction)
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

  void setVisRef()
  {
    gc_reference.tail(16) = gc_target.tail(16);
    mRP.predict(gc_target.tail(12));
    gc_reference[0] += mRP.getDX() * cos(ang_vel * control_dt_);
    gc_reference[1] += mRP.getDX() * sin(ang_vel * control_dt_);
    gc_reference[2] = mRP.getHeight();
    vis_ref->setGeneralizedCoordinate(gc_reference);
  }

  void setVisRef2()
  {
    gc_reference.tail(16) = gc_target.tail(16);
    mRP.predict(gc_target.tail(12));
    gc_reference[0] = root_trj[current_count][0];
    gc_reference[1] = root_trj[current_count][1] + 0.7;
    gc_reference[2] = 0.27;
    vis_ref->setGeneralizedCoordinate(gc_reference);
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

  void readReference(std::string ref_file)
  {
    // Reader ==> Save as vector
    std::ifstream human_input(ref_file);
    if(!human_input){
      std::cout << "Expected valid input file name" << std::endl;
    }
    while(!human_input.eof()){
      Eigen::Vector3d rootTj;
      Eigen::VectorXd humanoid(16);
      for(int i = 0 ; i < 3; ++i)
      {
        float ang;
        human_input >> ang;
        rootTj[i] = ang;
      }
      for(int i = 0 ; i < 16; ++i)
      {
        float ang;
        human_input >> ang;
        humanoid[i] = ang;
      }
      root_trj.push_back(rootTj);
      trajectories.push_back(humanoid);
    }
    root_trj.pop_back();
    trajectories.pop_back();
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


  void updatedSupportPolygon()
  {
    contact_feet.clear();
    contact_feet.shrink_to_fit();
    float contact_thresh = 0.025;
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

  void generateRandomSeq()
  {
    int cnt = 0; 
    Eigen::Quaterniond prev_quat(gc_init_[3],gc_init_[4],gc_init_[5],gc_init_[6]);
    Eigen::VectorXd prev_pose = gc_init_.tail(12);
    motionBlender(prev_quat, prev_quat, prev_pose, prev_pose, 20);
    cnt+= 20;

    while(cnt < max_cnt)
    {
      // First decide the leg to manipulate
      std::random_device rd;
      std::mt19937 generator(rd());
      float h_tg  = height_distribution(generator);
      float alpha_tg = swing_distribution(generator);
      float Ae = Ae_distribution(generator);
      float phase_speed = phase_speed_distribution(generator);
      int change_cnt = change_distribution(generator);
      ang_vel = anglularvel_distribution(generator);
      Eigen::VectorXd noise = Eigen::VectorXd::Zero(12);
      for(int i =0; i<12; ++i)
      {
        unsigned seed_temp = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine gen(seed_temp);
        noise[i] = noise_distribution(gen);
      }
      mTG.change_Cs(0.0);
      mTG.set_Ae(Ae);
      mTG.get_tg_parameters(1.0, alpha_tg, h_tg);
      for(int i = 0; i < change_cnt; ++i)
      {
        get_current_target(phase_speed, noise);
      }
      cnt += change_cnt;
    }
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
      trajectories.push_back(total);
    }
  }

  void get_current_target(float phase_speed, Eigen::VectorXd noise)
  {
    Eigen::VectorXd target(16);
    gc_ = alien_->getGeneralizedCoordinate().e();
    phi += 2 * M_PI * control_dt_ * phase_speed;
    phi = fmod(phi, 2 * M_PI);
    mTG.manual_timing_update(phi);
    Eigen::Quaterniond quat_target;
    angle_cummul += ang_vel * control_dt_;
    angle_cummul = fmod(angle_cummul, 2* M_PI);
    quat_target = Eigen::AngleAxisd(angle_cummul, Eigen::Vector3d::UnitZ());
    target.head(4) = Eigen::Vector4d(quat_target.w(), quat_target.x(),quat_target.y(),quat_target.z());
    target.tail(12) = mTG.get_u();
    target.tail(12) += noise;
    trajectories.push_back(target);
  }

  void manualizeTGparams()
  {  
    mTG.change_Cs(0.0);
    mTG.get_tg_parameters(1.0, alpha_manual, h_manual);
    phase_speed = speed_manual;
  } 

  float stepRef() final
  {
    return 0;
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    gc_target = trajectories[current_count];
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

    updatedSupportPolygon();
    updateObservation();

    float totalReward = CalcReward();

    if(visualizeThisStep_) {
      visualizeSupportPolygon();
      gui::rewardLogger.log("Root_reward", root_imit_reward);
      gui::rewardLogger.log("Angle_reward", joint_imit_reward);
      gui::rewardLogger.log("Ori_reward", ori_imit_reward);
      gui::rewardLogger.log("End_reward", end_imit_reward);
      gui::rewardLogger.log("Freq_reward", freq_reward);
      auto vis = raisim::OgreVis::get();
      vis->select(ref_graphics->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }
    current_count++;
    if(!mStopFlag) 
      return totalReward;
    else
      return -1E10;
  }

  float stepTest(const Eigen::Ref<EigenVec>& action) final {
    gc_target = trajectories[current_count];
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
     for(int i = 0; i < gc_.size()-1;++i)
    {
      writeRobot << std::setprecision(3) << std::fixed << gc_[i] << " ";
    }
    writeRobot << std::setprecision(3) << std::fixed << gc_[gc_.size()-1];
    writeRobot << std::endl;

    updatedSupportPolygon();
    updateObservation();

    float totalReward = CalcReward();

    if(visualizeThisStep_) {
      visualizeSupportPolygon();
      gui::rewardLogger.log("Root_reward", root_imit_reward);
      gui::rewardLogger.log("Angle_reward", joint_imit_reward);
      gui::rewardLogger.log("Ori_reward", ori_imit_reward);
      gui::rewardLogger.log("End_reward", end_imit_reward);
      gui::rewardLogger.log("Freq_reward", freq_reward);
      auto vis = raisim::OgreVis::get();
      vis->select(ref_graphics->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }
    current_count++;
    if(!mStopFlag) 
      return totalReward;
    else
      return -1E10;
  }

  void curriculumUpdate(int update) final {
    // TGcurriculum(update);
    TGcurriculum(20000);
    DRcurriculum(15000);
  } 
  void TGcurriculum(int update)
  {
    int curr1 = 2000;
    int curr2 = 4000;
    int curr3 = 6000;
    int curr4 = 11000;
    int curr5 = 13000;
    int curr6 = 15000;
    if(update >= curr1 && update <curr2) 
    {
      Ae_distribution =  std::uniform_real_distribution<double> (0.10, 0.12);
      height_distribution =  std::uniform_real_distribution<double> (0.24, 0.29);
      phase_speed_distribution =  std::uniform_real_distribution<double> (0.8, 1.2);
      anglularvel_distribution = std::uniform_real_distribution<double> (deg2rad(-10), deg2rad(10));
    }
    else if(update >= curr2 && update <curr3) 
    {
      Ae_distribution =  std::uniform_real_distribution<double> (0.09, 0.13);
      height_distribution =  std::uniform_real_distribution<double> (0.23, 0.30);
      swing_distribution =  std::uniform_real_distribution<double> (deg2rad(0), deg2rad(18));
      phase_speed_distribution =  std::uniform_real_distribution<double> (0.8, 1.2);
      anglularvel_distribution = std::uniform_real_distribution<double> (deg2rad(-10), deg2rad(10));
      // change_distribution = std::uniform_int_distribution<int>(20, 30);
    }
    else if(update >= curr3 && update <curr4) 
    {
      Ae_distribution =  std::uniform_real_distribution<double> (0.13, 0.18);
      height_distribution =  std::uniform_real_distribution<double> (0.26, 0.28);
      swing_distribution =  std::uniform_real_distribution<double> (deg2rad(0), deg2rad(20));
      phase_speed_distribution =  std::uniform_real_distribution<double> (0.8, 1.2);
      anglularvel_distribution = std::uniform_real_distribution<double> (deg2rad(-10), deg2rad(10));
      // change_distribution = std::uniform_int_distribution<int>(30, 30);
    }
    else if(update >= curr4) 
    {
      Ae_distribution =  std::uniform_real_distribution<double> (0.05, 0.19);
      height_distribution =  std::uniform_real_distribution<double> (0.22, 0.34);
      swing_distribution =  std::uniform_real_distribution<double> (deg2rad(0), deg2rad(25));
      phase_speed_distribution =  std::uniform_real_distribution<double> (0.3, 1.2);
      // noise_distribution =  std::uniform_real_distribution<double> (-deg2rad(15), deg2rad(15));
      // anglularvel_distribution = std::uniform_real_distribution<double> (deg2rad(-25), deg2rad(25));
      anglularvel_distribution = std::uniform_real_distribution<double> (deg2rad(-10), deg2rad(10));
      change_distribution = std::uniform_int_distribution<int>(15, 40);
    }
    // else if(update >= curr5 )
    // {
    //   Ae_distribution =  std::uniform_real_distribution<double> (0.05, 0.19);
    //   height_distribution =  std::uniform_real_distribution<double> (0.22, 0.33);
    //   swing_distribution =  std::uniform_real_distribution<double> (deg2rad(0), deg2rad(25));
    //   phase_speed_distribution =  std::uniform_real_distribution<double> (0.3, 1.2);
    //   // phase_speed_distribution =  std::uniform_real_distribution<double> (0.3, 1.8);
    //   noise_distribution =  std::uniform_real_distribution<double> (-deg2rad(1), deg2rad(1));
    //   anglularvel_distribution = std::uniform_real_distribution<double> (deg2rad(-20), deg2rad(20));

    //   // anglularvel_distribution = std::uniform_real_distribution<double> (deg2rad(-27), deg2rad(27));
    //   // change_distribution = std::uniform_int_distribution<int>(7, 50);
    // }
    // else if(update >= curr6)
    // {
    //   Ae_distribution =  std::uniform_real_distribution<double> (0.05, 0.18);
    //   height_distribution =  std::uniform_real_distribution<double> (0.22, 0.34);
    //   swing_distribution =  std::uniform_real_distribution<double> (deg2rad(0), deg2rad(25));
    //   phase_speed_distribution =  std::uniform_real_distribution<double> (0.3, 1.0);
    //   anglularvel_distribution = std::uniform_real_distribution<double> (deg2rad(-35), deg2rad(35));
    //   noise_distribution =  std::uniform_real_distribution<double> (-deg2rad(15), deg2rad(15));
    //   change_distribution = std::uniform_int_distribution<int>(15, 40);
    // }
  }

  void DRcurriculum(int update)
  {
    int curr1 = 1000;
    int curr2 = 2000;
    int curr3 = 5000;
    int curr4 = 7000;
    int curr5 = 10000;

    // Here implement the final distribution
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
      latency_distribution = std::uniform_real_distribution<double> (0.95, 1.5);

    }
    else if(update > curr5)
    {
      mass_distribution = std::uniform_real_distribution<double> (0.75, 1.2);
      pgain_distribution = std::uniform_real_distribution<double> (0.7, 1.3);
      dgain_distribution = std::uniform_real_distribution<double> (0.7, 1.3);
      latency_distribution = std::uniform_real_distribution<double> (0.95, 1.6);
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

    // root_imit_reward = exp(-root_imit_scale * root_imit_err);
    root_imit_reward = 1;
    joint_imit_reward = exp(-joint_imit_scale * joint_imit_err);
    ori_imit_reward = exp(-ori_imit_scale * ori_imit_err);
    end_imit_reward = exp(-end_imit_scale * end_imit_err);
    freq_reward = exp(- freq_scale * freq_err);

    reward = main_w * root_imit_reward * joint_imit_reward * ori_imit_reward * end_imit_reward 
      + freq_w * freq_reward;
    if(isnan(reward)) return 0;
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

  float orientationImitationErr()
  {
    float err = 0.;
    float epsilon = 1e-6;
    // Eigen::Matrix3d rot_m1;
    // Eigen::Matrix3d rot_m2;
    // rot_m1  = Eigen::Quaterniond(gc_[3],gc_[4],gc_[5],gc_[6]);
    // rot_m2  = Eigen::Quaterniond(gc_target[0],gc_target[1],gc_target[2],gc_target[3]);
    // Eigen::Vector3d current_dir = rot_m1 * Eigen::Vector3d::UnitX();
    // Eigen::Vector3d next_dir = rot_m2 * Eigen::Vector3d::UnitX();
    // double value = (current_dir[0] * next_dir[0] + current_dir[1] * next_dir[1]) / 
    //   (std::sqrt(pow(current_dir[0], 2) + pow(current_dir[1], 2)) * std::sqrt(pow(next_dir[0], 2) + pow(next_dir[1], 2)));
    // value = std::min(1.0 - epsilon, value);
    // value = std::max(-1.0 + epsilon, value);
    // float angle = acos(value);
    // err = angle * angle;
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
      err += (footPositionR.e() - footPositionA.e()).squaredNorm();
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
  std::vector<GraphicObject>* anymalVisual_;
  std::vector<GraphicObject>* ref_graphics;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, torque_, gc_dummy_init;
  Eigen::VectorXd gc_prev, gc_target, gc_reference;
  double desired_fps_ = 30.;
  int visualizationCounter_=0;
  float effort_limit = 33.5 * 0.90;
  Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_;
  Eigen::VectorXd obDouble_, obScaled_;
  std::set<size_t> footIndices_;

  int current_count;
  Eigen::VectorXd prev_vel, joint_acc;
  std::vector<std::deque<double>> torque_sequence;
  RootPredictor mRP;
  TrajectoryGenerator mTG;

  std::vector<Eigen::VectorXd> action_history;
  std::vector<Eigen::VectorXd> state_history;
  std::vector<Eigen::VectorXd> reference_history;
  std::vector<Eigen::VectorXd> trajectories;
  std::vector<Eigen::Vector3d> root_trj;

  int max_cnt = 452; int human_cnt;
  bool test_mode = false;
  int foot_order[4] = {0,1,3,2};

  // Reward Shapers
  double main_w, freq_w;
  double root_imit_scale, joint_imit_scale, ori_imit_scale, end_imit_scale, freq_scale;
  double root_imit_reward, joint_imit_reward, ori_imit_reward, end_imit_reward, freq_reward;

  // Kp Settings
  double init_pgain = 100;
  double init_dgain = 2;
  Eigen::VectorXd jointPgain, jointDgain;
  double phi, phase_speed;
  double ang_vel, angle_cummul;
  std::vector<int> contact_feet;

  // Imitation Learning
  std::uniform_int_distribution<int> number_distribution;
  std::uniform_real_distribution<double> Ae_distribution;
  std::uniform_real_distribution<double> height_distribution;
  std::uniform_real_distribution<double> swing_distribution;
  std::uniform_real_distribution<double> phase_speed_distribution;
  std::uniform_real_distribution<double> noise_distribution;
  std::uniform_real_distribution<double> anglularvel_distribution;
  std::uniform_int_distribution<int> initial_distribution;
  std::uniform_int_distribution<int> change_distribution;

  // Parameters for domain randomization
  std::uniform_real_distribution<double> mass_distribution;
  std::uniform_real_distribution<double> pgain_distribution;
  std::uniform_real_distribution<double> dgain_distribution;
  std::uniform_real_distribution<double> friction_distribution; 
  std::uniform_real_distribution<double> latency_distribution; 

  float default_friction_coeff;
  std::vector<double> original_mass;
  
  std::vector<Eigen::VectorXd> human_point_list;
  Eigen::Vector4i currentContact;
  Eigen::Vector4i prevContact;
  Eigen::VectorXd kin_gc_prev;
  float init_foot;
  std::ofstream writeRobot;
};

}