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
#include "../include/AnalyticIK.hpp"

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
    vis_ref = new raisim::ArticulatedSystem(resourceDir_+"/a1/a1/a1.urdf");       
    tracking_ball = new raisim::Sphere(0.0001, 0.0001);
    nBalls = 0;
    for(int i = 0 ; i < nBalls; ++i)
    {
      thrown_balls.push_back(world_->addSphere(0.05, 0.1));
    }

    alien_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    ground = world_->addGround(0, "floor");
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
    gc_prev.setZero(gcDim_); gc_target.setZero(4 + nJoints_);

    /// this is nominal configuration of anymal
    gc_init_ <<  0, 0, 0.27, 1.0, 0.0, 0.0, 0.0, 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99);
    alien_->setState(gc_init_, gv_init_);
    ref_dummy->setState(gc_init_, gv_init_);
    vis_ref->setState(gc_init_, gv_init_);
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

    obDim_ = 164;
    actionDim_ = nJoints_;

    throwingFlag = false;
    actionStd_.setZero(nJoints_);  actionMean_.setZero(nJoints_);
    obMean_.setZero(obDim_); obStd_.setZero(obDim_);
    prev_vel.setZero(nJoints_); joint_acc.setZero(nJoints_);

    float ah = deg2rad(30), at = deg2rad(30), ac = deg2rad(30);
    float mh = 0, mt = M_PI /2, mc = -1.80;
    actionStd_ << ah, at, ac, ah, at, ac, ah, at, ac, ah, at, ac;
    // actionMean_ << mh, mt, mc, mh, mt, mc, mh, mt, mc, mh, mt, mc;

    freq_w = 0.1;
    main_w = 1.0 - freq_w;

    joint_imit_scale = 10.0;
    ori_imit_scale = 1.0;
    end_imit_scale = 10.0;
    support_scale = 10.0;
    freq_scale = 1e-4;

    current_count = 1;
    change_count = 30;
    ori_count = 30;
    throw_count = 25;

    mCOM = Eigen::Vector3d(0, 0, 0.27);
    mCOM[2] = 0;
    mCOP = mCOM;
    mDCM = mCOM;

    hip_distribution = std::uniform_real_distribution<double> (deg2rad(-5),deg2rad(5));
    thigh_distribution = std::uniform_real_distribution<double> (deg2rad(80), deg2rad(85));
    calf_distribution = std::uniform_real_distribution<double>(deg2rad(-140), deg2rad(-130));
    x_distribution = std::uniform_real_distribution<double>(deg2rad(-1), deg2rad(1));
    y_distribution = std::uniform_real_distribution<double>(deg2rad(-1), deg2rad(1));
    z_distribution = std::uniform_real_distribution<double>(deg2rad(-1), deg2rad(1));

    balllocation_distribution = std::uniform_real_distribution<double>(0, 2 * M_PI);
    ballspeed_distribution = std::uniform_real_distribution<double>(7, 8);
    theta1_distribution = std::uniform_real_distribution<double>(deg2rad(-10), deg2rad(10));
    theta2_distribution = std::uniform_real_distribution<double>(deg2rad(-10), deg2rad(10));
    rad_distribution = std::uniform_real_distribution<double>(0.0, 0.04);
    phi_distribution = std::uniform_real_distribution<double>(-M_PI, M_PI);

    hstep_distribution = std::uniform_real_distribution<double>(0.08, 0.17);
    dstep_distribution = std::uniform_real_distribution<double>(0.01, 0.05);

    change_distribution = std::uniform_int_distribution<int>(30, 30);
    xflag_distribution = std::uniform_int_distribution<int>(0, 1);
    yflag_distribution = std::uniform_int_distribution<int>(0, 1);
    zflag_distribution = std::uniform_int_distribution<int>(0, 1);
    reminder_distribution = std::uniform_int_distribution<int>(0, 3);

    gui::rewardLogger.init({"Angle_reward", "Ori_reward", "Support_reward", "End_reward", "Freq_reward"});

    /// indices of links that should not make contact with ground
    footIndices_.insert(alien_->getBodyIdx("FL_calf"));
    footIndices_.insert(alien_->getBodyIdx("FR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RL_calf"));
    start = true;

    /// visualize if it is the first environment
    if (visualizable_) {
      auto vis = raisim::OgreVis::get();      
      auto& list = raisim::OgreVis::get()->getVisualObjectList();

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
      for(int i = 0; i < nBalls; ++i)
        vis->createGraphicalObject(thrown_balls[i], "ball " + std::to_string(i), "coral");
      tracking_graphics = vis->createGraphicalObject(tracking_ball, "Tracker", "None");

      desired_fps_ = 1.0/ control_dt_;
      vis->setDesiredFPS(desired_fps_);
      vis->select(tracking_graphics->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(-2.1), Ogre::Radian(-1.1), 4, true);
      vis->addVisualObject("cop", "sphereMesh", "aqua_marine", {0.02, 0.02, 0.02}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("com", "sphereMesh", "medium_purple", {0.02, 0.02, 0.02}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("dcm", "sphereMesh", "dark_magenta", {0.02, 0.02, 0.02}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
    }
  }

  ~ENVIRONMENT() final = default;

  void init() final { }

  void kinect_init() final{}

  void reset() final
  {
    alien_->setState(gc_init_, gv_init_);
    gc_prev = gc_init_;
    gc_reference = gc_init_;
    gc_reference[1] += 0.7;
    gc_target = gc_init_.tail(nJoints_ + 4);

    prev_vel.setZero();
    joint_acc.setZero();

    mCOM = Eigen::Vector3d(0, 0, 0.27);
    height1 = mCOM[2];
    height2 = mCOM[2];
    mCOM[2] = 0;
    mCOP = mCOM;
    mDCM = mCOM;
    pastOri = Eigen::Quaterniond(1,0,0,0);
    futureOri = Eigen::Quaterniond(1,0,0,0);
    armChangeFlag = false;
    rearSteppingFlag = false;
    std::random_device rd;
    std::mt19937 generator(rd());
    int reminder = reminder_distribution(generator);
    reminderFlag = (reminder == 0);

    joint_imit_reward = 0.0;
    ori_imit_reward = 0.0;
    end_imit_reward = 0.0;
    support_reward = 0.0;
    freq_reward = 0.0;

    current_count = 1;
    desired_angleH = gc_init_[7];
    desired_angleT = gc_init_[8];
    desired_angleC = gc_init_[9];
    desired_angleHC = desired_angleH;
    desired_angleTC = desired_angleT;
    desired_angleCC = desired_angleC;
    desired_angleHF = desired_angleH;
    desired_angleTF = desired_angleT;
    desired_angleCF = desired_angleC;
    // curriculumUpdate_init(25000);

    for(int i = 0; i <nBalls ; ++i){
      thrown_balls[i]->setPosition(100, 100, 100+2*i);
      thrown_balls[i]->setLinearVelocity(raisim::Vec<3>{0.,0.,0.});
    }
    updateObservation();
    tracking_ball->setPosition(0, 0.5, 0.3);

    if(visualizable_){
      gui::rewardLogger.clean();
      auto& list = raisim::OgreVis::get()->getVisualObjectList();
      list["com"].setPosition(mCOM);
      list["dcm"].setPosition(mDCM);
      list["cop"].setPosition(mCOP);
      // resetPolygon();
    }
  }

  void updatedSupportPolygon()
  {
    int cnt_feet = 0;
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

  void update_random_leaning_angle()
  {
    std::random_device rd;
    std::mt19937 generator(rd());
    desired_angleHF = hip_distribution(generator);
    desired_angleTF = thigh_distribution(generator);
    desired_angleCF = calf_distribution(generator);
  }
  
  void update_full_leaning_anlge()
  {
    std::random_device rd;
    std::mt19937 generator(rd());
    auto temphip_distribution = std::uniform_real_distribution<double> (-deg2rad(45), deg2rad(45));
    auto tempthigh_distribution = std::uniform_real_distribution<double> (-deg2rad(55), deg2rad(235));
    auto tempcalf_distribution = std::uniform_real_distribution<double>(-deg2rad(150), -deg2rad(50));
    desired_angleHF = temphip_distribution(generator);
    desired_angleTF = tempthigh_distribution(generator);
    desired_angleCF = tempcalf_distribution(generator);
  }

  void interpolate_desired()
  {
    int interpol = (current_count) % change_count;
    float percentile = (float) interpol / (float) change_count;
    desired_angleH = desired_angleHC * (1.0 - percentile) + desired_angleHF * percentile;
    desired_angleT = desired_angleTC * (1.0 - percentile) + desired_angleTF * percentile;
    desired_angleC = desired_angleCC * (1.0 - percentile) + desired_angleCF * percentile;
  }

  void ballthrowing()
  {
    std::random_device rd;
    std::mt19937 generator(rd());
    // Root are scaled to have + 2cm gap
    float x_root = 0.269, y_root = 0.196;
    float regionTheta1 = atan(y_root / x_root);
    float regionTheta2 = M_PI - regionTheta1;
    float regionTheta3 = M_PI + regionTheta1;
    float regionTheta4 = - regionTheta1;

    // 1) decide the hitting point and set the position
    float current_theta = balllocation_distribution(generator);
    float x_contact, y_contact;
    Eigen::Vector3d base_dir;
    Eigen::Vector3d rot_dir;
    if(current_theta == M_PI /2) {
      x_contact = 0; y_contact = y_root;
      base_dir = Eigen::Vector3d(0, -1, 0);
      rot_dir = Eigen::Vector3d(1, 0, 0);
    }
    else if(current_theta == 3 * M_PI /2) {
      x_contact = 0; y_contact = -y_root;
      base_dir = Eigen::Vector3d(0, 1, 0);
      rot_dir = Eigen::Vector3d(1, 0, 0);
    }
    else if(current_theta >= regionTheta1 && current_theta < regionTheta2)
    {
      // Region 2
      y_contact = y_root;
      x_contact = y_contact / tan(current_theta);
      base_dir = Eigen::Vector3d(0, -1, 0);
      rot_dir = Eigen::Vector3d(1, 0, 0);
    }
    else if(current_theta >= regionTheta2 && current_theta < regionTheta3)
    {
      // Region 3
      x_contact = - x_root;
      y_contact = x_contact * tan(current_theta);
      base_dir = Eigen::Vector3d(1, 0, 0);
      rot_dir = Eigen::Vector3d(0, 1, 0);
    }
    else if(current_theta >= regionTheta3 && current_theta < regionTheta4 + M_PI * 2)
    {
      // Region 4
      y_contact = - y_root;
      x_contact = y_contact / tan(current_theta);
      base_dir = Eigen::Vector3d(0, 1, 0);
      rot_dir = Eigen::Vector3d(1, 0, 0);
    }
    else
    {
      // Region 1
      x_contact = x_root;
      y_contact = x_contact * tan(current_theta);
      base_dir = Eigen::Vector3d(-1, 0, 0);
      rot_dir = Eigen::Vector3d(0, 1, 0);
    }

    Eigen::Matrix3d current_ori = alien_->getBaseOrientation().e();
    Eigen::Vector3d current_pose = alien_->getBasePosition().e();
    Eigen::Vector3d ballInitPose = current_pose + current_ori * Eigen::Vector3d(x_contact, y_contact, 0);
    thrown_balls[0]->setPosition(ballInitPose);
    // 2) decide speed
    float ballspeed = ballspeed_distribution(generator);
    // 3) decide direction
    float comp_dir = theta1_distribution(generator);
    float z_dir = theta2_distribution(generator);
    Eigen::Matrix3d rotator;
    rotator = Eigen::AngleAxisd(comp_dir, rot_dir) * Eigen:: AngleAxisd(z_dir, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d LinSpeed = ballspeed * rotator * base_dir;
    thrown_balls[0]->setLinearVelocity(LinSpeed);
  }

  void land_arm(int NthArm)
  {
    std::random_device rd;
    std::mt19937 generator(rd());
    float rad = rad_distribution(generator);
    phi = phi_distribution(generator);
    ref_dummy->setGeneralizedCoordinate(gc_init_);
    raisim::Vec<3> original_landing;
    ref_dummy->getFramePosition(FR_FOOT+3 * NthArm, original_landing);
    Eigen::Vector3d target_land = original_landing.e() + Eigen::Vector3d(rad * cos(phi), rad * sin(phi),0);
    Eigen::Vector3d sol;
    // gc_ = alien_->getGeneralizedCoordinate().e();
    // ref_dummy->setGeneralizedCoordinate(gc_);
    if(NthArm%2 == 0)
    {
      AnalyticRIK* rik = new AnalyticRIK(ref_dummy, NthArm/2);
      rik->setReady();
      rik->setCurrentPose(gc_init_);
      rik->setTarget(target_land);
      rik->solveIK();
      sol = rik->getSolution();
      rik->reset();
      delete rik;
    }
    else
    {
      AnalyticLIK* lik = new AnalyticLIK(ref_dummy, NthArm/2);
      lik->setReady();
      lik->setCurrentPose(gc_init_);
      lik->setTarget(target_land);
      lik->solveIK();
      sol = lik->getSolution();
      lik->reset();
      delete lik;
    }
    desired_angleHF = sol[0];
    desired_angleTF = sol[1];
    desired_angleCF = sol[2];
  }

  void updateStepParam()
  {
    // d refers to half of the desired step width
    std::random_device rd;
    std::mt19937 generator(rd());
    d_step = dstep_distribution(generator);
    h_step = hstep_distribution(generator);
    phi = phi_distribution(generator);
    // phi = 0.0;
    ref_dummy->getFramePosition(RR_FOOT+3 * rear_dir, original_landing_rear);
  }

  void updateRearFootStep()
  {
    float elltheta = ((current_count -1) % change_count) * M_PI / (change_count);
    float ellrad = d_step * h_step / std::sqrt(pow(h_step * cos(elltheta), 2)  + pow(d_step* sin(elltheta), 2));
    Eigen::Vector3d target_land = original_landing_rear.e() + Eigen::Vector3d((d_step - ellrad * cos(elltheta)) * cos(phi), (d_step - ellrad * cos(elltheta)) * sin(phi), ellrad * sin(elltheta));
    Eigen::Vector3d sol;
    if(rear_dir == 0)
    {
      AnalyticRIK* rik = new AnalyticRIK(ref_dummy, 1);
      rik->setReady();
      rik->setCurrentPose(gc_init_);
      rik->setTarget(target_land);
      rik->solveIK();
      sol = rik->getSolution();
      rik->reset();
      delete rik;
    }
    else
    {
      AnalyticLIK* lik = new AnalyticLIK(ref_dummy, 1);
      lik->setReady();
      lik->setCurrentPose(gc_init_);
      lik->setTarget(target_land);
      lik->solveIK();
      sol = lik->getSolution();
      lik->reset();
      delete lik;
    }
    // std::cout << (target_land - original_landing_rear.e()).transpose() <<std::endl;
    gc_target[10 + 3 * rear_dir] = sol[0];
    gc_target[11 + 3 * rear_dir] = sol[1];
    gc_target[12 + 3 * rear_dir] = sol[2];
  }

  void updateFinalFootStep()
  {
    float elltheta = M_PI;
    float ellrad = d_step * h_step / std::sqrt(pow(h_step * cos(elltheta), 2)  + pow(d_step* sin(elltheta), 2));
    Eigen::Vector3d target_land = original_landing_rear.e() + Eigen::Vector3d((d_step - ellrad * cos(elltheta)) * cos(phi), (d_step - ellrad * cos(elltheta)) * sin(phi), ellrad * sin(elltheta));
    Eigen::Vector3d sol;
    if(rear_dir == 0)
    {
      AnalyticRIK* rik = new AnalyticRIK(ref_dummy, 1);
      rik->setReady();
      rik->setCurrentPose(gc_init_);
      rik->setTarget(target_land);
      rik->solveIK();
      sol = rik->getSolution();
      rik->reset();
      delete rik;
    }
    else
    {
      AnalyticLIK* lik = new AnalyticLIK(ref_dummy, 1);
      lik->setReady();
      lik->setCurrentPose(gc_init_);
      lik->setTarget(target_land);
      lik->solveIK();
      sol = lik->getSolution();
      lik->reset();
      delete lik;
    }
    // std::cout << ellrad <<std::endl;
    // std::cout << (target_land - original_landing_rear.e()).transpose() <<std::endl;
    // std::cout << target_land.transpose() <<std::endl;
    gc_target[10 + 3 * rear_dir] = sol[0];
    gc_target[11 + 3 * rear_dir] = sol[1];
    gc_target[12 + 3 * rear_dir] = sol[2];
  }

  void update_body_angle()
  {
    int xflag, yflag, zflag;
    float x, y, z;
    std::random_device rd;
    std::mt19937 generator(rd());
    xflag = xflag_distribution(generator);
    yflag = yflag_distribution(generator);
    zflag = zflag_distribution(generator);
    x = x_distribution(generator);
    y = y_distribution(generator);
    z = z_distribution(generator);
    futureOri = Eigen::AngleAxisd(x * xflag, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(y * yflag, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(z * zflag,Eigen:: Vector3d::UnitZ());
    pastOri = Eigen::Quaterniond(gc_target[0], gc_target[1], gc_target[2], gc_target[3]);
  }

  void interpolate_body_angle()
  {
    float interpol = float((current_count)%ori_count) / float(ori_count);
    // std::cout<< interpol << std::endl;
    Eigen::Quaterniond current = pastOri.slerp(interpol, futureOri).normalized();
    // std::cout << current.coeffs() <<  std::endl;
    gc_target[0] = current.w();
    gc_target[1] = current.x();
    gc_target[2] = current.y();
    gc_target[3] = current.z();
  }

  void change_manipulation_arm(int side)
  {
    gc_target[4 + 3 * side] = desired_angleHF;
    gc_target[5 + 3 * side] = desired_angleTF;
    gc_target[6 + 3 * side] = desired_angleCF;
    desired_angleHF = gc_target[7 - 3* side];
    desired_angleTF = gc_target[8 - 3* side];
    desired_angleCF = gc_target[9 - 3* side];
    desired_angleH = gc_target[7 - 3* side];
    desired_angleT = gc_target[8 - 3* side];
    desired_angleC = gc_target[9 - 3* side];
    armChangeFlag = (side == 0);
    current_count += 10;
  }

  Eigen::VectorXd safePDtarget(Eigen::VectorXd pTarget)
  {
    auto jointLimits = alien_->getJointLimits();
    for(int i = 0; i < nJoints_; ++i)
    {
      if(jointLimits[6 + i][0] > pTarget[i]) pTarget[i] = jointLimits[6 + i][0];
      else if(jointLimits[6 + i][1] < pTarget[i]) pTarget[i] = jointLimits[6 + i][1];
    }
    return pTarget;
  }

  float stepRef() final
  {    
    return 0;
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    if(rearSteppingFlag){
      updateRearFootStep();
    }
    else{
      if(armChangeFlag)
      {
        gc_target[7] = desired_angleH;
        gc_target[8] = desired_angleT;
        gc_target[9] = desired_angleC;
      }
      else
      {
        gc_target[4] = desired_angleH;
        gc_target[5] = desired_angleT;
        gc_target[6] = desired_angleC;
      }
    }

    Eigen::VectorXd currentAction= action.cast<double>();
    pTarget12_ = currentAction;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += gc_target.tail(nJoints_);
    pTarget_.tail(nJoints_) = safePDtarget(pTarget12_);
    if(visualizable_) setVisRef();
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
    alien_->getState(gc_, gv_);
    updatedSupportPolygon();
    updateCOMDCM();
    updateCOP();
    updateObservation();
    tracking_ball->setPosition(gc_[0], 0.5, 0.3);

    float totalReward = CalcReward();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("Angle_reward", joint_imit_reward);
      gui::rewardLogger.log("Ori_reward", ori_imit_reward);
      gui::rewardLogger.log("Support_reward", support_reward);
      gui::rewardLogger.log("End_reward", end_imit_reward);
      gui::rewardLogger.log("Freq_reward", freq_reward);
      auto vis = raisim::OgreVis::get();

      vis->select(anymalVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(-0.40), Ogre::Radian(-0.6), 4, true);
    }

    if(current_count%throw_count == 0 && throwingFlag)
    {
      ballthrowing();
    }

    if(current_count%change_count == 0){
      if(current_count == change_count * 2)
        update_full_leaning_anlge();
      else if(current_count == change_count * 3)
        land_arm(0);
      else if(current_count == change_count * 4)
      { 
        // if(reminderFlag){
          change_manipulation_arm(0);
          rearSteppingFlag = false;
        // }
        // else{
          // rear_dir = 0;
          // updateStepParam();
          // rearSteppingFlag = true;
        // }
      }
      // else if(current_count == change_count * 4)
      // {
      //   if(rearSteppingFlag){
      //     updateFinalFootStep();
      //     rearSteppingFlag = false;
      //   }
      //   else
      //     update_full_leaning_anlge();
      // }

      else if(current_count == change_count * 5 ||current_count == change_count * 6)
        update_random_leaning_angle();
      else if(current_count == change_count * 7)
      {
        if(reminderFlag)
          land_arm(1);
        else
          update_random_leaning_angle();
        // else{
        //   rear_dir = 1;
        //   updateStepParam();
        //   rearSteppingFlag = true;        
        // }
      }
      else if(current_count == change_count * 8)
      { 
        if(reminderFlag)
          change_manipulation_arm(1);
        else
          update_random_leaning_angle();
      }
      // else if(current_count == change_count * 7)
      // {
      //   if(reminderFlag)
      //     change_manipulation_arm(1);
      //   else{
      //     updateFinalFootStep();
      //     rearSteppingFlag = false;
      //   }
      // }
      // // else if(current_count == change_count * 8)
      // //   land_arm(0);
      // // else if(current_count == change_count * 9)
      // // { 
      // //   change_manipulation_arm(0);
      // // }
      else
        update_full_leaning_anlge();

      desired_angleHC = desired_angleH;
      desired_angleTC = desired_angleT;
      desired_angleCC = desired_angleC;
    }

    interpolate_desired();
    current_count++;

    if(!mStopFlag) 
      return totalReward;
    else{
      // if (visualizable_) resetPolygon();
      return -1E10;
    }
  }

  float stepTest(const Eigen::Ref<EigenVec>& action) final {
    if(start)
    {
      initSupportPolygon();
      start = false;
    }
    // curriculumUpdate_init(25000);
    // curriculumUpdate_init(30000);
    /// action scaling
    if(rearSteppingFlag){
      updateRearFootStep();
    }
    else{
      if(armChangeFlag)
      {
        gc_target[7] = desired_angleH;
        gc_target[8] = desired_angleT;
        gc_target[9] = desired_angleC;
      }
      else
      {
        gc_target[4] = desired_angleH;
        gc_target[5] = desired_angleT;
        gc_target[6] = desired_angleC;
      }
    }

    Eigen::VectorXd currentAction_PD = action.cast<double>();
    Eigen::VectorXd currentAction(nJoints_);
    currentAction = currentAction_PD.head(nJoints_);

    pTarget12_ = currentAction;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += gc_target.tail(12);
    pTarget_.tail(nJoints_) = pTarget12_;
    if(visualizable_) setVisRef();
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
    alien_->getState(gc_, gv_);
    updatedSupportPolygon();
    updateCOMDCM();
    updateCOP();
    visualizeSupportPolygon();
    updateObservation();
    tracking_ball->setPosition(gc_[0], 0.5, 0.3);

    float totalReward = CalcReward();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("Angle_reward", joint_imit_reward);
      gui::rewardLogger.log("Ori_reward", ori_imit_reward);
      gui::rewardLogger.log("Support_reward", support_reward);
      gui::rewardLogger.log("End_reward", end_imit_reward);
      gui::rewardLogger.log("Freq_reward", freq_reward);
      // auto vis = raisim::OgreVis::get();

      // vis->select(anymalVisual_->at(0), false);
      // vis->getCameraMan()->setYawPitchDist(Ogre::Radian(-2.1), Ogre::Radian(-1.1), 4, true);
    }
    if(current_count%change_count == 0){
      if(current_count == change_count * 2)
        update_full_leaning_anlge();
      else if(current_count == change_count * 3)
        land_arm(0);
      else if(current_count == change_count * 4)
      { 
        // if(reminderFlag){
        change_manipulation_arm(0);
        rearSteppingFlag = false;
      }
      else if(current_count == change_count * 5 ||current_count == change_count * 6)
        update_random_leaning_angle();
      else if(current_count == change_count * 7)
      {
        land_arm(1);
      }
      else if(current_count == change_count * 8)
      { 
        // if(reminderFlag){
        change_manipulation_arm(1);
      }
      else
        update_full_leaning_anlge();

      desired_angleHC = desired_angleH;
      desired_angleTC = desired_angleT;
      desired_angleCC = desired_angleC;
    }

    // if(current_count%throw_count == 0 && throwingFlag)
    // {
    //   ballthrowing();
    // }
    // if(current_count%change_count == 0){
    //   if(current_count == change_count * 1)
    //     update_full_leaning_anlge();
    //   else if(current_count == change_count * 2)
    //     land_arm(0);
    //   else if(current_count == change_count * 3)
    //   { 
    //     if(reminderFlag){
    //       change_manipulation_arm(0);
    //       rearSteppingFlag = false;
    //     }
    //     else{
    //       rear_dir = 0;
    //       updateStepParam();
    //       rearSteppingFlag = true;
    //     }
    //   }
    //   else if(current_count == change_count * 4)
    //   {
    //     if(rearSteppingFlag){
    //       updateFinalFootStep();
    //       rearSteppingFlag = false;
    //     }
    //     else
    //       update_random_leaning_angle();
    //   }
    //   else if(current_count == change_count * 5)
    //   {
    //     if(reminderFlag)
    //       land_arm(1);
    //     else{
    //       rear_dir = 1;
    //       updateStepParam();
    //       rearSteppingFlag = true;        
    //     }
    //   }
    //   else if(current_count == change_count * 6)
    //   {
    //     if(reminderFlag)
    //       change_manipulation_arm(1);
    //     else{
    //       updateFinalFootStep();
    //       rearSteppingFlag = false;
    //     }
    //   }
    //   else if(current_count == change_count * 8)
    //     land_arm(0);
    //   else if(current_count == change_count * 9)
    //   { 
    //     change_manipulation_arm(0);
    //   }
    //   else
    //     update_random_leaning_angle();

    //   desired_angleHC = desired_angleH;
    //   desired_angleTC = desired_angleT;
    //   desired_angleCC = desired_angleC;
    // }

    interpolate_desired();
    current_count++;

    if(!mStopFlag) 
      return totalReward;
    else{
      // if (visualizable_) resetPolygon();
      return -1E10;
    }
  }

  void curriculumUpdate(int update) final {
    int curr1 = 500;
    int curr2 = 1000;
    int curr3 = 2000;
    curriculumUpdate_init(update);
  } 

  void curriculumUpdate_init(int update) {
    int curr1 = 500;
    int curr2 = 1000;
    int curr3 = 2000;
    int curr4 = 2500;
    int curr5 = 3500;
    int curr6 = 4000; 
    int curr7 = 5000; 
    int curr8 = 6000; 
    int curr9 = 8000; 
    int curr10 = 10000; 
    int curr11 = 17000; 
    int curr12 = 20000; 
    int curr13 = 22000; 
    int curr14 = 24000; 
    int curr15 = 26000; 

    if(update >= curr1 && update <curr2) 
    {
      hip_distribution = std::uniform_real_distribution<double> (-deg2rad(10), deg2rad(10));
      thigh_distribution = std::uniform_real_distribution<double> (deg2rad(0), deg2rad(85));
      calf_distribution = std::uniform_real_distribution<double>(-deg2rad(110), -deg2rad(85));
    }
    else if(update >= curr2 && update <curr3) 
    {
      hip_distribution = std::uniform_real_distribution<double> (-deg2rad(15), deg2rad(15));
      thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(5), deg2rad(100));
      calf_distribution = std::uniform_real_distribution<double>(-deg2rad(120), -deg2rad(75));
      change_distribution = std::uniform_int_distribution<int>(20, 30);
    }
    else if(update >= curr3 && update <curr4) 
    {
      hip_distribution = std::uniform_real_distribution<double> (-deg2rad(20), deg2rad(20));
      thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(15), deg2rad(150));
      calf_distribution = std::uniform_real_distribution<double>(-deg2rad(125), -deg2rad(70));
    }
    else if(update >= curr4 && update <curr5) 
    {
      hip_distribution = std::uniform_real_distribution<double> (-deg2rad(30), deg2rad(30));
      thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(30), deg2rad(180));
      calf_distribution = std::uniform_real_distribution<double>(-deg2rad(130), -deg2rad(60));
      change_distribution = std::uniform_int_distribution<int>(10, 40);
    }
    else if(update >= curr5 && update <curr6) 
    {
      hip_distribution = std::uniform_real_distribution<double> (-deg2rad(40), deg2rad(40));
      thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(45), deg2rad(200));
      calf_distribution = std::uniform_real_distribution<double>(-deg2rad(140), -deg2rad(55));
      // change_distribution = std::uniform_int_distribution<int>(6, 45);
    }
    else if(update >= curr6 ) 
    {
      hip_distribution = std::uniform_real_distribution<double> (-deg2rad(45), deg2rad(45));
      thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(55), deg2rad(235));
      calf_distribution = std::uniform_real_distribution<double>(-deg2rad(150), -deg2rad(50));
      change_distribution = std::uniform_int_distribution<int>(10, 40);
    }
    // else if(update >= curr7 && update <curr8) 
    // {
    //   throwingFlag = true;
    // }
    // else if(update >= curr8 && update <curr9) 
    // {
    //   ballspeed_distribution = std::uniform_real_distribution<double>(7, 10);
    //   theta1_distribution = std::uniform_real_distribution<double>(deg2rad(-15), deg2rad(15));
    //   theta2_distribution = std::uniform_real_distribution<double>(deg2rad(-15), deg2rad(15));
    // }
    // else if(update >= curr9 && update <curr10) 
    // {
    //   ballspeed_distribution = std::uniform_real_distribution<double>(7, 11);
    //   theta1_distribution = std::uniform_real_distribution<double>(deg2rad(-20), deg2rad(20));
    //   theta2_distribution = std::uniform_real_distribution<double>(deg2rad(-20), deg2rad(20));

    // }
    // else if(update >= curr10 && update <curr11) 
    // {
    //   hip_distribution = std::uniform_real_distribution<double> (-deg2rad(45), deg2rad(45));
    //   thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(55), deg2rad(235));
    //   calf_distribution = std::uniform_real_distribution<double>(-deg2rad(150), -deg2rad(50));
    //   change_distribution = std::uniform_int_distribution<int>(10, 40);
    //   ballspeed_distribution = std::uniform_real_distribution<double>(7, 13);
    //   theta1_distribution = std::uniform_real_distribution<double>(deg2rad(-30), deg2rad(30));
    //   theta2_distribution = std::uniform_real_distribution<double>(deg2rad(-30), deg2rad(30));
    //   throwingFlag = true;
    // }
    // else if(update >= curr11 && update <curr12) 
    // {
    //   // Arm Chaning
    //   throwingFlag = false;
    // }
    // else if(update >= curr12 && update <curr13) 
    // {
    //   hip_distribution = std::uniform_real_distribution<double> (-deg2rad(15), deg2rad(15));
    //   thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(5), deg2rad(100));
    //   calf_distribution = std::uniform_real_distribution<double>(-deg2rad(120), -deg2rad(75));
    // }
    // else if(update >= curr13 && update <curr14) 
    // {
    //   hip_distribution = std::uniform_real_distribution<double> (-deg2rad(30), deg2rad(30));
    //   thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(30), deg2rad(180));
    //   calf_distribution = std::uniform_real_distribution<double>(-deg2rad(130), -deg2rad(60));
    // }
    // else if(update >= curr14 && update < curr15)
    // {
    //   hip_distribution = std::uniform_real_distribution<double> (-deg2rad(45), deg2rad(45));
    //   thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(55), deg2rad(235));
    //   calf_distribution = std::uniform_real_distribution<double>(-deg2rad(150), -deg2rad(50));
    // }
    // else if(update >= curr15)
    // {
    //   ballspeed_distribution = std::uniform_real_distribution<double>(7, 13);
    //   theta1_distribution = std::uniform_real_distribution<double>(deg2rad(-30), deg2rad(30));
    //   theta2_distribution = std::uniform_real_distribution<double>(deg2rad(-30), deg2rad(30));
    //   throwingFlag = true;
    // }
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
      alien_->setGeneralizedCoordinate(gc_prev);
      gc_ = gc_prev;
      return 0;
    }

    joint_acc = (gv_.tail(nJoints_) - prev_vel) / control_dt_;
    prev_vel = gv_.tail(nJoints_);

    joint_imit_err = jointImitationErr();
    ori_imit_err = orientationImitationErr();
    // end_imit_err = endEffectorImitationErr();
    end_imit_err = endEffectorImitationErr2();
    // end_imit_err = endEffectorImitationErrLoss();
    support_err = CalcSupportHardErr();
    freq_err = CalcAccErr();

    joint_imit_reward = exp(-joint_imit_scale * joint_imit_err);
    ori_imit_reward = exp(-ori_imit_scale * ori_imit_err);
    end_imit_reward = exp(-end_imit_scale * end_imit_err);
    // end_imit_reward = exp(-end_imit_scale * end_imit_err);
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
    for(int i = 0; i < 3; ++i){
      if(rearSteppingFlag)
        err += pow((gc_target[10 + i + 3 * rear_dir] - gc_[13+ i + 3 * rear_dir]),2);
      else{
        if(armChangeFlag)
          err += pow((gc_target[7+ i] - gc_[10 + i]), 2);
        else
          err += pow((gc_target[4 + i] - gc_[7 + i]),2);
      }
    }
    return err;
  }

  float orientationImitationErr()
  {
    float err = 0.;
    // Use slerp
    Eigen::Quaterniond target(gc_target[0], gc_target[1], gc_target[2], gc_target[3]);
    Eigen::Quaterniond current(gc_[3],gc_[4],gc_[5],gc_[6]);
    err = pow(current.angularDistance(target), 2);
    return err;
  }

  float endEffectorImitationErr2()
  {
    float err = 0.;
    Eigen::VectorXd gc_extended = gc_;
    // gc_extended.segment(0,2) = gc_.segment(0, 2);
    gc_extended.tail(12) = gc_target.tail(12);
    ref_dummy->setGeneralizedCoordinate(gc_extended);
    // Use endeffector position of moving arm
    int i =0;
    if(armChangeFlag) i = 1;
    if(rearSteppingFlag) i = 2 + rear_dir;
    raisim::Vec<3> footPositionA;
    raisim::Vec<3> footPositionR;
    alien_->getFramePosition(FR_FOOT+ 3 * i, footPositionA);
    ref_dummy->getFramePosition(FR_FOOT+ 3 * i, footPositionR);
    err += (footPositionR.e() - footPositionA.e()).norm();
    return err;
  }

  float endEffectorImitationErr()
  {
    float err = 0.;
    Eigen::VectorXd gc_extended = gc_init_;
    // gc_extended.segment(0,2) = gc_.segment(0, 2);
    gc_extended.tail(16) = gc_target;
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
        float current = distToSegment(mCOM, current1, current1);
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
        float current = distToSegment(mCOM, current1, current1);
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
        float current = distToSegment(mCOM, current1, current1);
        if(min_d > current) min_d = current;
      }
      error += min_d * min_d;
    } 
    contact_feet.clear();
    contact_feet.shrink_to_fit();
    if(isnan(error)) return 100000;
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

    for(int i = 0; i < gc_.size(); ++i)
    {
      if(isnan(gc_[i])) return true;
    }

    /// if the contact body is not feet
    for(auto& contact: alien_->getContacts()){
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end()) {
        // if(world_->getObject(contact.getPairObjectIndex())->getObjectType() == HALFSPACE){
          return true;
        // }
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
  raisim::ArticulatedSystem* ref_dummy;
  raisim::ArticulatedSystem* vis_ref;
  raisim::Sphere* tracking_ball;
  std::vector<raisim::Sphere*> thrown_balls;
  std::vector<GraphicObject> * anymalVisual_;
  std::vector<GraphicObject>* tracking_graphics;
  raisim::Ground* ground;

  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, torque_, gc_dummy_init;
  Eigen::VectorXd gc_prev, gc_target, gc_reference;
  double desired_fps_ = 30.;
  int visualizationCounter_=0;
  float effort_limit = 33.5 * 0.90;
  Eigen::Vector3d mCOP, mCOM, mDCM;
  Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_;
  Eigen::VectorXd obDouble_, obScaled_;
  std::set<size_t> footIndices_;
  std::set<size_t> foot_hipIndices_;
  Eigen::VectorXd prev_vel, joint_acc;
  std::vector<std::deque<double>> torque_sequence;
  float d_step, h_step;

  int nBalls;
  bool start, throwingFlag, armChangeFlag, rearSteppingFlag, reminderFlag;
  float height1, height2;
  std::vector<int> contact_feet;
  int foot_order[4] = {0,1,3,2};
  int current_count, change_count, ori_count, throw_count;
  double desired_angleH, desired_angleT, desired_angleC;
  double desired_angleHC, desired_angleTC, desired_angleCC;
  double desired_angleHF, desired_angleTF, desired_angleCF;
  Eigen::Vector3d supportCentre;
  raisim::Vec<3> original_landing_rear;
  float poly_scale = 0.5;
  float phi;
  int rear_dir;

  int max_cnt = 302;

  std::vector<Eigen::VectorXd> action_history;
  std::vector<Eigen::VectorXd> state_history;
  std::vector<Eigen::VectorXd> reference_history;
  std::vector<Eigen::VectorXd> trajectories;

  // Reward Shapers
  double main_w, freq_w;
  double joint_imit_scale, ori_imit_scale, support_scale, end_imit_scale, freq_scale;
  double joint_imit_reward, ori_imit_reward, support_reward, end_imit_reward, freq_reward;

  Eigen::Quaterniond pastOri, futureOri;

  std::uniform_real_distribution<double> hip_distribution;
  std::uniform_real_distribution<double> thigh_distribution;
  std::uniform_real_distribution<double> calf_distribution;
  std::uniform_real_distribution<double> x_distribution;
  std::uniform_real_distribution<double> y_distribution;
  std::uniform_real_distribution<double> z_distribution;
  std::uniform_real_distribution<double> balllocation_distribution;
  std::uniform_real_distribution<double> ballspeed_distribution;
  std::uniform_real_distribution<double> theta1_distribution;
  std::uniform_real_distribution<double> theta2_distribution;
  std::uniform_real_distribution<double> rad_distribution;
  std::uniform_real_distribution<double> phi_distribution;
  std::uniform_real_distribution<double> dstep_distribution;
  std::uniform_real_distribution<double> hstep_distribution;
  std::uniform_int_distribution<int> change_distribution;
  std::uniform_int_distribution<int> xflag_distribution;
  std::uniform_int_distribution<int> yflag_distribution;
  std::uniform_int_distribution<int> zflag_distribution;
  std::uniform_int_distribution<int> reminder_distribution;

  // Parameters for domain randomization
  std::uniform_real_distribution<double> mass_distribution;
  std::uniform_real_distribution<double> pgain_distribution;
  std::uniform_real_distribution<double> dgain_distribution;
  std::uniform_real_distribution<double> friction_distribution; 

  // Kp Settings
  double init_pgain = 100;
  double init_dgain = 2;
  Eigen::VectorXd jointPgain, jointDgain;
};

} 