
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
    alien_ = world_->addArticulatedSystem(resourceDir_+"/aliengo/aliengo.urdf");
    ref_dummy = new raisim::ArticulatedSystem(resourceDir_+"/aliengo/aliengo.urdf");    
    ik_dummy = new raisim::ArticulatedSystem(resourceDir_+"/aliengo/aliengo.urdf");    
    alien_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    auto ground = world_->addGround();
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
    gc_init_ << 0, 0, 0.40, 1.0, 0.0, 0.0, 0.0,0.0, 0.790522,-1.382, -0.0, 0.790522,-1.382,0.0, 0.790522,-1.382,0.0, 0.790522,-1.382;
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
    nRefClips = 12;
    obDim_ = 47;
    actionDim_ = nJoints_;

    actionStd_.setZero(nJoints_);  actionMean_.setZero(nJoints_);
    obMean_.setZero(obDim_); obStd_.setZero(obDim_);
    prev_vel.setZero(nJoints_); joint_acc.setZero(nJoints_);

    getKeyPoses("/home/sonic/Project/Alien_Gesture/env/rsc/new_motion.txt");

    float ah = deg2rad(30), at = deg2rad(30), ac = deg2rad(30);
    float mh = 0, mt = M_PI /2, mc = -1.80;
    actionStd_ << ah, at, ac, ah, at, ac, ah, at, ac, ah, at, ac;
    // actionMean_ << mh, mt, mc, mh, mt, mc, mh, mt, mc, mh, mt, mc;

    main_w = 1.0;

    joint_imit_scale = 10.0;
    ori_imit_scale = 5.0;
    end_imit_scale = 1.0;

    gui::rewardLogger.init({"Angle_reward", "Ori_reward", "End_reward"});

    /// indices of links that should not make contact with ground
    footIndices_.insert(alien_->getBodyIdx("FL_calf"));
    footIndices_.insert(alien_->getBodyIdx("FR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RL_calf"));
    foot_hipIndices_.insert(alien_->getBodyIdx("FL_hip"));
    foot_hipIndices_.insert(alien_->getBodyIdx("FR_hip"));
    foot_hipIndices_.insert(alien_->getBodyIdx("RR_hip"));
    foot_hipIndices_.insert(alien_->getBodyIdx("RL_hip"));
    foot_hipIndices_.insert(alien_->getBodyIdx("FL_thigh"));
    foot_hipIndices_.insert(alien_->getBodyIdx("FR_thigh"));
    foot_hipIndices_.insert(alien_->getBodyIdx("RR_thigh"));
    foot_hipIndices_.insert(alien_->getBodyIdx("RL_thigh"));
    foot_hipIndices_.insert(alien_->getBodyIdx("FL_calf"));
    foot_hipIndices_.insert(alien_->getBodyIdx("FR_calf"));
    foot_hipIndices_.insert(alien_->getBodyIdx("RR_calf"));
    foot_hipIndices_.insert(alien_->getBodyIdx("RL_calf"));

    // for(int i = 0; i < nJoints_; ++i)
    // {
    //   std::deque<double> zero_(32, 0.0);
    //   torque_sequence.push_back(zero_);
    //   zero_.shrink_to_fit();
    // }
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

      desired_fps_ = 1.0/ control_dt_;
      vis->setDesiredFPS(desired_fps_);
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
    ref_dummy->setState(gc_init_, gv_init_);
    ik_dummy->setState(gc_init_, gv_init_);
    current_count = 0;

    prev_vel.setZero();
    joint_acc.setZero();
    gc_prev = gc_init_;
    gc_target = gc_init_.tail(nJoints_ + 4);

    joint_imit_reward = 0.0;
    ori_imit_reward = 0.0;
    end_imit_reward = 0.0;

    pTarget_.setZero();
    pTarget_.tail(nJoints_) = gc_init_.tail(nJoints_);
    alien_->setPdTarget(pTarget_, vTarget_);

    clearKeys();
    setKeyPoseIndices();
    motionGenerator();

    // Only turn on at the test step
    // test_extension();

    updateObservation();

    if(visualizable_)
      gui::rewardLogger.clean();
  }

  void test_extension()
  {
    nRefClips = 50;
    clearKeys();
    setKeyPoseIndices();
    motionGenerator(60);
  }

  float stepRef() final
  {
    return 0;
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    gc_target = motion_clips[current_count];
    Eigen::VectorXd currentAction_PD = action.cast<double>();
    Eigen::VectorXd currentAction(nJoints_);
    currentAction = currentAction_PD.head(nJoints_);

    pTarget12_ = currentAction;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += gc_target.tail(12);
    pTarget_.tail(nJoints_) = pTarget12_;

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
      auto vis = raisim::OgreVis::get();

      vis->select(anymalVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }
    if(current_count < motion_clips.size()-2)
      current_count++;
    if(!mStopFlag) 
      return totalReward;
    else
      return -1E10;
  }

  float stepTest(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    gc_target = motion_clips[current_count];
    Eigen::VectorXd currentAction_PD = action.cast<double>();
    Eigen::VectorXd currentAction(nJoints_);
    currentAction = currentAction_PD.head(nJoints_);
    pTarget12_ = currentAction;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += gc_target.tail(12);
    pTarget_.tail(nJoints_) = pTarget12_;

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
      /// reset camera
      auto vis = raisim::OgreVis::get();

      vis->select(anymalVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }
    if(current_count < motion_clips.size()-2)
      current_count++;
    if(!mStopFlag) 
      return totalReward;
    else
      return -1E10;
  }

  void getKeyPoses(std::string filename)
  {
    std::ifstream keyposefile(filename);
    if(!keyposefile){
      std::cout << "Expected valid input file name" << std::endl;
    }
    int robot_ang = 12 + 4;
    key_poses.push_back(gc_init_.tail(4 + nJoints_));
    while(!keyposefile.eof()){
      Eigen::VectorXd one_key_pose(robot_ang);
      for(int i = 0 ; i < robot_ang; ++i)
      {
        float end;
        keyposefile >> end;
        one_key_pose[i] = end;
      }
      key_poses.push_back(one_key_pose);
    }
    keyposefile.close();
    number_distribution =  std::uniform_int_distribution<int> (0, key_poses.size()-2);
  }

  void clearKeys()
  {
    current_key_indices.clear();
    current_key_indices.shrink_to_fit();
    motion_clips.clear();
    motion_clips.shrink_to_fit();
  }

  void setKeyPoseIndices()
  {
    current_key_indices.push_back(0);
    current_key_indices.push_back(0);
    // Allows overlapping, not much possibility
    for(int i = 0 ; i < nRefClips-1; ++i)
    {
      std::random_device rd;
      // unsigned seed_temp = std::chrono::system_clock::now().time_since_epoch().count();
      // std::default_random_engine generator(seed_temp);
      std::mt19937 gen(rd());
      int n = number_distribution(gen);
      current_key_indices.push_back(n);
    }
  }

  void motionGenerator(int elapsing = 10)
  {
    // Now we have N poses to be interpolated (From the begining)
    int inbetween = int(elapsing * int(desired_fps_) / nRefClips);
    for(int i = 0; i < nRefClips; ++i)
    {
      Eigen::Quaterniond q1 = Eigen::Quaterniond(key_poses[current_key_indices[i]][0], key_poses[current_key_indices[i]][1], key_poses[current_key_indices[i]][2], key_poses[current_key_indices[i]][3]);
      Eigen::Quaterniond q2 = Eigen::Quaterniond(key_poses[current_key_indices[i+1]][0], key_poses[current_key_indices[i+1]][1], key_poses[current_key_indices[i+1]][2], key_poses[current_key_indices[i+1]][3]);
      Eigen::VectorXd now = key_poses[current_key_indices[i]].tail(nJoints_);
      Eigen::VectorXd next = key_poses[current_key_indices[i+1]].tail(nJoints_);
      for(int j = 0 ; j < inbetween ; ++j)
      {
        Eigen::Quaterniond slerped = q1.slerp(float(j) / float(inbetween), q2);
        Eigen::VectorXd interpol = (now *(inbetween - j) / inbetween) + (next *j / inbetween);
        Eigen::VectorXd total(4 + nJoints_);
        slerped.normalize();
        total[0] = slerped.w();
        total[1] = slerped.x();
        total[2] = slerped.y();
        total[3] = slerped.z();
        total.tail(nJoints_) = interpol;
        motion_clips.push_back(total);
      }
    }
    motion_clips.push_back(key_poses[current_key_indices[current_key_indices.size()-1]]);
    raisim::Vec<3> tempPositionL;
    ik_dummy->getFramePosition(FL_FOOT, tempPositionL);
    float x_shifter = tempPositionL[0];
    float y_shifter = tempPositionL[1];
    for(int current_idx =0;current_idx < motion_clips.size(); ++current_idx)
    {
      auto current_joint = motion_clips[current_idx];
      Eigen::VectorXd gc_r = ref_dummy->getGeneralizedCoordinate().e();
      for(int j = 0 ; j < 16; ++j)
      {
        gc_r[3+j] = current_joint[j];
      }
      ik_dummy->setGeneralizedCoordinate(gc_r);
      float min_foot = 1000;
      float body_xshifter, body_yshifter;
      for(int i =0 ; i < 4; ++i)
      {
        raisim::Vec<3> tempPosition;
        ik_dummy->getFramePosition(FR_FOOT + 3 * i, tempPosition);
        float height = (tempPosition[2]-0.0165);
        if(i == 1){
          body_xshifter = x_shifter - tempPosition[0];
          body_yshifter = y_shifter - tempPosition[1];
        }
        if(min_foot > height) min_foot = height;
      }
      gc_r[2] -= min_foot + 0.009;
      gc_r[0] += body_xshifter;
      gc_r[1] += body_yshifter;

      ik_dummy->setGeneralizedCoordinate(gc_r);

      mrl->isHumanContact(0,0);
      mrl->getPreviousRealPose(gc_init_);
      mrl->getCurrentRawPose(gc_r);
      mrl->reshapeMotion();
      Eigen::VectorXd gc_reshaped = mrl->getReshapedMotion();
      ref_dummy->setGeneralizedCoordinate(gc_reshaped);
      motion_clips[current_idx] = gc_reshaped.tail(4 + nJoints_);
    }
    ref_dummy->setGeneralizedCoordinate(gc_init_);
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
    // std::cout << 

    joint_imit_reward = exp(-joint_imit_scale * joint_imit_err);
    ori_imit_reward = exp(-ori_imit_scale * ori_imit_err);
    end_imit_reward = exp(-end_imit_scale * end_imit_err);

    reward = main_w * joint_imit_reward * ori_imit_reward * end_imit_reward;
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
  MotionReshaper *mrl;
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* alien_;
  raisim::ArticulatedSystem* ik_dummy;
  raisim::ArticulatedSystem* ref_dummy;
  std::vector<GraphicObject> * anymalVisual_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, torque_, gc_dummy_init;
  Eigen::VectorXd gc_prev, gc_target;
  double desired_fps_ = 30.;
  int visualizationCounter_=0;
  float effort_limit = 40.0 * 0.90;
  Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_;
  Eigen::VectorXd obDouble_, obScaled_;
  std::set<size_t> footIndices_;
  std::set<size_t> foot_hipIndices_;

  int current_count;
  Eigen::VectorXd prev_vel, joint_acc;

  std::vector<std::deque<double>> torque_sequence;

  // Reward Shapers
  double main_w;
  double joint_imit_scale, ori_imit_scale, end_imit_scale;
  double joint_imit_reward, ori_imit_reward, end_imit_reward;

  // Kp Settings
  double init_pgain = 300;
  double init_dgain = 10;
  Eigen::VectorXd jointPgain, jointDgain;
  float pGainScaler = 150.0;
  float pGainShift = 180.0;
  // float pGainOffSet = 30.0;

  // Imitation Learning
  int nRefClips;
  std::vector<Eigen::VectorXd> key_poses;
  std::vector<int> current_key_indices;
  std::vector<Eigen::VectorXd> motion_clips;
  std::uniform_int_distribution<int> number_distribution;
};

} 