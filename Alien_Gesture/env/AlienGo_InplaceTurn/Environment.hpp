#include <stdlib.h>
#include <cstdint>
#include <set>
#include <chrono>
#include <random>
#include <fstream>
#include "math.h"
#include <raisim/OgreVis.hpp>
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

// This code is for learning PMTG with PD gain
// Have mTG to control trajectory
// Input : TG phase / v_desired / m_gait / state s ==> 1 + 1 + 1 + 15
// Output: TG parameters / u_fb / kp ==> 3 + 12 + 12 
// Move with PD controller 
// Mapping function : TBD


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
    // raisim::World::setActivationKey("/home/sonic/Libraries/Raisim/activation_Sunwoo_SNU.raisim");
    /// add objects
    alien_ = world_->addArticulatedSystem(resourceDir_+"/urdf/aliengo.urdf");
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
    gc_init_ <<  0, 0, 0.38, 1.0, 0.0, 0.0, 0.0, 0.0, 0.765076, -1.5303, 0.0, 0.765017, -1.53019, 0.0, 0.765574, -1.53102, 0.0, 0.765522, -1.53092;
    alien_->setState(gc_init_, gv_init_);

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

    obDim_ = 21;
    actionDim_ = nJoints_ * 2 + 3;
    actionStd_.setZero(nJoints_);
    obMean_.setZero(obDim_); obStd_.setZero(obDim_);

    float ltc, lcf;
    float dt = 0.033333;
    raisim::Vec<3> thigh_location;
    alien_->getFramePosition(alien_->getFrameByName("FL_thigh_joint"), thigh_location);
    raisim::Vec<3> calf_location;
    alien_->getFramePosition(alien_->getFrameByName("FL_calf_joint"), calf_location);
    raisim::Vec<3> foot_location;
    alien_->getFramePosition(alien_->getFrameByName("FL_foot_fixed"), foot_location);
    ltc = (thigh_location.e() - calf_location.e()).norm();
    lcf = (calf_location.e() - foot_location.e()).norm();
    mTG = TrajectoryGenerator(RUNNING_TROT, ltc, lcf, dt);

    /// action & observation scaling
    // actionStd_.setConstant(deg2rad(10));
    // actionStd_ << deg2rad(10), deg2rad(5), deg2rad(5), deg2rad(10), deg2rad(5), deg2rad(5),deg2rad(10), deg2rad(5), deg2rad(5),deg2rad(10), deg2rad(5), deg2rad(5);
    actionStd_ << deg2rad(45), deg2rad(10), deg2rad(10), deg2rad(45), deg2rad(10), deg2rad(10), deg2rad(45), deg2rad(10), deg2rad(10), deg2rad(45), deg2rad(10), deg2rad(10);

    freq_w = 0.0;    
    torque_w = 0.0;
    main_w = 1.0 - freq_w - torque_w;

    speed_scale = 1.0;
    dir_scale = 10.;
    stb_scale = 1.;
    torque_scale = 1e-4;
    freq_scale = 1e-4;

    current_count = 0;
    scaler = -1.0 / 2000;

    gui::rewardLogger.init({"SpeedReward", "DirectionReward", "StableReward", "TorqueReward", "FrequencyReward"});

    /// indices of links that should not make contact with ground
    footIndices_.insert(alien_->getBodyIdx("FL_calf"));
    footIndices_.insert(alien_->getBodyIdx("FR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RL_calf"));
    speed_distribution = std::uniform_real_distribution<double> (0.0, 0.0);
    angle_distribution = std::uniform_real_distribution<double> (deg2rad(-5), deg2rad(5));
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
      vis->addVisualObject("ref_speed_indicator", "arrowMesh", "yellow", {0.03, 0.03, 0.03}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("cur_speed_indicator", "arrowMesh", "red", {0.03, 0.03, 0.03}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      
      auto list = vis->getVisualObjectList();
      raisim::Vec<3> pos_tempr;
      alien_->getFramePosition(FR_HIP, pos_tempr);
      raisim::Vec<3> pos_templ;
      alien_->getFramePosition(FL_HIP, pos_templ);
      Eigen::Vector3d half = (pos_tempr.e() + pos_templ.e()) / 2;
      raisim::Mat<3,3> rot;
      raisim::Vec<3> direction = {1, 0, 0};
      raisim::zaxisToRotMat(direction, rot);
      list["ref_speed_indicator"].setPosition(half);
      list["ref_speed_indicator"].setScale(Eigen::Vector3d(0.05, 0.05, 0.05));
      list["ref_speed_indicator"].setOrientation(rot); 
      list["cur_speed_indicator"].setPosition(half);
      list["cur_speed_indicator"].setScale(Eigen::Vector3d(0.05, 0.05, 0.05));
      list["cur_speed_indicator"].setOrientation(rot); 

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

    gm.setSafeZone();
    gm.setMax_wait(30);

    image = cv::Mat::ones(colorHeight,colorWidth,CV_8UC3);
    namedWindow("Human_Motion", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("Human_Motion", 1800, 1200);
    cv::Size frame_size(colorWidth, colorHeight);
    vod = cv::VideoWriter("/home/sonic/Project/Alien_Gesture/Human.avi", 
    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), desired_fps_, frame_size, true);
    prev_foot =  pp;
    startFlag = true;
    deviceFlag = false;
    auto vis = raisim::OgreVis::get();
    for(int i =0 ; i < 31; ++i){
      vis->addVisualObject("bone" + std::to_string(i), "cylinderMesh", "red", {0.015, 0.015, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
    }
  }


  void reset() final
  {
    alien_->setState(gc_init_, gv_init_);
    current_count = 0;
    curriculum_cnt = 0;
    v_desired = 0.0;
    v_target = 0.0;
    v_current = 0.0;
    a_desired = 0.0;
    a_target = 0.0;
    mode_gait = STOP;
    change_count = 15;

    pTarget_.setZero();
    pTarget_.tail(nJoints_) = gc_init_.tail(12);
    alien_->setPdTarget(pTarget_, vTarget_);
    jointPgain.tail(nJoints_).setConstant(init_pgain);
    jointDgain.tail(nJoints_).setConstant(1.4 * std::sqrt(init_pgain));
    mTG.reset();
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
    Eigen::VectorXd rawGain = currentAction_PD.segment(nJoints_, nJoints_) * pGainScaler;
    double f_tg = currentAction_PD[nJoints_ * 2 + 0] * f_scaler + f_shifter;
    double alpha_tg = currentAction_PD[nJoints_ * 2 + 1]* alpha_scaler + alpha_shifter;
    double h_tg = currentAction_PD[nJoints_ * 2 + 2]* h_scaler + h_shifter;

    Eigen::VectorXd meanShift(nJoints_);
    meanShift.setConstant(pGainShift);
    rawGain = rawGain + meanShift;
    for(int i = 0; i < rawGain.size();++i)
      if(rawGain[i] < 0.0) rawGain[i] = 0.0;
    jointPgain.tail(nJoints_) = rawGain;
    jointPgain.tail(nJoints_).setConstant(init_pgain);
    jointDgain.tail(nJoints_) = jointPgain.tail(nJoints_).cwiseSqrt();
    jointDgain = jointDgain * 1.4;

    mTG.change_gait((GaitType)mode_gait);
    mTG.get_tg_parameters(f_tg, alpha_tg, h_tg);
    mTG.update();

    // alien_->setPdGains(jointPgain, jointDgain);
    pTarget12_ = currentAction;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += mTG.get_u();
    pTarget_.tail(nJoints_) = pTarget12_;

    alien_->setPdTarget(pTarget_, vTarget_);

    auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
    auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);    

    Eigen::Matrix3d body_rot;
    Eigen::Vector3d world_vel;
    Eigen::Vector3d body_vel;

    for(int i=0; i<loopCount; i++) {
      world_->integrate();  
      alien_->getState(gc_, gv_);
      body_rot = alien_->getBaseOrientation().e();
      world_vel = Eigen::Vector3d(gv_[0], gv_[1], 0);
      body_vel = body_rot.inverse() * world_vel;
      v_current = body_vel[0];
      if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
      {
        auto vis = raisim::OgreVis::get();
        auto& list = raisim::OgreVis::get()->getVisualObjectList();
        visualize_arrow();
        vis->renderOneFrame();
      }      
      visualizationCounter_++;
    }

    if(abs(v_target - v_desired) < v_acc) v_target = v_desired;
    else if(v_target > v_desired)
    {
      v_target -= v_acc;
    }
    else if(v_target < v_desired)
    {
      v_target += v_acc;
    }
    if(abs(a_target - a_desired) < a_acc) a_target = a_desired;
    else if(a_target > a_desired)
    {
      a_target -= a_acc;
    }
    else if(a_target < a_desired)
    {
      a_target += a_acc;
    }

    updateObservation();

    float totalReward = CalcReward();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("SpeedReward", speed_reward);
      gui::rewardLogger.log("DirectionReward", dir_reward);
      gui::rewardLogger.log("StableReward", stb_reward);
      gui::rewardLogger.log("TorqueReward", torque_reward);
      gui::rewardLogger.log("FrequencyReward", freq_reward);

      /// reset camera
      auto vis = raisim::OgreVis::get();

      vis->select(anymalVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }
    current_count++;
    if(current_count%change_count == 0){
      mode_gait = RUNNING_TROT;
      change_angle_var();
      change_count = 60;
    }
    // change_run_stop();
    if(!mStopFlag) 
      return totalReward;
    else
      return -1E10;
  }

  float stepTest(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    curriculum_speed_distribution_mod(3, speed_distribution);
    curriculum_angle_distribution(3, angle_distribution);
    Eigen::VectorXd currentAction_PD = action.cast<double>();
    Eigen::VectorXd currentAction(nJoints_);
    currentAction = currentAction_PD.head(nJoints_);
    Eigen::VectorXd rawGain = currentAction_PD.segment(nJoints_, nJoints_) * pGainScaler;
    double f_tg = currentAction_PD[nJoints_ * 2 + 0] * f_scaler + f_shifter;
    double alpha_tg = currentAction_PD[nJoints_ * 2 + 1]* alpha_scaler + alpha_shifter;
    double h_tg = currentAction_PD[nJoints_ * 2 + 2]* h_scaler + h_shifter;

    Eigen::VectorXd meanShift(nJoints_);
    meanShift.setConstant(pGainShift);
    rawGain = rawGain + meanShift;
    for(int i = 0; i < rawGain.size();++i)
      if(rawGain[i] < 0.0) rawGain[i] = 0.0;
    jointPgain.tail(nJoints_) = rawGain;
    jointDgain.tail(nJoints_) = jointPgain.tail(nJoints_).cwiseSqrt();
    jointDgain = jointDgain * 1.4;

    mTG.change_gait((GaitType)mode_gait);
    mTG.get_tg_parameters(f_tg, alpha_tg, h_tg);
    mTG.update();
    // alien_->setPdGains(jointPgain, jointDgain);
    pTarget12_ = currentAction;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += mTG.get_u();

    pTarget_.tail(nJoints_) = pTarget12_;

    alien_->setPdTarget(pTarget_, vTarget_);
    auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
    auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);    

    Eigen::Matrix3d body_rot;
    Eigen::Vector3d world_vel;
    Eigen::Vector3d body_vel;
    for(int i=0; i<loopCount; i++) {
      world_->integrate();  
      alien_->getState(gc_, gv_);
      // GetFromDevice();
      body_rot = alien_->getBaseOrientation().e();
      world_vel = Eigen::Vector3d(gv_[0], gv_[1], 0);
      body_vel = body_rot.inverse() * world_vel;
      v_current = body_vel[0];
      if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
      {
        auto vis = raisim::OgreVis::get();
        auto& list = raisim::OgreVis::get()->getVisualObjectList();
        visualize_arrow();
        vis->renderOneFrame();
      }      
      visualizationCounter_++;
    }

    // bgra2Mat();
    // vod.write(image);
    // imshow("Human_Motion", image);
    // cv::waitKey(1);
    if(abs(v_target - v_desired) < v_acc) v_target = v_desired;
    else if(v_target > v_desired)
    {
      v_target -= v_acc;
    }
    else if(v_target < v_desired)
    {
      v_target += v_acc;
    }
    if(abs(a_target - a_desired) < a_acc) a_target = a_desired;
    else if(a_target > a_desired)
    {
      a_target -= a_acc;
    }
    else if(a_target < a_desired)
    {
      a_target += a_acc;
    }

    updateObservation();

    float totalReward = CalcReward();
    // float totalReward = 0;
    // update_from_skel();
    if(visualizeThisStep_) {
      gui::rewardLogger.log("SpeedReward", speed_reward);
      gui::rewardLogger.log("DirectionReward",  dir_reward);
      gui::rewardLogger.log("StableReward", stb_reward);

      /// reset camera
      auto vis = raisim::OgreVis::get();

      vis->select(anymalVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(-1.8), Ogre::Radian(-1.3), 3, true);
    }
    current_count++;
    if(current_count%change_count == 0){
      mode_gait = RUNNING_TROT;
      change_angle_var();
      change_count = 60;
    }
    // change_run_stop();
    if(!mStopFlag) 
      return totalReward;
    else
      return -1E10;
  }

  void curriculumUpdate(int update) final {
    int curr1 = 700;
    int curr2 = 1500;
    int curr3 = 2000;
    int curr4 = 3000;
    if(update > curr1 && update < curr2)
    {
      curriculum_angular_velocity_distribution(1, angle_distribution);
    }
    else if(update > curr2 && update < curr3)
    {
      curriculum_angular_velocity_distribution(2, angle_distribution);
    }
    else if(update > curr3 && update < curr4)
    {
      curriculum_angular_velocity_distribution(3, angle_distribution);
    }
    else if(update > curr4)
    {
      curriculum_angular_velocity_distribution(4, angle_distribution);
    }
  }

  void change_angle_var()
  {
    unsigned seed_temp = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed_temp);
    a_desired = angle_distribution(generator);
  }


  void visualize_arrow()
  {
    auto vis = raisim::OgreVis::get();
    auto& list = raisim::OgreVis::get()->getVisualObjectList();
    raisim::Vec<3> pos_tempr;
    alien_->getFramePosition(FR_HIP, pos_tempr);
    raisim::Vec<3> pos_templ;
    alien_->getFramePosition(FL_HIP, pos_templ);
    Eigen::Vector3d half = (pos_tempr.e() + pos_templ.e()) / 2;
    half[2]+= arrow_offset;


    raisim::Mat<3,3> rot;
    Eigen::Matrix3d rootrotation = alien_->getBaseOrientation().e();
    raisim::Vec<3> direction = {rootrotation(0,0), rootrotation(1,0), rootrotation(2,0)}; 
    raisim::zaxisToRotMat(direction, rot);


    Eigen::Matrix3d aa;
    aa = Eigen::AngleAxisd(a_target, Eigen::Vector3d(0,0,1));
    Eigen::Vector3d aaa =  aa * rootrotation * Eigen::Vector3d(1,0,0);
    raisim::Vec<3> dir_ref ={aaa[0], aaa[1], aaa[2]};
    raisim::Mat<3,3> rot_ref;
    raisim::zaxisToRotMat(dir_ref, rot_ref);

    Eigen::Vector3d half2 = half;
    half2[2] +=arrow_offset;
    list["ref_speed_indicator"].setPosition(half);
    list["cur_speed_indicator"].setPosition(half2);
    list["ref_speed_indicator"].setScale(Eigen::Vector3d(0.1, 0.1,  0.1));
    list["cur_speed_indicator"].setScale(Eigen::Vector3d(0.1, 0.1,  0.1));
    list["ref_speed_indicator"].setOrientation(rot_ref);
    list["cur_speed_indicator"].setOrientation(rot); 
  }

  float CalcReward()
  {   
    float reward = 0;
    double speed_err = 0;
    double dir_err = 0;
    double stb_err = 0;
    double torque_err = 0;
    double freq_err = 0;

    alien_->getState(gc_, gv_);
    if(isnan(gc_[0]))
    {
      return 0;
    }
    speed_err = CalcSpeedErr();
    dir_err = CalcDirErr();
    stb_err = CalcStbErr();
    torque_err = CalcTorqueErr();
    freq_err = CalcFreqErr();

    speed_reward = exp(-speed_scale * speed_err);
    dir_reward = exp(-dir_scale * dir_err);
    stb_reward = exp(-stb_scale * stb_err);
    // torque_reward = exp(-torque_scale * torque_err);
    freq_reward = exp(-freq_scale * freq_err);

    // reward = speed_reward * dir_reward;
    // reward = main_w * speed_reward * dir_reward + torque_w * torque_reward + freq_w * freq_reward;
    reward = main_w * speed_reward * dir_reward * stb_reward;
    return reward;
  }

  float CalcSpeedErr()
  {
    float err = 0.0;
    err = pow((v_current - v_target),2);
    return err;
  }

  float CalcDirErr()
  {
    Eigen::Vector3d xvec(1.0, 0.0, 0.0);
    float rot_err = 0.;
    rot_err = pow(gv_[2] - a_target,2);
    return rot_err;
  }

  float CalcStbErr()
  {
    Eigen::Vector3d zvec(0.0, 0.0, 1.0);
    float rot_err = 0.;
    raisim::Vec<4> quat;
    raisim::Mat<3, 3> R;
    for(int j = 0; j<4; ++j)
      quat[j] = gc_[j+3];
    quatToRotMat(quat, R);
    Eigen::Matrix3d R_curr = R.e();
    Eigen::Vector3d current = R_curr * zvec;
    // current[2] = 0.0;

    float angle;
    if(current.norm() < 1e-10)
      angle = 0.0;
    else
      angle = acos(current.dot(zvec)/current.norm());
    return angle * angle;
  }

  float CalcTorqueErr()
  {
    float total_torque = alien_->getGeneralizedForce().squaredNorm();
    return total_torque;
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

  void updateExtraInfo() final {
    extraInfo_["base height"] = gc_[2];
  }

  void updateObservation() {
    alien_->getState(gc_, gv_);
    obScaled_.setZero(obDim_);
    TG_phase = mTG.get_phase();

    /// update observations
    obScaled_[0] = TG_phase;
    obScaled_[1] = v_desired;
    obScaled_[2] = v_target;
    obScaled_[3] = v_current;
    obScaled_[4] = a_target;
    obScaled_[5] = a_desired;
    obScaled_.segment(6, 12) = gc_.tail(12);
    Eigen::Quaterniond quat(gc_[3], gc_[4], gc_[5], gc_[6]);
    Eigen::AngleAxisd aa(quat);
    Eigen::Vector3d rotation = aa.angle()*aa.axis();
    obScaled_.segment(18, 3) = rotation;
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
    k4abt_tracker_shutdown(tracker);
    k4abt_tracker_destroy(tracker);

    k4a_device_stop_cameras(device);
    k4a_device_close(device);
  }


  void GetFromDevice() {
    k4a_capture_t sensorCapture = nullptr;
    k4a_wait_result_t getCaptureResult = k4a_device_get_capture(device, &sensorCapture, 0); // timeout_in_ms is set to 0

    if (getCaptureResult == K4A_WAIT_RESULT_SUCCEEDED)
    {
        // timeout_in_ms is set to 0. Return immediately no matter whether the sensorCapture is successfully added
        // to the queue or not.
        k4a_wait_result_t queueCaptureResult = k4abt_tracker_enqueue_capture(tracker, sensorCapture, 0);

        k4a_image_t colorImage = k4a_capture_get_color_image(sensorCapture);
        img_bgra32 = k4a_image_get_buffer(colorImage);

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
          skeleton = body.skeleton;
          bodies.push_back(body);
          if(current_count > delay_cnt){
            visualize_bones(bodies[0]);
            bodies.pop_front();
          }
          else
            visualize_bones(bodies[0]);
        }
        //Release the bodyFrame    
        k4abt_frame_release(bodyFrame);
    }
  }

  void visualize_bones(k4abt_body_t body)
  {
    float scaler = -1. / 2000;
    float y_shift = 0.5;
    auto& list = raisim::OgreVis::get()->getVisualObjectList();
    k4a_float3_t jointRPosition = body.skeleton.joints[K4ABT_JOINT_FOOT_RIGHT].position;
    k4a_float3_t jointLPosition = body.skeleton.joints[K4ABT_JOINT_FOOT_LEFT].position;
    k4a_float3_t joint0Position = body.skeleton.joints[K4ABT_JOINT_FOOT_RIGHT].position;
    if(jointLPosition.xyz.y > jointRPosition.xyz.y) joint0Position.xyz.y = jointLPosition.xyz.y;

    for(size_t boneIdx = 0; boneIdx < g_boneList.size(); boneIdx++)
    {
      k4abt_joint_id_t joint1 = g_boneList[boneIdx].first;
      k4abt_joint_id_t joint2 = g_boneList[boneIdx].second;
      k4a_float3_t joint1Position = body.skeleton.joints[joint1].position;
      k4a_float3_t joint2Position = body.skeleton.joints[joint2].position;
      Eigen::Vector3d point1 = scaler * Eigen::Vector3d(joint1Position.xyz.z- joint0Position.xyz.z ,joint1Position.xyz.x - joint0Position.xyz.x, joint1Position.xyz.y- joint0Position.xyz.y);
      Eigen::Vector3d point2 = scaler * Eigen::Vector3d( joint2Position.xyz.z- joint0Position.xyz.z ,joint2Position.xyz.x - joint0Position.xyz.x, joint2Position.xyz.y- joint0Position.xyz.y);
      Eigen::Vector3d half = (point1 + point2) / 2 ;
      half[0] += gc_[0];
      half[1] += gc_[1] - y_shift ;
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

  void update_from_skel()
  {
    gm.getCurrentPose(skeleton);
    float temp = gm.LaggedSpeed();
    if(temp > 1.5) v_desired = 1.5;
    else if(!isnan(temp)) v_desired = temp;
    // change_run_stop();
    // change_run();
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
  std::vector<GraphicObject> * anymalVisual_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, torque_, gc_dummy_init;
  Eigen::VectorXd gc_prev1, gc_prev2;
  double desired_fps_ = 30.;
  int visualizationCounter_=0;
  float effort_limit = 40. * 0.85;
  Eigen::VectorXd actionStd_, obMean_, obStd_;
  Eigen::VectorXd obDouble_, obScaled_;
  std::set<size_t> footIndices_;

  bool startFlag, deviceFlag;

  int change_count, current_count;
  int curriculum_cnt;

  double TG_phase, v_desired, v_target, v_current;
  double a_desired, a_target;
  double v_acc = 1.0 / 30., a_acc = deg2rad(10) / 30.;
  int mode_gait;
  std::vector<std::deque<double>> torque_sequence;


  // Reward Shapers
  double main_w, torque_w, freq_w;
  double speed_scale, dir_scale, stb_scale, torque_scale, freq_scale;
  double speed_reward, dir_reward, stb_reward, torque_reward, freq_reward;

  // Kp Settings
  double init_pgain = 100;
  Eigen::VectorXd jointPgain, jointDgain;
  float pGainScaler = 60.0;
  float pGainShift = 80.0;
  // float pGainOffSet = 30.0;

  // Set f : 0.3 ~ 2.0Hz, alpha : 0 - 3 degree, h : 0.28 ~ 0.52m
  float f_scaler = 1.3; float f_shifter = 1.5;
  float alpha_scaler = deg2rad(20); float alpha_shifter = deg2rad(20);
  float h_scaler = 0.12; float h_shifter = 0.4;
  float arrow_offset = 0.06;

  // Kinect Settings
  float z_offset = 0.85;
  k4a_device_t device = nullptr;
  k4abt_tracker_t tracker = nullptr;
  std::vector<int> connection;
  float scaler;
  float z_standard;
  int depthWidth, depthHeight, colorWidth, colorHeight;
  cv::Mat image;
  uint8_t* img_bgra32;
  cv::VideoWriter vod;
  int delay_cnt;
  std::deque<k4abt_body_t> bodies;

  TrajectoryGenerator mTG;
  GaitMapper gm;

  Eigen::Matrix3d Rinit, Rcomp;
  float hum2rob;
  Eigen::Vector3d prev_foot;
  Eigen::Vector3d pp;
  raisim::Vec<3> force; 
  std::uniform_real_distribution<double> speed_distribution;
  std::uniform_real_distribution<double> angle_distribution;
  std::uniform_int_distribution<int> mode_distribution;

  bool init_pose_flag;
  k4abt_skeleton_t skeleton;
  //IK solvers
};

}