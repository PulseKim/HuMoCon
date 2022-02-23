#include <stdlib.h>
#include <cstdint>
#include <set>
#include <chrono>
#include <random>
#include <fstream>
#include <raisim/OgreVis.hpp>
#include "RaisimGymEnv.hpp"
#include "visSetupCallback2.hpp"
#include "Utilities.h"
#include "math.h"
#include "IKSolver.hpp"
#include "MotionPlanner.hpp"

#include <k4arecord/playback.h>
#include <k4a/k4a.h>
#include <k4abt.h>
#include <IpTNLP.hpp>
#include <IpIpoptApplication.hpp>

#include "../visualizer_alien/raisimKeyboardCallback.hpp"
#include "../visualizer_alien/helper.hpp"
#include "../visualizer_alien/guiState.hpp"
#include "../visualizer_alien/raisimBasicImguiPanel.hpp"

#define deg2rad(ang) ((ang) * M_PI / 180.0)
#define rad2deg(ang) ((ang) * 180.0 / M_PI)

std::uniform_real_distribution<double> height_distribution(0.17, 0.30);



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

    READ_YAML(double, control_dt_, cfg["control_dt"]);
    control_fps = floor(1.0 / control_dt_ + 0.5);
    motion_count = 0;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    torque_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);
    prev_action1.setZero(nJoints_); prev_action2.setZero(nJoints_); prev_action3.setZero(nJoints_); 
    

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.44, 1.0, 0.0, 0.0, 0.0, 0.01689, 0.27114, -0.86400, 0.01283, 0.27160, -0.86473, 0.01689, 0.27114, -0.86400, 0.01283, 0.27156, -0.86473;

    prev_action1 = gc_init_.tail(nJoints_);
    prev_action2 = gc_init_.tail(nJoints_);
    prev_action3 = gc_init_.tail(nJoints_);

    gc_dummy_init = gc_init_;
    // gc_dummy_init[1] = coll_off;
    terminalFlag = false;
    startFlag = true; 

    /// set pd gains
    double pgain = 500;
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(pgain);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(2*std::sqrt(pgain));
    alien_->setPdGains(jointPgain, jointDgain);
    alien_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    gc_prev1 = gc_init_;
    gc_prev2 = gc_init_;

    obDim_ = 100; 
    actionDim_ = nJoints_;
    actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obMean_.setZero(obDim_); obStd_.setZero(obDim_);

    /// action & observation scaling
    actionMean_.setConstant(0);
    actionStd_.setConstant(0.53);

    x_off = 0;
    y_off = 0;

    pose_w = 0.59;
    vel_w = 0.01;
    end_eff_w = 0.2;
    root_pose_w = 0.15;
    root_vel_w = 0.05;

    pose_scale = 20;
    vel_scale = 0.1;
    end_eff_scale = 20;
    root_pose_scale = 20;
    root_pose_rot_scale = 10;
    root_vel_scale = 2;
    root_vel_rot_scale = 0.2;

    scaler = -1.0 /1300;
    startpoint = 15;


    // gui::rewardLogger.init({"poseReward", "velReward","endReward"});
    gui::rewardLogger.init({"poseReward", "velReward","endReward", "rootPoseReward", "rootVelReward"});

    /// indices of links that should not make contact with ground
    footIndices_.insert(alien_->getBodyIdx("FL_calf"));
    footIndices_.insert(alien_->getBodyIdx("FR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RL_calf"));

    names.push_back("floating_base");
    names.push_back("floating_base");
    names.push_back("FL_foot_fixed");
    names.push_back("FR_foot_fixed");
    names.push_back("RL_foot_fixed");
    names.push_back("RR_foot_fixed");

    mIK = new IKSolver(ref_dummy, names);
   
    
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
      vis->createGraphicalObject(ref_dummy, "dummy");
      vis->createGraphicalObject(ground, 10, "floor", "checkerboard_green");
      desired_fps_ = 60.;
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
    for(int i =0 ; i <3;++i){ 
      vis->addVisualObject("sphere" + std::to_string(i), "sphereMesh", "orange", {0.03, 0.03, 0.03}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);

    }
    startFlag = true;
    for(int i = 0; i < 3; ++i)
      points.push_back(Eigen::Vector3d(0,0,0));
  }

  void reset() final
  {
    alien_->setState(gc_init_, gv_init_);
    ref_dummy ->setState(gc_dummy_init, gv_init_);
    ref_pose = gc_dummy_init;
    gc_prev1 = gc_init_;
    gc_prev2 = gc_init_;
    terminalFlag = false;

    unsigned seed_temp = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed_temp);
    min_h = height_distribution(generator);

    Eigen::VectorXd timings;
    timings.setZero(2);
    timings[0] = 0.8;
    timings[1] = 5.0 * (20 * min_h /3);
    KeyMotionGen(min_h, 0.44, timings);


    prev_action1 = gc_init_.tail(nJoints_);
    prev_action2 = gc_init_.tail(nJoints_);
    prev_action3 = gc_init_.tail(nJoints_);

    motion_count = 0;
    balance_count = 0;
    terminal_count = 0;
    jumpFlag = false;

    updateObservation();    
    if(visualizable_)
      gui::rewardLogger.clean();
  }

  float stepRef() final
  {
    auto pose_ =  motionClip_[motion_count];
    ref_dummy->setGeneralizedCoordinate(pose_);

    auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
    auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);
    for(int i=0; i<loopCount; i++) {
      raisim::OgreVis::get()->renderOneFrame();
    }
    motion_count++;

    if(visualizeThisStep_) {
      /// reset camera
      auto vis = raisim::OgreVis::get();

      vis->select(anymalVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 5, true);
    }
    if(motion_count == motionClip_.size())
    {
      mStopFlag = true;
    }
    if(!mStopFlag) 
      return 0;
    else{
      stopRecordingVideo();
      return -1E10;
    }
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    Eigen::VectorXd currentAction = action.cast<double>();
    pTarget12_ = currentAction;
    ref_pose =  motionClip_[motion_count];
    ref_vel = speedClip_[motion_count];
    pTarget12_ += ref_pose.tail(nJoints_);
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    // pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

    alien_->setPdTarget(pTarget_, vTarget_);

    auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
    auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);
    
    for(int i=0; i<loopCount; i++) {
      world_->integrate();  
      if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
      {
        auto vis = raisim::OgreVis::get();
        auto& list = raisim::OgreVis::get()->getVisualObjectList();
        // ref_pose[1] += coll_off;
        ref_dummy->setGeneralizedCoordinate(ref_pose);
        // ref_pose[1] -= coll_off;
        vis->renderOneFrame();
      }
      
      visualizationCounter_++;
    }

    updateObservation();

    float totalReward = CalcRewardImitate();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("poseReward", pose_reward);
      gui::rewardLogger.log("velReward",  vel_reward);
      gui::rewardLogger.log("endReward", end_eff_reward);
      gui::rewardLogger.log("rootPoseReward", root_pose_reward);
      gui::rewardLogger.log("rootVelReward", root_vel_reward);
      /// reset camera
      auto vis = raisim::OgreVis::get();

      vis->select(anymalVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }
    prev_action3 = prev_action2;
    prev_action2 = prev_action1;
    prev_action1 = pTarget12_;

    if(balance_count < 0){
      totalReward = 0;
      terminal_count++;
      if(terminal_count > 10)
        terminalFlag = true;
    }
    else if(balance_count < startpoint)
      balance_count++;
    else if(motion_count == motionClip_.size() - 2)
    {
      balance_count = -50;
      motion_count = 0;
      jumpFlag = false;
    }
    else if(balance_count == startpoint){
      jumpFlag = true;
      motion_count++;
    }

    if(isnan(totalReward))
      totalReward = 0;

    if(!mStopFlag) 
      return totalReward;
    else
      return -1E10;
  }

  float stepTest(const Eigen::Ref<EigenVec>& action) final {
    GetFromDevice();
    Eigen::VectorXd currentAction = action.cast<double>();
    pTarget12_ = currentAction;
    ref_vel = gv_init_;
    pTarget12_ += ref_pose.tail(nJoints_);
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    // pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

    alien_->setPdTarget(pTarget_, vTarget_);

    auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
    auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);    

    for(int i=0; i<loopCount; i++) {
      world_->integrate();  
      if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
      {
        auto vis = raisim::OgreVis::get();
        auto& list = raisim::OgreVis::get()->getVisualObjectList();
        for(int i =0 ; i < points.size();++i){
          list["sphere" + std::to_string(i)].setPosition(points[i]);
        }
        // ref_pose[1] += coll_off;
        ref_dummy->setGeneralizedCoordinate(ref_pose);
        // ref_pose[1] -= coll_off;
        vis->renderOneFrame();
      }
      
      visualizationCounter_++;
    }
    
    updateObservation();

    float totalReward = CalcRewardImitate();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("poseReward", pose_reward);
      gui::rewardLogger.log("velReward",  vel_reward);
      gui::rewardLogger.log("endReward", end_eff_reward);
      gui::rewardLogger.log("rootPoseReward", root_pose_reward);
      gui::rewardLogger.log("rootVelReward", root_vel_reward);
      /// reset camera
      auto vis = raisim::OgreVis::get();

      vis->select(anymalVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }
    prev_action3 = prev_action2;
    prev_action2 = prev_action1;
    prev_action1 = pTarget12_;

    if(isnan(totalReward))
      totalReward = 0;

    if(!mStopFlag) 
      return totalReward;
    else
      return -1E10;
  }

  float CalcRewardImitate()
  {   
    float reward = 0;
    double pose_err = 0;
    double vel_err = 0;
    double end_eff_err = 0;
    double root_pose_err = 0;
    double root_pose_rot_err = 0;
    double root_vel_err = 0;
    double root_vel_rot_err = 0;

    alien_->getState(gc_, gv_);
    if(isnan(gc_[0]))
    {
      return 0;
    }

    pose_err = CalcPoseErr();
    vel_err =  CalcVelErr();
    end_eff_err = CalcEndErr();
    CalcRootPosErr(root_pose_err, root_pose_rot_err);
    CalcRootVelErr(root_vel_err, root_vel_rot_err);

    pose_reward = exp(-pose_scale * pose_err);
    vel_reward = exp(-vel_scale * vel_err);
    end_eff_reward = exp(-end_eff_scale * end_eff_err);
    root_pose_reward = exp(-root_pose_scale * root_pose_err);
    root_vel_reward = exp(-root_vel_scale * root_vel_err -root_vel_rot_scale * root_vel_rot_err);

    double augmented_w = 1.0 - root_vel_w - vel_w;
    // reward = pose_w * pose_reward + vel_w * vel_reward + end_eff_w * end_eff_reward
    //   + root_pose_w * root_pose_reward + root_vel_w * root_vel_reward;
    // reward = augmented_w * pose_reward * end_eff_reward * root_pose_reward + root_vel_w * root_vel_reward + vel_w * vel_reward;
    reward = pose_reward * end_eff_reward;

    return reward;
  }

  double CalcPoseErr(){
    double pose_err = 0.0;
    int start = 7;
    // Eigen::VectorXd scaler_temp;
    // scaler.setZero(nJoints_);
    // scaler.setConstant(1.0);

    for(int i = 0 ; i <nJoints_ ; ++i)
    {
      pose_err += pow((ref_pose[i+start] - gc_[i+start]),2);
    }

    // std::cout << ref_pose_quat[7] << gc_[7] << std::endl;

    return pose_err;
  }

  double CalcVelErr(){
    double vel_err = 0.0;
    int start = 6;
    for(int i = 0 ; i <nJoints_ ; ++i)
    {
      vel_err += pow((ref_vel[i+start] - gv_[i+start]),2);
    }
    return vel_err;
  }

  double CalcEndErr(){
    double pose_err = 0.0;
    raisim::Vec<3> alien_root;
    raisim::Vec<3> ref_root;
    alien_->getBasePosition(alien_root);
    ref_dummy->getBasePosition(ref_root);

    for(int i = 2; i < names.size();++i)
    {
      raisim::Vec<3> alienPosition;
      raisim::Vec<3> refPosition;
      alien_->getFramePosition(alien_->getFrameByName(names[i]), alienPosition);
      ref_dummy->getFramePosition(ref_dummy->getFrameByName(names[i]), refPosition);
      // refPosition[1] -=coll_off;
      raisim::Vec<3> posdiff;
      vecsub(alien_root, alienPosition);
      vecsub(ref_root, refPosition);
      vecsub(refPosition, alienPosition, posdiff);
      pose_err += posdiff.squaredNorm();
    }
    return pose_err;
  }

  void CalcRootPosErr(double& pose_err, double& rot_err)
  {
    Eigen::Vector3d posediff;
    for(int i= 0 ; i < 2 ; ++i){
      posediff[i] = ref_pose[i] - gc_[i];
    }
    posediff[2]= 0;
    pose_err = posediff.squaredNorm();
    Eigen::Vector4d quat;
    for(int j = 0; j<4; ++j)
      quat[j] = gc_[j+3];
    Eigen::Vector3d axis;
    quat2euler(quat, axis);

    Eigen::Vector4d quat2;
    for(int j = 0; j<4; ++j)
      quat2[j] = ref_pose[j+3];
    Eigen::Vector3d axis2;
    quat2euler(quat2, axis2);

    Eigen::Vector3d angdiff;
    for(int i= 0 ; i < 3 ; ++i){
      angdiff[i] = axis2[i] - axis[i];
    }
    rot_err = angdiff.squaredNorm();
  }

  void CalcRootVelErr(double& pose_err, double& rot_err)
  {
    Eigen::Vector3d posediff;
    Eigen::Vector3d angdiff;
    for(int i= 0 ; i < 3 ; ++i){
      posediff[i] = ref_vel[i] - gv_[i];
      angdiff[i] = ref_vel[i+3] - gv_[i+3];
    }
    pose_err = posediff.squaredNorm();      
    rot_err = angdiff.squaredNorm();
  }

  void KeyMotionGen(double low_h, double high_h, Eigen::VectorXd timing){
    std::vector<Eigen::Vector3d> Ends;
    std::vector<bool> linear;
    std::vector<Eigen::VectorXd> Keys;
    Keys.push_back(gc_init_);
    Ends.push_back(Eigen::Vector3d(0.0,0.0,low_h));
    linear.push_back(true);
    Ends.push_back(Eigen::Vector3d(0, 0, 0));
    linear.push_back(false);


    for(int i = 2; i < names.size();++i)
    {
      raisim::Vec<3> footPosition;
      ref_dummy->getFramePosition(ref_dummy->getFrameByName(names[i]), footPosition);
      Ends.push_back(Eigen::Vector3d(footPosition[0],footPosition[1],0));
      linear.push_back(true);
    }
    Keys.push_back(mIK->solve(Ends, linear, 10000));
    Ends[0] = Eigen::Vector3d(0.0, 0.0, high_h);
    Keys.push_back(mIK->solve(Ends, linear, 10000));

    mMP = new MotionPlanner(Keys, control_fps, timing);
    motionClip_ = mMP->getResult();
    speedClip_ = mMP->getVelocity();

  }

  void updateExtraInfo() final {
    extraInfo_["base height"] = gc_[2];
  }

  void updateObservation() {
    alien_->getState(gc_, gv_);
    obScaled_.setZero(obDim_);

    Eigen::Vector4d quat;
    Eigen::Vector3d euler;
    for(int i=0;i<4;++i)
      quat[i] = gc_[i+3];
    quat2euler(quat, euler);

    Eigen::Vector4d quat1;
    Eigen::Vector3d euler1;
    for(int i=0;i<4;++i)
      quat1[i] = gc_prev1[i+3];
    quat2euler(quat1, euler1);

    Eigen::Vector4d quat2;
    Eigen::Vector3d euler2;
    for(int i=0;i<4;++i)
      quat2[i] = gc_prev2[i+3];
    quat2euler(quat2, euler2);

    Eigen::VectorXd prediction;
    prediction = ref_pose;

    Eigen::Vector4d quat_pred;
    Eigen::Vector3d euler_pred;
    for(int i=0;i<4;++i)
      quat_pred[i] = prediction[i+3];
    quat2euler(quat_pred, euler_pred);

    /// update observations
    obScaled_.segment(0, 3) = euler2;
    obScaled_.segment(3, 12) = gc_prev2.tail(12);
    obScaled_.segment(15, 3) = euler1;
    obScaled_.segment(18, 12) = gc_prev1.tail(12);
    obScaled_.segment(30, 3) = euler;
    obScaled_.segment(33, 12) = gc_.tail(12);

    obScaled_.segment(45, 12) = prev_action3;
    obScaled_.segment(57, 12) = prev_action2;
    obScaled_.segment(69, 12) = prev_action1;
    obScaled_.segment(81, 3) = prediction.head(3);
    obScaled_.segment(84, 3) = euler_pred;
    obScaled_.segment(87, 12) = prediction.tail(12);
    obScaled_[99] = min_h;

    /// update previous
    gc_prev2 = gc_prev1;
    gc_prev1 = gc_;
  }

  void SetFromDevice(InputSettings inputSettings) {
    VERIFY(k4a_device_open(0, &device), "Open K4A Device failed!");

    // Start camera. Make sure depth camera is enabled.
    k4a_device_configuration_t deviceConfig = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    deviceConfig.depth_mode = inputSettings.DepthCameraMode;
    deviceConfig.color_resolution = K4A_COLOR_RESOLUTION_OFF;
    VERIFY(k4a_device_start_cameras(device, &deviceConfig), "Start K4A cameras failed!");

    // Get calibration information
    k4a_calibration_t sensorCalibration;
    VERIFY(k4a_device_get_calibration(device, deviceConfig.depth_mode, deviceConfig.color_resolution, &sensorCalibration),
        "Get depth camera calibration failed!");
    int depthWidth = sensorCalibration.depth_camera_calibration.resolution_width;
    int depthHeight = sensorCalibration.depth_camera_calibration.resolution_height;

    // Create Body Tracker
    k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
    tracker_config.processing_mode = inputSettings.CpuOnlyMode ? K4ABT_TRACKER_PROCESSING_MODE_CPU : K4ABT_TRACKER_PROCESSING_MODE_GPU;
    VERIFY(k4abt_tracker_create(&sensorCalibration, tracker_config, &tracker), "Body tracker initialization failed!");
  }

  void CloseDevice(){
    std::cout << "Finished body tracking processing!" << std::endl;

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
        uint32_t numBodies = k4abt_frame_get_num_bodies(bodyFrame);
        for (uint32_t i = 0; i < numBodies; i++)
        {
          k4abt_body_t body;
          VERIFY(k4abt_frame_get_body_skeleton(bodyFrame, i, &body.skeleton), "Get skeleton from body frame failed!");
          body.id = k4abt_frame_get_body_id(bodyFrame, 0);

          k4a_float3_t leftWristJoint = body.skeleton.joints[K4ABT_JOINT_WRIST_LEFT].position;
          k4a_float3_t leftElbowJoint = body.skeleton.joints[K4ABT_JOINT_ELBOW_LEFT].position;
          k4a_float3_t leftShoulderJoint = body.skeleton.joints[K4ABT_JOINT_SHOULDER_LEFT].position;
          points[0] = scaler * Eigen::Vector3d(-(leftShoulderJoint.xyz.x - leftShoulderJoint.xyz.x), (leftShoulderJoint.xyz.z - leftShoulderJoint.xyz.z), (leftShoulderJoint.xyz.y - leftShoulderJoint.xyz.y));
          points[1] = scaler * Eigen::Vector3d(-(leftElbowJoint.xyz.x - leftShoulderJoint.xyz.x), (leftElbowJoint.xyz.z - leftShoulderJoint.xyz.z), (leftElbowJoint.xyz.y - leftShoulderJoint.xyz.y));
          points[2] = scaler * Eigen::Vector3d(-(leftWristJoint.xyz.x - leftShoulderJoint.xyz.x), (leftWristJoint.xyz.z - leftShoulderJoint.xyz.z), (leftWristJoint.xyz.y - leftShoulderJoint.xyz.y));
          for(int i =0 ; i < points.size();++i){
            points[i][0] +=x_off;
            points[i][1] +=y_off;
            points[i][2] +=z_offset;
          }
          double angle = angle_finder(points[0], points[1], points[2]);
          if(startFlag)
          {
            base_angle = angle;
            startFlag = false;
          }
          // Mapping Height
          angle -= base_angle;
          double weight = 0.15;
          double desired_h = 0.43 + angle * weight;
          auto smallIK = new IKSolver(ref_dummy, names);

          std::vector<Eigen::Vector3d> Ends;
          std::vector<bool> linear;
          Ends.push_back(Eigen::Vector3d(0.0,0.0,desired_h));
          linear.push_back(true);
          Ends.push_back(Eigen::Vector3d(0, 0, 0));
          linear.push_back(false);
          for(int i = 2; i < names.size();++i)
          {
            raisim::Vec<3> footPosition;
            ref_dummy->getFramePosition(ref_dummy->getFrameByName(names[i]), footPosition);
            Ends.push_back(Eigen::Vector3d(footPosition[0],footPosition[1],0));
            linear.push_back(true);
          }
          ref_pose = smallIK->solve(Ends, linear, 1000);
        }
        //Release the bodyFrame    
        k4abt_frame_release(bodyFrame);
    }
  }

  float angle_finder(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d c)
  {
    float A = (b-c).norm();
    float B = (c-a).norm();
    float C = (a-b).norm();
    float angle = acos((C*C + A*A - B*B)/(2 * C * A));
    return angle;
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obScaled_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = 0.f;

    /// if the contact body is not feet
    for(auto& contact: alien_->getContacts())
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end()) {
        return true;
      }

    if(terminalFlag)
      return true;

    return false;
  }

  void setSeed(int seed) final {
    std::srand(seed);
  }

  void close() final {
  }

  void quat2euler(Eigen::Vector4d quat, Eigen::Vector3d& axis){
    const double norm = (std::sqrt(quat[1]*quat[1] + quat[2]*quat[2] + quat[3]*quat[3]));
    if(fabs(norm) < 1e-12) {
      axis[0] = 0;
      axis[1] = 0;
      axis[2] = 0;
    }
    else{
      const double normInv = 1.0/norm;
      const double angleNomrInv = std::acos(std::min(quat[0],1.0))*2.0*normInv;
      axis[0] = quat[1] * angleNomrInv;
      axis[1] = quat[2] * angleNomrInv;
      axis[2] = quat[3] * angleNomrInv;
    }
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
  Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_;
  Eigen::VectorXd obDouble_, obScaled_;
  std::set<size_t> footIndices_;
  int control_fps;

  double pose_w, vel_w, end_eff_w, root_pose_w, root_vel_w;
  double pose_scale, vel_scale, end_eff_scale, root_pose_scale, root_pose_rot_scale, root_vel_scale, root_vel_rot_scale;
  double pose_reward, vel_reward, end_eff_reward, root_pose_reward, root_vel_reward;

  float x_off, y_off;
  int joint_st = 7;

  Eigen::VectorXd prev_action3, prev_action2, prev_action1;

  int motion_count;
  std::vector<std::string> motion_seed;
  std::vector<std::string> names;
  int turn_signal;
  std::string motion_path;
  Eigen::VectorXd ref_pose ,ref_pose_quat;
  Eigen::VectorXd ref_vel;
  float coll_off = -2.0;
  int motion_times;
  bool terminalFlag;
  int random_signal, wait_count;

  IKSolver* mIK;
  MotionPlanner* mMP;
  std::vector<Eigen::VectorXd> motionClip_;
  std::vector<Eigen::VectorXd> speedClip_;
  bool jumpFlag;
  int balance_count, startpoint;
  int terminal_count;
  double min_h;
  int steady_count = 20;
  int current_count;

  bool startFlag;
  double base_angle;
  float z_offset = 0.85;
  k4a_device_t device = nullptr;
  k4abt_tracker_t tracker = nullptr;
  std::vector<Eigen::Vector3d> points;
  k4a_float3_t startOrigin;
  float scaler;
  float angle_smoother;
  
};

}

