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
#include "IKIPopt.hpp"
#include "MotionPlanner.hpp"
#include "RobotInfo.hpp"

#include <k4arecord/playback.h>
#include <k4a/k4a.h>
#include <k4abt.h>

#include <IpTNLP.hpp>
#include <IpIpoptApplication.hpp>

#include "../visualizer_alien/raisimKeyboardCallback.hpp"
#include "../visualizer_alien/helper.hpp"
#include "../visualizer_alien/guiState.hpp"
#include "../visualizer_alien/raisimBasicImguiPanel.hpp"

// This code is for learning pace with desried speed

#define deg2rad(ang) ((ang) * M_PI / 180.0)
#define rad2deg(ang) ((ang) * 180.0 / M_PI)

#define LEFT_HIP 10
#define LEFT_THIGH 11
#define LEFT_CALF 12

std::uniform_real_distribution<double> speed_distribution(0.1, 3.0);

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

    READ_YAML(double, control_dt_, cfg["control_dt"])

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    torque_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);
    
    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.37, 1.0, 0.0, 0.0, 0.0, 0.000356295, 0.765076, -1.5303, 5.69796e-05, 0.765017, -1.53019, 0.000385732, 0.765574, -1.53102, 8.34913e-06, 0.765522, -1.53092;
    alien_->setState(gc_init_, gv_init_);
    ref_dummy->setState(gc_init_, gv_init_);

    /// set pd gains
    double pgain = 120;
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(pgain);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(1.4 * std::sqrt(pgain));
    alien_->setPdGains(jointPgain, jointDgain);
    alien_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    obDim_ = 72;
    actionDim_ = nJoints_;
    actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obMean_.setZero(obDim_); obStd_.setZero(obDim_);

    /// action & observation scaling
    actionMean_ = gc_init_.tail(nJoints_);
    // actionStd_.setConstant(0.6);
    actionStd_ << 0.2, 0.6, 0.6, 0.2, 0.6, 0.6, 0.2, 0.6, 0.6, 0.2, 0.6, 0.6;

    speed_w = 0.7;
    tracking_w = 0.2;
    torque_w = 0.1;

    speed_scale = 20.;
    track_scale = 10.;
    rootrot_scale = 5.;
    torque_scale = 4e-5;

    current_count = 0;
    change_count = 24;
    scaler = -1.0 / 2000;

    gui::rewardLogger.init({"SpeedReward", "TracktorReward", "RootRotReward", "TorqueReward"});

    /// indices of links that should not make contact with ground
    footIndices_.insert(alien_->getBodyIdx("FL_calf"));
    footIndices_.insert(alien_->getBodyIdx("FR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RL_calf"));

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
    for(int i =0 ; i < 6;++i){ 
      vis->addVisualObject("sphere" + std::to_string(i), "sphereMesh", "orange", {0.02, 0.02, 0.02}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
    }
    for(int i =0 ; i < 5; ++i){
      vis->addVisualObject("bone" + std::to_string(i), "cylinderMesh", "red", {0.015, 0.015, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
    }
    vis->addVisualObject("sphere_indicator_pr", "sphereMesh", "orange", {0.03, 0.03, 0.03}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);

    connection.push_back(3);
    connection.push_back(4);
    connection.push_back(5);
    connection.push_back(0);
    connection.push_back(1);
    connection.push_back(2);
    raisim::Vec<3> tempPosition;
    ref_dummy->getFramePosition(ref_dummy->getFrameByName("FL_foot_fixed"), tempPosition);
    // pp = tempPosition.e(); 

    startFlag = true;
    deviceFlag = false;
    for(int i = 0; i < 6; ++i)
      points.push_back(Eigen::Vector3d(0,0,0));
  }

  void reset() final
  {
    alien_->setState(gc_init_, gv_init_);
    ref_dummy ->setState(gc_init_, gv_init_);
    
    current_count = 0;
    pTarget_.setZero();
    pTarget_.tail(nJoints_) = gc_init_.tail(12);
    alien_->setPdTarget(pTarget_, vTarget_);
    current_speed = 0.0;
    update_desired_speed();

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
    Eigen::VectorXd currentAction = action.cast<double>();
    pTarget12_ = currentAction;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += 0.3 * actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;
    alien_->setPdTarget(pTarget_, vTarget_);

    auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
    auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);    
    current_count++;

    Eigen::Matrix3d body_rot;
    Eigen::Vector3d world_vel;
    Eigen::Vector3d body_vel;
    for(int i=0; i<loopCount; i++) {
      world_->integrate();  
      alien_->getState(gc_, gv_);
      body_rot = alien_->getBaseOrientation().e();
      world_vel = Eigen::Vector3d(gv_[0], gv_[1], 0);
      body_vel = body_rot.inverse() * world_vel;
      current_speed = body_vel[0];

      if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
      {
        auto vis = raisim::OgreVis::get();
        visualize_arrow();        
        vis->renderOneFrame();
      }      
      visualizationCounter_++;
    }

    updateObservation();

    float totalReward = CalcReward();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("SpeedReward",  speed_reward);
      gui::rewardLogger.log("TracktorReward", track_reward);
      gui::rewardLogger.log("RootRotReward",  rootrot_reward);
      gui::rewardLogger.log("TorqueReward",  torque_reward);

      /// reset camera
      auto vis = raisim::OgreVis::get();

      vis->select(anymalVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }
    
    if(current_count%change_count == 0){
      update_desired_speed();
    }

    if(!mStopFlag) 
      return totalReward;
    else
      return -1E10;
  }

  float stepTest(const Eigen::Ref<EigenVec>& action) final {
    Eigen::VectorXd currentAction = action.cast<double>();
    pTarget12_ = currentAction;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += 0.3 * actionMean_;
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
      current_speed = body_vel[0];
      if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
      {
        auto vis = raisim::OgreVis::get();
        auto& list = raisim::OgreVis::get()->getVisualObjectList();
        visualize_arrow();

        for(int i =0 ; i < points.size();++i){
          list["sphere" + std::to_string(i)].setPosition(points[i]);
        }
        for(int i =0 ; i < connection.size()-1;++i){
          Eigen::Vector3d half = (points[connection[i+1]] + points[connection[i]]) / 2 ;
          float len = (points[connection[i+1]] - points[connection[i]]).norm();
          raisim::Mat<3,3> rot;
          Eigen::Vector3d way = (points[connection[i+1]] - points[connection[i]]).normalized();
          raisim::Vec<3> direction = {way[0], way[1], way[2]};
          raisim::zaxisToRotMat(direction, rot);

          list["bone" + std::to_string(i)].setPosition(half);
          list["bone" + std::to_string(i)].setScale(Eigen::Vector3d(0.015, 0.015, len));
          list["bone" + std::to_string(i)].setOrientation(rot);
        }
        vis->renderOneFrame();
        if(startFlag){
          vis->select(anymalVisual_->at(0), false);
          vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.1), 3, true);
        }
      }      
      visualizationCounter_++;  
    }
    updateObservation();

    float totalReward = CalcReward();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("SpeedReward",  speed_reward);
      gui::rewardLogger.log("TracktorReward", track_reward);
      gui::rewardLogger.log("RootRotReward",  rootrot_reward);
      gui::rewardLogger.log("TorqueReward",  torque_reward);

      /// reset camera
      // auto vis = raisim::OgreVis::get();

      // vis->select(anymalVisual_->at(0), false);
      // vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.1), 3, true);
    }

    current_count++;
    if(current_count%change_count == 0){
      update_desired_speed();
    }

    if(!mStopFlag) 
      return totalReward;
    else{
      CloseDevice();
      return -1E10;
    }
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
    Eigen::Vector3d half2 = half;
    half2[2] +=arrow_offset;
    list["ref_speed_indicator"].setPosition(half);
    list["cur_speed_indicator"].setPosition(half2);
    list["ref_speed_indicator"].setScale(Eigen::Vector3d(0.1, 0.1, desired_speed * 0.1));
    list["cur_speed_indicator"].setScale(Eigen::Vector3d(0.1, 0.1, current_speed * 0.1));
    list["ref_speed_indicator"].setOrientation(rot);
    list["cur_speed_indicator"].setOrientation(rot); 
  }

  float CalcReward()
  {   
    float reward = 0;
    double speed_err = 0;
    double track_err = 0;
    double rootrot_err = 0;
    double torque_err = 0;

    alien_->getState(gc_, gv_);
    if(isnan(gc_[0])) 
      return 0;
    speed_err = CalcSpeedErr();
    track_err = CalcTrackErr();
    rootrot_err = CalcRootRotErr();
    torque_err = CalcTorqueErr();

    speed_reward = exp(-speed_scale * speed_err);
    track_reward = exp(-track_scale * track_err);
    rootrot_reward = exp(-rootrot_scale * rootrot_err);
    torque_reward = exp(-torque_scale * torque_err);

    reward = speed_w * speed_reward +  tracking_w * track_reward * rootrot_reward + torque_w * torque_reward;

    return reward;
  }

  float CalcSpeedErr()
  {
    float speed_diff = current_speed - desired_speed;
    return speed_diff * speed_diff;
  }

  float CalcTrackErr()
  {
    float track_diff = gc_[1];
    return track_diff * track_diff;
  }

  float CalcRootRotErr()
  {
    Eigen::Vector3d xvec(1.0, 0.0, 0.0);
    float rot_err = 0;
    raisim::Vec<4> quat;
    raisim::Mat<3, 3> R;
    for(int j = 0; j<4; ++j)
      quat[j] = gc_[j+3];
    quatToRotMat(quat, R);
    Eigen::Matrix3d R_curr = R.e();
    Eigen::Vector3d current = R_curr * xvec;
    current[2] = 0.0;

    float angle;
    if(current.norm() < 1e-10)
      angle = 0;
    else
      angle = acos(current.dot(xvec)/current.norm());
    rot_err = angle * angle;
    return rot_err;
  }
  float CalcTorqueErr()
  {
    float total_torque = alien_->getGeneralizedForce().squaredNorm();
    return total_torque;
  }

  void updateExtraInfo() final {
    extraInfo_["base height"] = gc_[2];
  }

  void updateObservation() {
    alien_->getState(gc_, gv_);
    obScaled_.setZero(obDim_);

    /// update observations
    obScaled_[0] = desired_speed;
    obScaled_[1] = current_speed;
    obScaled_[2] = gc_[2];
    obScaled_.segment(3, 12) = gc_.tail(12);
    Eigen::Quaterniond quat(gc_[3], gc_[4], gc_[5], gc_[6]);
    Eigen::AngleAxisd aa(quat);
    Eigen::Vector3d rotation = aa.angle()*aa.axis();
    obScaled_.segment(15, 3) = rotation;
    obScaled_.segment(18, 3) = gv_.segment(0,3);
    auto frameNames = allFrames();
    for(int i = 0; i < frameNames.size(); ++i){
      raisim::Vec<3> tempPose;
      alien_->getFramePosition(alien_->getFrameByName(frameNames[i]), tempPose);
      obScaled_.segment(21 + 3*i, 3) = tempPose.e();
    }
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
        // uint32_t numBodies = k4abt_frame_get_num_bodies(bodyFrame);
        uint32_t numBodies = 1;
        for (uint32_t i = 0; i < numBodies; i++)
        {
          k4abt_body_t body;
          VERIFY(k4abt_frame_get_body_skeleton(bodyFrame, i, &body.skeleton), "Get skeleton from body frame failed!");
          body.id = k4abt_frame_get_body_id(bodyFrame, 0);

          k4a_float3_t leftWristJoint = body.skeleton.joints[K4ABT_JOINT_WRIST_LEFT].position;
          k4a_float3_t leftElbowJoint = body.skeleton.joints[K4ABT_JOINT_ELBOW_LEFT].position;
          k4a_float3_t leftShoulderJoint = body.skeleton.joints[K4ABT_JOINT_SHOULDER_LEFT].position;
          k4a_float3_t leftClavicleJoint = body.skeleton.joints[K4ABT_JOINT_CLAVICLE_LEFT].position;
          k4a_float3_t rightClavicleJoint = body.skeleton.joints[K4ABT_JOINT_CLAVICLE_RIGHT].position;
          k4a_float3_t NeckJoint = body.skeleton.joints[K4ABT_JOINT_NECK].position;
          NeckOrientation = body.skeleton.joints[K4ABT_JOINT_NECK].orientation;
          
          points[0] = scaler * Eigen::Vector3d((leftShoulderJoint.xyz.x - NeckJoint.xyz.x), (leftShoulderJoint.xyz.z - NeckJoint.xyz.z), (leftShoulderJoint.xyz.y - NeckJoint.xyz.y));
          points[1] = scaler * Eigen::Vector3d((leftElbowJoint.xyz.x - NeckJoint.xyz.x), (leftElbowJoint.xyz.z - NeckJoint.xyz.z), (leftElbowJoint.xyz.y - NeckJoint.xyz.y));
          points[2] = scaler * Eigen::Vector3d((leftWristJoint.xyz.x - NeckJoint.xyz.x), (leftWristJoint.xyz.z - NeckJoint.xyz.z), (leftWristJoint.xyz.y - NeckJoint.xyz.y));
          points[3] = scaler * Eigen::Vector3d((rightClavicleJoint.xyz.x - NeckJoint.xyz.x), (rightClavicleJoint.xyz.z - NeckJoint.xyz.z), (rightClavicleJoint.xyz.y - NeckJoint.xyz.y));
          points[4] = scaler * Eigen::Vector3d((NeckJoint.xyz.x - NeckJoint.xyz.x), (NeckJoint.xyz.z - NeckJoint.xyz.z), (NeckJoint.xyz.y - NeckJoint.xyz.y));
          points[5] = scaler * Eigen::Vector3d((leftClavicleJoint.xyz.x - NeckJoint.xyz.x), (leftClavicleJoint.xyz.z - NeckJoint.xyz.z), (leftClavicleJoint.xyz.y - NeckJoint.xyz.y));
          for(int i =0 ; i < points.size();++i){
            points[i][2] +=z_offset;
          }
          deviceFlag = true;
        }
        //Release the bodyFrame    
        k4abt_frame_release(bodyFrame);
    }
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obScaled_.cast<float>();
  }

  void update_desired_speed()
  {
    unsigned seed_temp = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed_temp);
    desired_speed = speed_distribution(generator);
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
  raisim::ArticulatedSystem* ref_dummy;
  std::vector<GraphicObject> * anymalVisual_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, torque_;
  double desired_fps_ = 60.;
  int visualizationCounter_=0;
  Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_;
  Eigen::VectorXd obDouble_, obScaled_;
  std::set<size_t> footIndices_;

  double current_speed, desired_speed;
  bool startFlag, deviceFlag;
  float arrow_offset = 0.06;

  int change_count, current_count;

  double speed_w, tracking_w, torque_w;
  double speed_scale, track_scale, rootrot_scale, torque_scale;
  double speed_reward, track_reward, rootrot_reward, torque_reward;

  // Kinect Settings
  float z_offset = 0.85;
  k4a_device_t device = nullptr;
  k4abt_tracker_t tracker = nullptr;
  std::vector<Eigen::Vector3d> points;
  std::vector<int> connection;
  k4a_float3_t startOrigin;
  float scaler;
  k4a_quaternion_t NeckOrientation;
};

}