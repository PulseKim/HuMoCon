#include <stdlib.h>
#include <cstdint>
#include <set>
#include <chrono>
#include <random>
#include <fstream>
#include <raisim/OgreVis.hpp>
#include "RaisimGymEnv.hpp"
#include "visSetupCallback2.hpp"
#include "motionParser.hpp"
#include "BVHCallBack.hpp"
#include "Utilities.h"
#include "math.h"

#include <k4arecord/playback.h>
#include <k4a/k4a.h>
#include <k4abt.h>

#include "../visualizer_alien/raisimKeyboardCallback.hpp"
#include "../visualizer_alien/helper.hpp"
#include "../visualizer_alien/guiState.hpp"
#include "../visualizer_alien/raisimBasicImguiPanel.hpp"

#define deg2rad(ang) ((ang) * M_PI / 180.0)
#define rad2deg(ang) ((ang) * 180.0 / M_PI)

std::uniform_real_distribution<double> angle_distribution(deg2rad(-20),deg2rad(20));

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
    gc_init_ << 0, 0, 0.43, 1.0, 0.0, 0.0, 0.0, 0.045377, 0.667921, -1.23225,0.045377, 0.667921, -1.23225, 0.045377, 0.667921, -1.23225, 0.045377, 0.667921, -1.23225;
    alien_->setState(gc_init_, gv_init_);

    /// set pd gains
    double pgain = 350;
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(pgain);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(2*std::sqrt(pgain));
    alien_->setPdGains(jointPgain, jointDgain);
    alien_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
    pTarget_.tail(nJoints_) = gc_init_.tail(12);
    alien_->setPdTarget(pTarget_, vTarget_);

    desired_angle = 0;

    // Have to implement this part
    obDim_ = 5 + nJoints_; 
    actionDim_ = nJoints_;
    actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obMean_.setZero(obDim_); obStd_.setZero(obDim_);

    /// action & observation scaling
    actionMean_ = gc_init_.tail(nJoints_);
    // actionStd_.setConstant(0.6);
    actionStd_ << 0.3, 0.6, 0.6, 0.3, 0.6, 0.6, 0.3, 0.6, 0.6, 0.3, 0.6, 0.6;

    obMean_ << 0,
               Eigen::VectorXd::Constant(4, 0.0), 
               gc_init_.tail(nJoints_);

    obStd_ << 0 , /// joint angles
              Eigen::VectorXd::Constant(4, 0.0),
              Eigen::VectorXd::Constant(nJoints_, 0.7);

    angle_w = 0.3;
    end_w = 0.3;

    angle_scale = 30;
    end_scale = 5;
    current_count = 0;
    change_count = 30;

    scaler = -1.0 /800;
    angle_smoother = 0.5;

    gui::rewardLogger.init({"AngleReward", "EndReward"});

    end_names.push_back("FR_foot_fixed");
    end_names.push_back("FL_foot_fixed");
    end_names.push_back("RR_foot_fixed");
    end_names.push_back("RL_foot_fixed");

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
      vis->createGraphicalObject(ground, 10, "floor", "checkerboard_green");
      desired_fps_ = 60.;
      vis->setDesiredFPS(desired_fps_);

      vis->addVisualObject("random_arrow1", "arrowMesh", "red", {0.1, 0.1, 0.2}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("random_arrow2", "arrowMesh", "blue", {0.1, 0.1, 0.2}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);

      auto& list = raisim::OgreVis::get()->getVisualObjectList();
      Eigen::Matrix3d rot1;
      rot1 = Eigen::AngleAxisd(0.5*M_PI,  Eigen::Vector3d::UnitY());
      Eigen::Matrix3d rot2;
      Eigen::Quaterniond quat(1, 0, 0, 0);
      rot2 = Eigen::AngleAxisd(quat) * Eigen::AngleAxisd(0.5*M_PI,  Eigen::Vector3d::UnitY());

      list["random_arrow1"].offset = {0,0,0.49};
      list["random_arrow1"].scale = {0.3,0.3,0.3};
      list["random_arrow1"].setOrientation(rot1);
      list["random_arrow2"].offset = {0,0,0.49};
      list["random_arrow2"].scale = {0.3,0.3,0.3};
      list["random_arrow2"].setOrientation(rot2);
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
    desired_angle = 0;
    current_count = 0;
    pTarget_.setZero();
    pTarget_.tail(nJoints_) = gc_init_.tail(12);
    alien_->setPdTarget(pTarget_, vTarget_);

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
    // currentAction *= 0.5;
    pTarget12_ = currentAction;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += 1.0 * actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

    alien_->setPdTarget(pTarget_, vTarget_);

    auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
    auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);    

    
    for(int i=0; i<loopCount; i++) {
      world_->integrate();  
      if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
      {
        auto vis = raisim::OgreVis::get();
        alien_->getState(gc_, gv_);
        auto& list = raisim::OgreVis::get()->getVisualObjectList();
        list["random_arrow1"].offset = {gc_[0],gc_[1],gc_[2] + 0.1};
        list["random_arrow2"].offset = {gc_[0],gc_[1],gc_[2] + 0.1};

        Eigen::Matrix3d rot2;
        Eigen::Quaterniond quat(gc_[3], gc_[4], gc_[5], gc_[6]);
        rot2 = Eigen::AngleAxisd(quat) * Eigen::AngleAxisd(0.5*M_PI,  Eigen::Vector3d::UnitY());
        list["random_arrow2"].setOrientation(rot2);
        
        vis->renderOneFrame();
      }
      
      visualizationCounter_++;
    }

    updateObservation();

    float totalReward = CalcReward();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("AngleReward", 1.0 * angle_reward);
      gui::rewardLogger.log("EndReward",  1.0 * end_reward);

      /// reset camera
      auto vis = raisim::OgreVis::get();

      vis->select(anymalVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }
    
    current_count++;
    if(current_count == change_count){
      update_leaning_angle();
      Eigen::Matrix3d rot1;
      rot1 = Eigen::AngleAxisd(desired_angle,  Eigen::Vector3d::UnitY())
            * Eigen::AngleAxisd(0.5*M_PI,  Eigen::Vector3d::UnitY());
      auto& list = raisim::OgreVis::get()->getVisualObjectList();
      list["random_arrow1"].setOrientation(rot1);
      current_count = 0;
    }

    if(!mStopFlag) 
      return totalReward;
    else
      return -1E10;
  }

  float stepTest(const Eigen::Ref<EigenVec>& action) final {
   /// action scaling
    Eigen::VectorXd currentAction = action.cast<double>();
    // currentAction *= 0.5;
    pTarget12_ = currentAction;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += 1.0 * actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

    alien_->setPdTarget(pTarget_, vTarget_);

    auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
    auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);    

    
    for(int i=0; i<loopCount; i++) {
      world_->integrate();
      GetFromDevice();  
      if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
      {
        auto vis = raisim::OgreVis::get();
        alien_->getState(gc_, gv_);
        auto& list = raisim::OgreVis::get()->getVisualObjectList();
        list["random_arrow1"].offset = {gc_[0],gc_[1],gc_[2] + 0.1};
        list["random_arrow2"].offset = {gc_[0],gc_[1],gc_[2] + 0.1};

        Eigen::Matrix3d rot2;
        Eigen::Quaterniond quat(gc_[3], gc_[4], gc_[5], gc_[6]);
        rot2 = Eigen::AngleAxisd(quat) * Eigen::AngleAxisd(0.5*M_PI,  Eigen::Vector3d::UnitY());
        list["random_arrow2"].setOrientation(rot2);
        Eigen::Matrix3d rot1;
        rot1 = Eigen::AngleAxisd(desired_angle,  Eigen::Vector3d::UnitY())
              * Eigen::AngleAxisd(0.5*M_PI,  Eigen::Vector3d::UnitY());
        list["random_arrow1"].setOrientation(rot1);
        for(int i =0 ; i < points.size();++i){
          list["sphere" + std::to_string(i)].setPosition(points[i]);
        }
        vis->renderOneFrame();
      }
      visualizationCounter_++;
    }

    updateObservation();

    float totalReward = CalcReward();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("AngleReward", 1.0 * angle_reward);
      gui::rewardLogger.log("EndReward",  1.0 * end_reward);

      /// reset camera
      auto vis = raisim::OgreVis::get();

      vis->select(anymalVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }

    if(!mStopFlag) 
      return totalReward;
    else{
      CloseDevice();
      return -1E10;
    }
  }

  float CalcReward()
  {   
    float reward = 0;
    double angle_err = 0;
    double end_err = 0;

    alien_->getState(gc_, gv_);

    angle_err = CalcAngleErr();
    end_err = CalcEndErr();

    angle_reward = exp(-angle_scale * angle_err);
    end_reward = exp(-end_scale * end_err);

    reward = angle_reward * end_reward;

    return reward;
  }

  float CalcAngleErr()
  {
    Eigen::Quaterniond q;  
    q = Eigen::AngleAxisd(desired_angle, Eigen::Vector3d(0,1,0));
    Eigen::Quaterniond current(gc_[3], gc_[4], gc_[5], gc_[6]);
    float diff = current.angularDistance(q);
    return diff * diff;    
  }
  float CalcEndErr()
  {
    float diff = 0.0;
    for(int i =0 ; i < end_names.size() ; ++i)
    {
      raisim::Vec<3> tempPosition;
      alien_->getFramePosition(alien_->getFrameByName(end_names[i]), tempPosition);
      diff += tempPosition[2] * tempPosition[2];
    }
    return diff;
  }

  void updateExtraInfo() final {
    extraInfo_["base height"] = gc_[2];
  }

  void updateObservation() {
    alien_->getState(gc_, gv_);
    obDouble_.setZero(obDim_); obScaled_.setZero(obDim_);

    // /// update observations
    // obDouble_[0] = desired_angle;
    // obDouble_.segment(1, 4) = gc_.segment(3, 4);
    // obDouble_.segment(5, nJoints_) = gc_.tail(nJoints_);

    // obScaled_ = (obDouble_-obMean_).cwiseQuotient(obStd_);
    obScaled_[0] = desired_angle;
    obScaled_.segment(1, 4) = gc_.segment(3, 4);
    obScaled_.segment(4, nJoints_) = gc_.tail(nJoints_);
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
            points[i][2] +=z_offset;
          }
          double angle = angle_finder(points[0], points[1], points[2]);
          if(startFlag)
          {
            base_angle = angle;
            startFlag = false;
          }
          desired_angle = (angle - base_angle) * angle_smoother;
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

  void update_leaning_angle()
  {
    unsigned seed_temp = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed_temp);
    desired_angle += angle_distribution(generator);
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

  double desired_angle;
  int change_count, current_count;

  double angle_w, end_w;
  double angle_scale, end_scale;
  double angle_reward, end_reward;

  float z_offset = 0.85;
  k4a_device_t device = nullptr;
  k4abt_tracker_t tracker = nullptr;
  std::vector<Eigen::Vector3d> points;
  k4a_float3_t startOrigin;
  float scaler;
  float angle_smoother;
  bool startFlag;
  double base_angle;
  std::vector<std::string> end_names;

};

}
