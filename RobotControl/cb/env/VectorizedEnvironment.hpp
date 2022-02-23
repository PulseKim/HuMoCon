#ifndef SRC_GYMVECENV_HPP
#define SRC_GYMVECENV_HPP

#include <chrono>
#include <Eigen/Core>
#include <fstream>
#include "Control1.hpp"
#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include <k4arecord/playback.h>
#include <k4a/k4a.h>
#include <k4abt.h>
#include <raisim/OgreVis.hpp>

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "../include/BodyTrakingHelper.hpp"
#include "../include/PointCloudGenerator.hpp"
#include "../include/FloorDetector.hpp"
#include "../include/MotionFunction.hpp"
#include "../include/TrajectoryGenerator.hpp"
#include "../include/raisimBasicImguiPanel.hpp"
#include "../include/raisimKeyboardCallback.hpp"

using Dtype=float;
using EigenRowMajorMat=Eigen::Matrix<Dtype, -1, -1, Eigen::RowMajor>;
using EigenVec=Eigen::Matrix<Dtype, -1, 1>;
using EigenBoolVec=Eigen::Matrix<bool, -1, 1>;

void setupCallback() {
   auto vis = raisim::OgreVis::get();

  /// light
  vis->getLight()->setDiffuseColour(1, 1, 1);
  vis->getLight()->setCastShadows(true);
  Ogre::Vector3 lightdir(-1,0.0,-0.5);
  lightdir.normalise();
  vis->getLightNode()->setDirection({lightdir});
  vis->setCameraSpeed(300);
  Ogre::ColourValue fadeColour(0.8, 0.8, 0.8);
  vis->getSceneManager()->setFog(Ogre::FOG_LINEAR , fadeColour, 0.1, 8, 100);
  vis->getViewPort()->setBackgroundColour(Ogre::ColourValue(1.0,1.0,1.0));
  /// load  textures
  vis->addResourceDirectory(vis->getResourceDir() + "/material/gravel");
  vis->loadMaterialFile("gravel.material");

  vis->addResourceDirectory(vis->getResourceDir() + "/material/checkerboard");
  vis->loadMaterialFile("checkerboard.material");

  /// shdow setting
  vis->getSceneManager()->setShadowTechnique(Ogre::SHADOWTYPE_TEXTURE_ADDITIVE);
  vis->getSceneManager()->setShadowTextureSettings(4096, 1);

  /// scale related settings!! Please adapt it depending on your map size
  // beyond this distance, shadow disappears
  vis->getSceneManager()->setShadowFarDistance(15);
  // size of contact points and contact forces
  vis->setContactVisObjectSize(0.03, 0.4);
  // speed of camera motion in freelook mode
  vis->getCameraMan()->setTopSpeed(5);
}

class VectorizedEnvironment {

 public:
  explicit VectorizedEnvironment(std::string resourceDir, std::string cfg){}

  ~VectorizedEnvironment() {
  }
  int getObDim(){return obDim_;}
  int getActionDim(){return 12;}

  void init()
  {
    world_ = std::make_unique<raisim::World>();
    // std::cout << "init " << std::endl;
    obDim_ = 164;
    actionDim_ = 12;
    gc_init_.setZero(16); gc_target.setZero(16);
    gc_init_ << 1.0, 0.0, 0.0, 0.0, 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99);
    gc_target = gc_init_;
    pTarget12_.setZero(12);
    state_history.push_back(gc_init_);
    reference_history.push_back(gc_init_);
    vis_ref = new raisim::ArticulatedSystem("/home/sonic/Project/RobotControl/cb/rsc/a1_ref/a1/a1.urdf");
    ik_dummy = new raisim::ArticulatedSystem("/home/sonic/Project/RobotControl/cb/rsc/a1/a1/a1.urdf");
    for(int i = 0; i < 3; ++i)
    {
      reference_history.push_back(gc_init_);
      state_history.push_back(gc_init_);
      action_history.push_back(gc_init_.tail(12));
    }
    actionStd_.setZero(12);  actionMean_.setZero(12);
    float ah = 0.8028, at = 2.6179, ac = 0.8901;
    float mh = 0, mt = M_PI /2, mc = -1.806;
    actionStd_ << ah, at, ac, ah, at, ac, ah, at, ac, ah, at, ac;
    actionMean_ << mh, mt, mc, mh, mt, mc, mh, mt, mc, mh, mt, mc;
    startFlag = true;

    currentContact = Eigen::Vector4i(1,1,1,1);
    prevContact = Eigen::Vector4i(1,1,1,1);
    control_dt_ = 0.03333333;

    Eigen::VectorXd gc_(19);
    gc_.head(3) = Eigen::Vector3d(0,0,0.27);
    gc_.tail(16) = gc_init_;
    human2.setZero(32); human1.setZero(32);
    ik_dummy->setGeneralizedCoordinate(gc_);
    vis_ref->setGeneralizedCoordinate(gc_);
    for(int i = 0; i < 4; ++i)
    {
      raisim::Vec<3> footPositionA;
      ik_dummy->getFramePosition(FR_FOOT+3 * i, footPositionA);
      init_feet_poses.push_back(footPositionA.e());
    }
    kin_gc_prev = gc_;
    init_foot = init_feet_poses[0][2];
    delay = 0;
    rik = new AnalyticRIK(ik_dummy, 0);
    lik = new AnalyticLIK(ik_dummy, 0);
    mAIK = new AnalyticFullIK(ik_dummy);
    current_mode = TILT;
    mrl = new MotionReshaper(ik_dummy);
    save_human_skel.setZero(3 * K4ABT_JOINT_COUNT);
    current_state = stateSTAND;
    sitcount = 0;
    standcount = 0;
    rot_compensation = Eigen::Matrix3d::Identity();
    raisim::Vec<3> thigh_location;
    ik_dummy->getFramePosition(ik_dummy->getFrameByName("FL_thigh_joint"), thigh_location);
    raisim::Vec<3> calf_location;
    ik_dummy->getFramePosition(ik_dummy->getFrameByName("FL_calf_joint"), calf_location);
    raisim::Vec<3> foot_location;
    ik_dummy->getFramePosition(ik_dummy->getFrameByName("FL_foot_fixed"), foot_location);
    float ltc = (thigh_location.e() - calf_location.e()).norm();
    float lcf = (calf_location.e() - foot_location.e()).norm();
    float dt = 0.033333;
    mTG = TrajectoryGenerator(WALKING_TROT, ltc, lcf, dt);
    mTG.set_Ae(0.11);
    phi = M_PI;
  }

  void check_delay(int recorded_delay)
  {
    delay = recorded_delay;
  }

  void settingSimulation()
  {
    // Here setting the simulation environment for testing
    alien_ = world_->addArticulatedSystem("/home/sonic/Project/RobotControl/cb/rsc/a1/a1/a1.urdf");
    world_->setTimeStep(simulation_dt_);
    world_->setERP(0,0);
    Eigen::VectorXd gc_(19);
    Eigen::VectorXd gv_(18);
    gc_.head(3) = Eigen::Vector3d(0,0.,0.27);
    gc_.tail(16) = gc_init_;
    gv_.setZero();
    alien_->setState(gc_, gv_);
    int nJoints_ = 12;
    double init_pgain = 100;
    double init_dgain = 2;
    Eigen::VectorXd jointPgain, jointDgain;
    jointPgain.setZero(18); jointPgain.tail(nJoints_).setConstant(init_pgain);
    jointDgain.setZero(18); jointDgain.tail(nJoints_).setConstant(init_dgain);
    alien_->setPdGains(jointPgain, jointDgain);
    alien_->setGeneralizedForce(Eigen::VectorXd::Zero(18));
    Eigen::VectorXd torque_upperlimit = alien_->getUpperLimitForce().e();
    Eigen::VectorXd torque_lowerlimit = alien_->getLowerLimitForce().e();
    torque_upperlimit.tail(nJoints_).setConstant(33.5 * 0.90);
    torque_lowerlimit.tail(nJoints_).setConstant(-33.5 * 0.90);
    alien_->setActuationLimits(torque_upperlimit, torque_lowerlimit);
    physics_simulated = true;
    pTarget_.setZero(19);
    pTarget_.tail(nJoints_) = gc_init_.tail(nJoints_);
    alien_->setPdTarget(pTarget_, Eigen::VectorXd::Zero(18));
  }

  void initVis()
  {
    auto vis = raisim::OgreVis::get();
    /// these method must be called before initApp
    vis->setWorld(world_.get());
    vis->setWindowTitle("Alien");
    vis->setWindowSize(1920, 1080);
    vis->setImguiSetupCallback(imguiSetupCallback);
    vis->setImguiRenderCallback(imguiRenderCallBack);
    vis->setKeyboardCallback(raisimKeyboardCallback);
    vis->setSetUpCallback(setupCallback);
    vis->setAntiAliasing(2);
    vis->initApp();
    std::vector<double> heightVec;
    for(int i = 0; i < 81; ++ i)  heightVec.push_back(0.0);
    vis_sphere = new raisim::Sphere(0.001, 0.0001);
    raisim::Sphere *target_sphere = new raisim::Sphere(0.05, 0.0001);
    targetLocation = Eigen::Vector3d(0.35,-0.21,0.37);
    vis_sphere->setPosition(Eigen::Vector3d(-0.0,-0.4,0.5));
    target_sphere->setPosition(targetLocation);
    auto heightMap = world_->addHeightMap(9, 9, 80.0, 80.0, 0., 0., heightVec, "floor");
    vis->createGraphicalObject(heightMap,  "floor", "checkerboard_mine");
    if(physics_simulated){
      auto anymalVisual_ = vis->createGraphicalObject(alien_, "A1");
      float default_friction_coeff = 0.7;
      raisim::MaterialManager materials_;
      materials_.setMaterialPairProp("floor", "robot", default_friction_coeff, 0.0, 0.0);
      world_->updateMaterialProp(materials_);
      alien_->getCollisionBody("FR_foot/0").setMaterial("robot");
      alien_->getCollisionBody("FL_foot/0").setMaterial("robot");
      alien_->getCollisionBody("RL_foot/0").setMaterial("robot");
      alien_->getCollisionBody("RR_foot/0").setMaterial("robot");
    }
    auto vis_ref_graphics = vis->createGraphicalObject(vis_ref, "vis_ref");

    fixball = vis->createGraphicalObject(vis_sphere, "vis_sphere", "white");
    if(vis_sphere_flag){
      vis->addVisualObject("target_sphere", "sphereMesh","red", {0.05, 0.05, 0.05}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      auto& list = raisim::OgreVis::get()->getVisualObjectList();
      list["target_sphere"].setPosition(targetLocation);
    }
    for(int i =0 ; i < 31; ++i){
      vis->addVisualObject("bone" + std::to_string(i), "cylinderMesh", "magenta", {0.015, 0.015, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
    }
    
    // Front View: Ogre::Radian(deg2rad(-90)), Ogre::Radian(deg2rad(-90)) 1.5
    // RightPersp View: Ogre::Radian(deg2rad(-45)), Ogre::Radian(deg2rad(-90)) 1.5
    vis->select(fixball->at(0), false);
    vis->getCameraMan()->setYawPitchDist(Ogre::Radian(deg2rad(-135)), Ogre::Radian(deg2rad(-90)), 2.0, true);
    desired_fps_ = 1.0/ control_dt_;
    vis->setDesiredFPS(desired_fps_);
    vis->renderOneFrame();
    vis->startRecordingVideo("Video.mp4");
  }

  // resets all environments and returns observation
  void reset(Eigen::Ref<EigenRowMajorMat>& ob) 
  {
    updateObservation();
    observe(ob.row(0));
  }

  void resetSim(Eigen::Ref<EigenRowMajorMat>& ob) 
  {
    updateObservationSim();
    observe(ob.row(0));
  }

  void VingVingInit()
  {
    Eigen::Matrix3d m;
    m = Eigen::AngleAxisd(deg2rad(30), Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond ving(m);
    gc_init_[0] = ving.w();
    gc_init_[1] = ving.x();
    gc_init_[2] = ving.y();
    gc_init_[3] = ving.z();
    rot_recorded = ving;
  }

  void VingVingObserve()
  {
    Eigen::Matrix3d cur;
    cur = Eigen::Quaterniond(gc_target[0],gc_target[1],gc_target[2],gc_target[3]);
    Eigen::Matrix3d m;
    m = Eigen::AngleAxisd(deg2rad(30), Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond quat(m * cur);
    gc_target[0] = quat.w();
    gc_target[1] = quat.x();
    gc_target[2] = quat.y();
    gc_target[3] = quat.z();
  }

  struct InputSettings
  {
    k4a_depth_mode_t DepthCameraMode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    bool CpuOnlyMode = false;
    bool Offline = false;
    std::string FileName;
  };

  void kinect_init() 
  {
    InputSettings inputSettings;
    SetFromDevice(inputSettings);
    image = cv::Mat::ones(colorHeight,colorWidth,CV_8UC3);
    namedWindow("Human_Motion", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("Human_Motion", 2000, 1200);
    cv::Size frame_size(colorWidth, colorHeight);
    vod = cv::VideoWriter("/home/sonic/Project/RobotControl/human_video/Human.avi", 
    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30., frame_size, true);
    kinect_set = true;
  }

  void GetFromDevice() {
    k4a_capture_t sensorCapture = nullptr;
    k4a_wait_result_t getCaptureResult = k4a_device_get_capture(device, &sensorCapture, 0); // timeout_in_ms is set to 0

    if (getCaptureResult == K4A_WAIT_RESULT_SUCCEEDED)
    {
        // timeout_in_ms is set to 0. Return immediately no matter whether the sensorCapture is successfully added
        // to the queue or not.
      k4a_image_t colorImage = k4a_capture_get_color_image(sensorCapture);
      k4a_image_t depthImage = k4a_capture_get_depth_image(sensorCapture);
      img_bgra32 = k4a_image_get_buffer(colorImage);
      k4a_wait_result_t queueCaptureResult = k4abt_tracker_enqueue_capture(tracker, sensorCapture, 0);

      k4a_imu_sample_t imu_sample;
      if(imuInit){
        if (k4a_device_get_imu_sample(device, &imu_sample, 0) == K4A_WAIT_RESULT_SUCCEEDED)
        {
          pointCloudGenerator->Update(depthImage);
          // Get down-sampled cloud points.
          const int downsampleStep = 2;
          const auto& cloudPoints = pointCloudGenerator->GetCloudPoints(downsampleStep);
          const size_t minimumFloorPointCount = 1024 / (downsampleStep * downsampleStep);

          // Detect floor plane based on latest visual and inertial observations.
          const auto& maybeFloorPlane = floorDetector.TryDetectFloorPlane(cloudPoints, imu_sample, sensorCalibration, minimumFloorPointCount);
          // Visualize the floor plane.
          if (maybeFloorPlane.has_value())
          {
            // For visualization purposes, make floor origin the projection of a point 1.5m in front of the camera.
            Samples::Vector cameraOrigin = { 0, 0, 0 };
            Samples::Vector cameraForward = { 0, 0, 1 };

            auto p = maybeFloorPlane->ProjectPoint(cameraOrigin) + maybeFloorPlane->ProjectVector(cameraForward) * 1.5f;
            auto n = maybeFloorPlane->Normal;
            z_axis = Eigen::Vector3d(-n.Z, n.X, -n.Y).normalized();
            x_axis = Eigen::Vector3d(p.Z, p.X, 0).normalized();
            y_axis = z_axis.cross(x_axis).normalized();
            imuInit = false;
          }
        }
      }

      // Release the sensor capture once it is no longer needed.
      k4a_capture_release(sensorCapture);

      if (queueCaptureResult == K4A_WAIT_RESULT_FAILED)
      {
          std::cout << "Error! Add capture to tracker process queue failed!" << std::endl;
          stopFlag = true;
          CloseDevice();
      }
    }
    else if (getCaptureResult != K4A_WAIT_RESULT_TIMEOUT)
    {
        std::cout << "Get depth capture returned error: " << getCaptureResult << std::endl;
        stopFlag = true;
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
      uint32_t numBodies = k4abt_frame_get_num_bodies(bodyFrame);
      // uint32_t numBodies = 1;
      bool success = false;
      for (uint32_t i = 0; i < numBodies; i++)
      {
        k4abt_body_t body;
        VERIFY(k4abt_frame_get_body_skeleton(bodyFrame, i, &body.skeleton), "Get skeleton from body frame failed!");
        body.id = k4abt_frame_get_body_id(bodyFrame, i);
        k4abt_skeleton_t skel = body.skeleton;
        if(startFlag){
          prev_skeleton = skel;
          visualize_bones(body);
          break;
        }
        if(isContinuous(skel))
        {
          current_skeleton = skel;
          visualize_bones(body);
          prev_skeleton = skel;
          success = true;
          break;
        }        
      }
      if(!success) std::cout << "fail" << std::endl;
      //Release the bodyFrame    
      k4abt_frame_release(bodyFrame);
    }
  }

  bool isContinuous(k4abt_skeleton_t skel) 
  {
    float scaler = -1. /1600;
    Eigen::Matrix3d axis_orientation;
    axis_orientation.col(0) = x_axis;
    axis_orientation.col(1) = y_axis;
    axis_orientation.col(2) = z_axis;
    k4a_float3_t jointPosition = prev_skeleton.joints[K4ABT_JOINT_PELVIS].position;
    Eigen::Vector3d point_prev = scaler * axis_orientation.inverse() *  Eigen::Vector3d(jointPosition.xyz.z- joint0Position.xyz.z ,-jointPosition.xyz.x + joint0Position.xyz.x, 0.);
    k4a_float3_t jointPosition_cur = skel.joints[K4ABT_JOINT_PELVIS].position;
    Eigen::Vector3d point_cur = scaler * axis_orientation.inverse() *  Eigen::Vector3d(jointPosition_cur.xyz.z- joint0Position.xyz.z ,-jointPosition_cur.xyz.x + joint0Position.xyz.x, 0.);
    float rad = (point_cur - point_prev).norm();
    if(rad > confient_rad) return false;
    else return true;
  }

  void visualize_bones(k4abt_body_t body)
  {
    float scaler = -1. / 2300;
    float y_shift = 0.8;
    auto& list = raisim::OgreVis::get()->getVisualObjectList();
    if(startFlag){
      k4a_float3_t jointRPosition = body.skeleton.joints[K4ABT_JOINT_FOOT_RIGHT].position;
      k4a_float3_t jointLPosition = body.skeleton.joints[K4ABT_JOINT_FOOT_LEFT].position;
      joint0Position = body.skeleton.joints[K4ABT_JOINT_FOOT_RIGHT].position;
      if(jointLPosition.xyz.y > jointRPosition.xyz.y) joint0Position.xyz.y = jointLPosition.xyz.y;
      startFlag = false;
    }
    Eigen::Matrix3d axis_orientation;
    axis_orientation.col(0) = x_axis;
    axis_orientation.col(1) = y_axis;
    axis_orientation.col(2) = z_axis;
    for(size_t boneIdx = 0; boneIdx < g_boneList.size(); boneIdx++)
    {
      k4abt_joint_id_t joint1 = g_boneList[boneIdx].first;
      k4abt_joint_id_t joint2 = g_boneList[boneIdx].second;
      k4a_float3_t joint1Position = body.skeleton.joints[joint1].position;
      k4a_float3_t joint2Position = body.skeleton.joints[joint2].position;
      Eigen::Vector3d point1 = scaler * axis_orientation.inverse() * Eigen::Vector3d(joint1Position.xyz.z- joint0Position.xyz.z ,-joint1Position.xyz.x + joint0Position.xyz.x, joint1Position.xyz.y- joint0Position.xyz.y);
      Eigen::Vector3d point2 = scaler * axis_orientation.inverse() * Eigen::Vector3d(joint2Position.xyz.z- joint0Position.xyz.z ,-joint2Position.xyz.x + joint0Position.xyz.x, joint2Position.xyz.y- joint0Position.xyz.y);
      Eigen::Vector3d half = (point1 + point2) / 2 ;
      half[1] -= y_shift ;
      float len = (point2 - point1).norm();
      raisim::Mat<3,3> rot;
      Eigen::Vector3d way = (point2 - point1).normalized();
      raisim::Vec<3> direction = {way[0], way[1], way[2]};
      raisim::zaxisToRotMat(direction, rot);
      if(joint2 ==K4ABT_JOINT_HAND_RIGHT ||joint2 ==K4ABT_JOINT_HANDTIP_RIGHT || joint2 == K4ABT_JOINT_THUMB_RIGHT|| joint2 ==K4ABT_JOINT_HAND_LEFT ||joint2 ==K4ABT_JOINT_HANDTIP_LEFT || joint2 == K4ABT_JOINT_THUMB_LEFT)
      {
        list["bone" + std::to_string(boneIdx)].setPosition(half);
        list["bone" + std::to_string(boneIdx)].setScale(Eigen::Vector3d(0.0, 0.0, 0.0));
        list["bone" + std::to_string(boneIdx)].setOrientation(rot);
      }
      else{
        list["bone" + std::to_string(boneIdx)].setPosition(half);
        list["bone" + std::to_string(boneIdx)].setScale(Eigen::Vector3d(0.01, 0.01, len));
        list["bone" + std::to_string(boneIdx)].setOrientation(rot);
      }
    }
    root_x.setZero(); root_y.setZero(); root_z.setZero(); 
    chest_x.setZero(); chest_y.setZero(); chest_z.setZero(); 
    int saver = 0;
    for(size_t boneIdx = 0; boneIdx < K4ABT_JOINT_COUNT; boneIdx++)
    {
      k4a_float3_t jointPosition = body.skeleton.joints[boneIdx].position;
      k4abt_joint_confidence_level_t conf = body.skeleton.joints[boneIdx].confidence_level;
      Eigen::Vector3d point = scaler * axis_orientation.inverse() *  Eigen::Vector3d(jointPosition.xyz.z- joint0Position.xyz.z ,-jointPosition.xyz.x + joint0Position.xyz.x, jointPosition.xyz.y- joint0Position.xyz.y);
      point[1] -= y_shift;
      // if(conf == K4ABT_JOINT_CONFIDENCE_NONE) std::cout << "zero" << std::endl;
      // if(conf == K4ABT_JOINT_CONFIDENCE_LOW) std::cout << "low" << std::endl;
      save_human_skel.segment(boneIdx * 3, 3) =  point;

      if (boneIdx == K4ABT_JOINT_PELVIS) 
        root_z -= point;
      else if(boneIdx == K4ABT_JOINT_SPINE_NAVEL)
        root_z += point;
      else if(boneIdx == K4ABT_JOINT_HIP_RIGHT)
        root_y -= point;
      else if(boneIdx == K4ABT_JOINT_HIP_LEFT)
        root_y += point;
      else if(boneIdx == K4ABT_JOINT_SPINE_CHEST)
        chest_z -= point;
      else if(boneIdx == K4ABT_JOINT_NECK)
        chest_z += point;
      else if(boneIdx == K4ABT_JOINT_CLAVICLE_RIGHT)
        chest_y -= point;
      else if(boneIdx == K4ABT_JOINT_CLAVICLE_LEFT)
        chest_y += point;
    }
    root_x = root_y.cross(root_z);
    chest_x = chest_y.cross(chest_z);
    root_x.normalize(); root_y.normalize(); root_z.normalize();
    chest_x.normalize(); chest_y.normalize(); chest_z.normalize();
    // std::cout << root_x.transpose()<< std::endl;
    // std::cout << root_y.transpose()<< std::endl;
    // std::cout << root_z.transpose()<< std::endl;
  }

  void visualize_bones_list()
  {
    auto& list = raisim::OgreVis::get()->getVisualObjectList();   
    float scaler = 1.;
    Eigen::VectorXd current_keypose = scaler * human_point_list[current_count];

    for(size_t boneIdx = 0; boneIdx < g_boneList.size(); boneIdx++)
    {
      k4abt_joint_id_t joint1 = g_boneList[boneIdx].first;
      k4abt_joint_id_t joint2 = g_boneList[boneIdx].second;
      Eigen::Vector3d point1(current_keypose[3*joint1], current_keypose[3*joint1+1], current_keypose[3*joint1+2]);
      Eigen::Vector3d point2(current_keypose[3*joint2], current_keypose[3*joint2+1], current_keypose[3*joint2+2]);
      // point1 += human_shifter;
      // point2 += human_shifter;
      Eigen::Vector3d half = (point1 + point2) / 2 ;
      float len = (point2 - point1).norm();
      raisim::Mat<3,3> rot;
      Eigen::Vector3d way = (point2 - point1).normalized();
      raisim::Vec<3> direction = {way[0], way[1], way[2]};
      raisim::zaxisToRotMat(direction, rot);
      if(joint2 ==K4ABT_JOINT_HAND_RIGHT ||joint2 ==K4ABT_JOINT_HANDTIP_RIGHT || joint2 == K4ABT_JOINT_THUMB_RIGHT|| joint2 ==K4ABT_JOINT_HAND_LEFT ||joint2 ==K4ABT_JOINT_HANDTIP_LEFT || joint2 == K4ABT_JOINT_THUMB_LEFT)
      {
        list["bone" + std::to_string(boneIdx)].setPosition(half);
        list["bone" + std::to_string(boneIdx)].setScale(Eigen::Vector3d(0.0, 0.0, 0.0));
        list["bone" + std::to_string(boneIdx)].setOrientation(rot);
      }
      else{
        list["bone" + std::to_string(boneIdx)].setPosition(half);
        list["bone" + std::to_string(boneIdx)].setScale(Eigen::Vector3d(0.01, 0.01, len));
        list["bone" + std::to_string(boneIdx)].setOrientation(rot);
      }
    }
  }


  void calcRoot(k4abt_body_t body)
  {
    float scaler = -1. / 1800;
    float y_shift = 0.8;
    if(startFlag){
      k4a_float3_t jointRPosition = body.skeleton.joints[K4ABT_JOINT_FOOT_RIGHT].position;
      k4a_float3_t jointLPosition = body.skeleton.joints[K4ABT_JOINT_FOOT_LEFT].position;
      joint0Position = body.skeleton.joints[K4ABT_JOINT_FOOT_RIGHT].position;
      if(jointLPosition.xyz.y > jointRPosition.xyz.y) joint0Position.xyz.y = jointLPosition.xyz.y;
      // if(custom.getMvFlag())
      startFlag = false;
    }
    Eigen::Matrix3d axis_orientation;
    axis_orientation.col(0) = x_axis;
    axis_orientation.col(1) = y_axis;
    axis_orientation.col(2) = z_axis;

    root_x.setZero(); root_y.setZero(); root_z.setZero(); 
    chest_x.setZero(); chest_y.setZero(); chest_z.setZero(); 
    for(size_t boneIdx = 0; boneIdx < K4ABT_JOINT_COUNT; boneIdx++)
    {
      k4a_float3_t jointPosition = body.skeleton.joints[boneIdx].position;
      k4abt_joint_confidence_level_t conf = body.skeleton.joints[boneIdx].confidence_level;
      Eigen::Vector3d point = scaler * axis_orientation.inverse() *  Eigen::Vector3d(jointPosition.xyz.z- joint0Position.xyz.z ,-jointPosition.xyz.x + joint0Position.xyz.x, jointPosition.xyz.y- joint0Position.xyz.y);
      point[1] -= y_shift;

      if (boneIdx == K4ABT_JOINT_PELVIS) 
        root_z -= point;
      else if(boneIdx == K4ABT_JOINT_SPINE_NAVEL)
        root_z += point;
      else if(boneIdx == K4ABT_JOINT_HIP_RIGHT)
        root_y -= point;
      else if(boneIdx == K4ABT_JOINT_HIP_LEFT)
        root_y += point;
      else if(boneIdx == K4ABT_JOINT_SPINE_CHEST)
        chest_z -= point;
      else if(boneIdx == K4ABT_JOINT_NECK)
        chest_z += point;
      else if(boneIdx == K4ABT_JOINT_CLAVICLE_RIGHT)
        chest_y -= point;
      else if(boneIdx == K4ABT_JOINT_CLAVICLE_LEFT)
        chest_y += point;
    }
    root_x = root_y.cross(root_z);
    chest_x = chest_y.cross(chest_z);
    root_x.normalize(); root_y.normalize(); root_z.normalize();
    chest_x.normalize(); chest_y.normalize(); chest_z.normalize();
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
    VERIFY(k4a_device_get_calibration(device, deviceConfig.depth_mode, deviceConfig.color_resolution, &sensorCalibration),
        "Get depth camera calibration failed!");
    int depthWidth = sensorCalibration.depth_camera_calibration.resolution_width;
    int depthHeight = sensorCalibration.depth_camera_calibration.resolution_height;
    colorWidth = sensorCalibration.color_camera_calibration.resolution_width;
    colorHeight = sensorCalibration.color_camera_calibration.resolution_height;
    // Create Body Tracker
    k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
    tracker_config.processing_mode = inputSettings.CpuOnlyMode ? K4ABT_TRACKER_PROCESSING_MODE_CPU : K4ABT_TRACKER_PROCESSING_MODE_GPU;
    VERIFY(k4abt_tracker_create(&sensorCalibration, tracker_config, &tracker), "Body tracker initialization failed!");
    // window3d.Create("3D Visualization", sensorCalibration);
    VERIFY(k4a_device_start_imu(device), "Start IMU failed!");
    pointCloudGenerator =  new Samples::PointCloudGenerator(sensorCalibration);
  }

  bool KinectEmergencyStop(){return stopFlag;}

  void CloseDevice(){
    k4abt_tracker_shutdown(tracker);
    k4abt_tracker_destroy(tracker);

    k4a_device_stop_cameras(device);
    k4a_device_close(device);
  }

  void setState(int state){current_state = state;}

  bool isNeutral(){ return mNeutral; }

  void setDynamic(){
    processingFlag = false;
    mNeutral = false;
  }

  // This function changes the gc_target!

  void setNeutral()
  {
    kin_gc_prev.tail(16) = gc_init_;
    kin_gc_prev.head(3) = Eigen::Vector3d(0,0,0.27);
    mrl->setPrevPose(kin_gc_prev);
    prevContact = Eigen::Vector4i(1,1,1,1);
    if(current_state == stateSTAND)
    { 
      float rot_thr = 0.01;
      float change_step = 0.06;
      Eigen::Quaterniond rot_current(gc_target[0], gc_target[1],gc_target[2],gc_target[3]);
      float rot_dev = rot_current.angularDistance(rot_recorded);
      bool manipulateFlag = false;

      if(currentContact.squaredNorm() < 4) manipulateFlag = true;
      if(rot_dev < rot_thr && !manipulateFlag){
        if(physics_simulated){
          Eigen::Matrix3d rot_current = alien_->getBaseOrientation().e();
          Eigen::Vector3d current_dir = rot_current * Eigen::Vector3d::UnitX();
          float angle = atan2(current_dir[1] , current_dir[0]);
          rot_compensation = Eigen::AngleAxisd(-angle, Eigen::Vector3d::UnitZ());
        }
        else
        {
          // custom.updateCompensation();
        }
        mNeutral = true;
        return;
      }

      Eigen::Quaterniond slerped;
      if(change_step < rot_dev)
        slerped = rot_current.slerp(change_step/rot_dev, rot_recorded);
      else
        slerped = rot_recorded;

      Eigen::VectorXd raw_pose(19);
      raw_pose[0] = 0; raw_pose[1] =0;
      raw_pose[2] = 0.27;
      raw_pose[3] = slerped.w(); raw_pose[4] = slerped.x();
      raw_pose[5] = slerped.y(); raw_pose[6] = slerped.z();
      raw_pose.tail(12) = gc_init_.tail(12);
      mAIK->setCurrentPose(raw_pose);
      mAIK->setReady();  
      for(int i = 0 ; i < 4; ++i){
        mAIK->appendTargets(init_feet_poses[i]);
      }
      mAIK->solve(raw_pose);
      Eigen::VectorXd current_pose = mAIK->getSolution();
      mAIK->clearTargets();
      if(manipulateFlag)
      {
        float thr = 0.01;
        Eigen::Vector3d change_step(0.06, 0.12, 0.045);
        int moving = -1;
        for(int j =0 ; j < 4; ++j)
        {
          if(currentContact[j] == 0){
            moving = j;
            break;
          }
        }
        Eigen::Vector3d gc_current = gc_target.segment(4+3*moving,3);
        Eigen::Vector3d dev = current_pose.segment(3*moving,3) - gc_current;
        if(dev.norm() < thr){
          currentContact = Eigen::Vector4i(1,1,1,1);
          manipulateFlag = false;
        }
        if(manipulateFlag){
          for(int i = 0; i < 3; ++i)
          {
            if(dev[i] >= 0)
            {
              if(dev[i] > change_step[i]) gc_current[i] += change_step[i];
              else gc_current[i] = current_pose[3*moving + i];
            }
            else
            {
              if(dev[i] < -change_step[i]) gc_current[i] -= change_step[i];
              else gc_current[i] = current_pose[3*moving + i];
            }      
          } 
          current_pose.segment(3*moving,3) = gc_current;
        }
      }
      raw_pose.tail(12) = current_pose;
      gc_target.head(4) = raw_pose.segment(3,4);
      gc_target.tail(12) = current_pose;
      vis_ref->setGeneralizedCoordinate(raw_pose);
    } 
    else if(current_state == stateSIT)
    {      
      float rot_thr = 0.01;
      float change_step = 0.06;
      Eigen::Matrix3d m;
      m = Eigen::AngleAxisd(deg2rad(-90), Eigen::Vector3d::UnitY());
      Eigen::Quaterniond target_quat(m);
      Eigen::Quaterniond rot_current(gc_target[0], gc_target[1],gc_target[2],gc_target[3]);
      float rot_dev = rot_current.angularDistance(target_quat);

      float thr = 0.01;
      Eigen::Vector3d change_arm(0.06, 0.12, 0.045);
      Eigen::VectorXd targetArm(12);
      targetArm << 0.27, deg2rad(50), -1.05, -0.27, deg2rad(50), -1.05, 0., deg2rad(120), deg2rad(-120), 0., deg2rad(120), deg2rad(-120);
      Eigen::VectorXd gc_current = gc_target.tail(12);
      Eigen::VectorXd dev = targetArm - gc_current;
      if(rot_dev < rot_thr && dev.norm() < thr){
        mNeutral = true;
        return;
      }
      Eigen::Quaterniond slerped;
      if(change_step < rot_dev)
        slerped = rot_current.slerp(change_step/rot_dev, target_quat);
      else
        slerped = target_quat;

      for(int i = 0; i < 12; ++i)
      {
        if(dev[i] >= 0)
        {
          if(dev[i] > change_arm[i%3]) gc_current[i] += change_arm[i%3];
          else gc_current[i] = targetArm[i];
        }
        else
        {
          if(dev[i] < -change_arm[i%3]) gc_current[i] -= change_arm[i%3];
          else gc_current[i] = targetArm[i];
        }
      }     

      Eigen::VectorXd raw_pose(19);
      raw_pose[0] = 0; raw_pose[1] =0;
      raw_pose[2] = 0.27;
      raw_pose[3] = slerped.w(); raw_pose[4] = slerped.x();
      raw_pose[5] = slerped.y(); raw_pose[6] = slerped.z();
      raw_pose.tail(12) = gc_current;
      gc_target.head(4) = raw_pose.segment(3,4);
      gc_target.tail(12) = gc_current;
      vis_ref->setGeneralizedCoordinate(raw_pose);
    } 

    else if(current_state == stateWALK)
    {      
      float thr = 0.01;
      Eigen::Vector3d change_step(0.03, 0.08, 0.045);
      Eigen::VectorXd gc_current = gc_target.tail(12);
      gc_target.head(4) = Eigen::Vector4d(1,0,0,0);
      Eigen::VectorXd dev = gc_init_.tail(12) - gc_current;
      if(dev.norm() < thr){
        if(physics_simulated){
          Eigen::Matrix3d rot_current = alien_->getBaseOrientation().e();
          Eigen::Vector3d current_dir = rot_current * Eigen::Vector3d::UnitX();
          float angle = atan2(current_dir[1] , current_dir[0]);
          rot_compensation = Eigen::AngleAxisd(-angle, Eigen::Vector3d::UnitZ());
        }
        else
        {
          // Fill here!
          custom.updateCompensation();
        }
        mNeutral = true;
        return;
      }
      for(int i = 0; i < 12; ++i)
      {
        if(dev[i] >= 0)
        {
          if(dev[i] > change_step[i%3]) gc_current[i] += change_step[i%3];
          else gc_current[i] = gc_init_[i+4];
        }
        else
        {
          if(dev[i] < -change_step[i%3]) gc_current[i] -= change_step[i%3];
          else gc_current[i] = gc_init_[i+4];
        }      
      }     
      gc_target.tail(12) = gc_current;
      Eigen::VectorXd gc_raw(19);
      gc_raw[0] = 0; gc_raw[1] = 0; gc_raw[2] = 0.27;
      gc_raw.tail(16) = gc_target;
      vis_ref->setGeneralizedCoordinate(gc_raw);
    } 
  }

  bool doSit()
  {
    if(sitcount == 0) processingFlag = true;
    if(physics_simulated){
      gc_target.head(4) = alien_->getGeneralizedCoordinate().e().segment(3,4);
      gc_target.tail(12) = sitPDseq[sitcount];
      Eigen::VectorXd gc_raw(19);
      gc_raw.head(3) = alien_->getGeneralizedCoordinate().e().head(3);
      gc_raw.tail(16) = gc_target;
      vis_ref->setGeneralizedCoordinate(gc_raw);
    }
    else
    {
      gc_target.head(4) = custom.sendRobotState().head(4);
      gc_target.tail(12) = sitPDseq[sitcount];
      // Eigen::VectorXd gc_raw(19);
      // gc_raw.tail(16) = gc_target;
      // vis_ref->setGeneralizedCoordinate(gc_raw);
    }
    pTarget12_ = sitPDseq[sitcount];
    sitcount++;
    if(sitcount == sitPDseq.size()){
      Eigen::Matrix3d m;
      m = Eigen::AngleAxisd(deg2rad(-90), Eigen::Vector3d::UnitY());
      Eigen::Quaterniond current_quat(m);
      Eigen::Vector3d sittingRoot;
      sittingRoot[0] = -(0.1805 + 0.05);
      sittingRoot[1] = 0.0;
      sittingRoot[2] = 0.1805 + 0.1* sqrt(3);
      kin_gc_prev << sittingRoot[0], sittingRoot[1] ,sittingRoot[2], current_quat.w(), current_quat.x(), current_quat.y(), current_quat.z(), 0.27, deg2rad(50), -1.05, -0.27, deg2rad(50), -1.05, 0., deg2rad(120), deg2rad(-120), 0., deg2rad(120), deg2rad(-120);
      sitcount = 0;
      return true;
    }
    return false;
  }

  bool doStand()
  {
    if(standcount == 0) processingFlag = true;
    if(physics_simulated){
      gc_target.head(4) = alien_->getGeneralizedCoordinate().e().segment(3,4);
      gc_target.tail(12) = standPDseq[standcount];
      Eigen::VectorXd gc_raw(19);
      gc_raw.head(3) = alien_->getGeneralizedCoordinate().e().head(3);
      gc_raw.tail(16) = gc_target;
      vis_ref->setGeneralizedCoordinate(gc_raw);
    }
    else
    {
      gc_target.head(4) = custom.sendRobotState().head(4);
      gc_target.tail(12) = standPDseq[standcount];
      // Eigen::VectorXd gc_raw(19);
      // gc_raw.tail(16) = gc_target;
      // vis_ref->setGeneralizedCoordinate(gc_raw);
    }
    pTarget12_ = standPDseq[standcount];
    standcount++;
    if(standcount == standPDseq.size()){
      if(physics_simulated){
        Eigen::Matrix3d rot_current = alien_->getBaseOrientation().e();
        Eigen::Vector3d current_dir = rot_current * Eigen::Vector3d::UnitX();
        float angle = atan2(current_dir[1] , current_dir[0]);
        rot_compensation = Eigen::AngleAxisd(-angle, Eigen::Vector3d::UnitZ());
      }
      kin_gc_prev.tail(16) = gc_init_;
      kin_gc_prev.head(3) = Eigen::Vector3d(0,0,0.27);
      standcount = 0;
      return true;
    }
    return false;
  }

  int decideModeTransition(int mode)
  {
    if(current_mode == mode) return current_mode;
    if(current_mode == TILT)
    {
      if(tiltTransition()) current_mode = mode;
      return current_mode;
    }
    else if(current_mode == MANIPULATE)
    {
      if(manStop()){
        // Eigen::VectorXd gc_current = alien_->getGeneralizedCoordinate().e();
        // // rot_recorded = Eigen::Quaterniond(gc_current[3],gc_current[4],gc_current[5],gc_current[6]);
        // gc_current[0] = 0; gc_current[1] = 0; gc_current[2] = 0.27;
        // gc_current.tail(16) = gc_target;
        // ik_dummy->setGeneralizedCoordinate(gc_current);
        // for(int i = 0; i < 4; ++i)
        // {
        //   raisim::Vec<3> footPositionA;
        //   ik_dummy->getFramePosition(FR_FOOT+3 * i, footPositionA);
        //   init_feet_poses[i] = footPositionA.e();
        // } 
        current_mode = mode;
      }
      return current_mode;
    }
    else {
      current_mode = mode;
      return current_mode;
    }
  }

  bool tiltTransition()
  {
    float thr = 0.01;
    float change_step = 0.06;
    Eigen::Quaterniond rot_current(gc_target[0], gc_target[1],gc_target[2],gc_target[3]);
    float dev = rot_current.angularDistance(rot_recorded);
    std::cout << "rot angle diff " <<  dev << std::endl;
    if(dev < thr){
      // for(int i = 0; i < 3; ++i)
      // {
      //   reference_history[i] = gc_init_;
      //   state_history[i] = gc_init_;
      //   action_history[i] = gc_init_.tail(12);
      // }
      // reference_history[3] = gc_target;
      // state_history[3] = gc_target;
      return true;
    }
    Eigen::Quaterniond slerped;
    if(change_step < dev)
      slerped = rot_current.slerp(change_step/dev, rot_recorded);
    else
      slerped = rot_recorded;

    Eigen::VectorXd raw_pose(19);
    raw_pose[0] = 0; raw_pose[1] =0;
    raw_pose[2] = 0.27;
    raw_pose[3] = slerped.w(); raw_pose[4] = slerped.x();
    raw_pose[5] = slerped.y(); raw_pose[6] = slerped.z();
    raw_pose.tail(12) = gc_init_.tail(12);
    mAIK->setCurrentPose(raw_pose);
    mAIK->setReady();  
    for(int i = 0 ; i < 4; ++i){
      mAIK->appendTargets(init_feet_poses[i]);
    }
    mAIK->solve(raw_pose);
    Eigen::VectorXd current_pose = mAIK->getSolution();
    mAIK->clearTargets();
    gc_target.head(4) = raw_pose.segment(3,4);
    gc_target.tail(12) = current_pose;
    raw_pose.tail(12) = current_pose;
    vis_ref->setGeneralizedCoordinate(raw_pose);
    inTransition = true;
    return false;
  }

  bool trotStop()
  {
    float thr = 0.01;
    Eigen::Vector3d change_step(0.06, 0.15, 0.045);
    Eigen::VectorXd gc_current = gc_target.tail(12);
    Eigen::VectorXd dev = gc_init_.tail(12) - gc_current;
    if(dev.norm() < thr){
      for(int i = 0; i < 3; ++i)
      {
        reference_history[i] = gc_init_;
        state_history[i] = gc_init_;
        action_history[i] = gc_init_.tail(12);
      }
      reference_history[3] = gc_target;
      state_history[3] = alien_->getGeneralizedCoordinate().e().tail(16);
      return true;
    }
    for(int i = 0; i < 12; ++i)
    {
      if(dev[i] >= 0)
      {
        if(dev[i] > change_step[i%3]) gc_current[i] += change_step[i%3];
        else gc_current[i] = gc_init_[i+4];
      }
      else
      {
        if(dev[i] < -change_step[i%3]) gc_current[i] -= change_step[i%3];
        else gc_current[i] = gc_init_[i+4];
      }      
    }     
    gc_target.tail(12) = gc_current;
    Eigen::VectorXd gc_raw(19);
    gc_raw[0] = 0; gc_raw[1] = 0; gc_raw[2] = 0.27;
    gc_raw.tail(16) = gc_target;
    vis_ref->setGeneralizedCoordinate(gc_raw);
    inTransition = true;
    return false;
  }

  bool manStop()
  {
    float thr = 0.005;
    Eigen::Vector3d change_step(0.06, 0.12, 0.045);
    Eigen::VectorXd gc_current = gc_target.tail(12);
    Eigen::VectorXd dev = gc_init_.tail(12) - gc_current;
    if(dev.norm() < thr){
      // for(int i = 0; i < 3; ++i)
      // {
      //   reference_history[i] = gc_init_;
      //   state_history[i] = gc_init_;
      //   action_history[i] = gc_init_.tail(12);
      // }
      // reference_history[3] = gc_init_;
      // state_history[3] = gc_init_;
      return true;
    }
    for(int i = 0; i < 12; ++i)
    {
      if(dev[i] >= 0)
      {
        if(dev[i] > change_step[i%3]) gc_current[i] += change_step[i%3];
        else gc_current[i] = gc_init_[i+4];
      }
      else
      {
        if(dev[i] < -change_step[i%3]) gc_current[i] -= change_step[i%3];
        else gc_current[i] = gc_init_[i+4];
      }      
    }     
    gc_target.tail(12) = gc_current;
    Eigen::VectorXd gc_raw(19);
    gc_raw[0] = 0; gc_raw[1] = 0; gc_raw[2] = 0.27;
    gc_raw.tail(16) = gc_target;
    vis_ref->setGeneralizedCoordinate(gc_raw);
    inTransition = true;
    return false;
  }

  Eigen::Matrix3d root_orientation_from_skel()
  {
    Eigen::Matrix3d body_orientation;
    body_orientation.col(0) = root_x;
    body_orientation.col(1) = root_y;
    body_orientation.col(2) = root_z;
    return body_orientation;
  }

  Eigen::Quaterniond chest_from_skel(Eigen::Matrix3d root_ori)
  {
    Eigen::Matrix3d chest_orientation;
    chest_orientation.col(0) = chest_x;
    chest_orientation.col(1) = chest_y;
    chest_orientation.col(2) = chest_z;
    Eigen::Quaterniond chest(root_ori.inverse() * chest_orientation);
    return chest;
  }

  float lenJoints(k4a_float3_t a, k4a_float3_t b)
  {
    float x1 = a.xyz.x;
    float y1 = a.xyz.y;
    float z1 = a.xyz.z;
    float x2 = b.xyz.x;
    float y2 = b.xyz.y;
    float z2 = b.xyz.z;
    return sqrt(pow(x1-x2,2) + pow(y1-y2,2) + pow(z1-z2,2));
  }

  Eigen::VectorXd getHumanPose() 
  {
    // Orientation , Right Hand / Left Hand / Right Foot / Left Foot order
    // Since Kinect Provides Left to right ==> Use minus operation
    // Need converter
    Eigen::VectorXd human_pose(32);
    for(int i =0 ; i< 1; ++i)
    {
      GetFromDevice();
    }
    float arm_len, leg_len;

    Eigen::Matrix3d Root_ori = root_orientation_from_skel();
    Eigen::Quaterniond root_quat(Root_ori);
    auto ori_chest = chest_from_skel(Root_ori).normalized();

    human_pose[0] = ori_chest.w();
    human_pose[1] = ori_chest.x();
    human_pose[2] = ori_chest.y();
    human_pose[3] = ori_chest.z(); 

    human_pose[4] = root_quat.w();
    human_pose[5] = root_quat.x();
    human_pose[6] = root_quat.y();
    human_pose[7] = root_quat.z();

    for(int i = 0 ; i < 2; ++i){
      k4a_float3_t WristJoint = current_skeleton.joints[K4ABT_JOINT_WRIST_RIGHT - 7 * i].position;
      k4a_float3_t ElbowJoint = current_skeleton.joints[K4ABT_JOINT_ELBOW_RIGHT - 7 * i].position;
      k4a_float3_t ShoulderJoint = current_skeleton.joints[K4ABT_JOINT_SHOULDER_RIGHT - 7 * i].position;
      arm_len = lenJoints(ShoulderJoint, ElbowJoint) + lenJoints(ElbowJoint, WristJoint);
      // z, x, y order
      Eigen::Vector3d raw_arm = Eigen::Vector3d((-WristJoint.xyz.z + ShoulderJoint.xyz.z), (WristJoint.xyz.x - ShoulderJoint.xyz.x), (-WristJoint.xyz.y + ShoulderJoint.xyz.y)) / arm_len;
      Eigen::Vector3d converted = Root_ori.inverse() * raw_arm;
      Eigen::Vector3d raw_elbow = Eigen::Vector3d((-ElbowJoint.xyz.z + ShoulderJoint.xyz.z), (ElbowJoint.xyz.x - ShoulderJoint.xyz.x), (-ElbowJoint.xyz.y + ShoulderJoint.xyz.y)) / arm_len;
      Eigen::Vector3d converted_elbow = Root_ori.inverse() * raw_elbow;

      human_pose[8 + 3*i] = converted[0];
      human_pose[9 + 3*i] = converted[1];
      human_pose[10 + 3*i] = converted[2];
      human_pose[20 + 3*i] = converted_elbow[0];
      human_pose[21 + 3*i] = converted_elbow[1];
      human_pose[22 + 3*i] = converted_elbow[2];
    }
    k4a_float3_t PelvisJoint = current_skeleton.joints[K4ABT_JOINT_PELVIS].position;
    for(int i = 0 ; i < 2; ++i){
      k4a_float3_t FootJoint = current_skeleton.joints[K4ABT_JOINT_FOOT_RIGHT - 4 * i].position;
      k4a_float3_t AnkleJoint = current_skeleton.joints[K4ABT_JOINT_ANKLE_RIGHT - 4 * i].position;
      k4a_float3_t KneeJoint = current_skeleton.joints[K4ABT_JOINT_KNEE_RIGHT - 4 * i].position;
      k4a_float3_t HipJoint = current_skeleton.joints[K4ABT_JOINT_HIP_RIGHT - 4 * i].position;
      leg_len = lenJoints(HipJoint, KneeJoint) + lenJoints(KneeJoint, AnkleJoint) + lenJoints(AnkleJoint, FootJoint);
      // z, x, y order
      Eigen::Vector3d raw_leg = Eigen::Vector3d((-FootJoint.xyz.z + HipJoint.xyz.z), (FootJoint.xyz.x - HipJoint.xyz.x), (-FootJoint.xyz.y + HipJoint.xyz.y)) / leg_len;
      Eigen::Vector3d converted = Root_ori.inverse() * raw_leg;
      Eigen::Vector3d raw_knee = Eigen::Vector3d((-KneeJoint.xyz.z + HipJoint.xyz.z), (KneeJoint.xyz.x - HipJoint.xyz.x), (-KneeJoint.xyz.y + HipJoint.xyz.y)) / leg_len;
      Eigen::Vector3d converted_knee = Root_ori.inverse() * raw_knee;

      human_pose[14 + 3*i] = converted[0];
      human_pose[15 + 3*i] = converted[1];
      human_pose[16 + 3*i] = converted[2];
      human_pose[26 + 3*i] = converted_knee[0];
      human_pose[27 + 3*i] = converted_knee[1];
      human_pose[28 + 3*i] = converted_knee[2];

    }
    bgra2Mat();
    vod.write(image);
    // imshow("Human_Motion", image);
    // cv::waitKey(1);
    return human_pose;
  }

  Eigen::VectorXd getHumanSkeleton()
  {
    return save_human_skel;
  }
  Eigen::VectorXd getCurrentRobot()
  {
    return alien_->getGeneralizedCoordinate().e();
  }
  Eigen::VectorXd getCurrentReference()
  {
    return vis_ref->getGeneralizedCoordinate().e();
  }


  Eigen::VectorXd getHumanData()
  {
    Eigen::VectorXd currentHuman = getHumanPose();
    // return getHumanSingle(currentHuman);
    return getHumanSeq(currentHuman);
  }

  Eigen::VectorXd getHumanSingle(Eigen::VectorXd currentHuman)
  {
    return currentHuman;
  }

  Eigen::VectorXd getHumanSeq(Eigen::VectorXd currentHuman)
  {
    // To Do
    Eigen::VectorXd human_pose(96);
    human_pose.head(32) = human2;
    human_pose.segment(32, 32) = human1;
    human_pose.tail(32) = currentHuman;
    human2 = human1;
    human1 = currentHuman;
    return human_pose;
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
  }

  void startMotion()
  {
    // InitEnvironment();
    loop_control = new LoopFunc("control_loop", custom.dt,    boost::bind(&Custom::RobotControl, &custom));
    loop_udpSend = new LoopFunc("udp_send",     custom.dt, 3, boost::bind(&Custom::UDPSend,      &custom));
    loop_udpRecv = new LoopFunc("udp_recv",     custom.dt, 3, boost::bind(&Custom::UDPRecv,      &custom));

    loop_udpSend->start();
    loop_udpRecv->start();
    loop_control->start();
  }

  void startMotionPD()
  {
    // InitEnvironment();
    // loop_control = new LoopFunc("control_loop", custom.dt,    boost::bind(&Custom::RobotControl_PD, &custom));
    // loop_udpSend = new LoopFunc("udp_send",     custom.dt, 3, boost::bind(&Custom::UDPSend,      &custom));
    // loop_udpRecv = new LoopFunc("udp_recv",     custom.dt, 3, boost::bind(&Custom::UDPRecv,      &custom));

    // loop_udpSend->start();
    // loop_udpRecv->start();
    // loop_control->start();
  }

  void control(int frame)
  {

    unsigned int oneframe = 33000;
    for(int i = 0; i < frame; ++i)
    {
      usleep(oneframe);
      // custom.changeTest();
    }
  }

  bool isMoving(){return custom.getMvFlag();}

  void step(Eigen::Ref<EigenRowMajorMat> &action, Eigen::Ref<EigenRowMajorMat> &ob)
  {    
    if(!processingFlag){
      Eigen::VectorXd currentAction = action.row(0).cast<double>();
      pTarget12_ = currentAction;
      pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
      pTarget12_ += actionMean_;
    }
    custom.getControlSignal(pTarget12_);
    std::cout << pTarget12_.transpose() << std::endl;
    if(custom.getMvFlag()){
      action_history.erase(action_history.begin());
      action_history.push_back(pTarget12_);
    }
    auto vis = raisim::OgreVis::get();
    vis->renderOneFrame();
    unsigned int oneframe = std::max(33000-delay, 1);
    std::cout << "frame " << (int)oneframe << std::endl;
    usleep(oneframe);
    updateObservation();
    if(custom.getMvFlag()) current_count++;
    observe(ob.row(0));
    delay = 0;
  }

  void stepTest(Eigen::Ref<EigenRowMajorMat> &action)
  {    
    Eigen::VectorXd currentAction = action.row(0).cast<double>();
    pTarget12_ = currentAction;
    std::cout << pTarget12_.transpose() << std::endl;
    custom.getControlSignal(pTarget12_);
    unsigned int oneframe = 33000;
    for(int i = 0; i < 1; ++i)
    {
      usleep(oneframe);
    }    
    updateObservation();
  }

  void stepSimulation(Eigen::Ref<EigenRowMajorMat> &action, Eigen::Ref<EigenRowMajorMat> &ob)
  {
    if(!processingFlag){
      Eigen::VectorXd currentAction = action.row(0).cast<double>();
      pTarget12_ = currentAction;
      pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
      pTarget12_ += actionMean_;
    }
    auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
    auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);    
    action_history.erase(action_history.begin());
    action_history.push_back(pTarget12_);
    pTarget_.tail(12) = pTarget12_;
    alien_->setPTarget(pTarget_);


    Eigen::VectorXd gc_ref = vis_ref->getGeneralizedCoordinate().e();
    visualizeFlag = true;
    if(visualizeFlag) gc_ref[2] = 0.27;
    else gc_ref[2] = -10000;
    vis_ref->setGeneralizedCoordinate(gc_ref);

    for(int i=0; i<loopCount; i++) {
      world_->integrate();  
      if(visualizationCounter_ % visDecimation == 0)
      {
        auto vis = raisim::OgreVis::get();
        if(vis_sphere_flag){
          int j;
          for(j = 0; j < 2; ++j){
            raisim::Vec<3> temp;
            alien_->getFramePosition(FR_FOOT + 3 * i, temp);
            if((temp.e() - targetLocation).norm() < 0.06){
              auto *ent = vis->getSceneManager()->getEntity("target_sphere");
              ent->setMaterialName("blue"); 
              break;
            }
          }
          if(j == 2)
          {
            auto *ent = vis->getSceneManager()->getEntity("target_sphere");
            ent->setMaterialName("red"); 
          }
        }        
        if(current_state == stateWALK)
          vis_sphere->setPosition(Eigen::Vector3d(alien_->getBasePosition()[0],-0.4,0.5));
        vis->select(fixball->at(0), false);
        vis->getCameraMan()->setYawPitchDist(Ogre::Radian(deg2rad(-135)), Ogre::Radian(deg2rad(-90)), 2.0, true);
        vis->renderOneFrame();
      }      
      visualizationCounter_++;
    }
    current_count++;
    updateObservationSim();
    observe(ob.row(0));
  }

  void stepSimulation2(Eigen::Ref<EigenRowMajorMat> &action, Eigen::Ref<EigenRowMajorMat> &ob)
  {
    gc_target = trajectories[current_count];
    Eigen::VectorXd currentAction = action.row(0).cast<double>();
    pTarget12_ = currentAction;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
    auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);    
    action_history.erase(action_history.begin());
    action_history.push_back(pTarget12_);
    pTarget_.tail(12) = pTarget12_;
    alien_->setPTarget(pTarget_);

    for(int i=0; i<loopCount; i++) {
      world_->integrate();  
      if(visualizationCounter_ % visDecimation == 0)
      {
        auto vis = raisim::OgreVis::get();
        vis->renderOneFrame();
      }      
      visualizationCounter_++;
    }
    current_count++;
    // auto vis = raisim::OgreVis::get();
    // vis->renderOneFrame();
    updateObservationSim();
    observe(ob.row(0));
  }

  void updateSimObs(Eigen::Ref<EigenRowMajorMat> &ob)
  {
    updateObservationSim();
    observe(ob.row(0));
  }

  void updateRecorded()
  {
    gc_target = trajectories[human_cnt];
    human_cnt++;
    Eigen::VectorXd gc_raw;
    gc_raw.setZero(19);
    gc_raw.head(3)= Eigen::Vector3d(0,0,0.27);
    gc_raw.tail(16) = gc_target;
    vis_ref->setGeneralizedCoordinate(gc_raw);
  }

  void getContactStep(const Eigen::Ref<EigenVec> &contact, const Eigen::Ref<EigenVec> &pose)
  {
    estimateContact(contact);
    getReferencePoses(pose);
  }


  void getWalkStep(const Eigen::Ref<EigenVec> &contact, const Eigen::Ref<EigenVec> &pose)
  {
    estimateContactWalk(contact);
    getReferencePoseWalk(pose);
  }

  void getParamLocoDel(const Eigen::Ref<EigenVec> &ori, const Eigen::Ref<EigenVec> &params)
  {
    getReferencePoseParamLocoDel(ori, params);
  }

  void getParamLoco(const Eigen::Ref<EigenVec> &ori, const Eigen::Ref<EigenVec> &params)
  {
    getReferencePoseParamLoco(ori, params);
  }

  void getSitStep(const Eigen::Ref<EigenVec> &pose)
  {
    getReferencePoseSit(pose);
  }

  void getStandStep(const Eigen::Ref<EigenVec> &contact, const Eigen::Ref<EigenVec> &tilt, const Eigen::Ref<EigenVec> &man)
  {
    estimateContactStand(contact);
    getReferencePoseStand(tilt, man);
  }

  void estimateContact(const Eigen::Ref<EigenVec>& contact)
  {
    Eigen::VectorXd contact_vec = contact.cast<double>();
    if(current_mode == TILT){
      for(int i = 0; i <4; ++i)
      {
        currentContact[i] = 1;
      }
      return;
    }

    for(int i = 0; i <4; ++i)
    {
      if(contact_vec[i] > 0.0)
        currentContact[i] = 0;
      else
        currentContact[i] = 1;
    }
    // This is used temporary to avoid multi arm manipulation
    // For the safety
    if(current_mode == MANIPULATE){
      if(currentContact.squaredNorm() < 4)
      {
        int idx = -1;
        float max_cont = -1000;
        for(int i = 0; i < 4 ;++i)
        {
          if(contact_vec[i] > max_cont)
          {
            max_cont = contact_vec[i];
            idx = i;
          }
          currentContact[i] = 1;
        }
        currentContact[idx] = 0;
      }
    }
  }
  void estimateContactStand(const Eigen::Ref<EigenVec>& contact)
  {
    Eigen::VectorXd contact_vec = contact.cast<double>();
    for(int i = 0; i <2; ++i)
    {
      if(contact_vec[i] > 0.0)
        currentContact[i] = 0;
      else
        currentContact[i] = 1;
    }
    // std::cout<< " cont " << contact_vec.transpose() << std::endl;
    if(currentContact.squaredNorm() < 4)
    {
      if(prevContact.squaredNorm() < 4)
      {
        currentContact = prevContact;
        return;
      }
      int idx = -1;
      float max_cont = -1000;
      for(int i = 0; i < 2 ;++i)
      {
        if(contact_vec[i] > max_cont)
        {
          max_cont = contact_vec[i];
          idx = i;
        }
        currentContact[i] = 1;
      }
      currentContact[idx] = 0;
    }
  }

  void estimateContactWalk(const Eigen::Ref<EigenVec>& contact)
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
    // if(currentAction[0] > 0)
    //   gc_target.head(4) = currentAction.head(4);
    // else
    //   gc_target.head(4) = -currentAction.head(4);
    // gc_target.tail(12) = currentAction.tail(12);
    if(inTransition){
      inTransition = false;
      kin_gc_prev.tail(16) = gc_target;
      return;
    }
    if(current_mode == MANIPULATE)
      setReshapedMotionManipulate(currentAction);
    else
      setReshapedMotion(currentAction);
    // kin_gc_prev.tail(16) = gc_target;
  }

  void getReferencePoseStand(const Eigen::Ref<EigenVec>& tilt,const Eigen::Ref<EigenVec>& man)
  {
    Eigen::VectorXd tiltAction = tilt.cast<double>();
    Eigen::VectorXd manAction = man.cast<double>();
    setReshapedMotionStand(tiltAction, manAction);
  }

  void getReferencePoseWalk(const Eigen::Ref<EigenVec>& pose)
  {
    Eigen::VectorXd currentAction = pose.cast<double>();
    setReshapedMotionWalk(currentAction);
  }

  void getReferencePoseSit(const Eigen::Ref<EigenVec>& pose)
  {
    Eigen::VectorXd currentAction = pose.cast<double>();
    setReshapedMotionSit(currentAction);
  }

  void getReferencePoseParamLocoDel(const Eigen::Ref<EigenVec> &ori, const Eigen::Ref<EigenVec>& param)
  {
    Eigen::VectorXd current_ori = ori.cast<double>();
    Eigen::VectorXd currentParam = param.cast<double>();
    float h_tg = currentParam[0];
    float alpha_tg = currentParam[1];
    float Ae_value = currentParam[2];
    float del_phi = currentParam[3];
    if(del_phi > 0.08 * M_PI) del_phi = 0.08 * M_PI;
    phi+= del_phi;
    phi = fmod(phi, 2 * M_PI);
    mTG.change_Cs(0.0);
    mTG.get_tg_parameters(1.0, alpha_tg, h_tg);
    mTG.set_Ae(Ae_value);
    mTG.manual_timing_update(phi);
    gc_target.tail(12) = mTG.get_u();
    if(current_ori[0] > 0) gc_target.head(4) = current_ori;
    else gc_target.head(4) = -current_ori;
    Eigen::VectorXd gc_raw;
    gc_raw.setZero(19);
    gc_raw.head(3)= Eigen::Vector3d(0,0,0.27);
    gc_raw.tail(16) = gc_target;
    vis_ref->setGeneralizedCoordinate(gc_raw);
  } 

  void getReferencePoseParamLoco(const Eigen::Ref<EigenVec> &ori, const Eigen::Ref<EigenVec>& param)
  {
    Eigen::VectorXd current_ori = ori.cast<double>();
    Eigen::VectorXd currentParam = param.cast<double>();
    float h_tg = currentParam[0];
    float alpha_tg = currentParam[1];
    float Ae_value = currentParam[2];
    float del = currentParam[3] - phi; 
    if(del < 0) del += 2 * M_PI;
    if(del > 0.05 * M_PI) del  = 0.05 * M_PI;
    phi += del;
    phi = fmod(phi, 2 * M_PI);
    mTG.change_Cs(0.0);
    mTG.get_tg_parameters(1.0, alpha_tg, h_tg);
    mTG.set_Ae(Ae_value);
    mTG.manual_timing_update(phi);
    gc_target.tail(12) = mTG.get_u();
    if(current_ori[0] > 0) gc_target.head(4) = current_ori;
    else gc_target.head(4) = -current_ori;
    Eigen::VectorXd gc_raw;
    gc_raw.setZero(19);
    gc_raw.head(3)= Eigen::Vector3d(0,0,0.27);
    gc_raw.tail(16) = gc_target;
    vis_ref->setGeneralizedCoordinate(gc_raw);
  } 


  void setReshapedMotionStand(Eigen::VectorXd tilt, Eigen::VectorXd manipulate)
  {
    Eigen::Vector4i tiltContact = Eigen::Vector4i(1,1,1,1);
    // First do the tilting
    Eigen::VectorXd prev_target = gc_target;
    Eigen::VectorXd gc_raw;
    gc_raw.setZero(19);
    gc_raw.head(3)= Eigen::Vector3d(0,0,0.27);
    gc_raw.tail(16) = tilt;
    ik_dummy->setGeneralizedCoordinate(gc_raw);
    float current_foot = 10000;
    for(int i= 0; i < 4; ++i)
    {
      raisim::Vec<3> temp;
      ik_dummy->getFramePosition(FR_FOOT + 3 * i, temp);
      if(temp[2] < current_foot) current_foot = temp[2];
    }
    gc_raw[2] += (init_foot - current_foot);
    mrl->contactInformation(tiltContact);
    mrl->getPreviousRealPose(kin_gc_prev);
    mrl->getCurrentRawPose(gc_raw);
    mrl->getKinematicFixedPoseTest();
    Eigen::VectorXd gc_reshaped = mrl->getReshapedMotion();
    gc_target = gc_reshaped.tail(16);
    // std::cout << gc_target.transpose() - tilt.transpose()<< std::endl;
    // gc_target = tilt;

    if(tilt[0] > 0)
      gc_target.head(4) = gc_target.head(4);
    else
      gc_target.head(4) = -gc_target.head(4);
    prevContact = currentContact;

    for(int i = 0 ; i < 4; ++i)
    {
      if(currentContact[i] == 0)
      {
        float max_dev; 
        for(int j = 0; j <3; ++j){
          max_dev = 0.9 * mrl->getJointLimit(j);
          float current_angle = manipulate[4 + 3 * i + j];
          float dev = current_angle - prev_target[4 + 3 * i + j];
          if(dev > max_dev) current_angle = prev_target[4 + 3 * i + j]+ max_dev;
          else if(-max_dev > dev) current_angle = prev_target[4 + 3 * i + j] - max_dev;
          gc_target[4 + 3 * i+ j] = current_angle;
        }
      }
    }
    gc_raw.tail(16) = gc_target;
    mrl->setPrevPose(gc_raw);
    vis_ref->setGeneralizedCoordinate(gc_raw);
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
    Eigen::VectorXd gc_raw;
    gc_raw.setZero(19);
    gc_raw.head(3)= Eigen::Vector3d(0,0,0.27);
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
    // mrl->getDynStates(gc_, gc_prev);
    mrl->contactInformation(currentContact);
    mrl->getPreviousRealPose(kin_gc_prev);
    mrl->getCurrentRawPose(gc_raw);
    mrl->getKinematicFixedPoseTest();
    Eigen::VectorXd gc_reshaped = mrl->getReshapedMotion();
    gc_target = gc_reshaped.tail(16);
    if(currentAction[0] > 0)
      gc_target.head(4) = gc_target.head(4);
    else
      gc_target.head(4) = -gc_target.head(4);
    // if(human_cnt < 1) gc_target = gc_init_;
    prevContact = currentContact;
    gc_raw.tail(16) = gc_target;
    vis_ref->setGeneralizedCoordinate(gc_raw);
  }

  void setReshapedMotionManipulate(Eigen::VectorXd currentAction)
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
    Eigen::VectorXd gc_raw;
    gc_raw.setZero(19);
    gc_raw.head(3)= Eigen::Vector3d(0,0,0.27);
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
    // mrl->getDynStates(gc_, gc_prev);
    mrl->contactInformation(currentContact);
    mrl->getPreviousRealPose(kin_gc_prev);
    mrl->getCurrentRawPose(gc_raw);
    mrl->getKinematicFixedManipulate();
    Eigen::VectorXd gc_reshaped = mrl->getReshapedMotion();
    gc_target = gc_reshaped.tail(16);
    if(currentAction[0] > 0)
      gc_target.head(4) = gc_target.head(4);
    else
      gc_target.head(4) = -gc_target.head(4);
    // if(human_cnt < 1) gc_target = gc_init_;
    prevContact = currentContact;
    gc_raw.tail(16) = gc_target;
    vis_ref->setGeneralizedCoordinate(gc_raw);
  }

  void setReshapedMotionWalk(Eigen::VectorXd currentAction)
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
    Eigen::VectorXd gc_raw;
    gc_raw.setZero(19);
    gc_raw.head(7)<< 0,0,0.27,1,0,0,0;
    gc_raw.tail(12) = currentAction.tail(12);
    ik_dummy->setGeneralizedCoordinate(gc_raw);
    float current_foot = 10000;
    for(int i= 0; i < 4; ++i)
    {
      raisim::Vec<3> temp;
      ik_dummy->getFramePosition(FR_FOOT + 3 * i, temp);
      if(temp[2] < current_foot) current_foot = temp[2];
    }
    gc_raw[2] += (init_foot - current_foot);
    mrl->contactInformation(currentContact);
    mrl->getPreviousRealPose(kin_gc_prev);
    mrl->getCurrentRawPose(gc_raw);
    mrl->compensateWalk();
    mrl->getKinematicFixedPoseTest();
    Eigen::VectorXd gc_reshaped = mrl->getReshapedMotion();
    gc_target = gc_reshaped.tail(16);
    gc_target.head(4) = currentAction.head(4);

    if(currentAction[0] < 0)
      gc_target.head(4) = -gc_target.head(4);
    prevContact = currentContact;
    gc_raw.tail(16) = gc_target;
    vis_ref->setGeneralizedCoordinate(gc_raw);
  }

  void setReshapedMotionSit(Eigen::VectorXd currentAction)
  {
    Eigen::VectorXd gc_raw;
    gc_raw.setZero(19);
    gc_raw[0] = -(0.1805 + 0.05);
    gc_raw[1] = 0.0;
    gc_raw[2] = 0.1805 + 0.1* sqrt(3);
    auto prev_target = gc_target;
    gc_target = currentAction;
    Eigen::VectorXd rear_pose(6);
    for(int i = 0 ; i < 2; ++i)
    {
      float max_dev; 
      for(int j = 0; j <3; ++j){
        max_dev = 0.9 * mrl->getJointLimit(j);
        float current_angle = currentAction[4 + 3 * i + j];
        float dev = current_angle - prev_target[4 + 3 * i + j];
        if(dev > max_dev) current_angle = prev_target[4 + 3 * i + j]+ max_dev;
        else if(-max_dev > dev) current_angle = prev_target[4 + 3 * i + j] - max_dev;
        gc_target[4 + 3 * i+ j] = current_angle;
      }
    }
    if(currentAction[0] > 0)
      gc_target.head(4) = gc_target.head(4);
    else
      gc_target.head(4) = -gc_target.head(4);
    rear_pose << 0., deg2rad(120), deg2rad(-120), 0., deg2rad(120), deg2rad(-120);
    gc_target.tail(6) = rear_pose;
    gc_raw.tail(16) = gc_target;
    vis_ref->setGeneralizedCoordinate(gc_raw);
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
        if(1000 < y && y < 1050)
        {
          image.at<cv::Vec3b>(y, x)[0] = 0;
          image.at<cv::Vec3b>(y, x)[1] = 0;
          image.at<cv::Vec3b>(y, x)[2] = 255;
        }
      }
    }
  }

  void readRecordedHumanParam(std::string human_params)
  {
    std::ifstream human_input(human_params);
    if(!human_input){
      std::cout << "Expected valid input file name" << std::endl;
    }

    int triple_bone = 32;
    while(!human_input.eof()){
      Eigen::VectorXd humanoid(triple_bone);
      for(int i = 0 ; i < triple_bone; ++i)
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

  void readRecordedHumanSkel(std::string human_points)
  {
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
    human_point_list.pop_back();
    human_input.close();
  } 

  Eigen::VectorXd readHumanData()
  {
    visualize_bones_list();
    Eigen::VectorXd human_aug(96);
    int prev_2 = std::max(0, current_count - 2);
    int prev_1 = std::max(0, current_count - 1);
    human_aug.segment(0, 32) = human_param_list[prev_2];
    human_aug.segment(32, 32) = human_param_list[prev_1];
    human_aug.segment(64, 32) = human_param_list[current_count];
    return human_aug;
  }


  void finishSim()
  {
    auto vis = raisim::OgreVis::get();
    vis->closeApp();
    raisim::OgreVis::get()->stopRecordingVideoAndSave();
    if(kinect_set)
      CloseDevice();
  }

  void finish()
  {
    loop_control->shutdown();
    loop_udpSend->shutdown(); 
    loop_udpRecv->shutdown();
    delete loop_control;
    delete loop_udpSend;
    delete loop_udpRecv;
    loop_control = NULL;
    loop_udpSend = NULL;
    loop_udpRecv = NULL;
    raisim::OgreVis::get()->stopRecordingVideoAndSave();
    if(kinect_set)
      CloseDevice();
  }


  void observe(Eigen::Ref<EigenVec> ob) 
  {
    ob = obScaled_.cast<float>();
  }

  void updateObservation()
  {
    // Here, time horizon and ... 
    obScaled_.setZero(obDim_);
    if(custom.getMvFlag()){
      state_history.erase(state_history.begin());
      state_history.push_back(custom.sendRobotState());
      reference_history.erase(reference_history.begin());
      reference_history.push_back(gc_target);
    }

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

  void updateObservationSim()
  {
    // Here, time horizon and ... 
    obScaled_.setZero(obDim_);
    state_history.erase(state_history.begin());
    Eigen::VectorXd gc_mod(16);
    gc_mod = alien_->getGeneralizedCoordinate().e().tail(16);
    Eigen::Matrix3d current_rot = alien_->getBaseOrientation().e();
    Eigen::Quaterniond quat_mine(rot_compensation * current_rot);
    gc_mod[0] = quat_mine.w(); gc_mod[1] = quat_mine.x(); 
    gc_mod[2] = quat_mine.y(); gc_mod[3] = quat_mine.z();
    if(gc_mod[0] < 0)
      gc_mod.head(4) = -gc_mod.head(4);
    state_history.push_back(gc_mod);
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

  void generateRandomSeqMani()
  {
    // This is for reaching
    f_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(45), deg2rad(45));
    f_thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(55), deg2rad(153));
    f_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(150), -deg2rad(50));
    r_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(30), deg2rad(30));
    r_thigh_distribution = std::uniform_real_distribution<double> (deg2rad(-15), deg2rad(120));
    r_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(130), -deg2rad(100));
    change_distribution = std::uniform_int_distribution<int>(15, 50);
    rad_distribution = std::uniform_real_distribution<double>(0.0, 0.05);
    leg_distribution = std::uniform_int_distribution<int>(0, 3);

    int cnt = 0; 
    Eigen::VectorXd gc_init_full;
    gc_init_full.setZero(19);
    gc_init_full.head(3)= Eigen::Vector3d(0,0,0.27);
    gc_init_full.tail(16) = gc_init_;
    Eigen::Quaterniond prev_quat(gc_init_full[3],gc_init_full[4],gc_init_full[5],gc_init_full[6]);
    Eigen::VectorXd prev_pose = gc_init_full.tail(12);
    motionBlender(prev_quat, prev_quat, prev_pose, prev_pose, 20);
    cnt+= 20;
    int prev_leg = -1;

    while(cnt < max_cnt)
    {
      // First decide the leg to manipulate
      std::random_device rd;
      std::mt19937 generator(rd());
      int current_leg = leg_distribution(generator);
      Eigen::VectorXd current_pose = prev_pose;
      Eigen::Quaterniond current_quat = prev_quat;

      // Check whether the arm is the same or different
      if(prev_leg != current_leg && prev_leg != -1)
      {
        // Land the prev leg for the next movement and then increase the count
        int land_between = change_distribution(generator);

        // Define the target
        float phi = phi_distribution(generator);
        float rad = rad_distribution(generator);
        ik_dummy->setGeneralizedCoordinate(gc_init_full);
        raisim::Vec<3> original_landing;
        ik_dummy->getFramePosition(FR_FOOT+ 3 * prev_leg, original_landing);
        Eigen::Vector3d target_land = original_landing.e() + Eigen::Vector3d(rad * cos(phi), rad * sin(phi),0);
        Eigen::Vector3d sol;

        if(prev_leg % 2 == 0)
        {
          rik->setHipIdx(prev_leg/2);
          rik->setReady();
          rik->setCurrentPose(gc_init_full);
          rik->setTarget(target_land);
          rik->solveIK();
          sol = rik->getSolution();
          rik->reset();
        }
        else
        {
          lik->setHipIdx(prev_leg/2);
          lik->setReady();
          lik->setCurrentPose(gc_init_full);
          lik->setTarget(target_land);
          lik->solveIK();
          sol = lik->getSolution();
          lik->reset();
        }
        current_pose[3 * prev_leg + 0] = sol[0];
        current_pose[3 * prev_leg + 1] = sol[1];
        current_pose[3 * prev_leg + 2] = sol[2];

        motionBlender(prev_quat, current_quat, prev_pose, current_pose, land_between);
        cnt += land_between;
        // Here for stability
        motionBlender(current_quat, current_quat, current_pose, current_pose, 10);
        cnt+= 10;
        
        prev_quat = current_quat;
        prev_pose = current_pose;
      }

      int inbetween = change_distribution(generator);
      // Here distinguish rear and front leg
      if(current_leg < 2)
      {
        current_pose[3 * current_leg + 0] = f_hip_distribution(generator);
        current_pose[3 * current_leg + 1] = f_thigh_distribution(generator);
        current_pose[3 * current_leg + 2] = f_calf_distribution(generator);
      }
      else
      {
        current_pose[3 * current_leg + 0] = r_hip_distribution(generator);
        current_pose[3 * current_leg + 1] = r_thigh_distribution(generator);
        current_pose[3 * current_leg + 2] = r_calf_distribution(generator);
      }
      motionBlender(prev_quat, current_quat, prev_pose, current_pose, inbetween);
      prev_quat = current_quat;
      prev_pose = current_pose;
      cnt += inbetween;
      prev_leg = current_leg;
    }
  }

  void generateRandomSeqLoco()
  {
    // This is for locomotion
    Ae_distribution =  std::uniform_real_distribution<double> (0.05, 0.19);
    height_distribution =  std::uniform_real_distribution<double> (0.22, 0.33);
    swing_distribution =  std::uniform_real_distribution<double> (deg2rad(0), deg2rad(25));
    phase_speed_distribution =  std::uniform_real_distribution<double> (0.3, 1.8);
    noise_distribution =  std::uniform_real_distribution<double> (-deg2rad(1), deg2rad(1));
    anglularvel_distribution = std::uniform_real_distribution<double> (deg2rad(-27), deg2rad(27));
    // anglularvel_distribution =  std::uniform_real_distribution<double> (-deg2rad(0.), deg2rad(0.));
    change_distribution = std::uniform_int_distribution<int>(7, 50);

    // Ae_distribution =  std::uniform_real_distribution<double> (0.11, 0.11);
    // height_distribution =  std::uniform_real_distribution<double> (0.25, 0.28);
    // swing_distribution =  std::uniform_real_distribution<double> (deg2rad(10), deg2rad(15));
    // phase_speed_distribution =  std::uniform_real_distribution<double> (0.8, 1.2);
    // noise_distribution =  std::uniform_real_distribution<double> (-deg2rad(0.), deg2rad(0.));
    // anglularvel_distribution =  std::uniform_real_distribution<double> (-deg2rad(0.), deg2rad(0.));
    // change_distribution = std::uniform_int_distribution<int>(30, 30);

    raisim::Vec<3> thigh_location;
    ik_dummy->getFramePosition(ik_dummy->getFrameByName("FL_thigh_joint"), thigh_location);
    raisim::Vec<3> calf_location;
    ik_dummy->getFramePosition(ik_dummy->getFrameByName("FL_calf_joint"), calf_location);
    raisim::Vec<3> foot_location;
    ik_dummy->getFramePosition(ik_dummy->getFrameByName("FL_foot_fixed"), foot_location);
    float ltc = (thigh_location.e() - calf_location.e()).norm();
    float lcf = (calf_location.e() - foot_location.e()).norm();
    float dt = 0.033333;
    mTG = TrajectoryGenerator(WALKING_TROT, ltc, lcf, dt);
    mTG.set_Ae(0.11);
    phi = M_PI;
    ang_vel = 0.;
    angle_cummul = 0.;

    int cnt = 0; 
    Eigen::Quaterniond prev_quat(gc_init_[0],gc_init_[1],gc_init_[2],gc_init_[3]);
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

  void generateRandomSeqSit()
  {
    f_hip_distribution = std::uniform_real_distribution<double> (-deg2rad(40), deg2rad(40));
    f_thigh_distribution = std::uniform_real_distribution<double> (-deg2rad(45), deg2rad(150));
    f_calf_distribution = std::uniform_real_distribution<double>(-deg2rad(140), -deg2rad(55));
    x_distribution = std::uniform_real_distribution<double> (deg2rad(-2), deg2rad(2));
    y_distribution = std::uniform_real_distribution<double> (deg2rad(-15), deg2rad(20));
    z_distribution = std::uniform_real_distribution<double> (deg2rad(-2), deg2rad(2));
    change_distribution = std::uniform_int_distribution<int> (20,40);

    // start
    int cnt = 0; 
    Eigen::Quaterniond prev_quat(gc_init_[0],gc_init_[1],gc_init_[2],gc_init_[3]);
    Eigen::VectorXd prev_pose = gc_init_.tail(12);
    motionBlender(prev_quat, prev_quat, prev_pose, prev_pose, 30);
    cnt+= 30;

    std::vector<Eigen::Vector3d> targets;
    Eigen::VectorXd gc_adj;
    gc_adj.setZero(19);
    gc_adj.head(3)= Eigen::Vector3d(0,0,0.27);
    gc_adj.tail(16) = gc_init_;
    gc_adj[2] = 0.12;
    for(int i = 0; i < 4; ++i)
    {
      raisim::Vec<3> current;
      vis_ref->getFramePosition(FR_FOOT+ 3 * i, current);
      targets.push_back(current.e());
    }
    mAIK->setReady();
    mAIK->setFullTargets(targets);
    mAIK->solve(gc_adj);
    auto sol = mAIK->getSolution();
    mAIK->clearTargets();
    motionBlender(prev_quat, prev_quat, prev_pose, sol, 6);
    prev_pose = sol;

    Eigen::Matrix3d m;
    m = Eigen::AngleAxisd(deg2rad(-90), Eigen::Vector3d::UnitY());
    Eigen::Quaterniond current_quat(m);
    Eigen::VectorXd sitpose(12);
    sitpose << 0.27, deg2rad(50), -1.05, -0.27, deg2rad(50), -1.05, 0., deg2rad(120), deg2rad(-120), 0., deg2rad(120), deg2rad(-120);
    motionBlender(prev_quat, current_quat, prev_pose, sitpose, 12);
    int stay = 30;
    motionBlender(current_quat, current_quat, sitpose, sitpose, stay);
    prev_quat = current_quat;
    prev_pose = sitpose;
    cnt += 48;

    std::random_device rd;
    std::mt19937 generator(rd());
    // Initialize for some duration
    int inbetween = change_distribution(generator);
    while(cnt < max_cnt)
    {
      std::random_device rd;
      std::mt19937 generator(rd());
      int inbetween = change_distribution(generator);
      float x_angle = x_distribution(generator);
      float y_angle = y_distribution(generator);
      float z_angle = z_distribution(generator);
      Eigen::Matrix3d m;
      m = Eigen::AngleAxisd(x_angle, Eigen::Vector3d::UnitX())
        * Eigen::AngleAxisd(y_angle + deg2rad(-90),  Eigen::Vector3d::UnitY())
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
      motionBlender(prev_quat, current_quat, prev_pose, current_pose, inbetween);
      prev_quat = current_quat;
      prev_pose = current_pose;
      cnt += inbetween;

      int inbetween2 = change_distribution(generator) / 4;
      motionBlender(current_quat, current_quat, current_pose, current_pose, inbetween2);
      cnt+= inbetween2;
    }
  }

  void generateSitPD()
  {
    std::string ref_poses = "/home/sonic/Project/RobotControl/sitaction.txt";
    std::ifstream human_input(ref_poses);
    if(!human_input){
      std::cout << "Expected valid input file name" << std::endl;
    }

    int triple_bone = 12;
    while(!human_input.eof()){
      Eigen::VectorXd humanoid(triple_bone);
      for(int i = 0 ; i < triple_bone; ++i)
      {
        float ang;
        human_input >> ang;
        humanoid[i] = ang;
      }
      sitPDseq.push_back(humanoid);
    }
    sitPDseq.pop_back();
    human_input.close();
  }

  void generateStandPD()
  {
    std::string ref_poses = "/home/sonic/Project/RobotControl/standaction.txt";
    std::ifstream human_input(ref_poses);
    if(!human_input){
      std::cout << "Expected valid input file name" << std::endl;
    }

    int triple_bone = 12;
    while(!human_input.eof()){
      Eigen::VectorXd humanoid(triple_bone);
      for(int i = 0 ; i < triple_bone; ++i)
      {
        float ang;
        human_input >> ang;
        humanoid[i] = ang;
      }
      standPDseq.push_back(humanoid);
    }
    standPDseq.pop_back();
    human_input.close();
  }

  void get_current_target(float phase_speed, Eigen::VectorXd noise)
  {
    Eigen::VectorXd target(16);
    Eigen::VectorXd gc_ = ik_dummy->getGeneralizedCoordinate().e();
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

  void motionBlender(Eigen::Quaterniond q1, Eigen::Quaterniond q2, 
    Eigen::VectorXd now, Eigen::VectorXd next, int inbetween)
  {
    for(int j = 0 ; j < inbetween ; ++j)
    {
      std::random_device rd;
      std::mt19937 generator(rd());
      Eigen::Quaterniond slerped = q1.slerp(float(j) / float(inbetween), q2);
      Eigen::VectorXd interpol = (now *(inbetween - j) / inbetween) + (next *j / inbetween);
      Eigen::VectorXd total(4 + 12);
      slerped.normalize();
      total[0] = slerped.w();
      total[1] = slerped.x();
      total[2] = slerped.y();
      total[3] = slerped.z();
      total.tail(12) = interpol;
      trajectories.push_back(total);
    }
  }

 private:
  std::unique_ptr<raisim::World> world_;
  raisim::ArticulatedSystem* alien_;
  raisim::ArticulatedSystem* vis_ref;
  raisim::ArticulatedSystem* ik_dummy;
  raisim::Sphere *vis_sphere;
  std::vector<raisim::GraphicObject> * fixball;

  MotionReshaper *mrl;
  bool startFlag;
  int num_envs_ = 1;
  int obDim_ = 0, actionDim_ = 0;
  int human_cnt = 0;
  int visualizationCounter_ = 0;
  float desired_fps_;
  std::string resourceDir_;
  // Yaml::Node cfg_;
  Eigen::VectorXd pTarget12_;
  Eigen::VectorXd gc_target;
  Eigen::VectorXd robot_observation;
  Custom custom = Custom(LOWLEVEL);
  std::vector<Eigen::VectorXd> action_history;
  std::vector<Eigen::VectorXd> state_history;
  std::vector<Eigen::VectorXd> reference_history;
  Eigen::VectorXd actionMean_, actionStd_, obScaled_;
  Eigen::VectorXd gc_init_;

  bool physics_simulated = false;
  // std::vector<Eigen::VectorXd> raw_trj;
  Eigen::Vector4i currentContact;
  Eigen::Vector4i prevContact;
  Eigen::VectorXd pTarget_;

  LoopFunc* loop_control;
  LoopFunc* loop_udpSend;
  LoopFunc* loop_udpRecv;

  bool imuInit = true;

  int sitcount;
  int standcount;

  // Stop flag for a safety
  bool stopFlag = false; 
  k4a_device_t device = nullptr;
  k4abt_tracker_t tracker = nullptr;
  uint8_t* img_bgra32;
  int colorHeight;
  int colorWidth;
  cv::Mat image;
  cv::VideoWriter vod;
  Eigen::Vector3d x_axis,y_axis, z_axis;
  Eigen::Vector3d root_x,root_y, root_z;
  Eigen::Vector3d chest_x,chest_y, chest_z;
  k4a_float3_t joint0Position;
  Samples::PointCloudGenerator *pointCloudGenerator;
  Samples::FloorDetector floorDetector;
  k4a_calibration_t sensorCalibration;
  Eigen::VectorXd human1, human2;

  float control_dt_;
  int current_mode;  int reserved_mode;
  bool inTransition = false;
  std::vector<Eigen::VectorXd> init_feet_poses;
  Eigen::Quaterniond rot_recorded = Eigen::Quaterniond(1,0,0,0);
  Eigen::Vector3d targetLocation;
  Eigen::Matrix3d rot_compensation;
  TrajectoryGenerator mTG;

  int max_cnt = 610;
  std::vector<Eigen::VectorXd> trajectories;
  std::vector<Eigen::VectorXd> sitseq;
  std::vector<Eigen::VectorXd> sitPDseq;
  std::vector<Eigen::VectorXd> standseq;
  std::vector<Eigen::VectorXd> standPDseq;
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
  // Duration
  std::uniform_int_distribution<int> change_distribution;
  std::uniform_int_distribution<int> leg_distribution;

  std::uniform_int_distribution<int> number_distribution;
  std::uniform_real_distribution<double> Ae_distribution;
  std::uniform_real_distribution<double> height_distribution;
  std::uniform_real_distribution<double> swing_distribution;
  std::uniform_real_distribution<double> phase_speed_distribution;
  std::uniform_real_distribution<double> noise_distribution;
  std::uniform_real_distribution<double> anglularvel_distribution;

  std::uniform_real_distribution<double> x_distribution;
  std::uniform_real_distribution<double> y_distribution;
  std::uniform_real_distribution<double> z_distribution;
  int current_count = 0;
  Eigen::VectorXd save_human_skel;
  AnalyticRIK* rik;
  AnalyticLIK* lik;
  AnalyticFullIK* mAIK;
  Eigen::VectorXd kin_gc_prev;
  k4abt_skeleton_t current_skeleton;
  k4abt_skeleton_t prev_skeleton;
  float confient_rad = 0.3;
  float init_foot;
  double phi, phase_speed;
  double ang_vel, angle_cummul;
  double simulation_dt_ = 0.0025;
  int delay;
  std::vector<Eigen::VectorXd> human_point_list;
  std::vector<Eigen::VectorXd> human_param_list;

  int current_state;
  bool mNeutral = false;
  bool processingFlag = false;
  bool vis_sphere_flag = false;
  bool kinect_set = false;
};



#endif //SRC_RAISIMGYMVECENV_HPP
