#include <stdio.h>
#include <iostream>
#include <raisim/OgreVis.hpp>
#include "raisimBasicImguiPanel.hpp"
#include "raisimKeyboardCallback.hpp"
#include "helper.hpp"
#include "glm/ext.hpp"
#include "glm/glm.hpp"
#include "../env/include/TrajectoryGenerator.hpp"
#include "../env/include/GaitMapper.hpp"
#include "../env/include/AnalyticIK.hpp"
// #include "../env/include/SampleGenerator.hpp"
#include "../env/visualizer_alien/BodyTrakingHelper.hpp"
#include "../../Warping_Test/SampleGenerator/include/MotionFunction.hpp"
#include "../../Warping_Test/SampleGenerator/include/PointCloudGenerator.hpp"
#include "../../Warping_Test/SampleGenerator/include/FloorDetector.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <random>

#include <k4arecord/playback.h>
#include <k4a/k4a.h>
#include <k4abt.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"


// #include <BodyTrackingHelpers.h>
#include <Utilities.h>

void setupCallback() {
  auto vis = raisim::OgreVis::get();

  /// light
  vis->getLight()->setDiffuseColour(1, 1, 1);
  vis->getLight()->setCastShadows(true);
  Ogre::Vector3 lightdir(-3,-3,-0.5);
  lightdir.normalise();
  vis->getLightNode()->setDirection({lightdir});

  /// load  textures
  vis->addResourceDirectory(vis->getResourceDir() + "/material/checkerboard");
  vis->loadMaterialFile("checkerboard.material");
  std::cout<< vis->getResourceDir() << std::endl;

  /// shdow setting
  vis->getSceneManager()->setShadowTechnique(Ogre::SHADOWTYPE_TEXTURE_ADDITIVE);
  vis->getSceneManager()->setShadowTextureSettings(2048, 3);

  /// scale related settings!! Please adapt it depending on your map size
  // beyond this distance, shadow disappears
  vis->getSceneManager()->setShadowFarDistance(30);
  // size of contact points and contact forces
  vis->setContactVisObjectSize(0.06, .6);
  // speed of camera motion in freelook mode
  vis->getCameraMan()->setTopSpeed(5);
}

class Temporal{
public:
  Temporal()
  {
    world.setTimeStep(0.0025);
    world.setERP(0,0);

    real_time_factor = 1.0;
    fps = 30.0;

    auto vis = raisim::OgreVis::get();
    /// these method must be called before initApp
    vis->setWorld(&world);
    vis->setWindowTitle("Alien");
    vis->setWindowSize(1800, 1200);
    vis->setImguiSetupCallback(imguiSetupCallback);
    vis->setImguiRenderCallback(imguiRenderCallBack);
    vis->setKeyboardCallback(raisimKeyboardCallback);
    vis->setSetUpCallback(setupCallback);
    vis->setAntiAliasing(2);
    vis->setDesiredFPS(30);
    captureSkel = false;

    // auto alien = world.addArticulatedSystem(raisim::loadResource("a1/a1/a1.urdf"));
    alien = world.addArticulatedSystem(raisim::loadResource("a1/a1/a1.urdf"));
    ik_dummy = new raisim::ArticulatedSystem(raisim::loadResource("a1/a1/a1.urdf"));

    // raisim::gui::manualStepping = true;

    // /// starts visualizer thread
    vis->initApp();

    // /// create raisim objects
    auto ground = world.addGround(0, "terrain");
    vis->createGraphicalObject(ground, 20, "terrain", "checkerboard_green");

    auto alien_graphics = vis->createGraphicalObject(alien, "Alien");

    raisim::MaterialManager materials_;
    materials_.setMaterialPairProp("terrain", "robot", 0.1, 0.0, 0.0);
    // materials_.setMaterialPairProp("robot", "robot", 1.2, 0.0, 0.0);

    world.setERP(0.0, 0.0);

    world.updateMaterialProp(materials_);
    alien->getCollisionBody("FR_foot/0").setMaterial("robot");
    alien->getCollisionBody("FL_foot/0").setMaterial("robot");
    alien->getCollisionBody("RL_foot/0").setMaterial("robot");
    alien->getCollisionBody("RR_foot/0").setMaterial("robot");

    int gcDim_ = alien->getGeneralizedCoordinateDim();
    int gvDim_ = alien->getDOF();
    Eigen::VectorXd gc_;
    gc_init_.setZero(gcDim_); gc_.setZero(gcDim_);
    gc_init_ <<  0, 0, 0.27, 1.0, 0.0, 0.0, 0.0, 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99);
    alien->setGeneralizedCoordinate(gc_init_);
    // ref_dummy->setState(gc_init_, gv_init_);

    int nJoints_ = 12;
    float effort_limit = 33.5 * 0.90;
    // float effort_limit = 40.0* 0.8;
    Eigen::VectorXd torque_upperlimit = alien->getUpperLimitForce().e();
    Eigen::VectorXd torque_lowerlimit = alien->getLowerLimitForce().e();
    torque_upperlimit.tail(nJoints_).setConstant(effort_limit);
    torque_lowerlimit.tail(nJoints_).setConstant(-effort_limit);
    alien->setActuationLimits(torque_upperlimit, torque_lowerlimit);

    Eigen::VectorXd jointPgain; Eigen::VectorXd jointDgain; 
    jointPgain.setZero(gvDim_); jointPgain.tail(nJoints_).setConstant(200);
    jointDgain.setZero(gvDim_); jointDgain.tail(nJoints_).setConstant(10);
    pTarget_.setZero(gcDim_); 
    pTarget_.tail(nJoints_) = gc_init_.tail(nJoints_);
    vTarget_.setZero(gvDim_);
    alien->setPdGains(jointPgain, jointDgain);
    alien->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
    alien->setPdTarget(pTarget_, vTarget_);

    // world.setMaterialPairProp("floor", "FR_fOOT", 1.7, 0.0, 0.0);
    // world.setDefaultMaterial(1.7, 0.0, 0.0);
    Aik = new AnalyticFullIK(ik_dummy);

    state0 = 55; state1 = state0 + 15; 
    state2 = state1 + 10; state3 = state2+ 16;
    state4 = state3 + 5;
      //  set camera1
    vis->select(alien_graphics->at(0), false);
    vis->getCameraMan()->setYawPitchDist(Ogre::Radian(-2.17), Ogre::Radian(-1.3), 5, true);
    vis->startRecordingVideo("Video.mp4");

    // auto feet = readHumanFeet("/home/sonic/Project/Warping_Test/rsc/samples_motion.txt");
    // auto results = readResult12D("/home/sonic/Project/Warping_Test/rsc/motion_result.txt");
    // // auto results = readResult12D("/home/sonic/Project/Warping_Test/rsc/new_motion.txt");

    // int k =0;
    kinect_init();
    // MotionReshaper *mrl = new MotionReshaper(ref_dummy);
  }

  void begin_write()
  {

    std::string filePath1 = "/home/sonic/Project/Alien_Gesture/KinectRaisimtest/build/rsc/points_temp.txt";
    std::string filePath2 = "/home/sonic/Project/Alien_Gesture/KinectRaisimtest/build/rsc/params_temp.txt";
    // write File
    writePoints.open(filePath1);
    writeParams.open(filePath2);
  }

  struct InputSettings
  {
    k4a_depth_mode_t DepthCameraMode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    bool CpuOnlyMode = false;
    bool Offline = false;
    std::string FileName;
  };

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

  void CloseDevice(){
    k4abt_tracker_shutdown(tracker);
    k4abt_tracker_destroy(tracker);

    k4a_device_stop_cameras(device);
    k4a_device_close(device);
  }


  void visualize_bones(k4abt_body_t body)
  {
    float scaler = -1. / 1600;
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
        list["bone" + std::to_string(boneIdx)].setScale(Eigen::Vector3d(0.015, 0.015, len));
        list["bone" + std::to_string(boneIdx)].setOrientation(rot);
      }
    }
    root_x.setZero(); root_y.setZero(); root_z.setZero(); 
    chest_x.setZero(); chest_y.setZero(); chest_z.setZero(); 
    for(size_t boneIdx = 0; boneIdx < K4ABT_JOINT_COUNT; boneIdx++)
    {
      k4a_float3_t jointPosition = body.skeleton.joints[boneIdx].position;
      k4abt_joint_confidence_level_t conf = body.skeleton.joints[boneIdx].confidence_level;
      Eigen::Vector3d point = scaler * axis_orientation.inverse() *  Eigen::Vector3d(jointPosition.xyz.z- joint0Position.xyz.z ,-jointPosition.xyz.x + joint0Position.xyz.x, jointPosition.xyz.y- joint0Position.xyz.y);
      point[1] -= y_shift;
      // if(conf == K4ABT_JOINT_CONFIDENCE_NONE) std::cout << "zero" << std::endl;
      // if(conf == K4ABT_JOINT_CONFIDENCE_LOW) std::cout << "low" << std::endl;
      if(boneIdx ==K4ABT_JOINT_HAND_RIGHT ||boneIdx ==K4ABT_JOINT_HANDTIP_RIGHT || boneIdx == K4ABT_JOINT_THUMB_RIGHT|| boneIdx ==K4ABT_JOINT_HAND_LEFT ||boneIdx ==K4ABT_JOINT_HANDTIP_LEFT || boneIdx == K4ABT_JOINT_THUMB_LEFT)
      {
        list["joint" + std::to_string(boneIdx)].setPosition(point);
        list["joint" + std::to_string(boneIdx)].setScale(Eigen::Vector3d(0.0, 0.0, 0.0));
      }
      else
        list["joint" + std::to_string(boneIdx)].setPosition(point);

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
    std::cout << rad << std::endl;
    if(rad > confient_rad) return false;
    else return true;
  }

  bool isFlipped()
  {

  }

  void inferSkel()
  {

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
    // for(int i =0 ; i< 1; ++i)
    // {
    //   GetFromDevice();
    // }
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
    return human_pose;
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

  std::vector<Eigen::VectorXd> readResult12D(std::string file_result)
  {
    // Read angle && read result
    // Contain Eigen::VecXd ... Eigen::Vec3D
    // Solve IK
    std::vector<Eigen::VectorXd> result_angles12d;
    std::ifstream result_endeffector_file(file_result);
    if(!result_endeffector_file){
      std::cout << "Expected valid result file name" << std::endl;
    }

    int robot_ang = 12 + 4;

    while(!result_endeffector_file.eof()){
      Eigen::VectorXd result_sol(robot_ang);
      for(int i = 0 ; i < robot_ang; ++i)
      {
        float end;
        result_endeffector_file >> end;
        result_sol[i] = end;
      }
      result_angles12d.push_back(result_sol);
    }
    result_endeffector_file.close();
    return result_angles12d;
  }

  std::vector<float> readHumanFeet(std::string human_input)
  {
    // Read angle && read result
    // Contain Eigen::VecXd ... Eigen::Vec3D
    // Solve IK
    std::vector<float> right_left;
    std::ifstream input_angle_file(human_input);
    if(!input_angle_file){
      std::cout << "Expected valid input file name" << std::endl;
    }
    int arm_num_len = 55;

    while(!input_angle_file.eof()){
      Eigen::VectorXd humanoid(arm_num_len);
      for(int i = 0 ; i < arm_num_len; ++i)
      {
        float ang;
        input_angle_file >> ang;
      }
      // Dummy 
      for(int i = 0 ; i < 16; ++i)
      {
        float ang;
        input_angle_file >> ang;
        if(i == 12 || i == 15)
          right_left.push_back(ang);
      }
    }
    input_angle_file.close();
    return right_left;
  }

  void kinect_init() 
  {
    InputSettings inputSettings;
    SetFromDevice(inputSettings);
    auto vis = raisim::OgreVis::get();
    image = cv::Mat::ones(colorHeight,colorWidth,CV_8UC3);
    namedWindow("Human_Motion", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("Human_Motion", 2000, 800);
    cv::Size frame_size(colorWidth, colorHeight);
    vod = cv::VideoWriter("/home/sonic/Project/Alien_Gesture/human_video/Human.avi", 
    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30., frame_size, true);
    for(int i =0 ; i < 31; ++i){
      vis->addVisualObject("bone" + std::to_string(i), "cylinderMesh", "red", {0.015, 0.015, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
    }
    for(int i = 0 ; i < K4ABT_JOINT_COUNT; ++i)
    {
      if(i == K4ABT_JOINT_HAND_RIGHT)
      {
        vis->addVisualObject("joint" + std::to_string(i), "sphereMesh", "orange", {0.03, 0.03, 0.03}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);     
      }
      else
        vis->addVisualObject("joint" + std::to_string(i), "sphereMesh", "blue", {0.03, 0.03, 0.03}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
    }
  }


  // Implement the file writing code here
  // Capture the properties for mapping
  // And capture the point information to reconstruct the info
  void capture_current_skeleton()
  { 
    Eigen::VectorXd current_human = getHumanPose();
    if(isnan(current_human[0]) || isnan(current_human[10])) return;
    auto& list = raisim::OgreVis::get()->getVisualObjectList();
    for(size_t boneIdx = 0; boneIdx < K4ABT_JOINT_COUNT; boneIdx++)
    {
      auto joint_pose = list["joint" + std::to_string(boneIdx)].offset;
      writePoints << std::setprecision(3) << std::fixed << joint_pose[0]<< " ";
      writePoints << std::setprecision(3) << std::fixed << joint_pose[1]<< " ";
      writePoints << std::setprecision(3) << std::fixed << joint_pose[2]<< " ";
    }
    writePoints << std::endl;

    for(size_t i = 0; i < current_human.size(); i++)
    {
      writeParams << std::setprecision(3) << std::fixed << current_human[i]<< " ";
    }
    writeParams << std::endl;
    captureSkel = false;
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

  void generateSit()
  {
    Eigen::Quaterniond prev_quat(gc_init_[3],gc_init_[4],gc_init_[5],gc_init_[6]);
    Eigen::VectorXd prev_pose = gc_init_.tail(12);
    motionBlender(prev_quat, prev_quat, prev_pose, prev_pose, 10);
    std::vector<Eigen::Vector3d> targets;
    Eigen::VectorXd gc_adj = gc_init_;
    gc_adj[2] = 0.15;
    for(int i = 0; i < 4; ++i)
    {
      raisim::Vec<3> current;
      alien->getFramePosition(FR_FOOT+ 3 * i, current);
      targets.push_back(current.e());
    }
    Aik->setReady();
    // Aik->setCurrentPose(gc_adj);
    Aik->setFullTargets(targets);
    Aik->solve(gc_adj);
    auto sol = Aik->getSolution();
    Aik->clearTargets();
    motionBlender(prev_quat, prev_quat, prev_pose, sol, 6);
    prev_pose = sol;

    gc_adj[2] = (0.45 + 0.15)/2;

    Eigen::Matrix3d m;
    m = Eigen::AngleAxisd(deg2rad(-65), Eigen::Vector3d::UnitY());
    Eigen::Quaterniond current_quat(m);
    gc_adj[3] = current_quat.w(); gc_adj[4] = current_quat.x();
    gc_adj[5] = current_quat.y(); gc_adj[6] = current_quat.z();

    Aik->setReady();
    Aik->setFullTargets(targets);
    Aik->solve(gc_adj);
    auto sol2 = Aik->getSolution();
    Eigen::VectorXd onlyupper = prev_pose;
    onlyupper.head(6) << 0.26406, 0.5, -1.05, -0.277623, 0.5, -1.05;
    // onlyupper.head(6) = sol2.head(6);
    // onlyupper[7] = sol2[7];
    // onlyupper[10] = sol2[10];
    Aik->clearTargets();
    motionBlender(prev_quat, current_quat, prev_pose, onlyupper, 4);
    // motionBlender(current_quat, current_quat, onlyupper, onlyupper, 1);
    prev_quat = current_quat;
    prev_pose = onlyupper;

    m = Eigen::AngleAxisd(deg2rad(-90), Eigen::Vector3d::UnitY());
    Eigen::Quaterniond current_quat2(m);
    gc_adj[3] = current_quat2.w(); gc_adj[4] = current_quat2.x();
    gc_adj[5] = current_quat2.y(); gc_adj[6] = current_quat2.z();
    
    gc_adj[0] = -0.2805;
    gc_adj[2] = 0.1805 + 0.1* sqrt(3)+0.01;

    Aik->setReady();
    Aik->setFullTargets(targets);
    Aik->solve(gc_adj);
    auto sol3 = Aik->getSolution();
    onlyupper = sol3;
    onlyupper.head(6) << 0.26406, deg2rad(90), -1.05, -0.277623, 0.9, -1.05;

    // onlyupper.tail(6) << 0.26406, 0.8, -1.05, -0.277623, 0.8, -1.05;
    // onlyupper.head(6) = sol2.head(6);
    onlyupper[7] = deg2rad(120);
    onlyupper[8] = deg2rad(-120);
    onlyupper[10] = deg2rad(120);
    onlyupper[11] = deg2rad(-120);
    Aik->clearTargets();
    motionBlender(prev_quat, current_quat2, prev_pose, onlyupper, 7);
    motionBlender(current_quat2, current_quat2, onlyupper, onlyupper, 1);
    prev_quat = current_quat2;
  }

  void run()
  {
    auto vis = raisim::OgreVis::get();
    double time = 0;
    for(int i = 0 ; i < 10000; ++i) {
      time -= real_time_factor / fps;
      Eigen::VectorXd currentPtarget(12);
      Eigen::VectorXd desired = gc_init_;
      if(i < trajectories.size())
      {
        currentPtarget = trajectories[i].tail(12);
        desired.tail(16) = trajectories[i];
      }
      else
      {
        currentPtarget = trajectories[trajectories.size()-1].tail(12);
        desired.tail(16) = trajectories[trajectories.size()-1];
        desired[0] = -(0.1805 + 0.05);
        desired[2] = 0.1805 + 0.1* sqrt(3);
      }
      pTarget_.tail(12) = currentPtarget;
      // alien->setPdTarget(pTarget_, vTarget_);
      std::cout << pTarget_.tail(12).transpose() << std::endl;
      alien->setGeneralizedCoordinate(desired);
      while (time < 0.0) {
        // world.integrate();
        time += world.getTimeStep();
      }
      vis->renderOneFrame();
      unsigned int t = 1;
      // sleep(t);
      if(!stopFlag) break;
    } 
  }

  void runNWrite()
  {
    auto vis = raisim::OgreVis::get();
    double time = 0;
    for(int i = 0 ; i < 10000; ++i) {
      time -= real_time_factor / fps;
      while (time < 0.0) {
        world.integrate();
        time += world.getTimeStep();
      }
      GetFromDevice();
      if(captureSkel) capture_current_skeleton();
      vis->renderOneFrame();
      if(!stopFlag) break;
    } 
  }

  void runWriteMotion()
  {
    auto vis = raisim::OgreVis::get();
    double time = 0;
    for(int i = 0 ; i < 10000; ++i) {
      time -= real_time_factor / fps;
      while (time < 0.0) {
        world.integrate();
        time += world.getTimeStep();
      }
      GetFromDevice();
      bgra2Mat();
      vod.write(image);
      imshow("Human_Motion", image);
      cv::waitKey(1);
      capture_current_skeleton();
      vis->renderOneFrame();
      if(!stopFlag) break;
    } 
  }

  void closeSimple()
  {
    auto vis = raisim::OgreVis::get();
    vis->stopRecordingVideoAndSave();
    vis->closeApp();
  }
  void close()
  {
    writePoints.close();
    writeParams.close();
    auto vis = raisim::OgreVis::get();
    CloseDevice();
    vis->stopRecordingVideoAndSave();
    vis->closeApp();
  }


private:
  raisim::World world;
  raisim::ArticulatedSystem* alien;
  raisim::ArticulatedSystem* ik_dummy;
  k4a_device_t device = nullptr;
  k4abt_tracker_t tracker = nullptr;
  GaitMapper gm;
  uint8_t* img_bgra32;
  int colorHeight;
  int colorWidth;
  cv::Mat image;
  cv::VideoWriter vod;
  Eigen::Vector3d x_axis,y_axis, z_axis;
  Eigen::Vector3d root_x,root_y, root_z;
  Eigen::Vector3d chest_x,chest_y, chest_z;
  bool imuInit = true;
  k4a_float3_t joint0Position;
  bool startFlag = true;
  Samples::PointCloudGenerator *pointCloudGenerator;
  Samples::FloorDetector floorDetector;
  k4a_calibration_t sensorCalibration;
  k4abt_skeleton_t current_skeleton;
  k4abt_skeleton_t prev_skeleton;
  float confient_rad = 0.3;
  std::vector<Eigen::VectorXd> trajectories;
  std::ofstream writePoints, writeParams;
  int state0, state1, state2, state3, state4;
  double fps, real_time_factor;
  Eigen::VectorXd gc_init_;
  Eigen::VectorXd pTarget_; Eigen::VectorXd vTarget_;
  AnalyticFullIK* Aik;
};



int main(int argc, char **argv) {
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);
  raisim::World::setActivationKey(binaryPath.getDirectory() + "/../rsc/activation.raisim");
  /// create raisim world
  Temporal *tp = new Temporal();
  // tp->generateSit();
  // tp->run();
  // tp->closeSimple();

  tp->begin_write();
  tp->runWriteMotion();
  tp->close();

  // CloseDevice();

  return 0;
}