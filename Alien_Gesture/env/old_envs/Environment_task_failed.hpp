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

#include "../visualizer_alien/raisimKeyboardCallback.hpp"
#include "../visualizer_alien/helper.hpp"
#include "../visualizer_alien/guiState.hpp"
#include "../visualizer_alien/raisimBasicImguiPanel.hpp"

#define deg2rad(ang) ((ang) * M_PI / 180.0)
#define rad2deg(ang) ((ang) * 180.0 / M_PI)

#define LEFT_HIP 10
#define LEFT_THIGH 11
#define LEFT_CALF 12

// std::uniform_real_distribution<double> hip_distribution(-1.22, 1.22);
std::uniform_real_distribution<double> hip_distribution(-0.5, 0.5);
// std::uniform_real_distribution<double> thigh_distribution(-3.14, 3.14);
std::uniform_real_distribution<double> thigh_distribution(deg2rad(-40), deg2rad(40));
// std::uniform_real_distribution<double> calf_distribution(-2.78, -0.65);
std::uniform_real_distribution<double> calf_distribution(-1.7, -1.00);


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
    gc_init_ << 0, 0, 0.43, 1.0, 0.0, 0.0, 0.0, 0.045377, 0.667921, -1.23225,0.045377, 0.667921, -1.23225, 0.045377, 0.8, -1.4, 0.045377, 0.8, -1.4;
    // gc_init_ << 0, 0, 0.43, 1.0, 0.0, 0.0, 0.0, 0.045377, 0.667921, -1.23225,0.045377, 0.667921, -1.23225, 0.045377, 0.667921, -1.23225, 0.045377, 0.667921, -1.23225;
    alien_->setState(gc_init_, gv_init_);
    gc_init_ = init_pose_gen();
    alien_->setState(gc_init_, gv_init_);


    /// set pd gains
    double pgain = 300;
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(pgain);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(1.4 * std::sqrt(pgain));
    alien_->setPdGains(jointPgain, jointDgain);
    alien_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    // des angleR, current angleR, des angleL, current angleL ,N joints 
    obDim_ = 26 + nJoints_; 
    actionDim_ = nJoints_;
    actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obMean_.setZero(obDim_); obStd_.setZero(obDim_);

    /// action & observation scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_ << 0.3, 0.7, 0.7, 0.7, 0.7, 0.7, 0.3, 0.7, 0.7, 0.3, 0.7, 0.7;

    task_w = 0.4;
    balance_w = 0.6;

    task_scale = 30;
    end_scale = 30;
    root_scale = 1.0;
    support_scale = 1000;

    current_count = 0;
    change_count = 24;
    scaler = -1.0 / 2000;

    angle_smoother.push_back(0.7);
    angle_smoother.push_back(1.0);
    angle_smoother.push_back(0.7);

    end_names.push_back("FR_foot_fixed");
    // end_names.push_back("FL_foot_fixed");
    end_names.push_back("RR_foot_fixed");
    end_names.push_back("RL_foot_fixed");

    gui::rewardLogger.init({"TaskReward", "EndReward", "SupportReward", "RotReward"});

    /// indices of links that should not make contact with ground
    footIndices_.insert(alien_->getBodyIdx("FL_calf"));
    footIndices_.insert(alien_->getBodyIdx("FR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RR_calf"));
    footIndices_.insert(alien_->getBodyIdx("RL_calf"));

    desired_footIndices_.push_back(alien_->getBodyIdx("FR_calf"));
    desired_footIndices_.push_back(alien_->getBodyIdx("RR_calf"));
    desired_footIndices_.push_back(alien_->getBodyIdx("RL_calf"));

    designed_motion.push_back(Eigen::Vector3d(1.1, deg2rad(0), -0.7));
    designed_motion.push_back(Eigen::Vector3d(-0.8, deg2rad(80), -1.2));
    designed_motion.push_back(Eigen::Vector3d(1.1, -deg2rad(80), -1.8));
    designed_motion.push_back(Eigen::Vector3d(-1.1, -deg2rad(80), -0.8));
    
    for(int i = 0; i < end_names.size(); ++i)
    {
      raisim::Vec<3> tempPosition;
      alien_->getFramePosition(alien_->getFrameByName(end_names[i]), tempPosition);
      feetPoses.push_back(tempPosition.e());
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
      vis->addVisualObject("sphere_knee_indicator", "sphereMesh", "green", {0.03, 0.03, 0.03}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("sphere_foot_indicator", "sphereMesh", "green", {0.03, 0.03, 0.03}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("com", "sphereMesh", "blueEmit", {0.02, 0.02, 0.02}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("supportPoly0", "cylinderMesh", "red", {0.01, 0.01, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("supportPoly1", "cylinderMesh", "red", {0.01, 0.01, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      vis->addVisualObject("supportPoly2", "cylinderMesh", "red", {0.01, 0.01, 0.5}, false, raisim::OgreVis::RAISIM_OBJECT_GROUP);
      auto& list = raisim::OgreVis::get()->getVisualObjectList();
      
      for(int i = 0; i < 3; ++i){
        int j;
        if(i==2) j = 0;
        else j = i+1;
        Eigen::Vector3d half = (feetPoses[j] + feetPoses[i]) / 2 ;
        float len = (feetPoses[j] - feetPoses[i]).norm();
        raisim::Mat<3,3> rot;
        Eigen::Vector3d way = (feetPoses[j] - feetPoses[i]).normalized();
        raisim::Vec<3> direction = {way[0], way[1], way[2]};
        raisim::zaxisToRotMat(direction, rot);

        list["supportPoly"+ std::to_string(i)].setPosition(half);
        list["supportPoly"+ std::to_string(i)].setScale(Eigen::Vector3d(0.01, 0.01, len));
        list["supportPoly"+ std::to_string(i)].setOrientation(rot);
      }
      desired_fps_ = 60.;
      vis->setDesiredFPS(desired_fps_);
    }
  }

  Eigen::VectorXd init_pose_gen()
  {
    std::vector<Eigen::Vector3d> Ends;
    std::vector<bool> linear;
    std::vector<std::string> names;

    names.push_back("floating_base");
    names.push_back("floating_base");
    names.push_back("FL_foot_fixed");
    names.push_back("FR_foot_fixed");
    names.push_back("RL_foot_fixed");
    names.push_back("RR_foot_fixed");

    Ends.push_back(Eigen::Vector3d(0.0,0.0,0.36));
    linear.push_back(true);
    Ends.push_back(Eigen::Vector3d(0, 0, 0));
    linear.push_back(false);
    auto mIK = new IKSolver(alien_, names);

    for(int i = 2; i < names.size();++i)
    {
      raisim::Vec<3> footPosition;
      alien_->getFramePosition(alien_->getFrameByName(names[i].substr(0,2) + "_thigh_joint"), footPosition);
      Ends.push_back(Eigen::Vector3d(footPosition[0],footPosition[1],0));
      linear.push_back(true);
    }
    Eigen::VectorXd init_p = mIK->solve(Ends, linear, 10000);
    return init_p;
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
    connection.push_back(3);
    connection.push_back(4);
    connection.push_back(5);
    connection.push_back(0);
    connection.push_back(1);
    connection.push_back(2);

    startFlag = true;
    for(int i = 0; i < 6; ++i)
      points.push_back(Eigen::Vector3d(0,0,0));
  }

  void update_target_coords()
  {
    raisim::Mat<3,3> orientation_r;
    alien_->getBaseOrientation(orientation_r);
    Eigen::Matrix3d rot = orientation_r.e();

    raisim::Vec<3> ref_hip;
    raisim::Vec<3> ref_knee;
    raisim::Vec<3> ref_foot;

    ref_dummy->getFramePosition(ref_dummy->getFrameByName("FL_hip_joint"), ref_hip);
    ref_dummy->getFramePosition(ref_dummy->getFrameByName("FL_calf_joint"), ref_knee);
    ref_dummy->getFramePosition(ref_dummy->getFrameByName("FL_foot_fixed"), ref_foot);
    ref_knee_b = rot.transpose() * (ref_knee.e() - ref_hip.e());
    ref_foot_b = rot.transpose() * (ref_foot.e() - ref_hip.e());    

    raisim::Vec<3> alien_hip;
    raisim::Vec<3> alien_knee;
    raisim::Vec<3> alien_foot;

    alien_->getFramePosition(alien_->getFrameByName("FL_hip_joint"), alien_hip);
    alien_->getFramePosition(alien_->getFrameByName("FL_calf_joint"), alien_knee);
    alien_->getFramePosition(alien_->getFrameByName("FL_foot_fixed"), alien_foot);
    alien_knee_b = rot.transpose() * (alien_knee.e() - alien_hip.e());
    alien_foot_b = rot.transpose() * (alien_foot.e() - alien_hip.e());
  }


  void put_boxes()
  {
    std::vector<raisim::Box*> cubes;
    float mass = 0.5;
    auto vis = raisim::OgreVis::get();

    raisim::SingleBodyObject *ob = nullptr;
    cubes.push_back(world_->addBox(0.1, 0.5, 0.3, 100 * mass));
    vis->createGraphicalObject(cubes.back(), "cubes1", "blue");
    ob = cubes.back();
    Eigen::Vector3d pose;
    pose[0] = 0.4;
    pose[1] = 0.0;
    pose[2] = 0.15;
    ob->setPosition(pose);
    cubes.push_back(world_->addBox(0.08, 0.08, 0.08, mass));
    vis->createGraphicalObject(cubes.back(), "cubes2", "red");
    ob = cubes.back();
    pose[2] = 0.3 + 0.04;
    ob->setPosition(pose);
  }

  void reset() final
  {
    alien_->setState(gc_init_, gv_init_);
    ref_dummy ->setState(gc_init_, gv_init_);

    current_count = 0;
    curriculum_cnt = 0;
    desired_angleH = 0.05;
    desired_angleT = 1.6;
    desired_angleC = -1.7;

    pTarget_.setZero();
    pTarget_.tail(nJoints_) = gc_init_.tail(12);
    alien_->setPdTarget(pTarget_, vTarget_);
    mCOP = alien_->getCompositeCOM().e();
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
    
    
    for(int i=0; i<loopCount; i++) {
      world_->integrate();
      alien_->getState(gc_, gv_);
      CenterOfPressure();
      for(int i = 0; i < end_names.size(); ++i)
      {
        raisim::Vec<3> tempPosition2;
        alien_->getFramePosition(alien_->getFrameByName(end_names[i]), tempPosition2);
        feetPoses[i] = tempPosition2.e();
      }
      Eigen::VectorXd ref_pose = gc_;
      ref_pose[LEFT_HIP] = desired_angleH;
      ref_pose[LEFT_THIGH] = desired_angleT;
      ref_pose[LEFT_CALF] = desired_angleC;
      ref_dummy->setGeneralizedCoordinate(ref_pose);
      update_target_coords();

      if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
      {
        auto vis = raisim::OgreVis::get();
        auto& list = raisim::OgreVis::get()->getVisualObjectList();
        
        raisim::Vec<3> tempPosition;
        ref_dummy->getFramePosition(ref_dummy->getFrameByName("FL_calf_joint"), tempPosition);
        list["sphere_knee_indicator"].setPosition(tempPosition);
        ref_dummy->getFramePosition(ref_dummy->getFrameByName("FL_foot_fixed"), tempPosition);
        list["sphere_foot_indicator"].setPosition(tempPosition);

        Eigen::Vector3d com = alien_->getCompositeCOM().e();
        com[2] = 0;
        // list["com"].setPosition(com);
        list["com"].setPosition(mCOP);
        for(int i = 0; i < 3; ++i){
          int j;
          if(i==2) j = 0;
          else j = i+1;
          Eigen::Vector3d half = (feetPoses[j] + feetPoses[i]) / 2 ;
          float len = (feetPoses[j] - feetPoses[i]).norm();
          raisim::Mat<3,3> rot;
          Eigen::Vector3d way = (feetPoses[j] - feetPoses[i]).normalized();
          raisim::Vec<3> direction = {way[0], way[1], way[2]};
          raisim::zaxisToRotMat(direction, rot);

          list["supportPoly"+ std::to_string(i)].setPosition(half);
          list["supportPoly"+ std::to_string(i)].setScale(Eigen::Vector3d(0.01, 0.01, len));
          list["supportPoly"+ std::to_string(i)].setOrientation(rot);
        }
        vis->renderOneFrame();
      }      
      visualizationCounter_++;
    }

    updateObservation();

    float totalReward = CalcReward();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("TaskReward", task_reward);
      gui::rewardLogger.log("EndReward",  end_reward);
      gui::rewardLogger.log("SupportReward",  support_reward);
      gui::rewardLogger.log("RotReward",  root_reward);

      /// reset camera
      auto vis = raisim::OgreVis::get();

      vis->select(anymalVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }
    
    // current_count++;
    // if(current_count == change_count){
    //   update_random_leaning_angle();
    //   current_count = 0;
    // }

    if(!mStopFlag) 
      return totalReward;
    else
      return -1E10;
  }

  float stepTest(const Eigen::Ref<EigenVec>& action) final {
       /// action scaling
    Eigen::VectorXd currentAction = action.cast<double>();
    pTarget12_ = currentAction;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += 0.3 * actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;
    alien_->setPdTarget(pTarget_, vTarget_);

    auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
    auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);    
    
    
    for(int i=0; i<loopCount; i++) {
      world_->integrate();
      alien_->getState(gc_, gv_);
      CenterOfPressure();
      for(int i = 0; i < end_names.size(); ++i)
      {
        raisim::Vec<3> tempPosition2;
        alien_->getFramePosition(alien_->getFrameByName(end_names[i]), tempPosition2);
        feetPoses[i] = tempPosition2.e();
      }
      Eigen::VectorXd ref_pose = gc_;
      ref_pose[LEFT_HIP] = desired_angleH;
      ref_pose[LEFT_THIGH] = desired_angleT;
      ref_pose[LEFT_CALF] = desired_angleC;
      ref_dummy->setGeneralizedCoordinate(ref_pose);
      update_target_coords();
      if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
      {
        auto vis = raisim::OgreVis::get();
        auto& list = raisim::OgreVis::get()->getVisualObjectList();

        raisim::Vec<3> tempPosition;
        ref_dummy->getFramePosition(ref_dummy->getFrameByName("FL_calf_joint"), tempPosition);
        list["sphere_knee_indicator"].setPosition(tempPosition);
        ref_dummy->getFramePosition(ref_dummy->getFrameByName("FL_foot_fixed"), tempPosition);
        list["sphere_foot_indicator"].setPosition(tempPosition);

        Eigen::Vector3d com = alien_->getCompositeCOM().e();
        com[2] = 0;
        // list["com"].setPosition(com);
        list["com"].setPosition(mCOP);

        for(int i = 0; i < 3; ++i){
          int j;
          if(i==2) j = 0;
          else j = i+1;
          Eigen::Vector3d half = (feetPoses[j] + feetPoses[i]) / 2 ;
          float len = (feetPoses[j] - feetPoses[i]).norm();
          raisim::Mat<3,3> rot;
          Eigen::Vector3d way = (feetPoses[j] - feetPoses[i]).normalized();
          raisim::Vec<3> direction = {way[0], way[1], way[2]};
          raisim::zaxisToRotMat(direction, rot);

          list["supportPoly"+ std::to_string(i)].setPosition(half);
          list["supportPoly"+ std::to_string(i)].setScale(Eigen::Vector3d(0.01, 0.01, len));
          list["supportPoly"+ std::to_string(i)].setOrientation(rot);
        }
        vis->renderOneFrame();
      }      
      visualizationCounter_++;
    }

    updateObservation();

    float totalReward = CalcReward();

    if(visualizeThisStep_) {
      gui::rewardLogger.log("TaskReward", task_reward);
      gui::rewardLogger.log("EndReward",  end_reward);
      gui::rewardLogger.log("SupportReward",  support_reward);
      gui::rewardLogger.log("RotReward",  root_reward);

      /// reset camera
      auto vis = raisim::OgreVis::get();

      vis->select(anymalVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }
    
    // current_count++;
    // if(current_count == change_count){
    //   update_random_leaning_angle();
    //   current_count = 0;
    // }

    if(!mStopFlag) 
      return totalReward;
    else
      return -1E10;
  }

  bool isInside()
  {
    float crosser = 1.0;
    Eigen::Vector3d com = alien_->getCompositeCOM().e();
    for(int i = 0; i < end_names.size(); ++i)
    {
      int j;
      if(i==2) j = 0;
      else j = i+1;
      Eigen::Vector3d dir = feetPoses[j] - feetPoses[i];
      // crosser *= (com[0]*dir[1] - com[1]*dir[0]);
      crosser *= (mCOP[0]*dir[1] - mCOP[1]*dir[0]);
    }
    if(crosser > 0) return true;
    else  return false;
  }

  bool isInside(std::vector<Eigen::Vector3d> safe)
  {
    float crosser = 1.0;
    Eigen::Vector3d com = alien_->getCompositeCOM().e();
    for(int i = 0; i < end_names.size(); ++i)
    {
      int j;
      if(i==2) j = 0;
      else j = i+1;
      Eigen::Vector3d dir = safe[j] - safe[i];
      // crosser *= (com[0]*dir[1] - com[1]*dir[0]);
      crosser *= (mCOP[0]*dir[1] - mCOP[1]*dir[0]);
    }
    if(crosser > 0) return true;
    else  return false;
  }

  float distToSegment(Eigen::Vector3d pt, Eigen::Vector3d p1, Eigen::Vector3d p2)
  { 
    float dx = p2[0] - p1[0];
    float dy = p2[1] - p1[1];
    if((dx == 0) && (dy == 0))
    {
        // It's a point not a line segment.
        dx = pt[0] - p1[0];
        dy = pt[1] - p1[1];
        return dx * dx + dy * dy;
    }
    // Calculate the t that minimizes the distance.
    float t = ((pt[0] - p1[0]) * dx + (pt[1] - p1[1]) * dy) /(dx * dx + dy * dy);
    // See if this represents one of the segment's, end points or a point in the middle.
    if (t < 0)
    {
      dx = pt[0] - p1[0];
      dy = pt[1] - p1[1];
    }
    else if (t > 1)
    {
      dx = pt[0] - p2[0];
      dy = pt[1] - p2[1];
    }
    else
    {
      dx = pt[0] - (p1[0] + t * dx);
      dy = pt[1] - (p1[1] + t * dy);
    }
    return dx * dx + dy * dy;
  }

  std::vector<Eigen::Vector3d> SafeSP(float scale)
  {
    std::vector<Eigen::Vector3d> safers;
    Eigen::Vector3d centre;
    centre.setZero();
    for(int i = 0 ; i < 3; ++i)
      centre += feetPoses[i];
    centre /= 3;
    for(int i = 0; i < 3; ++i){
      Eigen::Vector3d current;
      current = scale * feetPoses[i] + (1.0 - scale) * centre;
      safers.push_back(current);
    }
    return safers;
  }

  float CalcReward()
  {   
    float reward = 0;
    double task_err = 0;
    double end_err = 0;
    double root_err = 0;
    double support_err = 0;

    alien_->getState(gc_, gv_);
    if(isnan(gc_[0]))
    {
      return 0;
    }

    task_err = CalcTaskErr();
    end_err = CalcEndErr();
    root_err = CalcRootRotErr();
    support_err = CalcSupportErr(false);

    task_reward = exp(-task_scale * task_err);
    end_reward = exp(-end_scale * end_err);
    root_reward = exp(-root_scale * root_err);
    support_reward = exp(-support_scale * support_err);

    // reward = angle_w * angle_reward + end_w * end_reward;
    // reward = balance_w * end_reward * support_reward * root_reward + task_w * task_reward;
    reward = end_reward * support_reward * root_reward * task_reward;
    reward = end_reward * support_reward * task_reward;

    return reward;
  }

  float CalcTaskErr()
  {
    float diff = 0.0;
    update_target_coords();
    Eigen::Vector3d knee_err = ref_knee_b - alien_knee_b;
    Eigen::Vector3d foot_err = ref_foot_b - alien_foot_b;
    diff += knee_err.squaredNorm();
    diff += foot_err.squaredNorm() * 5;
    return diff;
  }

  float CalcEndErr()
  {
    float diff = 0.0;
    auto contacts = alien_->getContacts();

    for(int i = 0; i < end_names.size(); ++i)
    {
      Eigen::Vector3d projected_diff;
      projected_diff.setZero();
      raisim::Vec<3> hip_c;
      alien_->getFramePosition(alien_->getFrameByName(end_names[i].substr(0,2) + "_thigh_joint"), hip_c);
      projected_diff[0] = hip_c[0] - feetPoses[i][0];
      projected_diff[1] = hip_c[1] - feetPoses[i][1];
      diff += projected_diff.squaredNorm() * 0.03;
    }

    for(int i =0 ; i < desired_footIndices_.size() ; ++i)
    {
      int j;
      for(j =0 ; j < contacts.size(); ++j)
        if(desired_footIndices_[i] == contacts[j].getlocalBodyIndex())
          break;
      if(j == contacts.size()){
        diff += feetPoses[i][2] * feetPoses[i][2];
      }
    }
    return diff;
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
      angle = 0.0;
    else
      angle = acos(current.dot(xvec)/current.norm());
    rot_err = angle * angle;
    return rot_err;
  }

  float CalcSupportErr(bool safe)
  {
    Eigen::Vector3d com = alien_->getCompositeCOM().e();
    float min_d = 100000;

    if(safe){
      auto safeZone = SafeSP(0.9);
      if(isInside(safeZone)) return 0;
      for(int i = 0; i < end_names.size(); ++i)
      {
        int j;
        if(i==2) j = 0;
        else j = i+1;
        // float current = distToSegment(com, safeZone[i], safeZone[j]);
        float current = distToSegment(mCOP, safeZone[i], safeZone[j]);
        if(min_d > current) min_d = current;
      }
    }

    else{    
      if(isInside()) return 0;
      for(int i = 0; i < end_names.size(); ++i)
      {
        int j;
        if(i==2) j = 0;
        else j = i+1;
        // float current = distToSegment(com, feetPoses[i], feetPoses[j]);
        float current = distToSegment(mCOP, feetPoses[i], feetPoses[j]);
        if(min_d > current) min_d = current;
      }
    }
    return min_d * min_d;
  }


  void CenterOfPressure(){
    Eigen::Vector3d COP;
    COP.setZero();
    float tot_mag = 0.0;
    for (auto &contact: alien_->getContacts()) {
      for(int i = 0 ; i < desired_footIndices_.size(); ++i){
        if(desired_footIndices_[i] == contact.getlocalBodyIndex())
        {
          auto current = contact.getPosition();
          auto zaxis = *contact.getImpulse();
          float mag = zaxis[2];
          tot_mag += mag;
          COP[0] = current[0] * mag;
          COP[1] = current[1] * mag;
        }
      }
    }
    if(tot_mag == 0.0){
      Eigen::Vector3d com = alien_->getCompositeCOM().e();
      com[2] = 0;
      mCOP = com;
      return;
    }
    COP /= tot_mag;
    mCOP = COP;
  }

  void updateExtraInfo() final {
    extraInfo_["base height"] = gc_[2];
  }

  void updateObservation() {
    alien_->getState(gc_, gv_);
    obScaled_.setZero(obDim_);
    update_target_coords();

    /// update observations
    obScaled_.segment(0, 3) = ref_knee_b;
    obScaled_.segment(3, 3) = ref_foot_b;
    obScaled_.segment(6, 3) = alien_knee_b;
    obScaled_.segment(9, 3) = alien_foot_b;
    obScaled_.segment(12, 12) = gc_.tail(12);
    Eigen::Quaterniond quat(gc_[3], gc_[4], gc_[5], gc_[6]);
    Eigen::AngleAxisd aa(quat);
    Eigen::Vector3d rotation = aa.angle()*aa.axis();
    obScaled_.segment(24, 3) = rotation;
    Eigen::Vector3d com = alien_->getCompositeCOM().e();
    // obScaled_.segment(27, 2) = com.segment(0,2);
    obScaled_.segment(27, 2) = mCOP.segment(0,2);
    for(int i = 0 ; i < 3; ++i)
    {
      obScaled_.segment(29 + 2 * i, 2) = feetPoses[i].segment(0,2);
    }
    auto contacts = alien_->getContacts();
    for(int i = 0; i <desired_footIndices_.size(); ++i){
      int flag = 1;
      int j;
      for(j =0 ; j < contacts.size(); ++j)
        if(desired_footIndices_[i] == contacts[j].getlocalBodyIndex())
          break;
      if(j == contacts.size())
        flag = 0;

      obScaled_[35 + i] = flag;
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
          k4a_quaternion_t leftShoulderOrientation = body.skeleton.joints[K4ABT_JOINT_SHOULDER_LEFT].orientation;
          
          points[0] = scaler * Eigen::Vector3d((leftShoulderJoint.xyz.x - NeckJoint.xyz.x), (leftShoulderJoint.xyz.z - NeckJoint.xyz.z), (leftShoulderJoint.xyz.y - NeckJoint.xyz.y));
          points[1] = scaler * Eigen::Vector3d((leftElbowJoint.xyz.x - NeckJoint.xyz.x), (leftElbowJoint.xyz.z - NeckJoint.xyz.z), (leftElbowJoint.xyz.y - NeckJoint.xyz.y));
          points[2] = scaler * Eigen::Vector3d((leftWristJoint.xyz.x - NeckJoint.xyz.x), (leftWristJoint.xyz.z - NeckJoint.xyz.z), (leftWristJoint.xyz.y - NeckJoint.xyz.y));
          points[3] = scaler * Eigen::Vector3d((rightClavicleJoint.xyz.x - NeckJoint.xyz.x), (rightClavicleJoint.xyz.z - NeckJoint.xyz.z), (rightClavicleJoint.xyz.y - NeckJoint.xyz.y));
          points[4] = scaler * Eigen::Vector3d((NeckJoint.xyz.x - NeckJoint.xyz.x), (NeckJoint.xyz.z - NeckJoint.xyz.z), (NeckJoint.xyz.y - NeckJoint.xyz.y));
          points[5] = scaler * Eigen::Vector3d((leftClavicleJoint.xyz.x - NeckJoint.xyz.x), (leftClavicleJoint.xyz.z - NeckJoint.xyz.z), (leftClavicleJoint.xyz.y - NeckJoint.xyz.y));
          for(int i =0 ; i < points.size();++i){
            points[i][2] +=z_offset;
          }

          plane = EquationPlane(points[4], points[3], points[5]);
          normal_vec = plane.segment(0,3);
          Eigen::VectorXd shoulder_plane = ParallelPlane(normal_vec, points[0]);
          if(startFlag)
          {
            Eigen::Vector3d start_eb1 = Projection(shoulder_plane, points[1]);
            startElbowVecPlane = start_eb1 - points[0];
          }
          Eigen::VectorXd shoulder_perp_plane = EquationPlane(points[0] + normal_vec, points[0], startElbowVecPlane + points[0]);
          if(startFlag){
            Eigen::Vector3d start_eb2 = Projection(shoulder_perp_plane, points[1]);
            startElbowVecPerp = start_eb2 - points[0];
          }
          Eigen::Vector3d plane_projected = Projection(shoulder_plane, points[1]);
          Eigen::Vector3d perp_projected = Projection(shoulder_perp_plane, points[1]);

          double angleH = angle_finder(startElbowVecPlane + points[0], points[0], plane_projected);
          double angleT = angle_finder(startElbowVecPerp + points[0], points[0], perp_projected);
          double angleC = angle_finder(points[0], points[1], points[2]);
          double sideH = (points[1] - points[0]).dot(shoulder_perp_plane.segment(0,3));
          double sideT = (points[1] - points[0]).dot(normal_vec);

          if(startFlag)
          {
            base_angleC = angleC;
            if(sideT > 0)
              side_indicatorT = 1;
            else
              side_indicatorT = -1;
            startFlag = false;
          }
          if(sideH  < 0)
            desired_angleH = (angleH) * angle_smoother[0] + gc_init_[LEFT_HIP];
          else
            desired_angleH = -(angleH) * angle_smoother[0] + gc_init_[LEFT_HIP];
          if(sideT * side_indicatorT >= 0)
            desired_angleT = -(angleT) * angle_smoother[1] + gc_init_[LEFT_THIGH];
          else
            desired_angleT = angleT * angle_smoother[1] + gc_init_[LEFT_THIGH];
          desired_angleC = (angleC - base_angleC) * angle_smoother[2] + gc_init_[LEFT_CALF];
          // std::cout << desired_angleT << std::endl;
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
    if(A < 1e-5 || B < 1e-5)
      return 0;
    float angle = acos((C*C + A*A - B*B)/(2 * C * A));
    // if(angle > deg2rad(180))
    //   angle -= deg2rad(180);
    return angle;
  }

  Eigen::VectorXd EquationPlane(Eigen::Vector3d p, Eigen::Vector3d q, Eigen::Vector3d r){
    Eigen::VectorXd plane;
    plane.setZero(4);
    Eigen::Vector3d normal_vec;
    Eigen::Vector3d vec1 = p - r;
    Eigen::Vector3d vec2 = q - r;
    normal_vec = vec1.cross(vec2);
    plane.segment(0, 3) = normal_vec;
    double d = normal_vec.dot(-r);
    plane[3] = d;
    return plane;
  }
  Eigen::VectorXd ParallelPlane(Eigen::Vector3d normal_vec, Eigen::Vector3d p){
    Eigen::VectorXd plane;
    plane.setZero(4);
    plane.segment(0, 3) = normal_vec;
    double d = normal_vec.dot(-p);
    plane[3] = d;
    return plane;
  }


  Eigen::Vector3d Projection(Eigen::VectorXd plane, Eigen::Vector3d point){
    Eigen::Vector3d projected;
    double t0 = -(plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3]) / (plane[0]* plane[0] + plane[1]* plane[1] +plane[2] * plane[2]);
    projected[0] = point[0] + plane[0] * t0;
    projected[1] = point[1] + plane[1] * t0;
    projected[2] = point[2] + plane[2] * t0;
    return projected;
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obScaled_.cast<float>();
  }

  void update_random_leaning_angle()
  {
    unsigned seed_temp = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed_temp);
    desired_angleH = hip_distribution(generator);
    desired_angleT = thigh_distribution(generator);
    desired_angleC = calf_distribution(generator);
  }

  void update_designed_leaning_angle()
  {
    unsigned seed_temp = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed_temp);
    desired_angleH = designed_motion[curriculum_cnt][0];
    desired_angleT = designed_motion[curriculum_cnt][1];
    desired_angleC = designed_motion[curriculum_cnt][2];
    curriculum_cnt++;
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
  std::vector<int> desired_footIndices_;

  double desired_angleH, desired_angleT, desired_angleC;
  bool startFlag;
  double base_angleH, base_angleT, base_angleC;

  int change_count, current_count;
  int curriculum_cnt;

  double balance_w, task_w;
  double task_scale, end_scale, root_scale, support_scale;
  double task_reward, end_reward, root_reward, support_reward;

  Eigen::Vector3d ref_knee_b, ref_foot_b, alien_knee_b, alien_foot_b;

  float y_offset = 0.3;
  float z_offset = 0.85;
  k4a_device_t device = nullptr;
  k4abt_tracker_t tracker = nullptr;
  std::vector<Eigen::Vector3d> points;
  std::vector<int> connection;
  std::vector<Eigen::Vector3d> designed_motion;

  k4a_float3_t startOrigin;
  float scaler;
  std::vector<double> angle_smoother;
  double upper_limit = -0.65;
  double lower_limit = -2.78;
  std::vector<CoordinateFrame> body_frames;
  int side_indicatorH;
  int side_indicatorT;

  double left_angle_steady, right_angle_steady;
  std::vector<std::string> end_names;
  std::vector<Eigen::Vector3d> feet;
  std::vector<Eigen::Vector3d> feetPoses;
  Eigen::VectorXd plane;
  Eigen::Vector3d normal_vec;
  Eigen::Vector3d startElbowVecPlane;
  Eigen::Vector3d startElbowVecPerp;
  Eigen::Vector3d mCOP;

};

}