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
#include "IKIPopt.hpp"
#include "MotionPlanner.hpp"
#include "RobotInfo.hpp" 
#include <iomanip>
// #include "MappingFunctions.hpp"

#include "../visualizer_alien/raisimKeyboardCallback.hpp"
#include "../visualizer_alien/helper.hpp"
#include "../visualizer_alien/guiState.hpp"
#include "../visualizer_alien/raisimBasicImguiPanel.hpp"

#define deg2rad(ang) ((ang) * M_PI / 180.0)
#define rad2deg(ang) ((ang) * 180.0 / M_PI)

#define LEFT_HIP 10
#define LEFT_THIGH 11
#define LEFT_CALF 12

std::uniform_real_distribution<double> hip_distribution(-1.22, 1.22);
std::uniform_real_distribution<double> thigh_distribution(deg2rad(-180), deg2rad(180));
std::uniform_real_distribution<double> calf_distribution(-2.78, -0.65);

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const YAML::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) 
  {
    /// add objects
    alien_ = world_->addArticulatedSystem(resourceDir_+"/urdf/aliengo.urdf");
    alien_->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);
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
    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.360573, 1.0, 0.0, 0.0, 0.0, 0.000356295, 0.765076, -1.5303, 5.69796e-05, 0.765017, -1.53019, 0.000385732, 0.765574, -1.53102, 8.34913e-06, 0.765522, -1.53092;
    // gc_init_ << 0, 0, 0.43, 1.0, 0.0, 0.0, 0.0, 0.045377, 0.667921, -1.23225,0.045377, 0.667921, -1.23225, 0.045377, 0.667921, -1.23225, 0.045377, 0.667921, -1.23225;
    alien_->setState(gc_init_, gv_init_);
    actionDim_ = nJoints_;
    /// set pd gains
    alien_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
    generate_data_set("train_set", 500000);
    generate_data_set("test_set", 100000);
    // des angleR, current angleR, des angleL, current angleL ,N joints 
    obDim_ = 1;
  }

  ~ENVIRONMENT() final = default;

  void init() final { }

  void reset() final
  {
    alien_->setState(gc_init_, gv_init_);
    desired_angleH = gc_init_[LEFT_HIP];
    desired_angleT = gc_init_[LEFT_THIGH];
    desired_angleC = gc_init_[LEFT_CALF];
    updateObservation();
  }

  void generate_data_set(std::string file_name, int data_size)
  {
    std::string filePath = resourceDir_+ "/../rsc/"+ file_name +".txt";
    std::cout << filePath << std::endl;
    // write File
    std::ofstream writeFile;
    writeFile.open(filePath);

    // Write max_length, max theta
    raisim::Vec<3> tempPositionh;
    alien_->getFramePosition(alien_->getFrameByName("FL_hip_joint"), tempPositionh);
    raisim::Vec<3> tempPositionc;
    alien_->getFramePosition(alien_->getFrameByName("FL_calf_joint"), tempPositionc);
    raisim::Vec<3> tempPositionf;
    alien_->getFramePosition(alien_->getFrameByName("FL_foot_fixed"), tempPositionf);
    float robot_arm = (tempPositionf.e() - tempPositionc.e()).norm() + (tempPositionc.e() - tempPositionh.e()).norm();
    float max_h = 1.22; float min_h = -1.22; 
    float max_t = deg2rad(180); float min_t = -deg2rad(180); 
    float max_c = -0.65; float min_c = -2.78;
    writeFile << std::setprecision(3) << robot_arm << " "<< max_h << " " << min_h << " " << max_t 
      << " " << min_t << " " << max_c << " " << min_c << std::endl;
    // Generate data
    for(int i =0; i < data_size; ++i)
    {
      update_random_leaning_angle();
      Eigen::VectorXd pose_new = gc_init_;
      pose_new[LEFT_HIP] = desired_angleH;
      pose_new[LEFT_THIGH] = desired_angleT;
      pose_new[LEFT_CALF] = desired_angleC;
      alien_->setGeneralizedCoordinate(pose_new);

      raisim::Vec<3> tempPositionh;
      alien_->getFramePosition(alien_->getFrameByName("FL_hip_joint"), tempPositionh);
      raisim::Vec<3> tempPositionf;
      alien_->getFramePosition(alien_->getFrameByName("FL_foot_fixed"), tempPositionf);
      Eigen::Vector3d relative = tempPositionf.e() - tempPositionh.e();
      writeFile << std::setprecision(3) << std::fixed << desired_angleH << " " << desired_angleT 
        << " " << desired_angleC << " " << relative[0] << " " << relative[1] << " " << relative[2] << std::endl;
    }
    writeFile.close();
  }



  float stepRef() final
  {
    return 0;
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    return -1E10;
  }

  float stepTest(const Eigen::Ref<EigenVec>& action) final {
    return -1E10;
  }


  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obScaled_.cast<float>();
  }


  void updateExtraInfo() final {
    // extraInfo_["base height"] = gc_[2];
  }

  void updateObservation() {
    alien_->getState(gc_, gv_);
    obScaled_.setZero(obDim_);
    /// update observations
    obScaled_[0] = 1;
  }

  void update_random_leaning_angle()
  {
    unsigned seed_temp = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed_temp);
    desired_angleH = hip_distribution(generator);
    desired_angleT = thigh_distribution(generator);
    desired_angleC = calf_distribution(generator);
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

  void get_current_torque(Eigen::Ref<EigenVec>& tau) {
    Eigen::VectorXd torque = alien_->getGeneralizedForce().e().tail(nJoints_);
    for(int i = 0; i < nJoints_; ++i)
      tau[i] = torque[i];
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
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, torque_,torque12_, gc_dummy_init;
  Eigen::VectorXd gc_prev1, gc_prev2;
  double desired_fps_ = 60.;
  int visualizationCounter_=0;
  float effort_limit = 44.4;
  Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_;
  Eigen::VectorXd obDouble_, obScaled_;
  std::set<size_t> footIndices_;
  std::vector<int> desired_footIndices_;

  double desired_angleH, desired_angleT, desired_angleC;
};

}