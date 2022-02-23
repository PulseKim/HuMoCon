#ifndef SRC_GYMVECENV_HPP
#define SRC_GYMVECENV_HPP

#include <Eigen/Core>
#include <fstream>
#include "Control1.hpp"
#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include <k4arecord/playback.h>
#include <k4a/k4a.h>
#include <k4abt.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"

using Dtype=float;
using EigenRowMajorMat=Eigen::Matrix<Dtype, -1, -1, Eigen::RowMajor>;
using EigenVec=Eigen::Matrix<Dtype, -1, 1>;
using EigenBoolVec=Eigen::Matrix<bool, -1, 1>;

class VectorizedEnvironment {

 public:
  explicit VectorizedEnvironment(std::string resourceDir, std::string cfg){
  }

  ~VectorizedEnvironment() {
  }
  int getObDim(){return obDim_;}
  int getActionDim(){return 12;}

  void init()
  {
    // std::cout << "init " << std::endl;
    human0.setZero(32); human1.setZero(32); human2.setZero(32);
    obDim_ = 196;
    actionDim_ = 12;
    gc_init_.setZero(16);
    gc_init_ << 1.0, 0.0, 0.0, 0.0, 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99), 0.0, deg2rad(49), deg2rad(-99);
    pTarget12_.setZero(12);
    state_history.push_back(gc_init_);
    for(int i = 0; i < 3; ++i)
    {
      state_history.push_back(gc_init_);
      action_history.push_back(gc_init_.tail(12));
    }
    actionStd_.setZero(12);  actionMean_.setZero(12);
    float ah = 0.8028, at = 2.6179, ac = 0.8901;
    float mh = 0, mt = M_PI /2, mc = -1.806;
    actionStd_ << ah, at, ac, ah, at, ac, ah, at, ac, ah, at, ac;
    actionMean_ << mh, mt, mc, mh, mt, mc, mh, mt, mc, mh, mt, mc;
    startFlag = true;
    // Here comment it when you using a kinect.
    // Only use for the recorded motion
    readHumanfile("/home/sonic/Project/Warping_Test/temp_rsc/shorts/params/params_rs03.txt");
    // updateHumanPose();
  }

  // resets all environments and returns observation
  void reset(Eigen::Ref<EigenRowMajorMat>& ob) 
  {
    updateHumanPose();
    observe(ob.row(0));
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
    loop_control = new LoopFunc("control_loop", custom.dt,    boost::bind(&Custom::RobotControl_PD, &custom));
    loop_udpSend = new LoopFunc("udp_send",     custom.dt, 3, boost::bind(&Custom::UDPSend,      &custom));
    loop_udpRecv = new LoopFunc("udp_recv",     custom.dt, 3, boost::bind(&Custom::UDPRecv,      &custom));

    loop_udpSend->start();
    loop_udpRecv->start();
    loop_control->start();
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
    Eigen::VectorXd currentAction = action.row(0).cast<double>();
    pTarget12_ = currentAction;
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    custom.getControlSignal(pTarget12_);
    if(custom.getMvFlag()){
      action_history.erase(action_history.begin());
      action_history.push_back(pTarget12_);
    }
    unsigned int oneframe = 33000;
    for(int i = 0; i < 1; ++i)
    {
      usleep(oneframe);
    }    
    updateHumanPose();
    observe(ob.row(0));
  }

  void stepTest(Eigen::Ref<EigenRowMajorMat> &action)
  {    
    Eigen::VectorXd currentAction = action.row(0).cast<double>();
    pTarget12_ = currentAction;
    custom.getControlSignal(pTarget12_);
    unsigned int oneframe = 33000;
    for(int i = 0; i < 1; ++i)
    {
      usleep(oneframe);
    }    
    updateHumanPose();
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
  }

  void kinect_init()
  {

  }

  void readHumanfile(std::string file)
  {
    std::ifstream human_input(file);
    if(!human_input){
      std::cout << "Expected valid input file name" << std::endl;
    }

    int num_params = 32;
    while(!human_input.eof()){
      Eigen::VectorXd humanoid(num_params);
      for(int i = 0 ; i < num_params; ++i)
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

  void observe(Eigen::Ref<EigenVec> ob) 
  {
    ob = obScaled_.cast<float>();
  }

  void updateHumanPose()
  {
    human0 = human_param_list[human_cnt];
    if(startFlag) 
    {
      human1 = human0;
      human2 = human0;
      startFlag = false;
    }
    if(custom.getMvFlag())
      human_cnt++;
    if(human_cnt >= human_param_list.size()-1) human_cnt = 0;
    updateObservation();
  }

  void updateObservation()
  {
    // Here, time horizon and ... 
    obScaled_.setZero(obDim_);
    if(custom.getMvFlag()){
      state_history.erase(state_history.begin());
      state_history.push_back(custom.sendRobotState());
    }

    // Human pose triplets
    obScaled_.segment(0, 32) = human2;
    obScaled_.segment(32, 32) = human1;
    obScaled_.segment(64, 32) = human0;
    for(int i = 0; i < 4; ++i)
    {
      obScaled_.segment(96 + 16 * i, 16) = state_history[i];
    }
    for(int i = 0; i < 3; ++i)
    {
      obScaled_.segment(160 + 12 * i, 12) = action_history[i];
    }
    if(custom.getMvFlag()){
      human2 = human1;
      human1 = human0;
    }
  }


 private:
  bool startFlag;
  int num_envs_ = 1;
  int obDim_ = 0, actionDim_ = 0;
  int human_cnt = 0;
  std::string resourceDir_;
  // Yaml::Node cfg_;
  Eigen::VectorXd pTarget12_;
  Eigen::VectorXd robot_observation;
  Eigen::VectorXd human0, human1, human2;
  Custom custom = Custom(LOWLEVEL);
  std::vector<Eigen::VectorXd> action_history;
  std::vector<Eigen::VectorXd> state_history;
  Eigen::VectorXd actionMean_, actionStd_, obScaled_;
  Eigen::VectorXd gc_init_;
  std::vector<Eigen::VectorXd> human_param_list;
  LoopFunc* loop_control;
  LoopFunc* loop_udpSend;
  LoopFunc* loop_udpRecv;
};



#endif //SRC_RAISIMGYMVECENV_HPP
