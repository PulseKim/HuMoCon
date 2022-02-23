#ifndef _CONTROL_ONE_HPP
#define _CONTROL_ONE_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include "unitree_legged_sdk/unitree_joystick.h"

#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdint.h>

#define deg2rad(ang) ((ang) * M_PI / 180.0)
#define rad2deg(ang) ((ang) * 180.0 / M_PI)

using namespace UNITREE_LEGGED_SDK;
using namespace std;
class Custom
{
public:
  Custom(uint8_t level): safe(LeggedType::A1), udp(level) {
    udp.InitCmdData(cmd);
    pTarget.setZero(12);
    init_gc_.setZero(12);
    for(int i = 0; i < 4; ++i)
    {
      pTarget[3 * i+0] = deg2rad(0.);
      pTarget[3 * i+1] = deg2rad(49);
      pTarget[3 * i+2] = deg2rad(-99);
    }
    initialize_time = 5.0;
    initialize_count = int(initialize_time / dt);
    warmcount = int(1.0 / dt);
    std::cout << initialize_count << std::endl;
    mvFlag = false;
    compensate = Eigen::Quaterniond(1,0,0,0);
  }

  bool getMvFlag(){return mvFlag;}
  void UDPRecv(){udp.Recv();}
  void UDPSend(){udp.Send();}
  void getControlSignal(Eigen::VectorXd control)
  {   
    if(mvFlag)
      pTarget = control;
  }

  void setInitializeTime(double time)
  {
    initialize_time = time;
    initialize_count = int(initialize_time / dt);
  }

  Eigen::VectorXd sendRobotState()
  {
    Eigen::Quaterniond current(state.imu.quaternion[0], state.imu.quaternion[1], state.imu.quaternion[2], state.imu.quaternion[3]);
    Eigen::Quaterniond modified =Eigen::Quaterniond(compensate.inverse() * original_root.inverse() * current).normalized();
    Eigen::VectorXd pose(16);
    pose[0] = modified.w();
    pose[1] = modified.x();
    pose[2] = modified.y();
    pose[3] = modified.z();
    for(int i = 0; i < 4; ++i)
    {
      pose[4 + 3 * i] = state.motorState[0 + 3 * i].q;
      pose[5 + 3 * i] = state.motorState[1 + 3 * i].q;
      pose[6 + 3 * i] = state.motorState[2 + 3 * i].q;
    }    
    return pose;
  }

  void warmStart()
  {
    for(int i = 0; i < 12; ++i ){
      cmd.motorCmd[i].dq = 0.;
      cmd.motorCmd[i].Kp = 200;
      cmd.motorCmd[i].Kd = 10;
      cmd.motorCmd[i].tau = 0.f;
    }
    original_root = Eigen::Quaterniond(state.imu.quaternion[0], state.imu.quaternion[1], state.imu.quaternion[2], state.imu.quaternion[3]);
    if(motiontime >= 0 && motiontime < warmcount){
      for(int i = 0; i< 12; ++i)
        init_gc_[i] = state.motorState[i].q;
    }
    float percentile = std::min(1.0, std::max(double(motiontime - warmcount) * 2.0 / double(initialize_count), 0.0));
    for(int i = 0; i < 4; ++i){
      pTarget[3 * i+0] = percentile * deg2rad(0.) + (1.- percentile) * init_gc_[3 * i];
      pTarget[3 * i+1] = percentile * deg2rad(49)+ (1.- percentile) * init_gc_[3 * i+1];
      pTarget[3 * i+2] = percentile * deg2rad(-99)+ (1.- percentile) * init_gc_[3 * i+2];
    }
  }

  void updateCompensation()
  {
    Eigen::Quaterniond current(state.imu.quaternion[0], state.imu.quaternion[1], state.imu.quaternion[2], state.imu.quaternion[3]);
    compensate = Eigen::Quaterniond(compensate.inverse() * original_root.inverse() * current).inverse().normalized();
  }


  void manipulationTask()
  {
    for(int i = 0; i < 12; ++i ){
      cmd.motorCmd[i].dq = 0.;
      cmd.motorCmd[i].Kp = 150;
      cmd.motorCmd[i].Kd = 10;
      cmd.motorCmd[i].tau = 0.f;
    }
  }
  
  void tiltingTask()
  {
    if(mvFlag)
    {
      for(int i = 0; i < 12; ++i ){
        cmd.motorCmd[i].dq = 0.;
        cmd.motorCmd[i].Kp = 100;
        cmd.motorCmd[i].Kd = 2;
        cmd.motorCmd[i].tau = 0.f;
      }
    }
  }
  
  void RobotControl()
  {
    motiontime++;
    udp.GetRecv(state);

    if(!mvFlag)
      warmStart();
    if(motiontime > initialize_count)
      mvFlag = true;

    // manipulationTask();
    tiltingTask();
    // For debug

    if(motiontime > warmcount){
      for(int i = 0; i < 12; ++i)
      {
        cmd.motorCmd[FR_0 + i].q = pTarget[i];
      }
    }
    if(motiontime > 400){
      safe.PowerProtect(cmd, state, 9);
    }
    udp.SetSend(cmd);
  }



public:
  double initialize_time;
  int initialize_count;
  Safety safe;
  UDP udp;
  LowCmd cmd = {0};
  LowState state = {0};
  double time_consume = 0;
  int motiontime = 0;
  int warmcount = 0;
  float dt = 0.002;     // 0.001~0.01
  xRockerBtnDataStruct _keyData;
  bool mvFlag;
  Eigen::Quaterniond original_root;
  Eigen::Quaterniond compensate;
  Eigen::VectorXd pTarget;
  Eigen::VectorXd init_gc_;
};

#endif
