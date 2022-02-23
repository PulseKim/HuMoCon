#ifndef  _TRAJECTORY_GENERATOR_HPP_
#define  _TRAJECTORY_GENERATOR_HPP_

#include <stdio.h>
#include <iostream>
#include <Eigen/Core>
#include "EnvMath.hpp"
#include "math.h"

#define CYCLE (2 * M_PI)

enum LegOrder
{
  FRONT_RIGHT = 0,
  FRONT_LEFT,
  REAR_RIGHT,
  REAR_LEFT,
  NUM_LEGS
};

enum GaitType
{
  WALKING_TROT = 0,
  RUNNING_TROT,
  TRANSVERSE_GALLOP,
  BOUND,
  STOP,
  JUST_WALK,
  CRAB_WALK,
  NUM_GAIT_TYPES
};

struct GaitPhase
{
  float beta;
  std::vector<float> delta_phase;
  // (int(NUM_LEGS));
};

class GaitPhaseMode
{
public: 
  GaitPhaseMode(){}
  GaitPhaseMode(GaitType desired_mode){
    mMode = desired_mode;
    mPhase.delta_phase.resize(NUM_LEGS, 0.0);
    update_gait_phase();
  }

  void update_gait_phase()
  {
    float swing, stance;
    if(mMode == WALKING_TROT)
    {
      stance = 0.6;
      swing = 1. - stance;
      mPhase.beta = swing / stance;
      mPhase.delta_phase[FRONT_RIGHT] = 0.0 * CYCLE;
      mPhase.delta_phase[FRONT_LEFT] = 0.5 * CYCLE;
      mPhase.delta_phase[REAR_RIGHT] = 0.5 * CYCLE;
      mPhase.delta_phase[REAR_LEFT] = 0.0 * CYCLE;
    }
    else if(mMode == RUNNING_TROT)
    {
      stance = 0.4;
      swing = 1. - stance;
      mPhase.beta = swing / stance;
      mPhase.delta_phase[FRONT_RIGHT] = 0.0 * CYCLE;
      mPhase.delta_phase[FRONT_LEFT] = 0.5 * CYCLE;
      mPhase.delta_phase[REAR_RIGHT] = 0.5 * CYCLE;
      mPhase.delta_phase[REAR_LEFT] = 0.0 * CYCLE;
    }
    else if(mMode == TRANSVERSE_GALLOP)
    {
      // Have to change here
      stance = 0.3;
      swing = 1. - stance;
      mPhase.beta = swing / stance;
      mPhase.delta_phase[FRONT_RIGHT] = 0.0 * CYCLE;
      mPhase.delta_phase[FRONT_LEFT] = 0.2 * CYCLE;
      mPhase.delta_phase[REAR_RIGHT] = 0.5 * CYCLE;
      mPhase.delta_phase[REAR_LEFT] = 0.0 * CYCLE;
    }
    else if(mMode == BOUND)
    {
      stance = 0.5;
      swing = 1. - stance;      
      mPhase.delta_phase[FRONT_RIGHT] = 0.0 * CYCLE;
      mPhase.delta_phase[FRONT_LEFT] = 0.0 * CYCLE;
      mPhase.delta_phase[REAR_RIGHT] = 0.5 * CYCLE;
      mPhase.delta_phase[REAR_LEFT] = 0.5 * CYCLE;
    }
    else if(mMode == STOP)
    {
      stance = 1.0;
      swing = 1. - stance;      
      mPhase.delta_phase[FRONT_RIGHT] = 0.0 * CYCLE;
      mPhase.delta_phase[FRONT_LEFT] = 0.0 * CYCLE;
      mPhase.delta_phase[REAR_RIGHT] = 0.0 * CYCLE;
      mPhase.delta_phase[REAR_LEFT] = 0.0 * CYCLE;
    }
    else if(mMode == JUST_WALK)
    {
      stance = 0.75;
      swing = 1. - stance;      
      mPhase.delta_phase[FRONT_RIGHT] = 0.0 * CYCLE;
      mPhase.delta_phase[FRONT_LEFT] = 0.5 * CYCLE;
      mPhase.delta_phase[REAR_RIGHT] = 0.75 * CYCLE;
      mPhase.delta_phase[REAR_LEFT] = 0.25 * CYCLE;
    }
    else if(mMode == CRAB_WALK)
    {
      stance = 0.6;
      swing = 1. - stance;      
      mPhase.delta_phase[FRONT_RIGHT] = 0.0 * CYCLE;
      mPhase.delta_phase[FRONT_LEFT] = 0.5 * CYCLE;
      mPhase.delta_phase[REAR_RIGHT] = 0.0 * CYCLE;
      mPhase.delta_phase[REAR_LEFT] = 0.5 * CYCLE;
    }
    mPhase.beta = stance / 1.0;
  }

  GaitPhase phase_getter(){return mPhase;}
  std::vector<float> delta_phase_getter(){return mPhase.delta_phase;}
  float idx_delta_phase_getter(LegOrder leg){return mPhase.delta_phase[leg];}
  float idx_delta_phase_getter(int leg){return mPhase.delta_phase[leg];}
  float beta_getter(){return mPhase.beta;}

private:
  GaitType mMode;
  GaitPhase mPhase;
  float A_st, A_sw;


};

class TrajectoryGenerator
{
public:
  // For each parameters, implies....
  // Cs : Centre of swing DOF
  // alpha_tg : amplitude of swing --> stride
  // h_tg : centre of extension --> walking height
  // Ae : amplitude of extension --> ground clearance
  // theta: extension difference swing and stance ---> Theta depends on Cs, htg, alphatg

  TrajectoryGenerator(){}
  TrajectoryGenerator(GaitType gait_type_, float ltc_, float lcf_, float dt_ = 0.1)
  {
    // Initiate the gait parameters

    phi_prev = 0.0;
    phi = 0.0;
    t_delta = dt_;
    u_tg.setZero(12);
    Cs = 0.0;

    l_tc = ltc_;
    l_cf = lcf_;
    // Initialize leg phase
    mGait_type = gait_type_;
    mGaitPhaseMode = GaitPhaseMode(mGait_type);
    for(int i = 0; i < NUM_LEGS; ++i){
      mLegPhase.push_back(0.0 + mGaitPhaseMode.idx_delta_phase_getter(i));
      t_prime.push_back(0.0);
      swing_tg.push_back(Cs);
      extension_tg.push_back(0.1);
    }
  }
  
  void reset()
  {
    phi_prev = 0.0;
    phi = 0.0;
    u_tg.setZero(12);
    for(int i = 0; i < NUM_LEGS; ++i){
      mLegPhase[i] = 0.0 + mGaitPhaseMode.idx_delta_phase_getter(i);
      t_prime[i] = 0.0;
      swing_tg[i] = Cs;
      extension_tg[i] = 0.1;
    }
  } 
  
  void resetExtTG()
  {
    for(int i = 0; i < NUM_LEGS; ++i){
      extension_tg[i] = 0.1;
    }
  }

  void change_Cs(float cs)
  {
    Cs = cs;
  }
  void change_gait(GaitType gait_type_)
  {
    mGait_type = gait_type_;
    mGaitPhaseMode = GaitPhaseMode(mGait_type);
  }

  float get_phase(){return phi;}


  Eigen::VectorXd update_and_get_u(float f_, float alpha_, float h_)
  {
    get_tg_parameters(f_, alpha_, h_);
    update();
    return u_tg;
  }

  void update()
  {
    update_phase();
    calculate_current_swing();
    calculate_current_extension();
    conversion_to_utg();
  }

  void manual_timing_update(float phi_current)
  {
    update_phase_manually(phi_current);
    calculate_current_swing();
    calculate_current_extension();
    conversion_to_utg();
  }

  Eigen::VectorXd get_u()
  {
    return u_tg;
  }

  void get_tg_parameters(float f_, float alpha_, float h_)
  {
    f_tg = f_;
    alpha_tg = alpha_;
    h_tg = h_;
    // ex_mid = angle_finder(l_tc, h_tg, l_cf);
    // calculate_parameters();
  }

  void calculate_parameters()
  {
    Ae_swing = 0.20;
  }

  void set_Ae(float param)
  {
    Ae_swing = param;
  }


  void update_phase_manually(float phi_current)
  {
    phi = fmod(phi_current, CYCLE);
    
    for(int i = 0; i < NUM_LEGS; ++i){
      calculate_phase_leg(i);
      calculate_t_prime(i);
    }
  }

  void update_phase()
  {
    float phi_current = phi_prev + CYCLE * f_tg * t_delta;
    phi_prev = phi;
    phi = fmod(phi_current, CYCLE);
    
    for(int i = 0; i < NUM_LEGS; ++i){
      calculate_phase_leg(i);
      calculate_t_prime(i);
    }
  }

  void calculate_phase_leg(int leg)
  {
    double phase_leg = phi + mGaitPhaseMode.idx_delta_phase_getter(leg);
    mLegPhase[leg] = fmod(phase_leg, CYCLE);
  }

  void calculate_t_prime(int leg)
  {
    float phase_leg = mLegPhase[leg];
    float beta = mGaitPhaseMode.beta_getter();
    if(phase_leg < CYCLE * beta)
      t_prime[leg] = phase_leg / (2. * beta);
    else
      t_prime[leg] = M_PI + (phase_leg - CYCLE * beta) / (2 * (1. - beta));
  }

  void calculate_current_swing()
  {
    for(int i = 0; i < NUM_LEGS; ++i)
      swing_tg[i] = Cs - alpha_tg * cos(t_prime[i]);
  }

  void calculate_current_extension()
  {
    float beta = mGaitPhaseMode.beta_getter();
    for(int i = 0; i < NUM_LEGS; ++i){
      float phase_leg = mLegPhase[i];
      float swing_diff = swing_tg[i] - Cs;
      if(phase_leg < CYCLE * beta){
        extension_tg[i] = h_tg * (1./cos(swing_diff)) + h_tg * tan(swing_diff) * sin(Cs) / sin(M_PI/2 - swing_diff - Cs);
      }
      else{
        extension_tg[i] = h_tg * (1./cos(swing_diff)) + Ae_swing * sin(t_prime[i]) + h_tg * tan(swing_diff) * sin(Cs) / sin(M_PI/2 - swing_diff - Cs);
      }
    }
  }

  void conversion_to_utg()
  {
    // Have to change here too..........
    // 1) Change Extension to calf joint --> m to rad
    // 2) Have to calculate the compensation
    // 3) Assign value
      
    for(int i = 0 ; i <NUM_LEGS; ++i)
    {
      float theta_c = angle_finder(l_tc, extension_tg[i], l_cf);
      float calf_angle = theta_c - M_PI;
      float comp = angle_finder(l_tc, l_cf, extension_tg[i]);
      u_tg[3 * i + 0] = 0;
      u_tg[3 * i + 1] = swing_tg[i] + comp;
      u_tg[3 * i + 2] = calf_angle;
    }
  }

  const Eigen::VectorXd generated_pose(){return u_tg;}



private:
  GaitType mGait_type;
  GaitPhaseMode mGaitPhaseMode;
  std::vector<float> mLegPhase;
  float f_tg, alpha_tg, h_tg;
  // float ex_mid;
  float Cs, Ae_swing;
  float phi_prev, phi, t_delta;
  std::vector<float> t_prime;
  std::vector<float> swing_tg, extension_tg;
  Eigen::VectorXd u_tg;
  float l_tc, l_cf;
};

#endif