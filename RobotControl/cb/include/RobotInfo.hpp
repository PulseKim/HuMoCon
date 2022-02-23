#ifndef _ROBOTINFO_H
#define _ROBOTINFO_H

#include <iostream>
#include <Eigen/Core>
#include "raisim/World.hpp"
#include "math.h"
#include <random>
#include "EnvMath.hpp"

#define RIGHT_HIP 7
#define RIGHT_THIGH 8
#define RIGHT_CALF 9
#define LEFT_HIP 10
#define LEFT_THIGH 11
#define LEFT_CALF 12


float AlienJointVelocityLimit = 26.5;
float A1JointVelocityLimit = 21.;

typedef enum
{
    TROT = 0,
    MANIPULATE,
    BALANCE,
    TILT,
    CRABLOCO,
    JUMP,
    SIT
} motionID;

typedef enum
{
    stateSTAND = 0,
    stateSIT,
    stateWALK
    
} balanceState;


typedef enum
{
    ROOT = 0,
    FR_HIP,
	FL_HIP,
	RR_HIP,
	RL_HIP,
	FLOATING,
	IMU,
	FR_THIGH,
	FR_CALF,
	FR_FOOT,
	FL_THIGH,
	FL_CALF,
	FL_FOOT,
	RR_THIGH,
	RR_CALF,
	RR_FOOT,
	RL_THIGH,
	RL_CALF,
	RL_FOOT
} frame2id;

typedef enum
{
    ROOTH = 0,
	jL5S1_rotx,
	jRightHip_rotx,
	jLeftHip_rotx,
	jL5S1_roty,
	jL4L3_rotx,
	jL4L3_roty,
	jL1T12_rotx,
	jL1T12_roty,
	jT9T8_rotx,
	jT9T8_roty,
	jT9T8_rotz,
	jT1C7_rotx,
	jRightC7Shoulder_rotx,
	jLeftC7Shoulder_rotx,
	jT1C7_roty,
	jT1C7_rotz,
	jC1Head_rotx,
	jC1Head_roty,
	jRightShoulder_rotx, // 19
	jRightShoulder_roty,
	jRightShoulder_rotz,
	jRightElbow_roty,
	jRightElbow_rotz,
	jRightWrist_rotx,
	jRightWrist_rotz,
	jLeftShoulder_rotx, // 26
	jLeftShoulder_roty,
	jLeftShoulder_rotz,
	jLeftElbow_roty,
	jLeftElbow_rotz,
	jLeftWrist_rotx,
	jLeftWrist_rotz,
	jRightHip_roty,
	jRightHip_rotz,
	jRightKnee_roty,
	jRightKnee_rotz,
	jRightAnkle_rotx,
	jRightAnkle_roty,
	jRightAnkle_rotz,
	jRightBallFoot_roty,
	jLeftHip_roty,
	jLeftHip_rotz,
	jLeftKnee_roty,
	jLeftKnee_rotz,
	jLeftAnkle_rotx,
	jLeftAnkle_roty,
	jLeftAnkle_rotz,
	jLeftBallFoot_roty
} frame2id_h;


typedef enum
{
	Pelvis = 0, 
	L5_f1,
	L5,
	L3_f1,
	L3,
	T12_f1,
	T12,
	T8_f1,
	T8_f2,
	T8,
	Neck_f1,
	Neck_f2,
	Neck,
	Head_f1,
	Head,
	RightShoulder,
	RightUpperArm_f1,
	RightUpperArm_f2,
	RightUpperArm,
	RightForeArm_f1,
	RightForeArm,
	RightHand_f1,
	RightHand,
	LeftShoulder,
	LeftUpperArm_f1,
	LeftUpperArm_f2,
	LeftUpperArm,
	LeftForeArm_f1,
	LeftForeArm,
	LeftHand_f1,
	LeftHand,
	RightUpperLeg_f1,
	RightUpperLeg_f2,
	RightUpperLeg,
	RightLowerLeg_f1,
	RightLowerLeg,
	RightFoot_f1,
	RightFoot_f2,
	RightFoot,
	RightToe,
	LeftUpperLeg_f1,
	LeftUpperLeg_f2,
	LeftUpperLeg,
	LeftLowerLeg_f1,
	LeftLowerLeg,
	LeftFoot_f1,
	LeftFoot_f2,
	LeftFoot,
	LeftToe
} body2id_h;


std::vector<std::string> allFrames()
{
	std::vector<std::string> frames;
	frames.push_back("ROOT");
	frames.push_back("FR_hip_joint");
	frames.push_back("FR_thigh_joint");
	frames.push_back("FR_calf_joint");
	frames.push_back("FR_foot_fixed");
	frames.push_back("FL_hip_joint");
	frames.push_back("FL_thigh_joint");
	frames.push_back("FL_calf_joint");
	frames.push_back("FL_foot_fixed");
	frames.push_back("RR_hip_joint");
	frames.push_back("RR_thigh_joint");
	frames.push_back("RR_calf_joint");
	frames.push_back("RR_foot_fixed");
	frames.push_back("RL_hip_joint");
	frames.push_back("RL_thigh_joint");
	frames.push_back("RL_calf_joint");
	frames.push_back("RL_foot_fixed");
	return frames;
}

std::vector<std::string> RootOutFrames()
{
	std::vector<std::string> frames;
	frames.push_back("FR_hip_joint");
	frames.push_back("FR_thigh_joint");
	frames.push_back("FR_calf_joint");
	frames.push_back("FR_foot_fixed");
	frames.push_back("FL_hip_joint");
	frames.push_back("FL_thigh_joint");
	frames.push_back("FL_calf_joint");
	frames.push_back("FL_foot_fixed");
	frames.push_back("RR_hip_joint");
	frames.push_back("RR_thigh_joint");
	frames.push_back("RR_calf_joint");
	frames.push_back("RR_foot_fixed");
	frames.push_back("RL_hip_joint");
	frames.push_back("RL_thigh_joint");
	frames.push_back("RL_calf_joint");
	frames.push_back("RL_foot_fixed");
	return frames;
}

std::vector<std::string> EndFrames()
{
	std::vector<std::string> frames;
	frames.push_back("FR_foot_fixed");
	frames.push_back("FL_foot_fixed");
	frames.push_back("RR_foot_fixed");
	frames.push_back("RL_foot_fixed");
	return frames;
}

void curriculum_distribution(int flag, 
	std::uniform_real_distribution<double>& hip_distribution,
	std::uniform_real_distribution<double>& thigh_distribution,
	std::uniform_real_distribution<double>& calf_distribution)
{
	if(flag == 1)
	{
		hip_distribution = std::uniform_real_distribution<double> (-0.4, 0.4);
    thigh_distribution = std::uniform_real_distribution<double> (deg2rad(-60), deg2rad(60));
    calf_distribution = std::uniform_real_distribution<double>(-1.9, -1.2);
	}
	else if(flag == 2)
	{
		hip_distribution = std::uniform_real_distribution<double> (-0.8, 0.8);
    thigh_distribution = std::uniform_real_distribution<double> (deg2rad(-100), deg2rad(100));
    calf_distribution = std::uniform_real_distribution<double>(-2.2, -1.0);
	}

	else if(flag == 3)
	{
		hip_distribution = std::uniform_real_distribution<double> (-1.22, 1.22);
    thigh_distribution = std::uniform_real_distribution<double> (deg2rad(-140), deg2rad(140));
    calf_distribution = std::uniform_real_distribution<double>(-2.5, -0.7);
	}
	else if(flag == 4)
	{
		hip_distribution = std::uniform_real_distribution<double> (-1.22, 1.22);
    thigh_distribution = std::uniform_real_distribution<double> (deg2rad(-180), deg2rad(180));
    calf_distribution = std::uniform_real_distribution<double>(-2.78, -0.65);
	}

	else{
		hip_distribution = std::uniform_real_distribution<double> (0.1, 0.1);
    thigh_distribution = std::uniform_real_distribution<double> (deg2rad(80), deg2rad(80));
    calf_distribution = std::uniform_real_distribution<double>(-1.9, -1.9);
	}

}

void curriculum_force_distribution(int flag, 
	std::uniform_real_distribution<double>& force)
{
	if(flag == 1)
	{
		force = std::uniform_real_distribution<double> (0.0, 15.0);
	}
	else if(flag == 2)
	{
		force = std::uniform_real_distribution<double> (15.0, 40.0);
	}

	else if(flag == 3)
	{
		force = std::uniform_real_distribution<double> (30.0, 80.0);
	}
	else if(flag == 4)
	{
		force = std::uniform_real_distribution<double> (30.0, 60.0);
	}
	else{
		force = std::uniform_real_distribution<double> (0.0, 1.0);
	}

}
void curriculum_speed_distribution(int flag, 
	std::uniform_real_distribution<double>& speed)
{
	float mid_vel = 1.0;
	float dev = 0.0;
	if(flag == 1)
	{
		dev = 0.1;
	}
	else if(flag == 2)
	{
		dev = 0.2;
	}

	else if(flag == 3)
	{
		dev = 0.4;
	}
	else{
		dev = 0.0;
	}
	speed = std::uniform_real_distribution<double> (mid_vel - dev, mid_vel + dev);
}

void curriculum_angle_distribution(int flag, 
	std::uniform_real_distribution<double>& angle)
{
	float mid_angle = 0.0;
	float dev = 0.0;
	if(flag == 1)
	{
		dev = deg2rad(30);
	}
	else if(flag == 2)
	{
		dev = deg2rad(50);
	}
	else if(flag == 3)
	{
		dev = deg2rad(70);
	}
	else if(flag == 4)
	{
		dev = deg2rad(90);
	}
	else{
		dev = deg2rad(0);
	}
	angle = std::uniform_real_distribution<double> (mid_angle - dev, mid_angle + dev);
}

void curriculum_angular_velocity_distribution(int flag, 
	std::uniform_real_distribution<double>& angle)
{
	float mid_angle = 0.0;
	float dev = 0.0;
	if(flag == 1)
	{
		dev = deg2rad(10);
	}
	else if(flag == 2)
	{
		dev = deg2rad(20);
	}
	else if(flag == 3)
	{
		dev = deg2rad(30);
	}
	else if(flag == 4)
	{
		dev = deg2rad(60);
	}
	else{
		dev = deg2rad(0);
	}
	angle = std::uniform_real_distribution<double> (mid_angle - dev, mid_angle + dev);
}

void curriculum_speed_distribution_mod(int flag, 
	std::uniform_real_distribution<double>& speed)
{
	float dev = 0.0;
	if(flag == 1)
	{
		// Running Trot is added
		speed = std::uniform_real_distribution<double> (0.5, 1.1);
	}
	else if(flag == 2)
	{
		// Running Trot is added
		speed = std::uniform_real_distribution<double> (0.0, 1.3);
	}
	else if(flag == 3)
	{
		// Transverse Gait is added
		speed = std::uniform_real_distribution<double> (0.0, 1.5);
	}

	else if(flag == 4)
	{
		// Bounding gait is added
		speed = std::uniform_real_distribution<double> (0.4, 1.7);
	}
	else{
		speed = std::uniform_real_distribution<double> (0.8, 1.1);
	}
}




#endif