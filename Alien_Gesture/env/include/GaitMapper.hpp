#ifndef  _GAIT_MAPPER_HPP_
#define  _GAIT_MAPPER_HPP_

#include <stdlib.h>
#include <iostream>
#include <Eigen/Core>
#include "EnvMath.hpp"
#include "math.h"
#include <k4a/k4a.h>
#include <k4abt.h>

#define EMPTY_LEG 0
#define LEFT_LEG 1
#define RIGHT_LEG 2

// Have to implement initiate funciton


class GaitMapper
{
public: 
	GaitMapper(){
		reset();
	}

	void reset()
	{
		mInit = false;	mStop = true;
		onGait = false; halfGait = false;
 		current_wait = 0;
		speed_cnt = 0;
		duration_cnt = 0;
		current_foot = EMPTY_LEG;
		opposite_foot = EMPTY_LEG;
		safe_adder = 15;
		max_wait = 30;
		gaiter = 25.;

	}
	void setMax_wait(int freq){max_wait = freq;}
	void setSafeZone(int safer){safezone = 25;}
	void setSafeZone(){safezone = 30;}
	void setSafeZoneFromCurrent()
	{
		updateAngle();
		float min_angle = std::min(leftKneeAngle, rightKneeAngle);
		safezone = min_angle + safe_adder;
	}

	void getCurrentPose(k4abt_skeleton_t cur_skeleton)
	{
		skeleton = cur_skeleton;
		detectGait();
	}


	void detectGait()
	{
		// Start: when one of the foot escape from safe_zone
		// End: when opposite foot lands on safe_zone
		if(current_wait >= max_wait)
		{
			mStop = true;
			mInit = false;
			onGait = false; halfGait = false;
			duration_cnt = 0;
			current_foot = EMPTY_LEG;
			opposite_foot = EMPTY_LEG;
		}
		updateAngle();
		if(isEndGait())
		{
			endGait();
		}
		if(onGait)
		{
			duration_cnt++;
			isHalfGait();
		}		
		if(isStartGait())
		{
			startGait();
		}
		current_wait++;

	}

	float safeSpeed(float speed)
	{
		if(speed > max_speed) return max_speed;
		if(speed < min_speed) return 0.0;
		return speed;
	}

	void updateAngle()
	{
		// Knee joint angles in degree.
		k4a_float3_t footLeft = skeleton.joints[K4ABT_JOINT_ANKLE_LEFT].position;
    k4a_float3_t kneeLeft = skeleton.joints[K4ABT_JOINT_KNEE_LEFT].position;
    k4a_float3_t torzoLeft = skeleton.joints[K4ABT_JOINT_HIP_LEFT].position;

    k4a_float3_t footRight = skeleton.joints[K4ABT_JOINT_ANKLE_RIGHT].position;
    k4a_float3_t kneeRight = skeleton.joints[K4ABT_JOINT_KNEE_RIGHT].position;
    k4a_float3_t torzoRight = skeleton.joints[K4ABT_JOINT_HIP_RIGHT].position;

    leftKneeAngle = k4a_Angle(torzoLeft, kneeLeft, footLeft);
    rightKneeAngle = k4a_Angle(torzoRight, kneeRight, footRight);
    // std::cout << leftKneeAngle << " and  " << rightKneeAngle << std::endl;
	}

	bool isStartGait()
	{
		if(onGait) return false;
		if(leftKneeAngle > safezone)
		{
			current_foot = LEFT_LEG;
			opposite_foot = RIGHT_LEG;
			onGait = true;
			return true;
		}
		else if(rightKneeAngle > safezone)
		{
			current_foot = RIGHT_LEG;
			opposite_foot = LEFT_LEG;
			onGait = true;
			return true;
		}

		return false;
	}

	void isHalfGait()
	{
		if(!onGait) return;
		if(halfGait) return;

		if(current_foot == LEFT_LEG && leftKneeAngle < rightKneeAngle)
		{
			halfGait = true;
			return;
		}
		if(current_foot == RIGHT_LEG && leftKneeAngle > rightKneeAngle)
		{
			halfGait = true;
			return;
		}
	}
 
	bool isEndGait()
	{
		if(!onGait) return false;
		if(!halfGait) return false;
		if(current_foot == LEFT_LEG && rightKneeAngle <= safezone)
			return true;
		if(current_foot == RIGHT_LEG && leftKneeAngle <= safezone)
			return true;
		return false;
	}

	void startGait()
	{
		if(mStop) mInit = true;
		current_wait = 0;
	}

	void endGait()
	{
		float speed = gaiter/(duration_cnt + 8);
		// std::cout << "duration : " << duration_cnt << " speed " << speed << std::endl;
		mSpeed.push_back(safeSpeed(speed));
		mDuration.push_back(duration_cnt);
		onGait = false;
		halfGait = false;
		current_foot = EMPTY_LEG;
		opposite_foot = EMPTY_LEG;
		duration_cnt = 0;
		current_wait = 0;
	}


	float LaggedSpeed()
	{
		// mInit becomes true when mStop && a foot tries to pop out from the safezone
		if(mInit)
		{
			if(speed_cnt < max_wait)
			{
				speed_cnt++;
				mStop = false;
			}
			else
			{
				speed_cnt = 0;
				mInit = false;
			}
			return 0.0;
		}
		if(mStop){
 		  return 0.0;
		}
		if(speed_cnt <= mDuration[0])
		{
			speed_cnt++;
		}
		else
		{
			speed_cnt = 0;
			mDuration.pop_front();
			double curr = mSpeed[0];
			mSpeed.pop_front();
			return curr;
		}
		return mSpeed[0];
	}

	void noiseReduction()
	{

	}


private:
	int safezone, safe_adder;
	float leftKneeAngle, rightKneeAngle;
	int current_foot, opposite_foot;
	float max_speed = 1.7; float min_speed = 0.5;
	bool mStop, mInit, onGait, halfGait;
	float current_wait, max_wait;
	std::deque<double> mSpeed; 
	std::deque<int> mDuration;
	int speed_cnt;
	int duration_cnt;
	float gaiter;
	k4abt_skeleton_t skeleton;
};		

#endif
