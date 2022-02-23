#ifndef _MOTIONPLANNER_H
#define _MOTIONPLANNER_H

#include <iostream>
#include <Eigen/Core>
#include "raisim/World.hpp"
#include "math.h"


#define CALCERROR 1e-4

class MotionPlanner
{
public:
	MotionPlanner(std::vector<Eigen::VectorXd> Motions, double fpsD, Eigen::VectorXd fpsC) : 
		KeyMotions(Motions), desired_fps(fpsD), current_fps(fpsC)
	{
		// Assume desired_fps > current_fps since we get key motions from the environment
		interval = current_fps / desired_fps;
		currentKnots.resize(4);
		current_portion = 0.0;
		HermiterInterpolator();
		MotionClip.push_back(Motions[Motions.size()-1]);
		SpeedCalculator();
	}

	MotionPlanner(std::vector<Eigen::VectorXd> Motions, double fpsD, double fpsC) : 
		KeyMotions(Motions), desired_fps(fpsD)
	{
		// Assume desired_fps > current_fps since we get key motions from the environment
		current_fps.setZero(Motions.size()-1);
		current_fps.setConstant(fpsC);
		interval = current_fps / desired_fps;
		currentKnots.resize(4);
		current_portion = 0.0;
		HermiterInterpolator();
		MotionClip.push_back(Motions[Motions.size()-1]);
		SpeedCalculator();
	}


	void MotionBlender(){

	}

	void SpeedCalculator(){
		for(int i = 0; i < MotionClip.size() -1; ++i)
		{
			Eigen::VectorXd current = MotionClip[i];
			Eigen::VectorXd future = MotionClip[i+1];
			Eigen::VectorXd currentVel;
			currentVel.setZero(current.size()-1);
			for(int j =0 ; j < 3; ++j)
			{
				currentVel[j] = (future[j] - current[j]) * desired_fps;
			}
			Eigen::VectorXd quat1(4);
			Eigen::Vector3d axis1;
			quat1[0] = current[3]; quat1[1] = current[4]; quat1[2] = current[5]; quat1[3] = current[6];
			const double norm1 = (std::sqrt(quat1[1]*quat1[1] + quat1[2]*quat1[2] + quat1[3]*quat1[3]));
			if(fabs(norm1) < 1e-12) {
			  axis1[0] = 0;
			  axis1[1] = 0;
			  axis1[2] = 0;
			}
			else{
			  const double normInv = 1.0/norm1;
			  const double angleNomrInv = std::acos(std::min(quat1[0],1.0))*2.0*normInv;
			  axis1[0] = quat1[1] * angleNomrInv;
			  axis1[1] = quat1[2] * angleNomrInv;
			  axis1[2] = quat1[3] * angleNomrInv;
			}
			Eigen::VectorXd quat2(4);
			Eigen::Vector3d axis2;

			quat2[0] = future[3]; quat2[1] = future[4]; quat2[2] = future[5]; quat2[3] = future[6];
			const double norm2 = (std::sqrt(quat2[1]*quat2[1] + quat2[2]*quat2[2] + quat2[3]*quat2[3]));
			if(fabs(norm2) < 1e-12) {
			  axis2[0] = 0;
			  axis2[1] = 0;
			  axis2[2] = 0;
			}
			else{
			  const double normInv = 1.0/norm2;
			  const double angleNomrInv = std::acos(std::min(quat2[0],1.0))*2.0*normInv;
			  axis2[0] = quat2[1] * angleNomrInv;
			  axis2[1] = quat2[2] * angleNomrInv;
			  axis2[2] = quat2[3] * angleNomrInv;
			}

			for(int j = 3; j < 6; ++j)
			{
				currentVel[j] = (axis2[j-3] - axis1[j-3]) * desired_fps;
			}
			for(int j=6; j < currentVel.size(); ++j)
			{
				currentVel[j] = (future[j+1] - current[j+1]) * desired_fps;
			}
			VelocityClip.push_back(currentVel);
		}
		Eigen::VectorXd currentVel;
		currentVel.setZero(MotionClip[0].size()-1);
		VelocityClip.push_back(currentVel);
	}


	double BezierBasis(int idx, double t){
		if(idx == 0)
			return pow(1 - t, 3);
		else if(idx == 1)
			return 3 * pow((1-t), 2) * pow(t, 1);
		else if(idx == 2)
			return 3 * pow((1-t), 1) * pow(t, 2);
		else
			return pow(t, 3);
	}

	void HermiterInterpolator()
	{
		for(int i = 0; i < KeyMotions.size() -1; ++i)
		{
			InitHermiteInterpolator(i);
			while(true){
				MotionClip.push_back(GetCurrentM());
				current_portion += interval[i];
				if(current_portion >= i + 1 - CALCERROR){
					break;
				}
			}
		}
	}

	void InitHermiteInterpolator(int index)
	{
		Eigen::VectorXd v0;
		Eigen::VectorXd v1;
		
		v0 = (KeyMotions[index + 1] - KeyMotions[index]);
		v1 = (KeyMotions[index + 1] - KeyMotions[index]);
		
		currentKnots[0] = KeyMotions[index];
		currentKnots[1] = KeyMotions[index] + v0 / 3.0;
		currentKnots[2] = KeyMotions[index + 1] - v1 / 3.0;
		currentKnots[3] = KeyMotions[index + 1];
	}

	Eigen::VectorXd GetCurrentM()
	{
		Eigen::VectorXd current;
		current.setZero(KeyMotions[0].size());
		double portion = current_portion - (int)(current_portion + CALCERROR);

		// Have to Implement Quaternion Splining
		for(int i = 0; i < 4 ; ++i)
			current += currentKnots[i] * BezierBasis(i, portion);
		return current;
	}

	std::vector<Eigen::VectorXd> getResult(){return MotionClip;}
	std::vector<Eigen::VectorXd> getVelocity(){return VelocityClip;}


private:
	double desired_fps;
	Eigen::VectorXd current_fps;
	Eigen::VectorXd interval;

	double current_portion;
	
	std::vector<Eigen::VectorXd> MotionClip;
	std::vector<Eigen::VectorXd> VelocityClip;
	std::vector<Eigen::VectorXd> currentKnots;
	std::vector<Eigen::VectorXd> KeyMotions;
};


#endif