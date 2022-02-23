#ifndef _CONTROLLER_HPP_
#define	_CONTROLLER_HPP_

#include <stdlib.h>
#include <iostream>
#include <Eigen/Core>
#include "raisim/World.hpp"
#include "math.h"
// #include "EnvMath.hpp"
#include "RobotInfo.hpp"

class Controller{
public:
	//Constructor
	Controller(raisim::ArticulatedSystem* robot, double timeStep)
	: mRobot(robot), mTimestep(timeStep)
	{
	}

	void jointControlSetter(double magKp, double magKd)
	{
		int nDofs = mRobot->getDOF();

	    mForces = Eigen::VectorXd::Zero(nDofs);
	    mKp = Eigen::MatrixXd::Identity(nDofs, nDofs);
	    mKd = Eigen::MatrixXd::Identity(nDofs, nDofs);

	    for(std::size_t i = 0; i < 6; ++i){
	        mKp(i,i) = 0;
	        mKd(i,i) = 0;
	    }
	    for(std::size_t i = 6; i < nDofs; ++i){
	        mKp(i,i) = magKp;
	        mKd(i,i) = magKd;
	    }
	    setTargetPosition(mRobot->getGeneralizedCoordinate().e());
	}
	void setTargetPosition(const Eigen::VectorXd& pose)
	{
		mTargetPositions = get_current_q(pose);
	}
	void clearForces()
	{
		mForces.setZero();
	}
	void addSPDForces()
	{
		Eigen::VectorXd q = get_current_q(mRobot->getGeneralizedCoordinate().e());
	    Eigen::VectorXd dq = mRobot->getGeneralizedVelocity().e();
	    Eigen::MatrixXd invM = (mRobot->getMassMatrix().e()  + mKd * mTimestep).inverse();
	    Eigen::VectorXd p = -mKp * (q + dq * mTimestep - mTargetPositions);
	    Eigen::VectorXd d = -mKd * dq;
	    Eigen::VectorXd qddot =  invM * (- mRobot->getNonlinearities().e() + p + d);
	    
	    mForces = p + d - mTimestep * mKd * qddot ;
	}
	void setFreeJointPosition();
	Eigen::VectorXd getForces(){return mForces;}
	Eigen::VectorXd get_current_q(Eigen::VectorXd gc_){
	    Eigen::VectorXd pose;
	    pose.setZero(mRobot->getDOF());
	    for(int i = 0 ; i < 3; i++)
	    {
	      pose[i] = gc_[i];
	    }
	    Eigen::Quaterniond quat(gc_[3], gc_[4], gc_[5], gc_[6]);
	    Eigen::AngleAxisd aa(quat);
	    Eigen::Vector3d rotation = aa.angle()*aa.axis();

	    for(int i = 0 ; i < 3; i++)
	    {
	      pose[i+3] = rotation[i];
	    }
	    for(int i = 7 ; i < gc_.size(); i++)
	    {
	      pose[i-1] = gc_[i];
	    }
	    return pose;
	  }

protected:
	raisim::ArticulatedSystem* mRobot;
	Eigen::VectorXd mTargetPositions;
	Eigen::VectorXd mForces;
	Eigen::MatrixXd mKp;
	Eigen::MatrixXd mKd;
	double mTimestep;
};
#endif