#ifndef _PD_CON_HPP_
#define	_PD_CON_HPP_

#include <stdlib.h>
#include <iostream>
#include <Eigen/Core>
#include "math.h"

class Controller{
public:
	//Constructor
	Controller()
	{
	}
	Controller(Eigen::VectorXd init_pose)
	{
		setTargetPosition(init_pose);
	}

	void jointControlSetter(double magKp, double magKd)
	{
		int nDofs = 12;
	    mForces.setZero(nDofs);
	    mKp = Eigen::MatrixXd::Identity(nDofs, nDofs);
	    mKd = Eigen::MatrixXd::Identity(nDofs, nDofs);
	    for(std::size_t i = 0; i < nDofs; ++i){
	        mKp(i,i) = magKp;
	        mKd(i,i) = magKd;
	    }
	}
	void setTargetPosition(const Eigen::VectorXd& pose)
	{
		mTargetPositions = pose;
	}
	void clearForces()
	{
		mForces.setZero();
	}
	void addPDForces(Eigen::VectorXd q, Eigen::VectorXd dq)
	{
	    Eigen::VectorXd p = -mKp * (q - mTargetPositions);
	    Eigen::VectorXd d = -mKd * dq;
	    mForces += p+d;
	    for(int i = 0 ; i < mForces.size();++i)
	    {
	    	mForces[i] = std::min(30. * 0.9, mForces[i]);
	    	mForces[i] = std::max(-30. * 0.9, mForces[i]);
	    }
	    std::cout << p.transpose() << std::endl;
	}
	void setFreeJointPosition();
	Eigen::VectorXd getForces(){return mForces;}

protected:
	Eigen::VectorXd mTargetPositions;
	Eigen::VectorXd mForces;
	Eigen::MatrixXd mKp;
	Eigen::MatrixXd mKd;
	double mTimestep;
};
#endif