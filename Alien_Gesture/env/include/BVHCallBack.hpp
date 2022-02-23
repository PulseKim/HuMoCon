#ifndef  _BVHCALLBACK_HPP_
#define  _BVHCALLBACK_HPP_
#include "BVH.hpp"
#include <stdio.h>
#include <stdlib.h>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/quaternion.hpp"
#include "GL/glut.h"
#include "glm/ext.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
using namespace std;

class BVHCallBack
{
public:
	BVHCallBack();
	~BVHCallBack();
	BVHCallBack(const char* path)
	{
		// Initialize BVH
		mBVH = new BVH(path);
		mJointList = mBVH->getJointList();

  		// Key Joint Parsing
  		targetJoint.push_back("LeftShoulder");
  		targetJoint.push_back("RightShoulder");
  		targetJoint.push_back("LeftUpLeg");
  		targetJoint.push_back("RightUpLeg");
  		targetJoint.push_back("LeftHandEndSite");
  		targetJoint.push_back("RightHandEndSite");
  		targetJoint.push_back("LeftFootEndSite");
  		targetJoint.push_back("RightFootEndSite");

  		// Init parameters
		y_angle = mBVH->getFrameY(0);
  		Z_rot = Eigen::AngleAxisd(y_angle*M_PI/180, Eigen::Vector3d::UnitZ());
  		Z_rot = Z_rot.inverse();
  		Scaler = Eigen::Matrix3d::Identity();

  		//Initialize Frame
  		setBVHFrame(0);
  		auto initPoses = currentPoses();

  		z_offset = 100;
		for(int i =0 ; i <initPoses.size();++i){
		  if(z_offset > initPoses[i][2])
		  	z_offset = initPoses[i][2];
		}
		rootRef = FK_raw("Hips");
		rootRef[2] = 0;
	}

	void setScaler()
	{
		float x_factor, y_factor, z_factor;
		float bvh_x, bvh_y, bvh_z;

  		setBVHFrame(0);
		bvh_x = abs(currentPose("LeftShoulder")[0] - currentPose("LeftUpLeg")[0]);
		bvh_y = abs(currentPose("RightShoulder")[1] - currentPose("LeftShoulder")[1]);
		bvh_z = abs(FK_raw("Hips")[2] - z_offset);

		// std::cout << bvh_x << bvh_y << bvh_z << std::endl;

		x_factor = robot_x / bvh_x;
		y_factor = robot_y / bvh_y;
		z_factor = robot_z / bvh_z;

		Scaler(0,0) *= x_factor;
		Scaler(1,1) *= y_factor;
		Scaler(2,2) *= z_factor;
		std::cout << "Scaler" << Scaler << std::endl;
	}

	void getXYZ(float x, float y, float z)
	{
		robot_x = x;
		robot_y = y;
		robot_z = z;
		robot_z = 0.47;
	}


	void setBVHFrame(int frame)
	{
		mBVH->matrixMoveTo(frame, 0.01);
	}

	Eigen::Vector3d FK_raw(string joint)
	{
		Eigen::Vector3d pose;
		glm::vec4 origin(0,0,0,1);
		auto current = mBVH->getJoint(joint);
		glm::vec4 result = current->matrix * origin;

		pose[0] = -result[0];
		pose[1] = result[2];
		pose[2] = result[1];

		return pose;
	}

	std::vector<Eigen::Vector3d> currentPoses()
	{
		Eigen::Vector3d rootRef = FK_raw("Hips");
		rootRef[2] = 0;

		std::vector<Eigen::Vector3d> jointPoses;

		for(int i =0; i < targetJoint.size(); ++i)
		{
			Eigen::Vector3d curJoint = FK_raw(targetJoint[i]);
			curJoint = curJoint - rootRef;
			curJoint = Z_rot * curJoint;
			curJoint = Scaler * curJoint;
			curJoint[2] -= z_offset;
			if(curJoint[2] < 0) curJoint[2] = 0;
			jointPoses.push_back(curJoint);
		}
		return jointPoses;
	}

	Eigen::Vector3d currentPose(std::string frame)
	{	
		Eigen::Vector3d curJoint = FK_raw(frame);
		curJoint = curJoint - rootRef;
		curJoint = Z_rot * curJoint;
		curJoint = Scaler * curJoint;
		curJoint[2] -= z_offset;
		if(curJoint[2] < 0) curJoint[2] = 0;
		return curJoint;
	}

	std::vector<std::string> getJointList()
	{
		return targetJoint;
	}

	int getNumFrames(){return mBVH->getNumFrames();}

private:
	BVH* mBVH;
	std::vector<JOINT*> mJointList;
	std::vector<string> targetJoint;
	float y_angle;
	float z_offset = 0;
	Eigen::Matrix3d Z_rot;
	Eigen::Vector3d rootRef;
	Eigen::Matrix3d Scaler;

	float robot_x, robot_y, robot_z;

};
	

#endif