#ifndef  _SAMPLE_GENERATOR_HPP_
#define  _SAMPLE_GENERATOR_HPP_
	
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <Eigen/Core>
#include <fstream>
#include <random>
#include "EnvMath.hpp"
#include "raisim/World.hpp"
#include "math.h"
#include "RobotInfo.hpp"

class SafetyFunction
{
public:
	SafetyFunction(){}
	SafetyFunction(std::string robot_urdf)
	{
		nJoints = 12;
		beginLeg = 7;
		mWorld = std::make_unique<raisim::World>();
    mWorld->setTimeStep(0.00001);
    std::cout <<robot_urdf << std::endl;
    mRobot = mWorld->addArticulatedSystem(robot_urdf);
    int gcDim_ = mRobot->getGeneralizedCoordinateDim();
    int gvDim_ = mRobot->getDOF();
    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
		gc_init_ <<  0, 0, 0.45, 1.0, 0.0, 0.0, 0.0, 0.0, 0.765076, -1.5303, 0.0, 0.765017, -1.53019, 0.0, 0.765574, -1.53102, 0.0, 0.765522, -1.53092;
		gc_ = gc_init_;
		mPose = gc_init_;
		mRobot->setState(gc_init_, gv_init_);
		joint_violate = 0; selfcol_violate = 0; 
		ground_form_violate = 0; ground_col_violate = 0;
		mLine = mWorld->addCylinder(0.0001, 0.1, 0.1);
		mPlane = mWorld->addBox(3.0, 3.0, 0.0001, 0.1);
		mLine->setPosition(100, 100, 100);
		mPlane->setPosition(-100, -100, -100);
		slope_max = deg2rad(30);
	}

	void setCurrentState2D(Eigen::VectorXd pose_)
	{	
		mPose = gc_init_;
		mPose[beginLeg] = pose_[beginLeg];
		mPose[beginLeg+2] = pose_[beginLeg+2];
		mRobot->setGeneralizedCoordinate(mPose);
	}
	
	void setCurrentState(Eigen::VectorXd pose_, std::vector<int> ground_flag_)
	{	
		mGroundFlag = ground_flag_;
		mPose.tail(16) = pose_.tail(16);
		// Here set states via body rotation, joint angles

		mRobot->setGeneralizedCoordinate(mPose);
	}

	void resetState()
	{
		mRobot->setState(gc_init_, gv_init_);
		clearState();
	}

	void clearState()
	{
		mGroundFlag.shrink_to_fit();
		mPose.setZero();
		nContacts = 0;
	}


	int jointLimitValueFunction()
	{
		mRobot->getState(gc_, gv_);
		if(joint_limit_violation()){joint_violate++; return 0;}
		return 1;
	}
	int selfCollisionValueFunction()
	{
		mRobot->getState(gc_, gv_);
		if(self_collision_violation()){selfcol_violate++; return 0;}
		return 1;
	}

	int groundValueFunction()
	{
		mRobot->getState(gc_, gv_);
		if(ground_formation_violation()){ground_form_violate++; return 0;}
		mWorld->integrate();
		if(ground_collision_violation()){ground_col_violate++; return 0;}
		return 1;
	}


	int staticValueFunction()
	{
		mRobot->getState(gc_, gv_);
		if(joint_limit_violation()){joint_violate++; return 0;}
		if(ground_formation_violation()){ground_form_violate++; return 0;}
		mWorld->integrate();
		if(self_collision_violation()){selfcol_violate++; return 0;}
		if(ground_collision_violation()){ground_col_violate++; return 0;}
		return 1;
	}
	int staticValueFunctionTest()
	{
		mRobot->getState(gc_, gv_);
		if(joint_limit_violation()){return 0;}
		if(ground_formation_violation()){return 0;}
		mWorld->integrate();
		if(self_collision_violation()){return 0;}
		if(ground_collision_violation()){return 0;}
		return 1;
	}

	//  Violate or Not functions
	bool joint_limit_violation()
	{
		auto jointLimits = mRobot->getJointLimits();
		for(int i = beginLeg-1;i<jointLimits.size();i++)
		{
			if(jointLimits[i][0] == 0 && jointLimits[i][1] == 0){ 
				jointLimits[i][0] = deg2rad(-180);
				jointLimits[i][1] = deg2rad(180);
			}
		}
		for(int i = beginLeg; i < nJoints; ++i)
		{
			// std::cout << gc_[i] << " and " << jointLimits[i][0] << " , " << jointLimits[i][1] << std::endl;
			if(gc_[i] < jointLimits[i-1][0]) return true;
			else if(gc_[i] > jointLimits[i-1][1]) return true;
		}
		return false;
	}


	bool self_collision_violation()
	{
		for(auto& contact: mRobot->getContacts()) {
      if(contact.isSelfCollision()) return true;
    }
		return false;
	}

	// Ground can be 
	bool ground_formation_violation()
	{
		// Only Formation
		nContacts = 0;
		for(int i = 0; i < mGroundFlag.size();++i)
		{
			if(mGroundFlag[i]) nContacts++;
		}
		// 0 - contact : Assume it is jump case
		if(nContacts == 0) return false;
		// 1 - contact : Assume it is landing case ==> Check body config, rest legs 
		else if(nContacts == 1) return false;
		// 2 - contact : Line formation ==> On Gait motion
		else if(nContacts == 2)
		{
			Eigen::Vector3d centre = Eigen::Vector3d::Zero();
			Eigen::Vector3d dir = Eigen::Vector3d::Zero();
			int cnt = 1;
			for(int i = 0; i < mGroundFlag.size();++i)
			{
				if(mGroundFlag[i]){
					raisim::Vec<3> pos_tempr;
    			mRobot->getFramePosition(FR_FOOT + 3*i, pos_tempr);
    			centre = centre + pos_tempr.e();
    			dir = dir + cnt * pos_tempr.e();
    			cnt = -1;
				}
			}
			centre = centre / 2;
			double height = dir.norm();
			// Make line by body roation + line by contact points
			raisim::Mat<3,3> rot;
      Eigen::Vector3d way = dir.normalized();
      raisim::Vec<3> direction = {way[0], way[1], way[2]};
      raisim::zaxisToRotMat(direction, rot);
      Eigen::Matrix3d rot_e = rot.e();
      mLine = new raisim::Cylinder(0.0001, height, 0.1);
	    mLine->setPosition(centre);
	    mLine->setOrientation(rot_e);
		}
		else if(nContacts == 3)
		{
			std::vector<Eigen::Vector3d> feet;
			for(int i = 0; i < mGroundFlag.size();++i)
			{
				if(mGroundFlag[i]){
					raisim::Vec<3> pos_temp;
    			mRobot->getFramePosition(FR_FOOT + 3*i, pos_temp);
    			feet.push_back(pos_temp.e());
				}
			}
			createPlane(feet);
			return false;
		}
		else if(nContacts == 4)
		{
			// Check if the four points can make a plane
			std::vector<Eigen::Vector3d> feet;
			for(int i = 0; i < mGroundFlag.size();++i)
			{
				raisim::Vec<3> pos_temp;
  			mRobot->getFramePosition(FR_FOOT + 3*i, pos_temp);
  			feet.push_back(pos_temp.e());
			}
			Eigen::VectorXd plane = createPlane(feet);
			Eigen::Vector3d normal = plane.segment(0,3);
			float admittance = 0.01 / 2.;
			double indicator = normal.dot(feet[3]) + plane[3];
			if(abs(indicator) > admittance) 
			{
				return true;
			}
		}
		return false;
	}

	Eigen::VectorXd createPlane(std::vector<Eigen::Vector3d> feet)
	{
		Eigen::VectorXd plane =EquationPlane(feet[0], feet[1], feet[2]);
		raisim::Mat<3,3> rot;
    raisim::Vec<3> direction = {plane[0], plane[1], plane[2]};
    raisim::zaxisToRotMat(direction, rot);
    Eigen::Matrix3d rot_e = rot.e();
    mPlane->setPosition(feet[0]);
    mPlane->clearPerObjectContact();
    mPlane->setOrientation(rot_e);
    return plane;
	}

	bool ground_collision_violation()
	{
		if(nContacts == 0) return false;
		else if(nContacts == 1)
		{
			return false;
		}
		else if(nContacts == 2)
		{
			// Check whether the line crossbody plane
			for(auto& contact: mRobot->getContacts()) {
	      if(mWorld->getObject(contact.getPairObjectIndex()) == mLine){
	      	return true;
	      } 
	    }
			// Check whether the line is above the body.

			// Destroy mLine
			mLine->setPosition(100, 100, 100);
			// mWorld->removeObject(mLine);
		}
		else if(nContacts > 2)
		{
			// Check whether the plane collides with body
			for(auto& contact: mRobot->getContacts()) {
	      if(mWorld->getObject(contact.getPairObjectIndex()) == mPlane){
	      	return true;
	      } 
	    }
			// Check whether the plane is above the body.
	    mPlane->setPosition(-100, -100, -100);
	    // mPlane->clearPerObjectContact();
			// mWorld->removeObject(mPlane);
		}
		return false;		
	}

	bool slope_angle_violation()
	{
		// x-z projection

		// y-z projection

	}



public:
	int joint_violate, selfcol_violate, ground_form_violate, ground_col_violate;


private:
	std::unique_ptr<raisim::World> mWorld;
	raisim::ArticulatedSystem* mRobot;
	int nJoints, beginLeg;
	Eigen::VectorXd gc_, gv_;
	Eigen::VectorXd gc_init_, gv_init_;
	std::vector<int> mGroundFlag;
	int nContacts;
	double slope_max;
	Eigen::VectorXd mPose;
  raisim::Cylinder* mLine;
  raisim::Box* mPlane;
};


class SampleGenerator
{
public: 
	SampleGenerator(){}
	SampleGenerator(bool static_, std::string robot_urdf){
	mStatic = static_;
	mSF = SafetyFunction(robot_urdf);
	pitch_distribution = std::uniform_real_distribution<double> (deg2rad(-180), deg2rad(180));
	roll_distribution = std::uniform_real_distribution<double> (deg2rad(-180), deg2rad(180));
	hip_distribution = std::uniform_real_distribution<double> (deg2rad(-180), deg2rad(180));
  thigh_distribution = std::uniform_real_distribution<double> (deg2rad(-180), deg2rad(180));
  calf_distribution = std::uniform_real_distribution<double>(deg2rad(-180), deg2rad(180));
  flag_distribution = std::uniform_int_distribution<int>(0,1);
  mPosition.setZero(19);
  mPosition[4] = 1;
  start = 7;
  mGroundFlag.assign(4, 0);
	}

	void enableJointLimit()
	{
		hip_distribution = std::uniform_real_distribution<double> (-1.21, 1.21);
    thigh_distribution = std::uniform_real_distribution<double> (deg2rad(-179), deg2rad(179));
    calf_distribution = std::uniform_real_distribution<double>(-2.77, -0.66);
	}

	void disableJointLimit()
	{
		hip_distribution = std::uniform_real_distribution<double> (deg2rad(-180), deg2rad(180));
    thigh_distribution = std::uniform_real_distribution<double> (deg2rad(-180), deg2rad(180));
    calf_distribution = std::uniform_real_distribution<double>(deg2rad(-180), deg2rad(180));
	}

	void successPoseGenerator()
	{

	}

	void randomPoseGenerator()
	{
		unsigned seed_temp;
		float hip_angle, thigh_angle, calf_angle;
		int flag;
		seed_temp = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed_temp);
    mPitch = pitch_distribution(generator);
    mRoll = roll_distribution(generator);
		setPitchRollToQuat(mPitch, mRoll);
		for(int i = 0; i < 4; ++i){
			seed_temp = std::chrono::system_clock::now().time_since_epoch().count();
	    generator = std::default_random_engine(seed_temp);
	    hip_angle = hip_distribution(generator);
	    thigh_angle = thigh_distribution(generator);
	    calf_angle = calf_distribution(generator);
	    flag = flag_distribution(generator);
	    mPosition[start + 3 * i + 0] = hip_angle;
	    mPosition[start + 3 * i + 1] = thigh_angle;
	    mPosition[start + 3 * i + 2] = calf_angle;
	    mGroundFlag[i] = flag;
		}		
	}

	void setPitchRollToQuat(float pitch, float roll)
	{
		Eigen::Matrix3d ZYX;
		ZYX = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitX());
		Eigen::Quaterniond quat(ZYX);
		mPosition[3] = quat.w();
		mPosition[4] = quat.x();
		mPosition[5] = quat.y();
		mPosition[6] = quat.z();
	}

	void randomVelocityGenerator()
	{

	}

	void generateTrainingSamples2D(std::string file_name, int data_size)
	{
		std::string filePath = "/home/sonic/Project/Alien_Gesture/KinectRaisimtest/"+ file_name +".txt";
    std::cout << filePath << std::endl;
    // write File
    std::ofstream writeFile;
    writeFile.open(filePath);

    int succ_cnt = 0;
    // Generate data
    for(int i =0; i < (int)data_size / 2; ++i)
    {
    	if(i%1000 == 0) std::cout << i << "th iteration" << std::endl;
      randomPoseGenerator();
      mSF.setCurrentState2D(mPosition);
      int successor = mSF.jointLimitValueFunction();
      writeFile << std::setprecision(3) << std::fixed << mPosition[start] << " ";
      writeFile << std::setprecision(3) << std::fixed << mPosition[start + 2] << " ";    	
      if(successor == 1) succ_cnt++;
      writeFile << successor << std::endl;
  		mSF.resetState();
    }
    int cnt = (int)data_size / 2;
    enableJointLimit();
    while(cnt < data_size)
    {
    	if(cnt%1000 == 0) std::cout << cnt << "th iteration" << std::endl;
      randomPoseGenerator();
      mSF.setCurrentState2D(mPosition);
      int successor = mSF.jointLimitValueFunction();
      if(successor == 1)
      {
				writeFile << std::setprecision(3) << std::fixed << mPosition[start] << " ";
      	writeFile << std::setprecision(3) << std::fixed << mPosition[start + 2] << " ";  	
	      succ_cnt++;
	      writeFile << successor << std::endl;
	      cnt++;
	    }
  		mSF.resetState();
    }
    writeFile.close();
    float succ_rate = (float)succ_cnt/data_size;
    std::cout << "success : " << succ_rate << std::endl;
    std::cout << "joint : " << mSF.joint_violate << std::endl;
    std::cout << "ground form : " << mSF.ground_form_violate << std::endl;
    std::cout << "self collision : " << mSF.selfcol_violate << std::endl;
    std::cout << "ground collision : " << mSF.ground_col_violate << std::endl;
	}

	void generateTrainingSamples(std::string file_name, int data_size)
  {
    std::string filePath = "/home/sonic/Project/Alien_Gesture/KinectRaisimtest/"+ file_name +".txt";
    std::cout << filePath << std::endl;
    // write File
    std::ofstream writeFile;
    writeFile.open(filePath);

    int succ_cnt = 0;
    // Generate data
    for(int i =0; i < (int)data_size / 2; ++i)
    {
    	if(i%1000 == 0) std::cout << i << "th iteration" << std::endl;
      randomPoseGenerator();
      mSF.setCurrentState(mPosition, mGroundFlag);
      int successor = mSF.staticValueFunction();
      writeFile << std::setprecision(3) << std::fixed << mPitch << " " << mRoll << " ";
      for(int j = 0; j < 12; ++j)
      	writeFile << std::setprecision(3) << std::fixed << mPosition[start + j] << " ";
      for(int j = 0; j < 4; ++j)
      	writeFile << mGroundFlag[j] << " ";      	
      if(successor == 1) succ_cnt++;
      writeFile << successor << std::endl;
  		mSF.resetState();
    }
    int cnt = (int)data_size / 2;
    enableJointLimit();
    while(cnt < data_size)
    {
    	if(cnt%1000 == 0) std::cout << cnt << "th iteration" << std::endl;
      randomPoseGenerator();
      mSF.setCurrentState(mPosition, mGroundFlag);
      int successor = mSF.staticValueFunctionTest();
      if(successor == 1)
      {
	      writeFile << std::setprecision(3) << std::fixed << mPitch << " " << mRoll << " ";
	      for(int j = 0; j < 12; ++j)
	      	writeFile << std::setprecision(3) << std::fixed << mPosition[start + j] << " ";
	      for(int j = 0; j < 4; ++j)
	      	writeFile << mGroundFlag[j] << " ";      	
	      succ_cnt++;
	      writeFile << successor << std::endl;
	      cnt++;
	    }
  		mSF.resetState();

    }
    writeFile.close();
    float succ_rate = (float)succ_cnt/data_size;
    std::cout << "success : " << succ_rate << std::endl;
    std::cout << "joint : " << mSF.joint_violate << std::endl;
    std::cout << "ground form : " << mSF.ground_form_violate << std::endl;
    std::cout << "self collision : " << mSF.selfcol_violate << std::endl;
    std::cout << "ground collision : " << mSF.ground_col_violate << std::endl;
  }


	void generateEnsembleSamples(std::string file_name, int data_size)
  {
    std::string filePath1 = "/home/sonic/Project/Alien_Gesture/KinectRaisimtest/"+ file_name +"1.txt";
    std::string filePath2 = "/home/sonic/Project/Alien_Gesture/KinectRaisimtest/"+ file_name +"2.txt";
    std::string filePath3 = "/home/sonic/Project/Alien_Gesture/KinectRaisimtest/"+ file_name +"3.txt";
    std::string filePath4 = "/home/sonic/Project/Alien_Gesture/KinectRaisimtest/"+ file_name +"_all.txt";


    // write File
    std::ofstream writeFile1;
    writeFile1.open(filePath1);
    std::ofstream writeFile2;
    writeFile2.open(filePath2);
    std::ofstream writeFile3;
    writeFile3.open(filePath3);
		std::ofstream writeFile4;
    writeFile4.open(filePath4);

    int succ_cnt1 = 0;
    int succ_cnt2 = 0;
    int succ_cnt3 = 0;
    int succ_cnt4 = 0;
    // Generate data
    for(int i =0; i < (int)data_size / 2; ++i)
    {
    	if(i%1000 == 0) std::cout << i << "th iteration" << std::endl;
      randomPoseGenerator();
      mSF.setCurrentState(mPosition, mGroundFlag);
      int successor1 = mSF.jointLimitValueFunction();
      int successor3 = mSF.groundValueFunction();
      int successor2 = mSF.selfCollisionValueFunction();
      int successor4 = successor1 * successor2 * successor3;
      writeFile1 << std::setprecision(3) << std::fixed << mPitch << " " << mRoll << " ";
      writeFile2 << std::setprecision(3) << std::fixed << mPitch << " " << mRoll << " ";
      writeFile3 << std::setprecision(3) << std::fixed << mPitch << " " << mRoll << " ";
      writeFile4 << std::setprecision(3) << std::fixed << mPitch << " " << mRoll << " ";
      for(int j = 0; j < 12; ++j){
      	writeFile1 << std::setprecision(3) << std::fixed << mPosition[start + j] << " ";
      	writeFile2 << std::setprecision(3) << std::fixed << mPosition[start + j] << " ";
      	writeFile3 << std::setprecision(3) << std::fixed << mPosition[start + j] << " ";
      	writeFile4 << std::setprecision(3) << std::fixed << mPosition[start + j] << " ";
      }
      for(int j = 0; j < 4; ++j){
      	writeFile1 << mGroundFlag[j] << " ";      	
      	writeFile2 << mGroundFlag[j] << " ";      	
      	writeFile3 << mGroundFlag[j] << " ";      	
      	writeFile4 << mGroundFlag[j] << " ";      	
      }
      if(successor1 == 1) succ_cnt1++;
      if(successor2 == 1) succ_cnt2++;
      if(successor3 == 1) succ_cnt3++;
      if(successor4 == 1) succ_cnt4++;

      writeFile1 << successor1 << std::endl;
      writeFile2 << successor2 << std::endl;
      writeFile3 << successor3 << std::endl;
      writeFile4 << successor4 << std::endl;
  		mSF.resetState();
    }
    int cnt1 = (int)data_size / 2;
    int cnt2 = (int)data_size / 2;
    int cnt3 = (int)data_size / 2;
    enableJointLimit();
    while(cnt1 < data_size)
    {
    	if(cnt1%1000 == 0) std::cout << "1st" << cnt1 << "th iteration" << std::endl;
      randomPoseGenerator();
      mSF.setCurrentState(mPosition, mGroundFlag);
      int successor1 = mSF.jointLimitValueFunction();
      int successor3 = mSF.groundValueFunction();
      int successor2 = mSF.selfCollisionValueFunction();
      int successor4 = successor1 * successor2 * successor3;
      if(successor1 == 1)
      {
	      writeFile1 << std::setprecision(3) << std::fixed << mPitch << " " << mRoll << " ";
	      writeFile4 << std::setprecision(3) << std::fixed << mPitch << " " << mRoll << " ";
	      for(int j = 0; j < 12; ++j){
	      	writeFile1 << std::setprecision(3) << std::fixed << mPosition[start + j] << " ";
	      	writeFile4 << std::setprecision(3) << std::fixed << mPosition[start + j] << " ";
	      }
	      for(int j = 0; j < 4; ++j){
	      	writeFile1 << mGroundFlag[j] << " ";      	
	      	writeFile4 << mGroundFlag[j] << " ";      	
	      }
	      succ_cnt1++;
	      writeFile1 << successor1 << std::endl;
	      writeFile4 << successor4 << std::endl;
	      cnt1++;
	      if(successor4 == 1) succ_cnt4++;
	    }
  		mSF.resetState();
    }
    disableJointLimit();
    while(cnt2 < data_size)
    {
    	if(cnt2%1000 == 0) std::cout<< "2nd" << cnt2 << "th iteration" << std::endl;
      randomPoseGenerator();
      mSF.setCurrentState(mPosition, mGroundFlag);
      int successor1 = mSF.jointLimitValueFunction();
      int successor3 = mSF.groundValueFunction();
      int successor2 = mSF.selfCollisionValueFunction();
      int successor4 = successor1 * successor2 * successor3;
      if(successor2 == 1)
      {
	      writeFile2 << std::setprecision(3) << std::fixed << mPitch << " " << mRoll << " ";
 	      writeFile4 << std::setprecision(3) << std::fixed << mPitch << " " << mRoll << " ";
	      for(int j = 0; j < 12; ++j){
	      	writeFile2 << std::setprecision(3) << std::fixed << mPosition[start + j] << " ";
	      	writeFile4 << std::setprecision(3) << std::fixed << mPosition[start + j] << " ";
	      }
	      for(int j = 0; j < 4; ++j){
	      	writeFile2 << mGroundFlag[j] << " ";      	
	      	writeFile4 << mGroundFlag[j] << " ";      	
	      }
	      succ_cnt2++;
	      writeFile2 << successor2 << std::endl;
	      writeFile4 << successor4 << std::endl;
	      cnt2++;
	      if(successor4 == 1) succ_cnt4++;
	    }
  		mSF.resetState();
    }
    while(cnt3 < data_size)
    {
    	if(cnt3%1000 == 0) std::cout << "3rd" << cnt3 << "th iteration" << std::endl;
      randomPoseGenerator();
      mSF.setCurrentState(mPosition, mGroundFlag);
      int successor1 = mSF.jointLimitValueFunction();
      int successor3 = mSF.groundValueFunction();
      int successor2 = mSF.selfCollisionValueFunction();
      int successor4 = successor1 * successor2 * successor3;
      if(successor3 == 1)
      {
	      writeFile3 << std::setprecision(3) << std::fixed << mPitch << " " << mRoll << " ";
	      writeFile4 << std::setprecision(3) << std::fixed << mPitch << " " << mRoll << " ";
	      for(int j = 0; j < 12; ++j){
	      	writeFile3 << std::setprecision(3) << std::fixed << mPosition[start + j] << " ";
	      	writeFile4 << std::setprecision(3) << std::fixed << mPosition[start + j] << " ";
	      }
	      for(int j = 0; j < 4; ++j){
	      	writeFile3 << mGroundFlag[j] << " ";      	
	      	writeFile4 << mGroundFlag[j] << " ";      	
	      }
	      succ_cnt3++;
	      writeFile3 << successor3 << std::endl;
 	      writeFile4 << successor4 << std::endl;
	      cnt3++;
	      if(successor4 == 1) succ_cnt4++;
	    }
	    else{
	    	writeFile3 << std::setprecision(3) << std::fixed << mPitch << " " << mRoll << " ";
	      writeFile4 << std::setprecision(3) << std::fixed << mPitch << " " << mRoll << " ";
	      for(int j = 0; j < 12; ++j){
	      	writeFile3 << std::setprecision(3) << std::fixed << mPosition[start + j] << " ";
	      	writeFile4 << std::setprecision(3) << std::fixed << mPosition[start + j] << " ";
	      }
	      for(int j = 0; j < 4; ++j){
	      	writeFile3 << mGroundFlag[j] << " ";      	
	      	writeFile4 << mGroundFlag[j] << " ";      	
	      }
	      writeFile3 << successor3 << std::endl;
 	      writeFile4 << successor4 << std::endl;
	      cnt3++;
	      if(successor4 == 1) succ_cnt4++;
	    }
  		mSF.resetState();
    }

    writeFile1.close();
    writeFile2.close();
    writeFile3.close();
    writeFile4.close();
    float succ_rate1 = (float)succ_cnt1/data_size;
    float succ_rate2 = (float)succ_cnt2/data_size;
    float succ_rate3 = (float)succ_cnt3/data_size;
    float succ_rate4 = (float)succ_cnt4/(2* data_size);
    std::cout << "success1 : " << succ_rate1 << std::endl;
    std::cout << "success2 : " << succ_rate2 << std::endl;
    std::cout << "success3 : " << succ_rate3 << std::endl;
    std::cout << "success4 : " << succ_rate4 << std::endl;
    std::cout << "joint : " << mSF.joint_violate << std::endl;
    std::cout << "ground form : " << mSF.ground_form_violate << std::endl;
    std::cout << "self collision : " << mSF.selfcol_violate << std::endl;
    std::cout << "ground collision : " << mSF.ground_col_violate << std::endl;
  }

private:
  std::uniform_real_distribution<double> pitch_distribution;
  std::uniform_real_distribution<double> roll_distribution;
  std::uniform_real_distribution<double> hip_distribution;
  std::uniform_real_distribution<double> thigh_distribution;
  std::uniform_real_distribution<double> calf_distribution;
  std::uniform_int_distribution<int> flag_distribution;
  Eigen::VectorXd mPosition;
  std::vector<int> mGroundFlag;
	float mPitch, mRoll;
	bool mStatic;
	int start;
	SafetyFunction mSF;
};


#endif