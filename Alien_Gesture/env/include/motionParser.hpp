#ifndef _motionParser_H_
#define _motionParser_H_

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <string>
#include <fstream>
#include <Eigen/Core>
#include <math.h>

using std::vector;
using std::string;
using std::ifstream;


class motionParser
{
public:
	motionParser(){
		frame_off.setZero();
		frame_rot = Eigen::Matrix3d::Identity();
		dt = 0.033333;
	}

	motionParser(Eigen::Vector3d prev_frame_off, Eigen::Matrix3d prev_frame_rot){
		frame_off = prev_frame_off;
		frame_rot = prev_frame_rot;
		dt = 0.033333;
	}

	~motionParser();
	vector<Eigen::VectorXd> motionClip(){return mMotionClip;}
	int motionLen(){return motion_len;}
	Eigen::Vector3d angleAxisComp(Eigen::Matrix3d compensator, Eigen::Vector3d temp){
		Eigen::Vector3d dst;
		double temp_angle = temp.norm();
	    Eigen::Vector3d temp_axis = temp / temp_angle;
	    Eigen::Matrix3d temp_rot;
	    temp_rot = Eigen::AngleAxisd(temp_angle, temp_axis);
	    Eigen::Matrix3d mod_rot = compensator * temp_rot;
	    Eigen::AngleAxisd mod_axis(mod_rot);
	    // std::cout << "angle : " << mod_axis.angle() << ", axis : " << mod_axis.axis().transpose() << std::endl;
	    dst = mod_axis.angle() * mod_axis.axis();
		return dst;
	}

	void motionBVH(const string &abs_file, int clip_len) {
		vector<Eigen::VectorXd> motion;
		ifstream motionFile(abs_file);
		if(!motionFile){
			std::cout << "Expected valid file name" << std::endl;

		}
		string temp;
		while(!motionFile.eof()){
			// motionFile >> temp;
			// motionFile >> temp;
			Eigen::VectorXd clip(clip_len);
			for(int i = 0 ; i < clip_len; ++i)
			{
				// motionFile >> temp;
				// string copy;
				// copy = temp;
				// copy.resize(copy.length()-1);
				// clip(i) = stof(copy);
				float ang;
				motionFile >> ang;
				clip(i) = ang;
			}
			// float ang;
			// motionFile >> ang;
			// clip(clip_len-1) = ang;
			motion.push_back(clip);
	    }
	    motionFile.close();
	    motion.pop_back();
		mMotionClip = motion;
		motion_len = mMotionClip.size();
	}

	void motionBVH_XAlign(const string &abs_file, int clip_len) {
		vector<Eigen::VectorXd> motion;
		ifstream motionFile(abs_file);
		if(!motionFile){
			std::cout << "Expected valid file name" << std::endl;

		}
		string temp;
		while(!motionFile.eof()){
			Eigen::VectorXd clip(clip_len);
			for(int i = 0 ; i < clip_len; ++i)
			{
				float ang;
				motionFile >> ang;
				clip(i) = ang;
			}
			motion.push_back(clip);
	    }
	    motionFile.close();
	    motion.pop_back();

	    Eigen::Matrix3d compensator;
	    compensator = Eigen::AngleAxisd(-M_PI/2, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(-M_PI/2, Eigen::Vector3d::UnitZ());

	   	for(int i=0; i< motion.size();++i)
	   	{
	   		Eigen::Vector4d quat;
	   		for(int j = 0; j<4; ++j)
		    	quat[j] = motion[i][j+3];
		    Eigen::Vector3d axis;
		    const double norm = (std::sqrt(quat[1]*quat[1] + quat[2]*quat[2] + quat[3]*quat[3]));
		    if(fabs(norm) < 1e-12) {
		      axis[0] = 0;
		      axis[1] = 0;
		      axis[2] = 0;
		    }
		    else{
		      const double normInv = 1.0/norm;
		      const double angleNomrInv = std::acos(std::min(quat[0],1.0))*2.0*normInv;
		      axis[0] = quat[1] * angleNomrInv;
		      axis[1] = quat[2] * angleNomrInv;
		      axis[2] = quat[3] * angleNomrInv;
		    }

		    Eigen::Vector3d modified = angleAxisComp(compensator, axis);

		    const double angle = modified.norm();
			if(angle < 1e-12) {
				quat = Eigen::Vector4d(1,0,0,0);
			}
			else{		
				const double angleInv = 1.0/angle;
				Eigen::Vector3d axis = angleInv * modified;
				double s = sin(0.5*angle);
				double c = cos(0.5*angle);
				quat = Eigen::Vector4d(c, s*axis[0], s*axis[1], s*axis[2]);
			}

		    for(int j = 0; j<4; ++j)
		    	motion[i][j+3] = quat[j];
	   	}

		mMotionClip = motion;
		motion_len = mMotionClip.size();
	}

	void motionBVH_XAlign2(const string &abs_file, int clip_len) {
		vector<Eigen::VectorXd> motion;
		ifstream motionFile(abs_file);
		if(!motionFile){
			std::cout << "Expected valid file name" << std::endl;

		}
		string temp;
		while(!motionFile.eof()){
			Eigen::VectorXd clip(clip_len);
			for(int i = 0 ; i < clip_len; ++i)
			{
				float ang;
				motionFile >> ang;
				clip(i) = ang;
			}
			motion.push_back(clip);
	    }
	    motionFile.close();
	    motion.pop_back();

	    Eigen::Vector3d init_xy = Eigen::Vector3d::Zero();
	    init_xy[0] = motion[0][0];
	    init_xy[1] = motion[0][1];

	    Eigen::Vector3d init_euler;
	    init_euler[0] = motion[0][3];
	    init_euler[1] = motion[0][4];
	    init_euler[2] = motion[0][5];
	    double angle = init_euler.norm();
	    Eigen::Vector3d init_axis = init_euler / angle;
	    Eigen::Matrix3d init_rot;
	    init_rot = Eigen::AngleAxisd(angle, init_axis);

	   	Eigen::Matrix3d compensator = init_rot.transpose();

	   	for(int i=0; i< motion.size();++i)
	   	{
		    for(int j = 0; j<3; ++j)
		    	motion[i][j] -= init_xy[j];

	   		Eigen::Vector3d temp;
	   		for(int j = 0; j<3; ++j)
		    	temp[j] = motion[i][j+3];

		    Eigen::Vector3d modified = angleAxisComp(compensator, temp);
		    for(int j = 0; j<3; ++j)
		    	motion[i][j+3] = modified[j];
	   	}
		mMotionClip = motion;
		motion_len = mMotionClip.size();
	}
	
	void motionBVH_XAlign3(const string &abs_file, int clip_len) {
		vector<Eigen::VectorXd> motion;
		ifstream motionFile(abs_file);
		if(!motionFile){
			std::cout << "Expected valid file name" << std::endl;

		}
		string temp;
		while(!motionFile.eof()){
			Eigen::VectorXd clip(clip_len);
			for(int i = 0 ; i < clip_len; ++i)
			{
				float ang;
				motionFile >> ang;
				clip(i) = ang;
			}
			motion.push_back(clip);
	    }
	    motionFile.close();
	    motion.pop_back();

	    Eigen::Vector3d init_euler;

	    const double norm = (std::sqrt(motion[0][4]*motion[0][4] + motion[0][5]*motion[0][5] +motion[0][6]*motion[0][6]));
	    if(fabs(norm) < 1e-12) {
	      init_euler[0] = 0;
	      init_euler[1] = 0;
	      init_euler[2] = 0;
	    }
	    else{
	      const double normInv = 1.0/norm;
	      const double angleNomrInv = std::acos(std::min(motion[0][3],1.0))*2.0*normInv;
	      init_euler[0] = motion[0][4] * angleNomrInv;
	      init_euler[1] = motion[0][5] * angleNomrInv;
	      init_euler[2] = motion[0][6]* angleNomrInv;
	    }

	    double angle = init_euler.norm();
	    Eigen::Vector3d init_axis = init_euler / angle;
	    Eigen::Matrix3d init_rot;
	    init_rot = Eigen::AngleAxisd(angle, init_axis);

	   	Eigen::Matrix3d compensator = init_rot.transpose();

	   	for(int i=0; i< motion.size();++i)
	   	{
		    Eigen::Vector4d quat;
	   		for(int j = 0; j<4; ++j)
		    	quat[j] = motion[i][j+3];
		    Eigen::Vector3d axis;
		    const double norm = (std::sqrt(quat[1]*quat[1] + quat[2]*quat[2] + quat[3]*quat[3]));
		    if(fabs(norm) < 1e-12) {
		      axis[0] = 0;
		      axis[1] = 0;
		      axis[2] = 0;
		    }
		    else{
		      const double normInv = 1.0/norm;
		      const double angleNomrInv = std::acos(std::min(quat[0],1.0))*2.0*normInv;
		      axis[0] = quat[1] * angleNomrInv;
		      axis[1] = quat[2] * angleNomrInv;
		      axis[2] = quat[3] * angleNomrInv;
		    }

		    Eigen::Vector3d modified = angleAxisComp(compensator, axis);

		    const double angle = modified.norm();
			if(angle < 1e-12) {
				quat = Eigen::Vector4d(1,0,0,0);
			}
			else{		
				const double angleInv = 1.0/angle;
				Eigen::Vector3d axis = angleInv * modified;
				double s = sin(0.5*angle);
				double c = cos(0.5*angle);
				quat = Eigen::Vector4d(c, s*axis[2], s*axis[1], s*axis[0]);
			}

		    for(int j = 0; j<4; ++j)
		    	motion[i][j+3] = quat[j];
	   	}

		mMotionClip = motion;
		motion_len = mMotionClip.size();
	}

	Eigen::VectorXd getFrame(int frame){
		int clip = frame % motion_len;
		auto current = mMotionClip[clip];

		Eigen::Vector3d temp1(current[0], current[1], current[2]);
		Eigen::Vector3d temp2(current[3], current[4], current[5]);

		Eigen::Vector3d modified = angleAxisComp(frame_rot, temp2);
		Eigen::Vector3d pose_mod = frame_rot * temp1;

		for(int i = 0; i < 3; ++i)
		{
			current[i] = frame_off[i] + pose_mod[i];
			current[i+3] = modified[i];
		}

		if(clip == motion_len - 1)
		{
			Eigen::Vector3d temp_rot(0, 0, current[5]);
			double angle = temp_rot.norm();
		    Eigen::Vector3d axis_last = temp_rot / angle;
			frame_rot = Eigen::AngleAxisd(angle, axis_last);
			frame_off = Eigen::Vector3d(current[0], current[1], 0);
		}
		return current;
	}

	void manualStepWidth()
	{
		int size = mMotionClip.size();
		for(int i =0; i < size ; ++i)
		{
			mMotionClip[i][6] -= 0.25 ;
			mMotionClip[i][9] += 0.25;
			mMotionClip[i][12] -= 0.20;
			mMotionClip[i][15] += 0.14;
		}
	}

	void getFramePoseVel(int frame, Eigen::VectorXd& pose, Eigen::VectorXd& vel){
		int clip = frame % motion_len;
		auto current = mMotionClip[clip];

		Eigen::Vector3d temp1(current[0], current[1], current[2]);
		Eigen::Vector3d temp2(current[3], current[4], current[5]);

		Eigen::Vector3d modified = angleAxisComp(frame_rot, temp2);
		Eigen::Vector3d pose_mod = frame_rot * temp1;

		for(int i = 0; i < 3; ++i)
		{
			current[i] = frame_off[i] + pose_mod[i];
			current[i+3] = modified[i];
		}

		if(clip == motion_len - 1)
		{
			Eigen::Vector3d temp_rot(0, 0, current[5]);
			double angle = temp_rot.norm();
		    Eigen::Vector3d axis_last = temp_rot / angle;
			frame_rot = Eigen::AngleAxisd(angle, axis_last);
			frame_off = Eigen::Vector3d(current[0], current[1], 0);
		}

		int next_clip = (frame+1) % motion_len;
		auto next = mMotionClip[next_clip];
		Eigen::Vector3d temp_pos_next(next[0], next[1], next[2]);
		Eigen::Vector3d temp_rot_next(next[3], next[4], next[5]);

		Eigen::Vector3d rot_next = angleAxisComp(frame_rot, temp_rot_next);
		Eigen::Vector3d pose_next = frame_rot * temp_pos_next;

		for(int i = 0; i < 3; ++i)
		{
			next[i] = frame_off[i] + pose_next[i];
			next[i+3] = rot_next[i];
		}

		pose = current;
		vel = (next - current) / dt;
	}

	void getMeanStd(Eigen::VectorXd& mean, Eigen::VectorXd& dev){
		for(int i = 0; i< mMotionClip.size(); ++i){
			Eigen::VectorXd current = mMotionClip[i];
			for(int j = 3; j < current.size();++j){
				mean[j-3] += current[j];
			}
		}
		mean /= mMotionClip.size();
		for(int i = 0; i< mMotionClip.size(); ++i){
			Eigen::VectorXd current = mMotionClip[i];
			for(int j = 3; j < current.size();++j){
				dev[j-3] += (current[j] - mean[j-3]) * (current[j] - mean[j-3]);
			}
		}
		dev /= (mMotionClip.size() - 1);
		for(int j = 0; j < dev.size();++j){
			dev[j] = sqrt(dev[j]);
		}
	}

private:
	vector<Eigen::VectorXd> mMotionClip;
	int motion_len;
	Eigen::Vector3d frame_off;
	Eigen::Matrix3d frame_rot;
	float dt;

};

#endif