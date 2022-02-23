#ifndef _IKIPOPT_H
#define _IKIPOPT_H

#include <iostream>
#include <Eigen/Core>
#include "raisim/World.hpp"
#include <IpTNLP.hpp>
#include <IpIpoptApplication.hpp>

class IKOptimization : public Ipopt::TNLP
{
public:
	IKOptimization(raisim::ArticulatedSystem* robot_, std::vector<std::string> frames, std::vector<Eigen::Vector3d> targets_) : 
	mRobot(robot_), target_frames(frames), mTargets(targets_){
		int gcDim_ = mRobot->getGeneralizedCoordinateDim();
	    int gvDim_ = mRobot->getDOF();
	    gc_.setZero(gcDim_);
	    gv_.setZero(gvDim_); 
		mInitialPositions = get_current_q();
		mSolution = mInitialPositions;
		jointLimits = mRobot->getJointLimits();
	}

	void setManualJointLimit(int idx, float x_l, float x_u)
	{
		jointLimits[idx][0] = x_l;
		jointLimits[idx][1] = x_u;
	}

	void resetJointLimit()
	{
		jointLimits = mRobot->getJointLimits();
	}

	const std::vector<Eigen::Vector3d>& GetTargets(){return mTargets;}
	void ClearTarget(){
		mTargets.clear();
	}
	void SetTarget(std::vector<Eigen::Vector3d> targets_){
		mTargets = targets_;
	}

	void ClearFrame(){
		target_frames.clear();
	}

	void SetFrame(std::vector<std::string> frames){
		target_frames = frames;
	}

	const Eigen::VectorXd& GetSolution(){
		return mSolution;
	}
	void SetSolution(const Eigen::VectorXd& sol){
		mSolution = sol;
	}

	~IKOptimization(){}

	bool get_nlp_info(Ipopt::Index& n, Ipopt::Index& m, Ipopt::Index& nnz_jac_g, Ipopt::Index& nnz_h_lag, IndexStyleEnum& index_style) override{
		n = mRobot->getDOF();
		m = 0;
		nnz_jac_g = 0;
		nnz_h_lag = n;
		index_style = TNLP::C_STYLE;
		return true;
	}

	bool get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u, Ipopt::Index m, Ipopt::Number* g_l, Ipopt::Number* g_u) override{
		int nRoot = 6;
		Eigen::VectorXd current = get_current_q();
		for(int i =0;i<nRoot;i++)
		{
			x_l[i] = current[i];
			x_u[i] = current[i];

		}
		for(int i =nRoot;i<jointLimits.size();i++)
		{
			if(jointLimits[i][0] == 0 && jointLimits[i][1] == 0){ 
				jointLimits[i][0] = -3.14;
				jointLimits[i][1] = 3.14;
			}
			x_l[i] = jointLimits[i][0];
			x_u[i] = jointLimits[i][1];
		}
		return true;
	}

	Eigen::AngleAxisd GetDiff(const Eigen::Quaterniond& diff)
	{
		Eigen::AngleAxisd diff1,diff2;
		diff1 = Eigen::AngleAxisd(diff);

		if(diff1.angle()>3.141592)
		{
			diff2.axis() = -diff1.axis();
			diff2.angle() = 3.141592*2 - diff1.angle();	
		}
		else
			diff2 = diff1;
		return diff2;
	}

	bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number* x, bool init_z, Ipopt::Number* z_L, Ipopt::Number* z_U,
							Ipopt::Index m, bool init_lambda, Ipopt::Number* lambda) override{
		mSavePositions = get_current_q();		
		if(init_x)
		{
			for(int i =0;i<n;i++){
				x[i] = mSavePositions[i];
			}
		}
		mSavePositions = get_current_q();
		return true;

	}

	bool eval_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number& obj_value) override{
		Eigen::VectorXd q(n);
		for(int i =0;i<n;i++)
			q[i] = x[i];
		set_robot_pose(q);
		obj_value = 0;
		for(int i = 0 ; i < mTargets.size() ; i ++){
			Eigen::Vector3d currentPart;
			currentPart = get_robot_pose(target_frames[i]);
			Eigen::Vector3d deviation = (currentPart - mTargets[i]);
			obj_value += deviation.norm();
		}

		return true;
	}

	bool eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number* grad_f) override{
		Eigen::VectorXd q(n),g(n);
		for(int i =0;i<n;i++)
			q[i] = x[i];
		g.setZero();
		set_robot_pose(q);
		for(int i = 0 ; i < mTargets.size() ; i ++){
			Eigen::Vector3d currentPart;
			currentPart = get_robot_pose(target_frames[i]);
			Eigen::Vector3d deviation = (currentPart - mTargets[i]);
			Eigen::MatrixXd J = get_jacobian(target_frames[i]);
			Eigen::MatrixXd J_inv = J.transpose()*(J*J.transpose()).inverse();
			g += J_inv*deviation;
		}
		for(int i =0;i<n;i++)
			grad_f[i] = g[i];
		return true;
	}

	bool eval_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Index m, Ipopt::Number* g) override{
		return true;
	}

	bool eval_jac_g( Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Index m, Ipopt::Index nele_jac, Ipopt::Index* iRow, Ipopt::Index *jCol, Ipopt::Number* values) override{
		return true;
	}
	bool eval_h(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number obj_factor, Ipopt::Index m, const Ipopt::Number* lambda,
				bool new_lambda, Ipopt::Index nele_hess, Ipopt::Index* iRow, Ipopt::Index* jCol, Ipopt::Number* values) override{
		int nnz = 0;
		if(values == NULL)
		{
			for(int i=0;i<n;i++)
			{
				iRow[nnz] = i;
				jCol[nnz++] = i;
			}
		}
		else
		{
			for(int i=0;i<n;i++)
			{
				values[nnz++] = 1.0;
			}
		}
	return true;
	}

	void finalize_solution(Ipopt::SolverReturn status, Ipopt::Index n, const Ipopt::Number* x, const Ipopt::Number* z_L, const Ipopt::Number* z_U, Ipopt::Index m, 
							const Ipopt::Number* g, const Ipopt::Number* lambda, Ipopt::Number obj_value, const Ipopt::IpoptData* ip_data, Ipopt::IpoptCalculatedQuantities* ip_cq) override{
		for(int i=0;i<n;i++)
			mSolution[i] = x[i];
		set_robot_pose(mSavePositions);
	}

	Eigen::VectorXd get_current_q(){
		Eigen::VectorXd pose;
		mRobot->getState(gc_, gv_);
		pose.setZero(gv_.size());
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

	Eigen::Vector3d get_robot_pose(const std::string frame_name)
	{
		Eigen::Vector3d pose;
		raisim::Vec<3> tempPosition;
		mRobot->getFramePosition(mRobot->getFrameByName(frame_name), tempPosition);
		pose = tempPosition.e();
		return pose;
	}

	void set_robot_pose(const Eigen::VectorXd pose_)
	{
		Eigen::VectorXd pose_quat(pose_.size()+1);
		for(int i = 0 ; i < 3; i++)
		{
		  pose_quat[i] = pose_[i];
		}
		raisim::Vec<4> quat;
		raisim::Vec<3> axis;
		axis[0] = pose_[3]; axis[1] = pose_[4]; axis[2] = pose_[5];
		raisim::eulerVecToQuat(axis, quat);
		for(int i = 0 ; i < 4; i++)
		{
		  pose_quat[i+3] = quat[i];
		}
		for(int i = 6 ; i < pose_.size(); i++)
		{
		  pose_quat[i+1] = pose_[i];
		}
		mRobot->setGeneralizedCoordinate(pose_quat);
	}

	Eigen::MatrixXd get_jacobian(const std::string joint_name)
	{
		mRobot->getState(gc_, gv_);
		Eigen::MatrixXd Jacobian(3,mRobot->getDOF());;
		Jacobian.setZero();

		auto& frame = mRobot->getFrameByName(joint_name);
		raisim::Vec<3> position_W;
		mRobot->getFramePosition(mRobot->getFrameByName(joint_name), position_W);
		mRobot->getDenseJacobian(frame.parentId, position_W, Jacobian);

		return Jacobian;
	}
	Eigen::MatrixXd get_RF_jacobian(const std::string joint_name)
	{
		mRobot->getState(gc_, gv_);
		Eigen::MatrixXd Jacobian(3,mRobot->getDOF());;
		Jacobian.setZero();

		auto& frame = mRobot->getFrameByName(joint_name);
		raisim::Vec<3> position_W;
		mRobot->getFramePosition(mRobot->getFrameByName(joint_name), position_W);
		mRobot->getDenseJacobian(frame.parentId, position_W, Jacobian);

		Jacobian.block(0,0,3,6).setZero();
		return Jacobian;
	}

protected:
	IKOptimization();
	IKOptimization(const IKOptimization&);
	IKOptimization& operator=(const IKOptimization&);
	raisim::ArticulatedSystem* mRobot;
	Eigen::VectorXd mSavePositions;
	Eigen::VectorXd	mInitialPositions;
	std::vector<std::string> target_frames;
	std::vector<Eigen::Vector3d> mTargets;
	Eigen::VectorXd	mSolution;
	Eigen::VectorXd gc_, gv_;
	std::vector<raisim::Vec<2>> jointLimits;
};

class IKOptimizationLeft : public Ipopt::TNLP
{
public:
	IKOptimizationLeft(raisim::ArticulatedSystem* robot_, Eigen::Vector3d target) : 
		mRobot(robot_), mTarget(target){
		int gcDim_ = mRobot->getGeneralizedCoordinateDim();
	    int gvDim_ = mRobot->getDOF();
	    gc_.setZero(gcDim_);
	    gv_.setZero(gvDim_); 
	    mRobot->getState(gc_, gv_);
		mInitialPositions = get_current_q();
		mSolution = Eigen::Vector3d(0,0,0);
		jointLimits = mRobot->getJointLimits();
		mCurrent.setZero(3);
		w_reg = 1E-1;
	}

	void setManualJointLimit(int idx, float x_l, float x_u)
	{
		jointLimits[idx][0] = x_l;
		jointLimits[idx][1] = x_u;
	}

	void resetJointLimit()
	{
		jointLimits = mRobot->getJointLimits();
	}

	const Eigen::Vector3d& GetTargets(){return mTarget;}
	void ClearTarget(){
		mTarget.setZero();
	}
	void SetTarget(Eigen::Vector3d target){
		mTarget = target;
	}

	void SetCurrentPose(Eigen::VectorXd current){
		mCurrent = current;
	}

	const Eigen::Vector3d& GetSolution(){
		return mSolution;
	}
	void SetSolution(const Eigen::VectorXd& sol){
		mSolution = sol;
	}

	const bool GetSuccStatus(){
		return mSuccess;
	}

	~IKOptimizationLeft(){}

	bool get_nlp_info(Ipopt::Index& n, Ipopt::Index& m, Ipopt::Index& nnz_jac_g, Ipopt::Index& nnz_h_lag, IndexStyleEnum& index_style) override{
		n = 3;
		m = 0;
		nnz_jac_g = 0;
		nnz_h_lag = n;
		index_style = TNLP::C_STYLE;
		return true;
	}

	bool get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u, Ipopt::Index m, Ipopt::Number* g_l, Ipopt::Number* g_u) override{
		for(int i =0 ;i<n;i++)
		{
			int j = i+9;
			if(jointLimits[j][0] == 0 && jointLimits[j][1] == 0){ 
				jointLimits[j][0] = -3.14;
				jointLimits[j][1] = 3.14;
			}
			x_l[i] = jointLimits[j][0];
			x_u[i] = jointLimits[j][1];
		}
		return true;
	}

	bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number* x, bool init_z, Ipopt::Number* z_L, Ipopt::Number* z_U,
							Ipopt::Index m, bool init_lambda, Ipopt::Number* lambda) override{
		mSavePositions = mInitialPositions;	
		if(init_x)
		{
			for(int i =0;i<n;i++){
				x[i] = mSavePositions[i];
			}
		}
		return true;

	}

	bool eval_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number& obj_value) override{
		Eigen::VectorXd q(n);
		for(int i =0;i<n;i++)
			q[i] = x[i];
		set_left_hand(q);
		obj_value = 0;
		Eigen::Vector3d currentPart;
		currentPart = get_robot_pose(targetFrame);
		Eigen::Vector3d deviation = (currentPart - mTarget);
		obj_value += 0.5 * deviation.squaredNorm();
		obj_value += 0.5 * w_reg * (q - mCurrent).squaredNorm();
		return true;
	}

	bool eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number* grad_f) override{
		Eigen::VectorXd q(n),g(n);
		for(int i =0;i<n;i++)
			q[i] = x[i];
		g.setZero();
		set_left_hand(q);

		Eigen::Vector3d currentPart;
		currentPart = get_robot_pose(targetFrame);
		Eigen::Vector3d deviation = (currentPart - mTarget);
		Eigen::MatrixXd J = get_left_jacobian();
		// Eigen::MatrixXd J_inv = J.transpose()*(J*J.transpose()).inverse();
		Eigen::MatrixXd J_inv = J.inverse();
		g += J_inv*deviation;
		g += w_reg * (q - mCurrent);

		for(int i =0;i<n;i++)
			grad_f[i] = g[i];
		return true;
	}

	bool eval_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Index m, Ipopt::Number* g) override{
		return true;
	}

	bool eval_jac_g( Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Index m, Ipopt::Index nele_jac, Ipopt::Index* iRow, Ipopt::Index *jCol, Ipopt::Number* values) override{
		return true;
	}
	bool eval_h(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number obj_factor, Ipopt::Index m, const Ipopt::Number* lambda,
				bool new_lambda, Ipopt::Index nele_hess, Ipopt::Index* iRow, Ipopt::Index* jCol, Ipopt::Number* values) override{
		int nnz = 0;
		if(values == NULL)
		{
			for(int i=0;i<n;i++)
			{
				iRow[nnz] = i;
				jCol[nnz++] = i;
			}
		}
		else
		{
			for(int i=0;i<n;i++)
			{
				values[nnz++] = 1.0;
			}
		}
	return true;
	}

	void finalize_solution(Ipopt::SolverReturn status, Ipopt::Index n, const Ipopt::Number* x, const Ipopt::Number* z_L, const Ipopt::Number* z_U, Ipopt::Index m, 
							const Ipopt::Number* g, const Ipopt::Number* lambda, Ipopt::Number obj_value, const Ipopt::IpoptData* ip_data, Ipopt::IpoptCalculatedQuantities* ip_cq) override{
		if(status == Ipopt::SUCCESS) mSuccess = true;
		else mSuccess = false;
		for(int i=0;i<n;i++)
			mSolution[i] = x[i];
		set_left_hand(mSavePositions);
	}

	Eigen::Vector3d get_current_q(){
		Eigen::Vector3d pose;
		mRobot->getState(gc_, gv_);
		pose = gc_.segment(10,3);
		return pose;
	}

	Eigen::Vector3d get_robot_pose(const std::string frame_name)
	{
		raisim::Mat<3,3> orientation_r;
    	mRobot->getBaseOrientation(orientation_r);
    	Eigen::Matrix3d body_rot= orientation_r.e();
		Eigen::Vector3d pose;
		raisim::Vec<3> tempPositionh;
		mRobot->getFramePosition(mRobot->getFrameByName("FL_hip_joint"), tempPositionh);
		raisim::Vec<3> tempPositionf;
		mRobot->getFramePosition(mRobot->getFrameByName(frame_name), tempPositionf);
		// pose = body_rot.inverse() * (tempPositionf.e() - tempPositionh.e());
		pose = tempPositionf.e();
		return pose;
	}

	void set_robot_pose(const Eigen::VectorXd pose_)
	{
		Eigen::VectorXd pose_quat(pose_.size()+1);
		for(int i = 0 ; i < 3; i++)
		{
		  pose_quat[i] = pose_[i];
		}
		raisim::Vec<4> quat;
		raisim::Vec<3> axis;
		axis[0] = pose_[3]; axis[1] = pose_[4]; axis[2] = pose_[5];
		raisim::eulerVecToQuat(axis, quat);
		for(int i = 0 ; i < 4; i++)
		{
		  pose_quat[i+3] = quat[i];
		}
		for(int i = 6 ; i < pose_.size(); i++)
		{
		  pose_quat[i+1] = pose_[i];
		}
		mRobot->setGeneralizedCoordinate(pose_quat);
	}


	void set_left_hand(const Eigen::VectorXd pose_)
	{
		mRobot->getState(gc_, gv_);
		Eigen::VectorXd pose_temp;
		pose_temp = gc_;
		for(int i =0 ; i < pose_.size(); ++i)
		{
			pose_temp[i+10] = pose_[i];
		}
		mRobot->setGeneralizedCoordinate(pose_temp);
	}

	Eigen::MatrixXd get_jacobian(const std::string joint_name)
	{
		mRobot->getState(gc_, gv_);
		Eigen::MatrixXd Jacobian(3,mRobot->getDOF());
		Jacobian.setZero();

		auto& frame = mRobot->getFrameByName(joint_name);
		raisim::Vec<3> position_W;
		mRobot->getFramePosition(mRobot->getFrameByName(joint_name), position_W);
		mRobot->getDenseJacobian(frame.parentId, position_W, Jacobian);

		return Jacobian;
	}

	Eigen::MatrixXd get_left_jacobian()
	{
		Eigen::MatrixXd leftJacobian(3,3);
		Eigen::MatrixXd fullJacobian(3,mRobot->getDOF());
		leftJacobian.setZero();
		fullJacobian.setZero();

		auto& frame = mRobot->getFrameByName(targetFrame);
		raisim::Vec<3> position_W;
		mRobot->getFramePosition(mRobot->getFrameByName(targetFrame), position_W);
		mRobot->getDenseJacobian(frame.parentId, position_W, fullJacobian);
		leftJacobian = fullJacobian.block(0,9,3,3);
		return leftJacobian;
	}


	Eigen::MatrixXd get_RF_jacobian(const std::string joint_name)
	{
		mRobot->getState(gc_, gv_);
		Eigen::MatrixXd Jacobian(3,mRobot->getDOF());;
		Jacobian.setZero();

		auto& frame = mRobot->getFrameByName(joint_name);
		raisim::Vec<3> position_W;
		mRobot->getFramePosition(mRobot->getFrameByName(joint_name), position_W);
		mRobot->getDenseJacobian(frame.parentId, position_W, Jacobian);

		Jacobian.block(0,0,3,6).setZero();
		return Jacobian;
	}

protected:
	IKOptimizationLeft();
	IKOptimizationLeft(const IKOptimizationLeft&);
	IKOptimizationLeft& operator=(const IKOptimizationLeft&);
	raisim::ArticulatedSystem* mRobot;
	Eigen::Vector3d mSavePositions;
	Eigen::Vector3d	mInitialPositions;
	Eigen::Vector3d mTarget;
	std::string targetFrame = "FL_foot_fixed";
	Eigen::Vector3d	mSolution;
	Eigen::VectorXd gc_, gv_;
	Eigen::VectorXd mCurrent;
	std::vector<raisim::Vec<2>> jointLimits;
	float w_reg;
	bool mSuccess;
};


#endif