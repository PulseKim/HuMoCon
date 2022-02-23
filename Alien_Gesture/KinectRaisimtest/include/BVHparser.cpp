#include <Eigen/Geometry>
#include "BVHparser.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <algorithm>
#define M_PI 3.14159265358979323846

using namespace std;

MotionNode::MotionNode()
{
	name = "";
	axisOrder = "";
	isRoot = false;
	isEnd = false;
	parent = nullptr;
	next = nullptr;
	childs = vector<MotionNode*>();
	channelNum = 0;
	// orientation = Eigen::AngleAxisd::Identity();
}
MotionNode* MotionNode::getParent()
{
	return parent;
}
vector<MotionNode*> MotionNode::getChilds()
{
	return childs;
}
void MotionNode::setParent(MotionNode* pnode)
{
	parent = pnode;
	pnode->addChild(this);
}
void MotionNode::addChild(MotionNode* cnode)
{
	childs.push_back(cnode);
}
void MotionNode::setRoot()
{
	isRoot = true;
}
void MotionNode::setEnd()
{
	isEnd = true;
}
void MotionNode::setName(string mname)
{
	name = mname;
}
void MotionNode::setAxisOrder(string maxisOrder)
{
	axisOrder = maxisOrder;
}
void MotionNode::setOffset(float x, float y, float z)
{
	offset[0] = x;
	offset[1] = y;
	offset[2] = z;
}
void MotionNode::setNext(MotionNode *nextNode)
{
	next = nextNode;
}
void MotionNode::setChannelNum(int mchannelNum)
{
	channelNum = mchannelNum;
}
bool MotionNode::checkEnd()
{
	return isEnd;
}
string MotionNode::getName_std()
{
	if(match_name_list.size()>=1)
		return match_name_list[0];
	else
	{
		cout<<"no such name in list"<<name<<endl;
		return "";
	}
}
string MotionNode::getName()
{
	return name;
}
string MotionNode::getAxisOrder()
{
	return axisOrder;
}
void MotionNode::getOffset(float* outOffset)
{
	for(int i=0;i<3;i++)
	{
		outOffset[i] = offset[i];
	}
}
float MotionNode::getOffset(int index)
{
	return offset[index];
}
int MotionNode::getChannelNum()
{
	return channelNum;
}
MotionNode* MotionNode::getNextNode()
{
	return next;
}
void MotionNode::initData(int frames)
{
	data = new float*[frames];
	for(int i = 0; i < frames; i++)
	{
		data[i] = new float[channelNum];
	}
}
bool MotionNode::ContainedInModel()
{
	if(match_name_list.size()==0)
		return false;
	else
		return true;
}
/// convert quaternion -> log(quaternion)
Eigen::Vector3d QuaternionToAngleAxis(Eigen::Quaterniond qt)
{
	double angle = 2.0*atan2(qt.vec().norm(), qt.w());
	if(std::abs(angle) < 1e-4)
		return Eigen::Vector3d::Zero();
	return angle * qt.vec().normalized();
}

/// convert log(quaternion) -> quaternion
Eigen::Quaterniond AngleAxisToQuaternion(Eigen::Vector3d angleAxis)
{
	Eigen::Quaterniond qt;
	if(angleAxis.norm() < 1e-4)
		return Eigen::Quaterniond(1,0,0,0);
	qt.vec() = angleAxis.normalized() * sin(angleAxis.norm());
	qt.w() = cos(angleAxis.norm());
	return qt;
}

void BVHparser::initMatchNameListForMotionNode(const char* path)
{
	string line;
	stringstream s;
	ifstream in(path);
    // std::cout<<"# reading file "<<path<<std::endl;
	string muscleName_std;
	string muscleName;
	MotionNode* curNode = getRootNode();
	float value;
	while(!in.eof())
	{
		getline(in, line);
		if(line.empty())
			break;
		s = stringstream(line);
		std::vector<string> match_name_list;
		while(!s.eof())
		{
			s>>muscleName;
			match_name_list.push_back(muscleName);
		}
		curNode = getRootNode();
		while(curNode!=NULL)
		{
			for(auto& name : match_name_list)
			{
				if(curNode->getName() == name)
				{
					curNode->match_name_list = match_name_list;
					break;
				}
			}
			curNode = curNode->getNextNode();
		}
	}
	in.close();
}

// BVHparser::BVHparser(const char* path)
// {

// }

BVHparser::BVHparser(const char* path, BVHType bvhType)
{
	mBvhType = bvhType;
	this->scale = 1.0;
	int lineNum = 0;
	int channelNum = 0;
	string channels[6];
	float offx, offy, offz;
	mPath = path;
	lowerBodyOnly = false;
	upperBodyOnly = false;
	ifstream in;
	in.open(path, ios::in);
	if(!in)
	{
		cerr << "Cannot open "<<path<<endl; exit(1);
	}
	string line;
	getline(in, line);										//HIERARCHY
	lineNum++;

	if(line.compare(0,9,"HIERARCHY")!=0)
	{ 
		cout << "Check the file format. The line number "<<lineNum<<" is not fit to the format"<<endl;
		cout<<line.compare("HIERARCHY")<<endl;
		cout<<line.length()<<endl;
		cout << line<<endl;
		exit(-1);
	}
	getline(in, line);										//ROOT Hips
	lineNum++;
	rootNode = new MotionNode();
	rootNode->setRoot();
	istringstream s(line);
	string bvh_keyword;
	string bvh_nodeName;
	s >> bvh_keyword; s >> bvh_nodeName;
	if(bvh_keyword.compare("ROOT")!=0)
	{
		cout << "Check the file format. The line number "<<lineNum<<" is not fit to the format"<<endl;
		cout << line<<endl;
		exit(-1);
	}
	rootNode->setName(bvh_nodeName);
	getline(in, line);										//{

	getline(in, line);										//	OFFSET 0.00 0.00 0.00
	s.str("");
	s = istringstream(line);
	s >> bvh_keyword; s >> offx; s >> offy; s >> offz;
	if(mBvhType == BVHType::BASKET)
		rootNode->setOffset(offx, offy-90, offz);
	else
		rootNode->setOffset(offx, offy, offz);


	getline(in, line);										//	CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation
	s.str("");
	s = istringstream(line);
	s >> bvh_keyword; s >> channelNum;
	string newAxisOrder = "";
	for(int i = 0; i < channelNum; i++)
	{
		s >> channels[i];
		if(channels[i].substr(1) == "rotation")
		{
			transform(channels[i].begin(), channels[i].end(), channels[i].begin(), ::tolower);
			newAxisOrder += channels[i][0];
		}
	}
	rootNode->setAxisOrder(newAxisOrder);
	rootNode->setChannelNum(channelNum);
	MotionNode* prevNode = rootNode;
	MotionNode* prevNode4NextNode = rootNode;
	getline(in, line);
	while(line.compare(0, 6, "MOTION") != 0)						
	{
		s.str("");
		s = istringstream(line);
		s >> bvh_keyword; s >> bvh_nodeName;
		if(bvh_keyword=="JOINT")							//	JOINT LeftUpLeg
		{
			MotionNode *newNode = new MotionNode();
			newNode->setName(bvh_nodeName);

			getline(in, line);								//	{
			getline(in, line);								//		OFFSET 3.64953 0.00000 0.00000
			s.str("");
			s = istringstream(line);
			s >> bvh_keyword; s >> offx; s >> offy; s >> offz;
			newNode->setOffset(offx, offy, offz);

			getline(in, line);								//		CHANNELS 3 Xrotation Yrotation Zrotation
			s.str("");
			s = istringstream(line);
			s >> bvh_keyword; s >> channelNum;

			newAxisOrder = "";
			for(int i = 0;i < channelNum;i++)
			{
				s >> channels[i];
				if(channels[i].substr(1) == "rotation")
				{
					transform(channels[i].begin(), channels[i].end(), channels[i].begin(), ::tolower);
					newAxisOrder += channels[i][0];
				}
			}
			newNode->setParent(prevNode);
			newNode->setChannelNum(channelNum);
			newNode->setAxisOrder(newAxisOrder);
			prevNode4NextNode->setNext(newNode);
			prevNode = newNode;
			prevNode4NextNode = newNode;
		}
		else if(bvh_keyword =="End")						//	End Site
		{
			MotionNode *newNode = new MotionNode();
			newNode->setName(prevNode->getName()+bvh_nodeName);
			newNode->setEnd();
			getline(in, line);								//	{
			getline(in, line);								//		OFFSET 3.64953 0.00000 0.00000
			s.str("");
			s = istringstream(line);
			s >> bvh_keyword; s >> offx; s >> offy; s >> offz;
			newNode->setOffset(offx, offy, offz);

			newNode->setParent(prevNode);
			getline(in, line);								//	}
		}
		else if(bvh_keyword =="}")
		{
			prevNode = prevNode->getParent();
		}
		else
		{
			cout << "Check the file format. "<< bvh_keyword <<" is right?" <<endl;
			exit(-1);
		}
		getline(in, line);
	}

// 	Start MotionNode										//MOTION
	string str1, str2;	//to get the string for format
	float fvalue;
	getline(in, line);										//Frames: 4692
	s.str("");
	s = istringstream(line);
	s >> str1; s >> fvalue;
	frames = fvalue;

	getline(in, line);										//Frame Time: 0.008333
	s.str("");
	s = istringstream(line);
	s >> str1; s >> str2; s >> fvalue;
	frameTime = fvalue;

	// cout << "frames : " << frames <<", frame time : " << frameTime << endl;

	float f[6];
	MotionNode *curNode;
	curNode = rootNode;
	while(curNode != nullptr)
	{
		curNode->initData(frames);
		curNode = curNode->getNextNode();
	}

	for(int i = 0; i < frames; i++)
	{
		curNode = rootNode;
		getline(in, line);
		s.str("");
		s = istringstream(line);
		while(curNode != nullptr)
		{
			for(int j = 0; j < curNode->getChannelNum(); j++)
			{
				s >> f[j];
				curNode->data[i][j] = f[j];
			}
			curNode = curNode->getNextNode();
		}

	}
	in.close();
	Eigen::Quaterniond q;
	Eigen::Vector3d euler;
	curNode = rootNode;
	for(int i = 0; i < frames; i++)
	{
		q = Eigen::Quaterniond::Identity();
		for(int j = 0; j < 3; j++)
		{
			if(curNode->getAxisOrder().substr(j,1).compare("x") == 0)
			{
				q = q * Eigen::AngleAxisd(curNode->data[i][j+3]/360.0 * 2.0 * M_PI, Eigen::Vector3d::UnitX());
			}
			else if(curNode->getAxisOrder().substr(j,1).compare("y") == 0)
			{
				q = q * Eigen::AngleAxisd(curNode->data[i][j+3]/360.0 * 2.0 * M_PI, Eigen::Vector3d::UnitY());
			}
			else
			{
				q = q * Eigen::AngleAxisd(curNode->data[i][j+3]/360.0 * 2.0 * M_PI, Eigen::Vector3d::UnitZ());
			}
		}
		float theta;
		theta = atan2(sqrt(q.x()*q.x() + q.y()*q.y() + q.z()*q.z()), q.w());
		Eigen::Vector3d root_axis = Eigen::Vector3d(q.x(), q.y(), q.z());
		root_axis.normalize();
		curNode->data[i][3] = 2.0 * theta * root_axis.x();
		curNode->data[i][4] = 2.0 * theta * root_axis.y();
		curNode->data[i][5] = 2.0 * theta * root_axis.z();
	}
	curNode = curNode->getNextNode();
	while(curNode != nullptr)
	{
		for(int i = 0; i < frames; i++)
		{
			q = Eigen::Quaterniond::Identity();
			for(int j = 0; j < 3; j++)
			{
				if(curNode->getAxisOrder().substr(j,1).compare("x") == 0)
				{
					q = q * Eigen::AngleAxisd(curNode->data[i][j]/360.0 * 2.0 * M_PI, Eigen::Vector3d::UnitX());
				}
				else if(curNode->getAxisOrder().substr(j,1).compare("y") == 0)
				{
					q = q * Eigen::AngleAxisd(curNode->data[i][j]/360.0 * 2.0 * M_PI, Eigen::Vector3d::UnitY());
				}
				else
				{
					q = q * Eigen::AngleAxisd(curNode->data[i][j]/360.0 * 2.0 * M_PI, Eigen::Vector3d::UnitZ());
				}
			}


			float theta;
			theta = atan2(sqrt(q.x()*q.x() + q.y()*q.y() + q.z()*q.z()), q.w());
			Eigen::Vector3d root_axis = Eigen::Vector3d(q.x(), q.y(), q.z());
			root_axis.normalize();
			curNode->data[i][0] = 2.0 * theta * root_axis.x();
			curNode->data[i][1] = 2.0 * theta * root_axis.y();
			curNode->data[i][2] = 2.0 * theta * root_axis.z();
		}
		curNode = curNode->getNextNode();
	}
	// this->initMatchNameListForMotionNode("../motion/bodyNameMatch.txt");
}



MotionNode* BVHparser::getRootNode()
{
	return rootNode;
}
string bvhpath_2_skelpath_(const char* path)
{
	string pathString; pathString = path;
	if(pathString.find(".bvh") != std::string::npos)
	{
		pathString.erase(pathString.find(".bvh"), 5);
	}
	return pathString;
}

string getFileName_(const char* path)
{
	string pathString; pathString = path;
	int cur = 0;
	while(pathString.find("/") != std::string::npos)
	{
		pathString.erase(0, pathString.find("/")+1);
	}

	return pathString.substr(0, pathString.find("."));
}

Eigen::Vector3d getShaft(float* childOffset)
{
	Eigen::Vector3d shaft((fabs(childOffset[0])>fabs(childOffset[1]) && fabs(childOffset[0])>fabs(childOffset[2]))? childOffset[0]/fabs(childOffset[0]): 0.0,
						(fabs(childOffset[1])>fabs(childOffset[0]) && fabs(childOffset[1])>fabs(childOffset[2]))? childOffset[1]/fabs(childOffset[1]) : 0.0,
						(fabs(childOffset[2])>fabs(childOffset[0]) && fabs(childOffset[2])>fabs(childOffset[1]))? childOffset[2]/fabs(childOffset[2]) : 0.0);
	return shaft;

}
// //node's size is fixed to 0.1, 0.1, 0.1
// #include <boost/filesystem.hpp>
// // #define PATH_MAX 100
// void BVHparser::writeSkelFile()
// {
// 	ofstream outfile;

// 	boost::filesystem::path dir("../data/skels/");

// 	if(!(boost::filesystem::exists(dir))){
// 	    // std::cout<<"Doesn't Exists"<<std::endl;
// 	    if (boost::filesystem::create_directory(dir))
// 	        std::cout << "../data/skels directory was successfully created !" << std::endl;
// 	}
// 	// if(n == 0)
// 	outfile.open("../data/skels/"+getFileName_(mPath)+".skel", ios::trunc);

// 	char resolved_path[PATH_MAX]; 
// 	realpath("../", resolved_path); 
// 	// printf("\n%s\n",resolved_path); 

// 	std::string absolutePathProject = resolved_path;

// 	skelFilePath = absolutePathProject+"/data/skels/"+getFileName_(mPath)+".skel";
// 	// else
// 	// 	outfile.open(bvhpath_2_skelpath_(mPath)+"Mocap.skel", ios::trunc);
// 	outfile << fixed << showpoint;
// 	outfile << setprecision(4);
// 	outfile << "<?xml version=\"1.0\" ?>" <<endl;
// 	outfile << "<skel version=\"1.0\">" <<endl;
// 	// outfile << "    <world name=\"world 1\">" <<endl;
// 	outfile << "		<physics>"<<endl;
//     outfile << "			<time_step>0.001</time_step>"<<endl;
//     outfile << "			<gravity>0 -9.81 0</gravity>"<<endl;
//     outfile << "			<collision_detector>bullet</collision_detector>"<<endl;
//     outfile << "		</physics>"<<endl;
//     outfile << "		<skeleton name=\""<< getFileName_(mPath) << "\">"<<endl;

//     //outfile << "		<skeleton name=\"biped\">"<<endl;
//     outfile << "			<transformation>0 0 0 0 0 0</transformation>"<<endl;
//     MotionNode *curNode = rootNode;
//     MotionNode *parentNode = nullptr;
//     int tabCount = 3;
//     float curOffset[3];
//     float parentOffset[3];
//     float childOffset[3];
//     float resize_factor = 0.01;
//     while(curNode != nullptr)
//     {

//     	parentNode = curNode->getParent();
//     	curNode->getOffset(curOffset);
//     	outfile << "			";
//     	outfile << "<body name=\"" <<curNode->getName() << "\">" <<endl;

//     	outfile << "			";

//     	// Eigen::AngleAxisd curOrientation = Eigen::AngleAxisd::Identity();
//     	while(parentNode!= nullptr)
//     	{
//     		parentNode->getOffset(parentOffset);
//     		for(int i = 0; i<3; i++)
//     		{
//     			curOffset[i] += parentOffset[i];
//     		}
//     		// curOrientation = parentNode->orientation * curOrientation;
//     		parentNode = parentNode->getParent();
//     	}
//     	outfile << "	<transformation>";

// 	   	outfile << curOffset[0]*resize_factor <<" "<< curOffset[1]*resize_factor<<" "<< curOffset[2]*resize_factor<<" ";

//     	if(curNode->getChilds().size()==1)
//     	{
//     		curNode->getChilds()[0]->getOffset(childOffset);
//     		Eigen::Vector3d shaft = getShaft(childOffset);

//     		if(shaft.norm() == 0.0)
//     		{
//     			outfile <<"0 0 0</transformation>" <<endl;
// 	   		}
//     		else
//     		{
//     			Eigen::Vector3d targetShaft(childOffset[0], childOffset[1], childOffset[2]);

//     			targetShaft.normalize();
//     			shaft = getShaft(childOffset);

//     			Eigen::VectorXd axis = shaft.cross(targetShaft);
//     			double cosAngle = shaft.dot(targetShaft);
//     			double angle = atan2(axis.norm(), cosAngle);
//     			axis.normalize();
//     			axis*= angle;
//     			// Eigen::AngleAxisd mNodeOrientation = Eigen::AngleAxisd(angle, axis);

//     			// curNode->orientation = mNodeOrientation;
//     			// axis = mNodeOrientation.axis();

//     			outfile<<axis[0]<<" "<<axis[1]<<" "<<axis[2]<<"</transformation>"<<endl;
//     		}
//     	}
//     	else
//     	{
//     		outfile <<"0 0 0</transformation>" <<endl;
//     	}

//     	outfile << "			";
//     	outfile << "	<inertia>" <<endl;
           
//     	outfile << "			";
//     	if(curNode->getName().compare(0, 4, "Head")==0)
//     	{
//     		outfile <<	"		<mass>"<<8.0<<"</mass>"<<endl;
//     	}
//     	else if(curNode->getName().compare(0, 5, "Spine")==0)
//     	{
//     		curNode->getChilds()[0]->getOffset(childOffset);
//        		outfile << "		<mass>";
//        		outfile <<5.0
//        		<<"</mass>"<<endl;
//     	}
//        	else if(curNode->getChilds().size()==1)
//        	{
//        		curNode->getChilds()[0]->getOffset(childOffset);
// 	    	outfile << "		<mass>"
// 	    	<<((fabs(childOffset[0])>fabs(childOffset[1]) && fabs(childOffset[0])>fabs(childOffset[2]))? fabs(childOffset[0]) : 5.0 )
// 	    	*((fabs(childOffset[1])>fabs(childOffset[0]) && fabs(childOffset[1])>fabs(childOffset[2]))? fabs(childOffset[1]) : 5.0 )
// 	    	*((fabs(childOffset[2])>fabs(childOffset[0]) && fabs(childOffset[2])>fabs(childOffset[1]))? fabs(childOffset[2]) : 5.0 ) *resize_factor
// 	    	<<"</mass>" <<endl;
//     	}
//     	else
//     	{
//     		outfile <<	"		<mass>"<<4.0<<"</mass>"<<endl;
//     	}
//     	outfile << "			";
//     	outfile << "		<offset>0.0 0 0.0</offset>" <<endl;   
                    
//     	outfile << "			";
//     	outfile << "	</inertia>" <<endl;

//     	outfile << "			";
//     	outfile << "	<visualization_shape>" <<endl;

//     	outfile << "			";

//    		curNode->getChilds()[0]->getOffset(childOffset);

//     	Eigen::VectorXd shaftScaled = getShaft(childOffset);
//     	double length = sqrt(pow(childOffset[0],2)+pow(childOffset[1],2)+pow(childOffset[2],2));
//     	shaftScaled*= length;
//    		outfile << "		<transformation>"
//   		<< (shaftScaled[0])/2.0 * resize_factor<<" "
//    		<< (shaftScaled[1])/2.0 * resize_factor<<" "
//    		<< (shaftScaled[2])/2.0 * resize_factor<<" "
//    		<<" 0 0 0</transformation>" <<endl;



//     	outfile << "			";
//     	outfile << "		<geometry>" <<endl;

//     	outfile << "			";
//     	outfile << "			<box>" <<endl;

//     	outfile << "			";
//     	if(curNode->getName().compare(0, 4, "Head")==0)
//     	{
//     		outfile <<	"				<size>0.3 0.3 0.3</size>"<<endl;
//     	}
//     	if(mBvhType == BVHType::BASKET && curNode->getName().compare(0, 5, "Spine")==0)
//     	{
//     		curNode->getOffset(curOffset);
//     		curNode->getChilds()[0]->getOffset(childOffset);
//        		outfile << "				<size>";

// 	       		outfile <<sqrt(pow(childOffset[0],2)+pow(childOffset[1],2)+pow(childOffset[2],2)) *resize_factor<<" " 
// 	       		<<"0.3 0.3</size>"<<endl;


//     	}
//        	else if(curNode->getChilds().size()==1)
//        	{
//        		curNode->getChilds()[0]->getOffset(childOffset);
//        		outfile << "				<size>";

//     		Eigen::Vector3d shaft = getShaft(childOffset);
//     		shaft *= sqrt(pow(childOffset[0],2)+pow(childOffset[1],2)+pow(childOffset[2],2));


//        		outfile <<((shaft[0] != 0)? fabs(shaft[0]) : 5.0 )*resize_factor<<" " 
//        		<<((shaft[1] != 0)? fabs(shaft[1]) : 5.0 )*resize_factor<<" "
//        		<<((shaft[2] != 0)? fabs(shaft[2]) : 5.0 )*resize_factor;
//        		outfile <<"</size>" <<endl;
//        	}
//        	else
//        	{

//        		outfile << "				<size>0.1 0.1 0.1</size>" <<endl;
//        	}

//     	outfile << "			";
//     	outfile << "			</box>" <<endl;

//     	outfile << "			";
//     	outfile << "		</geometry>" <<endl;

//     	outfile << "		<color> 1.0 1.0 1.0 </color>"<<endl;

//     	outfile << "			";
//     	outfile << "	</visualization_shape>" <<endl;



//     	outfile << "			";
//     	outfile << "	<collision_shape>" <<endl;

//     	outfile << "			";

//    		outfile << "		<transformation>"
//   		<< (shaftScaled[0])/2.0 * resize_factor<<" "
//    		<< (shaftScaled[1])/2.0 * resize_factor<<" "
//    		<< (shaftScaled[2])/2.0 * resize_factor<<" "
//    		<<" 0 0 0</transformation>" <<endl;



//     	outfile << "			";
//     	outfile << "		<geometry>" <<endl;

//     	outfile << "			";
//     	outfile << "			<box>" <<endl;

//     	outfile << "			";
//     	if(curNode->getName().compare(0, 4, "Head")==0)
//     	{
//     		outfile <<	"				<size>0.3 0.3 0.3</size>"<<endl;
//     	}
//     	if(mBvhType == BVHType::BASKET && curNode->getName().compare(0, 5, "Spine")==0)
//     	{
//     		curNode->getOffset(curOffset);
//     		curNode->getChilds()[0]->getOffset(childOffset);
//        		outfile << "				<size>";

// 	       		outfile <<sqrt(pow(childOffset[0],2)+pow(childOffset[1],2)+pow(childOffset[2],2)) *resize_factor<<" " 
// 	       		<<"0.3 0.3</size>"<<endl;

//     	}
//        	else if(curNode->getChilds().size()==1)
//        	{
//        		curNode->getChilds()[0]->getOffset(childOffset);
//        		outfile << "				<size>";

//     		Eigen::Vector3d shaft = getShaft(childOffset);
//     		shaft *= sqrt(pow(childOffset[0],2)+pow(childOffset[1],2)+pow(childOffset[2],2));


//        		outfile <<((shaft[0] != 0)? fabs(shaft[0]) : 5.0 )*resize_factor<<" " 
//        		<<((shaft[1] != 0)? fabs(shaft[1]) : 5.0 )*resize_factor<<" "
//        		<<((shaft[2] != 0)? fabs(shaft[2]) : 5.0 )*resize_factor;
//        		outfile <<"</size>" <<endl;
//        	}
//        	else
// 		{
//        		outfile << "				<size>0.1 0.1 0.1</size>" <<endl;
//        	}

//     	outfile << "			";
//     	outfile << "			</box>" <<endl;

//     	outfile << "			";
//     	outfile << "		</geometry>" <<endl;

//     	outfile << "			";
//     	outfile << "	</collision_shape>" <<endl;

//     	outfile << "			";
//     	outfile << "</body>" <<endl;
//     	curNode = curNode->getNextNode();
//     }

//     curNode = rootNode;
//     while(curNode != nullptr)
//     {
//     	outfile << "			";
//     	if(curNode->getName() == rootNode->getName())
//     		outfile << "<joint type=\"free\" name=\"j_"<< curNode->getName() << "\">" <<endl;
//     	else
//     		outfile << "<joint type=\"ball\" name=\"j_"<< curNode->getName() << "\">" <<endl;


//     	if(curNode->getName() != rootNode->getName())
//     	{
//     		outfile << "			";
//     		outfile << "	<transformation>0.0 0.0 0.0 0.0 0.0 0.0</transformation>" << endl;
//     	}

//     	outfile << "			";
//     	if(curNode->getName() == rootNode->getName())
//     		outfile << "	<parent>world</parent>" <<endl;
//     	else
//     		outfile << "	<parent>" << curNode->getParent()->getName() << "</parent>" <<endl;		

//     	outfile << "			";
//     	outfile << "	<child>" << curNode->getName() << "</child>" <<endl;

// 		// if(curNode->getName() != rootNode->getName())
//   //   	{
//   //   		outfile << "			";
//   //   		outfile << "	<axis_order>" << "xyz" << "</axis_order>" << endl;
//   //   	} 

//      	outfile << "			";
//     	if(curNode->getName() == rootNode->getName())
//     		outfile << "	<init_pos>0 0 0 0 0 0</init_pos>" <<endl;
//     	else
//     		outfile << "	<init_pos>0 0 0</init_pos>" <<endl;

//      	outfile << "			";
//     	if(curNode->getName() == rootNode->getName())     	
// 			outfile << "	<init_vel>0 0 0 0 0 0</init_vel>" <<endl;
// 		else
// 			outfile << "	<init_vel>0 0 0</init_vel>" <<endl;

//      	outfile << "			";    	
// 		outfile << "</joint>" <<endl;
// 		curNode = curNode->getNextNode();		
//     }
    

//     outfile << "		</skeleton>" <<endl;
//     // outfile << "	</world>" <<endl;
//     outfile << "</skel>" <<endl;
//     outfile.close();

//     // if(n == 0)
//     cout<< "Generating skel file is completed ! : "<< "../data/skels/"+getFileName_(mPath)+".skel" <<endl;
//     // else
//     // 	cout<< "Generating skel file is completed ! : "<< bvhpath_2_skelpath_(mPath)+"Mocap.skel" <<endl;
	
// }