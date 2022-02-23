#ifndef MotionNode_H
#define MotionNode_H

#include <Eigen/Dense>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <vector>
#include <string>
using namespace std;

enum BVHType
{
	CMU,
	BASKET,
	DOG
};


class MotionNode
{
	string name;
	bool isRoot;
	bool isEnd;
	MotionNode* parent;
	vector<MotionNode*> childs;
	float offset[3];
	string axisOrder;
	int channelNum;
	MotionNode* next;
public :
	float **data;
	MotionNode();
	MotionNode* getParent();
	vector<MotionNode*> getChilds();
	void setParent(MotionNode* pnode);
	void addChild(MotionNode* cnode);
	void setRoot();
	void setEnd();
	void setName(string mname);
	void setAxisOrder(string maxisOrder);
	void setOffset(float x, float y, float z);
	void setNext(MotionNode *nextNode);
	void setChannelNum(int mchannelNum);
	bool checkEnd();
	string getName();
	string getName_std();	//return for the name in body_name_match.txt 
	string getAxisOrder();
	void getOffset(float* outOffset);
	float getOffset(int index);
	int getChannelNum();
	MotionNode* getNextNode();
	void initData(int frames);
	bool ContainedInModel();
	Eigen::AngleAxisd orientation;
	vector<string> match_name_list;
};
#endif

#ifndef BVHparser_H
#define BVHparser_H
class BVHparser
{
public :
	const char* mPath;
	MotionNode* rootNode;
	bool lowerBodyOnly;
	bool upperBodyOnly;
	// BVHparser(const char* path);
	// BVHparser(const char* path, const char* match_name_path);
	BVHparser(const char* path, BVHType bvhType = BVHType::BASKET);
	int frames;
	float frameTime;
	MotionNode* getRootNode();
	void initMatchNameListForMotionNode(const char* path);
	// void writeSkelFile();
	std::string skelFilePath;
	double scale;
	BVHType mBvhType;
};
string getFileName_(const char* path);
Eigen::Vector3d QuaternionToAngleAxis(Eigen::Quaterniond qt);
Eigen::Quaterniond AngleAxisToQuaternion(Eigen::Vector3d angleAxis);
#endif