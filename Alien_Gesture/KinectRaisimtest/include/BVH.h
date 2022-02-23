#ifndef  _BVH_H_
#define  _BVH_H_

#define GLM_ENABLE_EXPERIMENTAL
#include <stdio.h>
#include <iostream>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/quaternion.hpp"
#include "GL/glut.h"
#include "glm/ext.hpp"

#include <vector>
#include <string>
#include <map>


enum  ChannelEnum
{
	X_ROTATION, Y_ROTATION, Z_ROTATION,
	X_POSITION, Y_POSITION, Z_POSITION
};

typedef unsigned int uint;

typedef struct{
	ChannelEnum type;
	uint index;
} CHANNEL;

typedef struct
{
	float x, y, z;
} OFFSET;

typedef struct JOINT JOINT;

struct RigidTransform {
    glm::vec3 translation;
    glm::quat quaternion;
};

struct JOINT
{
	std::string name;                     // joint name
	JOINT* parent;                        // joint parent	
	OFFSET offset;                        // offset data
	std::vector<CHANNEL*> channels;
	std::vector<JOINT*> children;         // joint's children	
	bool is_site;                         // if it is end site

    RigidTransform transform;             // transformation stored by a translation and a quaternion (for animation)

    glm::mat4 matrix;                     // transformation stored by 4x4 matrix (for reference only)
};

typedef struct
{
	unsigned int num_frames;              // number of frames
	unsigned int num_motion_channels;     // number of motion channels (of all joints per frame)
	float* data;                          // motion float data array
	float frame_time;                     // time per frame; FPS = 1/frame_time
} MOTION;

class BVH
{
private:
	JOINT* rootJoint;
	MOTION motionData;
	bool load_success;
	std::map<std::string,JOINT*> nameToJoint;
	std::vector<JOINT*> jointList;             // this list stores all pointers to joints
	std::vector<CHANNEL*> channels;

public:
	BVH();
	BVH(const char* filename);
	~BVH();

private:
	// write a joint info to bvh hierarchy part
	void writeJoint(JOINT* joint, std::ostream& stream, int level);	

	// clean up stuff, used in destructor
	void clear();

public:
	/************************************************************************** 
	 * functions for basic I/O - only loading is needed
	 **************************************************************************/
	
    // load a bvh file
	void load(const std::string& filename);  	

	// is the bvh file successfully loaded? 
	bool IsLoadSuccess() const { return load_success; }

public:
	/************************************************************************** 
	 * functions to retrieve bvh data 
	 **************************************************************************/
	
    // get root joint
	JOINT* getRootJoint(){ return rootJoint;}
	
    // get the JOINT pointer from joint name 
	JOINT* getJoint(const std::string name);	
	
    // get the pointer to mation data at frame_no
	// NOTE: frame_no is treated as frame_no % total_frames
	float* getMotionDataPtr(int frame_no);

	float getFrameY(int frame_no){
		frame_no%=motionData.num_frames;
		return motionData.data[frame_no*motionData.num_motion_channels + 5];
	}
	
    // get a pointer to the root joint 
	const JOINT* getRootJoint() const { return rootJoint; }
	
    // get the list of the joint
	const std::vector<JOINT*> getJointList() const {return jointList; }
	
    // get the number of frames 
	unsigned getNumFrames() const { return motionData.num_frames; }
	
    // get time per frame
	float getFrameTime() const { return motionData.frame_time; }	
	
public:
    // calculate JOINT's transformation for specified frame using matrix	
	void matrixMoveTo(unsigned frame, float scale);

	std::map<std::string,JOINT*>  mapGetter();

    // calculate JOINT's transformation for specified frame using quaternion
    void quaternionMoveTo(unsigned frame, float scale);

private:    
	void quaternionMoveJoint(JOINT* joint, float* mdata, float scale);	
    void matrixMoveJoint(JOINT* joint, float* mdata, float scale);	    	

};

#endif