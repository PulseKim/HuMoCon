#ifndef  _BVH_HPP_
#define  _BVH_HPP_

#define GLM_ENABLE_EXPERIMENTAL
#include <stdio.h>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/quaternion.hpp"
#include "GL/glut.h"
#include "glm/ext.hpp"
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
using namespace std;

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
	BVH(){
		load_success = false;
	}
	BVH(const char* filename){
		std::string filenamestr(filename);
		load_success = false;
		load(filename);
	}
	~BVH(){
		clear();
	}

    // load a bvh file
	void load(const std::string& bvh_file_name)
	{
	#define BUFFER_LENGTH 1024*4
	std::ifstream file;
	char line[BUFFER_LENGTH];
	char* token;
	char separater[] = " :,\t";
	std::vector<JOINT*> joint_stack;
	JOINT* joint = NULL;
	JOINT* new_joint = NULL;
	bool is_site = false;
	double x, y ,z;	
	clear();

	file.open(bvh_file_name, std::ios::in);

	if(!file)
	{
		cerr << "Cannot open "<<bvh_file_name<<endl; exit(1);
	}		


	while (! file.eof())
	{
		if (file.eof()) goto bvh_error;

		file.getline( line, BUFFER_LENGTH );
		token = strtok( line, separater );

		if ( token == NULL )  continue;

		if (strncmp( token, "{",1) == 0 )
		{
			joint_stack.push_back( joint );
			joint = new_joint;
			continue;
		}

		if ( strncmp( token, "}",1) == 0 )
		{
			joint = joint_stack.back();
			joint_stack.pop_back();
			is_site = false;
			continue;
		}

		if ( ( strncmp( token, "ROOT" ,4) == 0 ) ||
		     ( strncmp( token, "JOINT" ,5) == 0 ) )
		{
			new_joint = new JOINT();			
			new_joint->parent = joint;
			new_joint->is_site = false;
			new_joint->offset.x = 0.0;  new_joint->offset.y = 0.0;  new_joint->offset.z = 0.0;	
			//add to joint collection
			jointList.push_back( new_joint );
			//add children
			if ( joint )
				joint->children.push_back( new_joint );
			else
				rootJoint = new_joint; //the root
			token = strtok(NULL, separater );
			while ( *token == ' ')  token ++;

			for(int i=0; i< strlen(token)-1; ++i)
				new_joint->name.append(1, token[i]);
			nameToJoint[new_joint->name] = new_joint;
			continue;
		}

		if ( ( strncmp( token, "End", 3) == 0 ) )
		{			
			new_joint = new JOINT();			
			new_joint->parent = joint;
			new_joint->name = joint->name + "EndSite";
			new_joint->is_site = true;
			new_joint->channels.clear();
			//add children 
			if ( joint )
				joint->children.push_back( new_joint );
			else
				rootJoint = new_joint; //can an endsite be root? -cuizh
			//add to joint collection
			jointList.push_back( new_joint );
			nameToJoint[new_joint->name] = new_joint;
			is_site = true;			
			continue;
		}

		if ( strncmp( token, "OFFSET", 6) == 0 )
		{
			token = strtok( NULL, separater );
			x = token ? atof( token ) : 0.0;
			token = strtok( NULL, separater );
			y = token ? atof( token ) : 0.0;
			token = strtok( NULL, separater );
			z = token ? atof( token ) : 0.0;			
			joint->offset.x = x;
			joint->offset.y = y;
			joint->offset.z = z;
			continue;
		}

		if ( strcmp( token, "CHANNELS" ) == 0 )
		{
			token = strtok( NULL, separater );
			joint->channels.resize( token ? atoi( token ) : 0 );

			for ( uint i=0; i<joint->channels.size(); i++ )
			{
				CHANNEL* channel = new CHANNEL();
				channel->index = channels.size();
				channels.push_back(channel);
				joint->channels[i] = channel;

				token = strtok( NULL, separater );
				if ( strncmp( token, "Xrotation" ,9) == 0 ){
					channel->type = X_ROTATION;
				}
				else if ( strncmp( token, "Yrotation",9 ) == 0 ){
					channel->type = Y_ROTATION;
				}
				else if ( strncmp( token, "Zrotation",9 ) == 0 ){
					channel->type = Z_ROTATION;
				}
				else if ( strncmp( token, "Xposition" ,9) == 0 ){
					channel->type = X_POSITION;
				}
				else if ( strncmp( token, "Yposition" ,9) == 0 ){
					channel->type = Y_POSITION;
				}
				else if ( strncmp( token, "Zposition" ,9) == 0 ){
					channel->type = Z_POSITION;
				}
			}			
		}

		if ( strncmp( token, "MOTION", 6) == 0 )
			break;
	}
 
	file.getline( line, BUFFER_LENGTH );
	token = strtok( line, separater );
	if ( strcmp( token, "Frames" ) != 0 )  goto bvh_error;
	token = strtok( NULL, separater );
	if ( token == NULL )  goto bvh_error;
	motionData.num_frames = atoi( token );

	file.getline( line, BUFFER_LENGTH );
	token = strtok( line, ":" );
	if ( strcmp( token, "Frame Time" ) != 0 )  goto bvh_error;
	token = strtok( NULL, separater );
	if ( token == NULL )  goto bvh_error;
	motionData.frame_time = atof( token );

	motionData.num_motion_channels = channels.size();
	motionData.data = new float[motionData.num_frames*motionData.num_motion_channels];

	for (uint i=0; i<motionData.num_frames; i++)
	{
		file.getline( line, BUFFER_LENGTH );
		token = strtok( line, separater );
		for ( uint j=0; j<motionData.num_motion_channels; j++ )
		{
			if (token == NULL)
				goto bvh_error;
			motionData.data[i*motionData.num_motion_channels+j] = atof(token);
			token = strtok( NULL, separater );
		}
	}

	file.close();
	load_success = true;
	return;

bvh_error:
	file.close();
} 	

	// is the bvh file successfully loaded? 
	bool IsLoadSuccess() const { return load_success; }

    // get root joint
	JOINT* getRootJoint(){ return rootJoint;}
	
    // get the JOINT pointer from joint name 
	JOINT* getJoint(const std::string name)
	{
		std::map<std::string, JOINT*>::const_iterator i = nameToJoint.find(name);
		// std::cout << i->second->name << std::endl;
		JOINT* j = (i!=nameToJoint.end()) ? (*i).second:NULL; 
		if(j==NULL){
			std::cout<<"JOINT <"<<name<<"> is not loaded!\n";
		}
		return j;
	}

	float* getMotionDataPtr(int frame_no){
		frame_no%=motionData.num_frames;
		return motionData.data+frame_no*motionData.num_motion_channels;
	}

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
	
    // calculate JOINT's transformation for specified frame using matrix	
	void matrixMoveTo(unsigned frame, float scale)
	{
	    // we calculate motion data's array start index for a frame
	    unsigned start_index = frame * motionData.num_motion_channels;

	    // recursively transform skeleton
	    matrixMoveJoint(rootJoint, getMotionDataPtr(frame), scale);
	}	

private:    
    void matrixMoveJoint(JOINT* joint, float* mdata, float scale)
    {
	    // translate identity matrix to this joint's offset parameters
	    joint->matrix = glm::translate(glm::mat4(1.0),
	                                   glm::vec3(joint->offset.x*scale,
	                                             joint->offset.y*scale,
	                                             joint->offset.z*scale));
	    // transform based on joint channels
	    // end site will have channels.size() == 0, won't be an issue

	    for(uint i = 0; i < joint->channels.size(); i++)
	    {        		
	        // extract value from motion data        
	        CHANNEL *channel = joint->channels[i];
			float value = mdata[channel->index];
			switch(channel->type){
			case X_POSITION:
				joint->matrix = glm::translate(joint->matrix, glm::vec3(value*scale, 0, 0));
				break;
			case Y_POSITION:        
				joint->matrix = glm::translate(joint->matrix, glm::vec3(0, value*scale, 0));
				break;
			case Z_POSITION:        
				joint->matrix = glm::translate(joint->matrix, glm::vec3(0, 0, value*scale));
				break;
			case X_ROTATION:
				joint->matrix = glm::rotate(joint->matrix, glm::radians(value), glm::vec3(1, 0, 0));
				break;
			case Y_ROTATION:
				joint->matrix = glm::rotate(joint->matrix, glm::radians(value), glm::vec3(0, 1, 0));
				break;
			case Z_ROTATION:        
				joint->matrix = glm::rotate(joint->matrix, glm::radians(value), glm::vec3(0, 0, 1));
				break;
			}
		}

	    // apply parent's local transfomation matrix to this joint's LTM (local tr. mtx. :)
		// watch out for the order
	    if( joint->parent != NULL )
	        joint->matrix = joint->parent->matrix * joint->matrix;

	    // do the same to all children
	    for(std::vector<JOINT*>::iterator child = joint->children.begin(); child != joint->children.end(); ++child)
	        matrixMoveJoint(*child, mdata,scale);
	}

	// write a joint info to bvh hierarchy part
	void writeJoint(JOINT* joint, std::ostream& stream, int level);	

	// clean up stuff, used in destructor
	void clear(){
		nameToJoint.clear();
		jointList.clear();
		channels.clear();
		load_success = false;
		rootJoint = NULL;
	}

};

#endif