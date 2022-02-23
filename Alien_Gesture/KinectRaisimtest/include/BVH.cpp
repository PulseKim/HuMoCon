 
#include "BVH.h"

#include <fstream>
#include <sstream>
#include <iostream>
using namespace std;

BVH::BVH()
{
	load_success = false;
}

BVH::BVH(const char* filename)
{
	std::string filenamestr(filename);
	load_success = false;
	load(filename);
}

BVH::~BVH()
{
	clear();
}

void BVH::clear()
{
	nameToJoint.clear();
	jointList.clear();
	channels.clear();
	load_success = false;
	rootJoint = NULL;
}

void BVH::load(const std::string& bvh_file_name)
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

JOINT* BVH::getJoint(const std::string name)
{
	std::map<std::string, JOINT*>::const_iterator i = nameToJoint.find(name);
	// std::cout << i->second->name << std::endl;
	JOINT* j = (i!=nameToJoint.end()) ? (*i).second:NULL; 
	if(j==NULL){
		std::cout<<"JOINT <"<<name<<"> is not loaded!\n";
	}
	return j;
}


float* BVH::getMotionDataPtr(int frame_no)
{
	frame_no%=motionData.num_frames;
	return motionData.data+frame_no*motionData.num_motion_channels;
}

void BVH::matrixMoveJoint(JOINT* joint, float* mdata, float scale)
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


void BVH::matrixMoveTo(unsigned frame, float scale)
{
    // we calculate motion data's array start index for a frame
    // unsigned start_index = frame * motionData.num_motion_channels;

    // recursively transform skeleton
    matrixMoveJoint(rootJoint, getMotionDataPtr(frame), scale);
}


void BVH::quaternionMoveJoint(JOINT* joint, float* mdata, float scale)
{
    // translate to the offset and set the rotation to identity
    joint->transform.translation = glm::vec3(joint->offset.x * scale,
                                             joint->offset.y * scale,
                                             joint->offset.z * scale);
    joint->transform.quaternion = glm::quat();


    for (uint i = 0; i < joint->channels.size(); i++)
    {        		
        // extract value from motion data        
        CHANNEL *channel = joint->channels[i];
		float value = mdata[channel->index];
		switch(channel->type){
		case X_POSITION:	
			joint->transform.translation.x += value * scale;
			break;

		case Y_POSITION:  
			joint->transform.translation.y += value * scale;
			break;

		case Z_POSITION: 
			joint->transform.translation.z += value * scale;
			break;

		case X_ROTATION:
        {            
			joint->transform.quaternion = joint->transform.quaternion
				* glm::angleAxis(value, glm::vec3(1.f, 0.f, 0.f));
			break;
        }
		case Y_ROTATION:
        {	
			joint->transform.quaternion = joint->transform.quaternion
				* glm::angleAxis(value, glm::vec3(0.f, 1.f, 0.f));
			break;
        }
		case Z_ROTATION:        
        {		
			joint->transform.quaternion = joint->transform.quaternion
				* glm::angleAxis(value, glm::vec3(0.f, 0.f, 1.f));
			break;
        }
		}
	}

    // apply parent's transfomation matrix to this joint to make the transformation global
    if (joint->parent != NULL) {

		//Translation: begin from parent's end point 
		// and translate according to its global orientation
		joint->transform.translation = joint->parent->transform.translation
			+ (joint->parent->transform.quaternion * joint->transform.translation);

		//Global orientation of joint
		//is cross product of its orientation with the parent's orientation
		joint->transform.quaternion =
			glm::cross(joint->parent->transform.quaternion, joint->transform.quaternion);

    }

    // do the same to all children
    for(std::vector<JOINT*>::iterator child = joint->children.begin(); child != joint->children.end(); ++child)
        quaternionMoveJoint(*child, mdata, scale);
}

void BVH::quaternionMoveTo(unsigned frame, float scale)
{
    // we calculate motion data's array start index for a frame
    unsigned start_index = frame * motionData.num_motion_channels;

    // recursively transform skeleton
    quaternionMoveJoint(rootJoint, getMotionDataPtr(frame), scale);
}