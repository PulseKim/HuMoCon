#ifndef _MAPPING_FUNCTIONS_H
#define _MAPPING_FUNCTIONS_H

#include <k4arecord/playback.h>
#include <k4a/k4a.h>
#include <k4abt.h>
#include <Eigen/Core>
#include "raisim/World.hpp"


struct InputSettings
{
  k4a_depth_mode_t DepthCameraMode = K4A_DEPTH_MODE_NFOV_UNBINNED;
  bool CpuOnlyMode = false;
  bool Offline = false;
  std::string FileName;
};

class KinectMapper()
{
public:
	KinectMapper(float scaler_, float z_offset_, k4abt_joint_id_t standard_, std::vector<k4abt_joint_id_t> desired_joint_)
	{
		desired_joint = desired_joint_;
		standard = standard_;
	  scaler = scaler_;
	  z_offset = z_offset_;
		deviceFlag = false;
	}
	~KinectMapper(){}


	void kinect_init()
  {
    InputSettings inputSettings;
    SetFromDevice(inputSettings);
  }

  void SetFromDevice(InputSettings inputSettings) {
    VERIFY(k4a_device_open(0, &device), "Open K4A Device failed!");

    // Start camera. Make sure depth camera is enabled.
    k4a_device_configuration_t deviceConfig = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    deviceConfig.depth_mode = inputSettings.DepthCameraMode;
    deviceConfig.color_resolution = K4A_COLOR_RESOLUTION_OFF;
    VERIFY(k4a_device_start_cameras(device, &deviceConfig), "Start K4A cameras failed!");

    // Get calibration information
    k4a_calibration_t sensorCalibration;
    VERIFY(k4a_device_get_calibration(device, deviceConfig.depth_mode, deviceConfig.color_resolution, &sensorCalibration),
        "Get depth camera calibration failed!");
    int depthWidth = sensorCalibration.depth_camera_calibration.resolution_width;
    int depthHeight = sensorCalibration.depth_camera_calibration.resolution_height;

    // Create Body Tracker
    k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
    tracker_config.processing_mode = inputSettings.CpuOnlyMode ? K4ABT_TRACKER_PROCESSING_MODE_CPU : K4ABT_TRACKER_PROCESSING_MODE_GPU;
    VERIFY(k4abt_tracker_create(&sensorCalibration, tracker_config, &tracker), "Body tracker initialization failed!");
  }

  void CloseDevice(){
    std::cout << "Finished body tracking processing!" << std::endl;

    k4abt_tracker_shutdown(tracker);
    k4abt_tracker_destroy(tracker);

    k4a_device_stop_cameras(device);
    k4a_device_close(device);
  }

  void GetFromDevice() {
    k4a_capture_t sensorCapture = nullptr;
    k4a_wait_result_t getCaptureResult = k4a_device_get_capture(device, &sensorCapture, 0); // timeout_in_ms is set to 0

    if (getCaptureResult == K4A_WAIT_RESULT_SUCCEEDED)
    {
        // timeout_in_ms is set to 0. Return immediately no matter whether the sensorCapture is successfully added
        // to the queue or not.
        k4a_wait_result_t queueCaptureResult = k4abt_tracker_enqueue_capture(tracker, sensorCapture, 0);

        // Release the sensor capture once it is no longer needed.
        k4a_capture_release(sensorCapture);

        if (queueCaptureResult == K4A_WAIT_RESULT_FAILED)
        {
            std::cout << "Error! Add capture to tracker process queue failed!" << std::endl;
            CloseDevice();
        }
    }
    else if (getCaptureResult != K4A_WAIT_RESULT_TIMEOUT)
    {
        std::cout << "Get depth capture returned error: " << getCaptureResult << std::endl;
        CloseDevice();
    }

    // Pop Result from Body Tracker
    k4abt_frame_t bodyFrame = nullptr;
    k4a_wait_result_t popFrameResult = k4abt_tracker_pop_result(tracker, &bodyFrame, 0); // timeout_in_ms is set to 0
    
    if (popFrameResult == K4A_WAIT_RESULT_SUCCEEDED)
    {
        /************* Successfully get a body tracking result, process the result here ***************/
        // VisualizeResult(bodyFrame, window3d, depthWidth, depthHeight);
        // uint32_t numBodies = k4abt_frame_get_num_bodies(bodyFrame);
        uint32_t numBodies = 1;
        for (uint32_t i = 0; i < numBodies; i++)
        {
          k4abt_body_t body;
          VERIFY(k4abt_frame_get_body_skeleton(bodyFrame, i, &body.skeleton), "Get skeleton from body frame failed!");
          body.id = k4abt_frame_get_body_id(bodyFrame, 0);

          k4a_float3_t standard_joint;
          std::vector<k4a_float3_t> joint_pose;

          for(int i =0 ; i < desired_joint.size(); ++i)
          {
          	if(desired_joint[i] == standard)
          		standard_joint = body.skeleton.joints[desired_joint[i]].position;
          	joint_pose.push_back(body.skeleton.joints[desired_joint[i]].position);
          }

          for(int i = 0; i < joint_pose.size();++i)
          {
          	points.push_back(Eigen::Vector3d((joint_pose[i].xyz.x - standard_joint.xyz.x), (joint_pose[i].xyz.z - standard_joint.xyz.z), (joint_pose[i].xyz.y - standard_joint.xyz.y)));
          }

          for(int i =0 ; i < points.size();++i){
            points[i][2] +=z_offset;
          }
          deviceFlag = true;
        }
        //Release the bodyFrame    
        k4abt_frame_release(bodyFrame);
    }
  }

  std::vector<Eigen::Vector3d> PositionGetter()
  {
  	return points;
  }
  bool flagGetter()
  {
  	return deviceFlag;
  }


private:
  std::vector<Eigen::Vector3d> points;
  std::vector<k4abt_joint_id_t> desired_joint;
  k4abt_joint_id_t standard;
  bool deviceFlag;
  float scaler;
  float z_offset;

};

#endif