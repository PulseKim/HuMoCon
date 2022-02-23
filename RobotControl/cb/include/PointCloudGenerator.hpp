#ifndef _POINT_CLOUD_GENERATOR_H
#define _POINT_CLOUD_GENERATOR_H

#include <k4a/k4atypes.h>
#include <k4a/k4a.h>
#include <vector>
#include "Utilities.h"

typedef union
{
  /** XYZ or array representation of vector. */
  struct _xyz
  {
    int16_t x; /**< X component of a vector. */
    int16_t y; /**< Y component of a vector. */
    int16_t z; /**< Z component of a vector. */
  } xyz;         /**< X, Y, Z representation of a vector. */
  int16_t v[3];    /**< Array representation of a vector. */
} PointCloudPixel_int16x3_t;

namespace Samples
{
  class PointCloudGenerator
  {
  public:
    PointCloudGenerator() = default;
    PointCloudGenerator(const k4a_calibration_t& sensorCalibration)
    {
    int depthWidth = sensorCalibration.depth_camera_calibration.resolution_width;
    int depthHeight = sensorCalibration.depth_camera_calibration.resolution_height;
    // Create transformation handle
    m_transformationHandle = k4a_transformation_create(&sensorCalibration);

    VERIFY(k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
      depthWidth,
      depthHeight,
      depthWidth * (int)sizeof(PointCloudPixel_int16x3_t),
      &m_pointCloudImage_int16x3), "Create Point Cloud Image failed!");
		}
    ~PointCloudGenerator(){
  	 if (m_transformationHandle != nullptr)
	    {
        k4a_transformation_destroy(m_transformationHandle);
        m_transformationHandle = nullptr;
	    }

	    if (m_pointCloudImage_int16x3 != nullptr)
	    {
        k4a_image_release(m_pointCloudImage_int16x3);
        m_pointCloudImage_int16x3 = nullptr;
	    }
    }

    void Update(k4a_image_t depthImage)
    {
    	VERIFY(k4a_transformation_depth_image_to_point_cloud(
        m_transformationHandle,
        depthImage,
        K4A_CALIBRATION_TYPE_DEPTH,
        m_pointCloudImage_int16x3), "Transform depth image to point clouds failed!");
    }
    const std::vector<k4a_float3_t>& GetCloudPoints(int step)
    {
    	int width = k4a_image_get_width_pixels(m_pointCloudImage_int16x3);
	    int height = k4a_image_get_height_pixels(m_pointCloudImage_int16x3);

	    // Current SDK transforms a depth map to point cloud only as int16 type.
	    // The point cloud conversion below to float is relatively slow.
	    // It would be better for the SDK to provide this functionality directly.

	    const auto pointCloudImageBufferInMM = (PointCloudPixel_int16x3_t*)k4a_image_get_buffer(m_pointCloudImage_int16x3);

	    m_cloudPoints.resize(width * height / (step * step));
	    size_t cloudPointsIndex = 0;
	    for (int h = 0; h < height; h+= step)
	    {
	      for (int w = 0; w < width; w += step)
	      {
	        int pixelIndex = h * width + w;

	        // When the point cloud is invalid, the z-depth value is 0.
	        if (pointCloudImageBufferInMM[pixelIndex].xyz.z > 0)
	        {
	          const float MillimeterToMeter = 0.001f;
	          k4a_float3_t positionInMeter = {
	            static_cast<float>(pointCloudImageBufferInMM[pixelIndex].v[0])* MillimeterToMeter,
	            static_cast<float>(pointCloudImageBufferInMM[pixelIndex].v[1])* MillimeterToMeter,
	            static_cast<float>(pointCloudImageBufferInMM[pixelIndex].v[2])* MillimeterToMeter };

	          m_cloudPoints[cloudPointsIndex++] = positionInMeter;
	        }
	      }
	    }
	    m_cloudPoints.resize(cloudPointsIndex);
	    return m_cloudPoints;
    }

  private:
    k4a_transformation_t m_transformationHandle = nullptr;
    k4a_image_t m_pointCloudImage_int16x3 = nullptr;
    std::vector<k4a_float3_t> m_cloudPoints;
  };
}

#endif