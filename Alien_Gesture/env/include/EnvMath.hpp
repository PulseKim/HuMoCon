#ifndef _ENV_MATH_H
#define _ENV_MATH_H

#include <iostream>
#include <Eigen/Core>
#include "math.h"
#include <k4a/k4a.h>
#include <k4abt.h>


#define deg2rad(ang) ((ang) * M_PI / 180.0)
#define rad2deg(ang) ((ang) * 180.0 / M_PI)

Eigen::Vector3d clipper(Eigen::Vector3d speed_limit, Eigen::Vector3d speed_current, Eigen::Vector3d task, Eigen::Vector3d prev_foot, float control_dt_)
{
  Eigen::Vector3d abs_speed_current = speed_current.cwiseAbs();
  if(abs_speed_current[0] < speed_limit[0] && abs_speed_current[1] < speed_limit[1] && abs_speed_current[2] < speed_limit[2])
    return task;
  Eigen::Vector3d clipped_speed = speed_current;
  for(int i =0 ; i< 3; ++i){
    if(abs_speed_current[i] > speed_limit[i])
    {
      clipped_speed[i] = copysign(speed_limit[i], speed_current[i]);
    }
  }
  Eigen::Vector3d clipped_task = prev_foot + clipped_speed * control_dt_;
  return prev_foot;
}

float angle_finder(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d c)
{
  float A = (b-c).norm();
  float B = (c-a).norm();
  float C = (a-b).norm();
  if(A < 1e-5 || B < 1e-5)
    return 0;
  float angle = acos((C*C + A*A - B*B)/(2 * C * A));
  // if(angle > deg2rad(180))
  //   angle -= deg2rad(180);
  return angle;
}

float angle_finder(float A, float B, float C)
{
  if(A < 1e-5 || B < 1e-5)
    return 0;
  float angle = acos((C*C + A*A - B*B)/(2 * C * A));
  // if(angle > deg2rad(180))
  //   angle -= deg2rad(180);
  return angle;
}


void quat2euler(Eigen::Vector4d quat, Eigen::Vector3d& axis){
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
}

Eigen::Vector3d LineSegment(Eigen::Vector3d p, Eigen::Vector3d q)
{
  Eigen::Vector3d sub = p - q;
  return sub;
}

Eigen::VectorXd EquationPlane(Eigen::Vector3d p, Eigen::Vector3d q, Eigen::Vector3d r){
  Eigen::VectorXd plane;
  plane.setZero(4);
  Eigen::Vector3d normal_vec;
  Eigen::Vector3d vec1 = p - r;
  Eigen::Vector3d vec2 = q - r;
  normal_vec = vec1.cross(vec2);
  plane.segment(0, 3) = normal_vec;
  double d = normal_vec.dot(-r);
  plane[3] = d;
  return plane;
}
Eigen::VectorXd ParallelPlane(Eigen::Vector3d normal_vec, Eigen::Vector3d p){
  Eigen::VectorXd plane;
  plane.setZero(4);
  plane.segment(0, 3) = normal_vec;
  double d = normal_vec.dot(-p);
  plane[3] = d;
  return plane;
}


Eigen::Vector3d Projection(Eigen::VectorXd plane, Eigen::Vector3d point){
  Eigen::Vector3d projected;
  double t0 = -(plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3]) / (plane[0]* plane[0] + plane[1]* plane[1] +plane[2] * plane[2]);
  projected[0] = point[0] + plane[0] * t0;
  projected[1] = point[1] + plane[1] * t0;
  projected[2] = point[2] + plane[2] * t0;
  return projected;
}

float dot2D(Eigen::Vector3d r, Eigen::Vector3d l)
{
  return (r[0] * l[0] + r[1] * l[1]);

}
float cross2D(Eigen::Vector3d r, Eigen::Vector3d l)
{
  return (r[0] * l[1] - r[1] * l[0]);
}

float dist2D(Eigen::Vector3d r, Eigen::Vector3d l)
{
  return sqrt(pow(r[0] - l[0], 2) + pow(r[1] - l[1],2));
}


float distToSegment(Eigen::Vector3d p, Eigen::Vector3d s, Eigen::Vector3d e)
{ 
  float dx = e[0] - s[0];
  float dy = e[1] - s[1];
  if((dx == 0) && (dy == 0))
  {
      // It's a point not a line segment.
      dx = p[0] - s[0];
      dy = p[1] - s[1];
      return dx * dx + dy * dy;
  }

  Eigen::Vector3d sp = p - s;
  Eigen::Vector3d se = e - s;
  Eigen::Vector3d es = s - e;
  Eigen::Vector3d ep = p - e;

  if(dot2D(sp, se) * dot2D(es,ep) >= 0)
  {
    return abs(cross2D(sp, se) / dist2D(s,e));
  }
  else
  {
    return std::min(dist2D(s,p), dist2D(e,p));
  }
}



void FFT(short int dir,long m,double *x,double *y)
{
  long n,i,i1,j,k,i2,l,l1,l2;
  double c1,c2,tx,ty,t1,t2,u1,u2,z;
  /* Calculate the number of points */
  n = 1;
  for (i=0;i<m;i++)
    n *= 2;
  /* Do the bit reversal */
  i2 = n >> 1;
  j = 0;
  for (i=0;i<n-1;i++) {
	  if (i < j) {
      tx = x[i];
      ty = y[i];
      x[i] = x[j];
      y[i] = y[j];
      x[j] = tx;
      y[j] = ty;
	  }
	  k = i2;
	  while (k <= j) {
      j -= k;
      k >>= 1;
	  }
	  j += k;
  }

  /* Compute the FFT */
  c1 = -1.0;
  c2 = 0.0;
  l2 = 1;
  for (l=0;l<m;l++) {
    l1 = l2;
    l2 <<= 1;
    u1 = 1.0;
    u2 = 0.0;
    for (j=0;j<l1;j++) {
      for (i=j;i<n;i+=l2) {
        i1 = i + l1;
        t1 = u1 * x[i1] - u2 * y[i1];
        t2 = u1 * y[i1] + u2 * x[i1];
        x[i1] = x[i] - t1;
        y[i1] = y[i] - t2;
        x[i] += t1;
        y[i] += t2;
      }
      z =  u1 * c1 - u2 * c2;
      u2 = u1 * c2 + u2 * c1;
      u1 = z;
    }
    c2 = sqrt((1.0 - c1) / 2.0);
    if (dir == 1)
      c2 = -c2;
    c1 = sqrt((1.0 + c1) / 2.0);
  }

  /* Scaling for forward transform */
  if (dir == 1) {
    for (i=0;i<n;i++) {
      x[i] /= n;
      y[i] /= n;
    }
  }
}

std::vector<double> FFT_Mag(short int dir,long m,double *x,double *y)
{
  long n,i,i1,j,k,i2,l,l1,l2;
  double c1,c2,tx,ty,t1,t2,u1,u2,z;
  /* Calculate the number of points */
  n = 1;
  for (i=0;i<m;i++)
    n *= 2;
  /* Do the bit reversal */
  i2 = n >> 1;
  j = 0;
  for (i=0;i<n-1;i++) {
	  if (i < j) {
      tx = x[i];
      ty = y[i];
      x[i] = x[j];
      y[i] = y[j];
      x[j] = tx;
      y[j] = ty;
	  }
	  k = i2;
	  while (k <= j) {
      j -= k;
      k >>= 1;
	  }
	  j += k;
  }

  /* Compute the FFT */
  c1 = -1.0;
  c2 = 0.0;
  l2 = 1;
  for (l=0;l<m;l++) {
    l1 = l2;
    l2 <<= 1;
    u1 = 1.0;
    u2 = 0.0;
    for (j=0;j<l1;j++) {
      for (i=j;i<n;i+=l2) {
        i1 = i + l1;
        t1 = u1 * x[i1] - u2 * y[i1];
        t2 = u1 * y[i1] + u2 * x[i1];
        x[i1] = x[i] - t1;
        y[i1] = y[i] - t2;
        x[i] += t1;
        y[i] += t2;
      }
      z =  u1 * c1 - u2 * c2;
      u2 = u1 * c2 + u2 * c1;
      u1 = z;
    }
    c2 = sqrt((1.0 - c1) / 2.0);
    if (dir == 1)
      c2 = -c2;
    c1 = sqrt((1.0 + c1) / 2.0);
  }

  /* Scaling for forward transform */
  if (dir == 1) {
    for (i=0;i<n;i++) {
      x[i] /= n;
      y[i] /= n;
    }
  }

	std::vector<double> mag(n/2);
	for(int i=0;i<n/2;i++) {
    mag[i] = 2 * sqrt(pow(x[i],2)+pow(y[i],2));
	}
	return mag;
}

float k4a_Angle(k4a_float3_t A, k4a_float3_t B, k4a_float3_t C)
{
    k4a_float3_t AbVector;
    k4a_float3_t BcVector;

    AbVector.xyz.x = B.xyz.x - A.xyz.x;
    AbVector.xyz.y = B.xyz.y - A.xyz.y;
    AbVector.xyz.z = B.xyz.z - A.xyz.z;

    BcVector.xyz.x = C.xyz.x - B.xyz.x;
    BcVector.xyz.y = C.xyz.y - B.xyz.y;
    BcVector.xyz.z = C.xyz.z - B.xyz.z;

    float AbNorm = (float)sqrt(AbVector.xyz.x * AbVector.xyz.x + AbVector.xyz.y * AbVector.xyz.y + AbVector.xyz.z * AbVector.xyz.z);
    float BcNorm = (float)sqrt(BcVector.xyz.x * BcVector.xyz.x + BcVector.xyz.y * BcVector.xyz.y + BcVector.xyz.z * BcVector.xyz.z);

    k4a_float3_t AbVectorNorm;
    k4a_float3_t BcVectorNorm;

    AbVectorNorm.xyz.x = AbVector.xyz.x / AbNorm;
    AbVectorNorm.xyz.y = AbVector.xyz.y / AbNorm;
    AbVectorNorm.xyz.z = AbVector.xyz.z / AbNorm;

    BcVectorNorm.xyz.x = BcVector.xyz.x / BcNorm;
    BcVectorNorm.xyz.y = BcVector.xyz.y / BcNorm;
    BcVectorNorm.xyz.z = BcVector.xyz.z / BcNorm;

    float result = AbVectorNorm.xyz.x * BcVectorNorm.xyz.x + AbVectorNorm.xyz.y * BcVectorNorm.xyz.y + AbVectorNorm.xyz.z * BcVectorNorm.xyz.z;

    result = (float)std::acos(result) * 180.0f / 3.1415926535897f;
    return result;
}


#endif