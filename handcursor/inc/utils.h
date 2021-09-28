#include <math.h>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#define RAD_TO_DEG 57.2958

double sigmoidfunc(double x);

float angleBetween(cv::Point v1, cv::Point v2);

cv::Point2f rotate2d(const cv::Point2f& inPoint, const double& angRad);

cv::Point2f rotatePoint(const cv::Point2f& inPoint, const cv::Point2f& center, const double& angRad);

