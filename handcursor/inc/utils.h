#include <math.h>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#define RAD_TO_DEG 57.2958


double sigmoidfunc(double x){
    double sigmoid = 1.0 / (1.0 + exp(-x));
    return sigmoid;
}

float angleBetween(cv::Point v1, cv::Point v2)
{
    float len1 = sqrt(v1.x * v1.x + v1.y * v1.y);
    float len2 = sqrt(v2.x * v2.x + v2.y * v2.y);

    float dot = v1.x * v2.x + v1.y * v2.y;

    float a = dot / (len1 * len2);

    if (a >= 1.0)
        return 0.0;
    else if (a <= -1.0)
        return M_PI * RAD_TO_DEG;
    else
        return acos(a) * RAD_TO_DEG; // 0..PI
}