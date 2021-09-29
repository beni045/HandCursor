#include <iostream>
#include <opencv2/highgui.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "modelprocessor.h"
#include "utils.h"

struct TransformData {
    cv::Point2f offset;
    cv::Point2f center;
    double angleRad;
};

class HandDetector : public ModelProcessor {       
    private:            
        std::vector<float> anchors_;
        std::vector<float> LoadAnchors(std::string filepath);
        cv::Mat result_;
        cv::RotatedRect cropRect_;
        
        TransformData transdata_;
        
        void ExtraSetup();

        void Preprocess();
        void Postprocess();

        int FindWidest(std::vector<int> threshold_idxs);

        std::vector<cv::Point> FindKeypoints(int widest_idx);
        cv::Mat TransformPalm(cv::Point wrist, cv::Point middlefinger, float thirdpoint_scale);
        
    public:
        HandDetector(int resize_width, int resize_height, std::string model_path);
        cv::Mat GetResult();
        void TransformBack(std::vector<cv::Point2f>& inPoints);
        cv::RotatedRect GetCropRect();
};


