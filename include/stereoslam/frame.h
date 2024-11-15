
#pragma once

#ifndef STEREOSLAM_FRAME_H
#define STEREOSLAM_FRAME_H

#include "stereoslam/common_include.h"

// for Sophus
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

typedef Sophus::SE3d SE3;
typedef Sophus::SO3d SO3;

// for cv
#include <opencv2/core/core.hpp>

using cv::Mat;

// glog
#include <glog/logging.h>

namespace stereoslam
{
    
    struct Frame
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
            typedef std::shared_ptr<Frame> Ptr;
            unsigned long id_ = 0;
            unsigned long keyframe_id_ = 0;
            bool is_keyframe_ = false;
            double time_stamp_ ;
            SE3 pose_;
            std::mutex pose_mutex_;
            cv::Mat left_img_, right_img_;

            std::vector<std::shared_ptr<Feature>> features_left_;
            std::vector<std::shared_ptr<Feature>> features_right_;

            Frame(){}

            Frame(unsigned long id, double time_stamp, const SE3& pose, const cv::Mat& left_img, const cv::Mat& right_img); 

            SE3 Pose() {
                std::unique_lock<std::mutex> lock(pose_mutex_);
                return pose_;
            }    

            void SetPose(const SE3& pose) {
                std::unique_lock<std::mutex> lock(pose_mutex_);
                pose_ = pose;
            }       

            void SetKeyFrame();

            static std::shared_ptr<Frame> CreateFrame();

    };

} // namespace stereoslam



#endif