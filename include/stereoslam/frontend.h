
#pragma once

#ifndef FRONTEND_H
#define FRONTEND_H

#include <opencv2/features2d.hpp>

#include "stereoslam/common_include.h"
#include "stereoslam/frame.h"
#include "stereoslam/map.h"
#include "stereoslam/camera.h"

namespace stereoslam
{

    enum class FrontEndStatus
    {
        INITIALIZING,
        TRACKING_GOOD,
        TRACKING_BAD,
        LOST
    };

    class FrontEnd
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<FrontEnd> Ptr;    

        FrontEnd();

        bool AddFrame(std::shared_ptr<Frame> frame);

        void SetMap(Map::Ptr map) { map_ = map; }   


        FrontEndStatus GetStatus() const { return status_; }

        void SetCameras(Camera::Ptr& left_camera, Camera::Ptr& right_camera) {
            left_camera_ = left_camera;
            right_camera_ = right_camera;
        }

    private:
        bool Track();

        bool Reset();

        int TrackLastFrame();

        bool InsertKeyFrame();

        int EstimateCurrentPose();

        bool StereoInit();

        int DetectFeatures();

        int FindFeaturesInRight();

        bool BuilInitMap();



        Map::Ptr map_ = nullptr;
        FrontEndStatus status_ = FrontEndStatus::INITIALIZING;
        Camera::Ptr left_camera_ = nullptr;
        Camera::Ptr right_camera_ = nullptr;

        Frame::Ptr current_frame_ = nullptr;
        Frame::Ptr last_frame_ = nullptr;

        SE3 relative_motion_;

        int tracking_inliers_ = 0;  

        //params

        int num_features_ = 200;
        int num_features_init_ = 100;
        int num_features_tracking_ = 50;
        int num_features_tracking_bad_ = 20;
        int num_features_needed_for_keyframe_ = 80;

        cv::Ptr<cv::GFTTDetector> gftt_;
    };
} // namespace stereoslam

#endif