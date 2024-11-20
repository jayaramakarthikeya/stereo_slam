
#pragma once

#ifndef FEATURE_H
#define FEATURE_H

#include "stereoslam/common_include.h"
#include "stereoslam/mappoint.h"


namespace stereoslam
{
    struct Frame;

    //struct MapPoint;
    struct Feature {

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Feature> Ptr;

        std::weak_ptr<Frame> frame_; 
        cv::KeyPoint position_;
        std::weak_ptr<MapPoint> map_point_;

        bool is_outlier_ = false;
        bool is_on_left_ = true;

        Feature() {}

        Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint& position) : frame_(frame), position_(position) {}

        
    };
} // namespace stereoslam



#endif