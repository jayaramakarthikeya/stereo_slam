
#pragma once

#ifndef MAP_H
#define MAP_H

#include "stereoslam/common_include.h"
#include "stereoslam/mappoint.h"`

namespace stereoslam{
    struct Map{
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW; 

        typedef std::shared_ptr<Map> Ptr;
        typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarkType;
        typedef std::unordered_map<unsigned long, Frame::Ptr> KeyFrameType;
        

        Map() {}

        void InsertKeyFrame(Frame::Ptr frame);

        void InsertMapPoint(MapPoint::Ptr map_point);

        LandmarkType GetAllMappoints() {
            std::unique_lock<std::mutex> lock(data_mutex_);
            return landmarks_;
        }

        KeyFrameType GetAllKeyFrames() {
            std::unique_lock<std::mutex> lock(data_mutex_);
            return keyframes_;
        }

        LandmarkType GetActiveMappoints() {
            std::unique_lock<std::mutex> lock(data_mutex_);
            return active_landmarks_;
        }

        KeyFrameType GetActiveKeyFrames() {
            std::unique_lock<std::mutex> lock(data_mutex_);
            return active_keyframes_;
        }

        void CleanMap();

        private:

        std::mutex data_mutex_;

        LandmarkType landmarks_, active_landmarks_;
        KeyFrameType keyframes_, active_keyframes_;
        
        Frame::Ptr current_keyframe_ = nullptr;

        int num_active_keyframes = 7;
    };
}

#endif