
#pragma once

#ifndef BACKEND_H
#define BACKEND_H

#include "stereoslam/common_include.h"
#include "stereoslam/camera.h"
#include "stereoslam/map.h"

namespace stereoslam {
    
    class Backend
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Backend> Ptr;

        Backend();

        void SetCameras(Camera::Ptr& left_camera, Camera::Ptr& right_camera) {
            left_camera_ = left_camera;
            right_camera_ = right_camera;
        }

        void UpdateMap();

        void SetMap(Map::Ptr map) { map_ = map; } 

        void Stop();         


    private:
        Camera::Ptr left_camera_ = nullptr;
        Camera::Ptr right_camera_ = nullptr;
        Map::Ptr map_ = nullptr;
        std::thread backend_thread_;
        std::mutex data_mutex_;
        std::condition_variable map_update_;

        std::atomic<bool> running_ = false;

        void Optimize(Map::KeyFrameType& active_keyframes, Map::LandmarkType& active_landmarks); 

        void BackendLoop();
        
    };
}
#endif