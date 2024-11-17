
#pragma once

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "stereoslam/common_include.h"
#include "stereoslam/feature.h"
#include <Eigen/Core>


namespace stereoslam {
    struct MapPoint {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        typedef std::shared_ptr<MapPoint> Ptr;

        unsigned long id_ = 0;
        bool is_outlier_ = false;
        Vec3 pos_ = Vec3::Zero();

        std::mutex data_mutex_;
        int observed_times_ = 0;
        std::list<std::weak_ptr<Feature>> observations_;


        MapPoint() {}

        MapPoint(long id, Vec3 pos);

        Vec3 GetPos() {
            std::unique_lock<std::mutex> lock(data_mutex_);
            return pos_;
        }

        void SetPos(Vec3 pos) {
            std::unique_lock<std::mutex> lock(data_mutex_);
            pos_ = pos;
        }

        void AddObservation(std::shared_ptr<Feature> feature) {
            std::unique_lock<std::mutex> lock(data_mutex_);
            observations_.emplace_back(feature);
            observed_times_++;
        }

        void RemoveObservation(std::shared_ptr<Feature> feature);

        std::list<std::weak_ptr<Feature>> GetObservations() {
            std::unique_lock<std::mutex> lock(data_mutex_);
            return observations_;
        }

        static MapPoint::Ptr CreateMapPoint();
    };
}



#endif