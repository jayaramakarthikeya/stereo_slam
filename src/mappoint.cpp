
#include "stereoslam/mappoint.h"
#include "stereoslam/feature.h"

namespace stereoslam {

    MapPoint::MapPoint(long id, Vec3 pos) : id_(id), pos_(pos) {}

    MapPoint::Ptr MapPoint::CreateMapPoint() {
        static unsigned long factory_id = 0;
        MapPoint::Ptr new_map_point(new MapPoint);
        new_map_point->id_ = factory_id++;
        return new_map_point;
    }


    void MapPoint::RemoveObservation(std::shared_ptr<Feature> feature) {
        std::unique_lock<std::mutex> lock(data_mutex_);
        for(auto observation = observations_.begin(); observation != observations_.end(); observation++) {
            if(observation->lock() == feature) {
                observations_.erase(observation);
                feature->map_point_.reset();
                observed_times_--;
                break;
            }
        }
    }
}