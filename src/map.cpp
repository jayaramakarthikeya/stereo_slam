

#include "stereoslam/map.h"

namespace stereoslam {

    void Map::InsertKeyFrame(Frame::Ptr frame) {
        current_keyframe_ = frame;
        if(keyframes_.find(frame->id_) == keyframes_.end()) {
            keyframes_[frame->id_] = frame;
            active_keyframes_[frame->id_] = frame;
        } else {
            keyframes_[frame->id_] = frame;
            active_keyframes_[frame->id_] = frame;
        }

        if(active_keyframes_.size() > num_active_keyframes) {
            RemoveOldKeyFrames();
        }
    }

    void Map::InsertMapPoint(MapPoint::Ptr map_point) {
        landmarks_[map_point->id_] = map_point;
        active_landmarks_[map_point->id_] = map_point;
    }

    void Map::RemoveOldKeyFrames() {
        if(current_keyframe_ == nullptr) return;

        double max_dis = 0, min_dis = 9999;
        double max_kf_id = 0, min_kf_id = 0;
        auto Twc = current_keyframe_->Pose().inverse();

        for(auto iter = active_keyframes_.begin(); iter != active_keyframes_.end(); iter++) {
            double dis = (iter->second->Pose() * Twc).log().norm();
            if(dis > max_dis) {
                max_dis = dis;
                max_kf_id = iter->first;
            }
            if(dis < min_dis) {
                min_dis = dis;
                min_kf_id = iter->first;
            }
        }

        const double min_th = 0.2;
        Frame::Ptr frame_to_remove = nullptr;
        if(min_dis < min_th) {
            frame_to_remove = active_keyframes_.at(min_kf_id);
        }
        else {
            frame_to_remove = active_keyframes_.at(max_kf_id);
        }

        active_keyframes_.erase(frame_to_remove->id_);

        for(auto feat: frame_to_remove->features_left_) {
            auto mp = feat->map_point_.lock();
            if (mp) mp->RemoveObservation(feat);
        }

        for(auto feat: frame_to_remove->features_right_) {
            auto mp = feat->map_point_.lock();
            if (mp) mp->RemoveObservation(feat);
        }

        CleanMap();
    }

    void Map::CleanMap() {
        int cnt_landmarks_removed = 0;

        for(auto iter = active_landmarks_.begin(); iter != active_landmarks_.end(); ) { 
            if(iter->second->observed_times_ == 0) {
                active_landmarks_.erase(iter);
                cnt_landmarks_removed++;
            } else {
                iter++;
            }
        }

        LOG(INFO) << "Removed " << cnt_landmarks_removed << " landmarks";
    }

}