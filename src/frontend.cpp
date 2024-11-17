
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "stereoslam/frontend.h"
#include "stereoslam/algorithm.h"

namespace stereoslam
{
    FrontEnd::FrontEnd() {
        gftt_ = cv::GFTTDetector::create(num_features_, 0.01, 20);

    }

    bool FrontEnd::AddFrame(std::shared_ptr<Frame> frame) {
        current_frame_ = frame;

        switch (status_)
        {
        case FrontEndStatus::INITIALIZING:
            StereoInit();
            break;
        case FrontEndStatus::TRACKING_GOOD:
        case FrontEndStatus::TRACKING_BAD:
            Track();
            break;
        case FrontEndStatus::LOST:
            Reset();
            break;
        }

        last_frame_ = current_frame_;
        return true;
    }

    bool FrontEnd::StereoInit() {
        int num_features_left = DetectFeatures();
        int num_features_right = FindFeaturesInRight();
        if(num_features_right < num_features_init_) {
            return false;
        }

        bool build_map_success = BuilInitMap();
        if(build_map_success) {
            status_ = FrontEndStatus::TRACKING_GOOD;
        }
        return build_map_success;
    }

    int FrontEnd::DetectFeatures() {
        cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);
        for (auto &feat: current_frame_->features_left_) {
            cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                  feat->position_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
        }
        std::vector<cv::KeyPoint> keypoints;
        gftt_->detect(current_frame_->left_img_, keypoints, mask);

        int cnt_detected = 0;
        for (auto &keypoint: keypoints) {
            std::shared_ptr<Feature> feature(new Feature(current_frame_, keypoint));
            current_frame_->features_left_.push_back(feature);
            cnt_detected++;
        }
        LOG(INFO) << "Detect " << cnt_detected << " features";
        return cnt_detected;
    }

    int FrontEnd::FindFeaturesInRight() {
        
        std::vector<cv::Point2f> points_left, points_right;
        for(auto& kp: current_frame_->features_left_) {
            points_left.push_back(kp->position_.pt);
            auto mp = kp->map_point_.lock();
            if(mp) {
                Vec2 px = right_camera_->world2pixel(mp->pos_, current_frame_->Pose());
                points_right.push_back(cv::Point2f(px[0], px[1]));
            }
            else {
                points_right.push_back(kp->position_.pt);
            }
        }

        std::vector<uchar> status;
        cv::Mat error;
        cv::calcOpticalFlowPyrLK(current_frame_->left_img_, current_frame_->right_img_, points_left, points_right, status, error,
                cv::Size(11,11), 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

        int cnt_detected = 0;
        for(int i = 0; i < status.size(); i++) {
            if(status[i]) {
                cv::KeyPoint keypoint(points_right[i],7);
                Feature::Ptr feature(new Feature(current_frame_, keypoint));
                feature->is_on_left_ = false;
                current_frame_->features_right_.push_back(feature);
                cnt_detected++;
            }
            else {
                current_frame_->features_right_.push_back(nullptr);
            }
        }
        LOG(INFO) << "Find " << cnt_detected << " features in right image";
        return cnt_detected;
        
    }

    bool FrontEnd::BuilInitMap() {
        std::vector<SE3> poses {left_camera_->pose(), right_camera_->pose()};
        int triangluated_points = 0;
        for(int i=0;i<current_frame_->features_left_.size();i++) {
            if(current_frame_->features_right_[i] == nullptr) continue; 
            
            std::vector<Vec3> points{
            left_camera_->pixel2camera(
                Vec2(current_frame_->features_left_[i]->position_.pt.x,
                     current_frame_->features_left_[i]->position_.pt.y)),
            right_camera_->pixel2camera(
                Vec2(current_frame_->features_right_[i]->position_.pt.x,
                     current_frame_->features_right_[i]->position_.pt.y))};
            Vec3 pworld = Vec3::Zero();


            if(triangulation(poses, points, pworld) && pworld[2] > 0) {
                MapPoint::Ptr map_point = MapPoint::CreateMapPoint();
                map_point->SetPos(pworld);
                map_point->AddObservation(current_frame_->features_left_[i]);
                map_point->AddObservation(current_frame_->features_right_[i]);
                current_frame_->features_left_[i]->map_point_ = map_point;
                current_frame_->features_right_[i]->map_point_ = map_point;
                map_->InsertMapPoint(map_point);
                triangluated_points++;
            }
        }
        LOG(INFO) << "New Landmarks: " << triangluated_points ;
        return true;
    }


}