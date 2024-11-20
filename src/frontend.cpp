
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "stereoslam/frontend.h"
#include "stereoslam/algorithm.h"
#include "stereoslam/g2o_types.h"
#include "stereoslam/config.h"

namespace stereoslam
{
    FrontEnd::FrontEnd() {
        gftt_ =
        cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);
        num_features_init_ = Config::Get<int>("num_features_init");
        num_features_ = Config::Get<int>("num_features");

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
        //LOG(INFO) << "Pose matrix:" << current_frame_->Pose().matrix();
        last_frame_ = current_frame_;
        return true;
    }

    bool FrontEnd::Track() {
        if(last_frame_) {
            current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
        }

        int num_track_last = TrackLastFrame();
        tracking_inliers_ = EstimateCurrentPose();

        

        std::cout << "tracking inliers: " << tracking_inliers_ << std::endl;

        if(tracking_inliers_ > num_features_tracking_) {
            status_ = FrontEndStatus::TRACKING_GOOD;
        }
        else if (tracking_inliers_ > num_features_tracking_bad_) {
            status_ = FrontEndStatus::TRACKING_BAD;
        }
        else {
            status_ = FrontEndStatus::LOST;
        }
        
        InsertKeyFrame();   
        relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();
        
        if (viewer_) viewer_->AddCurrentFrame(current_frame_);
        return true;
    }

    bool FrontEnd::InsertKeyFrame() {
        if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
            // still have enough features, don't insert keyframe
            return false;
        }
        // current frame is a new keyframe
        current_frame_->SetKeyFrame();
        map_->InsertKeyFrame(current_frame_);

        LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
                << current_frame_->keyframe_id_;

        SetObservationsForKeyFrame();
        DetectFeatures();  // detect new features

        // track in right image
        FindFeaturesInRight();
        // triangulate map points
        TriangulateNewPoints();
        // update backend because we have a new keyframe
        backend_->UpdateMap();

        if (viewer_) viewer_->UpdateMap();

        return true;
    }

    void FrontEnd::SetObservationsForKeyFrame() {
        for (auto &feat : current_frame_->features_left_) {
            auto mp = feat->map_point_.lock();
            if (mp) mp->AddObservation(feat);
        }
    }

    int FrontEnd::TriangulateNewPoints() {
        std::vector<SE3> poses{left_camera_->pose(), right_camera_->pose()};
        SE3 current_pose_Twc = current_frame_->Pose().inverse();
        int cnt_triangulated_pts = 0;
        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
            if (current_frame_->features_left_[i]->map_point_.expired() &&
                current_frame_->features_right_[i] != nullptr) {
                
                std::vector<Vec3> points{
                    left_camera_->pixel2camera(
                        Vec2(current_frame_->features_left_[i]->position_.pt.x,
                            current_frame_->features_left_[i]->position_.pt.y)),
                    right_camera_->pixel2camera(
                        Vec2(current_frame_->features_right_[i]->position_.pt.x,
                            current_frame_->features_right_[i]->position_.pt.y))};
                Vec3 pworld = Vec3::Zero();

                if (triangulation(poses, points, pworld) && pworld[2] > 0) {
                    auto new_map_point = MapPoint::CreateMapPoint();
                    pworld = current_pose_Twc * pworld;
                    new_map_point->SetPos(pworld);
                    new_map_point->AddObservation(
                        current_frame_->features_left_[i]);
                    new_map_point->AddObservation(
                        current_frame_->features_right_[i]);

                    current_frame_->features_left_[i]->map_point_ = new_map_point;
                    current_frame_->features_right_[i]->map_point_ = new_map_point;
                    map_->InsertMapPoint(new_map_point);
                    cnt_triangulated_pts++;
                }
            }
        }
        LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
        return cnt_triangulated_pts;
    }

    int FrontEnd::TrackLastFrame() {
        std::vector<cv::Point2f> points_left, points_right;
        for(auto& kp: last_frame_->features_left_) {
            
            auto mp = kp->map_point_.lock();
            if(mp) {
                Vec2 px = left_camera_->world2pixel(mp->pos_, current_frame_->Pose());
                points_right.push_back(cv::Point2f(px[0], px[1]));
                points_left.push_back(kp->position_.pt);
            }
            else {
                points_right.push_back(kp->position_.pt);
                points_left.push_back(kp->position_.pt);
            }
        }

        //std::cout << "points_left: " << points_left.size() << std::endl;
        //cv::Mat prevPtsMat = cv::Mat(points_left).reshape(1).convertTo(prevPtsMat, CV_32F);

        std::vector<uchar> status;
        std::vector<float> error;
        cv::calcOpticalFlowPyrLK(last_frame_->left_img_, current_frame_->left_img_, points_left, points_right, status, error,
                cv::Size(11,11), 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

        int cnt_detected = 0;
        for(int i = 0; i < status.size(); i++) {
            if(status[i]) {
                cv::KeyPoint keypoint(points_right[i],7);
                Feature::Ptr feature(new Feature(current_frame_, keypoint));
                feature->is_on_left_ = true;
                feature->map_point_ = last_frame_->features_left_[i]->map_point_;
                current_frame_->features_left_.push_back(feature);
                cnt_detected++;
            }
            // else {
            //     current_frame_->features_right_.push_back(nullptr);
            // }
        }
        LOG(INFO) << "Find " << cnt_detected << " features in right image";
        return cnt_detected;
    }

    int FrontEnd::EstimateCurrentPose() {
        typedef g2o::BlockSolver_6_3 BlockSolverType;
        typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(
                g2o::make_unique<LinearSolverType>()
            )
        );

        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);

        VertexPose* vertex_pose = new VertexPose();
        vertex_pose->setId(0);
        vertex_pose->setEstimate(current_frame_->Pose());
        optimizer.addVertex(vertex_pose);

        Mat33 K = left_camera_->K();

        int index = 1;
        std::vector<EdgeProjectionPoseOnly*> edges;
        std::vector<Feature::Ptr> tracked_features;
        for (auto feature: current_frame_->features_left_) {
            auto mp = feature->map_point_.lock();
            if(mp) {
                tracked_features.push_back(feature);
                EdgeProjectionPoseOnly* edge = new EdgeProjectionPoseOnly(mp->pos_, K);
                edge->setId(index);
                edge->setVertex(0, vertex_pose);
                edge->setMeasurement(toVec2(feature->position_.pt));
                edge->setInformation(Mat22::Identity());
                edge->setRobustKernel(new g2o::RobustKernelHuber);
                edges.push_back(edge);
                optimizer.addEdge(edge);
                index++;
            }
        }

        const double chi2_th = 5.991;
        int cnt_outlier = 0;

        for(int i=0;i<4;i++) {
            vertex_pose->setEstimate(current_frame_->Pose());
            optimizer.initializeOptimization();
            optimizer.optimize(10);
            cnt_outlier = 0;
            for (int j = 0; j < edges.size(); j++) {
                EdgeProjectionPoseOnly* edge = edges[j];
                if(tracked_features[j]->is_outlier_) {
                    edge->computeError();
                }

                if(edge->chi2() > chi2_th) {
                    tracked_features[j]->is_outlier_ = true;
                    edge->setLevel(1);
                    cnt_outlier++;
                } 
                else {
                    tracked_features[j]->is_outlier_ = false;
                    edge->setLevel(0);
                }

                if(i == 2) {
                    edge->setRobustKernel(nullptr);
                }
            }
        }

        LOG(INFO) << "Outliers in pose estimating is: " << cnt_outlier;

        current_frame_->SetPose(vertex_pose->estimate());
        
        for (auto &feat : tracked_features) {
            if (feat->is_outlier_) {
                feat->map_point_.reset();
                feat->is_outlier_ = false;  // maybe we can still use it in future
            }
        }
        return tracked_features.size() - cnt_outlier;
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
            if (viewer_) {
                viewer_->AddCurrentFrame(current_frame_);
                viewer_->UpdateMap();
            }
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
        LOG(INFO) << "Found " << cnt_detected << " features in right image";
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
                auto map_point = MapPoint::CreateMapPoint();
                map_point->SetPos(pworld);
                map_point->AddObservation(current_frame_->features_left_[i]);
                map_point->AddObservation(current_frame_->features_right_[i]);
                current_frame_->features_left_[i]->map_point_ = map_point;
                current_frame_->features_right_[i]->map_point_ = map_point;
                map_->InsertMapPoint(map_point);
                triangluated_points++;
            }
        }

        current_frame_->SetKeyFrame();
        map_->InsertKeyFrame(current_frame_);
        backend_->UpdateMap();
        LOG(INFO) << "New Landmarks: " << triangluated_points ;
        return true;
    }

    bool FrontEnd::Reset() {
        LOG(INFO) << "Reset FrontEnd";

        return true;
        
    }

}