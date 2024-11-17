

#include "stereoslam/frame.h"


namespace stereoslam {

    Frame::Frame(unsigned long id, double time_stamp, const SE3& pose, const cv::Mat& left_img, const cv::Mat& right_img) : id_(id), time_stamp_(time_stamp), pose_(pose), left_img_(left_img), right_img_(right_img) {}

    Frame::Ptr Frame::CreateFrame() {
        static unsigned long factory_id = 0;
        Frame::Ptr new_frame(new Frame);
        new_frame->id_ = factory_id++;
        return new_frame;
    }

    void Frame::SetKeyFrame() {
        unsigned long keyframe_factory_id = 0;
        is_keyframe_ = true;
        keyframe_id_ = keyframe_factory_id++;
    }
}