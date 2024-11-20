

#ifndef VISUAL_ODOMETRY_H
#define VISUAL_ODOMETRY_H

#include "stereoslam/common_include.h"
#include "stereoslam/frame.h"
#include "stereoslam/map.h"
#include "stereoslam/frontend.h"
#include "stereoslam/dataset.h"
#include "stereoslam/viewer.h"


namespace stereoslam
{
    class VisualOdometry {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<VisualOdometry> Ptr;

        /// constructor with config file
        VisualOdometry(std::string &config_path);

        /**
         * do initialization things before run
         * @return true if success
         */
        bool Init();

        /**
         * start vo in the dataset
         */
        void Run();

        /**
         * Make a step forward in dataset
         */
        bool Step();

        FrontEndStatus GetFrontendStatus() const { return frontend_->GetStatus(); }

    private:
        bool inited_ = false;
        std::string config_file_path_;

        FrontEnd::Ptr frontend_ = nullptr;
        Backend::Ptr backend_ = nullptr;
        Map::Ptr map_ = nullptr;
        Viewer::Ptr viewer_ = nullptr;

        // dataset
        Dataset::Ptr dataset_ = nullptr;
    };

}

#endif