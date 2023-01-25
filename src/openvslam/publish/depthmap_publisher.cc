#include "openvslam/data/landmark.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/data/map_database.h"
#include "openvslam/publish/depthmap_publisher.h"

#include <spdlog/spdlog.h>

namespace openvslam {
namespace publish {

depthmap_publisher::depthmap_publisher(const std::shared_ptr<config>& cfg, data::map_database* depthmap_db)
    : cfg_(cfg), depthmap_db_(depthmap_db) {
    spdlog::debug("CONSTRUCT: publish::depthmap_publisher");
}

depthmap_publisher::~depthmap_publisher() {
    spdlog::debug("DESTRUCT: publish::depthmap_publisher");
}

void depthmap_publisher::set_current_cam_pose(const Mat44_t& cam_pose_cw) {
    std::lock_guard<std::mutex> lock(mtx_cam_pose_);
    cam_pose_cw_ = cam_pose_cw;
}

Mat44_t depthmap_publisher::get_current_cam_pose() {
    std::lock_guard<std::mutex> lock(mtx_cam_pose_);
    return cam_pose_cw_;
}

unsigned int depthmap_publisher::get_keyframes(std::vector<data::keyframe*>& all_keyfrms) {
    all_keyfrms = depthmap_db_->get_all_keyframes();
    return depthmap_db_->get_num_keyframes();
}

unsigned int depthmap_publisher::get_landmarks(std::vector<data::landmark*>& all_landmarks,
                                          std::set<data::landmark*>& local_landmarks) {
    all_landmarks = depthmap_db_->get_all_landmarks();
    const auto _local_landmarks = depthmap_db_->get_local_landmarks();
    local_landmarks = std::set<data::landmark*>(_local_landmarks.begin(), _local_landmarks.end());
    return depthmap_db_->get_num_landmarks();
}

unsigned int depthmap_publisher::get_landmarks_dense(std::vector<data::landmark*>& all_landmarks) {
    all_landmarks = depthmap_db_->get_all_landmarks();
    return depthmap_db_->get_num_landmarks();
}

} // namespace publish
} // namespace openvslam
