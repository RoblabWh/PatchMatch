#include "depthmap_computation_module.h"
#include "openvslam/data/keyframe.h"
#include <opencv2/core/eigen.hpp>
#include "openvslam/data/map_database.h"
#include "openvslam/data/landmark.h"

#include <list>
#include <mutex>
#include <thread>
#include <memory>
#include <iostream>
#include <spdlog/spdlog.h>
#include <math.h>

namespace openvslam {

//! Constructor
dense_pointcloud_module::dense_pointcloud_module(const std::shared_ptr<config>& cfg, data::map_database* depthmap_db) : 
cfg_(cfg),
depthmap_db_(depthmap_db),
DepthmapPruner_(new dense::DepthmapPruner()) {
}

//! Destructor
dense_pointcloud_module::~dense_pointcloud_module() {

}

/*
//! Set the ?? module
void dense_pointcloud_module::set_depthmap_computation_module(mapping_module* mapper) {
    mapper_ = mapper;
}
*/


//-----------------------------------------
// main process
void dense_pointcloud_module::save_pruned_depth(cv::Mat &depth_pruned,  std::string &id, std::string &path_to_save) {

    
    double dMin;
    double dMax;
    cv::Mat depth_pruned_scaled;
    cv::Mat depth_pruned_img;

    cv::minMaxLoc(depth_pruned, &dMin, &dMax);
    depth_pruned_scaled = depth_pruned *  (255/(dMax-dMin));
    depth_pruned_scaled.convertTo(depth_pruned_img, CV_8UC1);

    std::string path_depth_pruned_full = path_to_save + "/depth_pruned_" + id + ".png" ;
    cv::imwrite(path_depth_pruned_full, depth_pruned_img);
}

//! Run main loop of the global optimization module
void dense_pointcloud_module::run() {
    
    //auto video = cv::VideoCapture("/media/marcsryzen/Volume/Videos/insta_tests/onex2/Ewald/Zechenwand_small.mp4", cv::CAP_FFMPEG);
    //auto video = cv::VideoCapture("/media/marcsryzen/Volume1/equi_small.mp4", cv::CAP_FFMPEG);
    //auto video = cv::VideoCapture("/media/marcsryzen/Volume/Videos/insta_tests/onex2/Garten/Garten.mp4", cv::CAP_FFMPEG);

    is_terminated_ = false;
    int pruned_id = 0;
    int nLms = 0;
    DepthmapPruner_->SetMinViews(1);
    DepthmapPruner_->SetDepthQueueSize(5);
    DepthmapPruner_->SetSameDepthThreshold(cfg_->SameDepthThreshold);


    std::vector<data::keyframe*> keyfrms_to_process;
    
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

        // check if termination is requested
        if (terminate_is_requested()) {
            // terminate and break
            terminate();
            break;
        }

        // check if pause is requested
        if (pause_is_requested()) {
            // pause and wait
            pause();
            // check if termination or reset is requested during pause
            while (is_paused() && !terminate_is_requested() && !reset_is_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
            }
        }

        // check if reset is requested
        if (reset_is_requested()) {
            // reset and continue
            DepthmapPruner_->reset();
            keyfrms_to_process.clear(); //does not free the underlying keyframes, so no memory leak (shared keyframes with depthmap compute module)
            pruned_id=0;
            reset();
            continue;
        }

        // if the queue is empty, the following process is not needed
        if (!keyframe_is_queued()) {
            continue;
        }

        // dequeue
        {
            std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    
            if (keyfrms_queue_.size() > 0) {
                data::keyframe* cur_keyfrm = keyfrms_queue_.front();
                keyfrms_to_process.push_back(cur_keyfrm);
                keyfrms_queue_.pop_front();
                dense::DepthmapCleanerResult* cleaned_depth_Result = cur_keyfrm->get_cleaned_result();
                if (cleaned_depth_Result == nullptr) {
                    spdlog::warn("Keyframe return nullptr for cleaned depth result: " + std::to_string(cur_keyfrm->id_));
                }
                else {
                    cv::Mat rgb = cur_keyfrm->get_rgb_thumb();
                    DepthmapPruner_->AddView(cleaned_depth_Result, rgb);
                }
            }
        }

        if (DepthmapPruner_->IsReadyForCompute()) {

            //if (depthmap_db_->get_num_keyframes() == 1) {
            //    depthmap_db_->pop_front_keyframe();
            //}

            /*
            std::vector<cv::Vec3d> merged_points;
            std::vector<cv::Vec3d> merged_normals;
            std::vector<cv::Vec3d> merged_colors;
            std::vector<cv::Vec2i> merged_keypts;
            */
            
            //dense::DepthmapCleanerResult* cleaned_depth_Result = keyfrm_densify->get_cleaned_result();
            dense::DepthmapPrunerResult* pruned_depth_result = new dense::DepthmapPrunerResult;
            
            DepthmapPruner_->Prune(pruned_depth_result); //merged_points, merged_normals, merged_colors, merged_keypts);
            DepthmapPruner_->PopHead();

            data::keyframe* keyfrm_densify =  keyfrms_to_process[pruned_id];
            depthmap_db_->add_keyframe(keyfrm_densify);
            pruned_id++;

            
            std::string s_src_id = std::to_string(keyfrm_densify->src_frm_id_);
            std::string path = cfg_->PathImgsKeyfrms_; //"/home/marcsryzen/openvslam_debug/build/zechenwand/keyfrm";
            save_pruned_depth(pruned_depth_result->pruned_depth, s_src_id , path);
            
            int color_id = 0;
            std::vector<cv::Vec2i>::iterator it_keypts = pruned_depth_result->keypts.begin();
            for (std::vector<cv::Vec3d>::iterator it = pruned_depth_result->points3d.begin(); it != pruned_depth_result->points3d.end(); it++) {

                cv::Vec3d point_w = *it;
                cv::Vec2i keypt = *it_keypts;
                Vec3_t point_w_eig(point_w(0),point_w(1),point_w(2));
                data::landmark* lm = new data::landmark(nLms, keyfrm_densify->id_,  point_w_eig, keyfrm_densify, 1, 1, depthmap_db_);
                

                keyfrm_densify->add_keypt_dense(keypt);
                keyfrm_densify->add_landmark_dense(lm, nLms);
                
                float c_b = pruned_depth_result->colors[color_id][0];
                float c_g = pruned_depth_result->colors[color_id][1];
                float c_r = pruned_depth_result->colors[color_id][2];
                lm->setColor(c_r, c_g, c_b);
                depthmap_db_->add_landmark(lm);

                color_id++;
                nLms++;
            }

            //DepthmapPruner_->PopHeadUnlessOne();
            //pruned_id++;
        }
    }
    spdlog::info("terminate dense pointcloud module");
}

void dense_pointcloud_module::queue_keyframe(data::keyframe* keyfrm_) {

    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    //data::keyframe* keyfrm = new data::keyframe(*keyfrm_);
    
    keyfrms_queue_.push_back(keyfrm_);
}

void dense_pointcloud_module::queue_keyframes(std::vector<data::keyframe*> &keyfrms) {

    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);

    for (std::vector<data::keyframe*>::iterator it = keyfrms.begin(); it != keyfrms.end(); it++) {
        data::keyframe* keyfrm_ = *it;
        data::keyframe* keyfrm = new data::keyframe(*keyfrm_);
        keyfrms_queue_.push_back(keyfrm);
    }
}

bool dense_pointcloud_module::keyframe_is_queued() const {
    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    return (!keyfrms_queue_.empty());
}

//-----------------------------------------
// management for reset process

//! Request to reset the global optimization module
//! (NOTE: this function waits for reset)
void dense_pointcloud_module::request_reset() {

    {
        std::lock_guard<std::mutex> lock(mtx_reset_);
        reset_is_requested_ = true;
    }

    // BLOCK until reset
    while (true) {
        {
            std::lock_guard<std::mutex> lock(mtx_reset_);
            if (!reset_is_requested_) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(3000));
    }
}

void dense_pointcloud_module::reset() {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    
    //depthmap_db_->clear();
    reset_is_requested_ = false;

    {
        std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
        keyfrms_queue_.clear();
    }
    
}

bool dense_pointcloud_module::reset_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    return reset_is_requested_;
}

//-----------------------------------------
// management for pause process

//! Request to pause the global optimization module
//! (NOTE: this function does not wait for pause)
void dense_pointcloud_module::request_pause() {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    pause_is_requested_ = true;
}

//! Check if the global optimization module is requested to be paused or not
bool dense_pointcloud_module::pause_is_requested() const {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    return pause_is_requested_;
}

//! Check if the global optimization module is paused or not
bool dense_pointcloud_module::is_paused() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return is_paused_;
    
}

void dense_pointcloud_module::pause() {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    is_paused_ = true;
}

//! Resume the global optimization module
void dense_pointcloud_module::resume() {

    std::lock_guard<std::mutex> lock1(mtx_pause_);
    std::lock_guard<std::mutex> lock2(mtx_terminate_);

    // if it has been already terminated, cannot resume
    if (is_terminated_) {
        return;
    }

    is_paused_ = false;
    pause_is_requested_ = false;
}

//-----------------------------------------
// management for terminate process

//! Request to terminate the global optimization module
//! (NOTE: this function does not wait for terminate)
void dense_pointcloud_module::request_terminate() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    terminate_is_requested_ = true;
}
//! Check if the global optimization module is terminated or not
bool dense_pointcloud_module::is_terminated() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return is_terminated_;
}
bool dense_pointcloud_module::terminate_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return terminate_is_requested_;
}

void dense_pointcloud_module::terminate() {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    std::lock_guard<std::mutex> lock2(mtx_terminate_);
    is_paused_ = true;
    is_terminated_ = true;
}

//-----------------------------------------
// management for loop DC

//! Check if loop DC is running or not
bool dense_pointcloud_module::loop_DC_is_running() const {
    return true;
}

//! Abort the loop DC externally
//! (NOTE: this function does not wait for abort)
void dense_pointcloud_module::abort_loop_DC() {

}


}  // namespace openvslam
