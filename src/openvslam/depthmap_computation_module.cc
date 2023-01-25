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
// #include <filesystem>

namespace openvslam {

static bool cmp_lms(const cv::Vec3d* lm1, const cv::Vec3d* lm2){

    bool lm1_smaller = (*lm1).dot(*lm1) < (*lm2).dot(*lm2);

    return lm1_smaller;
}

//! Constructor
depthmap_computation_module::depthmap_computation_module(const std::shared_ptr<config>& cfg, data::map_database* depthmap_db) : 
cfg_(cfg),
depthmap_db_(depthmap_db),
DepthmapEstimator_(new dense::DepthmapEstimator()),
DepthmapCleaner_(new dense::DepthmapCleaner()),
DepthmapPruner_(new dense::DepthmapPruner()) {
}

//! Destructor
depthmap_computation_module::~depthmap_computation_module() {

}

//! Set the mapping module
void depthmap_computation_module::set_mapping_module(mapping_module* mapper) {
    mapper_ = mapper;
}

//! Set the pcl module
void depthmap_computation_module::set_dense_pointcloud_module(dense_pointcloud_module* pointcloud_constructor) {
   dense_pointcloud_module_ = pointcloud_constructor;
}


void depthmap_computation_module::save_keyfrm_img(cv::Mat& keyfrm_rgb, std::string &id, std::string &path_to_save) {
    std::string path_rgb_full = path_to_save + "/rgb_" + id + ".png";
    cv::imwrite(path_rgb_full, keyfrm_rgb);
}

void depthmap_computation_module::save_cleaned_depth(cv::Mat &depth_cleaned,  std::string &id, std::string &path_to_save) {

    
    double dMin;
    double dMax;
    cv::Mat depth_cleaned_scaled;
    cv::Mat depth_cleaned_img;

    cv::minMaxLoc(depth_cleaned, &dMin, &dMax);
    depth_cleaned_scaled = depth_cleaned *  (255/(dMax-dMin));
    depth_cleaned_scaled.convertTo(depth_cleaned_img, CV_8UC1);

    std::string path_depth_cleaned_full = path_to_save + "/depth_cleaned_" + id + ".png" ;
    cv::imwrite(path_depth_cleaned_full, depth_cleaned_img);
}

void depthmap_computation_module::save_keyfrm_depth_and_score(cv::Mat &depth, cv::Mat &score, std::string &id, std::string &path_to_save) {

    double dMin;
    double dMax;
    cv::Mat depth_scaled;
    cv::Mat depth_img;
    cv::Mat score_scaled;
    cv::Mat score_img;

    cv::minMaxLoc(depth, &dMin, &dMax);
    depth_scaled = depth * (255/(dMax-dMin));
    depth_scaled.convertTo(depth_img, CV_8UC1);
    std::string path_depth_full = path_to_save + "/depth_" + id + ".png" ;
    cv::imwrite(path_depth_full, depth_img);

    
    //score_scaled = (score + 1) * (255/2);
    //cv::normalize(score, score_scaled, 0, 255, cv::NORM_MINMAX);
    cv::minMaxLoc(score, &dMin, &dMax);
    score_scaled = (score + 1) * (255/2);
    score_scaled.convertTo(score_img, CV_8UC1);
    std::string path_score_full = path_to_save + "/score_" + id + ".png" ;
    cv::imwrite(path_score_full, score_img);
}


cv::Mat depthmap_computation_module::get_keyfrm_gray(
    cv::Mat& keyfrm_rgb, std::string &path_to_save, std::string &id, const int scale_down, bool save_gray) {
    
    cv::Mat gray;
    cv::Mat gray_;
    cv::Mat gray_scaled;

    cv::cvtColor(keyfrm_rgb, gray_, cv::COLOR_BGR2GRAY);
    cv::Size size = gray_.size();

    if (true) {
        int height = size.height/scale_down;
        int width = size.width/scale_down;

        cv::resize(gray_, gray_scaled, cv::Size(width, height),0,0, cv::INTER_NEAREST);
        
    }
    gray = gray_scaled;
    if (save_gray) {
        std::string path_gray_full = path_to_save + "/gray_" + id + ".png" ;
        cv::imwrite(path_gray_full, gray);
    }
    
    return gray;
}

void depthmap_computation_module::compute_depth_intervall(data::keyframe* keyfrm, double &min_depth, double &max_depth) {
    cur_keyfrm_->computePercentileDepth_10_90(min_depth, max_depth);
    spdlog::info("depth from lms: " + std::to_string(min_depth) + ", " + std::to_string(max_depth));

    //min_depth *= 0.9;
    //min_depth = (min_depth > 1e-5) ? min_depth : 1e-5;
    min_depth = 1e-5;
    max_depth *= 1.1;
}

void depthmap_computation_module::get_landmark_coords(data::keyframe* keyfrm, std::vector<cv::Vec3d> &lms) {

    std::set<data::landmark*> lms_ = cur_keyfrm_->get_valid_landmarks();
    for (std::set<data::landmark*>::iterator it = lms_.begin(); it != lms_.end(); it++) {
        data::landmark* m = *it;
        Vec3_t v = m->get_pos_in_world();
        cv::Vec3d c2;
        
        cv::eigen2cv(v, c2);
        lms.push_back(c2);
    }
}


float depthmap_computation_module::score_view(data::keyframe* prev_keyfrm, data::keyframe* cur_keyfrm) {

    float score = 0;
    std::vector<cv::Vec3d> lms_prev_keyfrm;
    std::vector<cv::Vec3d> lms_cur_keyfrm;
    std::vector<cv::Vec3d*> lms_common;
    double theta_min = M_PI / 60;
    double theta_max = M_PI / 6;
    Vec3_t prev_pos_ = prev_keyfrm->get_cam_center();
    cv::Vec3d prev_pos;
    cv::eigen2cv(prev_pos_, prev_pos);
    Vec3_t cur_pos_ = cur_keyfrm->get_cam_center();
    cv::Vec3d cur_pos;
    cv::eigen2cv(cur_pos_, cur_pos);

    get_landmark_coords(prev_keyfrm, lms_prev_keyfrm);
    get_landmark_coords(cur_keyfrm, lms_cur_keyfrm);

    std::vector<cv::Vec3d*> test1;
    std::vector<cv::Vec3d*> test2;
    for (int i=0; i < lms_prev_keyfrm.size(); i++) {
        cv::Vec3d* t = &lms_prev_keyfrm[i];
        test1.push_back(t);
    }

    for (int i=0; i < lms_cur_keyfrm.size(); i++) {
        cv::Vec3d* t = &lms_cur_keyfrm[i];
        test2.push_back(t);
    }


    std::set_intersection(
        test1.begin(), test1.end(), test2.begin(),
        test2.end(), std::back_inserter(lms_common), cmp_lms);

    for (int i=0; i < lms_common.size(); i++) {
        cv::Vec3d lm = *lms_common[i]; 
        cv::Vec3d dist_prev = prev_pos - lm;
        cv::Vec3d dist_cur = cur_pos - lm;
        double nom = dist_cur.dot(dist_prev);
        double var1 = dist_cur.dot(dist_cur);
        double var2 = dist_prev.dot(dist_prev);

        double theta = acos(nom/sqrt(var1*var2));
        if (theta > theta_min && theta < theta_max) {
            score += 1.0f;
        }       
    }                


    return score / lms_common.size();
}


//-----------------------------------------
// main process

//! Run main loop of the global optimization module
void depthmap_computation_module::run() {
    
    //auto video = cv::VideoCapture("/media/marcsryzen/Volume/Videos/insta_tests/onex2/Ewald/Zechenwand_small.mp4", cv::CAP_FFMPEG);
    //auto video = cv::VideoCapture("/media/marcsryzen/Volume1/equi_small.mp4", cv::CAP_FFMPEG);
    //auto video = cv::VideoCapture("/media/marcsryzen/Volume/Videos/insta_tests/onex2/Garten/Garten.mp4", cv::CAP_FFMPEG);

    int neighbs = 2;
    is_terminated_ = false;
    DepthmapPruner_->SetMinViews(1);
    DepthmapPruner_->SetDepthQueueSize(5);
    DepthmapCleaner_->SetMinConsistentViews(3);
    DepthmapCleaner_->SetDepthQueueSize(5);
    DepthmapCleaner_->SetSameDepthThreshold(cfg_->SameDepthThreshold);
    DepthmapPruner_->SetSameDepthThreshold(cfg_->SameDepthThreshold);
    DepthmapEstimator_->SetMinImagesCompute(neighbs+1);
    DepthmapEstimator_->SetMinPatchSD(cfg_->MinPatchSD);
    DepthmapEstimator_->SetPatchSize(cfg_->PatchSize);
    DepthmapEstimator_->SetPatchMatchIterations(cfg_->Iterations);
    DepthmapEstimator_->SetMinScore(0.1);

    std::string path_to_save = cfg_->PathImgsKeyfrms_;
    bool imgs_to_save = path_to_save.compare("") != 0;
    // if (imgs_to_save && !std::filesystem::exists(path_to_save))
    // {
    //     std::filesystem::create_directory(path_to_save);
    // }

    unsigned int nLms = 0;

    int src_id=0;
    int depth_id=neighbs/2;
    int clean_id=depth_id+2;
    std::vector<data::keyframe*> keyfrms_to_process;
    int keyfrm_idx = 0;
    auto tp_00 = std::chrono::steady_clock::now();
    auto tp_01 = std::chrono::steady_clock::now();

    openvslam::data::keyframe* prev_keyfrm = NULL;

    std::vector<openvslam::data::keyframe*> current_keyfrm_wdw;
    //std::vector<dense::DepthmapEstimatorResult> depthResults;

    //freopen( "/home/marcsryzen/Dokumente/debug.txt", "w", stdout );

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
            DepthmapEstimator_->reset();
            DepthmapCleaner_->reset();
            //DepthmapPruner_->reset();
            
            
            //depthResults.clear();
            //delete[] prev_keyfrm;
            prev_keyfrm = NULL;
            //delete[] cur_keyfrm_;
            cur_keyfrm_ = NULL;
            //nLms = 0;

            src_id=0;
            depth_id=neighbs/2;
            clean_id=depth_id+2;
            //keyfrms_to_process.clear();
            //keyfrm_idx = 0;
            tp_00 = std::chrono::steady_clock::now();
            tp_01 = std::chrono::steady_clock::now();
            reset();
            current_keyfrm_wdw.clear();
            continue;
        }

        // if the queue is empty, the following process is not needed
        if (!keyframe_is_queued()) {
            continue;
        }

        // dequeue
        {
            std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    
            // dequeue -> cur_keyfrm_
            if (keyfrms_queue_.size() > 0) {
                cur_keyfrm_ = keyfrms_queue_.front();
                keyfrms_queue_.pop_front(); 
            }
        }

        //get_local_keyframes()
        if (cur_keyfrm_ != NULL) { //!cur_keyfrm_->will_be_erased() && cur_keyfrm_->id_>16 && cur_keyfrm_->id_ % 2 == 0

            cv::Mat rgb;
            cv::Mat gray;
            cv::Mat mask;
            cv::Matx33d R; 
            cv::Vec3d t;
            std::vector<cv::Vec3d> lms_keyfrm;

            double min_depth = 1e-5f;
            double max_depth = min_depth;

            bool suff = true;
            float score = 0;
            if (prev_keyfrm != NULL) {

                if (cur_keyfrm_->src_frm_id_ <= prev_keyfrm->src_frm_id_) {
                    spdlog::warn("curr src frm id: " + std::to_string(cur_keyfrm_->src_frm_id_) + " , but prev: " + std::to_string(prev_keyfrm->src_frm_id_));
                    //continue;
                }

                score = score_view(prev_keyfrm, cur_keyfrm_);
                suff = score > 0.25;

                spdlog::info("score: " + std::to_string(score) + " , sufficient stereo? : " + std::to_string(suff));
                if (!suff) {
                    cur_keyfrm_ = NULL;
                    continue;
                }

                 
                    
            }

            current_keyfrm_wdw.push_back(cur_keyfrm_);
            std::string s_src_frm_id = std::to_string(cur_keyfrm_->src_frm_id_);
            //video.set(1, cur_keyfrm_->src_frm_id_);
            //cv::Mat frame;
            //video.read(frame);
            
            //cv::resize(frame, rgb, cv::Size(1440,720));
            //cv::Mat img_gray;
            //cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);

            rgb = cur_keyfrm_->get_rgb_thumb();
            gray = cur_keyfrm_->get_gray_thumb();
            mask = cur_keyfrm_->get_mask_thumb();

            if (imgs_to_save) {
                
                std::string path_gray_full = path_to_save + "/gray_" + s_src_frm_id + ".png" ;
                cv::imwrite(path_gray_full, gray);
                save_keyfrm_img(rgb, s_src_frm_id, path_to_save);
            }
            
            
            cv::eigen2cv(cur_keyfrm_->get_rotation(), R);
            cv::eigen2cv(cur_keyfrm_->get_translation(), t);

            compute_depth_intervall(cur_keyfrm_, min_depth, max_depth);
            get_landmark_coords(cur_keyfrm_, lms_keyfrm);
            DepthmapEstimator_->AddView(R, t, gray, mask, lms_keyfrm, max_depth, min_depth);

            prev_keyfrm = cur_keyfrm_;
            cur_keyfrm_ = NULL;
            
            if (DepthmapEstimator_->IsReadyForCompute()) {

                const auto tp_1 = std::chrono::steady_clock::now();
                dense::DepthmapEstimatorResult *depthResult = new dense::DepthmapEstimatorResult;
                DepthmapEstimator_->ComputePatchMatch(depthResult);
                const auto tp_2 = std::chrono::steady_clock::now();
                const auto compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
                spdlog::info("Time for depthmap compute (cpu + gpu): " + std::to_string(compute_time));
                //depthResults.push_back(*depthResult);

                s_src_frm_id = std::to_string(current_keyfrm_wdw[depth_id]->src_frm_id_);
                depth_id++;
                if (imgs_to_save) {
                    save_keyfrm_depth_and_score(depthResult->depth, depthResult->score, s_src_frm_id, path_to_save);
                }

                
                DepthmapCleaner_->AddView(depthResult); //depthResult.R, depthResult.t, depthResult.depth);
                DepthmapEstimator_->PopHeadUnlessOne();
            }
            
            if (DepthmapCleaner_->IsReadyForCompute()) {
                
                dense::DepthmapCleanerResult* cleaned_depth_Result = new dense::DepthmapCleanerResult;
                DepthmapCleaner_->Clean(cleaned_depth_Result);
                current_keyfrm_wdw[clean_id]->set_cleaned_result(cleaned_depth_Result);
                

                s_src_frm_id = std::to_string(current_keyfrm_wdw[clean_id]->src_frm_id_);
                if (imgs_to_save) {
                    save_cleaned_depth(cleaned_depth_Result->cleaned_depth, s_src_frm_id, path_to_save);
                }

                dense_pointcloud_module_->queue_keyframe(current_keyfrm_wdw[clean_id]);
                DepthmapCleaner_->PopHeadUnlessOne();
                clean_id++;

                /*
                //depthResults.erase(depthResults.begin());
                cleaned_depth_Result = current_keyfrm_wdw[clean_id]->get_cleaned_result();
                
                std::vector<cv::Vec3d> merged_points;
                std::vector<cv::Vec3d> merged_normals;
                std::vector<cv::Vec3d> merged_colors;
                std::vector<cv::Vec2i> merged_keypts;
                cv::Mat rgb = current_keyfrm_wdw[clean_id]->get_rgb_thumb();
                DepthmapPruner_->AddView(cleaned_depth_Result, rgb);
               
                //depthResults.erase(depthResults.begin());

                DepthmapCleaner_->PopHeadUnlessOne();
                clean_id+=1;

                bool bComputed=false;
                int pruned = 0;
                if (DepthmapPruner_->IsReadyForCompute()) {
                    bComputed = true;
                    dense::DepthmapPrunerResult* pruned_depth_result = new dense::DepthmapPrunerResult;
                    DepthmapPruner_->Prune(pruned_depth_result);
                    
                    
                    //if (depthmap_db_->get_num_keyframes() == 1) {
                    //    depthmap_db_->pop_front_keyframe();
                    //}
                    openvslam::data::keyframe* computed_keyfr = current_keyfrm_wdw[clean_id-1-4]; // -2
                    //pruned++;
                    depthmap_db_->add_keyframe(computed_keyfr);                  
                    int c = 0;
                    std::vector<cv::Vec2i>::iterator it_keypts = pruned_depth_result->keypts.begin();
                    for (std::vector<cv::Vec3d>::iterator it = pruned_depth_result->points3d.begin(); it != pruned_depth_result->points3d.end(); it++) {
                        cv::Vec3d point_w = *it;
                        cv::Vec2i keypt = *it_keypts;
                        Vec3_t point_w_eig(point_w(0),point_w(1),point_w(2));
                        nLms++;
                        auto lm = new data::landmark(nLms, computed_keyfr->id_,  point_w_eig, computed_keyfr, 1, 1, depthmap_db_);
                        computed_keyfr->add_keypt_dense(keypt);
                        computed_keyfr->add_landmark_dense(lm, nLms);
                        
                        float c_b = pruned_depth_result->colors[c][0];
                        float c_g = pruned_depth_result->colors[c][1];
                        float c_r = pruned_depth_result->colors[c][2];
                        lm->setColor(c_r, c_g, c_b);
                        depthmap_db_->add_landmark(lm);
                        c++;
                    }
                    //DepthmapPruner_->DebugShiftHead();
                    DepthmapPruner_->PopHead();
                }*/
                
            }
        }
    }
    spdlog::info("terminate depthmap module");
}

void depthmap_computation_module::queue_keyframe(data::keyframe* keyfrm_) {

    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    data::keyframe* keyfrm = new data::keyframe(*keyfrm_);
    
    keyfrms_queue_.push_back(keyfrm);
}

void depthmap_computation_module::queue_keyframes(std::vector<data::keyframe*> &keyfrms) {

    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);

    for (std::vector<data::keyframe*>::iterator it = keyfrms.begin(); it != keyfrms.end(); it++) {
        data::keyframe* keyfrm_ = *it;
        data::keyframe* keyfrm = new data::keyframe(*keyfrm_);
        keyfrms_queue_.push_back(keyfrm);
    }
}

bool depthmap_computation_module::keyframe_is_queued() const {
    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    return (!keyfrms_queue_.empty());
}

//-----------------------------------------
// management for reset process

//! Request to reset the global optimization module
//! (NOTE: this function waits for reset)
void depthmap_computation_module::request_reset() {

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

void depthmap_computation_module::reset() {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    dense_pointcloud_module_->request_reset();
    /*
    for (std::list<openvslam::data::keyframe*>::iterator it=keyfrms_queue_.begin(); it != keyfrms_queue_.end();it++) {
        openvslam::data::keyframe* keyfrm = *it;
        delete keyfrm;
    }
    */
   {
       std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
       keyfrms_queue_.clear();
   }
    
    depthmap_db_->clear();
    reset_is_requested_ = false;
    
    
}

bool depthmap_computation_module::reset_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    return reset_is_requested_;
}

//-----------------------------------------
// management for pause process

//! Request to pause the global optimization module
//! (NOTE: this function does not wait for pause)
void depthmap_computation_module::request_pause() {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    pause_is_requested_ = true;
}

//! Check if the global optimization module is requested to be paused or not
bool depthmap_computation_module::pause_is_requested() const {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    return pause_is_requested_;
}

//! Check if the global optimization module is paused or not
bool depthmap_computation_module::is_paused() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return is_paused_;
    
}

void depthmap_computation_module::pause() {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    is_paused_ = true;
}

//! Resume the global optimization module
void depthmap_computation_module::resume() {

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
void depthmap_computation_module::request_terminate() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    terminate_is_requested_ = true;
}
//! Check if the global optimization module is terminated or not
bool depthmap_computation_module::is_terminated() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return is_terminated_;
}
bool depthmap_computation_module::terminate_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return terminate_is_requested_;
}

void depthmap_computation_module::terminate() {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    std::lock_guard<std::mutex> lock2(mtx_terminate_);
    is_paused_ = true;
    is_terminated_ = true;
}

//-----------------------------------------
// management for loop DC

//! Check if loop DC is running or not
bool depthmap_computation_module::loop_DC_is_running() const {
    return true;
}

//! Abort the loop DC externally
//! (NOTE: this function does not wait for abort)
void depthmap_computation_module::abort_loop_DC() {

}

/*
static bool cmp_keyfm_ids(const data::keyframe* kfrm1, const data::keyframe* kfrm2) {

    bool kfrm1_smaller = true;
    int id1 = kfrm1->id_;
    int id2 = kfrm2->id_;

    kfrm1_smaller = id1 <= id2;

    return kfrm1_smaller;
}
*/

}  // namespace openvslam
