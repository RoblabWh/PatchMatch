#ifndef OPENVSLAM_DENSE_POINTCLOUD_MODULE_H
#define OPENVSLAM_DENSE_POINTCLOUD_MODULE_H

#include "openvslam/config.h"
#include "openvslam/depthmap_computation_module.h"
#include "openvslam/depthmap.h"


#include <list>
#include <mutex>
#include <thread>
#include <memory>

namespace openvslam {

class depthmap_computation_module;

namespace data {
class keyframe;
class map_database;

} // namespace data


class dense_pointcloud_module {
public:
    //! Constructor
    dense_pointcloud_module(const std::shared_ptr<config>& cfg, data::map_database* map_db);

    //! Destructor
    ~dense_pointcloud_module();

    //! Set ?? module
    //void set_depthmap_compute_module(depthmap_computation_module* depthmap_estimator);

    //-----------------------------------------
    // main process

    //! Run main loop of the global optimization module
    void run();

    //! Queue a keyframe to the BoW database
    void queue_keyframe(data::keyframe* keyfrm);

    void queue_keyframes(std::vector<data::keyframe*> &keyfrms);
    void save_pruned_depth(cv::Mat &depth_pruned,  std::string &id, std::string &path_to_save);

    //-----------------------------------------
    // management for reset process

    //! Request to reset the global optimization module
    //! (NOTE: this function waits for reset)
    void request_reset();

    //-----------------------------------------
    // management for pause process

    //! Request to pause the global optimization module
    //! (NOTE: this function does not wait for pause)
    void request_pause();

    //! Check if the global optimization module is requested to be paused or not
    bool pause_is_requested() const;

    //! Check if the global optimization module is paused or not
    bool is_paused() const;

    //! Resume the global optimization module
    void resume();

    //-----------------------------------------
    // management for terminate process

    //! Request to terminate the global optimization module
    //! (NOTE: this function does not wait for terminate)
    void request_terminate();

    //! Check if the global optimization module is terminated or not
    bool is_terminated() const;

    //-----------------------------------------
    // management for loop DC

    //! Check if loop DC is running or not
    bool loop_DC_is_running() const;

    //! Abort the loop DC externally
    //! (NOTE: this function does not wait for abort)
    void abort_loop_DC();

private:
    //-----------------------------------------
    // main process

    //! config
    const std::shared_ptr<config> cfg_;

    //-----------------------------------------
    // database

    //! map database
    data::map_database* depthmap_db_ = nullptr;

    //-----------------------------------------
    // management for reset process

    //! mutex for access to reset procedure
    mutable std::mutex mtx_reset_;

    //! Check and execute reset
    bool reset_is_requested() const;

    //! Reset the global optimization module
    void reset();

    //! flag which indicates whether reset is requested or not
    bool reset_is_requested_ = false;

    //-----------------------------------------
    // management for pause process

    //! mutex for access to pause procedure
    mutable std::mutex mtx_pause_;

    //! Pause the global optimizer
    void pause();

    //! flag which indicates termination is requested or not
    bool pause_is_requested_ = false;
    //! flag which indicates whether the main loop is paused or not
    bool is_paused_ = false;

    //-----------------------------------------
    // management for terminate process

    //! mutex for access to terminate procedure
    mutable std::mutex mtx_terminate_;

    //! Check if termination is requested or not
    bool terminate_is_requested() const;

    //! Raise the flag which indicates the main loop has been already terminated
    void terminate();

    //! flag which indicates termination is requested or not
    bool terminate_is_requested_ = false;
    //! flag which indicates whether the main loop is terminated or not
    bool is_terminated_ = true;
    //-----------------------------------------
    // modules

    //! ?? module
    //mapping_module* mapper_ = nullptr;

    //-----------------------------------------
    // keyframe queue

    //! mutex for access to keyframe queue
    mutable std::mutex mtx_keyfrm_queue_;

    //! Check if keyframe is queued
    bool keyframe_is_queued() const;

    //! queue for keyframes
    std::list<data::keyframe*> keyfrms_queue_;
    std::unique_ptr<dense::DepthmapPruner> DepthmapPruner_ = nullptr;

};

} // namespace openvslam

#endif // OPENVSLAM_DEPTHMAP_COMPUATION_MODULE_H
