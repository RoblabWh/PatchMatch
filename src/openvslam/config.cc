#include "openvslam/config.h"
#include "openvslam/camera/perspective.h"
#include "openvslam/camera/fisheye.h"
#include "openvslam/camera/equirectangular.h"

#include <iostream>
#include <memory>
//#include <filesystem>

#include <spdlog/spdlog.h>

namespace openvslam {

config::config(const std::string& config_file_path)
    : config(YAML::LoadFile(config_file_path), config_file_path) {}

config::config(const YAML::Node& yaml_node, const std::string& config_file_path)
    : config_file_path_(config_file_path), yaml_node_(yaml_node) {
    spdlog::debug("CONSTRUCT: config");

    spdlog::info("config file loaded: {}", config_file_path_);

    //========================//
    // Load Camera Parameters //
    //========================//

    spdlog::debug("load camera model type");
    const auto camera_model_type = camera::base::load_model_type(yaml_node_);

    spdlog::debug("load camera model parameters");
    try {
        switch (camera_model_type) {
            case camera::model_type_t::Perspective: {
                camera_ = new camera::perspective(yaml_node_);
                break;
            }
            case camera::model_type_t::Fisheye: {
                camera_ = new camera::fisheye(yaml_node_);
                break;
            }
            case camera::model_type_t::Equirectangular: {
                camera_ = new camera::equirectangular(yaml_node_);
                break;
            }
        }
    }
    catch (const std::exception& e) {
        spdlog::debug("failed in loading camera model parameters: {}", e.what());
        delete camera_;
        camera_ = nullptr;
        throw;
    }

    //=====================//
    // Load ORB Parameters //
    //=====================//

    spdlog::debug("load ORB parameters");
    try {
        orb_params_ = feature::orb_params(yaml_node_);
    }
    catch (const std::exception& e) {
        spdlog::debug("failed in loading ORB parameters: {}", e.what());
        delete camera_;
        camera_ = nullptr;
        throw;
    }

    //==========================//
    // Load Tracking Parameters //
    //==========================//

    spdlog::debug("load tracking parameters");

    spdlog::debug("load depth threshold");
    if (camera_->setup_type_ == camera::setup_type_t::Stereo || camera_->setup_type_ == camera::setup_type_t::RGBD) {
        // ベースライン長の一定倍より遠いdepthは無視する
        const auto depth_thr_factor = yaml_node_["depth_threshold"].as<double>(40.0);

        switch (camera_->model_type_) {
            case camera::model_type_t::Perspective: {
                auto camera = static_cast<camera::perspective*>(camera_);
                true_depth_thr_ = camera->true_baseline_ * depth_thr_factor;
                break;
            }
            case camera::model_type_t::Fisheye: {
                auto camera = static_cast<camera::fisheye*>(camera_);
                true_depth_thr_ = camera->true_baseline_ * depth_thr_factor;
                break;
            }
            case camera::model_type_t::Equirectangular: {
                throw std::runtime_error("Not implemented: Stereo or RGBD of equirectangular camera model");
            }
        }
    }

    spdlog::debug("load depthmap factor");
    if (camera_->setup_type_ == camera::setup_type_t::RGBD) {
        depthmap_factor_ = yaml_node_["depthmap_factor"].as<double>(1.0);
    }

    //WriteImgsKeyfrmsTo
    //==========================//
    // Load Write ImgsKeyfrms option //
    //==========================//

    spdlog::debug("load option - Write Images from Keyframes");

    try {

        PathImgsKeyfrms_ = yaml_node["Images.WriteKeyfrmsImgsTo"].as<std::string>();
        //std::string PathImgsKeyfrms = yaml_node["Images.WriteKeyfrmsImgsTo"].as<std::string>();
        //PathImgsKeyfrms_ = std::filesystem::absolute(PathImgsKeyfrms);
    }
    catch(const std::error_code& e) {
        spdlog::info("failed in loading option - write Images from keyframes");
        PathImgsKeyfrms_= "" ;
    }
    catch (const std::exception& e) {
        spdlog::info("failed in loading option - write Images from keyframes: {}", e.what());
        PathImgsKeyfrms_= "" ;
    }

    try {
        int TypeKeyfrmsImgs = yaml_node["Images.TypeKeyfrmsImgs"].as<int>();
        switch (TypeKeyfrmsImgs) {
            case 0: TypeKeyfrmsImgs_ = ".png"; break;
            case 1: TypeKeyfrmsImgs_ = ".jpg"; break;
            default: TypeKeyfrmsImgs_ = ".jpg"; break;
        }
    }
    catch (const std::exception& e) {
        spdlog::debug("failed in loading option - image type: {}", e.what());
    }

    try {
        width = yaml_node["Dense.DepthmapWidth"].as<int>();
        height = yaml_node["Dense.DepthmapHeight"].as<int>();
        if (width <= 0 || height <= 0) {
            throw;
        }
    }
    catch (const std::exception& e) {
        spdlog::debug("failed in loading option - sizes of depthmaps: {}", e.what());
        width = 640;
        height = 320;
    }

    try {
        SameDepthThreshold = yaml_node["Dense.SameDepthThreshold"].as<float>();
        if (SameDepthThreshold < 0) {
            throw;
        }
    }
    catch (const std::exception& e) {
        spdlog::debug("failed in loading option - Same Depth Threshold: {}", e.what());
        SameDepthThreshold = 0.03;
    }

    try {
        MinPatchSD = yaml_node["Dense.MinPatchSD"].as<float>();
        if (MinPatchSD < 0) {
            throw;
        }
    }
    catch (const std::exception& e) {
        spdlog::debug("failed in loading option - Min Patch SD: {}", e.what());
        MinPatchSD = 0;
    }

    try {
        Iterations = yaml_node["Dense.Iterations"].as<int>();
        if (Iterations < 0) {
            throw;
        }
    }
    catch (const std::exception& e) {
        spdlog::debug("failed in loading option - PatchSize: {}", e.what());
        Iterations = 2;
    }

    try {
        PatchSize = 7;//yaml_node["Dense.PatchSize"].as<float>();
        if (PatchSize < 3 && PatchSize%2==0) {
            throw;
        }
    }
    catch (const std::exception& e) {
        spdlog::debug("failed in loading option - PatchSize: {}", e.what());
        PatchSize = 7;
    }
}

config::~config() {
    delete camera_;
    camera_ = nullptr;

    spdlog::debug("DESTRUCT: config");
}

std::ostream& operator<<(std::ostream& os, const config& cfg) {
    std::cout << "Camera Configuration:" << std::endl;
    cfg.camera_->show_parameters();

    std::cout << "ORB Configuration:" << std::endl;
    cfg.orb_params_.show_parameters();

    if (cfg.camera_->setup_type_ == camera::setup_type_t::Stereo || cfg.camera_->setup_type_ == camera::setup_type_t::RGBD) {
        std::cout << "Stereo Configuration:" << std::endl;
        std::cout << "- true baseline: " << cfg.camera_->true_baseline_ << std::endl;
        std::cout << "- true depth threshold: " << cfg.true_depth_thr_ << std::endl;
        std::cout << "- depth threshold factor: " << cfg.true_depth_thr_ / cfg.camera_->true_baseline_ << std::endl;
    }
    if (cfg.camera_->setup_type_ == camera::setup_type_t::RGBD) {
        std::cout << "Depth Image Configuration:" << std::endl;
        std::cout << "- depthmap factor: " << cfg.depthmap_factor_ << std::endl;
    }

    return os;
}

} // namespace openvslam
