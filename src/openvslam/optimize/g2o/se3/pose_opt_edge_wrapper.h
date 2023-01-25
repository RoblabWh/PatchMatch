#ifndef OPENVSLAM_OPTIMIZE_G2O_SE3_POSE_OPT_EDGE_WRAPPER_H
#define OPENVSLAM_OPTIMIZE_G2O_SE3_POSE_OPT_EDGE_WRAPPER_H

#include "openvslam/camera/perspective.h"
#include "openvslam/camera/fisheye.h"
#include "openvslam/camera/equirectangular.h"
#include "openvslam/optimize/g2o/se3/perspective_pose_opt_edge.h"
#include "openvslam/optimize/g2o/se3/equirectangular_pose_opt_edge.h"

#include <g2o/core/robust_kernel_impl.h>

namespace openvslam {

namespace data {
class landmark;
} // namespace data

namespace optimize {
namespace g2o {
namespace se3 {

template<typename T>
class pose_opt_edge_wrapper {
public:
    pose_opt_edge_wrapper() = delete;

    pose_opt_edge_wrapper(T* shot, shot_vertex* shot_vtx, const Vec3_t& pos_w,
                          const unsigned int idx, const float obs_x, const float obs_y, const float obs_x_right,
                          const float inv_sigma_sq, const float sqrt_chi_sq);

    virtual ~pose_opt_edge_wrapper() = default;

    inline bool is_inlier() const {
        return edge_->level() == 0;
    }

    inline bool is_outlier() const {
        return edge_->level() != 0;
    }

    inline void set_as_inlier() const {
        edge_->setLevel(0);
    }

    inline void set_as_outlier() const {
        edge_->setLevel(1);
    }

    inline bool depth_is_positive() const;

    ::g2o::OptimizableGraph::Edge* edge_;

    camera::base* camera_;
    T* shot_;
    const unsigned int idx_;
    const bool is_monocular_;
};

template<typename T>
pose_opt_edge_wrapper<T>::pose_opt_edge_wrapper(T* shot, shot_vertex* shot_vtx, const Vec3_t& pos_w,
                                                const unsigned int idx, const float obs_x, const float obs_y, const float obs_x_right,
                                                const float inv_sigma_sq, const float sqrt_chi_sq)
    : camera_(shot->camera_), shot_(shot), idx_(idx), is_monocular_(obs_x_right < 0) {
    // 拘束条件を設定
    switch (camera_->model_type_) {
        case camera::model_type_t::Perspective: {
            auto c = static_cast<camera::perspective*>(camera_);
            if (is_monocular_) {
                auto edge = new mono_perspective_pose_opt_edge();

                const Vec2_t obs{obs_x, obs_y};
                edge->setMeasurement(obs);
                edge->setInformation(Mat22_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;

                edge->pos_w_ = pos_w;

                edge->setVertex(0, shot_vtx);

                edge_ = edge;
            }
            else {
                auto edge = new stereo_perspective_pose_opt_edge();

                const Vec3_t obs{obs_x, obs_y, obs_x_right};
                edge->setMeasurement(obs);
                edge->setInformation(Mat33_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;
                edge->focal_x_baseline_ = camera_->focal_x_baseline_;

                edge->pos_w_ = pos_w;

                edge->setVertex(0, shot_vtx);

                edge_ = edge;
            }
            break;
        }
        case camera::model_type_t::Fisheye: {
            auto c = static_cast<camera::fisheye*>(camera_);
            if (is_monocular_) {
                auto edge = new mono_perspective_pose_opt_edge();

                const Vec2_t obs{obs_x, obs_y};
                edge->setMeasurement(obs);
                edge->setInformation(Mat22_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;

                edge->pos_w_ = pos_w;

                edge->setVertex(0, shot_vtx);

                edge_ = edge;
            }
            else {
                auto edge = new stereo_perspective_pose_opt_edge();

                const Vec3_t obs{obs_x, obs_y, obs_x_right};
                edge->setMeasurement(obs);
                edge->setInformation(Mat33_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;
                edge->focal_x_baseline_ = camera_->focal_x_baseline_;

                edge->pos_w_ = pos_w;

                edge->setVertex(0, shot_vtx);

                edge_ = edge;
            }
            break;
        }
        case camera::model_type_t::Equirectangular: {
            assert(is_monocular_);

            auto c = static_cast<camera::equirectangular*>(camera_);

            auto edge = new equirectangular_pose_opt_edge();

            const Vec2_t obs{obs_x, obs_y};
            edge->setMeasurement(obs);
            edge->setInformation(Mat22_t::Identity() * inv_sigma_sq);

            edge->cols_ = c->cols_;
            edge->rows_ = c->rows_;

            edge->pos_w_ = pos_w;

            edge->setVertex(0, shot_vtx);

            edge_ = edge;

            break;
        }
    }

    // loss functionを設定
    auto huber_kernel = new ::g2o::RobustKernelHuber();
    huber_kernel->setDelta(sqrt_chi_sq);
    edge_->setRobustKernel(huber_kernel);
}

template<typename T>
bool pose_opt_edge_wrapper<T>::depth_is_positive() const {
    switch (camera_->model_type_) {
        case camera::model_type_t::Perspective: {
            if (is_monocular_) {
                return static_cast<mono_perspective_pose_opt_edge*>(edge_)->mono_perspective_pose_opt_edge::depth_is_positive();
            }
            else {
                return static_cast<stereo_perspective_pose_opt_edge*>(edge_)->stereo_perspective_pose_opt_edge::depth_is_positive();
            }
        }
        case camera::model_type_t::Fisheye: {
            if (is_monocular_) {
                return static_cast<mono_perspective_pose_opt_edge*>(edge_)->mono_perspective_pose_opt_edge::depth_is_positive();
            }
            else {
                return static_cast<stereo_perspective_pose_opt_edge*>(edge_)->stereo_perspective_pose_opt_edge::depth_is_positive();
            }
        }
        case camera::model_type_t::Equirectangular: {
            return true;
        }
    }

    return true;
}

} // namespace se3
} // namespace g2o
} // namespace optimize
} // namespace openvslam

#endif // OPENVSLAM_OPTIMIZE_G2O_SE3_POSE_OPT_EDGE_WRAPPER_H
