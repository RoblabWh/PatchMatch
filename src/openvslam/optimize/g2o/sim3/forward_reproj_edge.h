#ifndef OPENVSLAM_OPTIMIZE_G2O_FORWARD_REPROJ_EDGE_H
#define OPENVSLAM_OPTIMIZE_G2O_FORWARD_REPROJ_EDGE_H

#include "openvslam/type.h"
#include "openvslam/optimize/g2o/sim3/transform_vertex.h"

#include <g2o/core/base_unary_edge.h>

namespace openvslam {
namespace optimize {
namespace g2o {
namespace sim3 {

class base_forward_reproj_edge : public ::g2o::BaseUnaryEdge<2, Vec2_t, transform_vertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    base_forward_reproj_edge();

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void computeError() final {
        // vertexからSim3(2->1)を取り出す
        const auto v1 = static_cast<const transform_vertex*>(_vertices.at(0));
        const ::g2o::Sim3& Sim3_12 = v1->estimate();
        // vertexからSE3(world->2)を取り出す
        const Mat33_t& rot_2w = v1->rot_2w_;
        const Vec3_t& trans_2w = v1->trans_2w_;

        // 点の座標系を変換(world->2)
        const Vec3_t pos_2 = rot_2w * pos_w_ + trans_2w;
        // Sim3を使ってさらに変換(2->1)
        const Vec3_t pos_1 = Sim3_12.map(pos_2);
        // 再投影誤差を計算
        const Vec2_t obs(_measurement);
        _error = obs - cam_project(pos_1);
    }

    virtual Vec2_t cam_project(const Vec3_t& pos_c) const = 0;

    Vec3_t pos_w_;
};

class perspective_forward_reproj_edge final : public base_forward_reproj_edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    perspective_forward_reproj_edge();

    inline Vec2_t cam_project(const Vec3_t& pos_c) const override {
        return {fx_ * pos_c(0) / pos_c(2) + cx_, fy_ * pos_c(1) / pos_c(2) + cy_};
    }

    double fx_, fy_, cx_, cy_;
};

class equirectangular_forward_reproj_edge final : public base_forward_reproj_edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    equirectangular_forward_reproj_edge();

    inline Vec2_t cam_project(const Vec3_t& pos_c) const override {
        const double theta = std::atan2(pos_c(0), pos_c(2));
        const double phi = -std::asin(pos_c(1) / pos_c.norm());
        return {cols_ * (0.5 + theta / (2 * M_PI)), rows_ * (0.5 - phi / M_PI)};
    }

    double cols_, rows_;
};

} // namespace sim3
} // namespace g2o
} // namespace optimize
} // namespace openvslam

#endif // OPENVSLAM_OPTIMIZE_G2O_FORWARD_REPROJ_EDGE_H
