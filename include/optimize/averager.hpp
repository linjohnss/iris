#pragma once
#include "core/math.hpp"
#include "core/util.hpp"

namespace vllm
{
namespace optimize
{
Eigen::Vector3f calcAverageTransform(const Eigen::Matrix3f& R, const Eigen::Vector3f& t, int n)
{
  Eigen::Matrix3f A = Eigen::Matrix3f::Zero();

  for (int i = 0; i < n; i++) {
    Eigen::Matrix3f tmp = Eigen::Matrix3f::Identity();
    for (int j = 0; j < i; j++) {
      tmp = R * tmp;
    }
    A += tmp;
  }

  return A.inverse() * t;
}

Eigen::Matrix4f calcVelocity(const std::list<Eigen::Matrix4f>& poses)
{
  Eigen::Matrix4f V = Eigen::Matrix4f::Identity();

  const int dt = static_cast<int>(poses.size()) - 2;

  Eigen::Matrix4f T0 = getNormalizedPose(*std::next(poses.begin()));
  Eigen::Matrix4f Tn = getNormalizedPose(*std::prev(poses.end()));

  V = T0 * Tn.inverse();
  Eigen::Matrix3f R = V.topLeftCorner(3, 3);
  Eigen::Vector3f t = V.topRightCorner(3, 1);
  Eigen::Matrix3f root_R = so3::exp(so3::log(R) / dt);
  V.topLeftCorner(3, 3) = root_R;
  V.topRightCorner(3, 1) = calcAverageTransform(root_R, t, dt);
  return V;
}
}  // namespace optimize
}  // namespace vllm
