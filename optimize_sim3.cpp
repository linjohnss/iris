#include "types_icp.hpp"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/stuff/sampler.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include <iostream>
#include <random>
#include <stdint.h>

void setVertexSE3(g2o::SparseOptimizer& optimizer)
{
  // set up rotation and translation for this node
  Eigen::Vector3d t(0, 0, 0);
  Eigen::Quaterniond q;
  q.setIdentity();

  Eigen::Isometry3d cam;  // camera pose
  cam = q;
  cam.translation() = t;

  // set up initial parameter
  g2o::VertexSE3* vc = new g2o::VertexSE3();
  vc->setEstimate(cam);
  vc->setId(0);  // vertex id
  std::cerr << t.transpose() << " | " << q.coeffs().transpose() << std::endl;

  // set first cam pose fixed
  vc->setFixed(true);

  // add to optimizer
  optimizer.addVertex(vc);
}

void setVertexSim3(g2o::SparseOptimizer& optimizer)
{
  // set up rotation and translation for this node
  Eigen::Vector3d t(0, 0, 1);
  Eigen::Quaterniond q;
  q.setIdentity();

  double r = 1.0;
  g2o::Sim3 sim3(q, t, r);

  // set up initial parameter
  g2o::VertexSim3Expmap* vc = new g2o::VertexSim3Expmap();
  vc->setEstimate(sim3);
  vc->setId(1);  // vertex id
  std::cerr << t.transpose() << " | " << q.coeffs().transpose() << " | " << r << std::endl;

  // add to optimizer
  optimizer.addVertex(vc);
}

int main()
{
  // noise in position[m]
  double euc_noise = 0.01;
  //  double outlier_ratio = 0.1;

  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(true);

  // variable-size block solver
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<g2o::BlockSolverX>(g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>()));
  optimizer.setAlgorithm(solver);


  // point cloud in R^3
  constexpr int N = 1000;
  std::vector<Eigen::Vector3d> true_points;
  for (size_t i = 0; i < N; ++i) {
    true_points.push_back(Eigen::Vector3d(
        (g2o::Sampler::uniformRand(0.0, 1.0) - 0.5) * 3,
        g2o::Sampler::uniformRand(0.0, 1.0) - 0.5,
        g2o::Sampler::uniformRand(0.0, 1.0) + 10));
  }

  setVertexSE3(optimizer);
  setVertexSim3(optimizer);

  for (size_t i = 0; i < true_points.size(); ++i) {
    // get Vertex
    g2o::VertexSE3* vp0 = dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(0)->second);
    g2o::VertexSim3Expmap* vp1 = dynamic_cast<g2o::VertexSim3Expmap*>(optimizer.vertices().find(1)->second);

    // calculate the relative 3D position of the point
    Eigen::Vector3d pt0, pt1;
    pt0 = vp0->estimate().inverse() * true_points[i];
    pt1 = vp1->estimate().inverse().map(true_points[i]);

    // add in noise
    pt0 += Eigen::Vector3d(
        g2o::Sampler::gaussRand(0.0, euc_noise),
        g2o::Sampler::gaussRand(0.0, euc_noise),
        g2o::Sampler::gaussRand(0.0, euc_noise));
    pt1 += Eigen::Vector3d(
        g2o::Sampler::gaussRand(0.0, euc_noise),
        g2o::Sampler::gaussRand(0.0, euc_noise),
        g2o::Sampler::gaussRand(0.0, euc_noise));

    // form edge, with normals in varioius positions
    Eigen::Vector3d nm0, nm1;
    nm0 << 0, static_cast<double>(i), 1;
    nm1 << 0, static_cast<double>(i), 1;
    nm0.normalize();
    nm1.normalize();

    // new edge with correct cohort for caching
    LLVM::Edge_Sim3_GICP* e = new LLVM::Edge_Sim3_GICP();

    e->setVertex(0, vp0);  // first viewpoint
    e->setVertex(1, vp1);  // second viewpoint

    LLVM::EdgeGICP meas;
    meas.pos0 = pt0;
    meas.pos1 = pt1;
    meas.normal0 = nm0;
    meas.normal1 = nm1;

    e->setMeasurement(meas);

    // meas = e->measurement();
    // e->information() = meas.prec0(0.01);  // use this for point-plane
    e->information().setIdentity();  // use this for point-point

    // set Huber kernel (default delta = 1.0)
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);

    optimizer.addEdge(e);
  }

  // move second cam off of its true position
  g2o::VertexSim3Expmap* vc = dynamic_cast<g2o::VertexSim3Expmap*>(optimizer.vertices().find(1)->second);
  g2o::Sim3 sim3 = vc->estimate();
  sim3.translation() = Eigen::Vector3d(0, 0, 0.2);
  sim3.scale() = 2.0;
  vc->setEstimate(sim3);

  optimizer.initializeOptimization();
  optimizer.computeActiveErrors();
  std::cout << "Initial chi2 = " << FIXED(optimizer.chi2()) << std::endl;

  optimizer.setVerbose(true);
  optimizer.optimize(10);

  // clang-format off
  std::cout << std::endl
            << "Second vertex should be near 0,0,1" 
            << std::endl;
  std::cout << dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(0)->second)->estimate().translation().transpose()
            << std::endl;
  std::cout << dynamic_cast<g2o::VertexSim3Expmap*>(optimizer.vertices().find(1)->second)->estimate().translation().transpose()
            << std::endl;
  std::cout << dynamic_cast<g2o::VertexSim3Expmap*>(optimizer.vertices().find(1)->second)->estimate().scale()
            << std::endl;
  std::cout << dynamic_cast<g2o::VertexSim3Expmap*>(optimizer.vertices().find(1)->second)->estimate()
            << std::endl;
  // clang-format on
}