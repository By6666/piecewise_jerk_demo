/**
 * only modify base on apollo autocar
 */

#pragma once

#include <tuple>
#include <utility>
#include <vector>
#include <time.h>
#include <iostream>
#include <algorithm>

#include "osqp/osqp.h"
#include "linear_layer.h"

namespace math {
namespace piecewise {

/*
 * @brief:
 * This class solve an optimization problem:
 * x
 * |
 * |                       P(s1, x1)  P(s2, x2)
 * |            P(s0, x0)                       ... P(s(k-1), x(k-1))
 * |P(start)
 * |
 * |________________________________________________________ s
 *
 * we suppose s(k+1) - s(k) == s(k) - s(k-1)
 *
 * Given the x, x', x'' at P(start),  The goal is to find x0, x1, ... x(k-1)
 * which makes the line P(start), P0, P(1) ... P(k-1) "smooth".
 */

typedef std::pair<bool, std::string> Status;

enum OptimizeStatus {
  SUCCESS = 0,
  FAIL = 1,
  OVERMAXITERATION = 2,
};

class PiecewiseJerkProblem {
 public:
  PiecewiseJerkProblem(const size_t num_of_knots, const double delta_s);

  explicit PiecewiseJerkProblem(const DifferentialLayer& layer);

  virtual ~PiecewiseJerkProblem() = default;

  void set_x_bounds(std::vector<std::pair<double, double>> x_bounds);

  void set_x_bounds(const double x_lower_bound, const double x_upper_bound);

  void set_dx_bounds(std::vector<std::pair<double, double>> dx_bounds);

  void set_dx_bounds(const double dx_lower_bound, const double dx_upper_bound);

  void set_ddx_bounds(std::vector<std::pair<double, double>> ddx_bounds);

  void set_ddx_bounds(const double ddx_lower_bound,
                      const double ddx_upper_bound);

  void set_dddx_bound(const double dddx_bound);

  void set_dddx_bound(const double dddx_lower_bound,
                      const double dddx_upper_bound);

  void set_weight_x(const double weight_x);

  void set_weight_dx(const double weight_dx);

  void set_weight_ddx(const double weight_ddx);

  void set_weight_dddx(const double weight_dddx);

  void set_scale_factor(const std::array<double, 3>& scale_factor) {
    scale_factor_ = scale_factor;
  }

  void set_x_ref(const double weight_x_ref, std::vector<double> x_ref);

  void set_end_state_ref(const std::array<double, 3>& weight_end_state,
                         const std::array<double, 3>& end_state_ref);

  virtual Status Optimize(const int max_iter = 4000);

  const std::vector<double>& opt_x() const { return x_; }

  const std::vector<double>& opt_dx() const { return dx_; }

  const std::vector<double>& opt_ddx() const { return ddx_; }

  const OSQPInfo solve_info() const { return solve_info_; }

  void set_solve_info(const OSQPInfo solve_info) { solve_info_ = solve_info; }

 protected:
  // naming convention follows osqp solver.
  virtual void CalculateKernel(std::vector<c_float>* P_data,
                               std::vector<c_int>* P_indices,
                               std::vector<c_int>* P_indptr);

  virtual void CalculateOffset(std::vector<c_float>* q);

  virtual void CalculateAffineConstraint(std::vector<c_float>* A_data,
                                         std::vector<c_int>* A_indices,
                                         std::vector<c_int>* A_indptr,
                                         std::vector<c_float>* lower_bounds,
                                         std::vector<c_float>* upper_bounds);

  virtual OSQPSettings* SolverDefaultSettings();

  OSQPData* FormulateProblem();

  void FreeData(OSQPData* data);

  template <typename T>
  T* CopyData(const std::vector<T>& vec) {
    T* data = new T[vec.size()];
    memcpy(data, vec.data(), sizeof(T) * vec.size());
    return data;
  }

 protected:
  double delta_s_;
  size_t num_of_knots_;
  DifferentialLayer layers_;
  std::array<double, 3> scale_factor_ = {{1.0, 1.0, 1.0}};

  // output
  std::vector<double> x_;
  std::vector<double> dx_;
  std::vector<double> ddx_;

  OSQPInfo solve_info_;
};

}  // namespace piecewise
}  // namespace math
