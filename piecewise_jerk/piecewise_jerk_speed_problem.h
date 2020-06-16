/**
 * only modify base on apollo autocar
 */

#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include "piecewise_jerk_problem.h"

namespace math {
namespace piecewise {

/*
 * @brief:
 * This class solve the path time optimization problem:
 * s
 * |
 * |                       P(t1, s1)  P(t2, s2)
 * |            P(t0, s0)                       ... P(t(k-1), s(k-1))
 * |P(start)
 * |
 * |________________________________________________________ t
 *
 * we suppose t(k+1) - t(k) == t(k) - t(k-1)
 *
 * Given the s, s', s'' at P(start),  The goal is to find t0, t1, ... t(k-1)
 * which makes the line P(start), P0, P(1) ... P(k-1) "smooth".
 */

class PiecewiseJerkSpeedProblem : public PiecewiseJerkProblem {
 public:
  PiecewiseJerkSpeedProblem(const size_t num_of_knots, const double delta_s);

  virtual ~PiecewiseJerkSpeedProblem() = default;

  //   void set_dx_ref(const double weight_dx_ref, const double dx_ref);

  //   void set_penalty_dx(std::vector<double> penalty_dx);

  void SetConsition(const std::array<double, 3>& init_state,
                    const std::array<double, 3>& end_state);

 protected:
  // naming convention follows osqp solver.
  void CalculateKernel(std::vector<c_float>* P_data,
                       std::vector<c_int>* P_indices,
                       std::vector<c_int>* P_indptr) override;

  void CalculateAffineConstraint(std::vector<c_float>* A_data,
                                 std::vector<c_int>* A_indices,
                                 std::vector<c_int>* A_indptr,
                                 std::vector<c_float>* lower_bounds,
                                 std::vector<c_float>* upper_bounds) override;

  OSQPSettings* SolverDefaultSettings() override;
};

}  // namespace piecewise
}  // namespace math
