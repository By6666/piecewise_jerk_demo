/**
 * only modify base on apollo autocar
 */

#include "piecewise_jerk_speed_problem.h"

namespace math {
namespace piecewise {

PiecewiseJerkSpeedProblem::PiecewiseJerkSpeedProblem(const size_t num_of_knots,
                                                     const double delta_s)
    : PiecewiseJerkProblem(num_of_knots, delta_s) {}

void PiecewiseJerkSpeedProblem::CalculateKernel(std::vector<c_float>* P_data,
                                                std::vector<c_int>* P_indices,
                                                std::vector<c_int>* P_indptr) {
  const int n = static_cast<int>(num_of_knots_);
  const int kNumParam = 3 * n;
  const int kNumValue = 4 * n - 1;
  std::vector<std::vector<std::pair<c_int, c_float>>> columns;
  columns.resize(kNumParam);
  int value_index = 0;

  // x(i)^2 * w_x_ref
  for (int i = 0; i < n; ++i) {
    columns[i].emplace_back(
        i, layers_[0].Weight().at(i) / (scale_factor_[0] * scale_factor_[0]));
    ++value_index;
  }

  // x(i)'^2 * (w_dx_ref + penalty_dx)
  for (int i = 0; i < n; ++i) {
    columns[n + i].emplace_back(
        n + i,
        layers_[1].Weight().at(i) / (scale_factor_[1] * scale_factor_[1]));
    ++value_index;
  }

  auto delta_s_square = delta_s_ * delta_s_;
  // x(n)''^2 * (w_ddx + 2 * w_dddx / delta_s^2)
  columns[2 * n].emplace_back(
      2 * n,
      (layers_[2].Weight().at(0) + layers_[3].Weight().at(0) / delta_s_square) /
          (scale_factor_[2] * scale_factor_[2]));
  ++value_index;

  for (int i = 1; i < n - 1; ++i) {
    columns[2 * n + i].emplace_back(
        2 * n + i, (layers_[2].Weight().at(i) +
                    2.0 * layers_[3].Weight().at(i) / delta_s_square) /
                       (scale_factor_[2] * scale_factor_[2]));
    ++value_index;
  }

  columns[3 * n - 1].emplace_back(
      3 * n - 1, (layers_[2].Weight().at(n - 1) +
                  layers_[3].Weight().at(n - 1) / delta_s_square) /
                     (scale_factor_[2] * scale_factor_[2]));
  ++value_index;

  // -2 * w_dddx / delta_s^2 * x(i)'' * x(i + 1)''
  for (int i = 0; i < n - 1; ++i) {
    columns[2 * n + i].emplace_back(
        2 * n + i + 1, -1.0 * layers_[3].Weight().at(i) / delta_s_square /
                           (scale_factor_[2] * scale_factor_[2]));
    ++value_index;
  }

  CHECK_EQ(value_index, kNumValue);

  int ind_p = 0;
  for (int i = 0; i < kNumParam; ++i) {
    P_indptr->push_back(ind_p);
    for (const auto& row_data_pair : columns[i]) {
      P_data->push_back(row_data_pair.second * 2.0);
      P_indices->push_back(row_data_pair.first);
      ++ind_p;
    }
  }
  P_indptr->push_back(ind_p);
}

void PiecewiseJerkSpeedProblem::CalculateAffineConstraint(
    std::vector<c_float>* A_data, std::vector<c_int>* A_indices,
    std::vector<c_int>* A_indptr, std::vector<c_float>* lower_bounds,
    std::vector<c_float>* upper_bounds) {
  // 3N params bounds on x, x', x''
  // 3(N-1) constraints on x, x', x''
  const int n = static_cast<int>(num_of_knots_);
  const int num_of_variables = 3 * n;
  const int num_of_constraints = num_of_variables + 3 * (n - 1);
  lower_bounds->resize(num_of_constraints);
  upper_bounds->resize(num_of_constraints);

  std::vector<std::vector<std::pair<c_int, c_float>>> variables(
      num_of_variables);

  int constraint_index = 0;
  // set x, x', x'' bounds
  for (int i = 0; i < num_of_variables; ++i) {
    if (i < n) {
      variables[i].emplace_back(constraint_index, 1.0);
      lower_bounds->at(constraint_index) =
          layers_[0].Boundary().at(i).first * scale_factor_[0];
      upper_bounds->at(constraint_index) =
          layers_[0].Boundary().at(i).second * scale_factor_[0];
    } else if (i < 2 * n) {
      variables[i].emplace_back(constraint_index, 1.0);

      lower_bounds->at(constraint_index) =
          layers_[1].Boundary().at(i - n).first * scale_factor_[1];
      upper_bounds->at(constraint_index) =
          layers_[1].Boundary().at(i - n).second * scale_factor_[1];
    } else {
      variables[i].emplace_back(constraint_index, 1.0);
      lower_bounds->at(constraint_index) =
          layers_[2].Boundary().at(i - 2 * n).first * scale_factor_[2];
      upper_bounds->at(constraint_index) =
          layers_[2].Boundary().at(i - 2 * n).second * scale_factor_[2];
    }
    ++constraint_index;
  }
  CHECK_EQ(constraint_index, num_of_variables);

  // x(i->i+1)''' = (x(i+1)'' - x(i)'') / delta_s
  for (int i = 0; i + 1 < n; ++i) {
    variables[2 * n + i].emplace_back(constraint_index, -1.0);
    variables[2 * n + i + 1].emplace_back(constraint_index, 1.0);
    lower_bounds->at(constraint_index) =
        layers_[3].Boundary().at(0).first * delta_s_ * scale_factor_[2];
    upper_bounds->at(constraint_index) =
        layers_[3].Boundary().at(0).second * delta_s_ * scale_factor_[2];
    ++constraint_index;
  }

  // x(i+1)' - x(i)' - 0.5 * delta_s * x(i)'' - 0.5 * delta_s * x(i+1)'' = 0
  for (int i = 0; i + 1 < n; ++i) {
    variables[n + i].emplace_back(constraint_index, -1.0 * scale_factor_[2]);
    variables[n + i + 1].emplace_back(constraint_index, 1.0 * scale_factor_[2]);
    variables[2 * n + i].emplace_back(constraint_index,
                                      -0.5 * delta_s_ * scale_factor_[1]);
    variables[2 * n + i + 1].emplace_back(constraint_index,
                                          -0.5 * delta_s_ * scale_factor_[1]);
    lower_bounds->at(constraint_index) = 0.0;
    upper_bounds->at(constraint_index) = 0.0;
    ++constraint_index;
  }

  // x(i+1) - x(i) - delta_s * x(i)'
  // - 1/3 * delta_s^2 * x(i)'' - 1/6 * delta_s^2 * x(i+1)''
  auto delta_s_sq_ = delta_s_ * delta_s_;
  for (int i = 0; i + 1 < n; ++i) {
    variables[i].emplace_back(constraint_index,
                              -1.0 * scale_factor_[1] * scale_factor_[2]);
    variables[i + 1].emplace_back(constraint_index,
                                  1.0 * scale_factor_[1] * scale_factor_[2]);
    variables[n + i].emplace_back(
        constraint_index, -delta_s_ * scale_factor_[0] * scale_factor_[2]);
    variables[2 * n + i].emplace_back(
        constraint_index,
        -delta_s_sq_ / 3.0 * scale_factor_[0] * scale_factor_[1]);
    variables[2 * n + i + 1].emplace_back(
        constraint_index,
        -delta_s_sq_ / 6.0 * scale_factor_[0] * scale_factor_[1]);

    lower_bounds->at(constraint_index) = 0.0;
    upper_bounds->at(constraint_index) = 0.0;
    ++constraint_index;
  }

  CHECK_EQ(constraint_index, num_of_constraints);

  int ind_p = 0;
  for (int i = 0; i < num_of_variables; ++i) {
    A_indptr->push_back(ind_p);
    for (const auto& variable_nz : variables[i]) {
      // coefficient
      A_data->push_back(variable_nz.second);

      // constraint index
      A_indices->push_back(variable_nz.first);
      ++ind_p;
    }
  }
  // We indeed need this line because of
  // https://github.com/oxfordcontrol/osqp/blob/master/src/cs.c#L255
  A_indptr->push_back(ind_p);
}

OSQPSettings* PiecewiseJerkSpeedProblem::SolverDefaultSettings() {
  // Define Solver default settings
  OSQPSettings* settings =
      reinterpret_cast<OSQPSettings*>(c_malloc(sizeof(OSQPSettings)));
  osqp_set_default_settings(settings);
  settings->eps_abs = 1e-4;
  settings->eps_rel = 1e-4;
  settings->eps_prim_inf = 1e-5;
  settings->eps_dual_inf = 1e-5;
  settings->polish = true;
  settings->verbose = true;
  settings->scaled_termination = false;
  settings->scaling = false;
  settings->adaptive_rho = true;
  settings->adaptive_rho_interval = 25;

  return settings;
}

void PiecewiseJerkSpeedProblem::SetConsition(
    const std::array<double, 3>& init_state,
    const std::array<double, 3>& end_state) {
  set_x_bounds(-1.0, end_state[0]);
  set_dx_bounds(0.0, init_state[1]);
  set_ddx_bounds(-6.0, 6.0);
  set_dddx_bound(-4.0, 4.0);

  set_weight_x(1e-4);
  set_weight_dx(1e-6);
  set_weight_ddx(1.0);
  set_weight_dddx(2);

  // set start init hard constrain
  CHECK_GE(init_state.size(), layers_.Rank());
  for (int i = 0; i < init_state.size(); ++i) {
    layers_[i].SetBoundary({init_state[i], init_state[i]}, 0);
  }

  // set end hard constrain
  for (int i = 1; i < end_state.size(); ++i) {
    layers_[i].SetBoundary({end_state[i], end_state[i]}, num_of_knots_ - 1);
  }

  // set end s target
  layers_[0].SetTarget(end_state[0], num_of_knots_ - 1);
  layers_[0].SetWeight(1.0, num_of_knots_ - 1);
}

}  // namespace piecewise
}  // namespace math
