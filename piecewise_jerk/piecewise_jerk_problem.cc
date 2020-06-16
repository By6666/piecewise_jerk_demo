/**
 * only modify base on apollo autocar
 */

#include "piecewise_jerk_problem.h"

namespace math {
namespace piecewise {

namespace {
constexpr double kMaxVariableRange = 1.0e10;
}  // namespace

PiecewiseJerkProblem::PiecewiseJerkProblem(const size_t num_of_knots,
                                           const double delta_s)
    : layers_(3, num_of_knots, delta_s) {
  CHECK_GE(num_of_knots, 2);
  num_of_knots_ = num_of_knots;
  delta_s_ = delta_s;
}

PiecewiseJerkProblem::PiecewiseJerkProblem(const DifferentialLayer& layer)
    : layers_(layer) {
  num_of_knots_ = layer.Label().size();
  CHECK_GE(num_of_knots_, 2);
  delta_s_ = layer.Label().at(1) - layer.Label().at(0);
}

OSQPData* PiecewiseJerkProblem::FormulateProblem() {
  // calculate kernel
  std::vector<c_float> P_data;
  std::vector<c_int> P_indices;
  std::vector<c_int> P_indptr;
  CalculateKernel(&P_data, &P_indices, &P_indptr);

  // calculate affine constraints
  std::vector<c_float> A_data;
  std::vector<c_int> A_indices;
  std::vector<c_int> A_indptr;
  std::vector<c_float> lower_bounds;
  std::vector<c_float> upper_bounds;
  CalculateAffineConstraint(&A_data, &A_indices, &A_indptr, &lower_bounds,
                            &upper_bounds);

  // calculate offset
  std::vector<c_float> q;
  CalculateOffset(&q);

  OSQPData* data = reinterpret_cast<OSQPData*>(c_malloc(sizeof(OSQPData)));
  CHECK_EQ(lower_bounds.size(), upper_bounds.size());

  size_t kernel_dim = 3 * num_of_knots_;
  size_t num_affine_constraint = lower_bounds.size();

  data->n = kernel_dim;
  data->m = num_affine_constraint;
  data->P = csc_matrix(kernel_dim, kernel_dim, P_data.size(), CopyData(P_data),
                       CopyData(P_indices), CopyData(P_indptr));
  data->q = CopyData(q);
  data->A =
      csc_matrix(num_affine_constraint, kernel_dim, A_data.size(),
                 CopyData(A_data), CopyData(A_indices), CopyData(A_indptr));
  data->l = CopyData(lower_bounds);
  data->u = CopyData(upper_bounds);
  return data;
}

Status PiecewiseJerkProblem::Optimize(const int max_iter) {
  OSQPData* data = FormulateProblem();

  // std::cout << layers_[0].DebugString() << std::endl;
  // std::cout << layers_[1].DebugString() << std::endl;
  // std::cout << layers_[2].DebugString() << std::endl;

  OSQPSettings* settings = SolverDefaultSettings();
  settings->max_iter = max_iter;
  // settings->time_limit = 0.02;

  auto init_start_time = clock();
  OSQPWorkspace* osqp_work = osqp_setup(data, settings);
  if (osqp_work == nullptr) {
    std::string msg = "Fail to setup osqp work!";
    std::cout << msg;
    FreeData(data);
    c_free(settings);
    return Status({false, msg});
  }
  auto solve_start_time = clock();
  osqp_solve(osqp_work);

  auto solve_finish_time = clock();

  std::cout << "setup time = "
            << (solve_start_time - init_start_time) / CLOCKS_PER_SEC * 1000
            << " ms, solve time = "
            << (solve_finish_time - solve_start_time) / CLOCKS_PER_SEC * 1000
            << " ms" << std::endl;
  auto status = osqp_work->info->status_val;

  set_solve_info(*(osqp_work->info));
  if (status != 1 && status != 2 && status != -2) {
    std::string msg =
        "failed optimization status: " + std::string(osqp_work->info->status);
    std::cout << msg;
    osqp_cleanup(osqp_work);
    FreeData(data);
    c_free(settings);
    return Status({false, msg});
  } else if (osqp_work->solution == nullptr) {
    std::string msg = "The solution from OSQP is nullptr";
    std::cout << msg;
    osqp_cleanup(osqp_work);
    FreeData(data);
    c_free(settings);
    return Status({false, msg});
  }

  // extract primal results
  x_.resize(num_of_knots_);
  dx_.resize(num_of_knots_);
  ddx_.resize(num_of_knots_);
  for (size_t i = 0; i < num_of_knots_; ++i) {
    x_.at(i) = osqp_work->solution->x[i] / scale_factor_[0];
    dx_.at(i) = osqp_work->solution->x[i + num_of_knots_] / scale_factor_[1];
    ddx_.at(i) =
        osqp_work->solution->x[i + 2 * num_of_knots_] / scale_factor_[2];
  }

  // Cleanup
  osqp_cleanup(osqp_work);
  FreeData(data);
  c_free(settings);
  if (status == -2) {
    std::string msg = "optimize over max iteration!";
    return Status({false, msg});
  } else {
    std::string msg = "OK ";
    return Status({true, msg});
  }
}  // namespace piecewise

void PiecewiseJerkProblem::CalculateAffineConstraint(
    std::vector<c_float>* A_data, std::vector<c_int>* A_indices,
    std::vector<c_int>* A_indptr, std::vector<c_float>* lower_bounds,
    std::vector<c_float>* upper_bounds) {
  // 3N params bounds on x, x', x''
  // 3(N-1) constraints on x, x', x''
  // 3 constraints on x_init_
  const int n = static_cast<int>(num_of_knots_);
  const int num_of_variables = 3 * n;
  const int num_of_constraints = num_of_variables + 2 * (n - 1);
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

void PiecewiseJerkProblem::CalculateKernel(std::vector<c_float>* P_data,
                                           std::vector<c_int>* P_indices,
                                           std::vector<c_int>* P_indptr) {
  const uint32_t dim = layers_.Rank() * num_of_knots_;
  P_data->resize(dim);
  P_indptr->resize(dim + 1);
  P_indices->resize(dim);
  for (uint32_t i = 0; i < 3 * num_of_knots_; i++) {
    uint32_t pos = i % num_of_knots_;
    uint32_t rank = i / num_of_knots_;
    P_indptr->at(i) = static_cast<c_int>(i);
    P_indices->at(i) = static_cast<c_int>(i);
    P_data->at(i) = layers_[rank].Weight().at(pos);
  }
  *(P_indptr->end() - 1) = dim;
}

void PiecewiseJerkProblem::CalculateOffset(std::vector<c_float>* q) {
  const uint32_t dim = layers_.Rank() * num_of_knots_;
  q->resize(dim);
  for (uint32_t i = 0; i < 3 * num_of_knots_; i++) {
    uint32_t pos = i % num_of_knots_;
    uint32_t rank = i / num_of_knots_;
    q->at(i) =
        -2.0 * layers_[rank].Weight().at(pos) * layers_[rank].Target().at(pos);
  }
}

OSQPSettings* PiecewiseJerkProblem::SolverDefaultSettings() {
  // Define Solver default settings
  OSQPSettings* settings =
      reinterpret_cast<OSQPSettings*>(c_malloc(sizeof(OSQPSettings)));
  osqp_set_default_settings(settings);
  settings->polish = true;
  settings->verbose = true;
  settings->scaled_termination = true;
  settings->scaling = false;
  settings->adaptive_rho = true;
  settings->adaptive_rho_interval = 25;
  settings->eps_abs = 2e-3;
  settings->eps_rel = 2e-3;
  return settings;
}

void PiecewiseJerkProblem::set_x_bounds(
    std::vector<std::pair<double, double>> x_bounds) {
  CHECK_EQ(x_bounds.size(), num_of_knots_);
  layers_[0].SetBoundary(x_bounds);
}

void PiecewiseJerkProblem::set_dx_bounds(
    std::vector<std::pair<double, double>> dx_bounds) {
  CHECK_EQ(dx_bounds.size(), num_of_knots_);
  layers_[1].SetBoundary(dx_bounds);
}

void PiecewiseJerkProblem::set_ddx_bounds(
    std::vector<std::pair<double, double>> ddx_bounds) {
  CHECK_EQ(ddx_bounds.size(), num_of_knots_);
  layers_[2].SetBoundary(ddx_bounds);
}

void PiecewiseJerkProblem::set_x_bounds(const double x_lower_bound,
                                        const double x_upper_bound) {
  layers_[0].SetBoundary(std::vector<std::pair<double, double>>(
      num_of_knots_, std::make_pair(x_lower_bound, x_upper_bound)));
}

void PiecewiseJerkProblem::set_dx_bounds(const double dx_lower_bound,
                                         const double dx_upper_bound) {
  layers_[1].SetBoundary(std::vector<std::pair<double, double>>(
      num_of_knots_, std::make_pair(dx_lower_bound, dx_upper_bound)));
}

void PiecewiseJerkProblem::set_ddx_bounds(const double ddx_lower_bound,
                                          const double ddx_upper_bound) {
  layers_[2].SetBoundary(std::vector<std::pair<double, double>>(
      num_of_knots_, std::make_pair(ddx_lower_bound, ddx_upper_bound)));
}

void PiecewiseJerkProblem::set_dddx_bound(const double dddx_bound) {
  layers_[3].SetBoundary(std::vector<std::pair<double, double>>(
      num_of_knots_, std::make_pair(-dddx_bound, dddx_bound)));
}

void PiecewiseJerkProblem::set_dddx_bound(const double dddx_lower_bound,
                                          const double dddx_upper_bound) {
  layers_[3].SetBoundary(std::vector<std::pair<double, double>>(
      num_of_knots_, std::make_pair(dddx_lower_bound, dddx_upper_bound)));
}

void PiecewiseJerkProblem::set_weight_x(const double weight_x) {
  layers_[0].SetWeight(std::vector<double>(num_of_knots_, weight_x));
}

void PiecewiseJerkProblem::set_weight_dx(const double weight_dx) {
  layers_[1].SetWeight(std::vector<double>(num_of_knots_, weight_dx));
}

void PiecewiseJerkProblem::set_weight_ddx(const double weight_ddx) {
  layers_[2].SetWeight(std::vector<double>(num_of_knots_, weight_ddx));
}

void PiecewiseJerkProblem::set_weight_dddx(const double weight_dddx) {
  layers_[3].SetWeight(std::vector<double>(num_of_knots_, weight_dddx));
}

void PiecewiseJerkProblem::FreeData(OSQPData* data) {
  delete[] data->q;
  delete[] data->l;
  delete[] data->u;

  delete[] data->P->i;
  delete[] data->P->p;
  delete[] data->P->x;

  delete[] data->A->i;
  delete[] data->A->p;
  delete[] data->A->x;
}

}  // namespace piecewise
}  // namespace math
