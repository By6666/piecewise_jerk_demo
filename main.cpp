#include <iostream>

#include "piecewise_jerk/piecewise_jerk_speed_problem.h"
#include "matplot/matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main() {
  double init_v = 8.0, init_a = 0.0, stop_distance = 15.0;
  const double delta_t = 0.02;
  const double total_time = 3.0;
  size_t num_of_knots = static_cast<size_t>(total_time / delta_t) + 1;

  init_a = std::min(init_a, 0.0);

  std::array<double, 3> init_s = {0.0, init_v, init_a};
  std::array<double, 3> end_s = {stop_distance, 0.0, 0.0};

  math::piecewise::PiecewiseJerkSpeedProblem project(num_of_knots, delta_t);
  project.SetConsition(init_s, end_s);

  auto success = project.Optimize();
  std::cout << success.second << std::endl;

  if (!success.first) return 0;

  // Extract output
  const std::vector<double>& s = project.opt_x();
  const std::vector<double>& ds = project.opt_dx();
  const std::vector<double>& dds = project.opt_ddx();
  std::vector<double> ddds(num_of_knots);

  std::vector<double> t(num_of_knots);
  for (int i = 0; i < num_of_knots; ++i) {
    t[i] = delta_t * i;
    if (i < num_of_knots - 1)
      ddds[i] = (dds[i + 1] - dds[i]) / delta_t;
    else
      ddds[i] = ddds[i - 1];
  }

  // for (size_t i = 0; i < num_of_knots; ++i) {
  //   std::cout << "For[" << delta_t * static_cast<double>(i) << "], s = " <<
  //   s[i]
  //             << ", v = " << ds[i] << ", a = " << dds[i] << std::endl;
  // }
  plt::title("piecewise jerk");
  plt::grid(true);
  plt::named_plot("s", t, s);
  plt::named_plot("v", t, ds);
  plt::named_plot("a", t, dds);
  plt::named_plot("j", t, ddds);
  plt::legend();
  plt::show();

  return 0;
}
