/**
 * only modify base on apollo autocar
 */

#pragma once

#include "glog/logging.h"

#include <tuple>
#include <utility>
#include <vector>
#include <string>
#include <cmath>

namespace math {
namespace piecewise {

enum TargetSource {
  DEFAULT,
  LEFT_BOUNDARY,
  RIGHT_BOUNDARY,
  BOTH_BOUNDARY,
  DECISION,
  HISTORY,
  KEEP_STRAIGHT,
  MANUAL,
};

enum TargetConfidence {
  IGNORE = 0,
  CONFLICT = 1,
  FLEXIBLE = 3,
  RELIABLE = 10,
  FOCUSED = 50,
  FIXED = 100,
};

struct LinearLayer {
 public:
  explicit LinearLayer(const uint32_t length = 2)
      : length_(length),
        target_(length, 0),
        boundary_(length, {-__DBL_MAX__, __DBL_MAX__}),
        weight_(length, 1e-6),
        source_(length, DEFAULT),
        confidence_(length, IGNORE) {
    CHECK_GE(length, 2);
  }

  LinearLayer(const LinearLayer& layer)
      : length_(layer.Length()),
        target_(layer.Target()),
        boundary_(layer.Boundary()),
        weight_(layer.Weight()),
        source_(layer.Source()),
        confidence_(layer.Confidence()) {}

  void SetId(const std::string id) { id_ = id; }

  std::string Id() const { return id_; }

  void SetLength(const uint32_t length);

  uint32_t Length() const { return length_; }

  void SetTarget(const std::vector<double>& target);

  void SetTarget(const double target, const uint32_t pos);

  const std::vector<double>& Target() const { return target_; }

  void SetBoundary(const std::vector<std::pair<double, double>>& boundary);

  void SetBoundary(const double boundary, const uint32_t pos, const bool upper);
  void SetBoundary(const std::pair<double, double> boundary,
                   const uint32_t pos);

  const std::vector<std::pair<double, double>>& Boundary() const {
    return boundary_;
  }

  void SetWeight(const std::vector<double>& weight);

  void SetWeight(const double weight, const uint32_t pos);

  const std::vector<TargetSource>& Source() const { return source_; }

  void SetSource(const std::vector<TargetSource>& source);

  void SetSource(const TargetSource source, const uint32_t pos);

  const std::vector<TargetConfidence>& Confidence() const {
    return confidence_;
  }

  void SetConfidence(const std::vector<TargetConfidence>& confidence);

  void SetConfidence(const TargetConfidence confidence, const uint32_t pos);

  const std::vector<double>& Weight() const { return weight_; }

  std::string DebugString();

 private:
  uint32_t length_;
  // uint32_t rank_;
  std::vector<double> target_;
  std::vector<std::pair<double, double>> boundary_;
  std::vector<double> weight_;
  std::string id_;
  std::vector<TargetSource> source_;
  std::vector<TargetConfidence> confidence_;
};

class DifferentialLayer {
 public:
  DifferentialLayer(const uint32_t rank, const std::vector<double>& label,
                    const double block_s)
      : rank_(rank),
        label_(label),
        layers_(rank + 1, LinearLayer(label.size())),
        block_s_(block_s) {}

  DifferentialLayer(const uint32_t rank, const uint32_t samples,
                    const double sample_dist);

  DifferentialLayer(const DifferentialLayer& layer);

  uint32_t Rank() const { return rank_; }

  void SetId(const std::string id);

  const std::string& Id() const { return id_; }

  LinearLayer& operator[](const uint32_t rank);

  const LinearLayer& operator[](const uint32_t rank) const;

  LinearLayer& at(const uint32_t rank);

  const LinearLayer& at(const uint32_t rank) const;

  const std::vector<LinearLayer>& Layers() const { return layers_; }

  void SetLabel(const std::vector<double>& label);

  const std::vector<double>& Label() const { return label_; }

  std::string DebugString();

  double BlockS() const { return block_s_; }

 private:
  uint32_t rank_;
  std::vector<double> label_;
  std::vector<LinearLayer> layers_;
  std::string id_;
  double block_s_ = -1000.0;
};
}  // namespace piecewise
}  // namespace math
