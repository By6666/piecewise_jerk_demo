/**
 * only modify base on apollo autocar
 */

#include "linear_layer.h"

namespace math {
namespace piecewise {

void LinearLayer::SetLength(const uint32_t length) {
  if (length < length_) {
    target_.erase(target_.begin() + length, target_.end());
    boundary_.erase(boundary_.begin() + length, boundary_.end());
    weight_.erase(weight_.begin() + length, weight_.end());
  } else {
    target_.resize(length, 0);
    boundary_.resize(length, {-INFINITY, INFINITY});
    weight_.resize(length, 1e-6);
  }
  length_ = length;
}

void LinearLayer::SetTarget(const std::vector<double>& target) {
  CHECK_EQ(target.size(), length_);
  target_ = target;
}

void LinearLayer::SetTarget(const double target, const uint32_t pos) {
  CHECK_LT(pos, length_);
  target_[pos] = target;
}

void LinearLayer::SetBoundary(
    const std::vector<std::pair<double, double>>& boundary) {
  CHECK_EQ(boundary.size(), length_);
  boundary_ = boundary;
}

void LinearLayer::SetBoundary(const double boundary, const uint32_t pos,
                              const bool upper) {
  CHECK_LT(pos, length_);
  if (upper) {
    boundary_[pos].second = boundary;
  } else {
    boundary_[pos].first = boundary;
  }
}
void LinearLayer::SetBoundary(const std::pair<double, double> boundary,
                              const uint32_t pos) {
  CHECK_LT(pos, length_);
  boundary_[pos] = boundary;
}

void LinearLayer::SetWeight(const std::vector<double>& weight) {
  CHECK_EQ(weight.size(), length_);
  weight_ = weight;
}

void LinearLayer::SetWeight(const double weight, const uint32_t pos) {
  CHECK_LT(pos, length_);
  weight_[pos] = weight;
}

void LinearLayer::SetSource(const std::vector<TargetSource>& source) {
  CHECK_EQ(source.size(), length_);
  source_ = source;
}

void LinearLayer::SetSource(const TargetSource source, const uint32_t pos) {
  CHECK_LT(pos, length_);
  source_[pos] = source;
}

void LinearLayer::SetConfidence(
    const std::vector<TargetConfidence>& confidence) {
  CHECK_EQ(confidence.size(), length_);
  confidence_ = confidence;
}

void LinearLayer::SetConfidence(const TargetConfidence confidence,
                                const uint32_t pos) {
  CHECK_LT(pos, length_);
  confidence_[pos] = confidence;
}

std::string LinearLayer::DebugString() {
  std::string debug_string = "\t" + id_ + "\n";
  for (uint32_t i = 0; i < length_; i++) {
    debug_string +=
        (std::to_string(i) + ":" + "\t" +
         "LowerBound: " + std::to_string(boundary_[i].first) + "\t" +
         "UpperBound: " + std::to_string(boundary_[i].second) + "\t" +
         "Target: " + std::to_string(target_[i]) + "\t" +
         "Weight: " + std::to_string(weight_[i]) + "\t" +
         "Source: " + std::to_string(static_cast<int>(source_[i])) +
         //                   "\t" + "Confidence: " +
         //                   std::to_string(static_cast<int>(confidence_[i])) +
         "\n");
  }
  return debug_string;
}

DifferentialLayer::DifferentialLayer(const uint32_t rank,
                                     const uint32_t samples,
                                     const double sample_dist)
    : rank_(rank) {
  CHECK_GE(samples, 2);
  layers_.reserve(rank + 1);
  label_.resize(samples);
  for (uint32_t i = 0; i < samples; i++) {
    label_[i] = sample_dist * i;
  }
  for (uint32_t i = 0; i <= rank; i++) {
    layers_.emplace_back(LinearLayer(samples));
  }
}

DifferentialLayer::DifferentialLayer(const DifferentialLayer& layer)
    : rank_(layer.Rank()),
      label_(layer.Label()),
      layers_(layer.Layers()),
      id_(layer.Id()),
      block_s_(layer.BlockS()) {}

LinearLayer& DifferentialLayer::operator[](const uint32_t rank) {
  CHECK_LT(rank, rank_ + 1);
  return layers_[rank];
}

const LinearLayer& DifferentialLayer::operator[](const uint32_t rank) const {
  CHECK_LT(rank, rank_ + 1);
  return layers_[rank];
}

LinearLayer& DifferentialLayer::at(const uint32_t rank) {
  CHECK_LT(rank, rank_ + 1);
  return layers_[rank];
}

const LinearLayer& DifferentialLayer::at(const uint32_t rank) const {
  CHECK_LT(rank, rank_ + 1);
  return layers_[rank];
}

void DifferentialLayer::SetLabel(const std::vector<double>& label) {
  CHECK_GE(label.size(), 2);
  for (uint32_t i = 0; i <= rank_; i++) {
    layers_[i].SetLength(label.size());
  }
}

void DifferentialLayer::SetId(const std::string id) {
  id_ = id;
  for (uint32_t i = 0; i <= rank_; i++) {
    layers_[i].SetId(id + "." + std::to_string(i) + "-th_derivative_layer");
  }
}

std::string DifferentialLayer::DebugString() {
  std::string debug_string = id_ + ":\n";
  for (uint32_t i = 0; i <= rank_; i++) {
    debug_string += ("At Rank " + std::to_string(i) + ":\n");
    debug_string += layers_[i].DebugString();
  }
  return debug_string;
}

}  // namespace piecewise
}  // namespace math
