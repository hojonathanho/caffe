#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
Dtype angle_error(Dtype x) {
  Dtype m = x - 2*M_PI*floor(x / (2*M_PI));
  return m <= M_PI ? m : 2*M_PI - m;
}

template <typename Dtype>
void AngleAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  for (int i = 0; i < 2; ++i) {
    CHECK_EQ(bottom[i]->channels(), 1);
    CHECK_EQ(bottom[i]->height(), 1);
    CHECK_EQ(bottom[i]->width(), 1);
  }
  (*top)[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void AngleAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();

  Dtype sum_angle_error = 0;
  for (int i = 0; i < num; ++i) {
    sum_angle_error += angle_error(bottom_data[i] - bottom_label[i]);
  }
  (*top)[0]->mutable_cpu_data()[0] = sum_angle_error / num;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(AngleAccuracyLayer);

}  // namespace caffe
