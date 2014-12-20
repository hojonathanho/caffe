#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CosineLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  for (int i = 0; i < 2; ++i) {
    CHECK_EQ(bottom[i]->height(), 1);
    CHECK_EQ(bottom[i]->width(), 1);
  }
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  tmp_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  ones_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  caffe_set(bottom[0]->count(), Dtype(1), ones_.mutable_cpu_data());
}

template <typename Dtype>
void CosineLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  caffe_cos(count, diff_.cpu_data(), tmp_.mutable_cpu_data());
  Dtype sum = caffe_cpu_dot(count, tmp_.cpu_data(), ones_.cpu_data());
  Dtype loss = (bottom[0]->num()*bottom[0]->channels() - sum) / bottom[0]->num() / Dtype(2);
  (*top)[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CosineLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / (*bottom)[i]->num() / Dtype(2);
      caffe_sin((*bottom)[i]->count(), diff_.cpu_data(), tmp_.mutable_cpu_data());
      caffe_cpu_axpby(
          (*bottom)[i]->count(),              // count
          alpha,                              // alpha
          tmp_.cpu_data(),                    // a
          Dtype(0),                           // beta
          (*bottom)[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CosineLossLayer);
#endif

INSTANTIATE_CLASS(CosineLossLayer);

}  // namespace caffe
