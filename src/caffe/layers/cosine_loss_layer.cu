#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Cos(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = cos(in[index]);
  }
}

template <typename Dtype>
__global__ void Sin(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = sin(in[index]);
  }
}

template <typename Dtype>
void CosineLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Cos<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, diff_.gpu_data(), tmp_.mutable_gpu_data());
  Dtype sum; caffe_gpu_dot(count, tmp_.gpu_data(), ones_.gpu_data(), &sum);
  Dtype loss = (bottom[0]->num()*bottom[0]->channels() - sum) / bottom[0]->num() / Dtype(2);
  (*top)[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CosineLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / (*bottom)[i]->num() / Dtype(2);
      Sin<Dtype><<<CAFFE_GET_BLOCKS((*bottom)[i]->count()), CAFFE_CUDA_NUM_THREADS>>>(
            (*bottom)[i]->count(), diff_.gpu_data(), tmp_.mutable_gpu_data());
      caffe_gpu_axpby(
          (*bottom)[i]->count(),              // count
          alpha,                              // alpha
          tmp_.gpu_data(),                    // a
          Dtype(0),                           // beta
          (*bottom)[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_CLASS(CosineLossLayer);

}  // namespace caffe
