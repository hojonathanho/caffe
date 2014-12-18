#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class CosineLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CosineLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 3, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 3, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    filler.Fill(this->blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~CosineLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }

  void TestForward() {
    const Dtype kErrorMargin = 1e-5;

    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    CosineLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
    const Dtype loss_weight_1 =
        layer_weight_1.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);

    // Manually calculate the loss here to make sure the layer is correct
    Dtype true_loss = 0;
    for (int i = 0; i < blob_bottom_data_->num(); ++i) {
      for (int j = 0; j < blob_bottom_data_->channels(); ++j) {
        Dtype a = blob_bottom_data_->cpu_data()[i*blob_bottom_data_->channels() + j];
        Dtype b = blob_bottom_label_->cpu_data()[i*blob_bottom_label_->channels() + j];
        true_loss += (Dtype(1) - cos(a - b)) / Dtype(2);
      }
    }
    true_loss /= Dtype(blob_bottom_data_->num());
    EXPECT_NEAR(true_loss, loss_weight_1, kErrorMargin);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    CosineLossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
    const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);
    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CosineLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(CosineLossLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(CosineLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  CosineLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

}  // namespace caffe
