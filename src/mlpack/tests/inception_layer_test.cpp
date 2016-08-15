/**
 * @file inception_layer_test.cpp
 * @author Nilay Jain
 *
 * Tests the inception layer.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>

#include <mlpack/methods/ann/layer/one_hot_layer.hpp>
#include <mlpack/methods/ann/layer/conv_layer.hpp>
#include <mlpack/methods/ann/layer/pooling_layer.hpp>
#include <mlpack/methods/ann/layer/softmax_layer.hpp>
#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/layer/inception_layer.hpp>
#include <mlpack/methods/ann/layer/concat_layer.hpp> 
#include <mlpack/methods/ann/layer/subnet_layer.hpp>

#include <mlpack/methods/ann/performance_functions/mse_function.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>

#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/cnn.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;


BOOST_AUTO_TEST_SUITE(InceptionLayerTest);

void Test(arma::mat m1, arma::mat m2)
{
  for (size_t i = 0; i < m1.n_elem; ++i)
    BOOST_REQUIRE_CLOSE(m1(i), m2(i), 1e-2);  
}

void Test(arma::cube m1, arma::cube m2)
{
  BOOST_REQUIRE_EQUAL(m1.n_slices, m2.n_slices);
  for (size_t i = 0; i < m1.n_slices; ++i)
    Test(m1.slice(i), m2.slice(i));  
}

void SampleInceptionNetwork()
{
  arma::cube input(6, 6, 60, arma::fill::randu);
  std::cout << "inception, input : " << arma::size(input) << std::endl;
  ConvLayer<> convLayer0(60, 192, 1, 1);
  BiasLayer2D<> biasLayer0(8);
  BaseLayer2D<> baseLayer0;
  InceptionLayer<> in(192, 64, 96, 128, 16, 32, 32);
  ConvLayer<> convLayer1(256, 10, 1, 1);
  BiasLayer2D<> biasLayer1(8);
  BaseLayer2D<> baseLayer1;
  LinearMappingLayer<> linearLayer0(360, 3);
  BiasLayer<> biasLayer2(3);
  SoftmaxLayer<> softmaxLayer0;

  OneHotLayer outputLayer;
  arma::mat Y = arma::zeros<arma::mat>(3, 1);
  auto modules = std::tie(convLayer0, biasLayer0, baseLayer0, in, 
                  convLayer1, biasLayer1, baseLayer1,
                  linearLayer0, biasLayer2, softmaxLayer0);

  CNN<decltype(modules), decltype(outputLayer),
      RandomInitialization, MeanSquaredErrorFunction> net(modules, outputLayer);


  RMSprop<decltype(net)> opt(net, 0.01, 0.88, 1e-8, 3 * input.n_slices, 0);

  net.Train(input, Y, opt);
  std::cout << "reached here, BuildSampleNetwork" << std::endl;
}

void ForwardPassTest(arma::cube& input, arma::cube& output, InceptionLayer<>& in)
{
  arma::cube inceptionLayerOutput;
  in.Forward(input, inceptionLayerOutput);
  Test(inceptionLayerOutput, output);
}

void BackwardPassTest(arma::cube& error, InceptionLayer<>& in)
{
  arma::cube d1, d2; //dummy cubes.
  in.Backward(d1, error, d2);
  arma::cube output = arma::ones(5, 5, 1);
  Test(in.base1.Delta(), output);
  Test(in.base3.Delta(), output);
  Test(in.base5.Delta(), output); 
  Test(in.bias1.Delta(), output); // since bias is zero.
  Test(in.bias3.Delta(), output);
  output = arma::ones(5, 5, 4);
  Test(in.conv3.Delta(), output);
}

void GradientUpdateTest(arma::cube& delta, InceptionLayer<>& in)
{
  arma::cube d1, d2;
  in.Gradient(d1, delta, d2);
  Test(in.bias1.Gradient(), delta.slice(0));
  Test(in.bias3.Gradient(), delta.slice(1));
  Test(in.bias5.Gradient(), delta.slice(2));
  Test(in.biasPool.Gradient(), delta.slice(3));
}
void SmallNetworkTest()
{
  arma::vec a = arma::linspace<arma::vec>(1, 25, 25);
  arma::cube input(5, 5, 3);
  for (size_t i = 0; i < input.n_slices; ++i)
  {
    int vec_idx = 0, row_idx = 0;
    for (size_t j = 0; j < input.n_rows; ++j)
    {
      input.slice(i).row(j) = a.subvec(vec_idx, vec_idx + 4).t();
      vec_idx += 5;
    }
  }

  //! set weights for the forward pass.
  arma::cube output(5, 5, 4);
  InceptionLayer<> in(3, 1, 4, 1, 6, 1, 1, 1);
  arma::cube conv1_w(1, 1, 3, arma::fill::ones);
  in.conv1.Weights() = conv1_w;
  
  arma::cube proj3_w(1, 1, 3 * 4);
  proj3_w.fill(1.0);
  in.proj3.Weights() = proj3_w;

  arma::mat id3 = arma::zeros<arma::mat>(3, 3);
  id3(1, 1) = 1;
  arma::cube conv3_w(3, 3, 4 * 1);
  for (size_t i = 0; i < conv3_w.n_slices; ++i) 
    conv3_w.slice(i) = id3;
  in.conv3.Weights() = conv3_w;

  arma::cube proj5_w(1, 1, 3 * 6);
  proj5_w.fill(1.0);
  in.proj5.Weights() = proj5_w;

  arma::mat id5 = arma::zeros<arma::mat>(5, 5);
  id5(2, 2) = 1;
  arma::cube conv5_w(5, 5, 6 * 1);
  for (size_t i = 0; i < conv5_w.n_slices; ++i) 
    conv5_w.slice(i) = id5;
  in.conv5.Weights() = conv5_w;

  // test for forward pass.
  arma::cube convPool_w(1, 1, 3 * 1);
  convPool_w.fill(1.0);
  in.convPool.Weights() = convPool_w;
  output.slice(0) = input.slice(0) * 3;
  output.slice(1) = input.slice(0) * 3 * 4;
  output.slice(2) = input.slice(0) * 3 * 6;
  output.slice(3) = input.slice(0) * 3;
  output.slice(3).row(0).fill(0);
  output.slice(3).row(4).fill(0);
  output.slice(3).col(0).fill(0);
  output.slice(3).col(4).fill(0);

  ForwardPassTest(input, output, in);

  // testing for backward pass.
  arma::cube error = arma::ones(5, 5, 4);
  BackwardPassTest(error, in);

  // test for gradient update.
  arma::cube delta = arma::zeros(5,5,4);
  for (size_t i = 0; i < delta.n_slices; ++i)
    delta(2, 2, i) = 1;
  GradientUpdateTest(delta, in);
}



BOOST_AUTO_TEST_CASE(SampleInceptionLayerTest)
{
  //SampleInceptionNetwork();
  SmallNetworkTest();
}
BOOST_AUTO_TEST_SUITE_END();
