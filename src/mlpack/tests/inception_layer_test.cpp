/**
 * @file convolutional_network_test.cpp
 * @author Marcus Edel
 *
 * Tests the convolutional neural network.
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

void BuildSampleNetwork()
{
  arma::cube input(28, 28, 192, arma::fill::randu);
  std::cout << "inception, input : " << arma::size(input) << std::endl;
  InceptionLayer<> in(192, 64, 96, 128, 16,  32,  32);
  LinearMappingLayer<> linearLayer0(256, 3);
  BiasLayer<> biasLayer0(3);
  SoftmaxLayer<> softmaxLayer0;

  OneHotLayer outputLayer;
  arma::mat Y = arma::zeros<arma::mat>(3, 192);
  auto modules = std::tie(in, linearLayer0, biasLayer0, softmaxLayer0);

  CNN<decltype(modules), decltype(outputLayer),
      RandomInitialization, MeanSquaredErrorFunction> net(modules, outputLayer);


  RMSprop<decltype(net)> opt(net, 0.01, 0.88, 1e-8, 3 * input.n_slices, 0);

  net.Train(input, Y, opt);
  std::cout << "reached here" << std::endl;
}
BOOST_AUTO_TEST_CASE(SampleInceptionLayerTest)
{
  BuildSampleNetwork();
}
BOOST_AUTO_TEST_SUITE_END();
