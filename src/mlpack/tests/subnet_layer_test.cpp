/**
 * @file conv_layer_test.cpp
 * @author Nilay Jain
 *
 * Tests the concat layer.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/layer/conv_layer.hpp>
#include <mlpack/methods/ann/layer/one_hot_layer.hpp>
#include <mlpack/methods/ann/layer/conv_layer.hpp>
#include <mlpack/methods/ann/layer/pooling_layer.hpp>
#include <mlpack/methods/ann/layer/softmax_layer.hpp>
#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
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


BOOST_AUTO_TEST_SUITE(SubnetLayerTest);
void Test(arma::mat m1, arma::mat m2)
{
  for (size_t i = 0; i < m1.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(m1(i), m2(i), 1e-2);  
}

void Test(arma::cube m1, arma::cube m2)
{
  BOOST_REQUIRE_EQUAL(m1.n_slices, m2.n_slices);
  for (size_t i = 0; i < m1.n_slices; ++i)
    Test(m1.slice(i), m2.slice(i));  
}

void SubnetLayerTest()
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
  ConvLayer<> convLayer0(3, 2, 1, 1);
  ConvLayer<> convLayer1(2, 3, 1, 1);
  auto JoinLayers = std::tie(convLayer0, convLayer1);
  size_t numLayers = 2;
  SubnetLayer<decltype(JoinLayers), arma::cube, arma::cube>
      subnetLayer0(numLayers, std::move(JoinLayers));

  LinearMappingLayer<> linearLayer0(75, 10);
  SoftmaxLayer<> softmaxLayer0;
  OneHotLayer outputLayer;


  BiasLayer2D<> biasLayer0(8);
  BaseLayer2D<> baseLayer0;


  auto modules = std::tie(subnetLayer0, baseLayer0, linearLayer0, softmaxLayer0);

  CNN<decltype(modules), decltype(outputLayer),
      RandomInitialization, MeanSquaredErrorFunction> net(modules, outputLayer);

  RMSprop<decltype(net)> opt(net, 0.01, 0.88, 1e-8, 10 * input.n_slices, 0);

  arma::mat Y = arma::zeros<arma::mat>(3, 1);

  net.Train(input, Y, opt); 
}
//! tests the concat_layer.
BOOST_AUTO_TEST_CASE(SubnetLayer)
{
  SubnetLayerTest();
}
BOOST_AUTO_TEST_SUITE_END();
