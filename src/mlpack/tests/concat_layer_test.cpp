#include <mlpack/core.hpp>

#include <mlpack/methods/ann/layer/conv_layer.hpp>
#include <mlpack/methods/ann/layer/one_hot_layer.hpp>
#include <mlpack/methods/ann/layer/conv_layer.hpp>
#include <mlpack/methods/ann/layer/pooling_layer.hpp>
#include <mlpack/methods/ann/layer/softmax_layer.hpp>
#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/layer/concat_layer.hpp>

#include <mlpack/methods/ann/performance_functions/mse_function.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>

#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/cnn.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(ConcatLayerTest);
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

void ConcatLayerTest()
{
  arma::cube input(5, 5, 3, arma::fill::ones);
  ConvLayer<> convLayer0(3, 2, 1, 1);
  ConvLayer<> convLayer1(3, 3, 1, 1);
  auto JoinLayers = std::tie(convLayer0, convLayer1);
  size_t numLayers = 2;
  ConcatLayer<decltype(JoinLayers), arma::cube, arma::cube> 
      concatLayer0(numLayers, std::move(JoinLayers));

  arma::mat id = arma::ones(1, 1);

  arma::cube convLayer0_w(1, 1, 3 * 2);
  for (size_t i = 0; i < convLayer0_w.n_slices; ++i)
    convLayer0_w.slice(i) = id;
  convLayer0.Weights() = convLayer0_w;

  arma::cube convLayer1_w(1, 1, 3 * 3);
  id(0, 0) = 2;
  for (size_t i = 0; i < convLayer1_w.n_slices; ++i)
    convLayer1_w.slice(i) = id;
  convLayer1.Weights() = convLayer1_w;

  arma::cube d1, d2; // dummy cubes.
  arma::cube output;

  //! Forward pass test for ConcatLayer...
  convLayer0.InputParameter() = convLayer1.InputParameter() = input;
  convLayer0.Forward(convLayer0.InputParameter(), convLayer0.OutputParameter());
  convLayer1.Forward(convLayer1.InputParameter(), convLayer1.OutputParameter());
  concatLayer0.Forward(d1, output);
  arma::cube desiredOutput = arma::join_slices(convLayer0.OutputParameter(), 
                              convLayer1.OutputParameter());
  Test(output, desiredOutput);
  
  //! Backward pass test for ConcatLayer.
  arma::cube error = arma::ones(5, 5, 5);
  concatLayer0.Backward(d1, error, d2);
  arma::cube backout0 = arma::ones(5, 5, 3) * 2;
  Test(convLayer0.Delta(), backout0);
  arma::cube backout1 = arma::ones(5, 5, 3) * 6;
  Test(convLayer1.Delta(), backout1);

  //! todo: Gradient update test for ConcatLayer.
}
//! tests the concat_layer.
BOOST_AUTO_TEST_CASE(ConcatLayer)
{
  ConcatLayerTest();
}
BOOST_AUTO_TEST_SUITE_END();
