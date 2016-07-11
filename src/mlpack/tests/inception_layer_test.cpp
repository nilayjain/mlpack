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
  arma::mat X;
  /*X.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  arma::uword nPoints = X.n_cols;
  for (arma::uword i = 0; i < nPoints; i++)
  {
    X.col(i) /= norm(X.col(i), 2);
  }

  // Build the target matrix.
  arma::mat Y = arma::zeros<arma::mat>(10, nPoints);
  for (size_t i = 0; i < nPoints; i++)
  {
    if (i < nPoints / 2)
    {
      Y.col(i)(5) = 1;
    }
    else
    {
      Y.col(i)(8) = 1;
    }
  }*/

  arma::cube input(28, 28, 192); input.randu();
  InceptionLayer<> in(192, 64, 96, 128, 16,  32,  32);
  OneHotLayer outputLayer;
  arma::mat Y = arma::zeros<arma::mat>(10, 192);
  auto modules = std::tie(in);

  CNN<decltype(modules), decltype(outputLayer),
      RandomInitialization, MeanSquaredErrorFunction> net(modules, outputLayer);


  RMSprop<decltype(net)> opt(net, 0.01, 0.88, 1e-8, 10 * input.n_slices, 0);

  net.Train(input, Y, opt);
  std::cout << "reached here" << std::endl;
}

BOOST_AUTO_TEST_SUITE_END();
