/**
 * @file googlenet.hpp
 * @author Nilay Jain
 *
 * Implementation of Googlenet.
 */
#ifndef MLPACK_METHODS_ANN_GOOGLENET_HPP
#define MLPACK_METHODS_ANN_GOOGLENET_HPP

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>

#include <mlpack/methods/ann/layer/one_hot_layer.hpp>
#include <mlpack/methods/ann/layer/conv_layer.hpp>
#include <mlpack/methods/ann/layer/pooling_layer.hpp>
#include <mlpack/methods/ann/pooling_rules/max_pooling.hpp>
#include <mlpack/methods/ann/pooling_rules/mean_pooling.hpp>
#include <mlpack/methods/ann/layer/softmax_layer.hpp>
#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/layer/inception_layer.hpp>
#include <mlpack/methods/ann/layer/concat_layer.hpp> 
#include <mlpack/methods/ann/layer/subnet_layer.hpp>
#include <mlpack/methods/ann/layer/connect_layer.hpp>
#include <mlpack/methods/ann/layer/dropout_layer.hpp>

#include <mlpack/methods/ann/performance_functions/mse_function.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>

#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/cnn.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {
class GoogleNet
{
 public:
  ConvLayer<> conv1, conv2, conv3, conv4;
  BaseLayer2D<RectifierFunction> base1, base2, base3, base4;
  BaseLayer<IdentityFunction, arma::cube, arma::cube> id1, id2;
  InceptionLayer<> inception3a, inception3b, inception4a, inception4b,
      inception4c, inception4d, inception4e, inception5a, inception5b;
  LinearMappingLayer<> linear1, linear2, linear3, linear4, linear5;
  DropoutLayer<> drop1, drop2, drop3;
  SoftmaxLayer<> softmax1, softmax2, softmax3;
  PoolingLayer<MaxPooling> pool1, pool2, pool3, pool4;
  PoolingLayer<MeanPooling> pool5, pool6, pool7;
  //ConnectLayer connect1, connect2;
  OneHotLayer output1, output2, output3, output4, output5;
  GoogleNet() :
    conv1(3, 64, 7, 7, 2, 2, 3, 3),
    pool1(3, 2),
    conv2(64, 192, 3, 3, 1, 1, 1, 1),
    pool2(3, 2),
    inception3a(192, 64, 96, 128, 16, 32, 32),
    inception3b(256, 128, 128, 192, 32, 96, 64),
    pool3(3, 2),
    inception4a(480, 192, 96, 208, 16, 48, 64),
    inception4b(512, 160, 112, 224, 24, 64, 64),
    inception4c(512, 128, 128, 256, 24, 64, 64),
    inception4d(512, 112, 144, 288, 32, 64, 64),
    inception4e(528, 256, 160, 320, 32, 128, 128),
    pool4(3, 2),
    inception5a(832, 256, 160, 320, 32, 128, 128),
    inception5b(832, 384, 192, 384, 48, 128, 128),
    pool5(7, 1),
    drop1(0.4),
    linear1(1024, 1000),
    pool6(5, 3),
    conv3(512, 128, 1, 1, 1, 1, 1, 1),
    linear2(4 * 4 * 128, 1024),
    drop2(0.7),
    linear3(1024, 1000),
    pool7(5, 3),
    conv4(528, 128, 1, 1, 1, 1, 1, 1),
    linear4(4 * 4 * 128, 1024),
    drop3(0.7),
    linear5(1024, 1000)
  {

    auto aux2 = std::tie(pool7, conv4, base4, linear4, drop3, linear5, softmax3);
    CNN<decltype(aux2), decltype(output3),
        RandomInitialization, MeanSquaredErrorFunction> auxNet2(aux2, output3);

    auto main3 = std::tie(id2, inception4e, pool4, inception5a, inception5b,
                          pool5, drop1, linear1, softmax1);
    CNN<decltype(main3), decltype(output1),
        RandomInitialization, MeanSquaredErrorFunction> mainNet3(main3, output1);

    auto connect2 = ConnectLayer<decltype(mainNet3), decltype(auxNet2)>(mainNet3, auxNet2);
    
    auto aux1 = std::tie(pool6, conv3, base3, linear2, drop2, linear3, softmax2);
    CNN<decltype(aux1), decltype(output2),
        RandomInitialization, MeanSquaredErrorFunction> auxNet1(aux1, output2);

    auto main2 = std::tie(id1, inception4b, inception4c, inception4d, connect2);
    CNN<decltype(main2), decltype(output4),
        RandomInitialization, MeanSquaredErrorFunction> mainNet2(main2, output4);    

    auto connect1 = ConnectLayer<decltype(mainNet2), decltype(auxNet1)>(mainNet2, auxNet1);

    auto main1 = std::tie(conv1, pool1, conv2, base2, pool2, inception3a, inception3b, 
                          pool3, inception4a, connect1);
    CNN<decltype(main1), decltype(output5),
        RandomInitialization, MeanSquaredErrorFunction> mainNet1(main1, output5);
  }

  void Train()
  {
     
  }
};

} // namespace ann
} // namspace mlpack

#endif
