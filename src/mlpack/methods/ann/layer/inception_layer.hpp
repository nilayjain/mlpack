/**
 * @file inception_layer.hpp
 * @author Nilay Jain
 *
 * Definition of the InceptionLayer class.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_INCEPTION_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_INCEPTION_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/activation_functions/rectifier_function.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/one_hot_layer.hpp>
#include <mlpack/methods/ann/layer/conv_layer.hpp>
#include <mlpack/methods/ann/layer/pooling_layer.hpp>
#include <mlpack/methods/ann/layer/softmax_layer.hpp>
#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>

using namespace mlpack;
using namespace mlpack::ann;
//using namespace mlpack::optimization;

/*
  this is a naive (hard coded) implementation of inception layer, 
  if we want to automate this (network in network) layering process, 
  we can make a sublayer class 
  (or a subnetwork class, which essentially is a collection of layers) , 
  pass the layers we want to construct as args, 
  and call the recursive procedures of Forward, Backward and Gradient as written 
  in cnn.hpp . We will implement these recursive procedures in a sublayer class. 
  I think we also need to make a concatenation layer which concatenates outputs
  and then divides/joins the backward/forward passes, respectively as done 
  here very simply using slices.

  so we will pass the layers as (conv1, bias1, base1- concat with - proj3, biasPorj3, basePorj3, conv3
                                  , bias3, base3 - concat with - ... and so on) to the sublayer class.

  (we need to see how we should pass the information that two layers must be concatenated)
  then inception layer would just be an example layer which is an instantiation of sublayer class with
  appropriate layers.

  do you think we should do it this way?

*/
namespace mlpack {
namespace ann /** Artificial Neural Network. */ {


//! input --> inception_layer --> output.

/*

Inception module of GoogLeNet.

It applies four different functions to the input array and concatenates
their outputs along the channel dimension. Three of them are 2D
convolutions of sizes 1x1, 3x3 and 5x5. Convolution paths of 3x3 and 5x5
sizes have 1x1 convolutions (called projections) ahead of them. The other
path consists of 1x1 convolution (projection) and 3x3 max pooling.

The output array has the same spatial size as the input. In order to
satisfy this, Inception module uses appropriate padding for each
convolution and pooling.

See: `Going Deeper with Convolutions <http://arxiv.org/abs/1409.4842>`_.

*/

// make all the layers..
// write forward method
// write backward method
// write gradient method
template<typename InputDataType = arma::cube,
         typename OutputDataType = arma::cube>
class InceptionLayer
{
 public:
  //! Locally-stored number of input maps.
  size_t inMaps;

  //! Locally-stored number of output maps.
  //size_t outMaps;

  //! bias value
  size_t bias;

  size_t out1, out3, out5, projSize3, projSize5, poolProj;

  //! Locally-stored delta object.
  OutputDataType delta;

  ConvLayer<> conv1;
  ConvLayer<> proj3;
  ConvLayer<> conv3;
  ConvLayer<> proj5;
  ConvLayer<> conv5;
  ConvLayer<> convPool;
  BaseLayer2D<RectifierFunction> base1, baseProj3, base3, baseProj5, base5, basePool;
  BiasLayer2D<> bias1, biasProj5, biasProj3, bias3, bias5, biasPool;
  PoolingLayer<MaxPooling> pool3;
  /**
   *
   * @param inMaps The number of input maps.
   * @param outMaps The number of output maps.
   *
    */
  InceptionLayer( const size_t inMaps,
                  const size_t out1,
                  const size_t projSize3,
                  const size_t out3,
                  const size_t projSize5,
                  const size_t out5,
                  const size_t poolProj,
                  const size_t bias = 0) :
      inMaps(inMaps),
      bias(bias),
      out1(out1),
      out3(out3),
      out5(out5),
      projSize5(projSize5),
      projSize3(projSize3),
      poolProj(poolProj),
      conv1(inMaps, out1, 1, 1),
      proj3(inMaps, projSize3, 1, 1),
      conv3(projSize3, out3, 3, 3, 1, 1, 1, 1),
      proj5(inMaps, projSize5, 1, 1),
      conv5(projSize5, out5, 5, 5, 1, 1, 2, 2),
      convPool(inMaps, poolProj, 1, 1, 1, 1, 1, 1),
      pool3(3),
      bias1(out1, bias),
      biasProj3(projSize3, bias),
      bias3(out3, bias),
      biasProj5(projSize5, bias),
      bias5(out5, bias),
      biasPool(poolProj, bias)
  {
    /*
      Example:
      input : 28 x 28 x 192.
      InceptionLayer<> in(192, 64, 96, 128, 16,  32,  32);

      conv1: 28 x 28 x 64

      proj3: 28 x 28 x 96
      conv3: 26 x 26 x 128

      proj5: 28 x 28 x 16
      conv5: 24 x 24 x 32

      pool3: 8 x 8 x 192 
      convPool: 8 x 8 x 32
    */
    /*
    set up all the layers.
      1x1 layer
  conv1(inMaps, out1, 1, 1);
  if(bias != 0)
  BiasLayer<> bias1(out1, bias);
  BaseLayer2D<RectifierFunction> base1;
  
  1x1 followed by 3x3 convolution
  ConvLayer<> proj3(inMaps, projSize3, 1, 1);
  if(bias != 0)
    BiasLayer<> biasProj3(projSize3, bias);
 BaseLayer2D<RectifierFunction> baseProj3;

  ConvLayer<> conv3(projSize3, out3, 3, 3);
  if(bias != 0)
   BiasLayer<> bias3(out3, bias);
  BaseLayer2D<RectifierFunction> base3;

  1x1 followd by 5x5 convolution
  ConvLayer<> proj5(inMaps, projSize5, 1, 1);
  if(bias != 0)
    BiasLayer<> biasProj5(projSize5, bias);
  BaseLayer2D<RectifierFunction> baseProj5;
  ConvLayer<> conv5(projSize5, out5, 5, 5);
  if(bias != 0)
    BiasLayer<> bias5(out5, bias);
  BaseLayer2D<RectifierFunction> base5;

  3x3 pooling follwed by 1x1 convolution
  PoolingLayer<MaxPooling> pool3(3);
  ConvLayer<> convPool(inMaps, poolProj, 1, 1);
  if(bias != 0)
    BiasLayer<> biasPool(poolProj, bias);
  BaseLayer2D<RectifierFunction> basePool;
*/
  std::cout << "Constructor called success" << std::endl;
  }

  // perform forward passes for all the layers.

  template<typename eT>
  void Forward(const arma::Cube<eT>& input, arma::Cube<eT>& output)
  {
    // Example input 28 x 28 x 192.
    conv1.InputParameter() = input;
    //this->InputParameter() = input;
    //! Forward pass for 1x1 conv path.

    conv1.Forward(conv1.InputParameter(), conv1.OutputParameter());
    // no InputParameter() update for bias term.
    bias1.Forward(conv1.OutputParameter(), bias1.OutputParameter());
    base1.InputParameter() = bias1.OutputParameter();
    base1.Forward(bias1.OutputParameter(), base1.OutputParameter());

    proj3.InputParameter() = input;
    proj3.Forward(input, proj3.OutputParameter());
    biasProj3.Forward(proj3.OutputParameter(), biasProj3.OutputParameter());
    baseProj3.InputParameter() = biasProj3.OutputParameter();
    baseProj3.Forward(biasProj3.OutputParameter(), baseProj3.OutputParameter());
    conv3.InputParameter() = baseProj3.OutputParameter();
    conv3.Forward(baseProj3.OutputParameter(), conv3.OutputParameter());
    bias3.Forward(conv3.OutputParameter(), bias3.OutputParameter());
    base3.InputParameter() = bias3.OutputParameter();
    base3.Forward(bias3.OutputParameter(), base3.OutputParameter());    
    
    proj5.InputParameter() = input;
    proj5.Forward(input, proj5.OutputParameter());
    biasProj5.Forward(proj5.OutputParameter(), biasProj5.OutputParameter());
    baseProj5.InputParameter() = biasProj5.OutputParameter();
    baseProj5.Forward(biasProj5.OutputParameter(), baseProj5.OutputParameter());
    conv5.InputParameter() = baseProj5.OutputParameter();
    conv5.Forward(baseProj5.OutputParameter(), conv5.OutputParameter());
    bias5.Forward(conv5.OutputParameter(), bias5.OutputParameter());
    base5.InputParameter() = bias5.OutputParameter();
    base5.Forward(bias5.OutputParameter(), base5.OutputParameter());
 
    pool3.InputParameter() = input;
    pool3.Forward(input, pool3.OutputParameter());
    convPool.InputParameter() = pool3.OutputParameter();
    convPool.Forward(pool3.OutputParameter(), convPool.OutputParameter());
    biasPool.Forward(convPool.OutputParameter(), biasPool.OutputParameter());
    basePool.InputParameter() = biasPool.OutputParameter();
    basePool.Forward(convPool.OutputParameter(), basePool.OutputParameter());

    //! assert that all have same number of rows and columns.

/*    std::cout << arma::size(base1.OutputParameter()) << std::endl;
    std::cout << arma::size(base3.OutputParameter()) << std::endl;
    std::cout << arma::size(base5.OutputParameter()) << std::endl;
    std::cout << arma::size(basePool.OutputParameter()) << std::endl;*/
    output = arma::join_slices( 
              arma::join_slices(
                arma::join_slices( 
                  base1.OutputParameter(), base3.OutputParameter() ), 
                  base5.OutputParameter() ), basePool.OutputParameter());

  }

  //! perform backward passes for all the layers.
  
  // Backward(error, network)
  // error : backpropagated error
  // g : calcualted gradient.
  // populate delta for all the layers.
  // size of delta = size of inputParameter.
  template<typename eT>
  void Backward(arma::Cube<eT>&, arma::Cube<eT>& error, arma::Cube<eT>& )
  {
    InputDataType in;
    base1.Backward(in, error.slices(0, base1.OutputParameter().n_slices - 1), base1.Delta());
    bias1.Backward(in, base1.Delta(), bias1.Delta());
    conv1.Backward(in, bias1.Delta(), conv1.Delta());

    base3.Backward(in, error.slices(base1.OutputParameter().n_slices, 
                base3.OutputParameter().n_slices - 1), base3.Delta());
    bias3.Backward(in, base3.Delta(), bias3.Delta());
    conv3.Backward(in, bias3.Delta(), conv3.Delta());
    baseProj3.Backward(in, conv3.Delta(), baseProj3.Delta());
    biasProj3.Backward(in, baseProj3.Delta(), biasProj3.Delta());
    proj3.Backward(in, biasProj3.Delta(), proj3.Delta());

    base5.Backward(in, error.slices(base3.OutputParameter().n_slices, 
                base5.OutputParameter().n_slices - 1), base5.Delta());

    bias5.Backward(in, base5.Delta(), bias5.Delta());
    conv5.Backward(in, bias5.Delta(), conv5.Delta());
    baseProj5.Backward(in, conv5.Delta(), baseProj5.Delta());
    biasProj5.Backward(in, baseProj5.Delta(), biasProj5.Delta());
    proj5.Backward(in, biasProj5.Delta(), proj5.Delta());

    basePool.Backward(in, error.slices(base5.OutputParameter().n_slices - 1, 
                          error.n_slices - 1), basePool.Delta());
    biasPool.Backward(in, basePool.Delta(), biasPool.Delta());
    convPool.Backward(in, biasPool.Delta(), convPool.Delta());
    pool3.Backward(in, convPool.Delta(), pool3.Delta());
  }

  template<typename eT>
  void Gradient(const arma::Cube<eT>&, arma::Cube<eT>& delta, arma::Cube<eT>&)
  {
    Delta(delta);
    conv1.Gradient(conv1.InputParameter(), bias1.Delta(), conv1.Gradient());
    bias1.Gradient(bias1.InputParameter(), delta.
        slices(0, bias1.OutputParameter().n_slices - 1), bias1.Gradient());
    /*base1.Gradient(base1.InputParameter(), delta.
        slices(0, base1.OutputParameter().n_slices - 1), base1.Gradient());*/
   
    proj3.Gradient(proj3.InputParameter(), biasProj3.Delta(), proj3.Gradient());
    biasProj3.Gradient(biasProj3.InputParameter(), conv3.Delta(), biasProj3.Gradient());
    //baseProj3.Gradient(baseProj3.InputParameter(), conv3.Delta(), baseProj3.Gradient());
    conv3.Gradient(conv3.InputParameter(), bias3.Delta(), conv3.Gradient());
    bias3.Gradient(bias3.InputParameter(), delta.
        slices(bias1.OutputParameter().n_slices, bias3.OutputParameter().n_slices - 1), bias3.Gradient());
   /* base3.Gradient(base3.InputParameter(), delta.
        slices(base1.OutputParameter().n_slices, base3.OutputParameter().n_slices - 1),
         base3.Gradient());
   */ 
    proj5.Gradient(proj5.InputParameter(), biasProj5.Delta(), proj5.Gradient());
    biasProj5.Gradient(biasProj5.InputParameter(), conv5.Delta(), biasProj5.Gradient());
    //baseProj5.Gradient(baseProj5.InputParameter(), conv5.Delta(), baseProj5.Gradient());
    conv5.Gradient(conv5.InputParameter(), bias5.Delta(), conv5.Gradient());
    bias5.Gradient(bias5.InputParameter(), delta.
        slices(bias3.OutputParameter().n_slices, bias5.OutputParameter().n_slices - 1), bias5.Gradient());
    /*base5.Gradient(base5.InputParameter(), delta.
        slices(base3.OutputParameter().n_slices, base5.OutputParameter().n_slices - 1),
         base5.Gradient());*/

    convPool.Gradient(convPool.InputParameter(), biasPool.Delta(), convPool.Gradient());
    biasPool.Gradient(biasPool.InputParameter(), delta.
        slices(bias5.OutputParameter().n_slices, delta.n_slices - 1), biasPool.Gradient());
    /*basePool.Gradient(basePool.InputParameter(), delta.
        slices(base5.OutputParameter().n_slices, delta.n_slices - 1),
         basePool.Gradient());*/
  }
  /*
  //! visual of subnetwork
  auto modules = 
      std::tie(conv1, bias1, base1,
              proj3, biasProj3, baseProj3,
              conv3, bias3, base3,
              proj5, biasProj5, baseProj5,
              conv5, bias5, base5,
              pool3, convPool, biasPool, basePool);
  */

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the input parameter.
  InputDataType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class InceptionLayer

} // namespace ann
} // namspace mlpack

#endif
