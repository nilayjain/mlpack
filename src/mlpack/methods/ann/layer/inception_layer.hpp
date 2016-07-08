/**
 * @file inception_layer.hpp
 * @author Nilay Jain
 *
 * Definition of the ConvLayer class.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_INCEPTION_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_INCEPTION_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/activation_functions/rectifier_function.hpp>

#include <mlpack/methods/ann/layer/one_hot_layer.hpp>
#include <mlpack/methods/ann/layer/conv_layer.hpp>
#include <mlpack/methods/ann/layer/pooling_layer.hpp>
#include <mlpack/methods/ann/layer/softmax_layer.hpp>
#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>


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

class InceptionLayer
{
   //! Locally-stored filter/kernel width.
  size_t wfilter;

  //! Locally-stored filter/kernel height.
  size_t hfilter;

  //! Locally-stored number of input maps.
  size_t inMaps;

  //! Locally-stored number of output maps.
  size_t outMaps;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored convolution Layer objects.
  ConvLayer<> conv1, proj3, conv3, proj5, conv5, convPool;

  //! Locally-stored bias layer objects.
  BiasLayer<> bias1, biasProj3, bias3, biasProj5, bias5, biasPool;

  //! Locally-stored base layer objects.
  BaseLayer2D<RectifierFunction> base1, baseProj3, base3, baseProj5, base5, basePool;

  //! Locally-stored pooling layer object.
  PoolingLayer<MaxPooling> pool3;

 public:
  /**
   * Create the ConvLayer object using the specified number of input maps,
   * output maps, filter size, stride and padding parameter.
   *
   * @param inMaps The number of input maps.
   * @param outMaps The number of output maps.
   * @param wfilter Width of the filter/kernel.
   * @param wfilter Height of the filter/kernel.
   * @param xStride Stride of filter application in the x direction.
   * @param yStride Stride of filter application in the y direction.
   * @param wPad Spatial padding width of the input.
   * @param hPad Spatial padding height of the input.
   */
  InceptionLayer( const size_t inMaps,
                  const size_t outMaps,
                  const size_t out1,
                  const size_t proj3,
                  const size_t out3,
                  const size_t proj5,
                  const size_t out5,
                  const size_t projPool,
                  const size_t bias = 0) :
      inMaps(inMaps),
      outMaps(outMaps),
      out1(out1),
      proj3(proj3),
      out3(out3),
      proj5(proj5),
      out5(out5),
      projPool(projPool),
      bias(bias)
  {
    // set up all the layers.

    // 1x1 layer
    conv1(inMaps, out1, 1, 1);
    if(bias != 0)
      bias1(out1, bias);
    //base layer already set up.

    // 1x1 followed by 3x3 convolution
    proj3(inMaps, proj3, 1, 1);
    if(bias != 0)
      biasProj3(proj3, bias);
    //base layer already set up.

    conv3(proj3, out3, 3, 3);
    if(bias != 0)
      bias3(out3, bias);
    //base layer already set up.    

    // 1x1 followd by 5x5 convolution
    proj5(inMaps, proj5, 1, 1);
    if(bias != 0)
      biasProj5(proj5, bias);
    //base layer already set up.
    conv5(proj5, out5, 5, 5);
    if(bias != 0)
      bias5(out5, bias);
    //base layer already set up.

    // 3x3 pooling follwed by 1x1 convolution
    pool3(3);
    convPool(inMaps, poolProj, 1, 1);
    if(bias != 0)
      biasPool(poolProj, bias);
    //base layer already set up.
  }

  // perform forward passes for all the layers.

  template<typename MatType = arma::mat,
           typename CubeType = arma::cube>
  void Forward(const CubeType& input, CubeType& output)
  {
    conv1.InputParameter() = input;

    //! Forward pass for 1x1 conv path.
    conv1.Forward(conv1.InputParameter(), conv1.OutputParameter());
    bias1.Forward(conv1.OutputParameter(), bias1.OutputParameter());
    base1.Forward(bias1.OutputParameter(), base1.OutputParameter());

    proj3.Forward(input, proj3.OutputParameter());
    biasProj3.Forward(proj3.OutputParameter(), biasProj3.OutputParameter());
    baseProj3.Forward(biasProj3.OutputParameter(), baseProj3.OutputParameter());
    conv3.Forward(baseProj3.OutputParameter(), conv3.OutputParameter());
    bias3.Forward(conv3.OutputParameter(), bias3.OutputParameter());
    base3.Forward(bias3.OutputParameter(), base3.OutputParameter());    
    
    proj5.Forward(input, proj5.OutputParameter());
    biasProj5.Forward(proj5.OutputParameter(), biasProj5.OutputParameter());
    baseProj5.Forward(biasProj5.OutputParameter(), baseProj5.OutputParameter());
    conv5.Forward(baseProj5.OutputParameter(), conv5.OutputParameter());
    bias5.Forward(conv5.OutputParameter(), bias5.OutputParameter());
    base5.Forward(bias5.OutputParameter(), base5.OutputParameter());

    pool3.Forward(input, pool3.OutputParameter());
    convPool.Forward(pool3.OutputParameter(), convPool.OutputParameter());
    biasPool.Forward(convPool.OutputParameter(), biasPool.OutputParameter());
    basePool.Forward(biasPool.OutputParameter(), basePool.OutputParameter());

    //! assert that all have same number of rows and columns.
    //! to do assertion...
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
  template<typename MatType = arma::mat,
           typename CubeType = arma::cube>
  void Backward(CubeType&, CubeType& error, CubeType& )
  {
    CubeType in;
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

    basePool.Backward(in, error.slices(base5.OutputParameter().n_slices - 1), 
                          error.n_slices - 1, basePool.Delta());
    biasPool.Backward(in, basePool.Delta(), biasPool.Delta());
    convPool.Backward(in, biasPool.Delta(), convPool.Delta());
    pool3.Backward(in, convPool.Delta(), pool3.Delta());
  }

  template<typename MatType = arma::mat,
           typename CubeType = arma::cube>
  void Gradient(const CubeType&, CubeType& delta, CubeType&)
  {
    //Delta(delta);
    conv1.Gradient(conv1.InputParameter(), bias1.Delta(), conv1.Gradient());
    bias1.Gradient(bias1.InputParameter(), base1.Delta(), bias1.Gradient());
    base1.Gradient(base1.InputParameter(), this->Delta().slices(), base1.Gradient());
   
    proj3.Gradient(proj3.InputParameter(), biasProj3.Delta(), proj3.Gradient());
    biasProj3.Gradient(biasProj3.InputParameter(), baseProj3.Delta(), biasProj3.Gradient());
    baseProj3.Gradient(baseProj3.InputParameter(), conv3.Delta(), baseProj3.Gradient());
    conv3.Gradient(conv3.InputParameter(), bias3.Delta(), conv3.Gradient());
    bias3.Gradient(bias3.InputParameter(), base3.Delta(), bias3.Gradient());
    base3.Gradient(base3.InputParameter(), this->Delta().slices(), base3.Gradient());

    proj5.Gradient(proj5.InputParameter(), biasProj5.Delta(), proj5.Gradient());
    biasProj5.Gradient(biasProj5.InputParameter(), baseProj5.Delta(), biasProj5.Gradient());
    baseProj5.Gradient(baseProj5.InputParameter(), conv5.Delta(), baseProj5.Gradient());
    conv5.Gradient(conv5.InputParameter(), bias5.Delta(), conv5.Gradient());
    bias5.Gradient(bias5.InputParameter(), base5.Delta(), bias5.Gradient());
    base5.Gradient(base5.InputParameter(), this->Delta().slices(), base5.Gradient());

    convPool.Gradient(convPool.InputParameter(), biasPool.Delta(), convPool.Gradient());
    biasPool.Gradient(biasPool.InputParameter(), basePool.Delta(), basePool.Gradient());
    basePool.Gradient(basePool.InputParameter(), this->Delta().slices(), basePool.Gradient());
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
  CubeType const& Delta() const { return delta; }
  //! Modify the delta.
  CubeType& Delta() { return delta; }
}; // class InceptionLayer

} // namespace ann
} // namspace mlpack
