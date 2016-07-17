/**
 * @file concat_layer.hpp
 * @author Nilay Jain
 *
 * Definition of the InceptionLayer class.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_CONCAT_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_CONCAT_LAYER_HPP

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

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {
/**
 * @tparam JoinLayers Contains all layer modules that need be concatenated.
 */
template <typename JoinLayers>
class ConcatLayer
{
 public:
  ConcatLayer(JoinLayers &&layers,
              size_t numLayers) :
  layers(std::forward<JoinLayers>(layers)),
  numLayers(numLayers)
  {

  }

  template<typename eT>
  void Forward(const arma::Cube<eT>& , arma::Cube<eT>& output)
  {
    ForwardTail(layers, output);
  }

  template<size_t I = 0, typename DataType, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  void ForwardTail(std::tuple<Tp...>& layers, DataType& output)
  {
    /* Nothing to do. */
  }

  template<size_t I = 0, typename DataType, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  void ForwardTail(std::tuple<Tp...>& layers, DataType& output)
  {
    output = arma::join_slices(output, std::get<I>(layers).OutputParameter());
    ForwardTail<I + 1, DataType, Tp...>(layers, output);
  }

  template<typename eT>
  void Backward(arma::Cube<eT>&, arma::Cube<eT>& error, arma::Cube<eT>& )
  {
    size_t slice_idx = 0;
    BackwardTail(layers, error, slice_idx);
  }

  template<size_t I = 0, typename DataType, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  BackwardTail(std::tuple<Tp...>& layers, const DataType& error,  size_t slice_idx)
  {
    /* Nothing to do. */
  }

  template<size_t I = 0, typename DataType, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  BackwardTail(std::tuple<Tp...>& layers, const DataType& error,  size_t slice_idx)
  {
    DataType subError = error.slices(slice_idx, 
        slice_idx + std::get<I>(layers).OutputParameter().n_slices - 1);
    slice_idx += std::get<I>(layers).OutputParameter().n_slices;
    std::get<I>(layers).Backward(std::get<I>(layers).OutputParameter(), subError, 
          std::get<I>(layers).Delta());
    BackwardTail<I + 1, DataType, Tp...>(layers, error, slice_idx);
  }

  template<typename eT>
  void Gradient(const arma::Cube<eT>&, arma::Cube<eT>& delta, arma::Cube<eT>&)
  {
    size_t slice_idx = 0;
    GradientTail(layers, delta, slice_idx);
  }

  template<size_t I = 0, typename DataType, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  GradientTail(std::tuple<Tp...>& layers, const DataType& error,  size_t slice_idx)
  {
    DataType deltaNext = delta.slices(slice_idx,
        slice_idx + )
  }
    
 private:

  //! Get the weights.
  OutputDataType const& Weights() const { return weights; }
  //! Modify the weights.
  OutputDataType& Weights() { return weights; }

  //! Get the input parameter.
  InputDataType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the gradient.
  OutputDataType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return gradient; }

  //! number of layers to concatenate.
  size_t numLayers;

  //! Instantiated convolutional neural network.
  JoinLayers layers;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
};

} // namespace ann
} // namspace mlpack

#endif
