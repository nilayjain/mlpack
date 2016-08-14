/**
 * @file subnet_layer.hpp
 * @author Nilay Jain
 *
 * Definition of the SubnetLayer class.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_SUBNET_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_SUBNET_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/network_util.hpp>


namespace mlpack {
namespace ann /** Artificial Neural Network. */ {
/**
 * @tparam LayerTypes represents the collection of layers that make subnet_layer.
 */
template <typename LayerTypes,
          typename InputDataType = arma::cube,
          typename OutputDataType = arma::cube>
class SubnetLayer
{
 public:
  
  SubnetLayer(const size_t numLayers,
              LayerTypes &&layers) :
              numLayers(numLayers),
  layers(std::forward<LayerTypes>(layers))
  {
    /* Nothing to do. */
  }

  template<typename eT>  
  void Forward(const arma::Cube<eT>& input, arma::Cube<eT>& )
  {
    Forward(input, layers);
  }

  template<typename eT>  
  void Backward(arma::Cube<eT>&, arma::Cube<eT>& error, arma::Cube<eT>& )
  {
    Backward(error, layers);
  }

  template<typename eT>
  void Gradient(const arma::Cube<eT>&, arma::Cube<eT>& delta, arma::Cube<eT>&)
  {
    UpdateGradients(layers);
  }

  template<size_t I = 0, typename DataType, typename... Tp>
  void Forward(const DataType& input, std::tuple<Tp...>& network)
  {
    std::get<I>(network).InputParameter() = input;

    std::get<I>(network).Forward(std::get<I>(network).InputParameter(),
                           std::get<I>(network).OutputParameter());

    ForwardTail<I + 1, Tp...>(network);
  }

  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  ForwardTail(std::tuple<Tp...>& network)
  {
    LinkParameter(network);
  }

  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  ForwardTail(std::tuple<Tp...>& network)
  {
    std::get<I>(network).Forward(std::get<I - 1>(network).OutputParameter(),
        std::get<I>(network).OutputParameter());

    ForwardTail<I + 1, Tp...>(network);
  }

    /**
   * Run a single iteration of the feed backward algorithm, using the given
   * error of the output layer. Note that we iterate backward through the
   * layer modules.
   */
  template<size_t I = 1, typename DataType, typename... Tp>
  typename std::enable_if<I < (sizeof...(Tp) - 1), void>::type
  Backward(const DataType& error, std::tuple<Tp...>& network)
  {
    std::get<sizeof...(Tp) - I>(network).Backward(
        std::get<sizeof...(Tp) - I>(network).OutputParameter(), error,
        std::get<sizeof...(Tp) - I>(network).Delta());

    BackwardTail<I + 1, DataType, Tp...>(error, network);
  }

  template<size_t I = 1, typename DataType, typename... Tp>
  typename std::enable_if<I == (sizeof...(Tp)), void>::type
  BackwardTail(const DataType& /* unused */,
               std::tuple<Tp...>& /* unused */) { /* Nothing to do here */ }

  template<size_t I = 1, typename DataType, typename... Tp>
  typename std::enable_if<I < (sizeof...(Tp)), void>::type
  BackwardTail(const DataType& error, std::tuple<Tp...>& network)
  {
    std::get<sizeof...(Tp) - I>(network).Backward(
        std::get<sizeof...(Tp) - I>(network).OutputParameter(),
        std::get<sizeof...(Tp) - I + 1>(network).Delta(),
        std::get<sizeof...(Tp) - I>(network).Delta());

    BackwardTail<I + 1, DataType, Tp...>(error, network);
  }

  /**
   * Iterate through all layer modules and update the the gradient using the
   * layer defined optimizer.
   */
  template<
      size_t I = 0,
      size_t Max = std::tuple_size<LayerTypes>::value - 1,
      typename... Tp
  >
  typename std::enable_if<I == Max, void>::type
  UpdateGradients(std::tuple<Tp...>& /* unused */) { /* Nothing to do here */ }

  template<
      size_t I = 0,
      size_t Max = std::tuple_size<LayerTypes>::value - 1,
      typename... Tp
  >
  typename std::enable_if<I < Max, void>::type
  UpdateGradients(std::tuple<Tp...>& network)
  {
    Update(std::get<I>(network), std::get<I>(network).OutputParameter(),
           std::get<I + 1>(network).Delta());

    UpdateGradients<I + 1, Max, Tp...>(network);
  }

  template<typename T, typename P, typename D>
  typename std::enable_if<
      HasGradientCheck<T, P&(T::*)()>::value, void>::type
  Update(T& layer, P& /* unused */, D& delta)
  {
    layer.Gradient(layer.InputParameter(), delta, layer.Gradient());
  }

  template<typename T, typename P, typename D>
  typename std::enable_if<
      !HasGradientCheck<T, P&(T::*)()>::value, void>::type
  Update(T& /* unused */, P& /* unused */, D& /* unused */)
  {
    /* Nothing to do here */
  }


  /**
   * Link the calculated activation with the connection layer.
   */
  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  LinkParameter(std::tuple<Tp...>& /* unused */) { /* Nothing to do here */ }

  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  LinkParameter(std::tuple<Tp...>& network)
  {
    if (!LayerTraits<typename std::remove_reference<
        decltype(std::get<I>(network))>::type>::IsBiasLayer)
    {
      std::get<I>(network).InputParameter() = std::get<I - 1>(
          network).OutputParameter();
    }

    LinkParameter<I + 1, Tp...>(network);
  }
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
 
 private:
  
  //! number of layers to concatenate.
  size_t numLayers;

  //! Instantiated convolutional neural network.
  LayerTypes layers;

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
} // namespace mlpack

#endif
