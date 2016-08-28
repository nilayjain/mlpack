/**
 * @file connect_layer.hpp
 * @author Nilay Jain
 * Definition of the ConnectLayer class.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_CONNECT_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_CONNECT_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/cnn.hpp>
#include <mlpack/methods/ann/network_util.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

// two networks can emerge from a connect layer.
template<
    typename NetworkTypeA,
    typename NetworkTypeB,
    typename InputDataType = arma::cube,
    typename OutputDataType = arma::mat
>
class ConnectLayer
{
 public:
  template<typename NetworkA, typename NetworkB>
  ConnectLayer(NetworkA networkA, NetworkB networkB):
    networkA(std::forward<NetworkA>(networkA)),
    networkB(std::forward<NetworkB>(networkB)),
    index(0),
    firstRun(true)
  {
    static_assert(std::is_same<typename std::decay<NetworkA>::type,
                  NetworkTypeA>::value,
                  "The type of networkA must be NetworkTypeA.");

    static_assert(std::is_same<typename std::decay<NetworkB>::type,
                  NetworkTypeB>::value,
                  "The type of networkB must be NetworkTypeB.");


    networkASize = NetworkSize(networkA.Layers());
    networkBSize = NetworkSize(networkB.Layers());
    weights.set_size(networkASize + networkBSize, 1);
  }

  template<size_t I = 0, typename... Tp, typename LayerTypes>
  typename std::enable_if<I == sizeof...(Tp), size_t>::type
  NetworkSize(LayerTypes& /* unused */)
  {
    return 0;
  }

  template<size_t I = 0, typename... Tp, typename LayerTypes>
  typename std::enable_if<I < sizeof...(Tp), size_t>::type
  NetworkSize(LayerTypes& network)
  {
    return LayerSize(std::get<I>(network), std::get<I>(
        network).OutputParameter()) + NetworkSize<I + 1, Tp...>(network);
  }

  template<size_t I = 0, typename... Tp, typename LayerTypes>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  NetworkWeights(arma::mat& weights,
                 LayerTypes& network,
                 size_t offset)
  {
    NetworkWeights<I + 1, Tp...>(weights, network,
        offset + LayerWeights(std::get<I>(network), weights,
        offset, std::get<I>(network).OutputParameter()));

  }

  template<size_t I = 0, typename... Tp, typename LayerTypes>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  NetworkWeights(arma::mat& /* unused */,
                 LayerTypes& /* unused */,
                 size_t /* unused */)
  {
    /* Nothing to do here */
  }

  
  void Forward(const InputDataType& input, OutputDataType& )
  {
    if (firstRun)
    {
      NetworkWeights(networkA.Parameter(), networkA.Layers(), 0);
      NetworkWeights(networkB.Parameter(), networkB.Layers(), networkASize);
      firstRun = false;
    }
    InputParameter() = input;
    networkA.Forward(input, networkA.Layers());
    networkA.OutputError(arma::mat(networkA.Responses().colptr(index),
                         networkA.Responses().n_rows, 1, false,
                         true), networkA.Error(), networkA.Layers());

    networkB.Forward(input, networkB.Layers());
    networkB.OutputError(arma::mat(networkB.Responses().colptr(index), 
                         networkB.Responses().n_rows, 1, false,
                         true), networkB.Error(), networkB.Layers());

    delta = networkA.Error() + networkB.Error();
    // for debug:
    // std::cout << networkA.Error().size() << std::endl;
    // std::cout << networkB.Error().size() << std::endl;
    // std::cout << input.size() << std::endl;
    storeDelta();
    index++;
  }


  template<typename InputType, typename InputErrorType, typename OutputErrorType>
  void Backward(const InputType& /* unused */,
                const InputErrorType& /* unused */,
                OutputErrorType& /* unused */)
  {
    networkA.Backward(networkA.Error(), networkA.Layers());
    networkB.Backward(networkB.Error(), networkB.Layers());
  }

  template<typename InputType, typename ErrorType, typename GradientType>
  void Gradient(const InputType& /* unused */,
                const ErrorType& /* unused */,
                GradientType& gradient)
  {
    NetworkGradients(gradient, networkA.Layers());
    NetworkGradients(gradient, networkB.Layers(), networkASize);

    networkA.UpdateGradients(networkA.Layers());
    networkB.UpdateGradients(networkB.Layers());
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

  template<class DeltaType = InputDataType>
  typename std::enable_if <std::is_same<DeltaType, arma::mat>::value, void>::type
  storeDelta()
  {
    /* Nothing to do */
  }
  template<class DeltaType = InputDataType>
  typename std::enable_if <!std::is_same<DeltaType, arma::mat>::value, void>::type
  storeDelta()
  {    
    arma::cube error(InputParameter().n_rows, InputParameter().n_cols, 
                        InputParameter().n_slices);
    size_t rowIdx = 0, colIdx = 0;
    for (size_t i = 0; i < InputParameter().n_slices; ++i)
    {
      error.slice(i) = delta.submat(rowIdx, 
                          rowIdx + InputParameter().n_rows - 1, 
                          colIdx, colIdx + InputParameter().n_cols - 1);
      rowIdx += InputParameter().n_rows;
      colIdx += InputParameter().n_cols;
    }

    Delta() = error;
  }

  //! Get the delta.
  template<class DeltaType = InputDataType>
  typename std::enable_if <std::is_same<DeltaType, arma::mat>::value, OutputDataType&>::type const
  Delta() const 
  {
    return delta; 
  }

  template<class DeltaType = InputDataType>
  typename std::enable_if <!std::is_same<DeltaType, arma::mat>::value, InputDataType&>::type const
  Delta() const 
  {
    return modifiedDelta;
  }
  
  //! Get the delta.
  template<class DeltaType = InputDataType>
  typename std::enable_if <std::is_same<DeltaType, arma::mat>::value, OutputDataType& >::type
  Delta() 
  {
    return delta; 
  }

  template<class DeltaType = InputDataType>
  typename std::enable_if <!std::is_same<DeltaType, arma::mat>::value, InputDataType&>::type
  Delta() 
  {
    return modifiedDelta;
  }

  //! Get the gradient.
  OutputDataType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return gradient; }
 
  NetworkTypeA networkA;
  NetworkTypeB networkB;

 private:

  size_t networkASize;
  size_t networkBSize;
  
  //! the index for the sample in the given dataset.
  size_t index;

  //! Locally-stored run parameter used to initalize the layer once.
  bool firstRun;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored modifiedDelta object.
  InputDataType modifiedDelta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! total error in the nets in connect_layer
  arma::mat error;
};

//! Layer traits for the Connect layer.
template<
    typename NetworkTypeA,
    typename NetworkTypeB,
    typename InputDataType,
    typename OutputDataType
>
class LayerTraits<ConnectLayer<NetworkTypeA, NetworkTypeB, 
                      InputDataType, OutputDataType> >
{
 public:
  static const bool IsBinary = false;
  static const bool IsOutputLayer = false;
  static const bool IsBiasLayer = false;
  static const bool IsLSTMLayer = false;
  static const bool IsConnection = false;
  static const bool IsConnectLayer = true;
};

} // namespace ann
} // namspace mlpack

#endif
