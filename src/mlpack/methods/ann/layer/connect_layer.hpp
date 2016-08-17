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

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

// two networks can emerge from a connect layer.
template<typename NetworkTypeA, typename NetworkTypeB>
class ConnectLayer
{
 public:
  template<typename NetworkA, typename NetworkB>
  ConnectLayer(NetworkA networkA, NetworkB networkB):
    networkA(std::forward<NetworkA>(networkA)),
    networkB(std::forward<NetworkB>(networkB))
  {
    static_assert(std::is_same<typename std::decay<NetworkA>::type,
                  NetworkTypeA>::value,
                  "The type of networkA must be NetworkTypeA.");

    static_assert(std::is_same<typename std::decay<NetworkB>::type,
                  NetworkTypeB>::value,
                  "The type of networkB must be NetworkTypeB.");
  }


  template<typename eT>
  void Forward(const arma::Cube<eT>& input, arma::Cube<eT>& )
  {
    networkA.Forward(input, networkA.Layers());
    networkB.Forward(input, networkB.Layers());
  }

  template<typename eT>
  void Backward(arma::Cube<eT>&, arma::Cube<eT>& error, arma::Cube<eT>& )
  {
    networkA.Backward(networkA.error, networkA.Layers());
    networkB.Backward(networkB.error, networkB.Layers());
  }

  template<typename eT>
  void Gradient(const arma::Cube<eT>&, arma::Cube<eT>& delta, arma::Cube<eT>&)
  {
    networkA.UpdateGradients(networkA.Layers());
    networkB.UpdateGradients(networkB.Layers());
  }

 private:

  NetworkTypeA networkA;
  NetworkTypeB networkB;
};

} // namespace ann
} // namspace mlpack

#endif
