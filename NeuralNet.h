#ifndef NEURAL_NET_H_
#define NEURAL_NET_H_

#include <cmath>
#include <vector>

class NeuralNet
{
  public:
    /// @brief create a randomized neural net with the specified layers
    /// @param _layer_sizes the size of each layer, in 
    NeuralNet(std::vector<int> _layer_sizes);

    std::vector<double> predict(const std::vector<double>& _inputs);

  private:
    const std::vector<int> m_layer_sizes;


    /// activations[layer][neuron]
    std::vector<std::vector<double>> m_activations;
    
    /// sum before going through activation function
    /// sums[layer][neuron]
    /// The 0th layer doesn't have sum
    std::vector<std::vector<double>> m_sums;

    /// biases[layer][neuron]
    /// The 0th layer doesn't have bias
    std::vector<std::vector<double>> m_biases;

    /// weights[starting_layer][starting_perceptron][ending_perceptron]
    std::vector<std::vector<std::vector<double>>> m_weights;


    /// Activation function - sigmoid
    double sigmoid(double x);

    double sigmoid_derivation(double x);
};

#endif // NEURAL_NET_H_