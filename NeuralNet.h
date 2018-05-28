#ifndef NEURAL_NET_H_
#define NEURAL_NET_H_

#include <cmath>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <iomanip>
#include <iostream>

class NeuralNet
{
  public:
    /// @brief create a randomized neural net with the specified layers
    /// @param _layer_sizes the size of each layer, in 
    NeuralNet(std::vector<int> _layer_sizes);

    /// @brief import a neural net from file
    /// @param _file_name name of file
    NeuralNet(const std::string _file_name);

    std::vector<double> predict(const std::vector<double>& _inputs);

    /// @brief batch training
    double train(const std::vector<std::vector<double>>& _batch_inputs, 
                 const std::vector<std::vector<double>>& _batch_outputs, 
                 double _learning_rate = 0.01);

    /// @brief online training
    double train(const std::vector<double>& _inputs,
                 const std::vector<double>& _outputs,
                 double _learning_rate = 0.01,
                 double _regularization_rate = 0.05);

    bool exportToFile(const std::string _file_name) const;

    const std::vector<std::vector<std::vector<double>>>& weights() const
    {
      return m_weights;
    }

  private:
    std::vector<int> m_layer_sizes;

    /// activations[layer][neuron]
    std::vector<std::vector<double>> m_activations;
    
    /// sum before going through activation function
    /// sums[layer][neuron]
    /// The 0th layer doesn't have sum
    std::vector<std::vector<double>> m_sums;

    /// weights[starting_layer][starting_neuron][ending_neuron]
    std::vector<std::vector<std::vector<double>>> m_weights;

    /// biases[layer][neuron]
    /// The 0th layer doesn't have bias
    std::vector<std::vector<double>> m_biases;

    ////////////////////////////////////////////////////////////////////////////
    /// @name Back propagation variables
    /// @{

    /// The negative gradient (-dC/dz[layer][neuron]) for each sum 
    /// (before activation)
    std::vector<std::vector<double>> m_sum_corrections;

    std::vector<std::vector<std::vector<double>>> m_weight_correction_accumulations;

    std::vector<std::vector<double>> m_bias_correction_accumulations;

    /// @}
    ////////////////////////////////////////////////////////////////////////////

    void   feedFoward(const std::vector<double>& _inputs);

    void   backPropagate(const std::vector<double>& _correct_outputs);

    /// Activation function - sigmoid
    double sigmoid(double x);

    double sigmoid_derivation(double sigmoid_x);

    double L1_derivation(double x, double regularization_rate);

};

#endif // NEURAL_NET_H_