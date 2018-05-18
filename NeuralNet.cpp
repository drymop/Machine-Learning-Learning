#include "NeuralNet.h"


NeuralNet::
NeuralNet(std::vector<int> _layer_sizes)
{

}


std::vector<double>
NeuralNet::
predict(const std::vector<double>& _inputs)
{
  // input is the activation of the 0th layer
  for (int neuron = 0; neuron < m_layer_sizes[0]; neuron++)
    m_activations[0][neuron] = _inputs[neuron];

  // feed foward process
  // for each ending layer
  for (int layer = 1; layer < m_layer_sizes.size(); layer++)
  {
    // for each ending neuron
    for (int end_neuron = 0; end_neuron < m_layer_sizes[layer]; end_neuron++)
    {
      // calculate the sum
      m_sums[layer][end_neuron] = m_biases[layer][end_neuron];
      for (int start_neuron = 0; start_neuron < m_layer_sizes[layer-1]; start_neuron++)
      {
        m_sums[layer][end_neuron] += 
            m_activations[layer-1][start_neuron] * 
            m_weights[layer-1][start_neuron][end_neuron];
      }
      // calculate activation
      m_activations[layer][end_neuron] = sigmoid(m_sums[layer][end_neuron]);
    }
  }

  return m_activations[m_layer_sizes.size()-1];
}


double
NeuralNet::
sigmoid(double x)
{
  double e_power_x = exp(x);
  return e_power_x / (e_power_x + 1);
}