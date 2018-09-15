#include "NeuralNet.h"

using std::vector;
using std::cout; using std::endl;

NeuralNet::
NeuralNet(vector<int> _layer_sizes)
    : m_layer_sizes ( _layer_sizes )
{
  //---------------
  // reserve space for sums and activations

  m_sums.reserve(m_layer_sizes.size());  
  m_sums.emplace_back();
  for (int i = 1; i < m_layer_sizes.size(); i++)
    m_sums.emplace_back( m_layer_sizes[i] );

  m_activations.reserve(m_layer_sizes.size());  
  for (int i = 0; i < m_layer_sizes.size(); i++)
    m_activations.emplace_back( m_layer_sizes[i] );
  
  //---------------
  // random intialize weights and biases
  
  // using a gaussian normal distribution of mean = 0, stddev = sqrt(2/fan-in)
  std::mt19937 mt(std::rand());
  std::normal_distribution<double> dist(0, 1);

  // weights
  m_weights.reserve(m_layer_sizes.size() - 1);
  for (int i = 0; i < m_layer_sizes.size() - 1; i++)
  {
    m_weights.emplace_back(
        m_layer_sizes[i], vector<double>(m_layer_sizes[i+1]) );
    // std dev for random weight of that layer
    double stddev = std::sqrt(2.0/m_layer_sizes[i]);

    for (int j = 0; j < m_layer_sizes[i]; j++)
      for (int k = 0; k < m_layer_sizes[i+1]; k++)
        m_weights[i][j][k] = dist(mt) * stddev;
  }

  // biases
  m_biases.reserve(m_layer_sizes.size());
  m_biases.emplace_back(); // 0th layer does not have bias
  for (int i = 1; i < m_layer_sizes.size(); i++)
  {
    m_biases.emplace_back( m_layer_sizes[i] );
    for (int j = 0; j < m_layer_sizes[i]; j++)
      m_biases[i][j] = dist(mt) * std::sqrt(2.0/m_layer_sizes[i-1]);
  }

  //---------------
  // reserve space for sum, weight and bias corrections
  
  // weights
  m_weight_correction_accumulations.reserve(m_layer_sizes.size() - 1);
  for (int i = 0; i < m_layer_sizes.size() - 1; i++)
    m_weight_correction_accumulations.emplace_back( 
        m_layer_sizes[i], vector<double>(m_layer_sizes[i+1]) );

  // biases
  m_bias_correction_accumulations.reserve(m_layer_sizes.size());
  m_bias_correction_accumulations.emplace_back(); // 0th layer doesn't have bias
  for (int i = 1; i < m_layer_sizes.size(); i++)
    m_bias_correction_accumulations.emplace_back(m_layer_sizes[i]);

  // sums
  m_sum_corrections.reserve(m_layer_sizes.size());
  m_sum_corrections.emplace_back(); // 0th layer doesn't have sum
  for (int i = 1; i < m_layer_sizes.size(); i++)
    m_sum_corrections.emplace_back(m_layer_sizes[i]);
}


NeuralNet::
NeuralNet(const std::string _file_name)
{
  std::ifstream ifs(_file_name);

  //---------------
  // import layer sizes
  int num_layers;
  ifs >> num_layers;

  m_layer_sizes.reserve(num_layers);
  for (int i = 0; i < num_layers; i++)
  {
    int layer_size;
    ifs >> layer_size;
    m_layer_sizes.push_back(layer_size);
  }

  //---------------
  // reserve space for sums and activations

  m_sums.reserve(m_layer_sizes.size());  
  m_sums.emplace_back();
  for (int i = 1; i < m_layer_sizes.size(); i++)
    m_sums.emplace_back( m_layer_sizes[i] );

  m_activations.reserve(m_layer_sizes.size());  
  for (int i = 0; i < m_layer_sizes.size(); i++)
    m_activations.emplace_back( m_layer_sizes[i] );
  
  //---------------
  // import weights and biases
  
  // weights
  m_weights.reserve(m_layer_sizes.size() - 1);
  for (int i = 0; i < m_layer_sizes.size() - 1; i++)
  {
    m_weights.emplace_back();
    m_weights[i].reserve(m_layer_sizes[i]);
    for (int j = 0; j < m_layer_sizes[i]; j++)
    {
      m_weights[i].emplace_back();
      m_weights[i][j].reserve(m_layer_sizes[i+1]);
      for (int k = 0; k < m_layer_sizes[i+1]; k++)
      {
        double weight;
        ifs >> weight;
        m_weights[i][j].push_back(weight);
      }
    }
  }

  // biases
  m_biases.reserve(m_layer_sizes.size());
  m_biases.emplace_back(); // 0th layer does not have bias
  for (int i = 1; i < m_layer_sizes.size(); i++)
  {
    m_biases.emplace_back();
    m_biases[i].reserve(m_layer_sizes[i]);
    for (int j = 0; j < m_layer_sizes[i]; j++)
    {
      double bias;
      ifs >> bias;
      m_biases[i].push_back(bias);
    }
  }

  //---------------
  // reserve space for sum, weight and bias corrections
  
  // weights
  m_weight_correction_accumulations.reserve(m_layer_sizes.size() - 1);
  for (int i = 0; i < m_layer_sizes.size() - 1; i++)
    m_weight_correction_accumulations.emplace_back( 
        m_layer_sizes[i], vector<double>(m_layer_sizes[i+1]) );

  // biases
  m_bias_correction_accumulations.reserve(m_layer_sizes.size());
  m_bias_correction_accumulations.emplace_back(); // 0th layer doesn't have bias
  for (int i = 1; i < m_layer_sizes.size(); i++)
    m_bias_correction_accumulations.emplace_back(m_layer_sizes[i]);

  // sums
  m_sum_corrections.reserve(m_layer_sizes.size());
  m_sum_corrections.emplace_back(); // 0th layer doesn't have sum
  for (int i = 1; i < m_layer_sizes.size(); i++)
    m_sum_corrections.emplace_back(m_layer_sizes[i]);
}


vector<double>
NeuralNet::
predict(const vector<double>& _inputs)
{
  feedFoward(_inputs);
  return m_activations[m_layer_sizes.size()-1];
}


double 
NeuralNet::
train(const vector<vector<double>>& _batch_inputs, 
      const vector<vector<double>>& _batch_outputs,
      double _learning_rate)
{
  // zero out the accumulations
  for (int i = 0; i < m_weight_correction_accumulations.size(); i++)
    for (int j = 0; j < m_weight_correction_accumulations[i].size(); j++)
      for (int k = 0; k < m_weight_correction_accumulations[i][j].size(); k++)
        m_weight_correction_accumulations[i][j][k] = 0;

  for (int i = 0; i < m_bias_correction_accumulations.size(); i++)
    for (int j = 0; j < m_bias_correction_accumulations[i].size(); j++)
      m_bias_correction_accumulations[i][j] = 0;

  // train
  int n_trainings = _batch_inputs.size();
  double avg_cost = 0;
  for (int i = 0; i < n_trainings; i++)
  {
    feedFoward   (_batch_inputs[i]);
    backPropagate(_batch_outputs[i]);
    // calculate cost
    for (int j = 0; j < m_layer_sizes.back(); j++)
    {
      double diff = m_activations.back()[j] - _batch_outputs[i][j];
      avg_cost += diff * diff;
    }
  }
  avg_cost = avg_cost/n_trainings;

  // perform correction
  double step = _learning_rate;// / n_trainings;
  for (int i = 0; i < m_weights.size(); i++)
    for (int j = 0; j < m_weights[i].size(); j++)
      for (int k = 0; k < m_weights[i][j].size(); k++)
        m_weights[i][j][k] += 
            m_weight_correction_accumulations[i][j][k] * step;

  for (int i = 0; i < m_biases.size(); i++)
    for (int j = 0; j < m_biases[i].size(); j++)
      m_biases[i][j] +=
          m_bias_correction_accumulations[i][j] * step;

  return avg_cost;
}


double 
NeuralNet::
train(const vector<double>& _inputs, 
      const vector<double>& _outputs,
      double _learning_rate,
      double _regularization_rate)
{
  double cost = 0;
  
  // zero out the accumulations, add regularization
  for (int i = 0; i < m_weight_correction_accumulations.size(); i++)
    for (int j = 0; j < m_weight_correction_accumulations[i].size(); j++)
      for (int k = 0; k < m_weight_correction_accumulations[i][j].size(); k++)
      {
        m_weight_correction_accumulations[i][j][k] = L1_derivation(m_weights[i][j][k], _regularization_rate);
        cost += std::abs(m_weights[i][j][k]);
      }

  for (int i = 0; i < m_bias_correction_accumulations.size(); i++)
    for (int j = 0; j < m_bias_correction_accumulations[i].size(); j++)
    {
      m_bias_correction_accumulations[i][j] = L1_derivation(m_biases[i][j], _regularization_rate);
      cost += std::abs(m_biases[i][j]);
    }

  cost *= _regularization_rate;

  // train
  feedFoward   (_inputs);
  backPropagate(_outputs);
  // calculate cost
  for (int i = 0; i < m_layer_sizes.back(); i++)
  {
    double diff = m_activations.back()[i] - _outputs[i];
    cost += diff * diff;
  }

  // perform correction
  double step = _learning_rate;
  for (int i = 0; i < m_weights.size(); i++)
    for (int j = 0; j < m_weights[i].size(); j++)
      for (int k = 0; k < m_weights[i][j].size(); k++)
      {
        m_weights[i][j][k] += 
            m_weight_correction_accumulations[i][j][k] * step;
      }

  for (int i = 0; i < m_biases.size(); i++)
    for (int j = 0; j < m_biases[i].size(); j++)
      m_biases[i][j] +=
          m_bias_correction_accumulations[i][j] * step;

  return cost;
}


void
NeuralNet::
feedFoward(const vector<double>& _inputs)
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
      // calculate the sum into that neuron
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
}


void 
NeuralNet::
backPropagate(const vector<double>& _correct_outputs)
{
  //------------------------------

  // calculate -dC/dz of the last (output) layer
  int last_layer = m_layer_sizes.size() - 1;
  for (int neuron = 0; neuron < m_layer_sizes[last_layer]; neuron++)
    m_sum_corrections[last_layer][neuron] 
        = 2 * ( _correct_outputs[neuron] - m_activations[last_layer][neuron] )
          * sigmoid_derivation(m_activations[last_layer][neuron]);

  // calculate -dC/dz of all layers
  for (int layer = last_layer - 1; layer >= 1; layer--)
  {
    // calculate sum correction of each neuron in layer
    for (int neuron = 0; neuron < m_layer_sizes[layer]; neuron++)
    {
      double sum_correction = 0;
      // add the influence of each next neuron
      for (int next_neuron = 0; next_neuron < m_layer_sizes[layer+1]; next_neuron++)
      {
        sum_correction += 
            m_sum_corrections[layer+1][next_neuron] *
            m_weights[layer][neuron][next_neuron] *
            sigmoid_derivation(m_activations[layer][neuron]);
      }
      m_sum_corrections[layer][neuron] = sum_correction;
    }
  }

  //------------------------------
  // calculate -dC/dw, -dC/db and add that too the accumulation
  
  // bias: 
  // -dC/db[L, i] = -dC/dz[L, i]
  for (int layer = 1; layer <= last_layer; layer++)
    for (int neuron = 0; neuron < m_layer_sizes[layer]; neuron++)
      m_bias_correction_accumulations[layer][neuron] +=
          m_sum_corrections[layer][neuron];

  // weights
  // -dC/dw[L, i->j] = -dC/ds[L+1, j] * a[L, i]
  for (int layer = 0; layer < last_layer; layer++)
    for (int neuron = 0; neuron < m_layer_sizes[layer]; neuron++)
      for (int next_neuron = 0; next_neuron < m_layer_sizes[layer+1]; next_neuron++)
        m_weight_correction_accumulations[layer][neuron][next_neuron] +=
            m_sum_corrections[layer+1][next_neuron] * m_activations[layer][neuron];
}


double
NeuralNet::
sigmoid(double x)
{
  double e_power_x = std::exp(x);
  return e_power_x / (e_power_x + 1);
}


double
NeuralNet::
sigmoid_derivation(double sigmoid_x)
{
  return sigmoid_x * (1 - sigmoid_x);
}


double
NeuralNet::
leakyRELU(double x)
{
  return (x > 0)? x : (0.001 * x);
}


double leakyRELU_derivation(double x)
{
  return (x > 0)? 1 : 0.001;
}



bool
NeuralNet::
exportToFile(const std::string _file_name) const
{
  std::ofstream ofs(_file_name);
  if (!ofs.is_open())
    return false;

  static std::string separator = " ";
  ofs.precision(17);
  
  // export layer sizes
  ofs << m_layer_sizes.size() << separator;
  for (auto size : m_layer_sizes)
    ofs << size << separator;

  // export weights
  for (auto& i : m_weights)
    for (auto& j : i)
      for (auto& k : j)
        ofs << k << separator;

  // export biases
  for (auto& i : m_biases)
    for (auto& j : i)
      ofs << j << separator;

  return true;
}

double 
NeuralNet::
L1_derivation(double x, double regularization_rate)
{
  if (x == 0.0)
    return 0;
  if (x < 0)
    return regularization_rate;
  return (-regularization_rate);
}
