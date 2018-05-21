#include <algorithm>
#include <iomanip>
#include <iostream>

#include <ctime>
#include "MNISTParser.h"
#include "NeuralNet.h"

using namespace std;

int main(int argc, char const *argv[])
{
  // seed random number
  srand (time(0));

  MNISTParser parser;
  struct MNISTData data  = parser();

  //NeuralNet net (vector<int>{28 * 28, 28, 10});
  NeuralNet net("MNIST_step_0_01.model");

  int num_epoch = 11;
  int batch_size = 128;
  int iter_per_epoch = data.train_inputs.size() / batch_size + 1;

  // use this to shuffle data
  vector<int> indexes;
  indexes.reserve(data.train_inputs.size());
  for (int i = 0; i < data.train_inputs.size(); i++)
    indexes.push_back(i);


  for (int epoch = 0; epoch < num_epoch; epoch++)
  {
    cout << "--------------------------------------------------------------------" << endl;
    cout << "EPOCH " << epoch << endl;

    random_shuffle( indexes.begin(), indexes.end() );
    for (int ii = 0; ii < indexes.size(); ii += 10000)
      cout << " " << indexes[ii];
    cout << endl;
    
    for (int iter = 0; iter < iter_per_epoch; iter++)
    {
      vector<vector<double>> batch_ins;
      vector<vector<double>> batch_outs;
      batch_ins.reserve(batch_size);
      batch_outs.reserve(batch_size);
      
      int start_index = iter * batch_size;
      int end_index = start_index + batch_size;
      if (end_index > data.train_inputs.size()) 
        end_index = data.train_inputs.size();
      for (int i = start_index; i < end_index; i++)
      {
        batch_ins .push_back(data.train_inputs[i]);
        batch_outs.push_back(data.train_outputs[i]);
      }

      double cost = net.train(batch_ins, batch_outs, 0.001);
      if (iter % 50 == 0)
      {
        cout << "Iter " << iter << ": " << cost << endl;
        net.exportToFile("MNIST.model");
      }
    }

  }


  return 0;
}