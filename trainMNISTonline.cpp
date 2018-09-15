#include <algorithm>
#include <iomanip>
#include <iostream>

#include <ctime>
#include "MNISTParser.h"
#include "NeuralNet.h"

using namespace std;

int main(int argc, char const *argv[])
{
  string load_file ("MNIST.model");
  string save_file ("MNIST.model");

  // seed random number
  srand (time(0));

  MNISTParser parser;
  struct MNISTData data  = parser();

  //NeuralNet net (vector<int>{28 * 28, 28, 10});
  NeuralNet net(load_file);

  int num_epoch = 11;

  for (int epoch = 0; epoch < num_epoch; epoch++)
  {
    cout << "--------------------------------------------------------------------" << endl;
    cout << "EPOCH " << epoch << endl;

    double avg_cost = 0;
    double prev_weight = 0;
    for (int i = 0; i < data.train_inputs.size(); i++)
    {
      double cost = net.train(data.train_inputs[i], data.train_outputs[i], 0.01, 0.0001);
      avg_cost += cost;
      
      if (i % 10000 == 0)
      {
        if (i)
          avg_cost /= 10000.0;
        cout << "Iter " << i << ": " << avg_cost << endl;
        avg_cost = 0;
        net.exportToFile(save_file);
      }
    }

  }


  return 0;
}