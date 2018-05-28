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

  NeuralNet net (vector<int>{28 * 28, 28, 10});
  //NeuralNet net(load_file);

  int num_epoch = 11;

  for (int epoch = 0; epoch < num_epoch; epoch++)
  {
    cout << "--------------------------------------------------------------------" << endl;
    cout << "EPOCH " << epoch << endl;

    double avg_cost = 0;
    double prev_weight = 0;
    for (int i = 0; i < data.train_inputs.size(); i++)
    {
      double cost = net.train(data.train_inputs[i], data.train_outputs[i], 0.1, 0.05);
      if (std::isnan(cost))
      {
        cout << "IS NAN " << i << endl;
        cout << "avg_cost " << avg_cost << endl;
        cout << "prev_weight" << prev_weight << endl;
      }
      avg_cost += cost;
      prev_weight = net.weights()[0][0][0];
      
      if (i % 1000 == 0)
      {
        avg_cost /= 1000;
        cout << "Iter " << i << ": " << avg_cost <<" "<< net.weights()[0][0][0] << endl;
        avg_cost = 0;
        net.exportToFile(save_file);
      }
    }

  }


  return 0;
}