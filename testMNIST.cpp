#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <tuple>

#include "MNISTParser.h"
#include "NeuralNet.h"

using namespace std;

void compare(const vector<double>& guess, const vector<double>& key, 
    double& confidence, double& cost, bool& correct)
{
  int label = -1;
  for (int i = 0; i < key.size(); i++)
  {
    if (key[i] == 1.0)
    {
      label = i;
      break;
    }
  }
  confidence = guess[label];

  cost = 0;
  for (int i = 0; i < guess.size(); i++)
  {
    double diff = key[i] - guess[i];
    cost += diff * diff;
  }

  int guess_label = 10;
  double max_conf = -1;
  for (int i = 0; i < guess.size(); i++)
  {
    if (guess[i] > max_conf)
    {
      max_conf = guess[i];
      guess_label = i;
    }
  }
  correct = (guess_label == label);
}


int main(int argc, char const *argv[])
{
  MNISTParser parser;
  struct MNISTData data  = parser();

  NeuralNet net("MNIST.model");

  // use this to shuffle data
  // to choose 10000 random trained data
  vector<int> indexes;
  indexes.reserve(data.train_inputs.size());
  for (int i = 0; i < data.train_inputs.size(); i++)
    indexes.push_back(i);
  srand(time(0));
  random_shuffle( indexes.begin(), indexes.end() );

  int n_inputs = data.test_inputs.size();

  cout << "Trained data" << endl;
  int num_correct = 0;
  double avg_cost = 0;
  double max_cost = -1;
  double min_cost = 11;
  double avg_conf = 0;
  double max_conf = -0.1;
  double min_conf = 1.1;

  for (int i = 0; i < n_inputs; i++)
  {
    vector<double> guess = net.predict( data.train_inputs[indexes[i]] );
    double cost = 0, confidence = 0;
    bool correct;
    compare(guess, data.train_outputs[indexes[i]], confidence, cost, correct);
    num_correct += correct;
    avg_cost += cost;
    min_cost = min(min_cost, cost);
    max_cost = max(max_cost, cost);
    avg_conf += confidence;
    min_conf = min(min_conf, confidence);
    max_conf = max(max_conf, confidence);
  }

  cout << "Correct " << num_correct << endl;
  cout << "Cost" << endl;
  cout << "  avg: " << avg_cost/n_inputs << endl;
  cout << "  min: " << min_cost << endl;
  cout << "  max: " << max_cost << endl;
  cout << "Confidence" << endl;
  cout << "  avg: " << avg_conf/n_inputs << endl;
  cout << "  min: " << min_conf << endl;
  cout << "  max: " << max_conf << endl;


  cout << "\nNew data" << endl;
  num_correct = 0;
  avg_cost = 0;
  max_cost = -1;
  min_cost = 11;
  avg_conf = 0;
  max_conf = -0.1;
  min_conf = 1.1;

  for (int i = 0; i < n_inputs; i++)
  {
    vector<double> guess = net.predict( data.test_inputs[i] );
    double cost = 0, confidence = 0;
    bool correct;
    compare(guess, data.test_outputs[i], confidence, cost, correct);
    num_correct += correct;
    avg_cost += cost;
    min_cost = min(min_cost, cost);
    max_cost = max(max_cost, cost);
    avg_conf += confidence;
    min_conf = min(min_conf, confidence);
    max_conf = max(max_conf, confidence);
  }

  cout << "Correct " << num_correct << endl;
  cout << "Cost" << endl;
  cout << "  avg: " << avg_cost/n_inputs << endl;
  cout << "  min: " << min_cost << endl;
  cout << "  max: " << max_cost << endl;
  cout << "Confidence" << endl;
  cout << "  avg: " << avg_conf/n_inputs << endl;
  cout << "  min: " << min_conf << endl;
  cout << "  max: " << max_conf << endl;

  return 0;
}