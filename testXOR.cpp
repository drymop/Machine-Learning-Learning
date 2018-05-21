#include <iomanip>
#include <iostream>

#include "NeuralNet.h"

using std::cout;
using std::endl;
using std::vector;

int main(int argc, char const *argv[])
{
  NeuralNet nn ({2, 3, 1});

  vector<vector<double>> X = 
  {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1},
  };
  vector<vector<double>> y = 
  {
    {0},
    {1},
    {1},
    {0},
  };

  for (int i = 0; i < 1000; i++)
  {
    nn.train(X, y, 1);
  }

  // test after train
  vector<vector<double>> X_test = 
  {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1},
  };

  for (int i = 0; i < X_test.size(); i++) {
    cout << "Predict " << X_test[i][0] << " " << X_test[i][1] << " : ";
    cout << nn.predict(X_test[i])[0] << endl;
  }


  return 0;
}