#ifndef MNIST_PARSER_H_
#define MNIST_PARSER_H_

#include <fstream>
#include <iostream>
#include <string>
#include <vector>


struct MNISTData {
  std::vector<std::vector<double>> train_inputs;
  std::vector<std::vector<double>> train_outputs;
  std::vector<std::vector<double>> test_inputs;
  std::vector<std::vector<double>> test_outputs;
};


class MNISTParser
{
  public:
    static const int NUM_TRAINS = 60000;
    static const int NUM_TESTS  = 10000;
    static const int IMG_SIZE   = 28 * 28;

    struct MNISTData operator() ();

  private:
    static const std::vector<double> PIXEL_CONVERSION_TABLE;

    static void readBytes(
        const std::string file_name, 
        std::vector<char>& buffer,
        int header_size, 
        int num_bytes);

    /// @brief Create a table that convert pixel value (0 - 255) to double
    /// value between 0 and 1
    static std::vector<double> initConversionTable();
};


#endif // MNIST_PARSER_H_