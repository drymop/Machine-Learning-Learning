#include "MNISTParser.h"

using std::vector;


const vector<double> MNISTParser::PIXEL_CONVERSION_TABLE = initConversionTable();

struct MNISTData
MNISTParser::
operator() ()
{
  struct MNISTData data;
  data.train_inputs  = vector<vector<double>> (NUM_TRAINS, vector<double>(IMG_SIZE));
  data.train_outputs = vector<vector<double>> (NUM_TRAINS, vector<double>(10, 0));
  data.test_inputs   = vector<vector<double>> (NUM_TESTS , vector<double>(IMG_SIZE));
  data.test_outputs  = vector<vector<double>> (NUM_TESTS , vector<double>(10, 0));

  vector<char> buffer;

  // read the training inputs
  readBytes("MNIST_database/train-images.idx3-ubyte", buffer, 16, NUM_TRAINS * IMG_SIZE);  
  for (int i = 0; i < NUM_TRAINS; i++)
  {
    for (int j = 0; j < IMG_SIZE; j++) {
      data.train_inputs[i][j] = PIXEL_CONVERSION_TABLE[ (unsigned char)buffer[i*IMG_SIZE+j] ];
    }
  }
  // read the test inputs
  readBytes("MNIST_database/t10k-images.idx3-ubyte", buffer, 16, NUM_TESTS * IMG_SIZE);  
  for (int i = 0; i < NUM_TESTS; i++)
  {
    for (int j = 0; j < IMG_SIZE; j++) {
      data.test_inputs[i][j] = PIXEL_CONVERSION_TABLE[ (unsigned char)buffer[i*IMG_SIZE+j] ];
    }
  }
  // read the training labels
  readBytes("MNIST_database/train-labels.idx1-ubyte", buffer, 8, NUM_TRAINS);
  for (int i = 0; i < NUM_TRAINS; i++)
    data.train_outputs[i][ buffer[i] ] = 1.0;      
  // read the test labels
  readBytes("MNIST_database/t10k-labels.idx1-ubyte", buffer, 8, NUM_TESTS);
  for (int i = 0; i < NUM_TESTS; i++)
    data.test_outputs[i][ buffer[i] ] = 1.0;    

  return data;
}


void
MNISTParser::
readBytes(const std::string file_name, 
          vector<char>& buffer,
          int header_size, 
          int num_bytes)
{

  buffer.resize(num_bytes);
  std::ifstream ifs(file_name, std::ios::binary);
  ifs.seekg(header_size); // skip past header
  ifs.read(&buffer[0], num_bytes);
}


vector<double> 
MNISTParser::
initConversionTable()
{
  vector<double> table (256);
  for (int i = 0; i < 256; i++) {
    table[i] = i / 256.0;
  }

  return table;
}