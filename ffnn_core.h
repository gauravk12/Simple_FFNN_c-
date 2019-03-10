#ifndef FFNN_CORE_H
#define FFNN_CORE_H

struct Layers
{
        int input_size; // number of input values
        int output_size; // number of output values 
        double* input_arr; // input array equal to input size
        double* output_arr; // output array equal to output size
        double* bias_arr; // bias array
        double** weight_arr; // weight array
        double* delta; // ruuning back propagation error
        double* errors_bias; // for back propagation for bias
        double** errors_weight; // for back propagation for weight
        struct Layers* next;
        struct Layers* prev;
};

typedef struct Layers Layers;

class ffnn_core
{
private:
        Layers* ffnn_in;
        Layers* ffnn_out;
        double learning_rate;
        void deinit();
        void random_generate();
public:
        ffnn_core();
        ~ffnn_core();
        void printffnn(char* weight_bias_file);
        void printoutput();
        void printinput();
        int init(int layers[], int size, double learn_rate, char* weight_bias_file);
        int feedforward(double* input_value, double* output_value, int input_size, int output_size);
        int trainbatch(double** input_values, double** expected_output, int input_size, int output_size, int batch_size);
};

#endif
