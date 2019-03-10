#include"ffnn_core.h"
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

using namespace std;

double sigmoid(double z)
{
	return (1.0/(1.0+exp(-z)));
}

double sigmoid_prime(double z)
{
	return sigmoid(z)*(1.0-sigmoid(z));
}

ffnn_core::ffnn_core()
{
	ffnn_in=NULL;
	ffnn_out=NULL;
}

ffnn_core::~ffnn_core()
{
	deinit();
}

void ffnn_core::printffnn(char* weight_bias_file)
{
	FILE* fp = stdout;
	if( weight_bias_file != NULL )
	{
		//TODO
	}
	fprintf(fp,"*******FFNN core weight values********\n");
	Layers* ptr = ffnn_in;
	int count = 0;
	while(ptr != NULL)
	{
		fprintf(fp,"Layers %d\n", count+1);
		fprintf(fp,"Inputs: %d \nOutputs: %d \n", ptr->input_size, ptr->output_size);
		fprintf(fp,"Bias_Array:- \n");
		for(int i=0;i<ptr->output_size;i++)
		{
			fprintf(fp,"%lf ",ptr->bias_arr[i]);
		}
		fprintf(fp,"\n");
		fprintf(fp,"Weight_Array:- \n");
		for(int i=0;i<ptr->output_size;i++)
		{
			for(int j=0;j<ptr->input_size;j++)
			{
				fprintf(fp,"%lf ", ptr->weight_arr[i][j]);
			}
			fprintf(fp,"\n");
		}
		count++;
		ptr = ptr->next;
	}
	fprintf(fp,"************FFNN Data End*******************\n");
}

void ffnn_core::printoutput()
{
	FILE* fp = stdout;
	fprintf(fp,"*************CNN output value ***************\n");
	Layers* ptr = ffnn_out;
	if( ptr != NULL)
	{
		for(int i=0;i<ptr->output_size;i++)
			fprintf(fp,"%lf ",ptr->output_arr[i]);
		fprintf(fp,"\n");
	}
	fprintf(fp,"*************output value end ****************\n");
}

void ffnn_core::printinput()
{
	FILE* fp = stdout;
	fprintf(fp,"*************CNN input value ******************\n");
	Layers* ptr = ffnn_in;
	if(ptr != NULL)
	{
		for(int i=0;i<ptr->input_size;i++)
			fprintf(fp,"%lf ",ptr->input_arr[i]);
		fprintf(fp,"\n");
	}
	fprintf(fp,"*************input value end *******************\n");
}


double rand_normal(double mean, double stddev)
{//Box muller method
    static double n2 = 0.0;
    static int n2_cached = 0;
    if (!n2_cached)
    {
        double x, y, r;
        do
        {
            x = 2.0*rand()/RAND_MAX - 1;
            y = 2.0*rand()/RAND_MAX - 1;

            r = x*x + y*y;
        }
        while (r == 0.0 || r > 1.0);
        {
            double d = sqrt(-2.0*log(r)/r);
            double n1 = x*d;
            n2 = y*d;
            double result = n1*stddev + mean;
            n2_cached = 1;
            return result;
        }
    }
    else
    {
        n2_cached = 0;
        return n2*stddev + mean;
    }
}


void ffnn_core::random_generate()
{
	// generating from a uniform distribution
	Layers* ptr = ffnn_in;
	srand(time(NULL));
	while(ptr != NULL)
	{
		for(int i=0;i<ptr->output_size;i++)
		{
			ptr->bias_arr[i] = rand_normal(0.0,1.0);
			for(int j=0;j<ptr->input_size;j++)
			{
				ptr->weight_arr[i][j] = rand_normal(0.0,1.0);
			}
		}
		ptr = ptr->next;
	}
}

void ffnn_core::deinit()
{
	Layers* ptr = ffnn_in;
	while(ptr != NULL)
	{
		if(ptr->input_arr != NULL)
			free(ptr->input_arr);
		if(ptr->output_arr != NULL)
			free(ptr->output_arr);
		if(ptr->bias_arr != NULL)
			free(ptr->bias_arr);
		if(ptr->delta != NULL)
			free(ptr->delta);
		if(ptr->errors_bias != NULL)
			free(ptr->errors_bias);
		if(ptr->errors_weight != NULL)
		{
			for(int j=0;j<ptr->output_size;j++)
			{
				if(ptr->errors_weight[j] != NULL)
					free(ptr->errors_weight[j]);
			}
			free(ptr->errors_weight);
		}
		if(ptr->weight_arr != NULL)
                {
                        for(int j=0;j<ptr->output_size;j++)
                        {
                                if(ptr->weight_arr[j] != NULL)
                                        free(ptr->weight_arr[j]);
                        }
                        free(ptr->weight_arr);
                }
		Layers* qtr = ptr;
		ptr = ptr->next;
		free(qtr);
	}
	ffnn_in = NULL;
	ffnn_out = NULL;
}

int ffnn_core::init(int layers[], int size, double learn_rate, char* weight_bias_file)
{
	if( ffnn_in != NULL)
	{
		return -1;
	}
	learning_rate = learn_rate;
	Layers* ptr = ffnn_in;
	for(int i=1;i<size;i++)
	{
		Layers* temp = (Layers*)malloc(sizeof(Layers));
		if( temp == NULL)
			goto err_handle;
		temp->input_size = layers[i-1];
		temp->output_size = layers[i];
		temp->input_arr = (double*)malloc(temp->input_size*sizeof(double));
		temp->output_arr = (double*)malloc(temp->output_size*sizeof(double));
		temp->bias_arr = (double*)malloc(temp->output_size*sizeof(double));
		temp->delta = (double*)malloc(temp->output_size*sizeof(double));
		temp->errors_bias = (double*)malloc(temp->output_size*sizeof(double));

		temp->errors_weight = (double**)malloc(temp->output_size*sizeof(double*));
		temp->weight_arr = (double**)malloc(temp->output_size*sizeof(double*));
		if((temp->input_arr == NULL)||(temp->output_arr == NULL)||(temp->bias_arr == NULL)||(temp->delta == NULL)||
			(temp->errors_bias == NULL)||(temp->errors_weight == NULL)||(temp->weight_arr == NULL))
			goto err_handle;
		for(int j=0;j<temp->output_size;j++)
		{
			temp->errors_weight[j] = (double*)malloc(temp->input_size*sizeof(double));
			temp->weight_arr[j] = (double*)malloc(temp->input_size*sizeof(double));
		}
		for(int j=0;j<temp->output_size;j++)
		{
			if(temp->errors_weight[j] == NULL)
				goto err_handle;
			if(temp->weight_arr[j] == NULL)
				goto err_handle;
		}
		if(ptr == NULL)
		{
			ffnn_in = temp;
			temp->prev = NULL;
		}
		else
		{
			temp->next = NULL;
			temp->prev = ptr;
			ptr->next = temp;
		}
		ptr = temp;
		temp->next = NULL;
		if( i == size-1)
			ffnn_out = ptr;
	}

	if(weight_bias_file == NULL)
	{
		random_generate();
	}

	return 0;

err_handle:
	deinit();
	return -1;
}

int ffnn_core::feedforward(double* input_value, double* output_value, int input_size, int output_size)
{
	if( ffnn_in->input_size != input_size)
		return -1;
	if( ffnn_out->output_size != output_size)
		return -1;
	Layers* ptr = ffnn_in;
	for(int j=0;j<input_size;j++)
		ptr->input_arr[j] = input_value[j];
	//printinput();
	while(ptr != NULL)
	{
		for(int j=0;j<ptr->output_size;j++)
		{
			double sum = 0.0;
			for(int k=0;k<ptr->input_size;k++)
			{
				sum += (ptr->weight_arr[j][k]*ptr->input_arr[k]);
			}
			sum += ptr->bias_arr[j];
			ptr->output_arr[j] = sigmoid(sum);
		}
			
		Layers* qtr = ptr;
		ptr = ptr->next;

		if( ptr != NULL)
		{
			// copy the input to next layer
			for(int j=0;j<qtr->output_size;j++)
			{
				ptr->input_arr[j] = qtr->output_arr[j];
			}
		}
	}
	//printoutput();
	if( output_value != NULL)
	{
		// copy the final output to array
        	for(int j=0;j<output_size;j++)
        	{
               		output_value[j] = ffnn_out->output_arr[j];
       		}
	}
	return 0;
}

int ffnn_core::trainbatch(double** input_values, double** expected_outputs, int input_size, int output_size, int batch_size)
{
	if( ffnn_in->input_size != input_size)
		return -1;
	if( ffnn_out->output_size != output_size)
		return -1;

	// initialize all the errors in layers to be zero
	Layers* ptr = ffnn_in;
	while(ptr != NULL)
	{
		for(int i=0;i<ptr->output_size;i++)
		{
			ptr->errors_bias[i] = 0.0;
			for(int j=0;j<ptr->input_size;j++)
			{
				ptr->errors_weight[i][j] = 0.0;
			}
		}
		ptr = ptr->next;
	}
	for(int i=0;i<batch_size;i++)
	{
		feedforward(input_values[i],NULL,input_size,output_size);
		//printinput();
		//printoutput();
		for(int j=0;j<ffnn_out->output_size;j++)
		{
			// cross entropy error delta  = activation - output;
			ffnn_out->delta[j] = (ffnn_out->output_arr[j] - expected_outputs[i][j]);
			ffnn_out->errors_bias[j] += ffnn_out->delta[j];
			for(int k=0;k<ffnn_out->input_size;k++)
			{
				// error_weights(i,j)=x(i)*delta(j);
				ffnn_out->errors_weight[j][k] += (ffnn_out->delta[j]*ffnn_out->input_arr[k]);
			}
		}
	
		// perform backpropagation
		Layers* ptr = ffnn_out->prev;
		while(ptr != NULL)
		{
			// calculate the new delta by backpropagation
			for(int j=0;j<ptr->output_size;j++)
			{
				// ptr->output_size == ptr->next->input_size
				double delta = 0.0;
				double z = 0.0;
				for(int k=0;k<ptr->next->output_size;k++)
				{
					delta += ptr->next->delta[k]*ptr->next->weight_arr[k][j];
				}
				for(int k=0;k<ptr->input_size;k++)
				{
					z += (ptr->weight_arr[j][k]*ptr->input_arr[k]);
				}
				z += ptr->bias_arr[j];
				// back propagation core equation given by
				// delta_l (j) = Sum_k(delta_l+1(k)*weight(j,k)) * sigmoid_prime( Z(j) );
				ptr->delta[j] = (delta*sigmoid_prime(z));
				ptr->errors_bias[j] += ptr->delta[j];
				for(int k=0;k<ptr->input_size;k++)
				{
					ptr->errors_weight[j][k] += (ptr->delta[j]*ptr->input_arr[k]);
				}
			}
			ptr = ptr->prev;
		}
	}
	// update weights as per learning from this batch training set
	ptr = ffnn_in;
	double multiplier = learning_rate/(1.0*batch_size);
	while(ptr != NULL)
	{
		for(int i=0;i<ptr->output_size;i++)
		{
			ptr->bias_arr[i] = ptr->bias_arr[i] - multiplier*ptr->errors_bias[i];
			for(int j=0;j<ptr->input_size;j++)
			{
				ptr->weight_arr[i][j] = ptr->weight_arr[i][j] - multiplier*ptr->errors_weight[i][j];
			}
		}
		ptr = ptr->next;
	}
	//printffnn(NULL);
    return 0;
}

int ffnn_test()
{
	ffnn_core c;
	int a[] = {784,30,1};
	c.init(a,3,10.0,NULL);
	//c.printffnn(NULL);
	double **input;
	double **output;
	input = (double**)malloc(10000*sizeof(double*));
	output = (double**)malloc(10000*sizeof(double*));
	for(int i=0;i<10000;i++)
	{
		input[i] = (double*)malloc(784*sizeof(double));
		output[i] = (double*)malloc(1*sizeof(double));
	}
	int correctcount = 0;
	while(correctcount  < 10000 )
	{
		for(int i=0;i<1000;i++)
		{
			double sum = 0;
			for(int j=0;j<784;j++)
			{
				input[i][j] = (1.0*(rand()%200 -100))/100.0;
				sum += input[i][j];
			}
			if(sum >= 0.0)
				output[i][0] = 1.0;
			else
				output[i][0] = 0.0;
		}
		c.trainbatch((double**)input,(double**)output,784,1,1000);
		// generating test data
		correctcount = 0;
		for(int i=0;i<10000;i++)
		{
			double sum = 0.0;
			for(int j=0;j<784;j++)
			{
				input[i][j] = (1.0*(rand()%200 - 100))/100.0;
				sum += input[i][j];
			}
			c.feedforward(input[i],output[i],784,1);
			if((sum >= 0.0)&&(output[i][0] > 0.9))
				correctcount++;
			else
			if((sum < 0.0)&&(output[i][0] < 0.1))
				correctcount++;
		}
		fprintf(stdout,"Accuracy:- {%d/%d}\n",correctcount,10000);
	}

	c.printffnn(NULL);

	for(int i=0;i<10000;i++)
	{
		free(input[i]);
		free(output[i]);
	}
	free(input);
	free(output);
	return 0;
}
