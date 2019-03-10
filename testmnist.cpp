#include <stdio.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include "ffnn_core.h"
#include <time.h>
//just for viewing MNIST images
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
//using namespace cv;

FILE* readMNIST_Imagefile_header(char* filename, int& rows, int &cols, int& images)
{
	FILE* fp = fopen(filename,"rb");
	int magicnum;
	fread(&magicnum,sizeof(int),1,fp);
	fread(&images,sizeof(int),1,fp);
	fread(&rows,sizeof(int),1,fp);
	fread(&cols,sizeof(int),1,fp);
	magicnum = ntohl(magicnum);
	images = ntohl(images);
	rows = ntohl(rows);
	cols = ntohl(cols);
	printf("magicnum:%d,images:%d,rows:%d,cols:%d\n",magicnum,images,rows,cols);
	//Mat drawing = Mat::zeros( cvSize(rows,cols), CV_8UC1 );
	//for(int i=0;i<images;i++)
	//{
	//	fread(drawing.data,1,rows*cols,fp);
	//	imshow( "Images", drawing );
	//	char ch = waitKey(1000);
	//	if(ch == 27)
	//		break;
	//}
	return fp;
}

FILE* readMNIST_Labelfile_header(char*filename)
{
	FILE* fp = fopen(filename,"rb");
	int magicnum;
	int labels;
	fread(&magicnum,sizeof(int),1,fp);
	fread(&labels,sizeof(int),1,fp);
	magicnum = ntohl(magicnum);
	labels = ntohl(labels);
	printf("magicnum:%d,labels:%d\n",magicnum,labels);
	return fp;
}

void readbatchdata(int indexarr[], int index, int batch_size, int rows, int cols, FILE* fp, double** input)
{
	int inputsize = (rows*cols);
	int fileoffset = 0;
	for(int i=0;i<batch_size;i++)
	{
		fileoffset = 16+(indexarr[index+i]*rows*cols);
		fseek(fp,fileoffset,SEEK_SET);
		for(int j=0;j<rows;j++)
		{
			for(int k=0;k<cols;k++)
                	{
                        	unsigned char val;
                        	fread(&val,1,1,fp);
                        	input[i][j*rows+k] = (1.0*val)/255.0;
                	}
		}
	}
}

void readbatchlabel(int indexarr[], int index, int batch_size, int size, FILE*fp, double** output)
{
	int fileoffset = 0;
	for(int i=0;i<batch_size;i++)
	{
		fileoffset = 8+indexarr[index+i];
		fseek(fp,fileoffset,SEEK_SET);
		unsigned char label;
		fread(&label,1,1,fp);
		for(int j=0;j<size;j++)
		{
			if(j==((int)label))
				output[i][j] = 1.0;
			else
				output[i][j] = 0.0;
		}
	}
}

void readValidationdata(int index, double* input, int rows, int cols, FILE* fp)
{
	int inputsize = rows*cols;
	int fileoffset = 16+ (index*(rows*cols));
        fseek(fp,fileoffset, SEEK_SET);
        for(int j=0;j<rows;j++)
        {
		for(int k=0;k<cols;k++)
		{
			unsigned char val;
                	fread(&val,1,1,fp);
			input[j*rows+k] = (1.0*val)/255.0;
		} 
        }
}

void readValidationlabel(int index, double* output, int size, FILE*fp)
{
	int fileoffset = 8+index;
        fseek(fp, fileoffset,SEEK_SET);
        unsigned char label;
        fread(&label,1,1,fp);
        for(int j=0;j<size;j++)
        {
        	if(j==((int)label))
        		output[j] = 1.0;
        	else
        		output[j] = 0.0;
        }
	//printf("validation label\n");
	//for(int j=0;j<size;j++)
	//{
	//	printf("%lf ",output[j]);
	//}
	//printf("\nvalidation label end \n");
}


void shuffle(int arr[], int count)
{
	for(int i=0;i<count;i++)
	{
		arr[i] = i;
	}
	for(int i=0;i<count;i++)
	{
		int swapindex = rand()%count;
		int temp = arr[i];
		arr[i] = arr[swapindex];
		arr[swapindex] = temp;
	}
}

int main(int argc, char* argv[])
{
    setbuf(stdout, NULL);
	if(argc < 5)
	{
		printf("usage: ./a.out <training images> <train labels> <testing images> <test labels> \n");
		return 0;
	}
	int tr_rows, tr_cols, ti_rows, ti_cols, tr_images, ti_images;
	FILE* tr = readMNIST_Imagefile_header(argv[1],tr_rows, tr_cols,tr_images);
	FILE* trl = readMNIST_Labelfile_header(argv[2]);
	FILE* ti = readMNIST_Imagefile_header(argv[3],ti_rows, ti_cols,ti_images);
	FILE* til = readMNIST_Labelfile_header(argv[4]);
	int validation_images = ti_images;
	int training_images = tr_images - validation_images;
	ffnn_core c;
	int a[3];
	a[0] = tr_rows*tr_cols;
	a[1] = 30;
	a[2] = 10;
	int batch_size = 10;
	double** input, **output;
	input = (double**)malloc(batch_size*sizeof(double*));
	output = (double**)malloc(batch_size*sizeof(double*));
	for(int i=0;i<batch_size;i++)
	{
		input[i] = (double*)malloc(a[0]*sizeof(double));
		output[i] = (double*)malloc(10*sizeof(double));
	}
	c.init(a,3,3.0,NULL);
	int epohcount = 0;
	int indexarr[60000];
	int correctvalidation = 0;
	int correcttest = 0;
	while( epohcount < 50)
	{
		shuffle(indexarr,tr_images);
		for(int i=0;i<tr_images;)
		{
			readbatchdata(indexarr,i,batch_size, tr_rows, tr_cols, tr, input);
			readbatchlabel(indexarr,i,batch_size, 10, trl, output);
			c.trainbatch(input,output,a[0],10,batch_size);
			i+= batch_size;
		}
		correctvalidation = 0;
                for(int i=training_images;i<tr_images;i++)
                {
                        readValidationdata(i,input[0],ti_rows, ti_cols, tr);
                        readValidationlabel(i,output[0],10,trl);
                        double act_output[10];
                        c.feedforward(input[0],act_output,a[0],10);
                        //c.printoutput();
                        int expindex = 0,actindex = 0;
                        float max = act_output[0];
                        for( int k=0;k<10;k++)
                        {
                                if( output[0][k] == 1.0)
                                        expindex = k;
                                if(act_output[k] > max)
                                {
                                        max = act_output[k];
                                        actindex = k;
                                }
                        }
                        if(expindex == actindex)
                                correctvalidation++;
                }
		correcttest = 0;
		for(int i=0;i<ti_images;i++)
                {
			readValidationdata(i,input[0],ti_rows, ti_cols, ti);
                	readValidationlabel(i,output[0],10,til);
                	double act_output[10];
                	c.feedforward(input[0],act_output,a[0],10);
                	//c.printoutput();
                 	int expindex = 0,actindex = 0;
                	float max = act_output[0];
                	for( int k=0;k<10;k++)
                 	{
				if( output[0][k] == 1.0)
					expindex = k;
				if(act_output[k] > max)
				{
					max = act_output[k];
					actindex = k;
                                }
                        }
                        if(expindex == actindex)
                        	correcttest++;
		}
		epohcount++;
		printf("Epoch %d, Accuracy(Validation Data) : %d/%d  Accuracy(Test Data) : %d/%d \n",epohcount,
			correctvalidation,validation_images, correcttest, ti_images);

	}
	return 0;
}
