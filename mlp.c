#ifndef READ_CSV_H
#define READ_CSV_H
#include <stdio.h>
#include <math.h>
#include <sys/timeb.h>
#include <sys/time.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#define MAX_LINE_SIZE 1048576 // 2^20
void read_csv(char*, int, int, double[259][12289]);
double x[259][12289];
int size = 208;
int nt = 4; // number of threads
double w1[12288][4];
double w2[4];
double b1[4];
double b2;
double s[4][49161]; //for each thread to store their results (gradient discent)
double st[49161];   //this collect the result of the threads
double learning_rate = 0.0075;
void initialize();
double sigmoid(double);
double tan_h(double);
void trainer(int);
void optimizer();
#endif
extern int
main(int argc, char *argv[])
{
	initialize();
	printf("initialization done : \n");
	struct timeval b, e;
	gettimeofday(&b,NULL);
	for(int i = 0; i < 1000; i++){
	optimizer();
        }
	gettimeofday(&e,NULL);
	printf("elapsed %li \n",((1000000*e.tv_sec+e.tv_usec)-(1000000*b.tv_sec+b.tv_usec)));
        printf("******************************************************** \n");
	double z1[4];
	double a1[4];
	double z2;
	double a2;
	double mean = 0;
	// test data set
	for(int i = 209; i < 259; i++){
		for(int j = 0; j < 4; j++){
			z1[j] = 0;
		}
		for(int j = 0; j < 4; j++){
			for(int k = 0; k < 12288; k++){
				z1[j] += x[i][k]*w1[k][j];
			}
			a1[j] = tan_h(z1[j]);
			  
		}
		z2 = a1[0]*w2[0] + a1[1]*w2[1] + a1[2]*w2[2] + a1[3]*w2[3]+b2;
		a2 = sigmoid(z2);
		if(a2 < 0.5)
		a2 = 0;
		else
			a2 = 1;
		mean += abs(x[i][12288] - a2);
	}
	printf("%f \n", (100-100*(mean/50)));
}
void initialize(){
char* file = "myfile3.csv";
double row = 259;
double col = 12289;
//initializing training and test example from csv file
read_csv(file, row, col, x);
//normalizing the input
for(int i = 0; i < 12288; i++){
	for(int j = 0; j < 259; j++){
		x[j][i] = x[j][i]/255;
	}
} 
//randomly initializing parameters
for(int i = 0; i < 4; i++){
	for(int j = 0; j < 12288; j++){
		w1[j][i] = 0.001*((double)rand() / RAND_MAX);
	}
	w2[i] = 0.001*((double)rand() / RAND_MAX);
}
}
void trainer(int start){
	double z1[4];
	double a1[4];
	double z2;
	double a2;
	double dz1[4];
	double dz2;
	int index = start/(size/nt);
	for(int i = 0; i < 49161; i++){
		s[index][i] = 0;
	}
	for(int i = start; i < start+size/nt; i++){
		
		for(int j = 0; j < 4; j++){
			z1[j] = 0;
		}
		// hiden layer calculation
		for(int j = 0; j < 4; j++){
			for(int k = 0; k < 12288; k++){
				z1[j] += x[i][k]*w1[k][j];
			}
			a1[j] = tan_h(z1[j]); 
		}
		// output layer calculation
		z2 = a1[0]*w2[0] + a1[1]*w2[1] + a1[2]*w2[2] + a1[3]*w2[3]+b2;
		a2 = sigmoid(z2);
		// gradient descent calculation
		dz2 = a2 - x[i][12288];
		
		for(int j = 49156; j < 49160; j++){
			s[index][j] += a1[j%4]*dz2;
		}
		s[index][49160] += dz2;
		
		for(int j = 0; j < 4; j++){
			dz1[j] = (w2[j] * dz2) * (1-(a1[j]*a1[j]));
			for(int k = j*12288; k < (j+1)*12288; k++){
				s[index][k] += dz1[j] * x[i][k%12288]; 
			}
		}
		
		for(int j = 49152; j < 49156; j++){
			s[index][j] += dz1[j%4];
		}
		
	}
	
}
void optimizer(){
	pthread_t thread1,thread2,thread3,thread4;
	pthread_create(&thread1,NULL,(void *) trainer,(void *) 0);
	pthread_create(&thread2,NULL,(void *) trainer,(void *) 52);
	pthread_create(&thread3,NULL,(void *) trainer,(void *) 104);
	pthread_create(&thread4,NULL,(void *) trainer,(void *) 156);
	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);
        pthread_join(thread3, NULL);
	pthread_join(thread4, NULL);
	for(int i = 0; i < 49161; i++){
		st[i] = 0;	
	}
	for(int i = 0; i < 49161; i++){
		for(int j = 0; j < 4; j++){
			st[i] += s[j][i];
		}
	}
	// updating parameters
	for(int i = 0; i < 4; i++){		
		for(int j = i*12288; j < (i+1)*12288; j++){
			w1[j%12288][i] -= ((st[j]/size)  * learning_rate);
		}	
	}
	for(int i = 49156; i < 49160; i++){
		w2[i%4] -= ((st[i]/size) * learning_rate);
	}
	for(int i = 49152; i < 49156; i++){
		b1[i%4] -= ((st[i]/size) * learning_rate);
	}
	b2 -= ((st[49160]/size) * learning_rate);
}
double sigmoid(double input){
	double output;
	output = 1.0 / (1.0 + exp(-input));
	return output;
}
double tan_h(double input){
	double output;
	output = tanh(input);
	return output;
}
void read_csv(char* filename, int rows, int cols, double data[259][12289]) {
    // Open file and perform sanity check
    FILE* fp = fopen(filename, "r");
    if (NULL == fp) {
        printf("Error opening %s file. Make sure you mentioned the file path correctly\n", filename);
        exit(0);
    }

    // Create memory to read a line/row from the file
    char* line = (char*)malloc(MAX_LINE_SIZE * sizeof(char));

    // Read the file line by line and save it in the matrix 'data'
    int i, j;
    for (i = 0; fgets(line, MAX_LINE_SIZE, fp) && i < rows; i++) {
        char* tok = strtok(line, ",");
        for (j = 0; tok && *tok; j++) {
            data[i][j] = atof(tok);
            tok = strtok(NULL, ",\n");
        }
    }

    // Free the allocated memory in Heap for line
    free(line);

    // Close the file
    fclose(fp);
}
