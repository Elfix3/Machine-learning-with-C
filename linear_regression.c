#include "stdio.h"
#include "stdlib.h"
#include "mem.h"

#include "time.h"

//Data Set size
#define DATA_SIZE 1000

//Model
struct {
    double w;
    double b;
} typedef Model;

//Handlers
static Model *model = NULL;
static double *x = NULL;
static double *y = NULL;

// Init handlers

void initModel(Model **m ,double w, double b){
    *m = malloc(sizeof(Model));
    (*m)->w = w;
    (*m)->b = b;
}

void init_data(double **v, size_t size){ //inits a vector randomly
    if(*v == NULL){
        *v = malloc(sizeof(double)*size);
        for(size_t i = 0; i< size; i++){
            (*v)[i] = ((float)rand() / (float)RAND_MAX) * 20.0f;
            
        }
        
    } else {
        printf("Already initalized\n");
        return; //already initalized
    }
}

void init_data_linear_dependance(double **v, size_t size, double *x, double w, double b, double error_range){ //inits data to simulate a linearity

    if(x == NULL){
        printf("Target data is not available\n");
        return;
    }
    if(*v == NULL){
        *v = malloc(sizeof(double)*size);
        for(size_t i = 0; i< size; i++){
            (*v)[i] = w*(x)[i]+ b + (((float)rand() / (float)RAND_MAX)-0.5)*error_range;
        }
        
    } else {
        printf("Already initalized\n");
        return; //already initalized
    }
}




//forward propagation
double model_output(Model *m, double input){
    return m->w*input + m->b;
}


//gradients computation

void grad_update(Model *m, double lr){
    double dW = 0;
    double dB = 0;

    for(int i = 0; i<DATA_SIZE;i++){    
        dW += x[i]*(model_output(m,x[i]) - y[i]);
        dB += (model_output(m,x[i]) - y[i]);
    }
    dW /= DATA_SIZE;
    dB /= DATA_SIZE;

    m->w -= lr*dW;
    m->b -= lr*dB;
}



//Print

void model_info(Model *m){
    if (m == NULL) {
        printf("Error, model unitialized\n");
    } else {
        printf("%f*x + %f\n",m->w,m->b);
    }
}

void print_data(double *v, size_t size){
    printf("[ ");
    for(size_t i = 0; i< size; i++){
        
        printf("%f ",*(v+i));
    }
    printf(" ]\n");
}


int main(){
    
    srand(time(NULL)); //random seed

    initModel(&model, 0.0f, 0.0f);
    init_data(&x,DATA_SIZE);
    init_data_linear_dependance(&y, DATA_SIZE, x, 1.3f, 4.0f, 0.3f);
    //print_data(x,DATA_SIZE);
    //print_data(y,DATA_SIZE);


    
    model_info(model);

    for(int epoch = 0; epoch <100000; epoch++){
        grad_update(model,0.001);
        
    }
    model_info(model);


    //grad_update()
    /* FILE *gp = popen("gnuplot -persistent", "w");

        fprintf(gp, "plot '-' with points\n");
        for(size_t i = 0; i < DATA_SIZE; i++){
            fprintf(gp, "%f %f\n", x[i], y[i]);
        }
        fprintf(gp, "e\n");
        pclose(gp); */


    free(model);
    free(x);
    free(y);
    return 0;
}


