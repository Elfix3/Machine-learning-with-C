#include "stdio.h"
#include "stdlib.h"
#include "mem.h"

#include "time.h"

//Data Set size
#define SQUARE(X)   ((X)*(X))
#define DATA_SIZE 100

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
        return;
    }
}




//forward propagation
double model_predict(Model *m, double input){
    return m->w*input + m->b;
}

//Loss function MSE
double loss(Model *m, double *x, double *y){
    double loss = 0;
    for(int i = 0; i<DATA_SIZE; i++){
        loss += SQUARE((model_predict(m,x[i]) - y[i]));
    }
    return loss/(2*DATA_SIZE);
}


//gradients computation
void grad_update(Model *m, double lr){
    double dW = 0;
    double dB = 0;

    for(int i = 0; i<DATA_SIZE;i++){    
        dW += x[i]*(model_predict(m,x[i]) - y[i]);
        dB += (model_predict(m,x[i]) - y[i]);
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
        printf("Model : %f*x + %f\tLoss : %f\n",m->w,m->b,loss(m,x,y));
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
    


    model_info(model);

    for(int epoch = 0; epoch <10000; epoch++){
        grad_update(model,0.01);
    }
    model_info(model);

    free(model);
    free(x);
    free(y);
    return 0;
}


