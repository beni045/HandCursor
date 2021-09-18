#include <math.h>


double sigmoidfunc(double x){
    double sigmoid = 1.0 / (1.0 + exp(-x));
    return sigmoid;
}

