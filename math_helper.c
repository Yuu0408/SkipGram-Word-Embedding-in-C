#include "math_helper.h"
#include <math.h>

// Function to calculate sigmoid
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Function to calculate softmax
void softmax(double *scores, int vocab_size, double *probs) {
    double sum = 0.0;
    int i;
    for (i = 0; i < vocab_size; i++) {
        probs[i] = exp(scores[i]);
        sum += probs[i];
    }
    for (i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
    }
}

