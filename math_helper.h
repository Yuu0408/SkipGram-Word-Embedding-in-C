#ifndef MATH_HELPER_H
#define MATH_HELPER_H

double sigmoid(double x);
void softmax(double *scores, int vocab_size, double *probs);

#endif
