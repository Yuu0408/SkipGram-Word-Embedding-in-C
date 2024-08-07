#ifndef TRAIN_H
#define TRAIN_H

#include "processing.h"

void train_model(char **tokens, int token_count, int window_size, int embedding_dim, double initial_learning_rate, double decay_rate, int decay_steps, int epochs, int batch_size, VocabItem *vocab, int vocab_size, int exist_model, char *model_filename, int x);
void forward_pass_batch(int *input_words, int batch_size, double **input_weights, double **output_weights, double **hidden_layer, double **output_layer, int vocab_size, int embedding_dim);
void backward_pass_batch(int *input_words, int *target_words, int batch_size, double **hidden_layer, double **output_probs, double **input_weights, double **output_weights, int vocab_size, int embedding_dim, double learning_rate);
void initialize_weights(int vocab_size, int embedding_dim, double **input_weights, double **output_weights);
void load_model(const char *filename, double **input_weights, double **output_weights, int vocab_size, int embedding_dim);
void save_model(const char *filename, double **input_weights, double **output_weights, int vocab_size, int embedding_dim);
void generate_training_data(int *token_indices, int token_count, int window_size, int **train_input_words, int **train_target_words, int *train_pair_count, int **valid_input_words, int **valid_target_words, int *valid_pair_count);
double learning_rate_schedule(double initial_learning_rate, int step, double decay_rate, int decay_steps);
void evaluate_model(double **input_weights, double **output_weights, int *input_words, int *target_words, int pair_count, int vocab_size, int embedding_dim, int batch_size, double *top_accuracy, int x);

#endif
