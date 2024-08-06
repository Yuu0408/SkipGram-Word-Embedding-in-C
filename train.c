#include "train.h"
#include "use_model.h"
#include "math_helper.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <time.h>

// Function to initialize weights for both input and output layers of the model
void initialize_weights(int vocab_size, int embedding_dim, double **input_weights, double **output_weights) {
    int i, j;
    for (i = 0; i < vocab_size; i++) {
        for (j = 0; j < embedding_dim; j++) {
            input_weights[i][j] = ((double) rand() / RAND_MAX - 0.5) / embedding_dim;   // Initialize input weights randomly centered around zero
            output_weights[i][j] = 0.0; // Initialize output weights to zero
        }
    }
}

// Function to perform a forward pass of the neural network for a batch of input words
void forward_pass_batch(int *input_words, int batch_size, double **input_weights, double **output_weights, double **hidden_layer, double **output_layer, int vocab_size, int embedding_dim) {
    int i, j, b;
    for (b = 0; b < batch_size; b++) {
        int input_word = input_words[b];
        // Copy embedding vector from input weights to hidden layer for the current word
        for (i = 0; i < embedding_dim; i++) {
            hidden_layer[b][i] = input_weights[input_word][i];
        }
        // Initialize output layer to zero and compute output using hidden layer and output weights
        for (i = 0; i < vocab_size; i++) {
            output_layer[b][i] = 0.0;
            for (j = 0; j < embedding_dim; j++) {
                output_layer[b][i] += hidden_layer[b][j] * output_weights[i][j];
            }
        }
    }
}

// Function to perform a backward pass to update weights based on the output error
void backward_pass_batch(int *input_words, int *target_words, int batch_size, double **hidden_layer, double **output_probs, double **input_weights, double **output_weights, int vocab_size, int embedding_dim, double learning_rate) {
    int i, j, b;
    for (b = 0; b < batch_size; b++) {
        int input_word = input_words[b];
        int target_word = target_words[b];
        double *grad_output = (double *) malloc(vocab_size * sizeof(double));
        double *grad_hidden = (double *) malloc(embedding_dim * sizeof(double));
        if (grad_output == NULL || grad_hidden == NULL) {
            printf("Error allocating memory for gradients: %s\n", strerror(errno));
            return;
        }

        // Compute output layer gradients
        for (i = 0; i < vocab_size; i++) {
            grad_output[i] = output_probs[b][i];
        }
        grad_output[target_word] -= 1.0;

        // Initialize gradients for hidden layer
        for (j = 0; j < embedding_dim; j++) {
            grad_hidden[j] = 0.0;
        }

        // Update output weights and accumulate gradients for hidden layer
        for (i = 0; i < vocab_size; i++) {
            for (j = 0; j < embedding_dim; j++) {
                double grad = grad_output[i] * hidden_layer[b][j];
                output_weights[i][j] -= learning_rate * grad;
                grad_hidden[j] += grad_output[i] * output_weights[i][j];
            }
        }

        // Update input weights based on accumulated gradients
        for (j = 0; j < embedding_dim; j++) {
            input_weights[input_word][j] -= learning_rate * grad_hidden[j];
        }

        free(grad_output);
        free(grad_hidden);
    }
}

// Function to train the neural network model
void train_model(char **tokens, int token_count, int window_size, int embedding_dim, double initial_learning_rate, double decay_rate, int decay_steps, int epochs, int batch_size, VocabItem *vocab, int vocab_size, int exist_model, char *model_filename, int x) {
    int i, j, b, epoch;

    double **input_weights = (double **) malloc(vocab_size * sizeof(double *));
    double **output_weights = (double **) malloc(vocab_size * sizeof(double *));
    if (input_weights == NULL || output_weights == NULL) {
        printf("Error allocating memory for weights: %s\n", strerror(errno));
        return;
    }
    for (i = 0; i < vocab_size; i++) {
        input_weights[i] = (double *) malloc(embedding_dim * sizeof(double));
        output_weights[i] = (double *) malloc(embedding_dim * sizeof(double));
        if (input_weights[i] == NULL || output_weights[i] == NULL) {
            printf("Error allocating memory for weight matrices: %s\n", strerror(errno));
            return;
        }
    }

    // Load existing model to continue the training process. If no existing model, initialize and train a new model
    if (exist_model == 0) {
        initialize_weights(vocab_size, embedding_dim, input_weights, output_weights);
    } else {
        load_model(model_filename, input_weights, output_weights, vocab_size, embedding_dim);
    }

    double **hidden_layer = (double **) malloc(batch_size * sizeof(double *));
    double **output_layer = (double **) malloc(batch_size * sizeof(double *));
    double **output_probs = (double **) malloc(batch_size * sizeof(double *));
    for (i = 0; i < batch_size; i++) {
        hidden_layer[i] = (double *) malloc(embedding_dim * sizeof(double));
        output_layer[i] = (double *) malloc(vocab_size * sizeof(double));
        output_probs[i] = (double *) malloc(vocab_size * sizeof(double));
    }

    int *train_input_words, *train_target_words, *valid_input_words, *valid_target_words;
    int train_pair_count, valid_pair_count;

    // Convert tokens to their indices in the vocabulary
    int *token_indices = (int *) malloc(token_count * sizeof(int));
    for (i = 0; i < token_count; i++) {
        for (j = 0; j < vocab_size; j++) {
            if (strcmp(tokens[i], vocab[j].word) == 0) {
                token_indices[i] = j;
                break;
            }
        }
    }
    // printf("Token Indices: \n");
    // for (i = 0; i < token_count; i++) {
    //     printf("%d\n", token_indices[i]);
    // }

    // Generate training and validation data
    generate_training_data(token_indices, token_count, window_size, &train_input_words, &train_target_words, &train_pair_count, &valid_input_words, &valid_target_words, &valid_pair_count);

    printf("Starting training process with %d training pairs and %d validation pairs...\n", train_pair_count, valid_pair_count);
    int total_batches = (train_pair_count + batch_size - 1) / batch_size; // Calculate total number of batches
    printf("Total number of batches: %d\n", total_batches);

    int step = 0;
    int *input_words = (int *) malloc(batch_size * sizeof(int));
    int *target_words = (int *) malloc(batch_size * sizeof(int));

    for (epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        printf("Epoch %d start\n", epoch);
        clock_t epoch_start = clock();

        // Process each batch
        for (i = 0; i < train_pair_count; i += batch_size) {
            int current_batch_size;
            if (i + batch_size <= train_pair_count) {
                current_batch_size = batch_size;
            } else {
                current_batch_size = train_pair_count - i;
            }
            printf("Processing batch %d with size %d...\n", i / batch_size, current_batch_size);

            // Setup batch data
            for (b = 0; b < current_batch_size; b++) {
                input_words[b] = train_input_words[i + b];
                target_words[b] = train_target_words[i + b];
            }

            // Calculate current learning rate based on the schedule
            double learning_rate = learning_rate_schedule(initial_learning_rate, step, decay_rate, decay_steps);
            printf("Learning rate at step %d: %f\n", step, learning_rate);
            step++;

            // Measure time for forward pass
            clock_t start_time = clock();
            forward_pass_batch(input_words, current_batch_size, input_weights, output_weights, hidden_layer, output_layer, vocab_size, embedding_dim);
            clock_t end_time = clock();
            double forward_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
            printf("Forward pass time: %f seconds\n", forward_time);

            // Measure time for softmax and loss calculation
            start_time = clock();
            for (b = 0; b < current_batch_size; b++) {
                softmax(output_layer[b], vocab_size, output_probs[b]);
                double batch_loss = -log(output_probs[b][target_words[b]]);
                total_loss += batch_loss;
            }
            end_time = clock();
            double softmax_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
            printf("Softmax and loss calculation time: %f seconds\n", softmax_time);

            // Measure time for backward pass
            start_time = clock();
            backward_pass_batch(input_words, target_words, current_batch_size, hidden_layer, output_probs, input_weights, output_weights, vocab_size, embedding_dim, learning_rate);
            end_time = clock();
            double backward_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
            printf("Backward pass time: %f seconds\n", backward_time);

            // Print loss for the current batch
            if ((i / batch_size) % 10 == 0) {
                printf("Epoch %d, Batch %d, Current word in subset: %d, Loss: %f\n", epoch, i / batch_size, i, total_loss / (i + current_batch_size));
            }
        }

        // Print epoch loss
        clock_t epoch_end = clock();
        double epoch_time = (double)(epoch_end - epoch_start) / CLOCKS_PER_SEC;
        printf("Epoch %d, Loss: %f, Time: %f seconds\n", epoch, total_loss / train_pair_count, epoch_time);

        // Evaluate model on validation set
        double accuracy;
        evaluate_model(input_weights, output_weights, train_input_words, train_target_words, train_pair_count, vocab_size, embedding_dim, batch_size, &accuracy, x);
        printf("Validation Top-%d Accuracy after epoch %d: %f\n", x, epoch, accuracy);
    }

    // Save the trained model
    save_model(model_filename, input_weights, output_weights, vocab_size, embedding_dim);

    // Free memory
    for (i = 0; i < vocab_size; i++) {
        free(input_weights[i]);
        free(output_weights[i]);
    }
    free(input_weights);
    free(output_weights);
    for (i = 0; i < batch_size; i++) {
        free(hidden_layer[i]);
        free(output_layer[i]);
        free(output_probs[i]);
    }
    free(hidden_layer);
    free(output_layer);
    free(output_probs);
    free(train_input_words);
    free(train_target_words);
    free(valid_input_words);
    free(valid_target_words);
    free(input_words);
    free(target_words);
    free(token_indices);
    printf("Training completed.\n");
}

// Function to calculate the learning rate based on the decay rate and step
double learning_rate_schedule(double initial_learning_rate, int step, double decay_rate, int decay_steps) {
    return initial_learning_rate * pow(decay_rate, (double) step / decay_steps);
}

// Function to save the trained model to a file
void save_model(const char *filename, double **input_weights, double **output_weights, int vocab_size, int embedding_dim) {
    FILE *file = fopen(filename, "wb");
    int i;
    if (file == NULL) {
        fprintf(stderr, "Error opening file for saving model: %s\n", strerror(errno));
        return;
    }
    fwrite(&vocab_size, sizeof(int), 1, file);
    fwrite(&embedding_dim, sizeof(int), 1, file);

    for (i = 0; i < vocab_size; i++) {
        fwrite(input_weights[i], sizeof(double), embedding_dim, file);
        fwrite(output_weights[i], sizeof(double), embedding_dim, file);
    }

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Function to generate training data from token indices using a sliding window approach
void generate_training_data(int *token_indices, int token_count, int window_size, int **train_input_words, int **train_target_words, int *train_pair_count, int **valid_input_words, int **valid_target_words, int *valid_pair_count) {
    int i, j, pair_count = 0;
    int max_pairs = token_count * window_size * 2;  // Maximum possible pairs

    // Temporary storage for all pairs
    int *temp_input_words = (int *) malloc(max_pairs * sizeof(int));
    int *temp_target_words = (int *) malloc(max_pairs * sizeof(int));

    if (temp_input_words == NULL || temp_target_words == NULL) {
        printf("Error allocating memory for temporary training data: %s\n", strerror(errno));
        return;
    }

    // Generate all possible (input, target) pairs
    // printf("generated pair: \n");
    for (i = 0; i < token_count; i++) {
        for (j = 1; j <= window_size; j++) {
            if (i - j >= 0) {
                temp_input_words[pair_count] = token_indices[i];
                temp_target_words[pair_count] = token_indices[i - j];
                // printf("%d | %d\n", temp_input_words[pair_count], temp_target_words[pair_count]);
                pair_count++;
            }
            if (i + j < token_count) {
                temp_input_words[pair_count] = token_indices[i];
                temp_target_words[pair_count] = token_indices[i + j];
                // printf("%d | %d\n", temp_input_words[pair_count], temp_target_words[pair_count]);
                pair_count++;
            }
        }
    }

    // Shuffle pairs
    for (i = pair_count - 1; i > 0; i--) {
        int index = rand() % (i + 1);
        int temp_input = temp_input_words[i];
        int temp_target = temp_target_words[i];
        temp_input_words[i] = temp_input_words[index];
        temp_target_words[i] = temp_target_words[index];
        temp_input_words[index] = temp_input;
        temp_target_words[index] = temp_target;
    }

    // Split into training and validation sets
    int train_size = (int)(pair_count * 1);
    int valid_size = pair_count - train_size;

    *train_input_words = (int *) malloc(train_size * sizeof(int));
    *train_target_words = (int *) malloc(train_size * sizeof(int));
    *valid_input_words = (int *) malloc(valid_size * sizeof(int));
    *valid_target_words = (int *) malloc(valid_size * sizeof(int));

    if (*train_input_words == NULL || *train_target_words == NULL || *valid_input_words == NULL || *valid_target_words == NULL) {
        printf("Error allocating memory for training/validation data: %s\n", strerror(errno));
        return;
    }

    // Copy the shuffled pairs into training and validation arrays
    for (i = 0; i < train_size; i++) {
        (*train_input_words)[i] = temp_input_words[i];
        (*train_target_words)[i] = temp_target_words[i];
    }

    for (i = 0; i < valid_size; i++) {
        (*valid_input_words)[i] = temp_input_words[train_size + i];
        (*valid_target_words)[i] = temp_target_words[train_size + i];
    }

    *train_pair_count = train_size;
    *valid_pair_count = valid_size;

    free(temp_input_words);
    free(temp_target_words);
}

// Function to evaluate the trained model on a validation set using top x accuracy
void evaluate_model(double **input_weights, double **output_weights, int *input_words, int *target_words, int pair_count, int vocab_size, int embedding_dim, int batch_size, double *top_accuracy, int x) {
    int i, j, b;

    int correct_predictions_top1 = 0;
    int correct_predictions = 0;

    double **hidden_layer = (double **) malloc(batch_size * sizeof(double *));
    double **output_layer = (double **) malloc(batch_size * sizeof(double *));
    double **output_probs = (double **) malloc(batch_size * sizeof(double *));
    for (i = 0; i < batch_size; i++) {
        hidden_layer[i] = (double *) malloc(embedding_dim * sizeof(double));
        output_layer[i] = (double *) malloc(vocab_size * sizeof(double));
        output_probs[i] = (double *) malloc(vocab_size * sizeof(double));
    }

    // Evaluate the model in batches
    for (i = 0; i < pair_count; i += batch_size) {
        int current_batch_size;
        if (i + batch_size <= pair_count) {
            current_batch_size = batch_size;
        } else {
            current_batch_size = pair_count - i;
        }

        // Process each batch
        for (b = 0; b < current_batch_size; b++) {
            int input_word = input_words[i + b];
            // printf("Input word: %d\n", input_word);
            for (j = 0; j < embedding_dim; j++) {
                hidden_layer[b][j] = input_weights[input_word][j];
            }
            for (j = 0; j < vocab_size; j++) {
                output_layer[b][j] = 0.0;
                for (int k = 0; k < embedding_dim; k++) {
                    output_layer[b][j] += hidden_layer[b][k] * output_weights[j][k];
                }
            }
            softmax(output_layer[b], vocab_size, output_probs[b]);



            // Check top-x predictions
            int top_predicted_words[x];
            for (j = 0; j < x; j++) {
                top_predicted_words[j] = -1;
            }
            for (j = 0; j < x; j++) {
                int max_index = -1;
                double max_val = -1.0;
                for (int k = 0; k < vocab_size; k++) {
                    int already_chosen = 0;
                    for (int l = 0; l < j; l++) {
                        if (top_predicted_words[l] == k) {
                            already_chosen = 1;
                            break;
                        }
                    }
                    if (!already_chosen && output_probs[b][k] > max_val) {
                        max_val = output_probs[b][k];
                        max_index = k;
                    }
                }
                top_predicted_words[j] = max_index;
            }
            // printf("Targeted word: %d\n", target_words[i + b]);
            // printf("Top predicted word: ");
            for (j = 0; j < x; j++) {
                // printf("%d ", top_predicted_words[j]);
                if (top_predicted_words[j] == target_words[i + b]) {
                    correct_predictions++;
                    break;
                }
            }
            // printf("\n");
        }
    }
    printf("Correction: %d | pair count: %d\n", correct_predictions, pair_count);

    // Free memory
    for (i = 0; i < batch_size; i++) {
        free(hidden_layer[i]);
        free(output_layer[i]);
        free(output_probs[i]);
    }
    free(hidden_layer);
    free(output_layer);
    free(output_probs);

    // Calculate and return top-x accuracy and average loss
    *top_accuracy = (double)correct_predictions / pair_count;
}

