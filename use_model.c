#include "use_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>

// Function to retrieves the word embedding for a given word index from the input weights matrix
void get_word_embedding(int word_index, double **input_weights, double *embedding, int embedding_dim) {
    for (int i = 0; i < embedding_dim; i++) {
        embedding[i] = input_weights[word_index][i];
    }
}

int predict_word(int input_word, double **input_weights, double **output_weights, int vocab_size, int embedding_dim) {
    double *hidden_layer = (double *)malloc(embedding_dim * sizeof(double));
    double *output_layer = (double *)malloc(vocab_size * sizeof(double));

    if (hidden_layer == NULL || output_layer == NULL) {
        printf("Error allocating memory for prediction: %s\n", strerror(errno));
        return -1;
    }

    for (int i = 0; i < embedding_dim; i++) {
        hidden_layer[i] = input_weights[input_word][i];
    }

    for (int i = 0; i < vocab_size; i++) {
        output_layer[i] = 0.0;
        for (int j = 0; j < embedding_dim; j++) {
            output_layer[i] += hidden_layer[j] * output_weights[i][j];
        }
    }

    double max_score = output_layer[0];
    for (int i = 1; i < vocab_size; i++) {
        if (output_layer[i] > max_score) {
            max_score = output_layer[i];
        }
    }

    double sum_exp = 0.0;
    for (int i = 0; i < vocab_size; i++) {
        output_layer[i] = exp(output_layer[i] - max_score);
        sum_exp += output_layer[i];
    }

    for (int i = 0; i < vocab_size; i++) {
        output_layer[i] /= sum_exp;
    }

    int predicted_word = 0;
    double max_prob = output_layer[0];
    for (int i = 1; i < vocab_size; i++) {
        if (output_layer[i] > max_prob) {
            max_prob = output_layer[i];
            predicted_word = i;
        }
    }

    free(hidden_layer);
    free(output_layer);

    return predicted_word;
}

void load_vocab(const char *filename, VocabItem **vocab, int *vocab_size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file to load vocabulary: %s\n", strerror(errno));
        return;
    }

    int capacity = 10000;
    *vocab = (VocabItem *)malloc(capacity * sizeof(VocabItem));
    if (!*vocab) {
        printf("Memory allocation for vocabulary failed: %s\n", strerror(errno));
        fclose(file);
        return;
    }

    *vocab_size = 0;
    char word[100];
    int count;
    while (fscanf(file, "%s %d", word, &count) == 2) {
        if (*vocab_size >= capacity) {
            capacity *= 2;
            *vocab = (VocabItem *)realloc(*vocab, capacity * sizeof(VocabItem));
            if (!*vocab) {
                printf("Memory reallocation for vocabulary failed: %s\n", strerror(errno));
                fclose(file);
                return;
            }
        }
        (*vocab)[*vocab_size].word = my_strdup(word);
        (*vocab)[*vocab_size].count = count;
        (*vocab_size)++;
    }

    fclose(file);
    printf("Vocabulary loaded from %s\n", filename);
}

const char* get_word_by_index(int index, VocabItem *vocab, int vocab_size) {
    if (index < 0 || index >= vocab_size) {
        return NULL;
    }
    return vocab[index].word;
}

double cosine_similarity(double *vec1, double *vec2, int size) {
    double dot_product = 0.0;
    double norm_vec1 = 0.0;
    double norm_vec2 = 0.0;
    for (int i = 0; i < size; i++) {
        dot_product += vec1[i] * vec2[i];
        norm_vec1 += vec1[i] * vec1[i];
        norm_vec2 += vec2[i] * vec2[i];
    }
    return dot_product / (sqrt(norm_vec1) * sqrt(norm_vec2));
}

void find_top_5_similar_words(int target_index, VocabItem *vocab, int vocab_size, double **input_weights, int embedding_dim, char **top_words) {
    if (target_index < 0 || target_index >= vocab_size) {
        printf("Invalid index '%d'.\n", target_index);
        return;
    }

    double top_similarities[5] = {-1, -1, -1, -1, -1};
    int top_indices[5] = {-1, -1, -1, -1, -1};

    for (int i = 0; i < vocab_size; i++) {
        if (i != target_index) {
            double similarity = cosine_similarity(input_weights[target_index], input_weights[i], embedding_dim);
            
            // Insert the similarity and index into the top 5 arrays
            for (int j = 0; j < 5; j++) {
                if (similarity > top_similarities[j]) {
                    for (int k = 4; k > j; k--) {
                        top_similarities[k] = top_similarities[k - 1];
                        top_indices[k] = top_indices[k - 1];
                    }
                    top_similarities[j] = similarity;
                    top_indices[j] = i;
                    break;
                }
            }
        }
    }

    for (int i = 0; i < 5; i++) {
        if (top_indices[i] != -1) {
            top_words[i] = vocab[top_indices[i]].word;
        } else {
            top_words[i] = NULL;
        }
    }
}