#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include "processing.h"
#include "use_model.h"


int main() {
    const char *model_filename = "word2vec_model_1K_checkpoint3.bin";
    const char *vocab_filename = "vocab.txt";
    int embedding_dim = 150; 
    int vocab_size;
    int i;

    srand(time(NULL));

    // Allocate memory for model weights
    double **input_weights;
    double **output_weights;
    VocabItem *vocab;

    // Load the vocabulary
    load_vocab(vocab_filename, &vocab, &vocab_size);

    // Allocate memory for model weights
    input_weights = (double **)malloc(vocab_size * sizeof(double *));
    output_weights = (double **)malloc(vocab_size * sizeof(double *));
    if (input_weights == NULL || output_weights == NULL) {
        printf("Error allocating memory for model weights: %s\n", strerror(errno));
        return 1;
    }
    for (i = 0; i < vocab_size; i++) {
        input_weights[i] = (double *)malloc(embedding_dim * sizeof(double));
        output_weights[i] = (double *)malloc(embedding_dim * sizeof(double));
        if (input_weights[i] == NULL || output_weights[i] == NULL) {
            printf("Error allocating memory for model weight matrices: %s\n", strerror(errno));
            return 1;
        }
    }
    // Load the model
    load_model(model_filename, input_weights, output_weights, vocab_size, embedding_dim);

    // Example: Use random tokens from the vocabulary
    int input_word = rand() % vocab_size;
    printf("Input word: %s\n", get_word_by_index(input_word, vocab, vocab_size));
    double *embedding = (double *)malloc(embedding_dim * sizeof(double));
    if (embedding == NULL) {
        printf("Error allocating memory for embedding: %s\n", strerror(errno));
        return 1;
    }
    get_word_embedding(input_word, input_weights, embedding, embedding_dim);

    printf("Embedding for the word '%s': ", get_word_by_index(input_word, vocab, vocab_size));
    for (i = 0; i < embedding_dim; i++) {
        printf("%f ", embedding[i]);
    }
    printf("\n");

    // Predict the next word
    int predicted_word_index = predict_word(input_word, input_weights, output_weights, vocab_size, embedding_dim);
    if (predicted_word_index == -1) {
        return 1;
    }
    const char* predicted_word = get_word_by_index(predicted_word_index, vocab, vocab_size);
    printf("Predicted word: %s\n", predicted_word);

    // Find the top 5 most similar words to the input word
    char *top_words[5];
    find_top_5_similar_words(input_word, vocab, vocab_size, input_weights, embedding_dim, top_words);
    printf("Top 5 similar words to '%s':\n", get_word_by_index(input_word, vocab, vocab_size));
    for (i = 0; i < 5; i++) {
        if (top_words[i] != NULL) {
            printf("%s\n", top_words[i]);
        }
    }

    // Free memory
    for (i = 0; i < vocab_size; i++) {
        free(input_weights[i]);
        free(output_weights[i]);
    }
    free(input_weights);
    free(output_weights);
    for (i = 0; i < vocab_size; i++) {
        free(vocab[i].word);
    }
    free(vocab);
    free(embedding);

    return 0;
}