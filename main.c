#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <errno.h>
#include "processing.h"
#include "train.h"
#include "use_model.h"
#include "math_helper.h"

#define MAX_STOPWORDS 1000

int file_exists(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file) {
        fclose(file);
        return 1;  // Return 1 if the file exists
    }
    return 0;  // Return 0 if the file does not exist
}

int main() {
    const char *corpus_filename = "text.txt";
    const char *tokens_filename = "tokens.txt";
    const char *model_filename = "word2vec_model.bin";
    const char *vocab_filename = "vocab.txt";
    const char *stopwords_filename = "stop_words_english.txt";
    int window_size = 2;
    int embedding_dim = 150;
    double initial_learning_rate = 0.05;
    double decay_rate = 0.96;
    int decay_steps = 50;
    int epochs = 5;
    int batch_size = 256;
    int top_x = 20;
    int i;

    char **tokens;
    int token_count;
    VocabItem *vocab = NULL;
    int vocab_size = 0;
    char *stopwords[MAX_STOPWORDS];
    int num_stopwords;

    // Load stopwords from file
    printf("Loading stopwords from file...\n");
    load_stopwords(stopwords_filename, stopwords, &num_stopwords, MAX_STOPWORDS);

    if (file_exists(tokens_filename) && file_exists(vocab_filename)) {
        // Load tokens from file
        printf("Loading tokens from file...\n");
        load_tokens(tokens_filename, &tokens, &token_count);
        printf("Loaded %d tokens from file.\n", token_count);
    } else {
        // Tokenize and save tokens if not already done
        printf("Reading and tokenizing corpus...\n");
        char *corpus = NULL;
        read_corpus(corpus_filename, &corpus);
        tokenize(corpus, &tokens, &token_count, stopwords, num_stopwords);
        save_tokens(tokens_filename, tokens, token_count);

        // Free allocated memory for corpus
        free(corpus);
    }

    if (file_exists(vocab_filename)) {
        // Load vocabulary from file
        printf("Loading vocabulary from file...\n");
        load_vocab(vocab_filename, &vocab, &vocab_size);
        printf("Loaded vocabulary of size %d from file.\n", vocab_size);
    } else {
        // Build and save vocabulary
        build_vocab(tokens, token_count, &vocab, &vocab_size);
        save_vocab(vocab_filename, vocab, vocab_size);
        printf("Vocabulary built and saved to %s\n", vocab_filename);
    }

    // Check if the model file exists
    if (file_exists(model_filename)) {
        // Load the pre-trained model
        printf("Loading pre-trained model from %s...\n", model_filename);
        double **input_weights = (double **) malloc(vocab_size * sizeof(double *));
        double **output_weights = (double **) malloc(vocab_size * sizeof(double *));
        if (input_weights == NULL || output_weights == NULL) {
            printf("Error allocating memory for model weights: %s\n", strerror(errno));
            return 1;
        }
        for (i = 0; i < vocab_size; i++) {
            input_weights[i] = (double *) malloc(embedding_dim * sizeof(double));
            output_weights[i] = (double *) malloc(embedding_dim * sizeof(double));
            if (input_weights[i] == NULL || output_weights[i] == NULL) {
                printf("Error allocating memory for model weight matrices: %s\n", strerror(errno));
                return 1;
            }
        }
        load_model(model_filename, input_weights, output_weights, vocab_size, embedding_dim);
        printf("Pre-trained model loaded. Continuing training...\n");
    } else {
        printf("No pre-trained model found. Training a new model from scratch...\n");
    }

    printf("Starting training...\n");
    clock_t start = clock();
    train_model(tokens, token_count, window_size, embedding_dim, initial_learning_rate, decay_rate, decay_steps, epochs, batch_size, vocab, vocab_size, file_exists(model_filename), (char *)model_filename, top_x);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time for %d epochs: %f seconds\n", epochs, time_spent);

    // Free allocated memory for tokens
    for (i = 0; i < token_count; i++) {
        free(tokens[i]);
    }
    free(tokens);

    // Free memory for vocabulary
    for (i = 0; i < vocab_size; i++) {
        free(vocab[i].word);
    }
    free(vocab);

    // Free memory for stopwords
    free_stopwords(stopwords, num_stopwords);

    printf("Program completed.\n");
    return 0;
}
