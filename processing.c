#include "processing.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

// Function for string duplication
char* my_strdup(const char* s) {
    char* p = malloc(strlen(s) + 1);
    if (p) strcpy(p, s);
    return p;
}

// Function to read an entire file into a memory buffer
void read_corpus(const char *filename, char **corpus) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file: %s\n", strerror(errno));
        return;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    *corpus = (char *)malloc((file_size + 1) * sizeof(char));
    if (!*corpus) {
        printf("Memory allocation failed\n");
        fclose(file);
        return;
    }

    fread(*corpus, sizeof(char), file_size, file);
    (*corpus)[file_size] = '\0';
    fclose(file);
}

// Function to load stopwords from a file
void load_stopwords(const char *filename, char **stopwords, int *num_stopwords, int max_stopwords) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Failed to open stopwords file: %s\n", strerror(errno));
        return;
    }

    char line[100];
    *num_stopwords = 0;

    while (fgets(line, sizeof(line), file) != NULL && *num_stopwords < max_stopwords) {
        // Remove newline character (if present)
        int len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') {
            line[len - 1] = '\0';
        }

        // Allocate memory for the new stopword
        stopwords[*num_stopwords] = (char *) malloc((len + 1) * sizeof(char));
        if (stopwords[*num_stopwords] == NULL) {
            printf("Memory allocation for stopword failed: %s\n", strerror(errno));
            fclose(file);
            return;
        }
        strcpy(stopwords[*num_stopwords], line);

        (*num_stopwords)++;
    }

    fclose(file);
}

// Function to check if a word is a stopword
int is_stopword(const char *word, char **stopwords, int num_stopwords) {
    for (int i = 0; i < num_stopwords; i++) {
        if (strcmp(word, stopwords[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

// Function to free memory allocated for stopwords
void free_stopwords(char **stopwords, int num_stopwords) {
    for (int i = 0; i < num_stopwords; i++) {
        free(stopwords[i]);
    }
    free(stopwords);
}

// Function to tokenize the corpus into words excluding stopwords
void tokenize(const char *corpus, char ***tokens, int *token_count, char **stopwords, int num_stopwords) {
    printf("Starting tokenization...\n");
    int MAX_TOKENS = 1000; // Limit maximum number of tokens to 500
    *tokens = (char **) malloc(MAX_TOKENS * sizeof(char *));
    if (*tokens == NULL) {
        printf("Memory allocation for tokens failed: %s\n", strerror(errno));
        return;
    }

    const char *start = corpus;
    const char *current = corpus;
    *token_count = 0;

    // Loop through corpus until the end of the string or maximum token count is reached
    while (*current != '\0' && *token_count < MAX_TOKENS) {
        if (*current == ' ' || *current == '\n') {  // Check for token delimiters (space or newline)
            if (current > start) {  // Calculate the length of the token
                size_t length = current - start;
                if (length >= 100) {    // Check if token exceeds maximum length
                    printf("Token length exceeds maximum allowed length\n");
                    return;
                }
                char *token = (char *) malloc((length + 1) * sizeof(char));
                if (token == NULL) {
                    printf("Memory allocation for a token failed: %s\n", strerror(errno));
                    return;
                }
                strncpy(token, start, length);
                token[length] = '\0';

                if (!is_stopword(token, stopwords, num_stopwords)) {
                    (*tokens)[*token_count] = token;
                    (*token_count)++;
                } else {
                    free(token);
                }
            }
            start = current + 1;
        }
        current++;
    }

    if (*token_count >= MAX_TOKENS) {   // Check if the maximum token limit is reached
        printf("Tokenization stopped after reaching the maximum token limit of %d\n", MAX_TOKENS);
    }

    // Process the last token if any
    if (current > start && *token_count < MAX_TOKENS) {
        size_t length = current - start;
        if (length >= 100) {
            printf("Token length exceeds maximum allowed length\n");
            return;
        }
        char *token = (char *) malloc((length + 1) * sizeof(char));
        if (token == NULL) {
            printf("Memory allocation for a token failed: %s\n", strerror(errno));
            return;
        }
        strncpy(token, start, length);
        token[length] = '\0';

        if (!is_stopword(token, stopwords, num_stopwords)) {
            (*tokens)[*token_count] = token;
            (*token_count)++;
        } else {
            free(token);
        }
    }

    printf("Token Count: %d\n", *token_count);
    printf("Tokenization succeeded\n");
}

// Function to save tokens to a file
void save_tokens(const char *filename, char **tokens, int token_count) {
    FILE *file = fopen(filename, "w");
    int i;
    if (!file) {
        printf("Error opening file to save tokens: %s\n", strerror(errno));
        return;
    }

    for (i = 0; i < token_count; i++) {
        fprintf(file, "%s\n", tokens[i]);
    }

    fclose(file);
    printf("Tokens saved to %s\n", filename);
}

// Function to load tokens from a file
void load_tokens(const char *filename, char ***tokens, int *token_count) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file to load tokens: %s\n", strerror(errno));
        return;
    }

    int capacity = 100000;   // Initial capacity for tokens array
    *tokens = (char **)malloc(capacity * sizeof(char *));
    if (!*tokens) {
        printf("Memory allocation for tokens failed: %s\n", strerror(errno));
        fclose(file);
        return;
    }

    *token_count = 0;
    char buffer[100];
    while (fgets(buffer, sizeof(buffer), file) && *token_count < capacity) {
        buffer[strcspn(buffer, "\n")] = '\0';
        (*tokens)[*token_count] = my_strdup(buffer);
        (*token_count)++;
    }

    fclose(file);
    printf("Tokens loaded from %s\n", filename);
}

// Function to build a vocabulary from tokens
void build_vocab(char **tokens, int token_count, VocabItem **vocab, int *vocab_size) {
    int capacity = 10000;   // Initial capacity for vocabulary
    int i;
    *vocab = (VocabItem *)malloc(capacity * sizeof(VocabItem));
    if (!*vocab) {
        printf("Memory allocation for vocabulary failed: %s\n", strerror(errno));
        return;
    }

    *vocab_size = 0;
    for (i = 0; i < token_count; i++) {
        int j;
        for (j = 0; j < *vocab_size; j++) {
            if (strcmp(tokens[i], (*vocab)[j].word) == 0) { // Check if token is already in vocabulary
                (*vocab)[j].count++;    // Increment count of existing word
                break;
            }
        }
        if (j == *vocab_size) {
            if (*vocab_size >= capacity) {  // Check if capacity is exceeded
                capacity *= 2;  // Double the capacity
                *vocab = (VocabItem *)realloc(*vocab, capacity * sizeof(VocabItem));
                if (!*vocab) {
                    printf("Memory reallocation for vocabulary failed: %s\n", strerror(errno));
                    return;
                }
            }
            (*vocab)[*vocab_size].word = my_strdup(tokens[i]);
            (*vocab)[*vocab_size].count = 1;
            (*vocab_size)++;
        }
    }

    printf("Vocabulary built. Size: %d\n", *vocab_size);
}

// Function to save vocabulary to a file
void save_vocab(const char *filename, VocabItem *vocab, int vocab_size) {
    FILE *file = fopen(filename, "w");
    int i;
    if (!file) {
        printf("Error opening file to save vocabulary: %s\n", strerror(errno));
        return;
    }

    for (i = 0; i < vocab_size; i++) {
        fprintf(file, "%s %d\n", vocab[i].word, vocab[i].count);
    }

    fclose(file);
    printf("Vocabulary saved to %s\n", filename);
}