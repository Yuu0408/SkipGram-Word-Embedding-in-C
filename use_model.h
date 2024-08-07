#ifndef USE_MODEL_H
#define USE_MODEL_H

#include "processing.h"
#include "train.h"

void get_word_embedding(int word_index, double **input_weights, double *embedding, int embedding_dim);
int predict_word(int input_word, double **input_weights, double **output_weights, int vocab_size, int embedding_dim);
const char* get_word_by_index(int index, VocabItem *vocab, int vocab_size);
void load_vocab(const char *filename, VocabItem **vocab, int *vocab_size);
void find_top_5_similar_words(int target_index, VocabItem *vocab, int vocab_size, double **input_weights, int embedding_dim, char **top_words);
double cosine_similarity(double *vec1, double *vec2, int size);

#endif
