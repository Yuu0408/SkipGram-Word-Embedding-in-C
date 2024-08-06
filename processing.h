#ifndef PROCESSING_H
#define PROCESSING_H

typedef struct {
    char *word;
    int count;
} VocabItem;

void read_corpus(const char *filename, char **corpus);
void tokenize(const char *corpus, char ***tokens, int *token_count, char **stopwords, int num_stopwords);
void save_tokens(const char *filename, char **tokens, int token_count);
void load_tokens(const char *filename, char ***tokens, int *token_count);
void build_vocab(char **tokens, int token_count, VocabItem **vocab, int *vocab_size);
void save_vocab(const char *filename, VocabItem *vocab, int vocab_size);
void load_stopwords(const char *filename, char **stopwords, int *num_stopwords, int max_stopwords);
int is_stopword(const char *word, char **stopwords, int num_stopwords);
void free_stopwords(char **stopwords, int num_stopwords);

#endif
