# SkipGram-Word-Embedding-in-C
 Skip-Gram Word Embedding Model implemented in C, utilizing data extracted from documents available on the Tokyo Institute of Technology website.

## How to Run the Program and Example Outputs

### 1. How to Run the Program

To effectively run the Skip-gram model, it is important to follow specific steps during the training and usage of the model.

#### 1.1 Training the Model

To train the model, navigate to the directory containing the source code and run the following commands.

First, set the necessary parameters for training in `main.c`. These parameters include:

- Maximum number of tokens
- Model file name
- Window size
- Embedding dimensions
- Initial learning rate
- Learning rate decay rate and step
- Number of epochs
- Batch size

These parameters can either be set directly in the code or passed as command-line arguments.

Additionally, you need to set the token limit in `preprocessing.c`. This is defined by the `MAX_TOKENS` constant in the `tokenize` function.

Compile the source files using the following command:

```bash
gcc main.c preprocessing.c train.c use_model.c math_helper.c -o word2vec -lm
```

This command compiles `main.c`, `preprocessing.c`, `train.c`, `use_model.c`, and `math_helper.c`, and generates the executable file `word2vec`.

Once the compilation is successful, start the training process with:
```bash
./word2vec
```

### 2. Example Outputs

This program created two models with different token counts: `word2vec model 1K` and `word2vec model 100K`. After training, we tested the models using the word `"teams"`.

#### 2.1 1K Token Model

The `word2vec model 1K` was trained with 1000 tokens using the following settings:

- **Window size**: 2
- **Embedding dimensions**: 150
- **Initial learning rate**: 0.05
- **Decay rate**: 0.96
- **Decay step**: 50
- **Epochs**: 50
- **Batch size**: 64
- **Evaluation**: Top 5 accuracy

##### Training Results:

```plaintext
Vocabulary size: 334
Training pairs: 3994
Total batches: 63
Accuracy: 0.674512
Time for 50 epochs: 198.523000 seconds
```

Test Output for the word "teams":
```plaintext
Word embedding: [-0.419234, 0.096878, 0.106686, ..., 0.226047]
Predicted word: joint
Top 5 similar words: joint, tmdu, visitors, organization, global
```

The 1K token model achieved a relatively high accuracy of around 67%, indicating that the word embeddings were properly learned even with a small dataset. The training time was short (198.523 seconds), demonstrating efficient learning. While the predicted word `"joint"` and the similar words like `"tmdu"`, `"visitors"`, `"organization"`, and `"global"` may not seem directly related to `"teams"`, it suggests that the model learned specific contexts.

#### 2.2 100K Token Model

The `word2vec model 100K` was trained with 100,000 tokens using the following settings:

- **Window size**: 2
- **Embedding dimensions**: 150
- **Initial learning rate**: 0.05
- **Decay rate**: 0.96
- **Decay step**: 50
- **Epochs**: 50
- **Batch size**: 256
- **Evaluation**: Top 20 accuracy

##### Training Results:

```plaintext
Vocabulary size: 11,407
Training pairs: 399,994
Total batches: 1563
Accuracy: 0.123977
Time for 5 epochs: 76300.608000 seconds
```

Test Output for the word "teams":
```plaintext
Word embedding: [0.010331, -0.027175, 0.013893, ..., 0.040569]
Predicted word: tokyo
Top 5 similar words: ward, donations, doctoratephd, hitotsubashi, membership
```

Despite using a larger dataset, the accuracy dropped to around 12%, which might indicate that the model struggled to capture word relationships due to the large volume of data, making it harder to identify meaningful connections. The significantly longer training time (around 21 hours) and potentially suboptimal parameter settings highlight the high computational cost. The predicted word `"tokyo"` and the similar words like `"ward"`, `"donations"`, `"doctoratephd"`, `"hitotsubashi"`, and `"membership"` show little relation to `"teams"`, suggesting that the model may not have properly learned word relationships.
