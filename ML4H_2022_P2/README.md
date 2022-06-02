# ML4H_2022_P2

## Where to put the data
Place the 3 text files (train.txt, dev.txt and test.txt) under the *data/* folder.

## Setting up the environment
`conda env create --file environment.yml`

`conda activate ml4h_project2`

Add the environment to jupyter \\
`python -m ipykernel install --user --name ml4h_project2 --display-name "ml4h_project2"`

## How the tasks are solved
The tasks are split into specific notebooks adressing either a complete task or some part of it.

## How to run the code.
TLDR: See **HOW TO RUN** parts and run things in the order they appear in the readme.
Otherwise, there are more detailed explanations on how things work.

## Task1
**HOW TO RUN** <br>
Open the **Task1** notebook and select "Run all cells". <br>

The notebook **Task1** solves the first task, by providing the following functionalities:
- Preprocess the data (each sentence in the dataset is tokenized) and persists the preprocessed result. This preprocessing is also used for Task2, so it's expected that this notebook is run first and foremost!
- Computes TF-IDF features
- Trains 2 baseline models: Naive Bayes and Linear classifiers (for which there is also hyperparameter search)

If one wants to compute different preprocessing configs than the original, one can switch the flags for *PreprocessingOptions*. By default, we only remove white spaces and punctuation (this had best performance), but stop words removal and lemmatization can be enabled if required (see *Config* section).

## Task2
As this task is more complex, it is split into multiple notebooks.

The starting point happens in notebook **Task2_ComputeEmbeddings**, which does the following:
- Loads the preprocessed data (assumed to have been already computed previously)
- Trains a Word2Vec or FastText model (based on the value of the EMBEDDING variable) on the preprocessed training data (persisted afterwards, so they only need to be trained once)
- Provides 2 ways to construct sentence embeddings: Concatenation and Summation, which are persisted after being computed.

### HOW TO RUN - Task 2 Part 1
Open the **Task2_ComputeEmbeddings** notebook and select "Run all cells" <br>

Options to configure in this notebook are the preprocessing options, Word2Vec and FastText model settings (cbow or Skip_N-gram, vectors size) and the max sentence size when concatenating (see the *Config* section)

Next, the **Task2_RNNModelTraining** loads the embeddings obtained by Concatenation and trains various RNN type models: simple RNN, LSTM, bidirectional LSTM, and one bidirectional LSTM with more layers and more units. **NOTE**: This step works only for Word2vec Embeddings.

### HOW TO RUN - Task 2 Part 2
Open **Task2_RNNModelTraining** notebook. <br><br>
If only inference is needed, skip the *Start training* section and run evaluation from a checkpoint directly. The model trained with the default settings will be provided. Otherwise, just run all cells in the notebook. <br>


### HOW TO RUN - Task 2 Part 3
Run all cells in **Task2_BaselineModels**

We also ran some linear models on Word2Vec and FastText sentence embeddings for the sake of comparison.

### HOW TO RUN - Task 2 Part 4
This task explores semantic relationships of word embeddings like Word2vec and FastText. You can run the notebook and see the visualizations created by "Run all cells" in **Task2_SemanticRelationships**. In the data loading section, you can change the value of the EMBEDDING variable to either "fasttext" or "word2vec" to create the visualizations for each of the embeddings respectively. For the last part(Analogies), the default running configuration we have provided works for 2 words. If you uncomment the commented lines and the right title in the next cell, it will work for 3 words too.

## Task3
**HOW TO RUN**
Open the **Task3** notebook and select "Run all cells". Your machine will then solve the first part by fine tuning a pretrained model where we only train the last layer (part 1), after that it will also fine tune the pretrained model on all layers (part 2). Because of computational limitations we were able to do those fine tuning experiments only with subsamples of the available dataset. We tested the performance for both parts (partial and full) with random subsamples of size 1k, 10k and 20k. To do this, run the notebooks three time and change the size of the training subsample at the location indicated in the notebook. Feel free to try out other subsample sizes. We recommend a GPU to run this notebook.



