# taken from https://www.tensorflow.org/text/tutorials/text_classification_rnn
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_tf_history(history):
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.subplot(1, 2, 2)
    plot_graphs(history, 'loss')


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

def visualize_word_vectors(word:str, pca_df, words_to_iterate):
  ax = pca_df.plot(x='pc1',y='pc2',kind="scatter",figsize=(10, 5),alpha=0)
  title_all_words_graph = "Visualization of word vectors after PCA"
  title_separate_graphs = "Visualization of word vectors after PCA for \'"+word+"\' and its most similiar words"
  if word == "":
    plt.title(title_all_words_graph)
  else:
    plt.title(title_separate_graphs)
  
  for txt in pca_df.index:
    if txt in words_to_iterate:
        x = pca_df.pc1.loc[txt]
        y = pca_df.pc2.loc[txt]
        ax.annotate(txt, (x,y))
  plt.show()

def heatmaps_of_feature_vectors(model, list_of_words, title_of_plot):
  indices_of_words = [model.get_index(word) for word in list_of_words] # The numerical indices of those words
  df = pd.DataFrame(model.get_normed_vectors()[indices_of_words], index=list_of_words)
  f, ax = plt.subplots(figsize=(10, 5))
  plt.title(title_of_plot)
  sns.heatmap(df, cmap='coolwarm')
