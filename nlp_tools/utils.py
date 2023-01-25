import nltk
import string
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from collections import Counter
from matplotlib.pyplot import figure

nltk.download('stopwords')

def remove_punctuation(self, string_input):
  """
    Removes punctuation: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
  """
  return string_input.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(self, string_input):
  """
    Removes numbers: '0123456789'
  """
  return string_input.translate(str.maketrans('', '', string.digits))

def stem_words(self, string_list):
  porter_stemmer = nltk.PorterStemmer()
  roots = [porter_stemmer.stem(i) for i in string_list]
  return roots

def remove_stop_words(self, string_list):
  stop_words = stopwords.words('english')
  new = []
  for word in string_list:
    if word.lower() not in stop_words:
      new.append(word)
  return new

# Visualize the data to see the class distributions in the training and test sets
def visualize_data_freq(dataset, data_split, labels):
  counts = Counter(dataset.values())
  figure(figsize=(15, 8), dpi=80)
  plt.rcParams['font.size'] = 15
  graph = plt.bar(range(len(labels)), 
                  counts.values(), 
                  color=['hotpink', 'yellow', 'lightgreen', 'violet'])
  plt.xticks(range(len(labels)), labels)
  plt.title(f"Distribution of classes for {data_split} dataset of AG News Dataset")
  plt.xlabel('Category')
  plt.ylabel('Frequency')

  i = 0
  for p in graph:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    plt.text(x + width/2,
            y + height*1.01,
            str(list(counts.values())[i]),
            ha='center')
    i = i + 1
  plt.show()
