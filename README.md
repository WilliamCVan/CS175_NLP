# CS175_NLP

<u>Datasets used:</u>
1. https://www.kaggle.com/team-ai/japaneseenglish-bilingual-corpus
2. https://nlp.stanford.edu/projects/jesc/ 

<u>Project layout:</u>  
![folder layout](./directoryLayout.PNG)

<u>Directions:</u>
1. Run genFileWikipedia.py to generate wikipedia_raw (jap, eng pairs)
2. parseTokenizeFiles.py for preprocessing (nltk, text sanitization)
3. util.py functions to tokenize japanese, add <EOS>, <SOS> tokens

<u>Colab Notebooks Links:</u>

1. Torchtext: https://colab.research.google.com/drive/1eTLcyfTZK4CfrZL4372I72DuXHclYyot?usp=sharing#scrollTo=N79gPKtJQMzi
2. Tensorflow: https://colab.research.google.com/drive/1JgxnnqolCddBtlJ1fSMIdhB4bW_MMW64