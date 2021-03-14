# CS175_NLP: TranslationWeeb

<u>Completed Trained Model Link:</u>

-https://drive.google.com/u/1/uc?id=1MkRyroGmtHM75ZdaTgJ4bScemO75qAST&export=download

<u>Datasets used:</u>
1. https://www.kaggle.com/team-ai/japaneseenglish-bilingual-corpus
2. https://nlp.stanford.edu/projects/jesc/ (Processed data file needed for project.ipynb available here: https://drive.google.com/u/0/uc?id=19_jCHSv3AYqFOiXdzi1MSwaYNfIfQJSh&export=download)

Folder: src file descriptions
1. encDecoderGRU.py - pretrained vector embeddings for encoder and decoder
2. encDecoderLSTM.py - basic sequence to sequence model using torchText for embedding layer
3. genFileWikipedia.py - generate wikipedia_raw (jap, eng pairs) 
4. torchAttn_v1.py - Pytorch tutorial, spacy for japanese tokenization, string encoded japanese for model inputs, attention layer
5. torchAttn_v2.py - Pytorch tutorial, Fugashi for japanese tokenization, string encoded japanese for model inputs, attention layer
6. torchAttn_v3.py - Pytorch tutorial, Fugashi for japanese tokenization, fed Unidic objects into langauge dictionary, attention layer
7. torchTT.ipynb - basic seq2seq model using pretrained embeddings using Word2Vec and fugashi for tokenization
8. torchTT.py - basic seq2seq model using torchText for embedding layer and spacy for tokenization
9. util.py - text preprocessing of corpus files generating final clean corpus files (nltk, text sanitization)
10. z_genPair.py - created the pickle file with <SOS> and <EOS> tokens to be loaded by model
11. z_genVectors.py - gensim vector embedding trained models
12. z_seq2seq_translation_tutorial.py - pytorch tutorial that was modified to translate japanese to english using attention
13. z_translateWEEB.py - Pytorch tutorial, using pretrained word embeddings (gensim) to translate japanese to english
