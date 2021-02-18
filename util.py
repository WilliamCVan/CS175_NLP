import fugashi
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# generates sentences for gensim's word2vec encoder
# uses fugashi to tokenize japanese text
class weebCorpus:
    filename = ""
    tagger = fugashi.Tagger()

    def __init__(self, f):
        self.filename = f

    def __iter__(self):
        for line in open(self.filename, mode="r", encoding="utf-8"):
            text = line.split("\t")[1]
            tokenized_text = [word.surface for word in self.tagger(text)]
            # print(tokenized_text)
            yield tokenized_text

def genVectors(corpus):
    sentences = weebCorpus(corpus)
    model = Word2Vec(sentences=sentences,
                     size=50,
                     window=5,
                     min_count=5,
                     workers=4,
                     iter=5)

    model.save("./weeb2vec.model")
    model.wv.save_word2vec_format("./weeb2vecnb.txt", binary=False)

# genVectors("./datafiles/wikipedia_raw")


import spacy
nlp = spacy.load("ja_core_news_sm")
def tokenizeJapanese(filename) -> list:
    tokens = list()

    with open(filename, mode="r", encoding="utf-8") as file_in:
        for line in file_in:
            eng, jap = line.split("\t")
            print("input: ", jap)
            sen = nlp(jap)
            for token in sen:
                print(token.text, token.pos_, token.dep_)
                tokens.append(token)
    return tokens
# processJapEnglishPairList("./datafiles/wikipedia_raw")


from math import floor

def get_training_and_testing_sets():
    split = 0.7
    with open("./datafiles/standford_test3.tsv", mode="r", encoding="utf-8") as file_in:
        total_lines = file_in.readlines()

        split_index = floor(len(total_lines) * split)
        training = total_lines[:split_index]
        testing = total_lines[split_index:]
    #return training, testing

    with open("./datafiles/standford_train4.tsv", mode="w", encoding="utf-8") as file_train:
        file_train.writelines(training)

    with open("./datafiles/standford_test4.tsv", mode="w", encoding="utf-8") as file_train:
        file_train.writelines(testing)

#get_training_and_testing_sets()


# import re
# pattern = re.compile('[^A-Za-z0-9 ]+')
# bcd = pattern.sub('', "i can't draw mr.")
# print(bcd)
#
# print(''.join([i for i in "き起こそう,とする事(象)が散見されています" if i.isalpha()])) # works for japanese

# with open("./datafiles/standford_raw_clean", mode="a", encoding="utf-8") as file_write:
#     with open("./datafiles/standford_raw", mode="r", encoding="utf-8") as file_read:
#         for line in file_read:
#             eng, jap = line.split("\t")
#             eng = pattern.sub('', eng)
#             jap = ''.join([i for i in jap if i.isalpha()])
#             file_write.write(eng + "\t" + jap + "\n")

