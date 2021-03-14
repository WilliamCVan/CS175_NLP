import fugashi
import gensim
import nltk
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

SOS_token = "SOS"
EOS_token = "EOS"

# pip install fugashi[unidic-lite]
# pip install nltk

# generates sentences for gensim's word2vec encoder
# uses fugashi to tokenize japanese text
class weebCorpus:
    filename = "";
    tagger = fugashi.Tagger()
    language = "japanese"
    
    def __init__(self, f, l):
        self.filename = f
        self.language = l
    
    def __iter__(self):
        tokenized_text = list()
        for line in open(self.filename, mode="r", encoding="utf-8"):
            if self.language == "japanese":
                text = line.split("\t")[1]
                tokenized_text = [SOS_token] + [word.surface for word in self.tagger(text)] + [EOS_token]
            else: # english
                text = line.split("\t")[0]
                tokenized_text = [SOS_token] + nltk.word_tokenize(text)
                tokenized_text.append(EOS_token)
            yield tokenized_text

def genVectors(corpus, language = "japanese"):
        sentences = weebCorpus(corpus, language)
        model = Word2Vec(sentences=sentences,
                         size=50,
                         window=5,
                         workers=4,
                         min_count=1,
                         iter=5)
        
        model.save("./" + language + "vectors.model")
        model.wv.save_word2vec_format("./" + language + "vectors_readable.txt", binary=False)
            
if __name__ == "__main__": 
    genVectors("./datafiles/standford_raw", language="japanese")
    genVectors("./datafiles/standford_raw", language="english")
    print("most similar to 時代: ")
    w1 = ["時代"]
    model = gensim.models.Word2Vec.load("japanesevectors.model")
    for thing in model.wv.most_similar(positive=w1,topn=6):
        print (thing[0])
    
        
    print("most similar to rice: ")
    w1 = ["rice"]
    model = gensim.models.Word2Vec.load("englishvectors.model")
    for thing in model.wv.most_similar(positive=w1,topn=6):
        print (thing[0])