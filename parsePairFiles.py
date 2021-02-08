import spacy

nlp = spacy.load("ja_core_news_sm")

def processJapEnglishPairList(filename) -> list:
    tokens = list()

    with open(filename, mode="r", encoding="utf-8") as file_in:
        listPairs = dict()
        for line in file_in:
            eng, jap = line.split("\t")
            print("input: ", jap)
            sen = nlp(jap)
            for token in sen:
                print(token.text, token.pos_, token.dep_)
                tokens.append(token)



if __name__ == "__main__":
    processJapEnglishPairList("./datafiles/wikipedia_raw")
    # processJapEnglishPairList("./datafiles/stanford_raw")
