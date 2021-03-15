import fugashi, nltk, pickle

tagger = fugashi.Tagger()

SOS_token = "<SOS>"
EOS_token = "<EOS>"


def genPairs():
    pairs = list()

    with open("./datafiles/standford_raw_clean", mode="r", encoding="utf-8") as file_in:
        for line in file_in:
            jap = [SOS_token] + [word.surface for word in tagger(line.split("\t")[1])]
            eng = [SOS_token] + nltk.word_tokenize(line.split("\t")[0])
            jap.append(EOS_token)
            eng.append(EOS_token)
            pairs.append((jap, eng))

    return pairs


if __name__ == "__main__":
    pairs = genPairs()

    with open("pairs.pickle", 'wb') as outfile:
        pickle.dump(pairs, outfile)

    for i in range(10):
        print("JAP: ", pairs[i][0], "\nENG: ", pairs[i][1])
