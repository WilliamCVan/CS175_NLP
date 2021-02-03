def processJapEnglishPairList() -> list:
    listPairs = list()

    with open("./datafiles/standford_raw", mode="r", encoding="utf-8") as file_in:
        listPairs = list()
        for line in file_in:
            eng, jap = line.split("\t")
            listPairs.append([jap, eng])
    # add tokenization/preprocessing logic


    with open("./datafiles/wikipedia_raw", mode="r", encoding="utf-8") as file_in:
        listPairs = list()
        for line in file_in:
            eng, jap = line.split("\t")
            listPairs.append([jap, eng])

    # add tokenization/preprocessing logic


if __name__ == "__main__":
    processJapEnglishPairList()
