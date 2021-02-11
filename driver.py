import util
from encDecoderGRU import EncoderRNN, DecoderRNN

if __name__ == "__main__":
    util.tokenizeJapanese("./datafiles/wikipedia_raw")
    util.genVectors("./datafiles/wikipedia_raw")

