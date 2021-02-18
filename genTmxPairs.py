import xml.etree.ElementTree as ET
import xml
import re
import html

def wikiCreateDEEnglishPairList() -> list:
    # write pairs out to file using <tab> as delimiter to match "standford_raw" dataset
    # fwrite = open("./datafiles/train_de-eng.tsv", "a", encoding="utf-8")
    # pattern = re.compile('[^A-Za-z0-9 ]+')
    #
    # with open("./datafiles/000_779.de.txt", mode="r", encoding="utf-8") as file_de:
    #     with open("./datafiles/000_779.en.txt", mode="r", encoding="utf-8") as file_en:

    fwrite = open("./datafiles/train_de-eng.tsv", "a", encoding="utf-8")
    pattern = re.compile('[^A-Za-z0-9 ]+')

    with open("./datafiles/002_792.de.txt", mode="r", encoding="utf-8") as file_de:
        with open("./datafiles/002_792.en.txt", mode="r", encoding="utf-8") as file_en:
            german = file_de.readlines()
            english = file_en.readlines()
            for idx, ger in enumerate(german):
                fwrite.write(pattern.sub('', english[idx]).replace(" apos", "").replace("  ", " ") + "\t" + ger.replace(" .", "").replace(" &quot;", "").replace(" ,", "").replace(" ?", "").replace("  ", " ").replace(" : &quot;", ""))


if __name__ == "__main__":
    wikiCreateDEEnglishPairList()