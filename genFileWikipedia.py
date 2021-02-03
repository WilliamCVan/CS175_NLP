import xml.etree.ElementTree as ET
import os

# https://stackoverflow.com/questions/1912434/how-do-i-parse-xml-in-python
# https://docs.python.org/3/library/xml.etree.elementtree.html

def wikiCreateJapEnglishPairList() -> list:
    listPairs = list()

    for path, subdirs, files in os.walk("./datafiles/wikipedia"):
        # if(path not in {"./datafiles/wikipedia\\CLT"}): # only include "culture" category
        #     continue

        try:
            for name in files:
                #print(os.path.join(path, name))

                #corpus = ET.parse('./datafiles/wikipedia/BDS/BDS00001.xml').getroot()
                # for child in corpus:
                #     print(child.tag, child.attrib)

                corpus = ET.parse(os.path.join(path, name)).getroot()

                #from the root, get all children that follow this node heirarchy, two different types of text blocks so two different for loops
                print(len(corpus.findall('par/sen')))
                print(len(corpus.findall('sec/par/sen')))

                for type_tag in corpus.findall('par/sen'):
                    value = list(type_tag) # get all children elements under the <sen> tag
                    japanese_child = value[0].text # 1st child
                    english_child = value[5].text # 6th child

                    # build [japanese, english] pairs
                    listPairs.append([english_child, japanese_child])


                for type_tag in corpus.findall('sec/par/sen'):
                    value = list(type_tag) # get all children elements under the <sen> tag
                    japanese_child = value[0].text # 1st child
                    english_child = value[5].text # 6th child

                    # build [japanese, english] pairs
                    listPairs.append([english_child, japanese_child])


            # write pairs out to file using <tab> as delimiter to match "standford_raw" dataset
            with open("./datafiles/wikipedia_raw", mode="a", encoding="utf-8") as file_in:
                for eng, jap in listPairs:
                    file_in.write(eng + "\t" + jap + "\n")
        except:
            pass


if __name__ == "__main__":
    wikiCreateJapEnglishPairList()