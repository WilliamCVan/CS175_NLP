import xml.etree.ElementTree as ET
import os

# https://stackoverflow.com/questions/1912434/how-do-i-parse-xml-in-python
# https://docs.python.org/3/library/xml.etree.elementtree.html

def wikiCreateJapEnglishPairList() -> list:
    listPairs = list()

    for path, subdirs, files in os.walk("./datafiles/wikipedia"):
        for name in files:
            #print(os.path.join(path, name))

            #corpus = ET.parse('./datafiles/wikipedia/BDS/BDS00001.xml').getroot()
            # for child in corpus:
            #     print(child.tag, child.attrib)

            corpus = ET.parse(os.path.join(path, name)).getroot()

            #from the root, get all children that follow this node heirarchy
            print(len(corpus.findall('par/sen')))
            print(len(corpus.findall('sec/par/sen')))

            for type_tag in corpus.findall('sen'):
                value = list(type_tag) # get all children elements under the <sen> tag
                japanese_child = value[0].text # 1st child
                english_child = value[5].text # 6th child

                # build [japanese, english] pairs
                listPairs.append([japanese_child, english_child])
                print()

            return listPairs


if __name__ == "__main__":
    wikiCreateJapEnglishPairList()