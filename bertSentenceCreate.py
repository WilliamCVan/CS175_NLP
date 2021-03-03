lines = open('./datafiles/standford_train4.tsv', encoding='utf-8').\
    read().strip().split('\n')
for lin in lines:
    eng, jap = lin.split("\t")
    writerENG = open('./datafiles/eng_train.sample', mode='a', encoding='utf-8')
    writerENG.write(eng + "\n")

    writerJAP = open('./datafiles/jap_train.sample', mode='a', encoding='utf-8')
    writerJAP.write(jap + "\n")



lines = open('./datafiles/standford_test4.tsv', encoding='utf-8').\
    read().strip().split('\n')
for lin in lines:
    eng, jap = lin.split("\t")
    writerENG = open('./datafiles/eng_test.sample', mode='a', encoding='utf-8')
    writerENG.write(eng + "\n")

    writerJAP = open('./datafiles/jap_test.sample', mode='a', encoding='utf-8')
    writerJAP.write(jap + "\n")