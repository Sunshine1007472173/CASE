# encoding: utf-8
import numpy as np
import math
from operator import attrgetter
from read import read
from read import pair
from majorityVoting import majorityV

class un_generate_label:
    class claim_with_labelDistance:
        def __init__(self, claim, truedistance, falsedistance):
            self.claim = claim
            self.truedistance = truedistance
            self.falsedistance = falsedistance

    def generate(self, readpath, resultpath, top_count):

        set_claim_file = readpath + "set_claim"
        claim_vectorMatrix = np.loadtxt(resultpath + "final_embeddings_claim.txt")#../../data1/un_pesudo/
        # write
        predictlabelfile = resultpath + "label_predict.txt"


        myRead = read()
        filename_claim = readpath + "claim"
        words_claim = myRead.read_words(filename_claim)
        dict_claim, reverse_dict_claim, count_list_claim = myRead.build_dict(words_claim)
        del words_claim

        f_read = open(set_claim_file,'r')
        booklist = list()
        for line in f_read.readlines():
            line = line.strip()
            booklist.append(line.split(" "))
        f_read.close()


        true_vector = np.zeros([claim_vectorMatrix.shape[1]], dtype=np.float32)
        true_count = 0

        # from majorityVoting
        my_majorityV = majorityV()
        mvClaimIndexList = my_majorityV.getTopClaims(readpath, dict_claim, int(top_count))
        for mv_claimIndex in mvClaimIndexList:
            true_vector += claim_vectorMatrix[mv_claimIndex]
            true_count += 1
        true_vector /= true_count


        book_distance_list = list()
        for book in booklist:
            thislist = list()
            for index in range(0,book.__len__()):
                claim_dictindex = book[index]
                claimindex = dict_claim[claim_dictindex]
                claim_vector = claim_vectorMatrix[claimindex]
                sim = math.exp(np.dot(claim_vector, true_vector))
                claim = self.claim_with_labelDistance(claim_dictindex, sim, 0)
                thislist.append(claim)
            thislist_sort = sorted(thislist, key = attrgetter("truedistance"), reverse=True)  # truedistance越大越好, truedistance是负的
            book_distance_list.append(thislist_sort)


        f_write = open(predictlabelfile, 'w')
        for book in book_distance_list:
            f_write.write(str(book[0].claim))
            f_write.write(" ")
            f_write.write("1")
            f_write.write(" ")
            f_write.write(str(book[0].truedistance))
            f_write.write(" ")
            f_write.write(str(book[0].falsedistance))
            f_write.write("\n")
            for index in range(1,book.__len__()):
                f_write.write(str(book[index].claim))
                f_write.write(" ")
                f_write.write("0")
                f_write.write(" ")
                f_write.write(str(book[index].truedistance))
                f_write.write(" ")
                f_write.write(str(book[index].falsedistance))
                f_write.write("\n")
        f_write.close()