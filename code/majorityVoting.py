# encoding: utf-8
from __future__ import division
from __future__ import print_function
import numpy as np
from operator import itemgetter, attrgetter
import time
import random

class majorityV:

    class majority_claim:
        def __init__(self, claim, count, sum=10):
            self._claim = claim
            self._count = count
            self._sum = sum

    def readcs(self, cs_file):
        f_read = open(cs_file, 'r')
        self.claim_count_dict = dict()
        for line in f_read.readlines():
            line = line.strip()
            words = line.split(" ")
            count = 0
            while count < words.__len__():
                source = int(words[count])
                claim = int(words[count + 1])
                if self.claim_count_dict.has_key(claim):
                    self.claim_count_dict[claim] += 1
                else:
                    self.claim_count_dict[claim] = 1
                count += 2
        f_read.close()

    def getTopClaims(self, path, claimDict, size):
        self.readcs(path + "cs")
        f_read = open(path + "set_claim", 'r')
        allTrueClaim_list = list()
        for line in f_read.readlines():
            line = line.strip()
            claimarray = line.split(" ")
            majority_claim_list = list()
            sum = 0.0
            for i in range(0, claimarray.__len__()):
                claim = int ( claimarray[i] )
                sum += self.claim_count_dict[claim]

            for i in range(0, claimarray.__len__()):
                claim = int ( claimarray[i] )
                my_majority_claim = self.majority_claim(claim, self.claim_count_dict[claim]/sum)
                majority_claim_list.append(my_majority_claim)

            random.shuffle(majority_claim_list)
            sorted_majority_claim_list = sorted(majority_claim_list, key = attrgetter("_count"), reverse = True)
            allTrueClaim_list.append(sorted_majority_claim_list[0])
        f_read.close()
        sorted_allTrueClaim_list = sorted(allTrueClaim_list, key = attrgetter("_count"), reverse = True)
        return_claimindex_list = list()
        print_list = list()
        for i in range(0,size):
            return_claimindex_list.append(claimDict[str(sorted_allTrueClaim_list[i]._claim)])
            print_list.append(str(sorted_allTrueClaim_list[i]._claim))
        print(print_list)
        return return_claimindex_list








# mymajorityV = majorityV()
# mymajorityV.readcs("../test/cs")
# mymajorityV.readset("../test/set_claim")