# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


class pair:
    def __init__(self, center, context):
        self._center = center
        self._context = context

class ss_pair:
    def __init__(self, center, context, samecount, diffcount):
        self._center = center
        self._context = context
        self._samecount = samecount
        self._diffcount = diffcount


class dataset:
    def __init__(self, pairlist, len):
        self._pairlist = pairlist
        self._len = len
        self._index = 0

class read:


    def read_words(self,filename):
        with open(filename) as f:
            words = tf.compat.as_str(f.read()).split()
        return words

    def build_dict(self, words):
        count = []
        count.extend(collections.Counter(words).most_common())  # all words in
        dictionary = dict()
        count_list = list()
        for word, count in count:
          dictionary[word] = len(dictionary)
          count_list.append(count)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return dictionary, reverse_dictionary, count_list


    def build_dataset(self, filename, centerDict, contextDict):
        with open(filename) as f:
            words = tf.compat.as_str(f.read()).split()

        pairlist = list()
        readIndex = 0
        while readIndex < words.__len__():
            myPair = pair(centerDict[words[readIndex]], contextDict[words[readIndex + 1]])
            pairlist.append(myPair)
            readIndex += 2
        del words
        return pairlist


    def build_book_claim(self, filename, claimDict):
        booklist_claim = list()
        claim_bookdict = dict()
        f_read = open(filename, 'r')
        for line in f_read.readlines():
            line = line.strip()
            line.replace("\r\n","")
            line.replace("\n","")
            words = line.split(" ")
            thisbooklist_claim = list()
            for i in range(0, words.__len__()):
                claimIndex = claimDict[words[i]]
                claim_bookdict[claimIndex] = booklist_claim.__len__()
                thisbooklist_claim.append(claimIndex)
            booklist_claim.append(thisbooklist_claim)

        return booklist_claim, claim_bookdict

    def build_source_claim(self, pairlist_cs):
        source_claim_dict = dict()
        claim_source_dict = dict()
        for pair in pairlist_cs:
            if source_claim_dict.has_key(pair._center):
                source_claim_dict[pair._center].append(pair._context)
            else:
                source_claim_dict[pair._center] = list()
                source_claim_dict[pair._center].append(pair._context)

            if claim_source_dict.has_key(pair._context):
                claim_source_dict[pair._context].append(pair._center)
            else:
                claim_source_dict[pair._context] = list()
                claim_source_dict[pair._context].append(pair._center)

        return source_claim_dict, claim_source_dict

    def build_sslist(self, claim_source_dict):
        sslist = list()
        for claimIndex in claim_source_dict:
            if claim_source_dict[claimIndex].__len__() == 1:
                continue
            elif claim_source_dict[claimIndex].__len__() == 2:
                continue
            else:
                for center_sourceIndex in claim_source_dict[claimIndex]:
                    thislist = list()
                    thislist.append(center_sourceIndex)
                    for context_sourceIndex in claim_source_dict[claimIndex]:
                        if context_sourceIndex == center_sourceIndex:
                            continue
                        thislist.append(context_sourceIndex)
                    sslist.append(thislist)
        return sslist


    def build_sspair_list(self, claim_source_dict, booklist, writefile):
        sspair_dict = dict()
        sum_same = 0
        sum_diff = 0
        for claimlist in booklist:
            # claimlist:一个book的所有claim
            for i in range(0, claimlist.__len__()):
                claimIndex = claimlist[i]
                sourcelist = claim_source_dict[claimIndex]
                if sourcelist.__len__() == 1:
                    continue
                for j in range(0, sourcelist.__len__()):
                    center_sourceIndex = sourcelist[j]
                    for k in range(j+1, sourcelist.__len__()):
                        context_sourceIndex = sourcelist[k]
                        pairname1 = str(center_sourceIndex) + "_" + str(context_sourceIndex)
                        pairname2 = str(context_sourceIndex) + "_" + str(center_sourceIndex)
                        if sspair_dict.has_key(pairname1):
                            sspair_dict[pairname1]._samecount += 1
                        elif sspair_dict.has_key(pairname2):
                            sspair_dict[pairname2]._samecount += 1
                        else:
                            sspair = ss_pair(center=center_sourceIndex, context=context_sourceIndex, samecount=1, diffcount=0)
                            sspair_dict[pairname1] = sspair
                        sum_same += 1
            for i in range(0, claimlist.__len__()):
                claimIndex = claimlist[i]
                sourcelist = claim_source_dict[claimIndex]
                othersourcelist = list()
                for j in range(i+1, claimlist.__len__()):
                    othersourcelist.extend(claim_source_dict[claimlist[j]])
                for center_sourceIndex in sourcelist:
                    for context_sourceIndex in othersourcelist:
                        pairname1 = str(center_sourceIndex) + "_" + str(context_sourceIndex)
                        pairname2 = str(context_sourceIndex) + "_" + str(center_sourceIndex)
                        if sspair_dict.has_key(pairname1):
                            sspair_dict[pairname1]._diffcount += 1
                        elif sspair_dict.has_key(pairname2):
                            sspair_dict[pairname2]._diffcount += 1
                        else:
                            sspair = ss_pair(center=center_sourceIndex, context=context_sourceIndex, samecount=0, diffcount=1)
                            sspair_dict[pairname1] = sspair
                        sum_diff += 1

        f_write = open(writefile,"w")
        for key in sspair_dict:
            f_write.write(str(key))
            f_write.write(": ")
            f_write.write(str(sspair_dict[key]._samecount))
            f_write.write(" ")
            f_write.write(str(sspair_dict[key]._diffcount))
            f_write.write("\n")
        f_write.close()

        pairlist_ss = list()
        pairlist_ss.extend(sspair_dict.values())
        print("same", sum_same)
        print("diff", sum_diff)
        return pairlist_ss











