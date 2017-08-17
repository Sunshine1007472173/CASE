# encoding: utf-8
# precision:


class evaluation:


    def evaluate(self, readpath, resultpath, lasting):
        # read
        truthfile = readpath + "myTruth"  #../../data1/
        predictlabelfile = resultpath + "label_predict.txt" #../../data1/un_pesudo/
        # write
        fliewith_truthandpredict = resultpath + "truth_predict_cl.txt"
        evaluationfile = resultpath + "evaluation_cl.txt"   #../../data1/un_pesudo/


        # groundTruth
        f_read = open(truthfile,'r')
        index_answer_dict = dict()
        for line in f_read.readlines():
            line = line.strip()
            index_answer_dict[int(line.split(" ")[0])] = 1
        f_read.close()

        # predictTruth
        index_predict_dict = dict()
        f_read = open(predictlabelfile,'r')
        for line in f_read.readlines():
            line = line.strip()
            index_predict_dict[int(line.split(" ")[0])] = int(line.split(" ")[1])
        f_read.close()

        truePositive = 0.0
        trueNegative = 0.0
        falsePositive = 0.0
        falseNegative = 0.0


        f_write = open(fliewith_truthandpredict, 'w')
        for claimIndex in index_answer_dict:
            if index_answer_dict[claimIndex] == 1:
                if index_predict_dict[claimIndex] == 1:
                    truePositive += 1
                    # print claimIndex
                else:
                    falseNegative += 1
            else:   #  index_answer_dict[claimIndex] = 0:
                if index_predict_dict[claimIndex] == 1:
                    falsePositive += 1
                else:
                    trueNegative += 1
            f_write.write(str(claimIndex))
            f_write.write(" ")
            f_write.write(str(index_answer_dict[claimIndex]))
            f_write.write(" ")
            f_write.write(str(index_predict_dict[claimIndex]))
            f_write.write("\n")

        f_write.close()


        f_write = open(evaluationfile,'w')
        # precision = truePositive / (truePositive + falsePositive)
        recall = truePositive / (truePositive + falseNegative)
        # accuracy = (truePositive + trueNegative)/(truePositive + trueNegative + falsePositive + falseNegative)
        # F1 = 2*precision*recall/(precision+recall)

        print "tp: " + str(truePositive)
        print "fp: " + str(falsePositive)
        print "tn: " + str(trueNegative)
        print "fn: " + str(falseNegative)
        # print "precision: " + str(precision)
        print "recall: " + str(recall)
        # print "accuracy: " + str(accuracy)
        # print "F1: " + str(F1)
        print "lastime: " + str(lasting)
        f_write.write("tp: " + str(truePositive))
        f_write.write("\n")
        f_write.write("fp: " + str(falsePositive))
        f_write.write("\n")
        f_write.write("tn: " + str(trueNegative))
        f_write.write("\n")
        f_write.write("fn: " + str(falseNegative))
        f_write.write("\n")
        # f_write.write("precision: " + str(precision))
        # f_write.write("\n")
        f_write.write("recall: " + str(recall))
        f_write.write("\n")
        # f_write.write("accuracy: " + str(accuracy))
        # f_write.write("\n")
        # f_write.write("F1: " + str(F1))
        # f_write.write("\n")
        f_write.write("lasttime: " + str(lasting))
        f_write.close()


