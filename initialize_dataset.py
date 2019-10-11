import csv

trainSetName = 'my_train2.txt'
testSetName = 'my_test2.txt'


def reduce_data(train_input, test_input):
    train_data = list()
    train_label = list()
    test_data = list()
    test_label = list()
    # train_set, test_set = extractDataSet()
    # for train in train_set:
    #     if train[1][0] in my_labels:
    #         if (train[1][0] == "grain") and ('corn' in train[1]):
    #             train_label.append("corn")
    #             train_data.append(train[0])
    #         elif (train[1][0] == "earn") or (train[1][0] == "acq") or (train[1][0] == "crude"):
    #             train_label.append(train[1][0])
    #             train_data.append(train[0])
    #
    # for test in test_set:
    #     if test[1][0] in my_labels:
    #         if (test[1][0] == "grain") and ('corn' in test[1]):
    #             test_label.append("corn")
    #             test_data.append(test[0])
    #         elif (test[1][0] == "earn") or (test[1][0] == "acq") or (test[1][0] == "crude"):
    #             test_label.append(test[1][0])
    #             test_data.append(test[0])
    with open(train_input) as f:
        reader = csv.reader(f)
        for ind,row in enumerate(reader):
            if ind!=0:
                train_label.append(row[1])
                train_data.append(row[0])
    f.close()
    with open(test_input) as f:
        reader = csv.reader(f)
        for ind, row in enumerate(reader):
            if ind != 0:
                test_label.append(row[1])
                test_data.append(row[0])
    f.close()

    return train_data, train_label, test_data, test_label

