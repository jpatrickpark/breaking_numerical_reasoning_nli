import os
import pickle as pkl

def save_log(log, label):
    """
    @log: (loss_list, val_acc_list)
    """
    if os.path.exists("results/log.pkl"):
        log_dict = pkl.load(open("results/log.pkl", "rb"))
    else:
        log_dict = {}
    log_dict[label] = log
    pkl.dump(log_dict, open("results/log.pkl", "wb"))
    print("log saved as: ", label)
    return 0

def word2int(word):
    try:
        return True, int(word)
    except:
        return False, 0

def text2int(textnum, numwords={}):
    textnum = textnum.lower()
    if not numwords:
        units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
        ]

        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

        scales = ["hundred", "thousand", "million", "billion", "trillion"]

        numwords["and"] = (1, 0) # scale, increment
        for idx, word in enumerate(units):  numwords[word] = (1, idx)
        for idx, word in enumerate(tens):       numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales): numwords[word] = (10 ** (idx * 3 or 2), 0)

    ordinal_words = {'first':1, 'second':2, 'third':3, 'fifth':5, 'eighth':8, 'ninth':9, 'twelfth':12}
    # ordinal_endings = [('ieth', 'y'), ('th', '')]

    textnum = textnum.replace('-', ' ')

    current = result = 0
    curstring = ""
    onnumber = False
    prev_a = False
    for word in textnum.split():
        if word in ordinal_words:
            curstring += word + " "
            result = current = 0
            onnumber = False
            # scale, increment = (1, ordinal_words[word])
            # current = current * scale + increment
            # if scale > 100:
            #     result += current
            #     current = 0
            # onnumber = True
        else:
            # for ending, replacement in ordinal_endings:
            #     if word.endswith(ending):
            #         word = "%s%s" % (word[:-len(ending)], replacement)

            if word not in numwords:
                isNum, val = word2int(word)
                if isNum:
                    current += val
                    onnumber = True
                    continue
                if word == "a":
                    prev_a = True
                    continue
                if onnumber:
                    prev_a = False
                    curstring += repr(result + current) + " "
                curstring += ("a" + " ") * prev_a + word + " "
                result = current = 0
                onnumber = False
                prev_a = False
            else:
                if word == "and" and not onnumber:
                    curstring += word + " "
                    continue
                scale, increment = numwords[word]

                if current == 0 and scale > 1:
                    current = 1
                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
                onnumber = True

    if onnumber:
        curstring += repr(result + current)

    return curstring