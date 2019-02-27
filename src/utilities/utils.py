import json
import os

convert_dict = {
    'a':1,
    'an':1,
    'one':1,
    'two':2,
    'three':3,
    'four':4,
    'five':5,
    'six':6,
    'seven':7,
    'eight':8,
    'nine':9,
    'ten':10,
    '1':1,
    '2':2,
    '3':3,
    '4':4,
    '5':5,
    '6':6,
    '7':7,
    '8':8,
    '9':9,
    '10':10
}
reverse_convert_dict = {
    1:'one',
    2:'two',
    3:'three',
    4:'four',
    5:'five',
    6:'six',
    7:'seven',
    8:'eight',
    9:'nine',
    10:'ten',
}

plural_words_for_numbers = ['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
singular_words_for_numbers = ['a', 'an', 'one',]
all_words_for_numbers = singular_words_for_numbers + plural_words_for_numbers
uncertain = ['group', 'some', 'most']

def read_jsonl(filename):
    with open(filename) as f:
        jlines = f.readlines()
    return [json.loads(jline.strip()) for jline in jlines]

def remove_parenthesis(pair):
    assert " " in pair[0] and " " in pair[1], "single-word sentence cannot be used."
    parentheses_table = str.maketrans({'(': None, ')': None})
    return " ".join(pair[0].translate(parentheses_table).split()), \
           " ".join(pair[1].translate(parentheses_table).split())

def return_sent_words_replaced_case_sensitive(sent, word_to_be_replaced, word_to_use):
    words_list = sent.strip().split()
    words_list[words_list.index(word_to_be_replaced)] = word_to_use
    return " ".join(words_list)

def return_pair_words_replaced(pair, word_to_be_replaced, word_to_use, second_word_to_use = None):
    if second_word_to_use == None:
        second_word_to_use = word_to_use
    return return_sent_words_replaced_case_sensitive(pair[0], word_to_be_replaced, word_to_use), \
        return_sent_words_replaced_case_sensitive(pair[1], word_to_be_replaced, second_word_to_use)

def return_num_pairs(more_first=False,include_1=False, bias=0):
    num_list = [2,3,4,5,6,7,8,9,10]
    if include_1:
        num_list.append(1)
    result = []
    for i in num_list:
        for j in num_list:
            if more_first:
                if i > j + bias:
                    result.append((i,j))
            else:
                if i + bias < j:
                    result.append((i,j))
    return result

def return_same_num_pairs(more_first=False,include_1=False, bias=0):
    num_list = [2,3,4,5,6,7,8,9,10]
    if include_1:
        num_list.append(1)
    result = []
    for i in num_list:
        for j in num_list:
            if more_first:
                if i == j + bias:
                    result.append((i,j))
            else:
                if i + bias == j:
                    result.append((i,j))
    return result

def num_pair_to_4_possible_mix(pair):
    n1, n2 = pair
    return (str(n1), str(n2)), (str(n1), reverse_convert_dict[n2]), (reverse_convert_dict[n1], str(n2)), (reverse_convert_dict[n1], reverse_convert_dict[n2])


def same_to_same_plural_number(sentence_pair, num_word_to_replace, bias=0):
    entailment_same_to_same_example = []
    for pair in return_same_num_pairs():
        if pair[0] == convert_dict[num_word_to_replace.lower()]:
            continue
        for possible_pair in num_pair_to_4_possible_mix(pair):
            entailment_same_to_same_example.append(
                (
                    return_sent_words_replaced_case_sensitive(
                         sentence_pair[0], 
                         num_word_to_replace, 
                         possible_pair[0].capitalize() if num_word_to_replace[0].isupper() else possible_pair[0]
                    ),
                    return_sent_words_replaced_case_sensitive(
                         sentence_pair[1], 
                         num_word_to_replace, 
                         possible_pair[1].capitalize() if num_word_to_replace[0].isupper() else possible_pair[1]
                    )
                )
            )
    return entailment_same_to_same_example


def same_to_different_plural_number(sentence_pair, num_word_to_replace, more_in_premise):
    entailment_same_to_different_example = []
    for pair in return_num_pairs(more_in_premise):
        for possible_pair in num_pair_to_4_possible_mix(pair):
            entailment_same_to_different_example.append(
                (
                    return_sent_words_replaced_case_sensitive(
                        sentence_pair[0], 
                        num_word_to_replace, 
                        possible_pair[0].capitalize() if num_word_to_replace[0].isupper() else possible_pair[0]
                    ),
                    return_sent_words_replaced_case_sensitive(
                        sentence_pair[1], 
                        num_word_to_replace, 
                        possible_pair[1].capitalize() if num_word_to_replace[0].isupper() else possible_pair[1]
                    )
                )
            )
    return entailment_same_to_different_example

def same_to_same_plural_number_with_addition(sentence_pair, num_words_to_replace, bias, more_in_premise):
    entailment_addition_same_example = []
    for pair in return_same_num_pairs(more_in_premise,bias=bias):
        if pair[0] == convert_dict[num_words_to_replace[0].lower()] and \
           pair[1] == convert_dict[num_words_to_replace[1].lower()]:
            continue
        for possible_pair in num_pair_to_4_possible_mix(pair):
            entailment_addition_same_example.append(
                (
                    return_sent_words_replaced_case_sensitive(
                        sentence_pair[0], 
                        num_words_to_replace[0], 
                        possible_pair[0].capitalize() if num_words_to_replace[0][0].isupper() else possible_pair[0]
                    ),
                    return_sent_words_replaced_case_sensitive(
                        sentence_pair[1], 
                        num_words_to_replace[1], 
                        possible_pair[1].capitalize() if num_words_to_replace[1][0].isupper() else possible_pair[1]
                    )
                )
            )
    return entailment_addition_same_example

def same_to_different_plural_number_with_addition(sentence_pair, num_words_to_replace, bias, more_in_premise):
    entailment_addition_same_to_different_example = []
    for pair in return_num_pairs(more_in_premise,bias=bias):
        for possible_pair in num_pair_to_4_possible_mix(pair):
            entailment_addition_same_to_different_example.append(
                (
                    return_sent_words_replaced_case_sensitive(
                        sentence_pair[0], 
                        num_words_to_replace[0], 
                        possible_pair[0].capitalize() if num_words_to_replace[0][0].isupper() else possible_pair[0]
                    ),
                    return_sent_words_replaced_case_sensitive(
                        sentence_pair[1], 
                        num_words_to_replace[1], 
                        possible_pair[1].capitalize() if num_words_to_replace[1][0].isupper() else possible_pair[1]
                    )
                )
            )
    return entailment_addition_same_to_different_example

def save_examples_for_ESIM(filename, example_list_list):
    global_pair_index = 0
    with open(filename,'w+') as f:
        for example_list, label in example_list_list:
            for pair in example_list:
                f.write("{}\t{}\t{}\t3\t4\t5\t6\t{}\n".format(label,pair[0],pair[1],global_pair_index))
                global_pair_index += 1

def save_examples_for_InferSent(filename, example_list_list):
    os.makedirs(filename, exist_ok=True)
    with open(filename+"/labels.test",'w+') as f:
        for example_list, label in example_list_list:
            for pair in example_list:
                f.write("{}\n".format(label))
    with open(filename+"/s1.test",'w+') as f:
        for example_list, label in example_list_list:
            for pair in example_list:
                f.write("{}\n".format(remove_parenthesis(pair)[0]))
    with open(filename+"/s2.test",'w+') as f:
        for example_list, label in example_list_list:
            for pair in example_list:
                f.write("{}\n".format(remove_parenthesis(pair)[1]))