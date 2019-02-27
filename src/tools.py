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