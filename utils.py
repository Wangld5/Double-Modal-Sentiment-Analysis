import torch
import os
from dataset import Pred_Dataset
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
import logging

logger = logging.getLogger("evaluation+positionencoding: ")
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler('text.txt')
handler.setLevel(logging.INFO)
logger.addHandler(handler)

def get_data(args, dataset, split_type):
    data_path = os.path.join(args.data_path, dataset.split('.')[0]) + f'_{split_type}.dt'
    if not os.path.exists(data_path):
        print(f'Creating new {split_type} data')
        data = Pred_Dataset(args.data_path, dataset, split_type)
        torch.save(data, data_path)
    else:
        print(f'Found cached {split_type} data')
        data = torch.load(data_path)
    return data

def save_model(model, name=''):
    torch.save(model, f'pre_trained_models/{name}.pt')

def load_model(name=''):
    model = torch.load(f'pre_trained_models/{name}.pt')
    return model
    
def eval_dataset(results, truths):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

    mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    logger.info(f"MAE: {mae}")
    logger.info(f"Correlation Coefficient: {corr}")
    logger.info(f"F1 score: {f_score}")
    logger.info(f"Accuracy: {accuracy_score(binary_truth, binary_preds)}")

    logger.info("-" * 50)
