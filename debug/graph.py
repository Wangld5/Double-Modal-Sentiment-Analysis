import pickle
import os
import sys
import torch
import matplotlib.pyplot as plt

preds = pickle.load(open('preds_cpu.pkl', 'rb'))
truths = pickle.load(open('truths_cpu.pkl', 'rb'))
preds_text = pickle.load(open('preds_text_only.pkl', 'rb'))
preds_vision = pickle.load(open('preds_vision_only.pkl', 'rb'))
plt.plot(range(preds.size), truths,  c='#90ed7d', label='ground_truths')
plt.plot(range(preds.size), preds, c='#f7a35c', label='fusion_predictions')
plt.plot(range(preds_vision.size), preds_text, c='#8085e7', label='text_predictions')
plt.plot(range(preds_vision.size), preds_vision, c='#f15c80', label='vision_predictions')
plt.xlabel('test account')
plt.ylabel('test accuracy')
plt.legend(loc='best')
plt.show()