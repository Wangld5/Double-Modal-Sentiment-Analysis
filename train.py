import torch
from utils import *
from my_transformers.bert_like import transformer
import os
import sys
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch.nn as nn
import pickle
from tqdm import tqdm, trange
import logging
import random

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

class MultiModalTrainer():
    def __init__(self, hyp_params, logger, model, load_train, load_valid, load_test):
        self.args = hyp_params
        self.logger = logger
        self.load_train_data = load_train
        self.load_valid_data = load_valid
        self.load_test_data = load_test
        self.model = model.to(self.args.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr = self.args.lr)
        self.loss_function = nn.MSELoss()
        vision_dim, text_dim, audio_dim = 35, 300, 74
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        
    def train(self):
        print("####### Start training ######")
        print(f" Num Epochs = {self.args.n_epochs}")
        print(f" Gradient Accumulation steps = {self.args.grad_accum}")
        
        set_seed(self.args)
        best_valid = 1e8
        best_epoch = 0
        attention = []
        for epoch in range(int(self.args.n_epochs)):
            self.train_epoch(epoch)
            valid_loss, results, truths = self.valid_epoch()
            test_loss, _, _, attn_score = self.test(epoch)
            attention.append(attn_score.cpu().numpy())
            print("-"*50)
            print('Epoch {:2d} | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, valid_loss, test_loss))
            eval_dataset(results, truths)
            print("-"*50)

            if valid_loss < best_valid:
                print(f"Saved model at pre_trained_models/{self.args.save_name}.pt!")
                save_model(self.model, name=self.args.save_name)
                best_valid = valid_loss
                best_epoch = epoch
        
        self.model = load_model(self.args.save_name)
        _, results, truths, attn_score = self.test(best_epoch)
        print(f'print the best result: {best_epoch}')
        eval_dataset(results, truths)
        with open('preds_text.pkl', 'wb') as f:
            pickle.dump(results.cpu().numpy(), f)
        with open('truths_text.pkl', 'wb') as w:
            pickle.dump(truths.cpu().numpy(), w)
        # attention.append(attn_score.cpu().numpy())
        # with open('attention.pkl', 'wb') as f:
        #     pickle.dump(attention, f)
    
    def train_epoch(self, epoch):
        self.model.train()
        tr_loss = 0.0
        proc_loss = 0
        proc_size = 0
        for i_batch, (batch_X, batch_Y) in enumerate(self.load_train_data):
            sample_idx, text, audio, vision = batch_X
            ground_truth = batch_Y.squeeze(-1)
            text, audio, vision, ground_truth = text.cuda(), audio.cuda(), vision.cuda(), ground_truth.cuda()

            preds, _, inner_loss = self.model(text, audio, vision)
            loss = self.loss_function(preds, ground_truth)

            if self.args.n_gpu > 1:
                loss = loss.mean() + inner_loss.mean()
            if self.args.grad_accum > 1:
                loss = loss / self.args.grad_accum
            
            loss.backward()
            tr_loss += loss.item()

            if (i_batch + 1) % self.args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.model.zero_grad()
            
            proc_loss += loss.item() * self.args.batch_size
            proc_size += self.args.batch_size
            if i_batch % self.args.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                print(f'Epoch {epoch} | Batch {i_batch} | Train Loss {avg_loss}')
                proc_loss = 0
                proc_size = 0

    
    def valid_epoch(self):
        self.model.eval()
        total_loss = 0.0
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y) in enumerate(self.load_valid_data):
                sample_idx, text, audio, vision = batch_X
                ground_truth = batch_Y.squeeze(-1)
                text, audio, vision, ground_truth = text.cuda(), audio.cuda(), vision.cuda(), ground_truth.cuda()
                batch_size = text.shape[0]

                preds, _, inner_loss = self.model(text, audio, vision)
                total_loss += (inner_loss.mean() + self.loss_function(preds, ground_truth)).item() * batch_size

                results.append(preds)
                truths.append(ground_truth)
        
        avg_loss = total_loss / len(self.load_valid_data)
        results = torch.cat(results)
        truths = torch.cat(truths)

        return avg_loss, results, truths

    def test(self, epoch):
        self.model.eval()
        total_loss = 0.0
        results = []
        truths = []

        with torch.no_grad():
            attention_score = 0
            for i_batch, (batch_X, batch_Y) in enumerate(self.load_test_data):
                sample_idx, text, audio, vision = batch_X
                ground_truth = batch_Y.squeeze(-1)
                text, audio, vision, ground_truth = text.cuda(), audio.cuda(), vision.cuda(), ground_truth.cuda()
                batch_size = text.shape[0]

                preds, attn_score, _ = self.model(text, audio, vision)
                if i_batch == 0:
                    attention_score = attn_score
                total_loss += self.loss_function(preds, ground_truth).item() * batch_size

                results.append(preds)
                truths.append(ground_truth)

        avg_loss = total_loss / len(self.load_test_data)
        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths, attention_score


            


        



