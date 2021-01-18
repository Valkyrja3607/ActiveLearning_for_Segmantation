import torch
import torch.nn as nn
import torch.optim as optim

from IOUEval import iouEval

import os
import numpy as np
from sklearn.metrics import accuracy_score

import sampler
import copy


class Solver:
    def __init__(self, args, test_dataloader, weight):
        self.args = args
        self.test_dataloader = test_dataloader

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.criterion = nn.NLLLoss2d()#ignore_index=0)

        self.sampler = sampler.AdversarySampler(self.args.budget, self.args)
    
    def adjust_learning_rate(self, optimizer, epoch):
        """
        Sets the learning rate to the initial LR decayed by 10 every 30 epochs
        """
        if self.args.lr_mode == 'step':
            lr = self.args.lr * (0.1 ** (epoch // self.args.lr_step))
        elif self.args.lr_mode == 'poly':
            lr = self.args.lr * (1 - epoch / self.args.train_epochs) ** 0.9
        else:
            raise ValueError('Unknown lr mode {}'.format(self.args.lr_mode))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label in dataloader:
                    yield img, label
        else:
            while True:
                for img, _ in dataloader:
                    yield img

    def train(self, querry_dataloader, val_dataloader, task_model, single_model, unlabeled_dataloader,):

        optim_task_model = optim.Adam(single_model.optim_parameters(),
                                self.args.lr,
                                weight_decay=self.args.weight_decay)

        task_model.train()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        task_model = task_model.to(device)
        self.criterion = self.criterion.to(device)
        best_IOU = 0
        best_acc = 0
        for i in range(self.args.train_epochs):
            self.adjust_learning_rate(optim_task_model, i)
            for j, (labeled_imgs, labels, _) in enumerate(querry_dataloader):

                labeled_imgs = labeled_imgs.to(device)
                labels = labels.to(device)

                # task_model step
                preds = task_model(labeled_imgs)[0]
                task_loss = self.criterion(preds, labels)
                optim_task_model.zero_grad()
                task_loss.backward()
                optim_task_model.step()

            print("Current training epochs: {}".format(i))
            print("Current task model loss: {:.4f}".format(task_loss.item()))

            overall_acc, per_class_acc, per_class_iu, mIOU = self.validate(task_model, val_dataloader)
            if mIOU > best_IOU:
                best_IOU = mIOU
                best_model = copy.deepcopy(task_model)
            best_acc = max(best_acc, overall_acc)

            print("current step: {} mIOU: {}".format(i, mIOU))
            print("all acc:", overall_acc)
            print("best IOU: ", best_IOU)

        best_model = best_model.to(device)

        overall_acc, per_class_acc, per_class_iu, final_mIOU = self.test(best_model)
        return final_mIOU, overall_acc, task_model

    def sample_for_labeling(self, task_model, unlabeled_dataloader):
        querry_indices = self.sampler.sample(
            task_model, unlabeled_dataloader, self.args.cuda,
        )

        return querry_indices

    def validate(self, task_model, loader):
        task_model.eval()
        iouEvalVal = iouEval(self.args.classes)

        total, correct = 0, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = task_model(imgs)[0]

            iouEvalVal.addBatch(preds.max(1)[1].detach(), labels.detach())
            overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalVal.getMetric()

        return overall_acc, per_class_acc, per_class_iu, mIOU

    def test(self, task_model):
        task_model.eval()
        iouEvalTest = iouEval(self.args.classes)

        total, correct = 0, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for imgs, labels, _ in self.test_dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = task_model(imgs)[0]

            iouEvalTest.addBatch(preds.max(1)[1].detach(), labels.detach())
            overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalTest.getMetric()

        return overall_acc, per_class_acc, per_class_iu, mIOU

