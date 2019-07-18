import torch
import torch.nn as nn
import torch.optim as optim
import os
from .utils import Logger


class Trainer:
    def __init__(self, model, train_loader, config):
        '''
            config: device, checkpoints_dir, log_dir, print_every_batch, clip
                    learning_rate, learning_rate_decay, learning_rate_decay_step, learning_rate_min
                    teacher_forcing_ratio, teacher_forcing_ratio_decay, teacher_forcing_ratio_decay_step, teacher_forcing_ratio_min
        '''
        self.model = model
        self.train_loader = train_loader
        self.device = config['device']

        self.criterian = nn.NLLLoss(ignore_index=train_loader.vocab.pad_token)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config['learning_rate_decay_step'],
                                                    gamma=config['learning_rate_decay'])
        self.learning_rate_min = config['learning_rate_min']

        self.checkpoints_dir = config['checkpoints_dir']

        logger_config = {
            'owner': 'Trainer.py',
            'log_dir' : config['log_dir'],
            'timezone' : 'Iran',
        }
        self.logger = Logger(logger_config)

        self.clip = config['clip']

        self.checkpoint_name = lambda id: 'snapshot-{:02d}.pt'.format(id)

        self.print_every_batch = config['print_every_batch']
        self.current_epoch = 0

        self.teacher_forcing_ratio = config['teacher_forcing_ratio']
        self.teacher_forcing_ratio_decay = config['teacher_forcing_ratio_decay']
        self.teacher_forcing_ratio_decay_step = config['teacher_forcing_ratio_decay_step']
        self.teacher_forcing_ratio_min = config['teacher_forcing_ratio_min']

    def train_one_epoch(self):
        self.logger('start training one epoch')
        self.model.train()
        epoch_ended = False
        batch_counter = 0
        epoch_loss = 0
        epoch_acc = 0
        predictions = []
        while not epoch_ended:
            x_train, y_train, epoch_ended = self.train_loader.get_next_batch()
            x_train = torch.from_numpy(x_train).float().to(self.device)
            y_train = torch.from_numpy(y_train).long().to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x_train, y_train, self.teacher_forcing_ratio)
            loss = self.criterian(logits[:, 1:, :].contiguous().view(-1, logits.shape[-1]),
                                  y_train[:, 1:].contiguous().view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            mask = (y_train != self.train_loader.vocab.pad_token)
            preds = logits.argmax(dim=2)
            acc = ((y_train == preds) * mask).sum().item() / mask.sum().item()
            if batch_counter % self.print_every_batch == 0:
                self.logger('Batch {}: loss={}, acc={}, lr={}'.format(batch_counter, loss, acc, get_lr(self.optimizer)))
            epoch_loss += loss.item()
            epoch_acc += acc
            for i in range(preds.shape[0]):
                pred = preds[i].cpu().numpy()
                predictions.append(self.train_loader.vocab.tensor2formula(pred, pretty=True))
            batch_counter += 1
        self.logger('Epoch finished, loss={} acc={}, lr={}'.format(epoch_loss/batch_counter, epoch_acc/batch_counter, get_lr(self.optimizer)))
        if get_lr(self.optimizer) > self.learning_rate_min:
            self.scheduler.step()
        if self.current_epoch % self.teacher_forcing_ratio_decay_step == 0:
            self.teacher_forcing_ratio = max(self.teacher_forcing_ratio_min,
                                            self.teacher_forcing_ratio_decay * self.teacher_forcing_ratio)
        self.current_epoch += 1
        self.logger('training one epoch finished.')
        return predictions, epoch_loss/batch_counter, epoch_acc/batch_counter

    def evaluate(self, eval_loader, method):
        self.logger('evaluation starts.')
        self.model.eval()
        start_token = self.train_loader.vocab.start_token
        max_len = self.train_loader.max_len
        predictions = []
        with torch.no_grad():
            epoch_ended = False
            batch_counter = 0
            epoch_loss = 0
            epoch_acc = 0
            while not epoch_ended:
                if eval_loader.has_label:
                    x_eval, y_eval, epoch_ended = eval_loader.get_next_batch()
                    y_eval = torch.from_numpy(y_eval).long().to(self.device)
                else:
                    x_eval, epoch_ended = eval_loader.get_next_batch()
                x_eval = torch.from_numpy(x_eval).float().to(self.device)
                if eval_loader.has_label:
                    max_len = y_eval.shape[1]
                preds, logits, attn_weights = self.model.generate(x_eval, start_token, max_len, method)
                if eval_loader.has_label:
                    epoch_loss += self.criterian(logits[:, 1:, :].contiguous().view(-1, logits.shape[-1]),
                                                 y_eval[:, 1:].contiguous().view(-1))
                    mask = (y_eval != self.train_loader.vocab.pad_token)
                    acc = ((y_eval == preds) * mask).sum().item() / mask.sum().item()
                    epoch_acc += acc
                for i in range(preds.shape[0]):
                    pred = preds[i].cpu().numpy()
                    predictions.append(self.train_loader.vocab.tensor2formula(pred, pretty=True))
                batch_counter += 1
                if batch_counter % self.print_every_batch == 0:
                    self.logger('batch {} completed.'.format(batch_counter))
            self.logger('evaluation finished.')
            if eval_loader.has_label:
                return predictions, attn_weights, epoch_loss/batch_counter, epoch_acc/batch_counter
            else:
                return predictions, attn_weights

    def save(self, epoch, loss, acc):
        if not os.path.isdir(self.checkpoints_dir):
            os.mkdir(self.checkpoints_dir)
        torch.save({
            'epoch' : epoch,
            'acc' : acc,
            'loss' : loss,
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'scheduler_state_dict' : self.scheduler.state_dict(),
        }, os.path.join(self.checkpoints_dir, self.checkpoint_name(epoch)))
        self.logger('Epoch {} saved.'.format(epoch))
        return os.path.join(self.checkpoints_dir, self.checkpoint_name(epoch))

    def load(self, checkpoint_name):
        checkpoint = torch.load(checkpoint_name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.logger('{} loaded.'.format(checkpoint_name))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
