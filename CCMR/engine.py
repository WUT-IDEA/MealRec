import math
import os
import torch
from tensorboardX import SummaryWriter
import datetime
from utils import use_optimizer, save_checkpoint


class Engine(object):
    """Meta Engine for training & evaluating model
    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self.nowtime = datetime.datetime.now()
        self.log_dir = './runs_3/' + self.config['alias'].format(self.config['model_name'],
                                                                 self.config['embed_shape'],
                                                                 self.config['meal_batch_size'],
                                                                 self.config['adam_lr']) \
                       + '_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self._writer = SummaryWriter(log_dir=self.log_dir)

        self.opt = use_optimizer(self.model, config)

        self.crit = self.BPRLoss

        self.meal_best_hit = 0
        self.meal_best_ndcg = 0

        self.batch_num_meal = 0
        self.meal_bpr_loss = 0
        self.record_b_num = 20

    def BPRLoss(self, up, un):
        return - torch.mean(torch.log(torch.sigmoid(up - un) + 1e-8))

    def train_single_batch(self, users, pos, neg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        if self.config['use_cuda'] is True:
            users, pos, neg = users.cuda(), pos.cuda(), neg.cuda()
        self.opt.zero_grad()
        pos_pred = self.model(users, pos)
        neg_pred = self.model(users, neg)
        loss = self.crit(pos_pred, neg_pred)
        loss.backward()
        self.opt.step()

        self.meal_bpr_loss += loss
        self.batch_num_meal += 1
        if self.batch_num_meal % self.record_b_num == 0:
            self._writer.add_scalar('meal_bpr_loss', self.meal_bpr_loss / self.record_b_num,
                                    global_step=int(self.batch_num_meal / self.record_b_num))
            self.meal_bpr_loss = 0
        return loss

    def train_an_epoch(self, train_meal_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()

        total_meal_loss = 0
        total_batch = 0
        b_n_batch_loss = 0
        for batch_id, batch in enumerate(train_meal_loader):
            batch_id += 1
            assert isinstance(batch[0], torch.LongTensor)
            user, pos, neg = batch[0], batch[1], batch[2]
            loss = self.train_single_batch(user, pos, neg)

            b_n_batch_loss += loss
            if batch_id % 200 == 0:
                print(
                    '[Training Epoch {}] Batch {}, meal Loss {}'.format(epoch_id, batch_id, b_n_batch_loss / 200))
                b_n_batch_loss = 0
            total_meal_loss += loss
            total_batch += 1

    def evaluate(self, evaluate_data, epoch_id, K=5):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        K2 = 2 * K

        with torch.no_grad():
            user, pos, negs = evaluate_data
            negs = negs.view(negs.size(0), 3, -1)
            if self.config['use_cuda'] is True:
                user = user.cuda()
                pos = pos.cuda()
                negs = negs.cuda()
            hit = 0
            ndcg = 0
            hit2 = 0
            ndcg2 = 0

            for i in range(len(user)):
                pos_val = self.model(user[i], pos[i])
                rank = 1  # 初始排名是第一
                for n in negs[i]:
                    neg_val = self.model(user[i].repeat(33), n)  # 批处理，每次33个
                    for n_v in neg_val:
                        if n_v >= pos_val:
                            rank += 1
                    if rank > K2:
                        break
                if rank <= K:
                    hit += 1
                    ndcg += math.log(2) / math.log(1 + rank)
                if rank <= K2:
                    hit2 += 1
                    ndcg2 += math.log(2) / math.log(1 + rank)

        HR_K, NDCG_K = (hit / len(user)), (ndcg / len(user))
        HR_2K, NDCG_2K = (hit2 / len(user)), (ndcg2 / len(user))
        print('[Evluating Epoch {}\n'.format(epoch_id))
        print('HR@K = {:.4f}, NDCG@K = {:.4f}\n'.format(HR_K, NDCG_K))
        print('HR@2K = {:.4f}, NDCG@2K = {:.4f}\n'.format(HR_2K, NDCG_2K))
        self._writer.add_scalar('HR@K', HR_K, global_step=epoch_id)
        self._writer.add_scalar('NDCG@K', NDCG_K, global_step=epoch_id)
        self._writer.add_scalar('HR@2K', HR_2K, global_step=epoch_id)
        self._writer.add_scalar('NDCG@2K', NDCG_2K, global_step=epoch_id)
        if self.config['save_checkpoint']:
            if HR_K > (self.meal_best_hit - 0.01) or (
                    HR_K == self.meal_best_hit and NDCG_K > self.meal_best_ndcg):
                self.save(epoch_id, HR_K, NDCG_K)
                if HR_K > self.meal_best_hit:
                    self.meal_best_hit = HR_K
                    self.meal_best_ndcg = NDCG_K
        return HR_K, NDCG_K

    def save(self, epoch_id, HR_K, NDCG_K):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        path = self.config['model_dir'] \
               + '/' + self.config['alias'].format(self.config['model_name'],
                                                   self.config['embed_shape'],
                                                   self.config['meal_batch_size'],
                                                   self.config['adam_lr']) \
               + '_' + self.nowtime.strftime('%Y-%m-%d-%H-%M-%S')
        if not os.path.exists(path):
            os.mkdir(path)
        model_dir = path + '/' + 'Epoch{}_HR@K{:.4f}_NDCG@K{:.4f}.model'.format(epoch_id, HR_K, NDCG_K)
        save_checkpoint(self.model, model_dir)
