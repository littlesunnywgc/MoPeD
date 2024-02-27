
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import abc
import torch.nn.utils as utils
from sklearn.metrics import classification_report, accuracy_score
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.init as init
import pickle
import json, os
import argparse
import config_file
import random
from PIL import Image
import sys
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
import logging

sys.path.append('/../image_part') #需要引用模块的地址

parser = argparse.ArgumentParser()
parser.description = "ini"
parser.add_argument("-t", "--task", type=str, default="pheme")
parser.add_argument("-g", "--gpu_id", type=str, default="0")
parser.add_argument("-c", "--config_name", type=str, default="single3.json")
parser.add_argument("-T", "--thread_name", type=str, default="Thread-1")
parser.add_argument("-d", "--description", type=str, default="exp_description")
parser.add_argument('--fc5_in_features', type=int, default=3448, help='Number of input features for fc5')
parser.add_argument('--fc4_in_features', type=int, default=2350, help='Number of input features for fc4')
parser.add_argument('--fc3_in_features', type=int, default=1524, help='Number of output features for fc3')
parser.add_argument('--fc2_in_features', type=int, default=650, help='Number of input features for fc2')
parser.add_argument('--fc1_in_features', type=int, default=300, help='Number of input features for fc1')
parser.add_argument('--kl_in_features', type=int, default=410, help='Number of input features for kl')
parser.add_argument('--sigma', type=int, default=1e-6, help='sigma')
args = parser.parse_args()


def process_config(config):
    for k,v in config.items():
        config[k] = v[0]
    return config

class PGD(object):

    def __init__(self, model, emb_name, epsilon=1., alpha=0.3):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]

class TransformerBlock(nn.Module):

    def __init__(self, input_size, d_k=16, d_v=16, n_heads=8, is_layer_norm=False, attn_dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))

        self.W_o = nn.Parameter(torch.Tensor(d_v*n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, episilon=1e-6):
        '''
        :param Q: (*, max_q_words, n_heads, input_size)
        :param K: (*, max_k_words, n_heads, input_size)
        :param V: (*, max_v_words, n_heads, input_size)
        :param episilon:
        :return:
        '''
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        Q_K_score = F.softmax(Q_K, dim=-1)
        Q_K_score = self.dropout(Q_K_score)

        V_att = Q_K_score.bmm(V)
        return V_att


    def multi_head_attention(self, Q, K, V):
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_v)

        V_att = self.scaled_dot_product_attention(Q_, K_, V_)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads*self.d_v)

        output = self.dropout(V_att.matmul(self.W_o))
        return output


    def forward(self, Q, K, V):
        '''
        :param Q: (batch_size, max_q_words, input_size)
        :param K: (batch_size, max_k_words, input_size)
        :param V: (batch_size, max_v_words, input_size)
        :return:  output: (batch_size, max_q_words, input_size)  same size as Q
        '''
        V_att = self.multi_head_attention(Q, K, V)

        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X
        return output

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.best_acc = 0
        self.init_clip_max_norm = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    @abc.abstractmethod
    def forward(self):
        pass

    def moped(self, x_tid, x_text, y, loss, i, total, params,pgd_word,image_features, text_features):
        self.optimizer.zero_grad()
        logit_defense = self.forward(x_tid, x_text,image_features, text_features)
        loss_classification = loss(logit_defense, y)
        loss_defense = loss_classification
        loss_defense.backward()

        K = 3
        pgd_word.backup_grad()
        for t in range(K):
            pgd_word.attack(is_first_attack=(t == 0))
            if t != K - 1:
                self.zero_grad()
            else:
                pgd_word.restore_grad()
            loss_adv= self.forward(x_tid, x_text,image_features, text_features)
            loss_adv = loss(loss_adv,y)
            loss_adv.backward()
        pgd_word.restore()

        self.optimizer.step()
        corrects = (torch.max(logit_defense, 1)[1].view(y.size()).data == y.data).sum()
        accuracy = 100 * corrects / len(y)
        print(
            'Batch[{}/{}] - loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total,
                                                                                                       loss_defense.item(),
                                                                                                       accuracy,
                                                                                                       corrects,
                                                                                                       y.size(0)))

    def fit(self, X_train_tid, X_train, y_train,
            X_dev_tid, X_dev, y_dev,image_features1, text_features1,image_features2, text_features2):

        if torch.cuda.is_available():
            self.cuda()
        batch_size = self.config['batch_size']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-3, weight_decay=0)

        X_train_tid = torch.LongTensor(X_train_tid)
        X_train = torch.LongTensor(X_train)
        y_train = torch.LongTensor(y_train)

        dataset = TensorDataset(X_train_tid, X_train, y_train,image_features1, text_features1)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        loss = nn.CrossEntropyLoss()
        params = [(name, param) for name, param in self.named_parameters()]
        pgd_word = PGD(self, emb_name='word_embedding', epsilon=6, alpha=1.8)

        for epoch in range(self.config['epochs']):
            print("\nEpoch ", epoch + 1, "/", self.config['epochs'])
            self.train()

            for i, data in enumerate(dataloader):
                total = len(dataloader)
                batch_x_tid, batch_x_text, batch_y,batch_image_features, batch_text_features = (item.cuda(device=self.device) for item in data)
                self.moped(batch_x_tid, batch_x_text, batch_y, loss, i, total, params, pgd_word,batch_image_features, batch_text_features)

                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)
            self.evaluate(X_dev_tid, X_dev, y_dev,image_features2, text_features2)

    def evaluate(self, X_dev_tid, X_dev, y_dev,image_features2, text_features2):
        y_pred = self.predict(X_dev_tid, X_dev,image_features2, text_features2)
        acc = accuracy_score(y_dev, y_pred)

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.state_dict(), self.config['save_path'])
            print(classification_report(y_dev, y_pred, target_names=self.config['target_names'], digits=5))
            print("Val set acc:", acc)
            print("Best val set acc:", self.best_acc)
            print("save model!!!   at ", self.config['save_path'])

    def predict(self, X_test_tid, X_test,image_features3, text_features3):
        if torch.cuda.is_available():
            self.cuda()
        self.eval()
        y_pred = []
        X_test_tid = torch.LongTensor(X_test_tid).cuda()
        X_test = torch.LongTensor(X_test).cuda()

        dataset = TensorDataset(X_test_tid, X_test,image_features3, text_features3)
        dataloader = DataLoader(dataset, batch_size=50)

        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_x_tid, batch_x_text,batch_image_features, batch_text_features = (item.cuda(device=self.device) for item in data)
                logits = self.forward(batch_x_tid, batch_x_text,batch_image_features, batch_text_features)
                predicted = torch.max(logits, dim=1)[1]
                y_pred += predicted.data.cpu().numpy().tolist()
        return y_pred

class resnet50():
    def __init__(self):
        self.newid2imgnum = config['newid2imgnum']
        self.model = models.resnet50(pretrained=True).cuda()
        self.model.fc = nn.Linear(2048, 300).cuda()
        torch.nn.init.eye_(self.model.fc.weight)
        self.path = os.path.dirname(os.getcwd()) + '/dataset/pheme/pheme_image/pheme_images_jpg/'
        self.trans = self.img_trans()
    def img_trans(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
        return transform
    def forward(self,xtid):
        img_path = []
        img_list = []
        for newid in xtid.cpu().numpy():
            imgnum = self.newid2imgnum[newid]
            imgpath = self.path + imgnum + '.jpg'
            im = np.array(self.trans(Image.open(imgpath)))
            im = torch.from_numpy(np.expand_dims(im, axis=0)).to(torch.float32)
            img_list.append(im)
        batch_img = torch.cat(img_list, dim=0).cuda()
        img_output = self.model(batch_img)
        return img_output

class Encoder(nn.Module):
    def __init__(self, z_dim=2):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(862, args.kl_in_features),
            nn.ReLU(True),
            nn.Linear(args.kl_in_features, z_dim * 2),
        )

    def forward(self, x):
        params = self.net(x)
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + args.sigma
        return Independent(Normal(loc=mu, scale=sigma), 1)

class KL(nn.Module):
    def __init__(self):
        super(KL, self).__init__()
        self.encoder_text = Encoder()
        self.encoder_image = Encoder()

    def forward(self, text_encoding, image_encoding):
        p_z1_given_text = self.encoder_text(text_encoding)
        p_z2_given_image = self.encoder_image(image_encoding)
        z1 = p_z1_given_text.rsample()
        z2 = p_z2_given_image.rsample()
        kl_1_2 = p_z1_given_text.log_prob(z1) - p_z2_given_image.log_prob(z1)
        kl_2_1 = p_z2_given_image.log_prob(z2) - p_z1_given_text.log_prob(z2)
        skl = (kl_1_2 + kl_2_1)/ 2.
        skl = nn.functional.sigmoid(skl)
        # skl = torch.sigmoid(skl)
        return skl

class MoPeD(NeuralNetwork):
    def __init__(self, config):
        super(MoPeD, self).__init__()
        self.config = config
        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape
        maxlen = config['maxlen']
        dropout_rate = config['dropout']

        self.mh_attention = TransformerBlock(input_size=862, n_heads=8, attn_dropout=0)
        self.word_embedding = nn.Embedding(num_embeddings=V, embedding_dim=D, padding_idx=0,
                                           _weight=torch.from_numpy(embedding_weights))
        self.image_embedding = resnet50()
        self.convs = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(args.fc5_in_features, args.fc4_in_features)  #
        self.fc4 = nn.Linear(args.fc4_in_features, args.fc3_in_features)  #
        self.fc3 = nn.Linear(args.fc3_in_features, args.fc2_in_features)  #
        self.fc2 = nn.Linear(args.fc2_in_features, args.fc1_in_features)
        self.fc1 = nn.Linear(in_features=args.fc1_in_features, out_features=config['num_classes'])
        self.alignfc_t = nn.Linear(in_features=300, out_features=300)
        self.init_weight()
        self.imgmemory = nn.Parameter(torch.zeros(1, 50))
        self.textmemory = nn.Parameter(torch.zeros(1, 50))

        self.h_score = KL()

    def init_weight(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc3.weight)
        init.xavier_normal_(self.fc4.weight)
        init.xavier_normal_(self.fc5.weight)

    def forward(self, X_tid, X_text,image_features, text_features):
        image_features = torch.squeeze(image_features)
        text_features = torch.squeeze(text_features)
        X_text = self.word_embedding(X_text)
        if self.config['user_self_attention'] == True:
            X_text = self.mh_attention(X_text, X_text, X_text)
        X_text = X_text.permute(0, 2, 1)
        iembedding = self.image_embedding.forward(X_tid)
        iembedding = torch.cat([iembedding,image_features],dim=1)
        conv_block = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(X_text))
            pool = max_pooling(act)
            pool = torch.squeeze(pool)
            conv_block.append(pool)

        text_feature = torch.cat(conv_block, dim=1)
        text_feature = torch.cat([text_feature,text_features],dim=1)
        bsz = text_feature.size()[0]
        imgmemory = self.imgmemory.expand(bsz, 50)
        image_feature = torch.cat([iembedding, imgmemory], dim=1)
        textmemory = self.textmemory.expand(bsz, 50)
        text_feature = torch.cat([text_feature, textmemory], dim=1)

        self_att_t = self.mh_attention(text_feature.view(bsz, -1, 862), text_feature.view(bsz, -1, 862), \
                                       text_feature.view(bsz, -1, 862))

        self_att_i = self.mh_attention(image_feature.view(bsz, -1, 862), image_feature.view(bsz, -1, 862), \
                                       image_feature.view(bsz, -1, 862))
        self_i = self_att_i.view(bsz, 862)
        self_t = self_att_t.view(bsz, 862)

        text_enhanced = self.mh_attention(self_att_i.view((bsz, -1, 862)), self_att_t.view((bsz, -1, 862)), \
                                          self_att_t.view((bsz, -1, 862))).view(bsz, 862)

        self_att_t = text_enhanced.view((bsz, -1, 862))

        co_att_ti = self.mh_attention(self_att_t, self_att_i, self_att_i).view(bsz, 862)
        co_att_it = self.mh_attention(self_att_i, self_att_t, self_att_t).view(bsz, 862)

        skl = self.h_score(self_i, self_t)
        w_unimodel = (1 - skl).unsqueeze(1)
        w_mutimodel = skl.unsqueeze(1)
        att_feature = torch.cat((w_unimodel * self_i,  w_mutimodel * co_att_it, w_mutimodel * co_att_ti, w_unimodel * self_t), dim=1)
        a1 = self.relu(self.dropout(self.fc5(att_feature)))
        a1 = self.relu(self.fc4(a1))
        a1 = self.relu(self.dropout(self.fc3(a1)))
        a1 = self.relu(self.fc2(a1))
        d1 = self.dropout(a1)
        output = self.fc1(d1)

        return output

def load_dataset():
    pre = os.path.dirname(os.getcwd()) + '/dataset/pheme/pheme_files/'

    X_train_tid, X_train, y_train, word_embeddings, adj = pickle.load(open(pre + "/train.pkl", 'rb'))
    X_dev_tid, X_dev, y_dev = pickle.load(open(pre + "/dev.pkl", 'rb'))
    X_test_tid, X_test, y_test = pickle.load(open(pre + "/test.pkl", 'rb'))
    config['embedding_weights'] = word_embeddings

    with open(pre + "train_features_label_pheme.pkl", 'rb') as f:
        image_features1, text_features1, y_train = pickle.load(f)
    image_features1 = [x.to('cpu').float() for x in image_features1]
    image_features1 = torch.stack([torch.Tensor(x) for x in image_features1])

    text_features1 = [x.to('cpu').float() for x in text_features1]
    text_features1 = torch.stack([torch.Tensor(x) for x in text_features1])

    with open(pre + "dev_features_label_pheme.pkl", 'rb') as f:
        image_features2, text_features2, y_dev = pickle.load(f)
    image_features2 = [x.to('cpu').float() for x in image_features2]
    image_features2 = torch.stack([torch.Tensor(x) for x in image_features2])

    text_features2 = [x.to('cpu').float() for x in text_features2]
    text_features2 = torch.stack([torch.Tensor(x) for x in text_features2])

    with open(pre + "/test_features_label_pheme.pkl", 'rb') as f:
        image_features3, text_features3, y_test = pickle.load(f)
    image_features3 = [x.to('cpu').float() for x in image_features3]
    image_features3 = torch.stack([torch.Tensor(x) for x in image_features3])

    text_features3 = [x.to('cpu').float() for x in text_features3]
    text_features3 = torch.stack([torch.Tensor(x) for x in text_features3])

    with open(pre + '/new_id_dic.json', 'r') as f:
        newid2mid = json.load(f)
        newid2mid = dict(zip(newid2mid.values(), newid2mid.keys()))
    content_path = os.path.dirname(os.getcwd()) + '/dataset/pheme/'
    with open(content_path + '/content.csv', 'r') as f:
        reader = csv.reader(f)
        result = list(reader)[1:]
        mid2num = {}
        for line in result:
            mid2num[line[1]] = line[0]
    newid2num = {}
    for id in X_train_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    for id in X_dev_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    for id in X_test_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    config['newid2imgnum'] = newid2num

    return X_train_tid, X_train, y_train, \
           X_dev_tid, X_dev, y_dev, \
           X_test_tid, X_test, y_test,\
           image_features1,text_features1,\
           image_features2,text_features2, \
           image_features3, text_features3



def train_and_test(model):
    model_suffix = model.__name__.lower().strip("text")+ '_' +"clip_unimodel_kl"
    res_dir = 'exp_result'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = os.path.join(res_dir, args.task)
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = os.path.join(res_dir, args.description)
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = config['save_path'] = os.path.join(res_dir, 'best_model_in_each_config')
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    config['save_path'] = os.path.join(res_dir, args.thread_name + '_' + 'config' + args.config_name.split(".")[
        0] + '_best_model_weights_' + model_suffix)
    #
    dir_path = os.path.join('exp_result', args.task, args.description)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if os.path.exists(config['save_path']):
        os.system('rm {}'.format(config['save_path']))

    X_train_tid, X_train, y_train, \
    X_dev_tid, X_dev, y_dev, \
    X_test_tid, X_test, y_test, \
    image_features1, text_features1,  \
    image_features2, text_features2,  \
    image_features3, text_features3,  = load_dataset()

    nn = model(config)

    nn.fit(X_train_tid, X_train, y_train,
           X_dev_tid, X_dev, y_dev,image_features1, text_features1,image_features2, text_features2)
    saved_model = torch.load(config['save_path'])
    nn.load_state_dict(saved_model)
    y_pred = nn.predict(X_test_tid, X_test,image_features3, text_features3)
    res = classification_report(y_test, y_pred, target_names=config['target_names'], digits=3, output_dict=True)
    for k, v in res.items():
        print(k, v)
    print("result:{:.4f}".format(res['accuracy']))
    res2={}
    res_final = {}
    res_final.update(res)
    res_final.update(res2)
    return res

config = process_config(config_file.config)

seed = config['seed']
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(seed)

model = MoPeD
train_and_test(model)






