import os

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import math

import logging,sys
from logging.handlers import RotatingFileHandler
import pandas as pd
import numpy as np
import time

'''参数'''
class Config:
    # 数据参数
    train_columns = list(range(1, 15))    # 要作为训练集的列
    feature_columns = list(range(5, 15))
    label_columns = [2, 4]                  # 要预测的列
    feature_in_train_index = (lambda x, y: [x.index(i) for i in y])(train_columns, feature_columns)
    label_in_train_index = (lambda x, y: [x.index(i) for i in y])(train_columns, label_columns)
    labels = ['IR','Tmax','Tavg','Tmin','chargetime','Skewness','Kurtosis','IntegralT','Slope','Intercept']

    predict_cycle = 1             # 预测步数

    # 网络参数
    features = len(feature_columns)
    output_size = len(label_columns)

    time_step = 10               # 数据处理步数
    batch_size = 64
    step_num = 2
    d_ff = 512
    hidden_size = 128
    dropout_rate = 0.0
    max_len = 1000
    relaxation_factor = 2.0
    eps = 1e-7

    # 训练参数
    do_train = True
    do_predict = True
    add_train = False           # 是否载入已有模型参数进行增量训练
    shuffle_train_data = True   # 是否对训练数据做shuffle
    use_cuda = True            # 是否使用GPU训练

    valid_data_rate = 0.2       # 验证数据占训练数据比例，验证集在训练过程使用，为了做模型和参数选择

    learning_rate = 1e-3        # 学习率
    epoch = 500                 # 整个训练集被训练多少遍，不考虑早停的前提下
    patience = 20               # 训练多少epoch，验证集没提升就停掉
    random_seed = 6            # 随机种子，保证可复现

    do_continue_train = False    # 每次训练把上一次的final_state作为下一次的init_state
    continue_flag = ""
    if do_continue_train:
        shuffle_train_data = False
        batch_size = 1
        continue_flag = "continue_"

    # 训练模式
    debug_mode = False  # 调试模式下，是为了跑通代码，追求快
    debug_num = 500  # 仅用debug_num条数据来调试

    # 框架参数
    used_frame = "pytorch"  # 选择的深度学习框架，不同的框架模型保存后缀不一样
    model_postfix = {"pytorch": ".pth", "keras": ".h5", "tensorflow": ".ckpt"}
    model_name = "model_" + continue_flag + used_frame + model_postfix[used_frame]

    # 路径参数
    train_name = "train"
    train_folder = "./DATA/" + train_name + "/"
    test_folder1 = "./DATA/prim_test/"
    test_folder2 = "./DATA/secn_test/"
    file_save_path = "./DATA/result/BLP_MLT_" + train_name + "/"

    i = 1;
    j = 1;
    k = 1;
    train_path = []
    test_path = []
    while i <= 41:
        train_data_path = train_folder + "train_" + str(i) + ".csv"
        train_path.append(train_data_path)
        i += 1
    while j <= 43:
        test_data_path = test_folder1 + "prim_test_" + str(j) + ".csv"
        test_path.append(test_data_path)
        j += 1
    while k <= 40:
        test_data_path = test_folder2 + "secn_test_" + str(k) + ".csv"
        test_path.append(test_data_path)
        k += 1

    cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    file_save_path = file_save_path + str(features) + "BLP_" + str(epoch) + "epoch_" + cur_time + "/"

    model_save_path = file_save_path + "model/"
    figure_save_path = file_save_path + "figure/"
    log_save_path = file_save_path + "log/"
    record_save_path = file_save_path + "record/"

    do_log_print_to_screen = True
    do_log_save_to_file = True                  # 是否将config和训练过程记录到log
    do_figure_save = True
    do_train_visualized = False         # 训练loss可视化，pytorch用visdom，tf用tensorboardX，实际上可以通用, keras没有
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)    # makedirs 递归创建目录
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    if not os.path.exists(record_save_path):
        os.mkdir(record_save_path)
    if do_train and (do_log_save_to_file or do_train_visualized):
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_save_path = log_save_path + cur_time + '_' + used_frame + "/"
        os.makedirs(log_save_path)


'''数据处理'''

class Data:
    def __init__(self, config):
        self.config = config
        self.start_num_in_test = 0      # 测试集中前几天的数据会被删掉，因为它不够一个time_step

    def read_data(self, data_path):                # 读取初始数据
        if self.config.debug_mode:
            init_data = pd.read_csv(data_path, nrows=self.config.debug_num,
                                    usecols=self.config.train_columns)
        else:
            init_data = pd.read_csv(data_path, usecols=self.config.train_columns)
        init_data = init_data.fillna(method='bfill') # 向后填充
        return init_data.values


    def get_train_and_valid_data(self, data_path):
        train_data = self.read_data(data_path=data_path)
        train_num = train_data.shape[0]-config.predict_cycle
        train_norm_data = self.get_norm_data(data=train_data, eps=config.eps)
        feature_data = train_norm_data[:train_num, self.config.feature_in_train_index]
        label_data = train_norm_data[self.config.predict_cycle: self.config.predict_cycle + train_num,
                                                     self.config.label_in_train_index]    # 将延后几天的数据作为label

        # 在非连续训练模式下，每time_step行数据会作为一个样本，两个样本错开一行，比如：1-20行，2-21行。。。。
        train_x = [feature_data[i:i+self.config.time_step] for i in range(train_num-config.time_step + 1)]
        train_y = [label_data[i:i+self.config.time_step] for i in range(train_num-config.time_step + 1)]

        train_x, train_y = np.array(train_x), np.array(train_y)

        self.train_x, self.valid_x, self.train_y, self.valid_y = train_test_split(train_x, train_y,
                                                                                  test_size=self.config.valid_data_rate,
                                                                                  random_state=self.config.random_seed,
                                                                                  shuffle=self.config.shuffle_train_data)  # 划分训练和验证集，并打乱

        return self.train_x, self.valid_x, self.train_y, self.valid_y

    def get_test_data(self, data_path, return_label_data=False):
        test_data = self.read_data(data_path)
        test_num = test_data.shape[0]
        test_norm_data = self.get_norm_data(data=test_data, eps=config.eps)
        feature_data = test_norm_data[:test_num,self.config.feature_in_train_index]
        sample_interval = min(feature_data.shape[0], self.config.time_step)     # 防止time_step大于测试集数量
        self.start_num_in_test = feature_data.shape[0] % sample_interval  # 这些天的数据不够一个sample_interval
        time_step_size = feature_data.shape[0] // sample_interval

        # 在测试数据中，每time_step行数据会作为一个样本，两个样本错开time_step行
        # 比如：1-20行，21-40行。。。到数据末尾。
        test_x = [feature_data[self.start_num_in_test+i*sample_interval : self.start_num_in_test+(i+1)*sample_interval]
                   for i in range(time_step_size)]
        if return_label_data:       # 实际应用中的测试集是没有label数据的
            label_data = test_norm_data[test_num + self.start_num_in_test:, self.config.feature_in_train_index]
            return np.array(test_x), label_data
        return np.array(test_x)

    def get_norm_data(self, data, eps):
        mean = np.mean(data, axis=0)  # 数据的均值和方差
        std = np.std(data, axis=0) + eps
        norm_data = (data - mean) / std  # 归一化，去量纲
        return norm_data

def load_logger(config):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    if config.do_log_print_to_screen:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                      fmt='[ %(asctime)s ] %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # FileHandler
    if config.do_log_save_to_file:
        file_handler = RotatingFileHandler(config.log_save_path + "out.log", maxBytes=1024000, backupCount=5)
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 把config信息也记录到log 文件中
        config_dict = {}
        for key in dir(config):
            if not key.startswith("_"):
                config_dict[key] = getattr(config, key)
        config_str = str(config_dict)
        config_list = config_str[1:-1].split(", '")
        config_save_str = "\nConfig:\n" + "\n'".join(config_list)
        logger.info(config_save_str)

    return logger


'''网络'''
class BLPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.features = config.features
        self.batch_size = config.batch_size
        self.time_step = config.time_step
        self.step_num = config.step_num
        self.relaxation_factor = config.relaxation_factor
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size
        self.eps = config.eps

        self.mlplayer = nn.Linear(config.features, config.hidden_size)
        self.positionlayer = PositionalEncoder(config.hidden_size, config.dropout_rate, config.max_len)
        self.encoder = nn.Sequential(
            EncoderLayer(config.hidden_size, config.d_ff, config.dropout_rate),
            Norm(config.hidden_size)
        )
        self.decoder = nn.Linear(config.hidden_size, config.output_size)

        self.norm_mask = Norm(config.features)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, src):
        M_explain = torch.zeros(src.shape).to(src.device)
        mask = torch.ones(src.shape).to(src.device)
        prior_scale_term = mask

        for i in range(self.step_num):
            mask = self.norm_mask(mask)
            mask = mask * prior_scale_term
            mask = nn.functional.softmax(mask, dim=-1)
            prior_scale_term = (self.relaxation_factor - mask) * prior_scale_term

            feature_out = src * mask;
            src_out = self.mlplayer(feature_out)
            en_in = self.positionlayer(src_out)
            en_out = self.encoder(en_in)
            de_out = self.decoder(en_out)

            step_importance = torch.sum(de_out, dim=2, keepdim=True)
            step_importance = torch.sum(step_importance, dim=1)
            step_importance = (step_importance - torch.min(step_importance)) / (
                        torch.max(step_importance) - torch.min(step_importance) + self.eps)
            M_explain += mask * step_importance.unsqueeze(dim=1)

        return de_out, M_explain

class PositionalEncoder(nn.Module):
    def __init__(self, hidden_size, dropout_rate, max_len):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# build an encoder layer with one multi-head attention layer and one # feed-forward layer
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, d_ff, dropout_rate):
        super().__init__()
        self.norm = Norm(hidden_size=hidden_size)
        self.ff = FeedForward(hidden_size=hidden_size, d_ff=d_ff, dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x_ff = self.ff(x)
        x2 = self.norm(x_ff)
        x = x + self.dropout(x2)

        return x


class Norm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()

        self.hidden_size = hidden_size
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.hidden_size))
        self.bias = nn.Parameter(torch.zeros(self.hidden_size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class FeedForward(nn.Module):
    def __init__(self, hidden_size, d_ff, dropout_rate):
        super().__init__()
        # set d_ff as a default to 512
        self.linear_1 = nn.Linear(hidden_size, d_ff)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(d_ff, hidden_size)

    def forward(self, x):
        x = self.linear_1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


'''还原数据'''
def restore_data(config: Config, data_path, origin_data: Data, logger, predict_norm_data: np.ndarray):
    test_data = origin_data.read_data(data_path)
    test_mean = np.mean(test_data, axis=0)  # 数据的均值和方差
    test_std = np.std(test_data, axis=0) + config.eps
    label_data = test_data[origin_data.start_num_in_test:,
                 config.label_in_train_index]

    predict_data = predict_norm_data * (test_std[config.label_in_train_index] + config.eps) + \
                   test_mean[config.label_in_train_index]  # 通过保存的均值和方差还原数据

    assert label_data.shape[0] == predict_data.shape[0], "The element number in origin and predicted data is different"

    return label_data, predict_data

def record_result(config, origin_data, label_data, predict_data):
    cycle = origin_data.loc[len(origin_data)-len(label_data):,'cycle']
    batteryQ_record = pd.DataFrame(columns=['cycle','true_capacity','true_rul','predict_capacity','predict_rul'])

    batteryQ_record['cycle'] = cycle
    batteryQ_record['true_capacity'] = label_data[:, 0]
    batteryQ_record['true_rul'] = label_data[:, 1]
    batteryQ_record['predict_capacity'] = predict_data[:, 0]
    batteryQ_record['predict_rul'] = predict_data[:, 1]

    return batteryQ_record

def restore_feature_imp(config, feature_imp):
    importance_record = pd.DataFrame(columns=config.labels)

    # importance_record['Qcharge'] = feature_imp[:, 0]
    importance_record['IR'] = feature_imp[:, 0]
    importance_record['Tmax'] = feature_imp[:, 1]
    importance_record['Tavg'] = feature_imp[:, 2]
    importance_record['Tmin'] = feature_imp[:, 3]
    importance_record['chargetime'] = feature_imp[:, 4]

    importance_record['Skewness'] = feature_imp[:, 5]
    importance_record['Kurtosis'] = feature_imp[:, 6]
    importance_record['IntegralT'] = feature_imp[:, 7]
    importance_record['Slope'] = feature_imp[:, 8]
    importance_record['Intercept'] = feature_imp[:, 9]

    return importance_record

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    config = Config()
    for key in dir(args):               # dir(args) 函数获得args所有的属性
        if not key.startswith("_"):     # 去掉 args 自带属性，比如__name__等
            setattr(config, key, getattr(args, key))   # 将属性值赋给Config

    logger = load_logger(config)

    data_gainer = Data(config)

    if config.do_train:
        device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")  # CPU训练还是GPU
        model = BLPModel(config).to(device)  # 如果是GPU训练， .to(device) 会把模型/数据复制到GPU显存中
        if config.add_train:  # 如果是增量训练，会先加载原模型参数
            model.load_state_dict(torch.load(config.model_save_path + config.model_name))
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = torch.nn.MSELoss()  # 这两句是定义优化器和loss

        valid_loss_min = float("inf")
        bad_epoch = 0
        train_loss_all = []
        valid_loss_all = []
        feature_importances = []

        for epoch in range(config.epoch):
            logger.info("Epoch {}/{}".format(epoch, config.epoch))
            res_explain_epoch = []
            train_loss_epoch = []
            valid_loss_epoch = []

            for train_data_path in config.train_path:
                torch.cuda.empty_cache()

                train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data(train_data_path)
                train_X, train_Y = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).float()  # 先转为Tensor
                train_loader = DataLoader(TensorDataset(train_X, train_Y),
                                          batch_size=config.batch_size)  # DataLoader可自动生成可训练的batch数据

                valid_X, valid_Y = torch.from_numpy(valid_X).float(), torch.from_numpy(valid_Y).float()
                valid_loader = DataLoader(TensorDataset(valid_X, valid_Y), batch_size=config.batch_size)

                model.train()  # pytorch中，训练时要转换成训练模式
                train_loss_array = []

                res_explain = torch.zeros(config.batch_size, config.time_step, config.features, dtype=torch.float,
                                          device=device)
                explain0 = torch.zeros(config.batch_size, config.time_step, config.features, dtype=torch.float,
                                       device=device)

                for _train_X, _train_Y in train_loader:
                    torch.cuda.empty_cache()
                    _train_X, _train_Y = _train_X.to(device), _train_Y.to(device)
                    optimizer.zero_grad()  # 训练前要将梯度信息置 0
                    pred_Y, M_explain = model(_train_X)  # 这里走的就是前向计算forward函数

                    loss = criterion(pred_Y, _train_Y)  # 计算loss
                    loss.backward()  # 将loss反向传播
                    optimizer.step()  # 用优化器更新参数
                    train_loss_array.append(loss.item())

                    M_explain = (M_explain - torch.mean(M_explain)) / (torch.std(M_explain) + config.eps)  # normalize
                    if (M_explain.shape[0] < config.batch_size):
                        M_explain = torch.cat((M_explain, explain0), axis=0)
                        M_explain = M_explain[0:config.batch_size, :, :]
                    torch.add(res_explain.detach(), M_explain.detach(), out=res_explain)

                res_explain = res_explain.detach().cpu().numpy()
                res_explain_epoch.append(res_explain)

                # 以下为早停机制，当模型训练连续config.patience个epoch都没有使验证集预测效果提升时，就停止，防止过拟合
                model.eval()  # pytorch中，预测时要转换成预测模式
                valid_loss_array = []
                for _valid_X, _valid_Y in valid_loader:
                    _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
                    pred_Y, _ = model(_valid_X)

                    loss = criterion(pred_Y, _valid_Y)  # 验证过程只有前向计算，无反向传播过程
                    valid_loss_array.append(loss.item())

                train_loss_cur = np.mean(train_loss_array)
                valid_loss_cur = np.mean(valid_loss_array)

                train_loss_epoch.append(train_loss_cur)
                valid_loss_epoch.append(valid_loss_cur)

            # 累计每轮mask
            res_explain_mean = np.mean(res_explain_epoch, axis=0)
            sum_explain = res_explain_mean.sum(axis=0)
            sum_explain = sum_explain.sum(axis=0)
            sum_explain = (sum_explain - np.min(sum_explain)) / (np.max(sum_explain) - np.min(sum_explain) + config.eps)
            sum_explain = (sum_explain / np.sum(sum_explain))  # [1, features]
            feature_importances.append(sum_explain)

            train_loss_mean = np.mean(train_loss_epoch)
            valid_loss_mean = np.mean(valid_loss_epoch)
            train_loss_all.append(train_loss_mean)
            valid_loss_all.append(valid_loss_mean)

            logger.info("The train loss is {:.6f}. ".format(train_loss_mean) +
                        "The valid loss is {:.6f}.".format(valid_loss_mean))

            if valid_loss_mean < valid_loss_min:
                valid_loss_min = valid_loss_mean
                bad_epoch = 0
                torch.save(model.state_dict(), config.model_save_path + config.model_name)  # 模型保存
            else:
                bad_epoch += 1
                if bad_epoch >= config.patience:  # 如果验证集指标连续patience个epoch没有提升，就停掉训练
                    logger.info(" The training stops early in epoch {}".format(epoch))
                    break

        feature_importances = np.array(feature_importances)
        imp_record = restore_feature_imp(config=config, feature_imp=feature_importances)
        imp_record.to_csv(config.record_save_path + "{}_featureImp_record.csv".format(config.train_name))

    # 目前存在问题：测试集中也需要输入测试目标
    if config.do_predict:
        # 加载模型
        device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
        model = BLPModel(config).to(device)
        model.load_state_dict(torch.load(config.model_save_path + config.model_name))  # 加载模型参数

        rul_result = []

        for k, test_data_path in enumerate(config.test_path):
            test_X, test_Y = data_gainer.get_test_data(test_data_path, return_label_data=True)
            # 获取测试数据
            test_X = torch.from_numpy(test_X).float()
            test_set = TensorDataset(test_X)
            test_loader = DataLoader(test_set, batch_size=1)

            # 先定义一个tensor保存预测结果
            result = torch.Tensor().to(device)

             # 预测过程
            model.eval()
            for _data in test_loader:
                data_X = _data[0].to(device)
                pred_X, _ = model(data_X)

                cur_pred = torch.squeeze(pred_X, dim=0)
                result = torch.cat((result, cur_pred), dim=0)

            pred_result = result.detach().cpu().numpy() # 这里输出的是未还原的归一化预测数据

            init_data = pd.read_csv(test_data_path)
            label_data, predict_data = restore_data(config, test_data_path, data_gainer, logger, pred_result)  # 储存预测结果和真实结果
            batteryQ_record = record_result(config, init_data, label_data, predict_data)
            batteryQ_record.to_csv(config.record_save_path + "{}_batteryQ_record.csv".format(k+1))

            true_rul = label_data[:, 1]
            predict_rul = predict_data[:, 1]
            rul = [true_rul[0],predict_rul[0]]
            rul_result.append(rul)

        rul_result = np.array(rul_result)
        rul_record = pd.DataFrame(columns=['true_rul','predict_rul'])
        rul_record['true_rul'] = rul_result[:,0]
        rul_record['predict_rul'] = rul_result[:,1]
        rul_record.to_csv(config.record_save_path + "rul_record.csv")