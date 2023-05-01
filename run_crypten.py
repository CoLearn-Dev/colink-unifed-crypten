import torch
import torch.nn.functional as F
import torch.nn as nn
import crypten
from crypten import mpc
from crypten.config import cfg
import flbenchmark.datasets
import flbenchmark.logging
import time
import json
import sys
import os
from sklearn.metrics import roc_auc_score

config = json.load(open(sys.argv[1], 'r'))

def compute_accuracy(output, labels):
    pred = output.argmax(1)
    correct = pred.eq(labels)
    correct_count = correct.sum(0, keepdim=True).float()
    accuracy = correct_count.mul_(1.0 / output.size(0))
    return accuracy.item()


def train(config):
    flbd = flbenchmark.datasets.FLBDatasets('./data')

    train_dataset, test_dataset = flbd.fateDatasets(config['dataset'])

    if config['dataset'] == 'breast_vertical':
        input_size = 30
        output_size = 2
        type = 'classification'
    elif config['dataset'] == 'motor_vertical':
        input_size = 11
        output_size = 1
        type = 'regression'
    elif config['dataset'] == 'default_credit_vertical':
        input_size = 23
        output_size = 2
        type = 'classification'
    elif config['dataset'] == 'dvisits_vertical':
        input_size = 12
        output_size = 1
        type = 'regression'
    elif config['dataset'] == 'give_credit_vertical':
        input_size = 10
        output_size = 2
        type = 'classification'
    elif config['dataset'] == 'student_vertical':
        input_size = 13
        output_size = 1
        type = 'regression'
    elif config['dataset'] == 'vehicle_scale_vertical':
        input_size = 18
        output_size = 4
        type = 'classification'


    if config['model'].startswith('mlp_'):
        sp = config['model'].split('_')
        if len(sp) == 2:
            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
                    self.fc1 = nn.Linear(input_size, int(sp[1]))
                    self.fc2 = nn.Linear(int(sp[1]), output_size)

                def forward(self, x):
                    out = self.fc1(x)
                    out = F.relu(out)
                    out = self.fc2(out)
                    return out
        elif len(sp) == 3:
            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
                    self.fc1 = nn.Linear(input_size, int(sp[1]))
                    self.fc2 = nn.Linear(int(sp[1]), int(sp[2]))
                    self.fc3 = nn.Linear(int(sp[2]), output_size)

                def forward(self, x):
                    out = self.fc1(x)
                    out = F.relu(out)
                    out = self.fc2(out)
                    out = F.relu(out)
                    out = self.fc3(out)
                    return out
        elif len(sp) == 4:
            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
                    self.fc1 = nn.Linear(input_size, int(sp[1]))
                    self.fc2 = nn.Linear(int(sp[1]), int(sp[2]))
                    self.fc3 = nn.Linear(int(sp[2]), int(sp[3]))
                    self.fc4 = nn.Linear(int(sp[3]), output_size)

                def forward(self, x):
                    out = self.fc1(x)
                    out = F.relu(out)
                    out = self.fc2(out)
                    out = F.relu(out)
                    out = self.fc3(out)
                    out = F.relu(out)
                    out = self.fc4(out)
                    return out
    elif config['model'] == 'logistic_regression':
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.linear = torch.nn.Linear(input_size, output_size)

            def forward(self, x):
                out = self.linear(x)
                out = F.sigmoid(out)
                return out
    elif config['model'] == 'linear_regression':
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.linear = torch.nn.Linear(input_size, output_size)

            def forward(self, x):
                out = self.linear(x)
                return out
    else:
        raise NotImplementedError('Model {} is not supported.'.format(config['model']))

    if config['training']['loss_func'] == 'cross_entropy':
        loss = crypten.nn.CrossEntropyLoss()
    elif config['training']['loss_func'] == 'mse_loss':
        loss = crypten.nn.MSELoss()
    elif config['training']['loss_func'] == 'l1_loss':
        loss = crypten.nn.L1Loss()
    else:
        raise NotImplementedError('Loss function {} is not supported.'.format(config['training']['loss_func']))

    if config['algorithm'] != 'mpc':
        raise NotImplementedError('Algorithm {} is not supported.'.format(config['algorithm']))

    if config['training']['optimizer'] != 'sgd':
        raise NotImplementedError('Optimizer {} is not supported.'.format(config['training']['optimizer']))

    crypten.common.serial.register_safe_class(Net)


    crypten.init()
    logger = flbenchmark.logging.Logger(id=crypten.comm.get().get_rank(), agent_type="client")
    with logger.preprocess_data():
        dataset = train_dataset
        alice_index = dataset.parties[0].column_name.index(dataset.options['unique_id'])
        alice_len = len(dataset.parties[0].column_name)
        if dataset.label_name in dataset.parties[0].column_name:
            label_src = 0
            label_index = dataset.parties[0].column_name.index(dataset.label_name)
            label = torch.tensor(dataset.parties[0].records).index_select(1, torch.tensor([label_index]))
            index = []
            for i in range(alice_len):
                if i != alice_index and i != label_index:
                    index.append(i)
            alice_feature = torch.tensor(dataset.parties[0].records).index_select(1, torch.tensor(index))
        else:
            index = list(range(0, alice_index))+list(range(alice_index+1, alice_len))
            alice_feature = torch.tensor(dataset.parties[0].records).index_select(1, torch.tensor(index))
        bob_index = dataset.parties[1].column_name.index(dataset.options['unique_id'])
        bob_len = len(dataset.parties[1].column_name)
        if dataset.label_name in dataset.parties[1].column_name:
            label_src = 1
            label_index = dataset.parties[1].column_name.index(dataset.label_name)
            label = torch.tensor(dataset.parties[1].records).index_select(1, torch.tensor([label_index]))
            index = []
            for i in range(bob_len):
                if i != bob_index and i != label_index:
                    index.append(i)
            bob_feature = torch.tensor(dataset.parties[1].records).index_select(1, torch.tensor(index))
        else:
            index = list(range(0, bob_index))+list(range(bob_index+1, bob_len))
            bob_feature = torch.tensor(dataset.parties[1].records).index_select(1, torch.tensor(index))

        # check data alignment
        alice_id = torch.tensor(dataset.parties[0].records).index_select(1, torch.tensor([alice_index])).long()
        bob_id = torch.tensor(dataset.parties[1].records).index_select(1, torch.tensor([bob_index])).long()
        assert torch.equal(alice_id, bob_id)

    with logger.training():
        # if config['bench_param']['device'] == 'gpu':
        #     torch.cuda.set_device(crypten.comm.get().get_rank())
        #     device = torch.device('cuda')
        # else:
        #     device = torch.device('cpu')
        device = torch.device('cpu')
        start_time = time.perf_counter()
        crypten.init()
        cfg.communicator.verbose = True
        alice_feature = alice_feature.to(device)
        bob_feature = bob_feature.to(device)
        label = label.to(device)
        alice_feature = crypten.cryptensor(alice_feature, src=0)
        bob_feature = crypten.cryptensor(bob_feature, src=1)
        feature = crypten.cat([alice_feature, bob_feature], dim=1)
        if type == 'classification':
            label = label.long().squeeze()
            label_eye = torch.eye(output_size, device=device)
            label_one_hot = label_eye[label]
            label_train = crypten.cryptensor(label_one_hot, src=label_src)
        elif type == 'regression':
            label_train = crypten.cryptensor(label, src=label_src)

        model_plaintext = Net()
        dummy_input = torch.empty(1, 1, input_size)
        model = crypten.nn.from_pytorch(model_plaintext, dummy_input)
        model.to(device)
        model.encrypt()
        model.train()

        optimizer = crypten.optim.SGD(model.parameters(), lr=config['training']['learning_rate'], **config['training']['optimizer_param'])
        num_epochs = config['training']['epochs']
        batch_size = config['training']['batch_size']
        num_samples = feature.size(0)
        if batch_size == -1:
            batch_size = num_samples
        for i in range(num_epochs):
            with logger.training_round() as t:
                t.report_metric('client_num', 2)
                with logger.computation() as c:
                    for j in range(0, num_samples, batch_size):
                        start, end = j, min(j+batch_size, num_samples)
                        x_train = feature[start:end]
                        y_train = label_train[start:end]
                        output = model(x_train)
                        loss_value = loss(output, y_train)
                        model.zero_grad()
                        loss_value.backward()
                        optimizer.step()
                    # c.report_metric('loss', loss_value.get_plain_text())

        end_time = time.perf_counter()

    with logger.model_evaluation() as e:
        output = model(feature)
        loss_value = loss(output, label_train).get_plain_text()
        if type == 'classification':
            output_plain = output.get_plain_text()
            if config['dataset'] == 'vehicle_scale_vertical':
                accuracy = compute_accuracy(output_plain, label)
                e.report_metric('accuracy', accuracy)
                crypten.print("Epoch: {:d} Loss: {:.6f} Acc: {:.4f}".format(i+1, loss_value, accuracy))
            else:
                auc = roc_auc_score(label.cpu(), output_plain[:, 1].cpu())
                e.report_metric('auc', auc)
                crypten.print("Epoch: {:d} Loss: {:.6f} AUC: {:.4f}".format(i+1, loss_value, auc))
        elif type == 'regression':
            e.report_metric('loss', loss_value.item())
            crypten.print("Epoch: {:d} Loss: {:.6f}".format(i+1, loss_value))
    logger.end()

    print("Time: {0:.4f}".format(end_time-start_time))
    communication_stats = crypten.comm.get().get_communication_stats()
    print(communication_stats)

if __name__ == '__main__':
    train(config)
