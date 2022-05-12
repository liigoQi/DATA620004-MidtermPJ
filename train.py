import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np 
import time as time
import argparse
# torch 
import torch 
import torch.nn as nn 
import torch.utils.data as data_utils
# CIFAR100
import torchvision.datasets as dataset
import torchvision.transforms as transforms
# resnet
from torchvision.models import resnet34
# data augmentation 
from cutout import Cutout
import mixup
import cutmix
# tensorboard 
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(66)

print('PyTorch version:', torch.__version__)
print('If CUDA:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('CuDNN version:', torch.backends.cudnn.version())

parser = argparse.ArgumentParser()
parser.add_argument('--data_aug', type=str, default='None', help='Choose data augment methods from cutout, mixup, cutmix and None. (default: None)')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train. (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate. (default: 0.01)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size. (default: 32)')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay. (default: 0.0001)')
parser.add_argument('--tuning', type=int, default=0, help='Parameters adjusting mode. 0 for False and 1 for True. (default: 0)')

args = parser.parse_args()
data_aug, epochs, lr, batch_size, weight_decay, if_tuning = args.data_aug, args.epochs, args.lr, args.batch_size, args.weight_decay, args.tuning

# pre-process
transform_train = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_test = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

if data_aug == 'cutout':
    transform_train.transforms.append(Cutout(length=8))

# dataset
train_data = dataset.CIFAR100(root="../datasets/cifar100", train=True, transform=transform_train, download=True)
test_data = dataset.CIFAR100(root="../datasets/cifar100", train=False, transform=transform_test, download=True)

if if_tuning:
    # use subset to find good parameters
    train_mask = list(range(len(train_data)))
    np.random.shuffle(train_mask)
    train_mask = train_mask[:100 * batch_size]
    test_mask = list(range(len(test_data)))
    np.random.shuffle(test_mask)
    test_mask = test_mask[:20 * batch_size]

    train_data = data_utils.Subset(train_data, train_mask)
    test_data = data_utils.Subset(test_data, test_mask)

train_loader = data_utils.DataLoader(dataset=train_data, batch_size=batch_size,  shuffle=True)
test_loader = data_utils.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# network 
net = resnet34(num_classes=100)
net.to(device)

optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

accuracy_batch = 0.0

train_start = time.time()
print('train start at {}'.format(train_start))

train_writer = SummaryWriter('runs/train-' + data_aug)
test_writer = SummaryWriter('runs/test-' + data_aug)

for i in range(0, epochs):
    # training
    train_correct = 0
    train_total = 0
    test_correct = 0 
    test_total = 0

    epoch_start = time.time()
    for j, (x_train, y_train) in enumerate(train_loader):
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        optimizer.zero_grad()
        if data_aug == 'None' or data_aug == 'cutout':
            y_pred = net(x_train)
            train_loss = loss_fn(y_pred, y_train)
        elif data_aug == 'mixup':
            x_train, y_a, y_b, lam = mixup.mixup_data(x_train, y_train)
            y_pred = net(x_train)
            train_loss = mixup.mixup_criterion(loss_fn, y_pred, y_a, y_b, lam)
        elif data_aug == 'cutmix':
            r = np.random.rand(1)
            if r < 0.5:
                x_train, y_a, y_b, lam = cutmix.cutmix_data(x_train, y_train)
                y_pred = net(x_train)
                train_loss = cutmix.cutmix_criterion(loss_fn, y_pred, y_a, y_b, lam)
            else:
                y_pred = net(x_train)
                train_loss = loss_fn(y_pred, y_train)

        train_loss.backward()
        optimizer.step()

        # train batch accuracy
        number_batch = y_train.size()[0]
        _, predicted = torch.max(y_pred.data, dim=1)
        correct_batch = (predicted == y_train).sum().item()
        accuracy_batch = 100 * correct_batch / number_batch
        train_correct += correct_batch
        train_total += number_batch

        train_writer.add_scalar('train_loss', train_loss.item(), global_step=int(i * len(train_data) / batch_size) + j)
        train_writer.add_scalar('train_acc', accuracy_batch, global_step=int(i * len(train_data) / batch_size) + j)

        if ((j + 1) % 100 == 0):
            print('train: epoch: {} | batch: {} | loss: {:.4f} | accuracy: {:.4f}%'.format(i + 1, j + 1, train_loss.item(), accuracy_batch))

    # evaluation 
    with torch.no_grad():
        for k, (x_test, y_test) in enumerate(test_loader):
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            outputs = net(x_test)
            test_loss = loss_fn(outputs, y_test)

            _, predicted = torch.max(outputs.data, 1)
            number_batch = y_test.size()[0]
            correct_batch = (predicted == y_test).sum().item()
            test_correct += correct_batch
            test_total += number_batch

            # test batch accuracy 
            accuracy_dataset = 100 * correct_batch / number_batch
            test_writer.add_scalar('test_loss', test_loss.item(), global_step=int(i * len(test_data) / batch_size) + k)
            test_writer.add_scalar('test_acc', accuracy_dataset, global_step=int(i * len(test_data) / batch_size) + k)

            if ((k + 1) % 100 == 0):
                print('test: epoch {} | batch {} | loss: {:.4f} | accuracy: {:.4f}%'.format(i + 1, k + 1, test_loss.item(), accuracy_dataset))
    
    epoch_end = time.time()
    epoch_cost = epoch_end - epoch_start
    print('epoch {} cost {}s '.format(i, epoch_cost))
    print('Training set accuracy: {:.4f}%'.format(train_correct / train_total * 100))
    print('Test set accuracy: {:.4f}%'.format(test_correct / test_total * 100))

    torch.cuda.empty_cache()

train_writer.close()
test_writer.close()

# save model
torch.save(net, 'model/resnet_model-' + data_aug + '.pkl')
torch.save(net.state_dict(), 'model/resnet_state-' + data_aug + '.pkl')

