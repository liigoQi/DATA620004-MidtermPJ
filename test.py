import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np 
import argparse
# torch 
import torch 
import torch.utils.data as data_utils
# CIFAR100
import torchvision.datasets as dataset
import torchvision.transforms as transforms

def set_seed(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(66)

parser = argparse.ArgumentParser()
parser.add_argument('--data_aug', type=str, default='None', help='Choose data augment methods from cutout, mixup, cutmix and None. (default: None)')

args = parser.parse_args()
data_aug = args.data_aug

batch_size = 32 

# pre-process
transform_test = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# dataset
test_data = dataset.CIFAR100(root="../datasets/cifar100", train=False, transform=transform_test, download=True)

test_loader = data_utils.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# network 
net = torch.load('model/resnet_model-' + data_aug + '.pkl')
net.to(device)

test_correct = 0 
test_total = 0

with torch.no_grad():
    for k, (x_test, y_test) in enumerate(test_loader):
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        outputs = net(x_test)
        _, predicted = torch.max(outputs.data, 1)
        number_batch = y_test.size()[0]
        correct_batch = (predicted == y_test).sum().item()
        test_correct += correct_batch
        test_total += number_batch

print('Test set accuracy: {:.4f}%'.format(test_correct / test_total * 100))

torch.cuda.empty_cache()

