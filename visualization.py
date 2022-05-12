# %%
import matplotlib.pyplot as plt
import torchvision.datasets as dataset
import torch 
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.utils.data as data_utils

from cutout import Cutout
import mixup
import cutmix

def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(66)

# %% 
# original graph
transform_train = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])

transform_test = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])

# dataset
train_data = dataset.CIFAR100(root="../datasets/cifar100", train=True, transform=transform_train, download=True)
test_data = dataset.CIFAR100(root="../datasets/cifar100", train=False, transform=transform_test, download=True)

batch_size = 32

train_loader = data_utils.DataLoader(dataset=train_data, batch_size=batch_size,  shuffle=False)
test_loader = data_utils.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

imgs, labels = next(iter(train_loader))
imgs = imgs[0:3]

images = utils.make_grid(imgs)
images = images.numpy().transpose(1, 2, 0)

plt.imshow(images)
plt.show()

# %%
# Normalization 
transform_train = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])

transform_test = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])

# dataset
train_data = dataset.CIFAR100(root="../datasets/cifar100", train=True, transform=transform_train, download=True)
test_data = dataset.CIFAR100(root="../datasets/cifar100", train=False, transform=transform_test, download=True)

batch_size = 32

train_loader = data_utils.DataLoader(dataset=train_data, batch_size=batch_size,  shuffle=False)
test_loader = data_utils.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

imgs, labels = next(iter(train_loader))
imgs = imgs[0:3]

images = utils.make_grid(imgs)
images = images.numpy().transpose(1, 2, 0)

plt.imshow(images)
plt.show()

# %%
cut = Cutout(n_holes=3)
imgs1 = []
for i in range(3):
    out = cut(imgs[i])
    imgs1.append(out)

images = utils.make_grid(imgs1)
images = images.numpy().transpose(1, 2, 0)
plt.imshow(images)
plt.show()

# %%
imgs2 , y_a, y_b, lam = mixup.mixup_data(imgs,labels, use_cuda=False)
images = utils.make_grid(imgs2)
images = images.numpy().transpose(1, 2, 0)
plt.imshow(images)
plt.show()

# %%
imgs3, y_a, y_b, lam = cutmix.cutmix_data(imgs, labels, use_cuda=False)
images = utils.make_grid(imgs3)
images = images.numpy().transpose(1, 2, 0)
plt.imshow(images)
plt.show()
# %%
