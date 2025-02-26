import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
import torch.nn as nn
import os
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import cv2
import numpy as np


from torchvision.utils import save_image



def generate_softmasked_inputs2(inputs_normal, inputs_g, inputs):

    diff = torch.abs(inputs_normal - inputs_g)  # (batch, 3, H, W)

    diff_per_image = diff.view(diff.size(0), -1)  # (batch, 3 * H * W)
    min_val, _ = diff_per_image.min(dim=1, keepdim=True)  # (batch, 1)
    max_val, _ = diff_per_image.max(dim=1, keepdim=True)  # (batch, 1)
    diff_normalized = (diff - min_val.view(-1, 1, 1, 1)) / (max_val - min_val + 1e-8).view(-1, 1, 1, 1)  # (batch, 3, H, W)

    softmask = diff_normalized.mean(dim=1, keepdim=True)  # (batch, 1, H, W)

    batch_size = inputs.size(0)
    shuffle_indices = list(range(batch_size))
    random.shuffle(shuffle_indices)

    shuffled_inputs_normal = inputs_normal[shuffle_indices]  # (batch, 3, H, W)

    new_inputs = inputs * softmask + shuffled_inputs_normal * (1 - softmask)  # (batch, 3, H, W)


    return new_inputs, shuffled_inputs_normal



def generate_softmasked_inputs(inputs_normal, inputs_g, inputs):
    diff = torch.abs(inputs_normal - inputs_g)  # (batch, 3, H, W)

    diff_per_image = diff.view(diff.size(0), -1)  # (batch, 3 * H * W)
    min_val, _ = diff_per_image.min(dim=1, keepdim=True)  # (batch, 1)
    max_val, _ = diff_per_image.max(dim=1, keepdim=True)  # (batch, 1)
    diff_normalized = (diff - min_val.view(-1, 1, 1, 1)) / (max_val - min_val + 1e-8).view(-1, 1, 1, 1)  # (batch, 3, H, W)


    softmask = diff_normalized.mean(dim=1, keepdim=True)  # (batch, 1, H, W)


    batch_size = inputs.size(0)
    shuffle_indices = list(range(batch_size))
    random.shuffle(shuffle_indices)

    shuffled_inputs_normal = inputs_normal[shuffle_indices]  # (batch, 3, H, W)

    new_inputs = inputs * softmask + shuffled_inputs_normal * (1 - softmask)  # (batch, 3, H, W)


    new_inputs_min = new_inputs.view(batch_size, -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    new_inputs_max = new_inputs.view(batch_size, -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    new_inputs = 2 * (new_inputs - new_inputs_min) / (new_inputs_max - new_inputs_min + 1e-8) - 1

    return new_inputs, shuffled_inputs_normal


def mix_inputs(inputs_aug, inputs_ori, prob):
    inputs_aug = inputs_aug.clone().detach()
    inputs_ori = inputs_ori.clone().detach()
    assert 0 <= prob <= 1, "Probability must be between 0 and 1."
    assert inputs_aug.shape == inputs_ori.shape, "Inputs must have the same shape."

    batch_size = inputs_aug.size(0)

    random_probs = torch.rand(batch_size, device=inputs_aug.device)


    mask = random_probs < prob


    mixed_inputs = torch.clone(inputs_ori)
    mixed_inputs[mask] = inputs_aug[mask]

    return mixed_inputs


def save_tensors_as_grid(inputs_aug, inputs_normal, inputs_g, inputs, output_path="grid_image.png"):
    batch_size, _, H, W = inputs_aug.shape


    grid = torch.cat((inputs_aug, inputs_normal, inputs_g, inputs), dim=3)  


    grid = (grid + 1) / 2.0


    save_image(grid, output_path, nrow=1)

def generate_masked_inputs(inputs_normal, inputs_g, inputs, threshold):

    diff = torch.abs(inputs_normal - inputs_g)  # (batch, 3, H, W)


    diff_per_image = diff.view(diff.size(0), -1)  # (batch, 3 * H * W)
    min_val, _ = diff_per_image.min(dim=1, keepdim=True)  # (batch, 1)
    max_val, _ = diff_per_image.max(dim=1, keepdim=True)  # (batch, 1)
    diff_normalized = (diff - min_val.view(-1, 1, 1, 1)) / (max_val - min_val + 1e-8).view(-1, 1, 1, 1)  # (batch, 3, H, W)


    mask = (diff_normalized.mean(dim=1, keepdim=True) > threshold).float()  # (batch, 1, H, W)


    batch_size = inputs.size(0)
    new_inputs = inputs.clone()
    shuffle_indices = list(range(batch_size))
    random.shuffle(shuffle_indices)

    for i in range(batch_size):

        new_inputs[i] = inputs[i] * mask[i] + inputs_normal[shuffle_indices[i]] * (1 - mask[i])

    return new_inputs

def color_difference_loss(img1, img2):

    img1_hsv = rgb_to_hsv(img1)
    img2_hsv = rgb_to_hsv(img2)


    hue1 = img1_hsv[:, 0, :, :]  
    hue2 = img2_hsv[:, 0, :, :]  


    loss = torch.mean((hue1 - hue2) ** 2)
    return loss

def rgb_to_hsv(rgb):

    max_rgb, _ = torch.max(rgb, dim=1, keepdim=True)  # [batch_size, 1, height, width]
    min_rgb, _ = torch.min(rgb, dim=1, keepdim=True)  # [batch_size, 1, height, width]
    delta = max_rgb - min_rgb  # [batch_size, 1, height, width]

  
    h = (max_rgb - rgb) / (delta + 1e-6) 

    mask = (delta == 0).float()  # [batch_size, 1, height, width]

    mask = mask.expand_as(h)  # [batch_size, 3, height, width]

    h = h * (1 - mask)  # [batch_size, 3, height, width]
    s = delta / (max_rgb + 1e-6)
    v = max_rgb

    return torch.cat([h, s, v], dim=1)  



import torch

def gram_matrix(x):
    batch_size, channels, height, width = x.size()
    features = x.view(batch_size, channels, height * width)
    G = torch.bmm(features, features.transpose(1, 2)) 
    G = G.div(channels * height * width)
    return G

def style_loss(img1, img2):
    G1 = gram_matrix(img1)
    G2 = gram_matrix(img2)
    
    loss = torch.mean((G1 - G2) ** 2)
    return loss


def calculate_loss(l1, l2, sigma1, sigma2):
    loss = (1 / sigma1**2) * l1 + (1 / sigma2**2) * l2 + 2 * torch.log(sigma1) + 2 * torch.log(sigma2)
    return loss

class LossBalancer:
    def __init__(self, init_sigma1=1.0, init_sigma2=1.0, device='cpu'):
        self.device = device
        self.sigma1 = torch.nn.Parameter(torch.tensor(init_sigma1, requires_grad=True, device=self.device))
        self.sigma2 = torch.nn.Parameter(torch.tensor(init_sigma2, requires_grad=True, device=self.device))

    def compute_loss(self, l1, l2):
        return calculate_loss(l1, l2, self.sigma1, self.sigma2)
    
class MultiUncertaintyWeightedLoss(nn.Module):
    def __init__(self, initial_sigma1=1.0, initial_sigma2=1.0, initial_sigma3=1.0):
        super(MultiUncertaintyWeightedLoss, self).__init__()
        self.log_sigma1 = nn.Parameter(torch.log(torch.tensor(initial_sigma1, dtype=torch.float32)))
        self.log_sigma2 = nn.Parameter(torch.log(torch.tensor(initial_sigma2, dtype=torch.float32)))
        self.log_sigma3 = nn.Parameter(torch.log(torch.tensor(initial_sigma3, dtype=torch.float32)))
    
    def forward(self, 
                output_1_g, output_2_g, output_3_g, output_concat_g, targets, 
                map1_g, map1_ori, map2_g, map2_ori, map3_g, map3_ori, 
                inputs_g, inputs_normal, 
                CELoss, reg_loss):

        L1 = CELoss(output_1_g, targets) + CELoss(output_2_g, targets) + CELoss(output_3_g, targets) + CELoss(output_concat_g, targets)
        L2 = reg_loss(map1_g, map1_ori) + reg_loss(map2_g, map2_ori) + reg_loss(map3_g, map3_ori)
        L3 = reg_loss(inputs_g, inputs_normal)
        

        sigma1 = torch.exp(self.log_sigma1)
        sigma2 = torch.exp(self.log_sigma2)
        sigma3 = torch.exp(self.log_sigma3)
        
        loss = (1 / (sigma1 ** 2)) * L1 + (1 / (sigma2 ** 2)) * L2 + (1 / (sigma3 ** 2)) * L3 + \
               2 * self.log_sigma1 + 2 * self.log_sigma2 + 2 * self.log_sigma3
        
        return loss


class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, initial_sigma1=1.0, initial_sigma2=1.0):
        super(UncertaintyWeightedLoss, self).__init__()

        self.log_sigma1 = nn.Parameter(torch.log(torch.tensor(initial_sigma1, dtype=torch.float32)))
        self.log_sigma2 = nn.Parameter(torch.log(torch.tensor(initial_sigma2, dtype=torch.float32)))
    
    def forward(self, output_1_g, output_2_g, output_3_g, output_concat_g, targets, inputs_g, inputs_normal, CELoss, reg_loss):

        L1 = CELoss(output_1_g, targets) + CELoss(output_2_g, targets) + CELoss(output_3_g, targets) + CELoss(output_concat_g, targets)
        L2 = reg_loss(inputs_g, inputs_normal)
        
        sigma1 = torch.exp(self.log_sigma1)
        sigma2 = torch.exp(self.log_sigma2)
        loss = (1 / (sigma1 ** 2)) * L1 + (1 / (sigma2 ** 2)) * L2 + 2 * self.log_sigma1 + 2 * self.log_sigma2
        
        return loss


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class GaussianBlur(object):
    def __init__(self, kernel_size=5, sigma=1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = F.gaussian_blur(img, self.kernel_size, self.sigma)
        return img
    

def apply_blur_with_mask(inputs, masks, kernel_size=15, sigma=5):
    """
    Apply heavy Gaussian blur to the regions of `inputs` corresponding to `1 - masks`, 
    while keeping the regions corresponding to `masks` unchanged.
    
    Args:
        inputs (torch.Tensor): Input images, shape (N, C, H, W).
        masks (torch.Tensor): Binary masks, shape (N, 1, H, W), with values 0 or 1.
        kernel_size (int): Size of the Gaussian kernel.
        sigma (float): Standard deviation of the Gaussian kernel.
    
    Returns:
        torch.Tensor: Processed images, same shape as `inputs`.
    """
    # Create a 1D Gaussian kernel
    x = torch.arange(kernel_size) - kernel_size // 2
    gauss_kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    gauss_kernel_1d = gauss_kernel_1d / gauss_kernel_1d.sum()  # Normalize the kernel

    # Expand to 2D Gaussian kernel
    gauss_kernel_2d = gauss_kernel_1d[:, None] @ gauss_kernel_1d[None, :]
    gauss_kernel_2d = gauss_kernel_2d.expand(inputs.size(1), 1, kernel_size, kernel_size).to(inputs.device)

    # Apply Gaussian blur using convolution
    padding = kernel_size // 2  # Ensure the output image has the same size as input
    blurred_inputs = F.conv2d(inputs, gauss_kernel_2d, padding=padding, groups=inputs.size(1))

    # Combine the blurred and original inputs based on the mask
    result = masks * inputs + (1 - masks) * blurred_inputs

    return result



def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class SkinLesionStyleLoss:
    def __init__(self, layers=None, device='cpu'):
        """
        A class to compute style loss for ensuring the style consistency between damaged and generated images.

        Parameters:
            layers (list of int): Indices of VGG layers to extract style features from.
                                  Defaults to [1, 6, 11, 20, 29] (conv1_1, conv2_1, etc.).
            device (str): Device to use ('cpu' or 'cuda').
        """
        self.device = device
        self.vgg = models.vgg19(pretrained=True).features.to(device).eval()
        self.layers = layers or [1, 6, 11, 20, 29]  # Default to common VGG layers
        self.loss_fn = nn.MSELoss()

    def _extract_features(self, image):
        """
        Extract features from the specified VGG layers.

        Parameters:
            image (torch.Tensor): Input image tensor of shape [1, 3, H, W].

        Returns:
            list of torch.Tensor: Features from the specified layers.
        """
        features = []
        x = image
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                features.append(self._gram_matrix(x))
        return features

    @staticmethod
    def _gram_matrix(features):
        """
        Compute the Gram matrix for a given feature map.

        Parameters:
            features (torch.Tensor): Feature map of shape [1, C, H, W].

        Returns:
            torch.Tensor: Gram matrix of shape [1, C, C].
        """
        B, C, H, W = features.size()
        features = features.view(B, C, -1)  # Reshape to [B, C, H*W]
        gram = torch.bmm(features, features.transpose(1, 2))  # Compute Gram matrix
        return gram / (C * H * W)  # Normalize by the number of elements

    def compute_loss(self, damaged_image, generated_image):
        """
        Compute the style loss between the damaged and generated images.

        Parameters:
            damaged_image (torch.Tensor): The original damaged image of shape [1, 3, H, W].
            generated_image (torch.Tensor): The generated image of shape [1, 3, H, W].

        Returns:
            torch.Tensor: Style loss.
        """
        damaged_features = self._extract_features(damaged_image.to(self.device))
        generated_features = self._extract_features(generated_image.to(self.device))
        loss = 0
        for damaged_gram, generated_gram in zip(damaged_features, generated_features):
            loss += self.loss_fn(damaged_gram, generated_gram)
        return loss



def save_tensor_image_cv2(tensor, filename="test.png"):
    # If the tensor is on the GPU, move it to CPU before converting to a numpy array
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Convert from [C, H, W] format to [H, W, C] format
    tensor = tensor.permute(1, 2, 0)

    # Min-Max normalization to scale tensor values to the range [0, 1]
    tensor_min = tensor.min()
    tensor_max = tensor.max()

    # Add small epsilon value to prevent division by zero
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-6)  # Normalize to [0, 1]

    # Scale the tensor values from [0, 1] to [0, 255]
    tensor = tensor * 255.0
    tensor = tensor.byte().numpy()  # Convert tensor to byte type and then to numpy array
    
    # OpenCV expects BGR format, so we convert from RGB to BGR
    tensor = cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)

    # Save the image using OpenCV
    cv2.imwrite(filename, tensor)
    print(f"Image saved to {filename}")
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    


def mk_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)



class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    
def set_dataset(dataset_name, fold):
    dataset_name = dataset_name.lower()


    if dataset_name == "isic2018":
        data_path = 'resized_dataset/ISIC_2018'
        num_class = 7
        inference_folder = "test"
        
    elif dataset_name == "isic2017":
        data_path = 'resized_dataset/ISIC_2017'
        num_class = 3
        inference_folder = "test"


    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Please choose from: 'isic2017', 'isic2018'.")
    
    
    return data_path, num_class, inference_folder


# Custom Dataset to preload data into memory
class InMemoryDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.data = []  # Store all images and labels
        self.classes = []  # Store class names

        # Use ImageFolder to read data
        image_folder = datasets.ImageFolder(root=root)
        self.classes = image_folder.classes  # Save class information
        
        for img_path, label in image_folder.imgs:
            img = Image.open(img_path).convert('RGB')  # Open image and convert to RGB format
            self.data.append((img, label))  # Load image and label into memory

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index]
        if self.transform:
            img = self.transform(img)  # Apply transformation
        return img, label
    
    