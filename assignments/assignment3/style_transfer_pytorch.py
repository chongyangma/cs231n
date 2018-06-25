import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image
import PIL
import numpy as np

from cs231n.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD


def preprocess(img, size=512):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in SQUEEZENET_STD.tolist()]),
        T.Normalize(mean=[-m for m in SQUEEZENET_MEAN.tolist()], std=[1, 1, 1]),
        T.Lambda(rescale),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

def features_from_img(imgpath, imgsize):
    img = preprocess(PIL.Image.open(imgpath), size=imgsize)
    img_var = Variable(img.type(dtype))
    return extract_features(img_var, cnn), img_var


dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

cnn = torchvision.models.squeezenet1_1(pretrained=True).features
cnn.type(dtype)
for param in cnn.parameters():
    param.requires_grad = False

def extract_features(x, cnn):
    """
    Use the CNN to extract features from the input image x.

    Inputs:
    - x: A PyTorch Variable of shape (N, C, H, W) holding a minibatch of images that
      will be fed to the CNN.
    - cnn: A PyTorch model that we will use to extract features.

    Returns:
    - features: A list of feature for the input images x extracted using the cnn model.
      features[i] is a PyTorch Variable of shape (N, C_i, H_i, W_i); recall that features
      from different layers of the network may have different numbers of channels (C_i) and
      spatial dimensions (H_i, W_i).
    """
    features = []
    prev_feat = x
    for i, module in enumerate(cnn._modules.values()):
        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features


def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.

    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_target: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """
    content_loss = content_weight * torch.sum((torch.pow(content_current - content_original, 2)))
    return content_loss


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.

    Inputs:
    - features: PyTorch Variable of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)

    Returns:
    - gram: PyTorch Variable of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    N, C, H, W = features.size()
    # Use torch.bmm for batch multiplication of matrices
    feat_reshaped = features.view(N, C, -1)
    gram = torch.bmm(feat_reshaped, feat_reshaped.transpose(1, 2))
    if normalize:
        return gram / (H*W*C)
    else:
        return gram


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.

    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Variable giving the Gram matrix the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].

    Returns:
    - style_loss: A PyTorch Variable holding a scalar giving the style loss.
    """
    style_loss = 0
    for i in range(len(style_layers)):
        gram = gram_matrix(feats[style_layers[i]])
        style_loss += style_weights[i] * torch.sum(torch.pow(gram-style_targets[i], 2))
    return style_loss


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.

    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss


def style_transfer(content_image, style_image, image_size, style_size, content_layer, content_weight,
                   style_layers, style_weights, tv_weight, iter_num, init_random = False):
    # Extract features for the content image
    content_img = preprocess(PIL.Image.open(content_image), size=image_size)
    content_img_var = Variable(content_img.type(dtype))
    feats = extract_features(content_img_var, cnn)
    content_target = feats[content_layer].clone()

    # Extract features for the style image
    style_img = preprocess(PIL.Image.open(style_image), size=style_size)
    style_img_var = Variable(style_img.type(dtype))
    feats = extract_features(style_img_var, cnn)
    style_targets = []
    for idx in style_layers:
        style_targets.append(gram_matrix(feats[idx].clone()))

    # Initialize output image to content image or nois
    if init_random:
        img = torch.Tensor(content_img.size()).uniform_(0, 1)
    else:
        img = content_img.clone().type(dtype)

    # We do want the gradient computed on our image!
    img_var = Variable(img, requires_grad=True)

    # Set up optimization hyperparameters
    initial_lr = 3.0
    decayed_lr = 0.1
    decay_lr_at = 180

    # Note that we are optimizing the pixel values of the image by passing
    # in the img_var Torch variable, whose requires_grad flag is set to True
    optimizer = torch.optim.Adam([img_var], lr=initial_lr)

    for t in range(iter_num):
        if t < 190:
            img.clamp_(-1.5, 1.5)
        optimizer.zero_grad()

        feats = extract_features(img_var, cnn)

        # Compute loss
        c_loss = content_loss(content_weight, feats[content_layer], content_target)
        s_loss = style_loss(feats, style_layers, style_targets, style_weights)
        t_loss = tv_loss(img_var, tv_weight)
        loss = c_loss + s_loss + t_loss

        loss.backward()

        # Perform gradient descents on our image values
        if t == decay_lr_at:
            optimizer = torch.optim.Adam([img_var], lr=decayed_lr)
        optimizer.step()

        if t % 100 == 0 or t == iter_num - 1:
            print('Iteration {}: {}'.format(t, loss.item()))
    save_image(deprocess(img), 'output.png')


parser = argparse.ArgumentParser(description='Style Transfer')
parser.add_argument('--iter_num', type=int, default=300, metavar='N',
                    help='number of iterations to optimize')
parser.add_argument('--content_image', type=str, default='styles/tubingen.jpg',
                    help='Input path to content image')
parser.add_argument('--style_image', type=str, default='styles/composition_vii.jpg',
                    help='Input path to style image')
parser.add_argument('--image_size', type=int, default=192, metavar='N',
                    help='size of smallest image dimension (used for content loss and generated image)')
parser.add_argument('--style_size', type=int, default=512, metavar='N',
                    help='size of smallest style image dimension')
parser.add_argument('--tv_weight', type=float, default='5e-2',
                    help='weight of total variation regularization term')
parser.add_argument('--init_random', action='store_true', default=False,
                    help='initialize the starting image to uniform random noise')
parser.add_argument('--content_layer', type=int, default=3,
                    help='layer to use for content loss')
parser.add_argument('--content_weight', type=float, default=5e-2,
                    help='weighting on content loss')
parser.add_argument('--style_layers', type=str, default='1, 4, 6, 7',
                    help='list of layers to use for style loss')
parser.add_argument('--style_weights', type=str, default='20000, 500, 12, 1',
                    help='list of weights to use for each layer in style_layers')


if __name__ == '__main__':
    args = parser.parse_args()

    style_layers_list = [int(item) for item in args.style_layers.split(',')]
    style_weights_list = [float(item) for item in args.style_weights.split(',')]
    assert len(style_layers_list) == len(style_weights_list), \
        "Numbers of layers and weights should match"

    style_transfer(args.content_image, args.style_image,
                   args.image_size, args.style_size,
                   args.content_layer, args.content_weight,
                   style_layers_list, style_weights_list,
                   args.tv_weight,
                   args.iter_num,
                   args.init_random
                   )
