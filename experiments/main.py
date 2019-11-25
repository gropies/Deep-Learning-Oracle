"""
## Quoted From: https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
"""

import os
import sys
import time
import numpy as np
from tqdm import tqdm, trange
import Mono_depth
import edge_model

import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from functions import cross_entropy_loss

from torchvision import datasets
from torchvision import transforms
from torchvision import models

import utils
from net import Net, Vgg16

from option import Options
from easydict import EasyDict as edict
import lap_loss
import depth_loss as dl

def main():
    # figure out the experiments type
    args = Options().parse()
    if args.subcommand is None:
        raise ValueError("ERROR: specify the experiment type")
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("ERROR: cuda is not available, try running on CPU")


    if args.subcommand == "train":
        # Training the model 
        train(args)

    elif args.subcommand == 'eval':
        # Test the pre-trained model
        evaluate(args)

    elif args.subcommand == 'optim':
        # Gatys et al. using optimization-based approach
        optimize(args)

    else:
        raise ValueError('Unknow experiment type')


def optimize(args):
    """    Gatys et al. CVPR 2017
    ref: Image Style Transfer Using Convolutional Neural Networks
    """
    # load the content and style target
    content_image = utils.tensor_load_rgbimage(args.content_image, size=args.content_size, keep_asp=True)
    content_image = content_image.unsqueeze(0)
    content_image = Variable(utils.preprocess_batch(content_image), requires_grad=False)
    content_image = utils.subtract_imagenet_mean_batch(content_image)
    style_image = utils.tensor_load_rgbimage(args.style_image, size=args.style_size)
    style_image = style_image.unsqueeze(0)    
    style_image = Variable(utils.preprocess_batch(style_image), requires_grad=False)
    style_image = utils.subtract_imagenet_mean_batch(style_image)

    # load the pre-trained vgg-16 and extract features
    vgg = Vgg16()
    utils.init_vgg16(args.vgg_model_dir)
    vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))
    if args.cuda:
        content_image = content_image.cuda()
        style_image = style_image.cuda()
        vgg.cuda()
    features_content = vgg(content_image)
    f_xc_c = Variable(features_content[1].data, requires_grad=False)
    features_style = vgg(style_image)
    gram_style = [utils.gram_matrix(y) for y in features_style]
    # init optimizer
    output = Variable(content_image.data, requires_grad=True)
    optimizer = Adam([output], lr=args.lr)
    mse_loss = torch.nn.MSELoss()
    # optimizing the images
    tbar = trange(args.iters)
    for e in tbar:
        utils.imagenet_clamp_batch(output, 0, 255)
        optimizer.zero_grad()
        features_y = vgg(output)
        content_loss = args.content_weight * mse_loss(features_y[1], f_xc_c)

        style_loss = 0.
        for m in range(len(features_y)):
            gram_y = utils.gram_matrix(features_y[m])
            gram_s = Variable(gram_style[m].data, requires_grad=False)
            style_loss += args.style_weight * mse_loss(gram_y, gram_s)

        total_loss = content_loss + style_loss
        total_loss.backward()
        optimizer.step()
        tbar.set_description(total_loss.data.cpu().numpy()[0])
    # save the image    
    output = utils.add_imagenet_mean_batch(output)
    utils.tensor_save_bgrimage(output.data[0], args.output_image, args.cuda)


def train(args):
    check_paths(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        kwargs = {'num_workers': 0, 'pin_memory': False}
    else:
        kwargs = {}

    transform = transforms.Compose([transforms.Scale(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)

    style_model = Net(ngf=args.ngf)
    if args.resume is not None:
        print('Resuming, initializing using weight from {}.'.format(args.resume))
        style_model.load_state_dict(torch.load(args.resume))
    print(style_model)
    optimizer = Adam(style_model.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()
    edgeloss = cross_entropy_loss()
    laploss = lap_loss.LapLoss()
    deploss = dl.MonodepthLoss(
                n=4,
                SSIM_w=0.85,
                disp_gradient_w=0.1, lr_w=1).cuda()

    vgg = Vgg16()
    utils.init_vgg16(args.vgg_model_dir)
    vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))
    
    edge_detect = edge_model.HED()
    if not os.path.exists(os.path.join(args.vgg_model_dir, 'edge.pytorch')):
        os.system(
            'wget --timestamping http://content.sniklaus.com/github/pytorch-hed/network-bsds500.pytorch -O ' + os.path.join(args.vgg_model_dir, 'edge.pytorch'))
    state = torch.load(os.path.join(args.vgg_model_dir, 'edge.pytorch'))
    for (src, dst) in zip(state.keys(), edge_detect.parameters()):
        dst.data[:] = state[src]

    if not os.path.exists(os.path.join(args.vgg_model_dir, 'depth.pth')):
        os.system(
            'wget https://p-def6.pcloud.com/cBZcIboSHZMPuC9HZZZFDKdN7Z2ZZVaFZkZkCcA0ZskZE5ZqXZp0Z10Z5kZHZI7Z0FZ8VZS7ZwVZb0ZYXZb5r97ZPbVM2erapaYylK4hGJglYSRc5oHk/monodepth_resnet18_001.pth -O ' + os.path.join(args.vgg_model_dir, 'depth.pth'))
    depth_model = Mono_depth.Model(edict({'data_dir':'data/kitti/train/',
                         'val_data_dir':'data/kitti/val/',
                         'model_path':'monodepth_resnet18_001.pth',
                         'output_directory':'data/output/',
                         'input_height':256,
                         'input_width':512,
                         'model':'resnet18_md',
                         'pretrained':True,
                         'mode':'eval',
                         'epochs':200,
                         'learning_rate':1e-4,
                         'batch_size': 8,
                         'adjust_lr':True,
                         'device':'cuda:0',
                         'do_augmentation':True,
                         'augment_parameters':[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
                         'print_images':False,
                         'print_weights':False,
                         'input_channels': 3,
                         'num_workers': 8,
                         'use_multiple_gpu': False}))
    depth_model.load('model/depth.pth')
    if args.cuda:
        style_model.cuda()
        vgg.cuda()

    style_loader = utils.StyleLoader(args.style_folder, args.style_size)

    tbar = trange(args.epochs)
    for e in tbar:
        style_model.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            x = Variable(utils.preprocess_batch(x))
            if args.cuda:
                x = x.cuda()

            style_v = style_loader.get(batch_id)
            style_model.setTarget(style_v)

            style_v = utils.subtract_imagenet_mean_batch(style_v)
            features_style = vgg(style_v)
            gram_style = [utils.gram_matrix(y) for y in features_style]

            y = style_model(x)
            xc = Variable(x.data.clone())

            y = utils.subtract_imagenet_mean_batch(y)
            xc = utils.subtract_imagenet_mean_batch(xc)

            features_y = vgg(y)
            features_xc = vgg(xc)

            f_xc_c = Variable(features_xc[1].data, requires_grad=False)

            content_loss = args.content_weight * mse_loss(features_y[1], f_xc_c)
            
            edge_detect.eval()
            edge_out = edge_detect(y)
            edge_loss = edgeloss(edge_out,xc)
            laplacian_loss = laploss(y,xc)
            laplacian_weight = 2.5
            depth_left = depth_model.model(y)
            depth_loss = deploss(depth_left,[y,xc])


            style_loss = 0.
            for m in range(len(features_y)):
                gram_y = utils.gram_matrix(features_y[m])
                gram_s = Variable(gram_style[m].data, requires_grad=False).repeat(args.batch_size, 1, 1, 1)
                style_loss += args.style_weight * mse_loss(gram_y, gram_s[:n_batch, :, :])

            total_loss = content_loss + style_loss + laplacian_weight * laplacian_loss + depth_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.data.item()
            agg_style_loss += style_loss.data.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                agg_content_loss / (batch_id + 1),
                                agg_style_loss / (batch_id + 1),
                                (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                tbar.set_description(mesg)

            
            if (batch_id + 1) % (4 * args.log_interval) == 0:
                # save model
                style_model.eval()
                style_model.cpu()
                save_model_filename = "Epoch_" + str(e) + "iters_" + str(count) + "_" + \
                    str(time.ctime()).replace(' ', '_') + "_" + str(
                    args.content_weight) + "_" + str(args.style_weight) + ".model"
                save_model_path = os.path.join(args.save_model_dir, save_model_filename)
                torch.save(style_model.state_dict(), save_model_path)
                style_model.train()
                style_model.cuda()
                tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

    # save model
    style_model.eval()
    style_model.cpu()
    save_model_filename = "Final_epoch_" + str(args.epochs) + "_" + \
        str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(style_model.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def check_paths(args):
    try:
        if not os.path.exists(args.vgg_model_dir):
            os.makedirs(args.vgg_model_dir)
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def evaluate(args):
    content_image = utils.tensor_load_rgbimage(args.content_image, size=args.content_size, keep_asp=True)
    content_image = content_image.unsqueeze(0)
    style = utils.tensor_load_rgbimage(args.style_image, size=args.style_size)
    style = style.unsqueeze(0)    
    style = utils.preprocess_batch(style)

    style_model = Net(ngf=args.ngf)
    model_dict = torch.load(args.model)
    model_dict_clone = model_dict.copy()
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model.load_state_dict(model_dict, False)

    if args.cuda:
        style_model.cuda()
        content_image = content_image.cuda()
        style = style.cuda()

    style_v = Variable(style)

    content_image = Variable(utils.preprocess_batch(content_image))
    style_model.setTarget(style_v)

    output = style_model(content_image)
    #output = utils.color_match(output, style_v)
    utils.tensor_save_bgrimage(output.data[0], args.output_image, args.cuda)


def fast_evaluate(args, basedir, contents, idx = 0):
    # basedir to save the data
    style_model = Net(ngf=args.ngf)
    style_model.load_state_dict(torch.load(args.model), False)
    style_model.eval()
    if args.cuda:
        style_model.cuda()
    
    style_loader = StyleLoader(args.style_folder, args.style_size, 
        cuda=args.cuda)

    for content_image in contents:
        idx += 1
        content_image = utils.tensor_load_rgbimage(content_image, size=args.content_size, keep_asp=True).unsqueeze(0)
        if args.cuda:
            content_image = content_image.cuda()
        content_image = Variable(utils.preprocess_batch(content_image))

        for isx in range(style_loader.size()):
            style_v = Variable(style_loader.get(isx).data)
            style_model.setTarget(style_v)
            output = style_model(content_image)
            filename = os.path.join(basedir, "{}_{}.png".format(idx, isx+1))
            utils.tensor_save_bgrimage(output.data[0], filename, args.cuda)
            print(filename)


if __name__ == "__main__":
   main()
