import timeit
import os
import argparse

import deeplab_resnet
from utils import *

import torch.optim as optim
from torch.autograd import Variable

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(traindata,valdata,args):
    net = deeplab_resnet.DeepLabv3_plus(nInputChannels=3, n_classes=args.n_classes, os=16, pretrained=True).to(device) 
    criterion = cross_entropy2d
    
    num_img_tr = len(traindata)
    num_img_ts = len(valdata)
    
    print(f"Num of Train Batch : {num_img_tr} ")
    print(f"Num of Validation Batch : {num_img_ts}")

    running_loss_tr = 0.0
    running_loss_ts = 0.0
    aveGrad = 0
    global_step = 0
    nTestInterval = args.nTestInterval
    snapshot = args.snapshot
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    
    print("Training Network")

    for epoch in range(args.nEpochs):
        start_time = timeit.default_timer()

        if epoch % args.epoch_size == args.epoch_size - 1:
            lr_ = lr_poly(args.lr, epoch, args.nEpochs, 0.9)
            print('(poly lr policy) learning rate: ', lr_)
            optimizer = optim.SGD(net.parameters(), lr=lr_, momentum=args.momentum, weight_decay=args.wd)

        net.train()
        for ii, sample_batched in enumerate(traindata):
            inputs, labels = sample_batched['image'], sample_batched['label']
            # Forward-Backward of the mini-batch
            inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
            global_step += inputs.data.shape[0]    
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net.forward(inputs)

            loss = criterion(outputs, labels, size_average=False, batch_average=True)
            running_loss_tr += loss.item()

            # Print stuff
            if ii % num_img_tr == (num_img_tr - 1):
                running_loss_tr = running_loss_tr / num_img_tr
                print('[Epoch: %d, numImages: %5d]' % (epoch+1, ii * args.batch_size + inputs.data.shape[0]))
                print('Loss: %f' % running_loss_tr)
                running_loss_tr = 0
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time) + "\n")

            # Backward the averaged gradient
            loss /= args.nAveGrad
            loss.backward()
            aveGrad += 1

            # Update the weights once in p['nAveGrad'] forward passes
            if aveGrad % args.nAveGrad == 0:
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

        # Save the model
        if (epoch % snapshot) == snapshot - 1:
            torch.save(net.state_dict(), os.path.join(args.output_dir, 'best.pth'))
            print("Save model at {}\n".format(os.path.join(args.output_dir, 'best.pth')))

        # One testing epoch
        if epoch % nTestInterval == (nTestInterval - 1):
            total_iou = 0.0
            net.eval()
            for ii, sample_batched in enumerate(valdata):
                inputs, labels = sample_batched['image'], sample_batched['label']

                # Forward pass of the mini-batch
                inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.no_grad():
                    outputs = net.forward(inputs)

                predictions = torch.max(outputs, 1)[1]

                loss = criterion(outputs, labels, size_average=False, batch_average=True)
                running_loss_ts += loss.item()

                total_iou += get_iou(predictions, labels,args.n_classes)

                # Print stuff
                if ii % num_img_ts == num_img_ts - 1:

                    miou = total_iou / (ii * args.batch_size + inputs.data.shape[0])
                    running_loss_ts = running_loss_ts / num_img_ts

                    print('Validation:')
                    print('[Epoch: %d, numImages: %5d]' % (epoch, ii * args.batch_size + inputs.data.shape[0]))
                    print('Loss: %f' % running_loss_ts)
                    print('MIoU: %f\n' % miou)
                    running_loss_ts = 0