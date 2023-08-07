import config
import timeit
import os
import deeplab_resnet

import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import make_grid

from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
def train(traindata,valdata,net,mode,n_classes):
    
    print(f'Current Mode == > {mode}')
    if mode == 'pretrained':
      net = deeplab_resnet.DeepLabv3_plus(nInputChannels=3, n_classes=n_classes, os=16, pretrained=True).to(device) 
    criterion = cross_entropy2d
    
    num_img_tr = len(traindata)
    num_img_ts = len(valdata)
    
    print("num_img_tr >>> ")
    print(num_img_tr)
    print("num_img_ts >>> ")
    print(num_img_ts)
    running_loss_tr = 0.0
    running_loss_ts = 0.0
    aveGrad = 0
    global_step = 0
    nTestInterval = config.nTestInterval
    snapshot = config.snapshot
    optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.wd)
    
    print("Training Network")

    for epoch in range(config.nEpochs):
        start_time = timeit.default_timer()

        if epoch % config.epoch_size == config.epoch_size - 1:
            lr_ = lr_poly(config.lr, epoch, config.nEpochs, 0.9)
            print('(poly lr policy) learning rate: ', lr_)
            optimizer = optim.SGD(net.parameters(), lr=lr_, momentum=config.momentum, weight_decay=config.wd)

        net.train()
        for ii, sample_batched in enumerate(traindata):

            # if ii % int(num_img_tr * 0.25) == 0:
            #     print(f"{num_img_tr} 중 {ii % int(num_img_tr * 0.25)}'s Batch 완료!")
                
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
                print('[Epoch: %d, numImages: %5d]' % (epoch+1, ii * config.batch_size + inputs.data.shape[0]))
                print('Loss: %f' % running_loss_tr)
                running_loss_tr = 0
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time) + "\n")

            # Backward the averaged gradient
            loss /= config.nAveGrad
            loss.backward()
            aveGrad += 1

            # Update the weights once in p['nAveGrad'] forward passes
            if aveGrad % config.nAveGrad == 0:
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

            # # Show 10 * 3 images results each epoch
            # if ii % (num_img_tr // 10) == 0:
            #     grid_image = make_grid(inputs[:3].clone().cpu().data, 3, normalize=True)
            #     grid_image = make_grid(decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy()), 3, normalize=False,
            #                            range=(0, 255))
            #     grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy()), 3, normalize=False, range=(0, 255))

        # Save the model
        if (epoch % snapshot) == snapshot - 1:
            torch.save(net.state_dict(), os.path.join(config.save_dir, 'best.pth'))
            print("Save model at {}\n".format(os.path.join(config.save_dir, 'best.pth')))

        # One testing epoch
        if epoch % nTestInterval == (nTestInterval - 1):
            total_iou = 0.0
            net.eval()
            for ii, sample_batched in enumerate(valdata):
                inputs, labels = sample_batched['image'], sample_batched['label']

                # Forward pass of the mini-batch
                inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
                inputs, labels = inputs.cuda(), labels.cuda()

                with torch.no_grad():
                    outputs = net.forward(inputs)

                predictions = torch.max(outputs, 1)[1]

                loss = criterion(outputs, labels, size_average=False, batch_average=True)
                running_loss_ts += loss.item()

                total_iou += get_iou(predictions, labels)

                # Print stuff
                if ii % num_img_ts == num_img_ts - 1:

                    miou = total_iou / (ii * config.batch_size + inputs.data.shape[0])
                    running_loss_ts = running_loss_ts / num_img_ts

                    print('Validation:')
                    print('[Epoch: %d, numImages: %5d]' % (epoch, ii * config.batch_size + inputs.data.shape[0]))
                    print('Loss: %f' % running_loss_ts)
                    print('MIoU: %f\n' % miou)
                    running_loss_ts = 0