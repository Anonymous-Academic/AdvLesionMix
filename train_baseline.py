from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import torch
import torchvision
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from skimage import feature as skif
import torch.nn.functional as F
import random
import shutil
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import timm
from sklearn.metrics import precision_recall_fscore_support as score
from utils import *
import torch.optim as optim

import argparse


parser = argparse.ArgumentParser(description='Organize Dataset')
parser.add_argument('--m', required=False, type=str, default='resnet34', help='model')
parser.add_argument('--d', required=False, type=str, default='isic2017', help='dataset')
parser.add_argument('--f', required=False, type=str, default=None, help='fold')
parser.add_argument('--i', required=False,type=int, default=224)
parser.add_argument('--s', required=False,type=int, default=0, help='seed')
parser.add_argument('--optim_times', required=False,type=int, default=1, help='optim times')

args, unparsed = parser.parse_known_args()


seed = args.s

input_size = args.i

    
def inference(net, criterion, test_loader):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = "cpu"
    net.to(device)

    

    score_list = []
    target_list = []
    pred_list = []
    
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs), Variable(targets)
        with torch.no_grad():
            outputs = net(inputs)

        score_list.append(outputs.softmax(dim=1).data.cpu())
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        pred_list.append(predicted.data.cpu().unsqueeze(0))
        target_list.append(targets.data.cpu().unsqueeze(0))

        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx % 50 == 0 or batch_idx == (test_loader.__len__()-1):
            print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

    pred_list = torch.cat(pred_list, axis=-1).squeeze().numpy()
    target_list = torch.cat(target_list, axis=-1).squeeze().numpy()
    score_list = torch.cat(score_list, axis=0).squeeze().numpy()
    
    accuracy = accuracy_score(pred_list, target_list)*100
    f1_micro = f1_score(target_list,pred_list,average='micro')
    f1_macro = f1_score(target_list,pred_list,average='macro')
    
    
    precision_micro, recall_micro, _, _ = score(torch.from_numpy(target_list), torch.from_numpy(pred_list), average='micro')
    precision_macro, recall_macro, _, _ = score(torch.from_numpy(target_list), torch.from_numpy(pred_list), average='macro')
    
    auc_micro = roc_auc_score(target_list, score_list, multi_class='ovr',average='micro')
    auc_macro = roc_auc_score(target_list, score_list, multi_class='ovr',average='macro')
    
    
    
    test_acc = 100. * float(correct) / total

    test_loss = test_loss / (idx + 1)
    print("Test Accuracy: {}%".format(test_acc))

    return accuracy, f1_micro, f1_macro, auc_micro,auc_macro, precision_micro, precision_macro, recall_micro, recall_macro


      

def test(net, criterion, test_loader):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0
    device = torch.device("cuda")

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs), Variable(targets)
        with torch.no_grad():
            outputs = net(inputs)

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

    test_acc = 100. * float(correct) / total

    test_loss = test_loss / (idx + 1)

    return test_acc, test_loss


def train(nb_epoch, batch_size, num_class, store_name, lr=0.002, data_path='', start_epoch=0, validation_folder="", inference_folder=""):
    # setup output
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)
    
  

    use_cuda = torch.cuda.is_available()
    print(use_cuda)


    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize((int(input_size/0.875), int(input_size/0.875))),
        transforms.RandomCrop(input_size, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    transform_test = transforms.Compose([
            transforms.Resize((int(input_size/0.875), int(input_size/0.875))),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    

    print("Loading training data...")
    train_set = InMemoryDataset(root=data_path + '/train', transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)  # num_workers=0 to avoid data duplication in memory
    
    
    print("Loading validation data...")
    val_set = InMemoryDataset(root=os.path.join(data_path, validation_folder), transform=transform_test)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0)

    print("Loading inference data...")
    infer_set = InMemoryDataset(root=os.path.join(data_path, inference_folder), transform=transform_test)
    infer_loader = DataLoader(infer_set, batch_size=4, shuffle=False, num_workers=0)
   
    
    

        
    if args.m == "resnet18":
        net = torchvision.models.resnet18(pretrained=True)  
        net.fc = nn.Linear(net.fc.in_features, num_class)    
    elif args.m == "resnet34":
        net = torchvision.models.resnet34(pretrained=True)  
        net.fc = nn.Linear(net.fc.in_features, num_class)  
    elif args.m == "vgg13":
        net = torchvision.models.vgg13(pretrained=True)  
        net.classifier[6] = nn.Linear(net.classifier[6].in_features, num_class) 
    elif args.m == "swins":
        net = timm.create_model("swin_small_patch4_window7_224.ms_in1k", pretrained=True, num_classes=num_class)
    elif args.m == "mobile050":
        net = timm.create_model("mobilenetv2_050.lamb_in1k", pretrained=True, num_classes=num_class)
    elif args.m == "mobile100":
        net = timm.create_model("mobilenetv2_100.ra_in1k", pretrained=True, num_classes=num_class)                                    
    elif args.m == "pvt":        
        net = timm.create_model("pvt_v2_b0.in1k", pretrained=True, num_classes=num_class)
        
    else:
        raise ValueError(f"Unsupported network type: {args.m}. Please choose from: 'resnet34', 'resnet18', 'vgg13', 'swins', 'mobile050', 'mobile100', 'pvt'.")
    
    

    netp = torch.nn.DataParallel(net)

    # GPU
    device = torch.device("cuda")
    net.to(device)

    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    max_val_acc = 0

    for epoch in tqdm(range(start_epoch, nb_epoch), ncols=80, unit='s', unit_scale=True):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            idx = batch_idx
            if inputs.shape[0] < batch_size:
                continue
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # update learning rater
 
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr)

            for ii in range(args.optim_times):
                optimizer.zero_grad()
                outputs = netp(inputs)
                loss = CELoss(outputs, targets)
                loss.backward()
                optimizer.step()
            

            #  training log
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            train_loss += loss.item() 
            

        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f |\n' % (
                epoch, train_acc, train_loss))


        val_acc, val_loss = test(net, CELoss, val_loader)
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            net.cpu()
            torch.save(net, './' + store_name + '/model.pth')
            net.to(device)
        with open(exp_dir + '/results_test.txt', 'a') as file:
            file.write('Iteration %d, test_acc = %.5f, test_loss = %.6f\n' % (
            epoch, val_acc, val_loss))
            
    trained_model = torch.load('./' + store_name + '/model.pth')
    accuracy, f1_micro, f1_macro, auc_micro, auc_macro, precision_micro, precision_macro, recall_micro, recall_macro = inference(trained_model, CELoss, infer_loader)
    with open(exp_dir + '/results_test.txt', 'a') as file:
        file.write('Inference Results: Accuracy = %.5f, F1_micro = %.5f, F1_macro = %.5f, Auc_micro = %.5f, Auc_macro = %.5f \n' % (
        accuracy, f1_micro, f1_macro, auc_micro, auc_macro))
        file.write('Inference Results: precision_micro = %.5f, precision_macro = %.5f, recall_micro = %.5f, recall_macro = %.5f \n' % (
        precision_micro, precision_macro, recall_micro, recall_macro))
         
              
 

if __name__ == '__main__':
    seed_everything(seed)
    
    data_path, num_class, inference_folder = set_dataset(args.d, args.f)
    
    lr = 0.002
    
    results_path ='results'

    mk_dir(results_path)
    

    task_result_path = os.path.join(results_path, 'baseline')
    mk_dir(task_result_path)


    
    pyname = os.path.basename(__file__).replace('.py','').replace('train_','')
    experiment_result_path = \
                        os.path.join(task_result_path, "{}_dataset_{}_model_{}_seed_{}_input_size_{}_fold_{}_optim_times_{}"
                                     .format(pyname,args.d,args.m,seed,input_size,args.f,args.optim_times))


    train(nb_epoch=100,             
            batch_size=16,
            num_class=num_class,
            lr = lr,      
            store_name= experiment_result_path,     
            data_path=data_path,          
            start_epoch=0,
            validation_folder="validation",
            inference_folder=inference_folder)         
