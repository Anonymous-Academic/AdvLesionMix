from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from PIL import Image
import torchvision.models
import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import timm
from tqdm import tqdm
from utils import *
from collections import Counter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import timm
import lpips
from sklearn.metrics import precision_recall_fscore_support as score
import argparse



parser = argparse.ArgumentParser(description='Organize Dataset')
parser.add_argument('--m', required=False, type=str, default='resnet18', help='model')
parser.add_argument('--d', required=False, type=str, default='isic2018', help='dataset')
parser.add_argument('--f', required=False, type=str, default=None, help='fold')
parser.add_argument('--i', required=False,type=int, default=224, help='input size')
parser.add_argument('--s', required=False,type=int, default=0, help='seed')



parser.add_argument('--gloss', required=False,type=str, default="type1c", help='gloss')
parser.add_argument('--rloss', required=False,type=str, default="mse", help='regression loss')
parser.add_argument('--alpha', required=False,type=float, default=1.0, help='alpha')
parser.add_argument('--beta', required=False,type=float, default=1.0, help='beta')
parser.add_argument('--gtype', required=False,type=str, default="Generator_Wrapper3", help='generator type')

parser.add_argument('--inner_dim', required=False,type=int, default=512, help='inner_dim')
parser.add_argument('--pad_normal', action='store_true', help='Set this flag to enable pad_normal')

parser.add_argument('--mix_ech', required=False,type=int, default=20, help='the epoch for starting mix')
parser.add_argument('--mix_rate', required=False,type=float, default=0.3, help='mix rate')
parser.add_argument('--mask_thd', required=False,type=float, default=0.3, help='threshold')
parser.add_argument('--mix_type', required=False,type=str, default="hard", help='mix type')


args, unparsed = parser.parse_known_args()
if not args.gloss=="type2":
   args.beta = None 
   
if args.gloss == "type1b":
    args.beta = None
    args.alpha = None
    args.rloss = None

seed = args.s
seed_everything(seed)
input_size = args.i

class NegativeL1Loss(nn.Module):
    def __init__(self):
        super(NegativeL1Loss, self).__init__()

    def forward(self, x1, x2):
        epsilon = 1e-8  
        
        x1_min = x1.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        x1_max = x1.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        x1_range = x1_max - x1_min + epsilon  
        x1_normalized = (x1 - x1_min) / x1_range

        x2_min = x2.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        x2_max = x2.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        x2_range = x2_max - x2_min + epsilon  
        x2_normalized = (x2 - x2_min) / x2_range

        x1_normalized = 2 * x1_normalized - 1
        x2_normalized = 2 * x2_normalized - 1

        loss_fn = nn.L1Loss()
        loss = -loss_fn(x1_normalized, x2_normalized)
        return loss

class Generator(nn.Module):
    def __init__(self, input_dim, num_layers, image_dim=6):
        super(Generator, self).__init__()
        image_processor = [nn.Conv2d(3, image_dim, 5, stride=1, padding=2),
                nn.InstanceNorm2d(image_dim),
                nn.ReLU(inplace=True)]   
        self.image_processor = nn.Sequential(*image_processor)   
        
        g = []         
        for _ in range(num_layers):
            out_dim = input_dim // 2
            g += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(input_dim, out_dim, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_dim),
                nn.ReLU(inplace=True),
            ]
            input_dim = out_dim
        g += [nn.Conv2d(input_dim, image_dim, 3, stride=1, padding=1), nn.ReLU(inplace=True)] 
        self.generator = nn.Sequential(*g)   
        
        o = [nn.ReflectionPad2d(3), nn.Conv2d(image_dim*2, 3, 7), nn.Tanh()]        
        self.output = nn.Sequential(*o)        

    def forward(self, im, feature):
        im  = self.image_processor(im)
        feature = self.generator(feature)

        return self.output(torch.cat((im, feature), dim=1)) 

class Features(nn.Module):
    def __init__(self, net_layers_FeatureHead):
        super(Features, self).__init__()
        self.net_layer_0 = nn.Sequential(*net_layers_FeatureHead[0:6])
        self.net_layer_1 = nn.Sequential(net_layers_FeatureHead[6])
        self.net_layer_2 = nn.Sequential(net_layers_FeatureHead[7])


    def forward(self, x):
        x1 = self.net_layer_0(x)
        
        x2 = self.net_layer_1(x1)
        x3 = self.net_layer_2(x2)

        return x1, x2, x3
    
class Network_Wrapper(nn.Module):
    def __init__(self, net_layers, num_class):
        super().__init__()
        self.Features = Features(net_layers)

        self.max_pool1 = nn.MaxPool2d(kernel_size=28, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=14, stride=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=7, stride=1)

        self.conv_block1 = nn.Sequential(
            BasicConv(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class)
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(1024, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(2048, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class),
        )

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class),
        )

    def forward(self, x):
        x1, x2, x3 = self.Features(x)
        
        # x1,x2,x3=x1.permute(0,3,1,2),x2.permute(0,3,1,2),x3.permute(0,3,1,2)

        map1 = x1.clone()
        x1_ = self.conv_block1(x1)
        x1_ = self.max_pool1(x1_)
        x1_f = x1_.view(x1_.size(0), -1)
        x1_c = self.classifier1(x1_f)

        map2 = x2.clone()
        x2_ = self.conv_block2(x2)
        x2_ = self.max_pool2(x2_)
        x2_f = x2_.view(x2_.size(0), -1)
        x2_c = self.classifier2(x2_f)

        map3 = x3.clone()
        x3_ = self.conv_block3(x3)
        x3_ = self.max_pool3(x3_)
        x3_f = x3_.view(x3_.size(0), -1)
        x3_c = self.classifier3(x3_f)

        x_c_all = torch.cat((x1_f, x2_f, x3_f), -1)
        f = x_c_all.clone()
        x_c_all = self.classifier_concat(x_c_all)

        return x1_c, x2_c, x3_c, x_c_all, map1, map2, map3, f

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
            output_1, output_2, output_3, output_concat, _, _, _, _= net(inputs)
            
            output_1, output_2, output_3, output_concat = \
                output_1[:,0:-1], output_2[:,0:-1], output_3[:,0:-1], output_concat[:,0:-1]

            outputs = output_1 + output_2 + output_3 + output_concat

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




def test(net, criterion, testloader):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    device = torch.device("cuda")

    for batch_idx, (inputs, targets) in enumerate(testloader):      
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        output_1, output_2, output_3, output_concat, _, _, _, _= net(inputs)
        
        output_1, output_2, output_3, output_concat = \
            output_1[:,0:-1], output_2[:,0:-1], output_3[:,0:-1], output_concat[:,0:-1]

        outputs_com = output_1 + output_2 + output_3 + output_concat
 
        loss = criterion(output_concat, targets)

        test_loss += loss.item()
        _, predicted = torch.max(output_concat.data, 1)
        _, predicted_com = torch.max(outputs_com.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct_com += predicted_com.eq(targets.data).cpu().sum()


    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc_en, test_loss

def freeze(net, freeze=True):
    if freeze:
        for param in net.parameters():
            param.requires_grad = False
        net.eval()
    else:
        for param in net.parameters():
            param.requires_grad = True
        net.train()

def train(nb_epoch, batch_size, store_name, start_epoch=0,  num_class=0, data_path = '', 
          validation_folder="validation", inference_folder="test", cycleGan=None):

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
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)  
    
    
    print("Loading validation data...")
    val_set = InMemoryDataset(root=os.path.join(data_path, validation_folder), transform=transform_test)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0)

    print("Loading inference data...")
    infer_set = InMemoryDataset(root=os.path.join(data_path, inference_folder), transform=transform_test)
    infer_loader = DataLoader(infer_set, batch_size=4, shuffle=False, num_workers=0)
    
    

    

    if args.m == "resnet34":
        from models.resnets_mix import Resnet34_mix
        net, g, d = Resnet34_mix(num_class+1, inner_dim=args.inner_dim, generator_wrapper=args.gtype)        
    elif args.m == "resnet18":
        from models.resnets_mix import Resnet18_mix
        net, g, d = Resnet18_mix(num_class+1, inner_dim=args.inner_dim, generator_wrapper=args.gtype)   
    elif args.m == "vgg13":
        from models.vggs_mix import vgg13_mix
        net, g, d = vgg13_mix(num_class+1, inner_dim=args.inner_dim, generator_wrapper=args.gtype) 
    elif args.m == "swins":
        from models.swin_mix import swin_s_mix
        net, g, d = swin_s_mix(num_class+1, inner_dim=args.inner_dim, generator_wrapper=args.gtype)  
    elif args.m == "mobile050":
        from models.mobilenets_mix import mobilenetv2_050_mix
        net, g, d = mobilenetv2_050_mix(num_class+1, inner_dim=args.inner_dim, generator_wrapper=args.gtype) 
    elif args.m == "mobile100":
        from models.mobilenets_mix import mobilenetv2_100_mix
        net, g, d = mobilenetv2_100_mix(num_class+1, inner_dim=args.inner_dim, generator_wrapper=args.gtype)                                         
    elif args.m == "pvt":
        from models.pvt_mix import pvt_mix
        net, g, d = pvt_mix(num_class+1, inner_dim=args.inner_dim, generator_wrapper=args.gtype) 
                                                                               
                                                                                     
    else:
        raise ValueError(f"Unsupported network type: {args.m}. Please choose from: 'resnet34', 'resnet18', 'vgg13', 'swins', 'mobile050', 'mobile100', 'pvt'.")

    
    params = count_parameters(net)
    print(f"Parameters: {params:.2f}M")
    
        


    

    netp = torch.nn.DataParallel(net, device_ids=[0])


    device = torch.device("cuda")
    net.to(device)
    g.to(device)
    d.to(device)
    cycleGan.to(device)
    

    CELoss = nn.CrossEntropyLoss()
    if not args.gloss == "type1b":
        if args.rloss=="mae":
            reg_loss = nn.L1Loss()
        elif args.rloss=="mse":
            reg_loss = nn.MSELoss()
        else:
            raise ValueError("unknown regression loss")
        

    balancer = LossBalancer(init_sigma1=1.0, init_sigma2=1.0, device=device)
    
    MAEloss = nn.L1Loss()
    MSELoss = nn.MSELoss()
    NegLoss = NegativeL1Loss()
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    BCELoss = nn.BCELoss()
    
    if args.gloss in["type4"]:
        un_loss = MultiUncertaintyWeightedLoss()
    elif args.gloss in["type3"]:
        un_loss = UncertaintyWeightedLoss()
    else:
        un_loss = None
    
    
    optimizer = optim.SGD([
        {'params': net.classifier_concat.parameters(), 'lr': 0.002},
        {'params': net.conv_block1.parameters(), 'lr': 0.002},
        {'params': net.classifier1.parameters(), 'lr': 0.002},
        {'params': net.conv_block2.parameters(), 'lr': 0.002},
        {'params': net.classifier2.parameters(), 'lr': 0.002},
        {'params': net.conv_block3.parameters(), 'lr': 0.002},
        {'params': net.classifier3.parameters(), 'lr': 0.002},
        {'params': net.Features.parameters(), 'lr': 0.0002}

    ],
        momentum=0.9, weight_decay=5e-4)
    
    optimizer_g = optim.Adam(
        g.parameters(),
        lr=0.0002, betas=(0.5, 0.999)
    )
    
    optimizer_d = optim.Adam(
        d.parameters(),
        lr=0.0002, betas=(0.5, 0.999)
    )    

    max_val_acc = 0
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    for epoch in tqdm(range(start_epoch, nb_epoch), desc="Epoch Progress"):
        net.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        train_loss5 = 0
        correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), 
                                                total=len(train_loader), 
                                                desc=f"Batch Progress (Epoch {epoch})", 
                                                leave=False):
            idx = batch_idx
            if inputs.shape[0] < batch_size:
                continue
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)
            
            with torch.no_grad():
                inputs_normal = cycleGan(inputs)
                inputs_normal = inputs_normal.clone().detach()
            
            target_counts = Counter(targets.cpu().numpy())
            min_count = min(target_counts.values())  
                      
            inputs_aux = inputs_normal[0:min_count, :, :, :]
            targets_aux = (torch.ones_like(targets) * num_class)[0:min_count]
            

                        
            
            inputs_ori = inputs.clone().detach()
            
            



            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])
                
            for nlr in range(len(optimizer_g.param_groups)):
                optimizer_g.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])
                
            for nlr in range(len(optimizer_d.param_groups)):
                optimizer_d.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])                


            freeze(netp, True)
            freeze(d, True)
            freeze(g, False)  
            optimizer_g.zero_grad()
            _, _, _, _, map1_ori, map2_ori, map3_ori, _ = netp(inputs_ori)
            if args.pad_normal:
                inputs_g = g(inputs_normal, [map1_ori,map2_ori,map3_ori])
            else:
                inputs_g = g(inputs_ori, [map1_ori,map2_ori,map3_ori])
            
            output_1_g, output_2_g, output_3_g, output_concat_g, map1_g, map2_g, map3_g, _ = netp(inputs_g)
            if args.gloss == "type1":
                loss_g = args.alpha*(CELoss(output_1_g, targets) +\
                            CELoss(output_2_g, targets) +\
                                CELoss(output_3_g, targets)+\
                                    CELoss(output_concat_g, targets))+\
                                        reg_loss(inputs_g, inputs_normal)
            elif args.gloss == "type1b":
                loss_g = CELoss(output_1_g, targets) +\
                            CELoss(output_2_g, targets) +\
                                CELoss(output_3_g, targets)+\
                                    CELoss(output_concat_g, targets)+\
                                        MSELoss(inputs_g, inputs_normal) + MAEloss(inputs_g, inputs_normal) + loss_fn_alex(inputs_g,inputs_normal).mean() 
            elif args.gloss == "type1c":
                loss_g1 = CELoss(output_1_g, targets) +\
                            CELoss(output_2_g, targets) +\
                                CELoss(output_3_g, targets)+\
                                    CELoss(output_concat_g, targets)
                loss_g2 = reg_loss(inputs_g, inputs_normal) 


                loss_g = balancer.compute_loss(loss_g1, loss_g2)
                                                                                
            elif args.gloss == "type1d":
                loss_g1 = CELoss(output_1_g, targets) +\
                            CELoss(output_2_g, targets) +\
                                CELoss(output_3_g, targets)+\
                                    CELoss(output_concat_g, targets)
                loss_g2 =  MSELoss(inputs_g, inputs_normal) + MAEloss(inputs_g, inputs_normal) + loss_fn_alex(inputs_g,inputs_normal).mean()                                     
                loss_g = balancer.compute_loss(loss_g1, loss_g2) 
                     
            elif args.gloss == "type1e":
                loss_g1 = CELoss(output_1_g, targets) +\
                            CELoss(output_2_g, targets) +\
                                CELoss(output_3_g, targets)+\
                                    CELoss(output_concat_g, targets)
                loss_g2 =  MSELoss(inputs_g, inputs_normal) + MAEloss(inputs_g, inputs_normal) + loss_fn_alex(inputs_g,inputs_normal).mean() +\
                    style_loss(inputs_g, inputs_normal) +  color_difference_loss(inputs_g, inputs_normal)                                  
                loss_g = balancer.compute_loss(loss_g1, loss_g2) 
                
                                                     
                loss_g = balancer.compute_loss(loss_g1, loss_g2)   
                                           
                                                                                                                           
            elif args.gloss == "type2":
                loss_g = args.alpha*(CELoss(output_1_g, targets) +\
                        CELoss(output_2_g, targets) +\
                            CELoss(output_3_g, targets)+\
                                CELoss(output_concat_g, targets))+\
                                    args.beta*(reg_loss(map1_g, map1_ori)+reg_loss(map2_g, map2_ori)+reg_loss(map3_g, map3_ori))+\
                                    reg_loss(inputs_g, inputs_normal)
            elif args.gloss == "type3":
                loss_g = un_loss(
                                output_1_g=output_1_g, 
                                output_2_g=output_2_g, 
                                output_3_g=output_3_g, 
                                output_concat_g=output_concat_g, 
                                targets=targets, 
                                inputs_g=inputs_g, 
                                inputs_normal=inputs_normal, 
                                CELoss=CELoss, 
                                reg_loss=reg_loss
                            )  
            elif args.gloss == "type4":
                loss_g = un_loss(
                    output_1_g, output_2_g, output_3_g, output_concat_g, targets, 
                    map1_g, map1_ori, map2_g, map2_ori, map3_g, map3_ori, 
                    inputs_g, inputs_normal, 
                    CELoss, reg_loss
                    )                                                                  
            else:
                raise ValueError("unkown gloss type")
            
            loss_g.backward()
            optimizer_g.step()
            
            if args.mix_type == "soft":
                inputs_aug,_ = generate_softmasked_inputs(inputs_normal, inputs_g, inputs)
            elif args.mix_type == "hard":
                inputs_aug = generate_masked_inputs(inputs_normal, inputs_g, inputs, args.mask_thd)
            else:
                raise ValueError("unkown mix type")
            
            
            freeze(netp, False)
            freeze(g, True)
            freeze(d, True)      

                            
            optimizer.zero_grad()
            if epoch>=args.mix_ech:
                inputs_ = mix_inputs(inputs_aug, inputs_ori, args.mix_rate)
                inputs_ = torch.cat([inputs_,inputs_aux],dim=0)                
            else:
                inputs_ = inputs_ori.clone().detach()
                inputs_ = torch.cat([inputs_,inputs_aux],dim=0)            

            targets_ = torch.cat([targets,targets_aux],dim=0)
            
            output_1, output_2, output_3, output_concat, _, _, _, _ = netp(inputs_)
            concat_loss = CELoss(output_1, targets_) + CELoss(output_2, targets_) + CELoss(output_3, targets_) + CELoss(output_concat, targets_) 
            concat_loss.backward()
            optimizer.step()  
            
         
            _, predicted = torch.max(output_concat.data, 1)
            total += targets_.size(0)
            correct += predicted.eq(targets_.data).cpu().sum()
            
            loss1 = concat_loss
            loss2 = concat_loss
            loss3 = concat_loss
            
            

            
            

            train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            train_loss4 += concat_loss.item()


        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_ATT: %.5f | Loss_concat: %.5f |\n' % (
                epoch, train_acc, train_loss, train_loss1 / (idx + 1), train_loss2 / (idx + 1), train_loss3 / (idx + 1),
                train_loss4 / (idx + 1), train_loss5 / (idx + 1)))

        if epoch < 100 or epoch >= 100:
            val_acc_com, val_loss = test(net, CELoss, val_loader)
            if val_acc_com > max_val_acc:
                max_val_acc = val_acc_com
                net.cpu()
                torch.save(net, './' + store_name + '/model.pth')
                net.to(device)
                g.cpu()
                torch.save(g, f'./{store_name}/model_g.pth')
                g.to(device)
              
            with open(exp_dir + '/results_test.txt', 'a') as file:
                file.write('Iteration %d, test_acc_combined = %.5f, test_loss = %.6f\n' % (
                epoch, val_acc_com, val_loss))
        else:
            net.cpu()
            torch.save(net, './' + store_name + '/model.pth')
            net.to(device)
            
            g.cpu()
            torch.save(g, f'./{store_name}/model_g.pth')
            g.to(device)

                
    trained_model = torch.load('./' + store_name + '/model.pth')
    accuracy, f1_micro, f1_macro, auc_micro, auc_macro, precision_micro, precision_macro, recall_micro, recall_macro = inference(trained_model, CELoss, infer_loader)
    with open(exp_dir + '/results_test.txt', 'a') as file:
        file.write('Inference Results: Accuracy = %.5f, F1_micro = %.5f, F1_macro = %.5f, Auc_micro = %.5f, Auc_macro = %.5f \n' % (
        accuracy, f1_micro, f1_macro, auc_micro, auc_macro))
        file.write('Inference Results: precision_micro = %.5f, precision_macro = %.5f, recall_micro = %.5f, recall_macro = %.5f \n' % (
        precision_micro, precision_macro, recall_micro, recall_macro))
         
                     
            

if __name__ == '__main__':
    import cycle_models
    netG_A2B = cycle_models.Generator(3, 3)
    netG_A2B.load_state_dict(torch.load("CycleGan_weights/weights/netG_A2B.pth"))
    netG_A2B.eval()
    
    
        
    data_path, num_class, inference_folder = set_dataset(args.d, args.f)
    
    results_path ='results'
    mk_dir(results_path)
    pyname = os.path.basename(__file__).replace('.py','').replace('train_','')
    task_result_path = os.path.join(results_path, 'proposed')
    mk_dir(task_result_path)    
    experiment_result_path = \
                        os.path.join(task_result_path, "{}_d_{}_m_{}_inner_dim_{}_s_{}_i_{}_f_{}_gloss_{}_regloss_{}_alpha_{}_beta_{}_generator_{}_pad_normal_{}_mix_rate_{}_mix_ech_{}_mask_thd_{}_mix_type_{}"
                                     .format(pyname,args.d,args.m,args.inner_dim,seed,input_size,args.f,args.gloss,args.rloss,args.alpha,args.beta,args.gtype,args.pad_normal,args.mix_rate,args.mix_ech,args.mask_thd, args.mix_type))



    mk_dir(experiment_result_path)
    
        
    train(nb_epoch=100,            
             batch_size=16,        
             store_name=experiment_result_path,    
             start_epoch=0,       
             num_class=num_class,
             data_path=data_path,
             validation_folder="validation",
             inference_folder=inference_folder,
             cycleGan = netG_A2B)        