from utils import *


class Discriminator(nn.Module):
    def __init__(self, dims, inner_dim):
        super(Discriminator, self).__init__()

        self.conv_block1 = nn.Sequential(
            BasicConv(dims[0], inner_dim//2, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(inner_dim//2, inner_dim, kernel_size=3, stride=1, padding=1, relu=True)
        )
        
        self.conv_block2 = nn.Sequential(
            BasicConv(dims[1], inner_dim//2, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(inner_dim//2, inner_dim, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(dims[2], inner_dim//2, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(inner_dim//2, inner_dim, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(inner_dim * 3),
            nn.Linear(inner_dim * 3, inner_dim//2),
            nn.BatchNorm1d(inner_dim//2),
            nn.ELU(inplace=True),
            nn.Linear(inner_dim//2, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
      

    def forward(self, features):
        f1 = self.pool(self.conv_block1(features[0])).view(features[0].size(0), -1)
        f2 = self.pool(self.conv_block2(features[1])).view(features[1].size(0), -1)
        f3 = self.pool(self.conv_block3(features[2])).view(features[2].size(0), -1)
        f = torch.concat((f1,f2,f3),dim=1)
        feature = self.classifier_concat(f)
        feature = torch.sigmoid(feature)
        return feature





class Generator(nn.Module):
    def __init__(self, input_dim, num_layers, image_dim=6):
        super(Generator, self).__init__()
        # image_processor = [nn.Conv2d(3, image_dim, 5, stride=1, padding=2),
        #         nn.InstanceNorm2d(image_dim),
        #         nn.ReLU(inplace=True)]   
        # self.image_processor = nn.Sequential(*image_processor)   
        
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
        
        # o = [nn.ReflectionPad2d(3), nn.Conv2d(image_dim*2, 3, 7), nn.Tanh()]        
        # self.output = nn.Sequential(*o)        

    def forward(self, feature):
        # im  = self.image_processor(im)
        feature = self.generator(feature)
        return feature

        # return self.output(torch.cat((im, feature), dim=1)) 



class Generator_Wrapper(nn.Module):
    def __init__(self, input_dims, num_layers, image_dim=6):
        super(Generator_Wrapper, self).__init__()
        image_processor = [nn.Conv2d(3, image_dim, 5, stride=1, padding=2),
                nn.InstanceNorm2d(image_dim),
                nn.ReLU(inplace=True)]   
        self.image_processor = nn.Sequential(*image_processor)   
        
        self.g1 = Generator(input_dims[0], num_layers[0])
        self.g2 = Generator(input_dims[1], num_layers[1])
        self.g3 = Generator(input_dims[2], num_layers[2])  
        
        o = [nn.ReflectionPad2d(3), nn.Conv2d(image_dim*4, 3, 7), nn.Tanh()]        
        self.output = nn.Sequential(*o)        

    def forward(self, im, features):
        im  = self.image_processor(im)
        
        feature1 = self.g1(features[0])
        feature2 = self.g2(features[1])
        feature3 = self.g3(features[2])

        return self.output(torch.cat((im, feature1, feature2, feature3), dim=1)) 



class Generator_Wrapper2(nn.Module):
    def __init__(self, input_dims, num_layers, image_dim=6):
        super(Generator_Wrapper2, self).__init__()
        image_processor = [nn.Conv2d(3, image_dim, 5, stride=1, padding=2),
                nn.InstanceNorm2d(image_dim),
                nn.ReLU(inplace=True)]   
        self.image_processor = nn.Sequential(*image_processor)   
        
        self.g1 = Generator(input_dims[0], num_layers[0])
        self.g2 = Generator(input_dims[1], num_layers[1])
        self.g3 = Generator(input_dims[2], num_layers[2])  
        
        o = [nn.ReflectionPad2d(1), nn.Conv2d(image_dim*4, 3, 3), nn.Tanh()]        
        self.output = nn.Sequential(*o)        

    def forward(self, im, features):
        im  = self.image_processor(im)
        
        feature1 = self.g1(features[0])
        feature2 = self.g2(features[1])
        feature3 = self.g3(features[2])

        return self.output(torch.cat((im, feature1, feature2, feature3), dim=1)) 



class Generator_Wrapper3(nn.Module):
    def __init__(self, input_dims, num_layers, image_dim=6):
        super(Generator_Wrapper3, self).__init__()
        image_processor = [nn.Conv2d(3, image_dim, 5, stride=1, padding=2),
                nn.InstanceNorm2d(image_dim),
                nn.ReLU(inplace=True)]   
        self.image_processor = nn.Sequential(*image_processor)   
        
        self.g1 = Generator(input_dims[0], num_layers[0])
        self.g2 = Generator(input_dims[1], num_layers[1])
        self.g3 = Generator(input_dims[2], num_layers[2])  
        
        o = [nn.Conv2d(image_dim*4, 3, 1), nn.Tanh()]        
        self.output = nn.Sequential(*o)        

    def forward(self, im, features):
        im  = self.image_processor(im)
        
        feature1 = self.g1(features[0])
        feature2 = self.g2(features[1])
        feature3 = self.g3(features[2])

        return self.output(torch.cat((im, feature1, feature2, feature3), dim=1)) 



class Generator_Wrapper3b(nn.Module):
    def __init__(self, input_dims, num_layers, image_dim=6):
        super(Generator_Wrapper3b, self).__init__()
        image_processor = [nn.Conv2d(3, 3, 5, stride=1, padding=2),
                nn.InstanceNorm2d(3),
                nn.ReLU(inplace=True)]   
        self.image_processor = nn.Sequential(*image_processor)   
        
        self.g1 = Generator(input_dims[0], num_layers[0])
        self.g2 = Generator(input_dims[1], num_layers[1])
        self.g3 = Generator(input_dims[2], num_layers[2])  
        
        o = [nn.Conv2d(image_dim*3, 3, 1), nn.Tanh()]        
        self.output = nn.Sequential(*o)        

    def forward(self, im, features):
        # im  = self.image_processor(im)
        
        feature1 = self.g1(features[0])
        feature2 = self.g2(features[1])
        feature3 = self.g3(features[2])

        return im + 0.3*self.output(torch.cat((feature1, feature2, feature3), dim=1)) 




class Generator_Wrapper4(nn.Module):
    def __init__(self, input_dims, num_layers, image_dim=6):
        super(Generator_Wrapper4, self).__init__()
        image_processor = [nn.Conv2d(3, image_dim, 5, stride=1, padding=2),
                nn.InstanceNorm2d(image_dim),
                nn.ReLU(inplace=True)]   
        self.image_processor = nn.Sequential(*image_processor)   
        
        self.g1 = Generator(input_dims[0], num_layers[0])
        self.g2 = Generator(input_dims[1], num_layers[1])
        self.g3 = Generator(input_dims[2], num_layers[2])  
        
        o = [nn.Conv2d(image_dim*3, 3, 1), nn.Tanh()]        
        self.output = nn.Sequential(*o)        

    def forward(self, im, features):
        
        feature1 = self.g1(features[0])
        feature2 = self.g2(features[1])
        feature3 = self.g3(features[2])

        return self.output(torch.cat((feature1, feature2, feature3), dim=1)) 



class Generator_Wrapper5(nn.Module):
    def __init__(self, input_dims, num_layers, image_dim=6):
        super(Generator_Wrapper5, self).__init__()
        image_processor = [nn.Conv2d(3, image_dim, 5, stride=1, padding=2),
                nn.InstanceNorm2d(image_dim),
                nn.ReLU(inplace=True)]   
        self.image_processor = nn.Sequential(*image_processor)   
        
        self.g1 = Generator(input_dims[0], num_layers[0])
        self.g2 = Generator(input_dims[1], num_layers[1])
        self.g3 = Generator(input_dims[2], num_layers[2])  
        
        o = [nn.ReflectionPad2d(3), nn.Conv2d(image_dim*3, 3, 7), nn.Tanh()]        
        self.output = nn.Sequential(*o)        

    def forward(self, im, features):
        # im  = self.image_processor(im)
        feature1 = self.g1(features[0])
        feature2 = self.g2(features[1])
        feature3 = self.g3(features[2])

        return self.output(torch.cat((feature1, feature2, feature3), dim=1)) 


class Generator_Wrapper6(nn.Module):
    def __init__(self, input_dims, num_layers, image_dim=6):
        super(Generator_Wrapper6, self).__init__()
        image_processor = [nn.Conv2d(3, image_dim, 5, stride=1, padding=2),
                nn.InstanceNorm2d(image_dim),
                nn.ReLU(inplace=True)]   
        self.image_processor = nn.Sequential(*image_processor)   
        
        self.g1 = Generator(input_dims[0], num_layers[0])
        self.g2 = Generator(input_dims[1], num_layers[1])
        self.g3 = Generator(input_dims[2], num_layers[2])  
        
        o = [nn.ReflectionPad2d(1), nn.Conv2d(image_dim*3, 3, 3), nn.Tanh()]        
        self.output = nn.Sequential(*o)        

    def forward(self, im, features):
        # im  = self.image_processor(im)
        
        feature1 = self.g1(features[0])
        feature2 = self.g2(features[1])
        feature3 = self.g3(features[2])

        return self.output(torch.cat((feature1, feature2, feature3), dim=1)) 




class Features(nn.Module):
    def __init__(self, net_layers_FeatureHead, breaks):
        super(Features, self).__init__()
        self.net_layer_0 = nn.Sequential(*net_layers_FeatureHead[0:breaks[0]])
        self.net_layer_1 = nn.Sequential(*net_layers_FeatureHead[breaks[0]:breaks[1]])
        self.net_layer_2 = nn.Sequential(*net_layers_FeatureHead[breaks[1]:breaks[2]])
        # print()



    def forward(self, x):
        x1 = self.net_layer_0(x)
        x2 = self.net_layer_1(x1)
        x3 = self.net_layer_2(x2)
        return x1, x2, x3
    
class Network_Wrapper(nn.Module):
    def __init__(self, net_layers, num_class, kernels, breaks, dims, inner_dim):
        super().__init__()
        self.Features = Features(net_layers,breaks)

        self.max_pool1 = nn.MaxPool2d(kernel_size=kernels[0], stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=kernels[1], stride=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=kernels[2], stride=1)

        self.conv_block1 = nn.Sequential(
            BasicConv(dims[0], inner_dim//2, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(inner_dim//2, inner_dim, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(inner_dim),
            nn.Linear(inner_dim, inner_dim//2),
            nn.BatchNorm1d(inner_dim//2),
            nn.ELU(inplace=True),
            nn.Linear(inner_dim//2, num_class)
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(dims[1], inner_dim//2, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(inner_dim//2, inner_dim, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(inner_dim),
            nn.Linear(inner_dim, inner_dim//2),
            nn.BatchNorm1d(inner_dim//2),
            nn.ELU(inplace=True),
            nn.Linear(inner_dim//2, num_class),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(dims[2], inner_dim//2, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(inner_dim//2, inner_dim, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(inner_dim),
            nn.Linear(inner_dim, inner_dim//2),
            nn.BatchNorm1d(inner_dim//2),
            nn.ELU(inplace=True),
            nn.Linear(inner_dim//2, num_class),
        )

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(inner_dim * 3),
            nn.Linear(inner_dim * 3, inner_dim//2),
            nn.BatchNorm1d(inner_dim//2),
            nn.ELU(inplace=True),
            nn.Linear(inner_dim//2, num_class),
        )

    def forward(self, x):
        x1, x2, x3 = self.Features(x)
        
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





def vgg13_mix(num_class, inner_dim=256, generator_wrapper="Generator_Wrapper"):
    
    This_Generator = globals().get(generator_wrapper)
    if This_Generator is None:
        raise ValueError(f"Generator {generator_wrapper} is not defined.")
        
    net = torchvision.models.vgg13(pretrained=True)
    net_layers = list(net.children())
    net_layers = net_layers[0]
    net_layers = list(net_layers.children())
    
    
    kernels = [28, 14, 7]
    dims = [256, 512, 512]
    breaks = [15, 20, 25]
    
    num_g_layers = [3, 4, 5]
    
    
    return Network_Wrapper(net_layers, num_class, kernels, breaks, dims, inner_dim), \
                This_Generator(dims, num_g_layers),\
                    Discriminator(dims, inner_dim)





if __name__ == "__main__":
    net, g, d = vgg13_mix(7)
    inputs = torch.rand(4,3,224,224)
    output_1, output_2, output_3, output_concat, map1, map2, map3, _ = net(inputs)
    inputs_g = g(inputs, [map1,map2,map3])



    print("inputs size:", inputs.size())
    print("output_1 size:", output_1.size())
    print("output_2 size:", output_2.size())
    print("output_3 size:", output_3.size())
    print("output_concat size:", output_concat.size())
    print("map1 size:", map1.size())
    print("map2 size:", map2.size())
    print("map3 size:", map3.size())
    print("inputs_g size:", inputs_g.size())


    print("net.Features.net_layer_0[-1]:", net.Features.net_layer_0[-1])
    print("net.Features.net_layer_1[-1]:", net.Features.net_layer_1[-1])
    print("net.Features.net_layer_2[-1]:", net.Features.net_layer_2[-1])