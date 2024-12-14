import torch
from torchvision.datasets import ImageFolder
from resnet import wide_resnet50_2
from de_resnet_new import de_wide_resnet50_2
from torch.nn import functional as F
import torchvision.transforms as transforms
from dataset import AugMixDatasetMVTec
from tqdm import tqdm
import os
from util import Tee
import sys


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                        b[item].view(b[item].shape[0], -1)))
    return loss


def loss_fucntion_last(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    item = 0
    loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                    b[item].view(b[item].shape[0], -1)))
    return loss


def loss_fucntion_l2(a, b):
    mse_loss = torch.nn.MSELoss()
    loss = mse_loss(a[0], a[1])
    loss += mse_loss(b[0], b[1])
    return loss



def train(_class_):
    print("###{:s}###".format(_class_))
    if _class_ in ["capsule", "pill", "screw", "zipper"]:
        epochs = 40
    else:
        epochs = 20
    learning_rate = 0.005
    batch_size = 8
    image_size = 256

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    resize_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
    ])
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train),
    ])

    train_path = '/data/mvtec/' + _class_ + '/train'  # update here
    train_data = ImageFolder(root=train_path, transform=resize_transform)
    train_data = AugMixDatasetMVTec(train_data, preprocess)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=learning_rate,
                                 betas=(0.5, 0.999))

    for epoch in range(epochs):
        print("########################################")
        bn.train()
        decoder.train()
        loss_list = []
        i = 0
        for normal, augmix_img, gray_img in train_dataloader:
            normal = normal.to(device)
            inputs_normal = encoder(normal)
            bn_normal = bn(inputs_normal)
            outputs_normal, ori_normal, a_normal, b_normal = decoder(bn_normal)

            augmix_img = augmix_img.to(device)
            inputs_augmix = encoder(augmix_img)
            bn_augmix = bn(inputs_augmix)
            outputs_augmix, ori_augmix, a_augmix, b_augmix = decoder(bn_augmix)

            gray_img = gray_img.to(device)
            inputs_gray = encoder(gray_img)
            bn_gray = bn(inputs_gray)

            loss_bn = loss_fucntion([bn_normal], [bn_augmix]) + loss_fucntion([bn_normal], [bn_gray])
            outputs_gray, ori_gray, a_gray, b_gray = decoder(bn_gray)

            loss_last_nor = loss_fucntion_last(outputs_normal, outputs_augmix) + loss_fucntion_last(outputs_normal,
                                                                                                outputs_gray)
            loss_last_ori = loss_fucntion_last(ori_normal, ori_augmix) + loss_fucntion_last(ori_normal,
                                                                                                ori_gray)

            loss_mse = loss_fucntion_l2(a_normal, b_normal) + loss_fucntion_l2(a_augmix, b_augmix) + loss_fucntion_l2(a_gray, b_gray)

            loss_last = loss_last_ori + 1 * loss_last_nor + loss_mse * 0.02

            loss_normal = loss_fucntion(inputs_normal, outputs_normal) \
                          + 0.05 * (loss_fucntion(inputs_augmix, outputs_augmix)) \
                          + 0.05 * (loss_fucntion(inputs_gray, outputs_gray))


            loss = loss_normal * 0.9 + loss_bn * 0.05 + loss_last * 0.05

            print("epoch {:d}/{:d} iter {:d}/{:d} loss={:.4f}".format(epoch + 1, epochs, i + 1, len(train_dataloader), loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            i = i + 1
        if (epoch + 1) % 20 == 0:
            ckp_path = save_path + 'mvtec_DINL_' + str(_class_) + '_' + str(epoch) + '.pth'
            torch.save({'bn': bn.state_dict(),
                        'decoder': decoder.state_dict()}, ckp_path)

    return


if __name__ == '__main__':
    save_path = './checkpoints/output/'  # change here
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    item_list = ['carpet', 'leather', 'grid', 'tile', 'wood', 'bottle', 'hazelnut', 'cable', 'capsule',
                 'pill', 'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper']
    sys.stdout = Tee(save_path + "out_train.txt")
    for i in item_list:
        train(i)
