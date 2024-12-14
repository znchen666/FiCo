import torch
from dataset import PACSDatasetOOD
from resnet_TTA import wide_resnet50_2
from de_resnet_new import de_wide_resnet50_2
import torchvision.transforms as transforms
from test import evaluation_ATTA
import os
from util import Tee
import sys
import csv

def test_PACS(_class_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    labels_dict = {
        0: 'dog',
        1: 'elephant',
        2: 'giraffe',
        3: 'guitar',
        4: 'horse',
        5: 'house',
        6: 'person'
    }
    name_dataset = labels_dict[_class_]
    print('Class: ', name_dataset)

    #load data
    size = 256
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    img_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(size),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])

    test_path_ID = '/data/PACS/image_list/photo_crossval_kfold.txt'
    test_path_OOD_art_painting = '/data/PACS/image_list/art_painting_crossval_kfold.txt'
    test_path_OOD_cartoon = '/data/PACS/image_list/cartoon_crossval_kfold.txt'
    test_path_OOD_sketch = '/data/PACS/image_list/sketch_crossval_kfold.txt'

    test_data_ID = PACSDatasetOOD(root=test_path_ID, transform=img_transforms, classname=name_dataset)
    test_data_OOD_art_painting = PACSDatasetOOD(root=test_path_OOD_art_painting, transform=img_transforms, classname=name_dataset)
    test_data_OOD_cartoon = PACSDatasetOOD(root=test_path_OOD_cartoon, transform=img_transforms, classname=name_dataset)
    test_data_OOD_sketch = PACSDatasetOOD(root=test_path_OOD_sketch, transform=img_transforms, classname=name_dataset)

    data_ID_loader = torch.utils.data.DataLoader(test_data_ID, batch_size=1, shuffle=False)
    data_OOD_art_painting_loader = torch.utils.data.DataLoader(test_data_OOD_art_painting, batch_size=1, shuffle=False)
    data_OOD_cartoon_loader = torch.utils.data.DataLoader(test_data_OOD_cartoon, batch_size=1, shuffle=False)
    data_OOD_sketch_loader = torch.utils.data.DataLoader(test_data_OOD_sketch, batch_size=1, shuffle=False)

    ckp_path_decoder = './PACS_checkpoints/output/' + 'PACS_DINL_' + name_dataset + '_19.pth'

    #load model
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    #load checkpoint
    ckp = torch.load(ckp_path_decoder)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'], strict=False)
    bn.load_state_dict(ckp['bn'], strict=False)
    decoder.eval()
    bn.eval()

    lamda = 0.5

    list_results = []
    auroc_sp = evaluation_ATTA(encoder, bn, decoder, data_ID_loader, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='PACS', _class_=_class_)
    print('Sample Auroc_ID {:.4f}'.format(auroc_sp))
    list_results.append(auroc_sp)

    auroc_sp = evaluation_ATTA(encoder, bn, decoder, data_OOD_art_painting_loader, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='PACS', _class_=_class_)
    print('Sample Auroc_art {:.4f}'.format(auroc_sp))
    list_results.append(auroc_sp)

    auroc_sp = evaluation_ATTA(encoder, bn, decoder, data_OOD_cartoon_loader, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='PACS', _class_=_class_)
    list_results.append(auroc_sp)
    print('Sample Auroc_cartoon {:.4f}'.format(auroc_sp))

    auroc_sp = evaluation_ATTA(encoder, bn, decoder, data_OOD_sketch_loader, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='PACS', _class_=_class_)
    list_results.append(auroc_sp)
    print('Sample Auroc_sketch {:.4f}'.format(auroc_sp))
    print(list_results)


    return list_results




save_path = './PACS_checkpoints/output/'  # change here
with open(save_path + 'results_crossval.csv', 'w') as csvfile:
    for i in range(0,7):
        data = test_PACS(i)
        writer = csv.writer(csvfile)
        write_data = i, float(data[0]) * 100, float(data[1]) * 100, float(data[2]) * 100, float(
            data[3]) * 100
        writer.writerow(write_data)

