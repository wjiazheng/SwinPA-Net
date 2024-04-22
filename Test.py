import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import imageio
from utils.dataloader import test_dataset
from PIL import Image
from lib.model import SwinPA

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--testsize', type=int,
                        default=384, help='testing size')

    parser.add_argument('--gpu_ids', type=int,
                        default=0, help='epoch number')

    parser.add_argument('--save_model', type=str,
                        default='.....')

    parser.add_argument('--pth_path', type=str,
                        default='.......')

    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu_ids)  # set your gpu device

    for _data_name in ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'Kvasir', 'ETIS-LaribPolypDB']:
        data_path = './dataset/test/{}/'.format(_data_name)

        save_path = '......./{}/'.format(_data_name)
        os.makedirs(save_path, exist_ok=True)
        model = SwinPA()
        model.load_state_dict(torch.load(opt.pth_path, map_location='cuda:0'))
        model.cuda()
        model.eval()
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        test_loader = test_dataset(image_root, gt_root, opt.testsize)

        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            out = model(image)
            out = F.interpolate(out, size=gt.shape, mode='bilinear', align_corners=False)
            res = out.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            Image.fromarray(((res > 0.5) * 255).astype(np.uint8)).save(os.path.join(save_path, name))
            print(i)