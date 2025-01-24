import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
from pdb import set_trace as stx
import numpy as np
import sys
from os import path as osp
cur_path = osp.abspath(
osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
print(cur_path)
sys.path.append(cur_path+'/CR-former')
import np_metric as img_met
import metrics_glf_cr as metrics_glf_cr

import logging
import datetime
from basicsr.models.archs.CR_Former_Net import CR_former
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
import numpy as np
logging.basicConfig(level = logging.CRITICAL,
    format = '%(asctime)s [%(levelname)s] at %(filename)s,%(lineno)d: %(message)s',
    datefmt = '%Y-%m-%d(%a)%H:%M:%S',
    filename = 'test_CR_FORMER_log_'+now+'.txt',
    filemode = 'w')


console = logging.StreamHandler()
console.setLevel(logging.CRITICAL)
formatter = logging.Formatter('[%(levelname)-8s] %(message)s') 
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

parser = argparse.ArgumentParser(description='Test on your own images')
parser.add_argument('--input_dir', default='/data/wy/data/cloud/T-Cloud/test/cloud/', type=str, help='Directory of input images or path of single image')
parser.add_argument('--input_truth_dir', default='/data/wy/data/cloud/T-Cloud/test/reference/', type=str, help='Directory of input images or path of single image')
parser.add_argument('--result_dir', default='./output/t-cloud-test', type=str, help='Directory for restored results')
parser.add_argument('--weights', default='experiments/t-cloud/net_g_best.pth', type=str, help='Path to weights')
parser.add_argument('--opt', default='option/T-CLOUD-CR-Former.yml', type=str, help='Path to config yml file')
args = parser.parse_args()

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_gray_img(filepath):
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)

def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)


inp_dir = args.input_dir
out_dir = os.path.join(args.result_dir)
inp_truth_dir = args.input_truth_dir

os.makedirs(out_dir, exist_ok=True)

extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']

if any([inp_dir.endswith(ext) for ext in extensions]):
    files = [inp_dir]
else:
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(inp_dir, '*.'+ext)))
    files = natsorted(files)

if any([inp_truth_dir.endswith(ext) for ext in extensions]):
    files_truth = [inp_truth_dir]
else:
    files_truth = []
    for ext in extensions:
        files_truth.extend(glob(os.path.join(inp_truth_dir, '*.'+ext)))
    files_truth = natsorted(files_truth)

if len(files) == 0:
    raise Exception(f'No files found at {inp_dir}')

# Get model weights and parameters
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

yaml_file= args.opt

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')

model = CR_former(**x['network_g']) 

checkpoint = torch.load(args.weights)
model.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model.cuda()

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


img_multiple_of = 8

MAE_vs = []
MSE_vs = []
RMSE_vs = []
BRMSE_vs = []
ssim_vs = []
psnr_vs = []
sam_vs = []
# NIQE_vs = []
logging.critical("start testing...")

with torch.no_grad():
    for file_ in tqdm(files):
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = load_img(file_)
            truth_file = file_.replace(args.input_dir,args.input_truth_dir)
            img_truth = load_img(truth_file)


        input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)
      
        # ground truth 
        input_truth = torch.from_numpy(img_truth).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)
       

        
        #model output 
        restored = model(input_)
      
       
        restored = torch.clamp(restored, 0, 1)

        # Unpad the output


        s2img1 = input_truth.clone()  #0-1 value
        fake_img1 = restored.clone()  #  0-1

       
        MAE_v = img_met.cloud_mean_absolute_error(s2img1, fake_img1)
        MSE_v = img_met.cloud_mean_squared_error(s2img1, fake_img1)
        RMSE_v = img_met.cloud_root_mean_squared_error(s2img1, fake_img1)
        BRMSE_v = img_met.cloud_bandwise_root_mean_squared_error(s2img1, fake_img1)
        psnr_v = metrics_glf_cr.PSNR(s2img1, fake_img1)
        ssim_v = metrics_glf_cr.SSIM( fake_img1, s2img1) 
        
      

        MAE_vs.append(np.asarray(MAE_v.cpu()))
        MSE_vs.append(np.asarray(MSE_v.cpu()))
        RMSE_vs.append(np.asarray(RMSE_v.cpu()))
        BRMSE_vs.append(np.asarray(BRMSE_v.cpu()))
        ssim_vs.append(np.asarray(ssim_v.cpu())) 
        psnr_vs.append(np.asarray(psnr_v))

       

        # spectral angle mapper
        mat = s2img1 * fake_img1
        mat = torch.sum(mat, 1)
        mat = torch.div(mat, torch.sqrt(torch.sum(s2img1 * s2img1, 1)))
        mat = torch.div(mat, torch.sqrt(torch.sum(fake_img1 * fake_img1, 1)))
        sam_v = torch.mean(torch.acos(torch.clamp(mat, -1, 1)) * torch.tensor(180) / np.pi)

        if not torch.isnan(sam_v):
              sam_vs.append(np.asarray(sam_v.cpu().detach().numpy()))

       



        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])
        input_1 = input_.permute(0, 2, 3, 1).cpu().detach().numpy()
        input_save = img_as_ubyte(input_1[0])
        input_tr = input_truth.permute(0, 2, 3, 1).cpu().detach().numpy()
        input_tr_save = img_as_ubyte(input_tr[0])

        f = os.path.splitext(os.path.split(file_)[-1])[0]
 
      
        save_img((os.path.join(out_dir, f+'_out.png')), restored)
        save_img((os.path.join(out_dir, f + '_input.png')), input_save)
        save_img((os.path.join(out_dir, f + '_truth.png')), input_tr_save)
        
        logging.critical(
            "MAE_v:{:.6f},MSE_v:{:.6f},RMSE_v:{:.6f},BRMSE_v:{:.6f},psnr_v:{:.6f},ssim_v:{:.6f},sam_v:{:.6f}, file_name:{}".format(
                MAE_v.cpu(), MSE_v.cpu(),
                RMSE_v.cpu(), BRMSE_v.cpu(),
                psnr_v, ssim_v,  sam_v,  file_))

      

    MAE_v = np.mean(MAE_vs)
    MSE_v = np.mean(MSE_vs)
    RMSE_v = np.mean(RMSE_vs)
    BRMSE_v = np.mean(BRMSE_vs)
    ssim_v = np.mean(ssim_vs)
    psnr_v = np.mean(psnr_vs)
    sam_v = np.mean(sam_vs)

    logging.critical(
        "MAE_m:{:.6f},MSE_m:{:.6f},RMSE_m:{:.6f},BRMSE_m:{:.6f},psnr_m:{:.6f},ssim_m:{:.6f},sam_m:{:.6f}".format(
            MAE_v, MSE_v, RMSE_v, BRMSE_v,
            psnr_v, ssim_v, sam_v))

    print(f"\nRestored images are saved at {out_dir}")
