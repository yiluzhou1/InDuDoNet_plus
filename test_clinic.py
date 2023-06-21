import os
import argparse
import numpy as np
import torch
from sklearn.cluster import  k_means
import scipy
from scipy import ndimage
import scipy.io as sio
import PIL
from PIL import Image
from CLINIC_metal.preprocess_clinic.preprocessing_clinic import clinic_input_data
from network.indudonet_plus import InDuDoNet_plus

import nibabel
import time
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

sigma = 1
smFilter =  sio.loadmat('deeplesion/gaussianfilter.mat')['smFilter']

miuAir = 0
miuWater=0.192
starpoint = np.zeros([3, 1])
starpoint[0] = miuAir
starpoint[1] = miuWater
starpoint[2] = 2 * miuWater
def image_get_minmax():
    return 0.0, 1.0
def proj_get_minmax():
    return 0.0, 4.0

def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max) 
    data = (data - data_min) / (data_max - data_min)
    data = data * 255.0
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)),0)
    return data

def test_image(allXma, allXLI, allM, allSma, allSLI, allTr, vol_idx, slice_idx):
    Xma = allXma[vol_idx][...,slice_idx]
    XLI = allXLI[vol_idx][...,slice_idx]
    M = allM[vol_idx][...,slice_idx]
    Sma = allSma[vol_idx][...,slice_idx]
    SLI = allSLI[vol_idx][...,slice_idx]
    Tr = allTr[vol_idx][...,slice_idx]
    #jow
    M = np.array(Image.fromarray(M).resize((416, 416), PIL.Image.Resampling.BILINEAR)) # maybe mismatch?
    Xprior = nmar_prior(XLI, M)
    Xprior = normalize(Xprior, image_get_minmax())  # *255
    Xma = normalize(Xma, image_get_minmax())  # *255
    XLI = normalize(XLI, image_get_minmax())
    Sma = normalize(Sma, proj_get_minmax())
    SLI = normalize(SLI, proj_get_minmax())
    Tr = 1-Tr.astype(np.float32)
    Tr = np.expand_dims(np.transpose(np.expand_dims(Tr, 2), (2, 0, 1)),0) # 1*1*h*w
    Mask = M.astype(np.float32)
    Mask = np.expand_dims(np.transpose(np.expand_dims(Mask, 2), (2, 0, 1)),0)
    return torch.Tensor(Xma).cuda(), torch.Tensor(XLI).cuda(), torch.Tensor(Mask).cuda(), \
       torch.Tensor(Sma).cuda(), torch.Tensor(SLI).cuda(), torch.Tensor(Tr).cuda(), torch.Tensor(Xprior).cuda()

def print_network(name, net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('name={:s}, Total number={:d}'.format(name, num_params))

def nmarprior(im,threshWater,threshBone,miuAir,miuWater,smFilter):
    imSm = ndimage.filters.convolve(im, smFilter, mode='nearest')
    priorimgHU = imSm
    priorimgHU[imSm <= threshWater] = miuAir
    h, w = imSm.shape[0], imSm.shape[1]
    priorimgHUvector = np.reshape(priorimgHU, h*w)
    region1_1d = np.where(priorimgHUvector > threshWater)
    region2_1d = np.where(priorimgHUvector < threshBone)
    region_1d = np.intersect1d(region1_1d, region2_1d)
    priorimgHUvector[region_1d] = miuWater
    priorimgHU = np.reshape(priorimgHUvector,(h,w))
    return priorimgHU
def nmar_prior(XLI, M):
    XLI[M == 1] = 0.192
    h, w = XLI.shape[0], XLI.shape[1]
    im1d = XLI.reshape(h * w, 1)
    best_centers, labels, best_inertia = k_means(im1d, n_clusters=3, init=starpoint, max_iter=300)
    threshBone2 = np.min(im1d[labels ==2])
    threshBone2 = np.max([threshBone2, 1.2 * miuWater])
    threshWater2 = np.min(im1d[labels == 1])
    imPriorNMAR = nmarprior(XLI, threshWater2, threshBone2, miuAir, miuWater, smFilter)
    return imPriorNMAR

def metal_artifact_reduction(data_path = 'CLINIC_metal/test/', save_path = "results/CLINIC_metal/" ):
    # data_path can be a folder or a single file
    # Build model
    print('Loading InDuDoNet_plus model ...\n')
    if 'opt' not in locals(): # running in another python file, not in command line
        opt = {
            'S': 10, # the number of total iterative stages
            'num_channel': 32, # the number of dual channels
            'T': 4, # the number of ResBlocks in every ProxNet
            'eta1': 1, # initialization for stepsize eta1
            'eta2': 5, # initialization for stepsize eta2
            'alpha': 0.5, # initialization for weight factor
            'model_dir': "pretrained_model/InDuDoNet+_latest.pt", # path to model and log files
            'data_path': data_path, # path to data
            'save_path': save_path, # path to save results
            'keep_originalshape': True, # whether to keep the original image shape
        }
    
    net = InDuDoNet_plus(opt).cuda()
    print_network("InDuDoNet", net)
    net.load_state_dict(torch.load(opt.model_dir))
    net.eval()
    time_test = 0
    count = 0
    print('--------------load---------------all----------------nii-------------')
    allXma, allXLI, allM, allSma, allSLI, allTr, allaffine, allfilename, alloriginalshape = clinic_input_data(opt.data_path)
    print('--------------test---------------all----------------nii-------------')
    for vol_idx in range(len(allXma)):
        print('test %d th volume.......' % vol_idx)
        num_s = allXma[vol_idx].shape[2]
        pre_Xout = np.zeros_like(allXma[vol_idx])
        pre_name = allfilename[vol_idx]
        originalshape = alloriginalshape[vol_idx]
        original_volume = np.zeros((originalshape[0], originalshape[1], num_s), dtype='float32')
        for slice_idx in range(num_s):
            Xma, XLI, M, Sma, SLI, Tr, Xprior  = test_image(allXma, allXLI, allM, allSma, allSLI, allTr, vol_idx, slice_idx)

            with torch.no_grad():
                if opt.use_GPU:
                    torch.cuda.synchronize()
                start_time = time.time()
                ListX, ListS, ListYS= net(Xma, XLI, Sma, SLI, Tr, Xprior)
            end_time = time.time()
            dur_time = end_time - start_time
            time_test += dur_time
            print('Times: ', dur_time)
            Xout= ListX[-1] / 255.0
            pre_Xout[..., slice_idx] = Xout.data.cpu().numpy().squeeze()
            # Convert to original size
            original_volume[..., slice_idx] = np.array(Image.fromarray(pre_Xout[..., slice_idx]).resize((originalshape[1], originalshape[0]), PIL.Image.Resampling.BILINEAR))
        # Save nii
        if opt.keep_originalshape == False:
            # using the default shape
            nibabel.save(nibabel.Nifti1Image(pre_Xout, allaffine[vol_idx]), Pred_nii + pre_name)
        else: # keep original shape
            nibabel.save(nibabel.Nifti1Image(original_volume, allaffine[vol_idx]), Pred_nii + pre_name)

    #     if vol_idx == 1:
    #         img = nibabel.load(Pred_nii + pre_name)
    #         qform = img.get_qform()
    #         img.set_qform(qform)
    #         sfrom = img.get_sform()
    #         img.set_sform(sfrom)
    #         nibabel.save(img, Pred_nii + pre_name)
    #     count += 1
    # print('Avg.time={:.4f}'.format(time_test / count))

def reload_nii(nii_path):
    img = nibabel.load(nii_path)
    qform = img.get_qform()
    img.set_qform(qform)
    sfrom = img.get_sform()
    img.set_sform(sfrom)
    nibabel.save(img, nii_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YU_Test")
    parser.add_argument("--model_dir", type=str, default="./pretrained_model/InDuDoNet+_latest.pt", help='path to model and log files')
    parser.add_argument("--data_path", type=str, default="CLINIC_metal/test/", help='path to training data')
    parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
    parser.add_argument("--save_path", type=str, default="results/CLINIC_metal/", help='path to training data')
    parser.add_argument("--keep_originalshape", type=str, default=False, help='whether to keep the original shape of the image')
    parser.add_argument('--num_channel', type=int, default=32, help='the number of dual channels')
    parser.add_argument('--T', type=int, default=4, help='the number of ResBlocks in every ProxNet')
    parser.add_argument('--S', type=int, default=10, help='the number of total iterative stages')
    parser.add_argument('--eta1', type=float, default=1, help='initialization for stepsize eta1')
    parser.add_argument('--eta2', type=float, default=5, help='initialization for stepsize eta2')
    parser.add_argument('--alpha', type=float, default=0.5, help='initialization for weight factor')
    opt = parser.parse_args()
    def mkdir(path):
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
            print("---  new folder...  ---")
            print("---  " + path + "  ---")
        else:
            print("---  There exsits folder " + path + " !  ---")
    Pred_nii = opt.save_path +'/X_mar/'
    mkdir(Pred_nii)    

    metal_artifact_reduction()
    # reload_nii(r"C:\Users\Image\jow\code\InDuDoNet_plus\results\CLINIC_metal\X_mar\9_image_9_xa_3d_mask_40_degs_512_new.nii.gz")

