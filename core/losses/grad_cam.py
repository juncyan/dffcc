import torch
import os
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

import argparse
from torch.utils.data import DataLoader
from common.cdloader import CDReader, TestReader
from changedetection.configs.config import get_config
from changedetection.models.MambaBCD import STMambaBCD
from models.model import ChangeACFM, ChangeMM,ChangeResMM
from models.model2 import ChangeSR



class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()


def normalize(img, mean=[0.485, 0.456, 0.406], std=[1, 1, 1]):
        # return img
        im = img.astype(np.float32, copy=False)
        mean = np.array(mean)[np.newaxis, np.newaxis, :]
        std = np.array(std)[np.newaxis, np.newaxis, :]
        im = im / 255.0
        im -= mean
        im /= std
        return im

# dataset_name = "GVLM_CD"
# dataset_name = "LEVIR_CD"
# dataset_name = "SYSU_CD"
dataset_name = "WHU_BCD"
# dataset_name = "CLCD"

# dataset_name = "MacaoCD"
# dataset_name = "SYSU_CD"
# dataset_name = "S2Looking"

dataset_path = '/mnt/data/Datasets/{}'.format(dataset_name)

parser = argparse.ArgumentParser(description="Training on SYSU/LEVIR-CD+/WHU-CD dataset")
parser.add_argument('--cfg', type=str, default='/home/jq/Code/VMamba/changedetection/configs/vssm1/vssm_base_224.yaml')
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+')

mparas = parser.parse_args()

    # with open(args.test_data_list_path, "r") as f:
    #     # data_name_list = f.read()
    #     test_data_name_list = [data_name.strip() for data_name in f]
    # args.test_data_name_list = test_data_name_list

config = get_config(mparas)
model = ChangeSR(
            pretrained="/home/jq/Code/weights/vssm_base_0229_ckpt_epoch_237.pth",
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            ) 


test_data = TestReader(path_root = dataset_path, mode="val", en_edge=False)
dataloader_test = DataLoader(dataset=test_data, batch_size=1, num_workers=0,
                                  shuffle=True, drop_last=True)

weight_path = r"/home/jq/Code/VMamba/output/whu_bcd/ChangeSR_2024_06_25_16/ChangeSR_best.pth"
checkpoint = torch.load(weight_path)
model_dict = {}
state_dict = model.state_dict()
for k, v in checkpoint.items():
    if k in state_dict:
        model_dict[k] = v
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)

model.cuda()
model.eval()
# model.encoder.layers[0]
target_layers = [model.main_clf]

model_name = "ChangeRS"
save_dir = f"/mnt/data/Results/cam/{dataset_name}/{model_name}/output"
if not os.path.exists(save_dir):
     os.makedirs(save_dir)

color_label = np.array([[0,0,0],[255,255,255]])

for _, (img1, img2, label, name) in enumerate(dataloader_test):
    input_tensor = torch.cat([img1, img2], 1).cuda()
    img1 = img1.squeeze().cpu().numpy()
    img1 = np.transpose(img1, [1,2,0])
    img2 = img2.squeeze().cpu().numpy()
    img2 = np.transpose(img2, [1,2,0])
    label = label.numpy()
    lm = np.argmax(label, 1)
    lm = lm.squeeze()
    lm = color_label[lm] / 255.0
    name = name[0]

    cam = GradCAM(model=model, target_layers=target_layers)

    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers) as cam:
    #   ...

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.

    targets = [SemanticSegmentationTarget(1, label)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img1, grayscale_cam, use_rgb=True)
    img = Image.fromarray(visualization)
    img.save(os.path.join(save_dir, name.replace(".png","_A.png")))
    # cv2.imwrite(os.path.join(save_dir, name.replace(".png","_A.png")), visualization)

    visualization = show_cam_on_image(img2, grayscale_cam, use_rgb=True)
    img = Image.fromarray(visualization)
    img.save(os.path.join(save_dir, name.replace(".png","_B.png")))
    # cv2.imwrite(os.path.join(save_dir, name.replace(".png","_B.png")), visualization)

    visualization = show_cam_on_image(lm, grayscale_cam, use_rgb=True)
    img = Image.fromarray(visualization)
    img.save(os.path.join(save_dir, name.replace(".png","_L.png")))
    # cv2.imwrite(os.path.join(save_dir, name.replace(".png","_L.png")), visualization)