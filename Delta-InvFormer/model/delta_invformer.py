import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  
import pytorch_lightning as pl
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation
from transformers.modeling_outputs import SemanticSegmenterOutput
from einops import rearrange
import matplotlib.pyplot as plt
from medpy import metric
import scipy.io
import h5py
from PIL import Image 
from thop import profile
from model.DiffTransformer.diff_transformers import SpatialBlock,TemporalBlock

mean = 41.744637110427604
segma = 47.91132784092548
                 
W = h5py.File("your weight matrix path",'r')
W = W['W']
W = np.transpose(W)
W = torch.from_numpy(W)
W = W.to('cuda')

mask = scipy.io.loadmat("your mask path")
mask = np.array(mask['mask'])
mask = torch.from_numpy(mask).float()
mask = mask.to('cuda')
mask = mask.to(dtype=torch.float32)

class FlopsModel(nn.Module):
    def __init__(self, backbone, projection):
        super().__init__()
        self.backbone = backbone
        self.projection = projection
        del self.backbone.pretrained_model

    def forward(self, x):
        outputs = self.backbone(x)
        print(outputs.logits.shape)
        logits = outputs.logits
        inversion_outputs = self.projection(logits)
        print(inversion_outputs.shape)
        return inversion_outputs

class MLP(nn.Module):
    def __init__(self,dim,ratio,bias=False):
        super().__init__()
        self.dim = dim
        self.r = ratio
        self.bias = bias
        self.mlp = nn.Sequential(
            nn.Linear(self.dim*2,self.dim*self.r,bias=self.bias),
            nn.GELU(),
            nn.Linear(self.dim*self.r,self.dim,bias=self.bias)
        )
    def forward(self,x):
        x = self.mlp(x)
        return x
    

class Projection(nn.Module):
    def __init__(self, input_dim=32*32, hidden_dim=32*32*4, output_dim=76*99):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=2,stride=2),#b,c,h/2,w/2
            nn.GELU(),
            nn.Conv2d(in_channels=16,out_channels=1,kernel_size=2,stride=2),#b,c,h/4,w/4
        )

        
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        b,c,h,w = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
        x = self.conv(x)
        x = x.view(b,c,-1)
        x = self.model(x)
        x = x.view(b,c,76,99)
        return x


class Delta_InvFormer(nn.Module):
    def __init__(self, frame_size):
        super().__init__()
        self.pretrained_model = SegformerForSemanticSegmentation.from_pretrained("/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c02/DATA/jinliye/sihao/scotch-and-soda-main/model/nvidia/segformer-b3-finetuned-ade-512-512", num_labels=1, ignore_mismatched_sizes=True)
        self.config = self.pretrained_model.config
        self.segformer = self.pretrained_model.segformer
        self.decode_head_1 = self.pretrained_model.decode_head

        self.frame_size = frame_size
        self.dims = [64, 128, 320, 512]
        self.nums_heads_list=[8, 8, 8, 8]
        self.x_sizes = [128, 64, 32, 16]
        self.y_sizes = [128, 64, 32, 16]
        self.frame_sizes=[frame_size] * 4
        self.position = [0,1,2]


        self.T_DiffFormer_1 = TemporalBlock(embed_dim=self.dims[self.position[0]], num_heads=self.nums_heads_list[self.position[0]],layers=3)
        self.T_DiffFormer_2 = TemporalBlock(embed_dim=self.dims[self.position[1]], num_heads=self.nums_heads_list[self.position[1]],layers=3)
        self.T_DiffFormer_3 = TemporalBlock(embed_dim=self.dims[self.position[2]], num_heads=self.nums_heads_list[self.position[2]],layers=3)

        self.S_DiffFormer_1 = SpatialBlock(embed_dim=self.dims[self.position[0]], num_heads=self.nums_heads_list[self.position[0]],layers=3)
        self.S_DiffFormer_2 = SpatialBlock(embed_dim=self.dims[self.position[1]], num_heads=self.nums_heads_list[self.position[1]],layers=3)
        self.S_DiffFormer_3 = SpatialBlock(embed_dim=self.dims[self.position[2]], num_heads=self.nums_heads_list[self.position[2]],layers=3)



        self.conv11 = nn.Conv2d(in_channels=self.dims[self.position[0]],out_channels=self.dims[self.position[0]]//2,kernel_size=1,stride=1)
        self.conv12 = nn.Conv2d(in_channels=self.dims[self.position[0]],out_channels=self.dims[self.position[0]]//2,kernel_size=1,stride=1)
        self.conv21 = nn.Conv2d(in_channels=self.dims[self.position[1]],out_channels=self.dims[self.position[1]]//2,kernel_size=1,stride=1)
        self.conv22 = nn.Conv2d(in_channels=self.dims[self.position[1]],out_channels=self.dims[self.position[1]]//2,kernel_size=1,stride=1)
        self.conv31 = nn.Conv2d(in_channels=self.dims[self.position[2]],out_channels=self.dims[self.position[2]]//2,kernel_size=1,stride=1)
        self.conv32 = nn.Conv2d(in_channels=self.dims[self.position[2]],out_channels=self.dims[self.position[2]]//2,kernel_size=1,stride=1)


    def forward(self, pixel_values, labels = None, output_attentions = None, output_hidden_states = True, return_dict = None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )


        outputs = self.segformer(
            pixel_values,
            output_attentions=True,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        attended_hidden_states = [encoder_hidden_states[0],encoder_hidden_states[1],encoder_hidden_states[2],encoder_hidden_states[3]]

        S_Feature_1 = self.S_DiffFormer_1(encoder_hidden_states[self.position[0]])
        T_Feature_1 = self.T_DiffFormer_1(encoder_hidden_states[self.position[0]])
        S_Feature_1 = self.conv11(S_Feature_1)
        T_Feature_1 = self.conv12(T_Feature_1)
        Fusion_Feature_1 = torch.cat([S_Feature_1,T_Feature_1],dim=1)
        

        S_Feature_2 = self.S_DiffFormer_2(encoder_hidden_states[self.position[1]])
        T_Feature_2 = self.T_DiffFormer_2(encoder_hidden_states[self.position[1]])
        S_Feature_2 = self.conv21(S_Feature_2)
        T_Feature_2 = self.conv22(T_Feature_2)
        Fusion_Feature_2 = torch.cat([S_Feature_2,T_Feature_2],dim=1)
         

        S_Feature_3 = self.S_DiffFormer_3(encoder_hidden_states[self.position[2]])
        T_Feature_3 = self.T_DiffFormer_3(encoder_hidden_states[self.position[2]])
        S_Feature_3 = self.conv31(S_Feature_3)
        T_Feature_3 = self.conv32(T_Feature_3)
        Fusion_Feature_3 = torch.cat([S_Feature_3,T_Feature_3],dim=1) 
        
        attended_hidden_states[self.position[0]] = Fusion_Feature_1
        attended_hidden_states[self.position[1]] = Fusion_Feature_2
        attended_hidden_states[self.position[2]] = Fusion_Feature_3

        logits = self.decode_head_1(attended_hidden_states)

        loss = None

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )


class LightningNetwork(pl.LightningModule):
    def __init__(self, configs: dict):
        super().__init__()
        self.backbone = Delta_InvFormer(configs['time_clips'])
        self.projection = Projection()
        self.experiment_name = configs["experiment_name"]
        self.output_dir = configs["output_dir"]

        self.loss_type = configs["loss_type"]
        self.finetune_learning_rate = configs["finetune_learning_rate"]
        self.scratch_learning_rate = configs["scratch_learning_rate"]
        self.inversion_learning_rate = configs["inversion_learning_rate"]

        self.optimizer = configs["optimizer"]
        self.lr_scheduler = configs["lr_scheduler"]

        self.max_epochs = configs["max_epochs"]
        self.time_clips = configs["time_clips"]


        if self.loss_type == "l1":
            self.segmentation_loss = nn.L1Loss()
        elif self.loss_type == "bce":            
            self.segmentation_loss = nn.BCELoss()
        elif self.loss_type == "mixed":
            from util.loss.losses import lovasz_hinge, binary_xloss, scotch_loss
            self.segmentation_loss = binary_xloss
            self.lovasz_hinge = lovasz_hinge
            self.contrastive_loss = scotch_loss
        else:
            raise Exception

        self.save_hyperparameters()

        self.to_pil = transforms.ToPILImage()
        self.thop = self.compute_flops()
    def get_noise(self, noise_type="gaussian",device='cuda', strength=0.05):
        if noise_type == "gaussian":
            noise = (torch.randn(
                1, 1, 512, 512,
            ) * strength).to(device)

        elif noise_type == "uniform":
            noise = torch.rand(
                1, 1, 512, 512,
            )
            noise = ((noise * 2.0 - 1.0) * strength).to(device)

        elif noise_type == "salt_pepper":
            rand = torch.rand(
                1, 1, 512, 512,
            )

            noise = torch.zeros(
                1, 1, 512, 512,
            )

            salt = rand < (strength / 2.0)
            pepper = (rand >= (strength / 2.0)) & (rand < strength)

            noise[salt] = 1.0
            noise[pepper] = -1.0
            noise = noise.to(device)

        else:
            raise ValueError("noise_type must be 'gaussian', 'uniform', or 'salt_pepper'.")

        noise = noise.repeat(1, 3, 1, 1)
        return noise
    
    def compute_flops(self):
        model = FlopsModel(
            copy.deepcopy(self.backbone).cpu(),
            copy.deepcopy(self.projection).cpu()
        ).eval()

        dummy_input = torch.randn(3, 3, 512, 512)

        with torch.no_grad():
            flop, params = profile(model, inputs=(dummy_input,), verbose=False)
        GFLOPS = flop/1e9
        Params = params/1e6
        print(f"FLOPs:{GFLOPS:.3f},Params:{Params:.3f}")
        return flop

    def forward(self, x: torch.Tensor):  # type: ignore
        print("Didn't implement this forward function.")

    def training_step(self, batch: dict, batch_idx: int):  # type: ignore
        exemplar, labels, inversion = batch['image'], batch['label'], batch["inversion"]

        if len(exemplar.size()) == 5 and len(labels.size()) == 5 and len(inversion.size())==5:
            exemplar = exemplar.flatten(start_dim=0, end_dim=1).contiguous() 
            labels = labels.flatten(start_dim=0, end_dim=1).contiguous() 
            inversion = inversion.flatten(start_dim=0, end_dim=1).contiguous()
        outputs = self.backbone(exemplar)
        logits = outputs.logits
        
        inversion_outputs = self.projection(logits)
        inversion_outputs = inversion_outputs
       

        if self.loss_type == "l1":
            final_outputs = F.sigmoid(final_outputs)
            total_loss = self.segmentation_loss(final_outputs, labels)
        elif self.loss_type == "bce":          
            final_outputs = F.sigmoid(final_outputs)  
            total_loss = self.segmentation_loss(final_outputs, labels.float())
        elif self.loss_type == "mixed":          

            inversion_loss = F.mse_loss(inversion_outputs, (inversion-mean)/segma,reduction='mean')
            total_loss = inversion_loss
            print(f"\ninversion_loss:{inversion_loss}")
        else:
            raise Exception


        return total_loss

    def test_step(self, batch: dict, batch_idx: int):  # type: ignore
        exemplar, exemplar_gt,inversion = batch['image'], batch['label'],batch['inversion']
        img_path_batch, label_path_batch, w_batch, h_batch = batch["image_path"], batch["label_path"], batch["w"], batch["h"]
        
        is_5d = False
        if len(exemplar.size()) == 5 and len(exemplar_gt.size()) == 5 and len(inversion.size())==5:
            is_5d = True
            exemplar = exemplar.view(-1, *exemplar.size()[2:])
            exemplar_gt = exemplar_gt.view(-1, *exemplar_gt.size()[2:])
            inversion = inversion.view(-1, *inversion.size()[2:])
        outputs = self.backbone(exemplar)
        logits = outputs.logits
        inversion_outputs = self.projection(logits)

        for batch_clip_idx in range(0, inversion_outputs.shape[0]):

            ground_truth = exemplar_gt[batch_clip_idx]
            inversion_prediction = inversion_outputs[batch_clip_idx]
            if is_5d:
                batch_num = int(batch_clip_idx / self.time_clips)
                clip_num = int(batch_clip_idx % self.time_clips)
                img_path, _, w, h = img_path_batch[clip_num][batch_num], label_path_batch[clip_num][batch_num], w_batch[clip_num][batch_num], h_batch[clip_num][batch_num]
            else:
                batch_idx = batch_clip_idx
                img_path, _, w, h = img_path_batch[batch_idx], label_path_batch[batch_idx], w_batch[batch_idx], h_batch[batch_idx]
            
            self.save_pred_with_exact_value(ground_truth,inversion_prediction, img_path, w, h)

    def on_test_epoch_end(self):
        shots=["your shot id"]
        for shot in shots:
            print(f"The current shotID:{shot}...")
            gt_dir = "dataset/NFDATASET/test/labels/"+shot # [You might need to change to GT path here if you put your data in a different location]
            pred_dir =  os.path.abspath(os.path.join(self.output_dir, "results", self.experiment_name, 'inversion_image/'+shot))
            ComputeError(gt_dir, pred_dir)
            gt_dir = "dataset/NFDATASET/test/inversion/"+shot # [You might need to change to GT path here if you put your data in a different location]
            pred_dir =  os.path.abspath(os.path.join(self.output_dir, "results", self.experiment_name, 'inversion/'+shot))
            ComputeInversionError(gt_dir, pred_dir,mask)
        return None

    

    def configure_optimizers(self) -> torch.optim.Adam:
        params = [
            {"params": self.backbone.segformer.parameters(), "lr": self.finetune_learning_rate},
            {"params": self.backbone.decode_head_1.parameters(), "lr": self.finetune_learning_rate},
            # Temproal DIFF Transoformer
            {"params": self.backbone.T_DiffFormer_1.parameters(), "lr": self.scratch_learning_rate},
            {"params": self.backbone.T_DiffFormer_2.parameters(), "lr": self.scratch_learning_rate},
            {"params": self.backbone.T_DiffFormer_3.parameters(), "lr": self.scratch_learning_rate},
            # Spatial DIFF Transformer
            {"params": self.backbone.S_DiffFormer_1.parameters(), "lr": self.scratch_learning_rate},
            {"params": self.backbone.S_DiffFormer_2.parameters(), "lr": self.scratch_learning_rate},
            {"params": self.backbone.S_DiffFormer_3.parameters(), "lr": self.scratch_learning_rate},
            # Fusion
            {"params": self.backbone.conv11.parameters(), "lr": self.scratch_learning_rate},
            {"params": self.backbone.conv12.parameters(), "lr": self.scratch_learning_rate},
            {"params": self.backbone.conv21.parameters(), "lr": self.scratch_learning_rate},
            {"params": self.backbone.conv22.parameters(), "lr": self.scratch_learning_rate},
            {"params": self.backbone.conv31.parameters(), "lr": self.scratch_learning_rate},
            {"params": self.backbone.conv32.parameters(), "lr": self.scratch_learning_rate},
            #projection
            {"params": self.projection.parameters(), "lr": self.scratch_learning_rate},
        ]

        # optimizer
        if self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(params)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=5e-4, nesterov=False)
        else:
            raise Exception


        return optimizer

    def log_image(self, image_dict):
        tensorboard = self.logger.experiment
        for image_name, image in image_dict.items():
            tensorboard.add_image("{}".format(image_name), image)

    def reverse_normalize(self, normalized_image):
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        inv_normalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        inv_tensor = inv_normalize(normalized_image)
        return inv_tensor 

    def save_pred_with_exact_value(self, ground_truth,inversion_prediction, img_path, w, h, ):
       
        inversion_pred = inversion_prediction.data.squeeze(0)
        inversion_pred = (inversion_pred*segma + mean)*mask
        inversion_pred = torch.clamp(inversion_pred,min=0,max=600)

        #mask
        inversion_pred = torch.multiply(inversion_pred,mask)

        E = inversion_pred[:-1,:-1].reshape((75*98,1))
        inversion_image = torch.matmul(W,E).reshape((378,405))
        inversion_image = inversion_image.cpu().numpy()
        inversion_image = Image.fromarray(inversion_image).convert('L')
        inversion_pred = inversion_pred.cpu().numpy()
        sub_name = img_path.split('/')

        check_mkdir(os.path.join(self.output_dir,'results',self.experiment_name,'inversion',sub_name[-2]))
        check_mkdir(os.path.join(self.output_dir,'results',self.experiment_name,'inversion_image',sub_name[-2]))

        path_inversion=os.path.join(self.output_dir,'results',self.experiment_name,'inversion',sub_name[-2],sub_name[-1].split('.')[0]+'.mat')
        path_inversion_image=os.path.join(self.output_dir,'results',self.experiment_name,'inversion_image',sub_name[-2],sub_name[-1].split('.')[0]+'.jpg')

        scipy.io.savemat(path_inversion,{'SA_E_result':inversion_pred})
        inversion_image.save(path_inversion_image)

def ComputeError(gt_dir,pred_dir):
        gt_name = os.listdir(gt_dir)
        pred_name = os.listdir(pred_dir)
        print(len(gt_name),len(pred_name))
        error = np.zeros((1,len(pred_name)))
        for i in range(len(pred_name)):
            name = gt_name[i]
            gt_path = os.path.join(gt_dir,name)
            pred_path = os.path.join(pred_dir,name)

            gt = torch.from_numpy(np.array(Image.open(gt_path).convert('L'))).float()
            pred = torch.from_numpy(np.array(Image.open(pred_path).convert('L'))).float()
            error_x = torch.mean((torch.abs(gt-pred)/(gt+1e-6)))
            error[0][i] = error_x
        mean = error.mean()
        std = error.std()
        print("Avg Error{:.3f}%".format(mean*100))
        return mean

def ComputeInversionError(gt_dir,pred_dir,mask):
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    SSIM = StructuralSimilarityIndexMeasure(data_range=600.0)
    gt_name = os.listdir(gt_dir)
    pred_name = os.listdir(pred_dir)
    mse_error = np.zeros((1,len(pred_name)))
    mae_error = np.zeros((1,len(pred_name)))
    ssim = np.zeros((1,len(pred_name)))
    # assert len(gt_name) == len(pred_name)
    for i in range(len(pred_name)):
        name = gt_name[i]
        gt_path = os.path.join(gt_dir,name)
        pred_path = os.path.join(pred_dir,name)
        gt = np.transpose(scipy.io.loadmat(gt_path)['SA_E_result'])
        pred = np.transpose(scipy.io.loadmat(pred_path)['SA_E_result'])

        a = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0)  # (1,1,75,98)
        b = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)  # (1,1,75,98)
        mse_error_x = (np.abs(gt-pred)**2).sum()/mask.sum()
        mae_error_x = (np.abs(gt-pred)).sum()/mask.sum()
        mse_error[0][i] = mse_error_x
        mae_error[0][i] = mae_error_x
        ssim_value = SSIM(a,b)
        ssim[0][i] = ssim_value
    mse = mse_error.mean()
    mae = mae_error.mean()

    print("MSE:{:.3f},MAE:{:.3f},SSIM:{:.3f}".format(mse,mae,ssim.mean()))
    return None

def visualization(W,E):
    E = transforms.Resize((99,76))(E)
    E = E.squeeze(0)*600
    E = E[0:-1,0:-1].reshape((75*98,1))
    image = torch.matmul(W,E)
    image = image.reshape((378,405))
    return image



IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_list(dir):
    images = []

    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)

                subname = path.split('/')
                images.append(os.path.join(subname[-2],subname[-1]))
    return images
