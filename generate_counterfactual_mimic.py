#%%
import os#; os.environ['CUDA_VISIBLE_DEVICES']='2'
from PIL import Image
from tqdm import tqdm
import argparse
import torchvision.transforms as T
import torch.nn as nn
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from blended_diffusion.guided_diffusion.guided_diffusion.script_util import (
    create_model_and_diffusion,
)
from blended_diffusion.optimization.dff_attack import (
    _map_img,
    _renormalize_gradient,
    compute_lp_dist,
    compute_lp_gradient,
)
import robustness.datasets
import robustness.model_utils
from robustness import imagenet_models

seed = 2228
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

#%%
# hps = get_config(get_arguments())
hps = argparse.Namespace(**{'timestep_respacing': '200',
                            'skip_timesteps': 100,
                            'lp_custom': 1.0,
                            'lp_custom_value': 0.15,
                            'lpips_sim_lambda': 0,
                            'l2_sim_lambda': 0,
                            'l1_sim_lambda': 0,
                            'range_lambda': 0,
                            'deg_cone_projection': 0,
                            'clip_guidance_lambda': 0,
                            'classifier_lambda': 0.1,
                            'num_imgs': 2048,
                            'batch_size': 6,
                            'seed': 2228,
                            'classifier_type': 3,
                            'second_classifier_type': -1,
                            'third_classifier_type': -1,
                            'use_blended': False,
                            'method': 'dvces',
                            'gen_type': 'p_sample',
                            'model_output_size': 256,
                            'enforce_same_norms': True,
                            'prompt': None,
                            'local_clip_guided_diffusion': False,
                            'aug_num': 1,
                            'TV_lambda': 0,
                            'ilvr_multi': 0,
                            'layer_reg': 0,
                            'layer_reg_value': 0,
                            'background_preservation_loss': False,
                            'not_use_init_image': False,
                            'denoise_dist_input': False,
                            'projecting_cone': False,
                            'verbose': False,
                            'invert_mask': False,
                            'enforce_background': True,
                            'gpu_id': 0,
                            'output_path': 'output',
                            'output_file': 'output.png',
                            'iterations_num': 1,
                            'save_video': False,
                            'export_assets': False,
                            'gpu': [0],
                            'mssim_lambda': 0,
                            'ssim_lambda': 0,
                            'quantile_cut': 0,
                            'config': 'imagenet1000.yml',
                            'class_id_spurious': -1,
                            'component_idx_spurious': -1,
                            'start_img_id': -1,
                            'pca_component_lambda': 0,
                            'classifier_size_1': 224,
                            'classifier_size_2': 224,
                            'classifier_size_3': 224,
                            'target_class': -1,
                            'dataset': 'imagenet',
                            'data_folder': '',
                            'project_folder': '.',
                            'consistent': False,
                            'step_lr': -1,
                            'nsigma': 1,
                            'model_types': None,
                            'ODI_steps': -1,
                            'fid_num_samples': 1,
                            'begin_ckpt': 1,
                            'end_ckpt': 1,
                            'adam': False,
                            'D_adam': False,
                            'D_steps': 0,
                            'model_epoch_num': 0,
                            'device_ids': [0],
                            'script_type': 'sampling',
                            'range_t': 0,
                            'down_N': 32,
                            'eps_project': 30,
                            'interpolation_int_1': 3,
                            'interpolation_int_2': 3,
                            'interpolation_int_3': 3,
                            'plot_freq': 5,
                            'world_size': 1,
                            'world_id': 0,
                            'variance': 1.0,
                            'device': torch.device(type='cuda', index=0)})

hps.device_ids = [int(hps.gpu[0])]
device = torch.device('cuda:'+str(hps.gpu[0]))
hps.device = device

torch.manual_seed(hps.seed)
random.seed(hps.seed)
np.random.seed(hps.seed)
    

t = T.Compose([
    T.Resize((256,256)),
    T.ToTensor()
])
class PNP_MIMIC(robustness.datasets.DataSet):
    def __init__(self, data_path, **kwargs):
        ds_name = 'pnp_mimic_dataset'
        ds_kwargs = {
            'num_classes' : 2,
            'mean': torch.tensor([0.5, 0.5, 0.5]),
            'std': torch.tensor([0.2, 0.2, 0.2]),
            'custom_class': None,
            'label_mapping': None,
            'transform_train': t,
            'transform_test': t,
        }
        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        super(PNP_MIMIC, self).__init__(ds_name, data_path, **ds_kwargs)
    
    def get_model(self, arch, pretrained):
        return imagenet_models.__dict__[arch](
            num_classes=self.num_classes
        )

class DiffusionAttack():
    def __init__(self, args, diffusion_ckpt_path, robust_classifier) -> None:
        self.args = args
        self.probs = None
        self.y = None
        self.writer = None
        self.small_const = 1e-12

        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)

        self.model_config = {'image_size': self.args.model_output_size, # 256
                            #  'num_channels': 256, # for imagenet checkpoints/256x256_diffusion_uncond.pt
                             'num_channels' : 128, # for guided-diffusion-cxr C:\Users\lab402\Projects\guided-diffusion-cxr\LOGDIR_for_rsna_lr1e-4_3e-5\model230000.pt
                             'num_res_blocks': 2,
                             'resblock_updown': True,
                             'num_heads': 4,
                             'num_heads_upsample': -1,
                             'num_head_channels': 64,
                            #  'attention_resolutions': '32,16,8', # for imagenet checkpoints/256x256_diffusion_uncond.pt
                             'attention_resolutions' : '16, 8', # for guided-diffusion-cxr C:\Users\lab402\Projects\guided-diffusion-cxr\LOGDIR_for_rsna_lr1e-4_3e-5\model230000.pt
                             'channel_mult': '',
                             'dropout': 0.0,
                             'class_cond': False,
                             'use_checkpoint': False,
                             'use_scale_shift_norm': True,
                             'use_fp16': True, # True,
                             'use_new_attention_order': False,
                             'learn_sigma': True,
                             'diffusion_steps': 1000,
                             'noise_schedule': 'linear',
                             'timestep_respacing': "200", #self.args.timestep_respacing,
                             'use_kl': False,
                             'predict_xstart': False,
                             'rescale_timesteps': True,
                             'rescale_learned_sigmas': False}
        self.diffusion_ckpt_path = diffusion_ckpt_path
        # self.robust_classifier_path = robust_classifier_path

        self.device = self.args.device
        print("Using device:", self.device)

        
        ############### DIFFUSION MODELS ##############
        
        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        
        ###### CUSTOM DIFFUSION MODELS ######
        self.model.load_state_dict(
            torch.load(self.diffusion_ckpt_path, map_location='cpu')
        )
        self.model.requires_grad_(False).eval().to(self.device)

        if args.device_ids is not None and len(args.device_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=args.device_ids)

        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()
                
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()
        args.device = self.device

        #########################################################
        ##################### CLASSIFIERS ####################
        #########################################################
        self.classifier = robust_classifier
        self.classifier.to(self.device)
        self.classifier.eval()



    def _compute_probabilities(self, x, classifier):
        # logits = classifier(_map_img(x))
        logits, _ = classifier(_map_img(x))
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return log_probs, probs

    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()

        return unscaled_timestep


    def perturb(self, x, y):
        self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        if x.shape[-1] != self.model_config["image_size"]:
            x = T.Resize(self.image_size)(x)
            print('shapes x after', x.shape)
        self.init_image = (x.to(self.device).mul(2).sub(1).clone())

        def cond_fn_clean(x, t, y=None, eps=None):
            grad_out = torch.zeros_like(x)
            x = x.detach().requires_grad_()
            t = self.unscale_timestep(t)
            with torch.enable_grad():
                out = self.diffusion.p_mean_variance(
                    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                )
                x_in = out["pred_xstart"]

            # compute classifier gradient
            keep_denoising_graph = self.args.denoise_dist_input
            with torch.no_grad():
                if self.args.classifier_lambda != 0:
                    with torch.enable_grad():
                        log_probs_1, probs_1 = self._compute_probabilities(x_in, self.classifier)
                        target_log_confs_1 = log_probs_1[range(1), y.view(-1)]
                        grad_class = torch.autograd.grad(target_log_confs_1.mean(), x,
                                                        retain_graph=keep_denoising_graph)[0]

                    if self.args.enforce_same_norms:
                        grad_, norm_ = _renormalize_gradient(grad_class, eps)
                        grad_class = self.args.classifier_lambda * grad_
                    else:
                        grad_class *= self.args.classifier_lambda

                    grad_out += grad_class

                # distance gradients
                if self.args.lp_custom: # and self.args.range_t < self.tensorboard_counter:
                    if not keep_denoising_graph:
                        diff = x_in - self.init_image
                        lp_grad = compute_lp_gradient(diff, self.args.lp_custom)
                    else:
                        with torch.enable_grad():
                            diff = x_in - self.init_image
                            lp_dist = compute_lp_dist(diff, self.args.lp_custom)
                            lp_grad = torch.autograd.grad(lp_dist.mean(), x)[0]
                    if self.args.quantile_cut != 0:
                        pass

                    if self.args.enforce_same_norms:
                        grad_, norm_ = _renormalize_gradient(lp_grad, eps)
                        lp_grad = self.args.lp_custom_value * grad_
                    else:
                        lp_grad *= self.args.lp_custom_value

                    grad_out -= lp_grad
                    
            return grad_out

        # gen_func = self.diffusion.p_sample_loop_progressive

        samples = self.diffusion.p_sample_loop_progressive(
            model = self.model,
            shape = (1, 3, 256, 256),
            clip_denoised=False,
            model_kwargs={
                "y": torch.tensor(y, device=self.device, dtype=torch.long)
                # torch.zeros([self.args.batch_size], device=self.device, dtype=torch.long)
            },
            cond_fn=cond_fn_clean,
            progress=False,
            skip_timesteps=self.args.skip_timesteps,
            init_image=self.init_image,
            postprocess_fn=None,
            randomize_class=False,
            resizers=None,
            range_t=self.args.range_t,
            eps_project=self.args.eps_project,
            ilvr_multi=self.args.ilvr_multi,
            seed=self.args.seed
        )

        total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1
        
        for i, sample in enumerate(samples):
            
            # pred = self.classifier(sample['pred_xstart'])
            # pred_for_target_class = pred[0][y.item()].item()
                        
            # fig, ax = plt.subplots(1, 1, figsize=(4,4))
            # arr = np.transpose(sample['pred_xstart'][0].add(1).div(2).clamp(0,1).detach().cpu().numpy(), (1,2,0))
            # ax.imshow(arr)
            # ax.set_title("target class pred: " + f"{pred_for_target_class:.4f}")
            # plt.show()
            # fig.savefig(f'./reverse_diffusion_steps_images_nonrobust_dino/{i}.jpg')
            
            if i == total_steps:
                sample_final = sample
                
        return sample_final["pred_xstart"].add(1).div(2).clamp(0, 1)

def generate_counterfactual(attack_module, image, target_class):
    image = image.unsqueeze(0)
    target = torch.as_tensor(int(target_class)).unsqueeze(0)
    adv_sample = attack_module.perturb(image, target)
    arr = adv_sample[0].detach().cpu().numpy()
    return arr





#%%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diffusion_ckpt_path', type=str,
                        default='/media/wonjun/m2/repos/pnp/guided-diffusion/training_checkpoints_oldcode_1171_1171/model050000.pt',
                        help='Path to pretraind diffusion model')
    parser.add_argument('--robust_classifier_path', type=str,
                        default = '/media/wonjun/m2/repos/pnp/classifiers/robust_red_936_937/30_checkpoint.pt',
                        help='Path to pretrained robust classifier')
    parser.add_argument('--mimic_path', type=str,
                        default='/media/wonjun/HDD8TB/mimic-cxr-jpg-resized512',
                        help='Path to MIMIC-CXR-JPG root directory')
    parser.add_argument('--dicom_id', type=str,
                        default='66b67252-000e4090-269c617a-1f7c366b-c07fbb46',
                        help='DICOM id of original input image')
    parser.add_argument('--labels', type=str,
                        default="{1:'Edema', 0:'No Finding'}",
                        help='dictionary (written out in string) containing the class labels used for training the robust classifier')
    parser.add_argument('--target_class', default=0, choices=[0,1], type=int,
                        help='class label of counterfactual')
    args = parser.parse_args("")

    negbio_labels = pd.read_csv(os.path.join(args.mimic_path, 'mimic-cxr-2.0.0-negbio.csv'))
    metadata = pd.read_csv(os.path.join(args.mimic_path, 'mimic-cxr-2.0.0-metadata.csv'))

    row = metadata[metadata['dicom_id']==args.dicom_id]

    labels = eval(args.labels)


    print("loading diffusion model and robust classifier...")
    ds = PNP_MIMIC('')
    classifier, _ = robustness.model_utils.make_and_restore_model(
        arch='resnet50',
        dataset=ds,
        resume_path=args.robust_classifier_path
    )
    # classifier.to('cuda')
    classifier.eval()

    att = DiffusionAttack(hps, args.diffusion_ckpt_path, classifier)


    original_image_path = os.path.join(args.mimic_path, 'files',
                            f"p{str(row['subject_id'].values[0])[:2]}",
                            f"p{str(row['subject_id'].values[0])}",
                            f"s{str(row['study_id'].values[0])}",
                            args.dicom_id + '.jpg')
    original_image = Image.open(original_image_path).convert("RGB")
    original_image = T.Compose([T.Resize((256,256)),T.ToTensor()])(original_image)


    original_pred = classifier(original_image.unsqueeze(0).to('cuda'))[0].detach().cpu()
    original_pred = torch.argmax(original_pred[0]).item()

    # print(f"Generating counterfactual, ie changing to target class {args.target_class, labels[args.target_class]}")
    counterfactual_arr = generate_counterfactual(
        att, original_image, target_class=args.target_class
    )

    counterfactual_tensor = torch.Tensor(counterfactual_arr)
    counterfactual_pred = classifier(counterfactual_tensor.unsqueeze(0).to('cuda'))[0].detach().cpu()
    counterfactual_pred = torch.argmax(counterfactual_pred[0]).item()

    print("\n\n========================== RESULTS ================================")
    print("\nGround-truth for original image: ")
    image_labels = negbio_labels[negbio_labels['study_id']==row['study_id'].values[0]].fillna(0.0)
    print(image_labels[labels.values()].to_string(index=False))
    print(f"\nTarget class (ie, desired class of generated counterfactual): {args.target_class, labels[args.target_class]}")
    print(f"\nClassifier's prediction for original image: {original_pred, labels[original_pred]}")
    print(f"\nClassifier's prediction for the counterfactual image: {counterfactual_pred, labels[counterfactual_pred]}")
    print("\n====================================================================")
    
    arr = (counterfactual_arr*255).astype(np.uint8)
    fig, ax = plt.subplots(1,2, figsize=(8, 4))
    ax[0].imshow(np.transpose(original_image.detach().cpu().numpy(), (1,2,0)))
    ax[0].set_title(f"Original")
    ax[1].imshow(np.transpose(arr, (1,2,0)))
    ax[1].set_title(f"Counterfactual")
    plt.show()
    
#%%
if __name__ == '__main__':
    main()