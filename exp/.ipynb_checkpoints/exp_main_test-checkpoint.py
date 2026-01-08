from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from utils.m4_summary import M4Summary
from utils.losses import mape_loss, mase_loss, smape_loss

from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, MTST, LRU
from layers import Koopa, KoopMamba, KoopMambaFFT, KoopMambaFFT_GLU, KoopBlock, KoopBlock_P, KoopBlock_noD
# KoopMambav2, KoopMambav3, KoopMamba_test, KoopMambav4, KoopMambav4_1, KoopMambav1_1, \
# KoopMambav6, KoopMambav7, KoopMambav2_3, KoopMambav4_4,KoopMambav4_1, KoopMambav4_5, KoopMambav4_6, KoopMambav8_4, KoopMambav4_7, \
# KoopMambav9, KoopMambav9_4, KoopMambav4_4_2layer,, FKoopMamba, KoopMambaFFT_GLUv2, KoopMamba_inv
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time
# import timm
# from timm.scheduler import CosineLRScheduler, PlateauLRScheduler

import warnings
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import pickle as pkl

from einops import rearrange

from timm.optim import create_optimizer_v2


import matplotlib.pyplot as plt
from scipy import linalg
import matplotlib


warnings.filterwarnings('ignore')
class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

        cfg = vars(args)
        self.use_mlflow = self.args.use_mlflow
        if self.use_mlflow:
            name = self.args.model_id
            project = self.args.mlflow_project
            experiment = mlflow.set_experiment(project)
            mlflow.start_run(run_name=name)
            mlflow.log_params(cfg)

    def _get_mask_spectrum(self):
        """
        get shared frequency spectrums
        """
        train_data, train_loader = self._get_data(flag='train')
        amps = 0.0
        for data in train_loader:
            lookback_window = data[0]
            # amps += abs(torch.fft.rfft(lookback_window, dim=1)).mean(dim=0)
            amps += abs(torch.fft.rfft(lookback_window, dim=1)).mean(dim=0).mean(dim=1)

        mask_spectrum = amps.topk(int(amps.shape[0]*self.args.alpha)).indices
        # mask_spectrum = amps.topk(int(amps.shape[0]*self.args.alpha), dim=0).indices #make it 2 Dim [L, C], different mask for each channel
        return mask_spectrum # as the spectrums of time-invariant component

    def _build_model(self):
        if self.args.data == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
            self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'MTST': MTST,
            'Koopa': Koopa,
            'KoopMamba': KoopMamba,
            'KoopMambaFFT': KoopMambaFFT,
            'GLU': KoopMambaFFT_GLU,
            'KoopBlock': KoopBlock,
            'KoopBlock_noD':KoopBlock_noD,
            'KoopBlock_P': KoopBlock_P,
            'LRU': LRU
        }
        # 'KoopMambaCI': KoopMambav2,
        # 'KoopRec': KoopMambav3,
        # 'KoopMambaFFN': KoopMambav4_1,
        # 'FFN': KoopMambav1_1,
        # 'MultiKoop': KoopMambav6,
        # 'DKoopMamba': KoopMambav8_4,
        # 'ChannelMix': KoopMambav4_6,
        # '2layer': KoopMambav4_4_2layer,
        # 'InvRNN': KoopMambav9,
        # 'GLU_freq': FKoopMamba,
        # 'GLUv2': KoopMambaFFT_GLUv2,
        # 'KoopMamba_inv': KoopMamba_inv,
        self.args.mask_spectrum = self._get_mask_spectrum()
        model = model_dict[self.args.model].Model(self.args).float().cuda()

        self.inv_loss_alpha = self.args.inv_loss_alpha


        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.l2)
        # model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.l2)
        model_optim = create_optimizer_v2(self.model.parameters(), opt='adamw', lr=self.args.learning_rate, weight_decay=self.args.l2)
        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'LRU' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        inv_in_out = None
                        if isinstance(outputs, (tuple, list)):
                            outputs, inv_in_out = outputs
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'LRU' in self.args.model:
                        outputs = self.model(batch_x)
                        if isinstance(outputs, (tuple, list)):
                            outputs, draw_list, attn_list = outputs
                    elif 'Rec' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        rec, outputs = outputs
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        inv_in_out = None
                        if isinstance(outputs, (tuple, list)):
                            outputs, inv_in_out = outputs


                f_dim = -1 if self.args.features == 'MS' else 0

                if 'TST' in self.args.model:
                    outputs = outputs[:, :, f_dim:]
                else:
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                batch_y_mark = torch.ones(true.shape)

                loss = criterion(pred, true)

                # if 'Rec' in self.args.model:
                #     rec, pred = outputs
                #     loss += criterion(rec, input)




                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # # to continue mtst on traffic
        # print('loading model')
        # model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(model_path))

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()


        if 'timm' in self.args.lradj:
            if 'cos' in self.args.lradj:
                scheduler = CosineLRScheduler(optimizer = model_optim,
                                              t_initial=train_steps - self.args.warmup_steps,
                                              lr_min=1e-8,
                                              warmup_t=self.args.warmup_steps,
                                              warmup_prefix=True,
                                              warmup_lr_init=1e-8
                                              )
            elif 'plateau' in self.args.lradj:
                scheduler = PlateauLRScheduler(optimizer=model_optim,
                                               decay_rate=self.args.decay_rate,
                                               patience_t=self.args.lr_patience,
                                               warmup_t=self.args.warmup_steps,
                                               warmup_lr_init=1e-8,
                                               lr_min=self.args.lr_min,
                                               mode='min'
                                        )

        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                                steps_per_epoch = train_steps,
                                                pct_start = self.args.pct_start,
                                                epochs = self.args.train_epochs,
                                                max_lr = self.args.learning_rate)



        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'LRU' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        inv_in_out = None
                        if isinstance(outputs, (tuple, list)):
                            outputs, inv_in_out = outputs


                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        # loss = criterion(forecast=outputs, target=batch_y, mask = batch_y_mark)
                        loss = criterion(outputs,batch_y)

                        if inv_in_out is not None:
                            inv_in, inv_out = inv_in_out[0], inv_in_out[1]
                            inv_loss = criterion(inv_in, inv_out)
                            loss += inv_loss * self.inv_loss_alpha


                        train_loss.append(loss.item())

                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'LRU' in self.args.model:
                            outputs = self.model(batch_x)
                            if isinstance(outputs, (tuple, list)):
                                outputs, draw_list, attn_list = outputs
                    elif 'Rec' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        rec, outputs = outputs
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    inv_in_out = None
                    if isinstance(outputs, (tuple, list)):
                        outputs, inv_in_out = outputs

                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:]


                    loss = criterion(outputs, batch_y)
                    if 'Rec' in self.args.model:
                        batch_x = batch_x.to(self.device)
                        loss += 0.5*criterion(rec, batch_x)

                    if inv_in_out is not None:
                        inv_in, inv_out = inv_in_out[0], inv_in_out[1]
                        inv_loss = criterion(inv_in, inv_out)
                        loss += inv_loss * self.inv_loss_alpha

                    train_loss.append(loss.item())



                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    print("Model Num. Parameters:", sum([p.numel() for p in self.model.parameters()]))
                    print("Mem. Info:", torch.cuda.max_memory_allocated())
                    exit()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            # print("Model Num. Parameters:", sum([p.numel() for p in self.model.parameters()]))
            # print("Mem. Info:", torch.cuda.max_memory_allocated())
            # exit()

            if self.use_mlflow:
                log_dict = {'train/loss': train_loss, 'vali/loss': vali_loss, 'test/loss': test_loss}
                mlflow.log_metrics(log_dict, step=epoch)
                lr = model_optim.param_groups[0]['lr']
                mlflow.log_metric('lr', lr, step=epoch)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if ('timm' in self.args.lradj and 'plateau' in self.args.lradj \
                    and model_optim.param_groups[0]['lr'] > self.args.lr_min) \
                    or epoch + 1 < self.args.warmup_steps:
                early_stopping.counter=0
                print("Reset Early stopping before Plateau Scheduler reach the lr_min")

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, metric=vali_loss)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):

        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            print(f"Number of GPUs available: {torch.cuda.device_count()}")
            print(f"Selected device: {self.device}")  # Check what is being passed
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        # visual always true in test function:
        self.args.visual = True

        # if self.args.visual:
        #     self.model.model.visual = True

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'LRU' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        inv_in_out = None
                        if isinstance(outputs, (tuple, list)):
                            outputs, inv_in_out = outputs
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'LRU' in self.args.model:
                            outputs = self.model(batch_x)
                            draw_list = None
                            attn_list = None
                            if isinstance(outputs, (tuple, list)):
                                outputs, draw_list, attn_list = outputs
                    elif 'Rec' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        rec, outputs = outputs
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        # inv_in_out = None
                        # if isinstance(outputs, (tuple, list)):
                        #     outputs, inv_in_out = outputs
                        branch_pred = None
                        if isinstance(outputs, (tuple, tuple)):
                            outputs, branch_pred = outputs

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze() # full_forecast
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                
                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                
                ########### isolate Koopman mode ######################

                # visualize_modal_contributions(self.model, batch_x, num_modes=1)

                # Analyze modes 0, 1, and 5
                analyze_koopman_cumulative(
                    self.model, 
                    batch_x, 
                    batch_y, 
                    save_path='eigenvalue/isolate',
                    dataset_name=self.args.data_path.split('.')[0]

                )
                # break
                # mode_idx=[0],  # Analyze multiple modes



                ### draw figures ########

                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if self.args.features == "S":
                #         variate_ids = [0]
                #     else:
                #         # variate_ids = [11,12,13,14]
                #         variate_ids = [0,1,2,6]
                    # for j in variate_ids:
                    #     gt = np.concatenate((input[0, :, j], true[0, :, j]), axis=0)
                    #     pd = pred[0, :, j]
                    #     # pd = np.concatenate((input[0, :, j], pred[0, :, j]), axis=0)
                    #     # time_step = np.arange(input.shape[1], gt.shape[1])
                    #     visual(gt, pd, os.path.join(folder_path, str(i) + f'_var{j}.pdf'))






        ########### plot eigen value and distribution #########
        # # Set global font sizes for better readability
        # matplotlib.rcParams['font.size'] = 14
        # matplotlib.rcParams['axes.labelsize'] = 16
        # matplotlib.rcParams['axes.titlesize'] = 18
        # matplotlib.rcParams['xtick.labelsize'] = 14
        # matplotlib.rcParams['ytick.labelsize'] = 14
        # matplotlib.rcParams['legend.fontsize'] = 14

        # # def analyze_koopman_eigenvalues(model, dataset_name="ETTh1"):
        # """
        # Analyze and visualize the eigenvalues of the Koopman operators in SKOLR.
        
        # Args:
        #     model: Trained SKOLR model
        #     dataset_name: Name of the dataset for the plot title
        # """
        # # Number of branches in the model
        # num_branches = len(self.model.blocks)
        
        # # Create a figure with subplots
        # fig, axes = plt.subplots(1, num_branches + 1, figsize=(6*num_branches, 5))
        
        # # Plot eigenvalues for each branch
        # all_eigenvalues = []
        # # colors = plt.cm.tab10(np.linspace(0, 1, num_branches))
        # colors = ['#1f77b4', '#ff7f0e']  # Blue, Orange
        
        # for i in range(num_branches):
        #     # Extract the matrix M_i from the i-th branch
        #     M_i = self.model.blocks[i].block.linearRNN.Whh.weight.detach().cpu().numpy()
            
        #     # Compute eigenvalues
        #     eigenvalues = linalg.eigvals(M_i)
        #     all_eigenvalues.extend(eigenvalues)
            
        #     # Plot on complex plane
        #     ax = axes[i]
        #     ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), s=80, alpha=0.7, 
        #               color=colors[i], label=f'Branch {i+1}')
            
        #     # Add unit circle for reference
        #     theta = np.linspace(0, 2*np.pi, 100)
        #     ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, linewidth=2)
            
        #     # Set axis limits and labels
        #     ax.set_xlim(-1.1, 1.1)
        #     ax.set_ylim(-1.1, 1.1)
        #     ax.set_aspect('equal')
        #     ax.set_title(f'Branch {i+1} Eigenvalues', fontsize=20, pad=15)
        #     ax.set_xlabel('Real Part', fontsize=18)
        #     ax.set_ylabel('Imaginary Part', fontsize=18)
        #     ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        #     ax.tick_params(width=2, length=6)
        #     for spine in ax.spines.values():
        #         spine.set_linewidth(1.5)
        #     ax.legend(frameon=True, fontsize=16)
            
        # # Plot all eigenvalues combined in the last subplot
        # ax = axes[-1]
        # for i in range(num_branches):
        #     branch_eigenvalues = linalg.eigvals(self.model.blocks[i].block.linearRNN.Whh.weight.detach().cpu().numpy())
        #     ax.scatter(np.real(branch_eigenvalues), np.imag(branch_eigenvalues), s=80, alpha=0.7, 
        #               color=colors[i], label=f'Branch {i+1}')
        
        # # Add unit circle for reference
        # theta = np.linspace(0, 2*np.pi, 100)
        # ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, linewidth=2)
        
        # # Set axis limits and labels
        # ax.set_xlim(-1.1, 1.1)
        # ax.set_ylim(-1.1, 1.1)
        # ax.set_aspect('equal')
        # ax.set_title(f'Combined Eigenvalues', fontsize=20, pad=15)
        # ax.set_xlabel('Real Part', fontsize=18)
        # ax.set_ylabel('Imaginary Part', fontsize=18)
        # ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        # ax.tick_params(width=2, length=6)
        # for spine in ax.spines.values():
        #     spine.set_linewidth(1.5)
        # ax.legend(frameon=True, fontsize=16)

        # dataset_name=self.args.data_path.split('.')[0]
        # plt.tight_layout()
        # plt.subplots_adjust(top=0.85)
        # plt.suptitle(f'Koopman Operator Eigenvalues: {dataset_name}', 
        #             fontsize=22, y=0.98)
        # plt.savefig(f'koopman_eigenvalues_{dataset_name}.pdf', 
        #            bbox_inches='tight', dpi=300)
        # plt.show()
        
        # # Additional analysis: eigenvalue magnitudes histogram
        # plt.figure(figsize=(12, 6))
        # for i in range(num_branches):
        #     branch_eigenvalues = linalg.eigvals(self.model.blocks[i].block.linearRNN.Whh.weight.detach().cpu().numpy())
        #     magnitudes = np.abs(branch_eigenvalues)
        #     plt.hist(magnitudes, bins=30, alpha=0.7, label=f'Branch {i+1}')
        
        # plt.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Unit Magnitude')
        # plt.title(f'Distribution of Eigenvalue Magnitudes: {dataset_name}', fontsize=20)
        # plt.xlabel('Magnitude (|λ|)', fontsize=18)
        # plt.ylabel('Frequency', fontsize=18)
        # plt.grid(True, alpha=0.3, linestyle='--')
        # plt.legend(fontsize=16)
        # plt.tight_layout()
        # plt.savefig(f'eigenvalue_magnitudes_{dataset_name}.pdf', 
        #            bbox_inches='tight', dpi=300)
        # plt.show()
        
        # return all_eigenvalues
    
    # Example usage:
    # analyze_koopman_eigenvalues(skolr_model, "ETTh1")



        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
            
        # print("Model Num. Parameters:", sum([p.numel() for p in self.model.parameters()]))
        # print("Mem. Info:", torch.cuda.max_memory_allocated())
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        preds = rearrange(preds, 'b l d -> b d l')
        trues = rearrange(trues, 'b l d -> b d l')
        inputx = rearrange(inputx, 'b l d -> b d l')


        # result save

        if self.args.data == 'm4':
            # result save
            folder_path = './m4_results/' + self.args.model + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
            forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
            forecasts_df.index.name = 'id'
            forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
            forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv')

            print(self.args.model)
            file_path = './m4_results/' + self.args.model + '/'
            if 'Weekly_forecast.csv' in os.listdir(file_path) \
                    and 'Monthly_forecast.csv' in os.listdir(file_path) \
                    and 'Yearly_forecast.csv' in os.listdir(file_path) \
                    and 'Daily_forecast.csv' in os.listdir(file_path) \
                    and 'Hourly_forecast.csv' in os.listdir(file_path) \
                    and 'Quarterly_forecast.csv' in os.listdir(file_path):
                m4_summary = M4Summary(file_path, self.args.root_path)
                # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
                smape_results, owa_results, mape, mase = m4_summary.evaluate()
                print('smape:', smape_results)
                print('mape:', mape)
                print('mase:', mase)
                print('owa:', owa_results)
            else:
                print('After all 6 tasks are finished, you can calculate the averaged index')

        else:
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            print(preds.shape)
            
            mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
            print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
            f = open("extendTest_result.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
            f.write('\n')
            f.write('\n')
            f.close()

            if self.use_mlflow:
                log_dict= {'best/test_mae': mae, 'best/test_mse': mse, "best/test_rse": rse}
                mlflow.log_metrics(log_dict)
                mlflow.end_run()


            # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
            # np.save(folder_path + 'pred.npy', preds)
            # np.save(folder_path + 'true.npy', trues)
            # np.save(folder_path + 'x.npy', inputx)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'LRU' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'LRU' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        # self.model.model.visual = True

        return

import math
import torch
import numpy as np
import matplotlib.pyplot as plt


def analyze_koopman_cumulative(model, x_enc, y_true, save_path='eigenvalue/isolate', dataset_name="ETTh1"):
    """
    Analyze how combinations of dominant Koopman modes approximate the full prediction.
    
    Args:
        model: Trained model
        x_enc: Input tensor
        y_true: Ground truth tensor
        save_path: Directory to save plots
        dataset_name: Name of the dataset for the plot title
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import linalg
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    

    # Mode combinations to analyze
    # mode_combinations = [np.array(range(20)),[0]]
    mode_combinations = [[0]]

    mode_labels = ["Top 1 mode"]
    
    
    # Number of branches in the model
    num_branches = len(model.blocks)
    
    # Generate full prediction
    with torch.no_grad():
        full_pred = model(x_enc, None, None, None)
        if isinstance(full_pred, tuple):
            full_pred = full_pred[0]
    
    # For each branch, analyze combinations of modes
    for branch_idx in range(num_branches):
        # Extract the Koopman matrix and compute eigenvalues
        K_i = model.blocks[branch_idx].block.linearRNN.Whh.weight.detach().cpu().numpy()
        eigenvalues = linalg.eigvals(K_i)
        
        # Sort by magnitude
        sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
        
        # Generate predictions for each combination
        predictions = []
        for combo in mode_combinations:
            # Get the indices of the selected modes (after sorting)
            if combo is not None:
                selected_indices = [sorted_indices[i] for i in combo if i < len(sorted_indices)]
            else:
                selected_indices = None
            # Generate prediction with these modes
            with torch.no_grad():
                combo_pred = isolate_koopman_mode(model, x_enc, 
                                               mode_idx=selected_indices, 
                                               conjugate_pair=True, 
                                               block_idx=branch_idx)
                if isinstance(combo_pred, tuple):
                    combo_pred = combo_pred[0]
                predictions.append(combo_pred) # append mode prediction for this branch
        
        # Create a figure showing how combinations approximate the full prediction
        B, L, C = x_enc.shape
        
        # Time steps for x-axis
        time_steps = np.arange(L + full_pred.shape[1])
        
        # Plot for each variable (up to 3)
        fig, axes = plt.subplots(min(C, 1), 1, figsize=(10, 3*min(C, 1)))
        if min(C, 1) == 1:
            axes = [axes]
        
        # Define colors for different combinations
        colors = ['#00CC44','#FF2D00'] # '#9900FF', '#ff7f0e','#1f77b4'] # green, red, purple, , orange, blue

        for var_idx in range(0, 1):
            # ax = axes[var_idx]
            ax = axes[0]

            # Plot history
            idx=0
            history = x_enc[idx, :, var_idx].cpu().numpy()
            ax.plot(time_steps[:L], history, 'k-', linewidth=1.5, label='History')
            
            # Last history point for continuity
            # last_history = history[-1]
            
            # Plot ground truth
            if y_true is not None:
                gt_data = y_true[idx, :, var_idx]
                # full_gt = np.concatenate([[last_history], gt_data])
                ax.plot(time_steps[L:L+len(gt_data)], gt_data, '#9900FF','-', linewidth=1.5, label='Ground Truth')
            
            # Plot full prediction
            full_data = full_pred[idx, :, var_idx].cpu().numpy()
            # full_combined = np.concatenate([[last_history], full_data])
            
            # ax.plot(time_steps[L:L+len(full_data)], full_data, 
            #         '#9900FF','-', linewidth=1.5, label='Full Prediction')
            
            ################ Plot each mode combination
            for i, (pred, combo, label) in enumerate(zip(predictions, mode_combinations, mode_labels)):
                pred_data = pred[idx, :, var_idx].cpu().numpy()
                # pred_full = np.concatenate([[last_history], pred_data])
                ax.plot(time_steps[L:L+len(pred_data)], pred_data, 
                        '--', color=colors[i], linewidth=1.5, label=label)
            
            # Set title and formatting
            # ax.set_title(f'Variable {var_idx+1}', fontsize=16)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=12)
            
            # Format axes
            ax.tick_params(width=2, length=6)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
        
        # Get current handles and labels
        handles, labels = ax.get_legend_handles_labels()
        
        # Create a dictionary to store unique labels and their first occurring handle
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels[label] = handle
        
        # Create new legend with only unique entries
        ax.legend(
            [unique_labels[label] for label in unique_labels], 
            list(unique_labels.keys()), 
            fontsize=12, loc='upper center',ncol=1, frameon=True
        )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        # plt.suptitle(f'Branch {branch_idx+1} - Selected Koopman Mode Contibution', 
        #             fontsize=18, y=0.98)
        plt.suptitle('Selected Koopman Mode Contribution', 
                    fontsize=18, y=0.98)
        
        # Save time series plot
        ts_path = os.path.join(save_path, f'branch{branch_idx+1}_cumulative_modes_{dataset_name}.pdf')
        plt.savefig(ts_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        #     ############# Plot combined branches prediction
        #     # Plot individual branch predictions
        #     branch_combo_data = []
        #     for i, (pred, combo, label) in enumerate(zip(predictions, mode_combinations, mode_labels)):
        #          pred_data = pred[0, :, var_idx].cpu().numpy()
        #          branch_combo_data.append(pred_data)
        #     # Sum the predictions from both branches (minus the duplicated last history point)
        #     combined_data = sum(branch_combo_data) / len(branch_combo_data)  # Average instead of sum
        #     # combined_full = np.concatenate([[last_history], combined_data])
        #     ax.plot(time_steps[L:L+len(combined_data)], combined_data, 
        #             '-', color=colors[-1], linewidth=2, 
        #             label=f'{label}')
            
        #     # Set title and formatting
        #     # ax.set_title(f'Variable {var_idx+1}', fontsize=16)
        #     ax.grid(True, alpha=0.3, linestyle='--')
        #     ax.legend(fontsize=10)
            
        #     # Format axes
        #     ax.tick_params(width=2, length=6)
        #     for spine in ax.spines.values():
        #         spine.set_linewidth(1.5)
        
        # plt.tight_layout()
        # plt.subplots_adjust(top=0.9)
        # plt.suptitle(f'Cumulative Mode Approximation', 
        #             fontsize=18, y=0.98)
        
        # # Save time series plot
        # ts_path = os.path.join(save_path, f'combined_branches_{dataset_name}.pdf')
        # plt.savefig(ts_path, bbox_inches='tight', dpi=300)



        
        ######## Also create an eigenvalue plot highlighting the combinations
        plt.figure(figsize=(3, 3))
        
        # Plot all eigenvalues with reduced opacity
        plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), s=80, alpha=0.3, 
                   color='gray')
        
        # Plot the modes in each combination with different colors
        for i, combo in enumerate(mode_combinations):
            # Get the actual indices
            if combo is not None:
                indices = [sorted_indices[j] for j in combo if j < len(sorted_indices)]
            else:
                indices = np.array(range(256))
            
            # Plot these eigenvalues
            for idx in indices:
                plt.scatter(np.real(eigenvalues[idx]), np.imag(eigenvalues[idx]), 
                           s=100, color=colors[i], edgecolor='black', alpha=0.9)
                
                # If it's a complex eigenvalue, also highlight its conjugate
                if np.abs(np.imag(eigenvalues[idx])) > 1e-10:
                    conj_idx = np.argmin(np.abs(eigenvalues - np.conj(eigenvalues[idx])))
                    if conj_idx != idx:
                        plt.scatter(np.real(eigenvalues[conj_idx]), np.imag(eigenvalues[conj_idx]), 
                                   s=100, color=colors[i], edgecolor='black', alpha=0.9)
        
        # Add custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], 
                  markersize=10, label=mode_labels[0]),
            # Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], 
            #       markersize=10, label=mode_labels[1]),
            # Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[2], 
            #       markersize=10, label=mode_labels[2])
        ]
        # plt.legend(handles=legend_elements, fontsize=12)
        plt.legend(handles=legend_elements, fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.15),ncol=2, frameon=True)

        
        # Add unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, linewidth=2)
        
        # Set axis limits and labels
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.axis('equal')
        plt.xlabel('Real Part', fontsize=14)
        plt.ylabel('Imaginary Part', fontsize=14)
        plt.title(f'Eigenvalues', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        # Save eigenvalue plot
        eigenvalue_path = os.path.join(save_path, f'branch{branch_idx+1}_eigen_highlight_{dataset_name}.pdf')
        plt.savefig(eigenvalue_path, bbox_inches='tight', dpi=300)
    
    print(f"Cumulative mode analysis complete. Plots saved to {save_path}")

# def analyze_koopman_mode(model, x_enc, y_true, mode_idx=[0], save_path='eigenvalue/isolate', dataset_name='ETTh1'):
#     """
#     Analyze and visualize specific Koopman modes across all branches.
    
#     Args:
#         model: Trained model
#         x_enc: Input tensor
#         y_true: Ground truth tensor
#         mode_idx: List of mode indices to analyze (after sorting by magnitude)
#         save_path: Directory to save plots
#         dataset_name: Name of the dataset for the plot title
#     """
#     import os
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from scipy import linalg
    
#     # Convert mode_idx to list if it's a single integer
#     if isinstance(mode_idx, int):
#         mode_idx = [mode_idx]
    
#     # Create save directory
#     os.makedirs(save_path, exist_ok=True)
    
#     # Number of branches in the model
#     num_branches = len(model.blocks)
    
#     # First, generate the eigenvalue visualization plot
#     fig, axes = plt.subplots(1, num_branches + 1, figsize=(6*num_branches, 5))
#     if num_branches == 1:
#         axes = [axes, axes]  # Handle case with only one branch
    
#     # Colors for different branches
#     branch_colors = ['#1f77b4', '#ff7f0e']  # Blue, Orange
    
#     # Colors for different modes
#     mode_colors = ['red', 'blue', 'green', 'purple', 'orange']
    
#     # Track the selected modes in each branch
#     all_selected_modes = []
    
#     # Plot eigenvalues for each branch
#     for i in range(num_branches):
#         # Extract the Koopman matrix from the branch
#         K_i = model.blocks[i].block.linearRNN.Whh.weight.detach().cpu().numpy()
        
#         # Compute eigenvalues
#         eigenvalues = linalg.eigvals(K_i)
        
#         # Sort by magnitude
#         sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
        
#         # Find each requested mode
#         branch_selected_modes = []
#         for m_idx, m in enumerate(mode_idx):
#             if m < len(sorted_indices):
#                 selected_mode_idx = sorted_indices[m]
#                 branch_selected_modes.append((selected_mode_idx, eigenvalues[selected_mode_idx], m_idx))
        
#         all_selected_modes.append((i, branch_selected_modes))
        
#         # Plot on complex plane
#         ax = axes[i]
        
#         # Plot all eigenvalues with reduced opacity
#         ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), s=80, alpha=0.3, 
#                   color=branch_colors[i], label=f'Branch {i+1}')
        
#         # Highlight the selected modes
#         for selected_idx, selected_val, m_position in branch_selected_modes:
#             mode_color = mode_colors[m_position % len(mode_colors)]
#             ax.scatter(np.real(selected_val), np.imag(selected_val), 
#                       s=150, color=mode_color, edgecolor='black', zorder=10)
            
#             # Add label
#             ax.text(np.real(selected_val), np.imag(selected_val), 
#                    f'M{mode_idx[m_position]}', fontsize=12, fontweight='bold')
            
#             # Find and highlight conjugate pair
#             closest_conj_idx = np.argmin(np.abs(eigenvalues - np.conj(selected_val)))
#             if closest_conj_idx != selected_idx:
#                 ax.scatter(np.real(eigenvalues[closest_conj_idx]), 
#                           np.imag(eigenvalues[closest_conj_idx]), 
#                           s=150, color=mode_color, alpha=0.5, edgecolor='black', zorder=10)
        
#         # Add unit circle for reference
#         theta = np.linspace(0, 2*np.pi, 100)
#         ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, linewidth=2)
        
#         # Set axis limits and labels
#         ax.set_xlim(-1.1, 1.1)
#         ax.set_ylim(-1.1, 1.1)
#         ax.set_aspect('equal')
#         ax.set_title(f'Branch {i+1} Eigenvalues', fontsize=20, pad=15)
#         ax.set_xlabel('Real Part', fontsize=18)
#         ax.set_ylabel('Imaginary Part', fontsize=18)
#         ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
#         ax.tick_params(width=2, length=6)
#         for spine in ax.spines.values():
#             spine.set_linewidth(1.5)
#         ax.legend(frameon=True, fontsize=16)
    
#     # Plot all branches combined in the last subplot
#     ax = axes[-1]
    
#     # Legend entries for the combined plot
#     custom_lines = []
#     custom_labels = []
    
#     for i, (branch_idx, branch_modes) in enumerate(all_selected_modes):
#         # Get eigenvalues for this branch
#         K_i = model.blocks[branch_idx].block.linearRNN.Whh.weight.detach().cpu().numpy()
#         eigenvalues = linalg.eigvals(K_i)
        
#         # Plot all eigenvalues with reduced opacity
#         ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), s=80, alpha=0.3, 
#                   color=branch_colors[i], label=f'Branch {branch_idx+1}')
        
#         # Add to legend
#         custom_lines.append(plt.Line2D([0], [0], marker='o', color='w', 
#                                       markerfacecolor=branch_colors[i], markersize=10))
#         custom_labels.append(f'Branch {branch_idx+1}')
        
#         # Highlight the selected modes
#         for selected_idx, selected_val, m_position in branch_modes:
#             mode_color = mode_colors[m_position % len(mode_colors)]
#             ax.scatter(np.real(selected_val), np.imag(selected_val), 
#                       s=150, color=mode_color, edgecolor='black', zorder=10)
            
#             # Add label for the mode
#             ax.text(np.real(selected_val), np.imag(selected_val), 
#                    f'B{branch_idx+1}M{mode_idx[m_position]}', fontsize=10)
            
#             # Add to legend for the first branch only (to avoid duplicates)
#             if i == 0:
#                 custom_lines.append(plt.Line2D([0], [0], marker='o', color='w',
#                                               markerfacecolor=mode_color, markersize=10))
#                 custom_labels.append(f'Mode {mode_idx[m_position]}')
    
#     # Add unit circle for reference
#     theta = np.linspace(0, 2*np.pi, 100)
#     ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, linewidth=2)
    
#     # Set axis limits and labels
#     ax.set_xlim(-1.1, 1.1)
#     ax.set_ylim(-1.1, 1.1)
#     ax.set_aspect('equal')
#     ax.set_title(f'Combined Eigenvalues', fontsize=20, pad=15)
#     ax.set_xlabel('Real Part', fontsize=18)
#     ax.set_ylabel('Imaginary Part', fontsize=18)
#     ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
#     ax.tick_params(width=2, length=6)
#     for spine in ax.spines.values():
#         spine.set_linewidth(1.5)
    
#     # Custom legend
#     ax.legend(custom_lines, custom_labels, frameon=True, fontsize=16)
    
#     # Adjust figure title
#     mode_str = ", ".join([str(m) for m in mode_idx])
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.85)
#     plt.suptitle(f'Modes [{mode_str}] Analysis: {dataset_name}', 
#                 fontsize=22, y=0.98)
    
#     # Save eigenvalue plot
#     eigenvalue_path = os.path.join(save_path, f'modes_{mode_str}_eigenvalues_{dataset_name}.pdf')
#     plt.savefig(eigenvalue_path, bbox_inches='tight', dpi=300)
    
#     # Generate full prediction
#     with torch.no_grad():
#         full_pred = model(x_enc, None, None, None)
#         if isinstance(full_pred, tuple):
#             full_pred = full_pred[0]
    
#     # Now generate separate time series plots for each branch and each mode
#     B, L, C = x_enc.shape

#     # For each branch and mode combination
#     for branch_idx, branch_modes in all_selected_modes:
#         for selected_idx, selected_val, m_position in branch_modes:
#             requested_mode = mode_idx[m_position]
            
#             # Generate predictions
#             with torch.no_grad():
#                 # Only the selected mode
#                 branch1_pred = isolate_koopman_mode(model, x_enc, 
#                                                   mode_idx=[selected_idx], 
#                                                   conjugate_pair=False, 
#                                                   block_idx=branch_idx)
#                 if isinstance(branch1_pred, tuple):
#                     branch1_pred = branch1_pred[0]
                
#                 # Selected mode + conjugate
#                 branch2_pred = isolate_koopman_mode(model, x_enc, 
#                                                   mode_idx=[selected_idx], 
#                                                   conjugate_pair=True, 
#                                                   block_idx=branch_idx)
#                 if isinstance(branch2_pred, tuple):
#                     branch2_pred = branch2_pred[0]
            
#             # Create time series plot
#             print("Creating ts plot")
#             fig, axes = plt.subplots(min(C, 3), 1, figsize=(12, 3*min(C, 3)))
#             if min(C, 3) == 1:
#                 axes = [axes]
            
#             # Time steps for x-axis
#             time_steps = np.arange(L + full_pred.shape[1])
            
#             # Plot each variable
#             for var_idx in range(min(C, 3)):
#                 ax = axes[var_idx]
                
#                 # Plot history
#                 history = x_enc[0, :, var_idx].cpu().numpy()
#                 ax.plot(time_steps[:L], history, 'k-', linewidth=2, label='History')
                
#                 # Last history point for continuity
#                 last_history = history[-1]
                
#                 # Plot ground truth
#                 if y_true is not None:
#                     gt_data = y_true[0, :, var_idx]
#                     full_gt = np.concatenate([[last_history], gt_data])
#                     ax.plot(time_steps[L-1:L+len(gt_data)], full_gt, 'g-', 
#                             linewidth=2, label='Ground Truth')
                
#                 # Get eigenvalue properties
#                 mag = np.abs(selected_val)
#                 phase = np.angle(selected_val)
#                 freq = phase / (2 * np.pi)
                
#                 # Plot Branch 1 (mode only)
#                 branch1_data = branch1_pred[0, :, var_idx].cpu().numpy()
#                 branch1_full = np.concatenate([[last_history], branch1_data])
#                 ax.plot(time_steps[L-1:L+len(branch1_data)], branch1_full, 
#                         '--', color=branch_colors[branch_idx % len(branch_colors)], linewidth=1.5, 
#                         label=f'Mode Only (|λ|={mag:.3f}, f={freq:.3f})')
                
#                 # Plot Branch 2 (mode + conjugate)
#                 branch2_data = branch2_pred[0, :, var_idx].cpu().numpy()
#                 branch2_full = np.concatenate([[last_history], branch2_data])
#                 ax.plot(time_steps[L-1:L+len(branch2_data)], branch2_full, 
#                         '-', color=branch_colors[branch_idx % len(branch_colors)], linewidth=1.5, 
#                         label=f'Mode+Conj. (|λ|={mag:.3f})')
                
#                 # Plot full prediction
#                 full_data = full_pred[0, :, var_idx].cpu().numpy()
#                 full_combined = np.concatenate([[last_history], full_data])
#                 ax.plot(time_steps[L-1:L+len(full_data)], full_combined, 
#                         'c-', linewidth=1.5, label='Full Model')
                
#                 ax.set_title(f'Variable {var_idx+1}', fontsize=16)
#                 ax.grid(True, alpha=0.3, linestyle='--')
#                 ax.legend(fontsize=12)
                
#                 # Format axes
#                 ax.tick_params(width=2, length=6)
#                 for spine in ax.spines.values():
#                     spine.set_linewidth(1.5)
            
#             plt.tight_layout()
#             plt.subplots_adjust(top=0.9)
#             plt.suptitle(f'Branch {branch_idx+1} - Mode {requested_mode} Contribution', 
#                         fontsize=18, y=0.98)
            
#             # Save time series plot
#             ts_path = os.path.join(save_path, f'branch{branch_idx+1}_mode{requested_mode}_{dataset_name}.pdf')
#             plt.savefig(ts_path, bbox_inches='tight', dpi=300)
    
#     print(f"Analysis complete. Plots saved to {save_path}")










def isolate_koopman_mode(model, x_enc, mode_idx=None, conjugate_pair=True, block_idx=None):
    """
    Forward pass with isolation of specific Koopman modes for analysis.
    
    Parameters:
    -----------
    model : Model
        The trained Koopman model
    x_enc : torch.Tensor
        Input tensor of shape [batch_size, seq_len, n_features]
    mode_idx : int or list, optional
        Index/indices of mode(s) to isolate within each block's LinearRNN
        If None, all modes are used
    conjugate_pair : bool
        Whether to include conjugate pairs for complex eigenvalues
    block_idx : int or list, optional
        Index/indices of block(s) to include in the forecast
        If None, all blocks are used but with mode isolation
        
    Returns:
    --------
    torch.Tensor
        Forecast using only the selected mode(s)
    """
    # Apply the same preprocessing as in the forward method
    mean_enc = x_enc.mean(1, keepdim=True).detach()
    x_enc_norm = x_enc - mean_enc
    std_enc = torch.sqrt(torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    x_enc_norm = x_enc_norm / std_enc
    
    # Get the block decomposition using Fourier filter
    ifft_results = model.disentanglement(x_enc_norm)
    
    # Define which blocks to use
    if block_idx is None:
        # Use all blocks but with mode isolation
        active_blocks = range(model.num_blocks)
    elif isinstance(block_idx, int):
        active_blocks = [block_idx]
    else:
        active_blocks = block_idx
    
    # Process each active block with mode isolation
    x_pred_list = []
    
    for i in active_blocks:
        # Get the block
        block = model.blocks[i]
        
        # Process the input through the block's encoder
        B, L, C = ifft_results[i].shape
        res = ifft_results[i]
        
        # Apply the same processing as in the TimeVarKP forward method
        freq = math.ceil(model.input_len / model.seg_len)
        padding_len = model.seg_len * freq - model.input_len
        res = torch.cat((res[:, L-padding_len:, :], res), dim=1)
        res = res.chunk(freq, dim=1)
        
        if block.CI:
            res = rearrange(torch.stack(res, dim=1), 'b f p c -> (b c) f p')
        else:
            res = rearrange(torch.stack(res, dim=1), 'b f p c -> b f (p c)')
        
        # Encode
        x_enc_latent = block.encoder(res)
        
        # Get the LinearRNN for modal analysis
        linearRNN = block.block.linearRNN
        
        # Get the Koopman operator matrix (K)
        K = linearRNN.Whh.weight.data
        
        # Perform eigendecomposition of K
        eigenvalues, eigenvectors = torch.linalg.eig(K)
        
        # Compute the inverse of eigenvectors
        eigenvectors_inv = torch.linalg.inv(eigenvectors)
        
        # Extract real components for computation
        eigenvectors_real = eigenvectors.real
        eigenvectors_inv_real = eigenvectors_inv.real
        
        # Initialize a mask for which modes to use
        mode_mask = torch.zeros(eigenvalues.shape[0], dtype=torch.bool, device=K.device)
        
        if mode_idx is not None:
            # Set selected mode to True
            if isinstance(mode_idx, int):
                mode_mask[mode_idx] = True
                
                # If complex and conjugate_pair is True, also include conjugate
                if conjugate_pair and torch.abs(eigenvalues[mode_idx].imag) > 1e-10:
                    # Find the index of the conjugate pair
                    conjugate_diff = torch.abs(eigenvalues - eigenvalues[mode_idx].conj())
                    conj_idx = torch.argmin(conjugate_diff)
                    if conj_idx != mode_idx:  # Ensure we're not picking the same index
                        mode_mask[conj_idx] = True
            else:
                # Handle list of mode indices
                for idx in mode_idx:
                    mode_mask[idx] = True
                    if conjugate_pair and torch.abs(eigenvalues[idx].imag) > 1e-10:
                        conjugate_diff = torch.abs(eigenvalues - eigenvalues[idx].conj())
                        conj_idx = torch.argmin(conjugate_diff)
                        if conj_idx != idx:
                            mode_mask[conj_idx] = True
        else:
            # Use all modes
            mode_mask = torch.ones_like(mode_mask)

        # Process each time step with mode isolation
        B, F, H = x_enc_latent.shape
        
        # Initialize hidden state
        h_t = torch.zeros(B, H, device=x_enc_latent.device)
        
        # Process each time step in the reconstructions
        rec = []
        for t in range(F):
            h_t = linearRNN.Wxh(x_enc_latent[:, t, :]) + linearRNN.Whh(h_t)
            rec.append(h_t.unsqueeze(1))
        
        # Get the last hidden state
        h_t = torch.cat(rec, dim=1)[:, -1, :]
        
        # Handle complex eigenvectors by working with real parts only
        # Convert to real-valued operations
        h_t_real = h_t.real() if h_t.is_complex() else h_t
        eigenvectors_real = eigenvectors.real
        eigenvectors_inv_real = eigenvectors_inv.real
        
        # Transform to the eigenbasis using real parts
        h_eig = torch.matmul(eigenvectors_inv_real, h_t_real.unsqueeze(-1)).squeeze(-1)
        
        # Apply the mode mask to filter out unwanted modes
        h_eig_masked = h_eig.clone()
        
        mask_expanded = mode_mask.unsqueeze(0).expand(h_eig.shape[0], -1)
        h_eig_masked = h_eig.clone()
        h_eig_masked[~mask_expanded] = 0
        
        # Transform back to the original basis
        h_t = torch.matmul(eigenvectors_real, h_eig_masked.unsqueeze(-1)).squeeze(-1)
        
        # Generate predictions with isolated modes
        outputs = []
        for _ in range(block.block.step):
            # Apply Koopman operator but only with the selected modes - handling complex values
            h_t_real = h_t.real() if h_t.is_complex() else h_t
            h_eig = torch.matmul(eigenvectors_inv_real, h_t_real.unsqueeze(-1)).squeeze(-1)
            
            # Use real part of eigenvalues to avoid complex number issues
            eigenvalues_real = eigenvalues.real
            h_eig = h_eig * eigenvalues_real * mode_mask
            
            h_t = torch.matmul(eigenvectors_real, h_eig.unsqueeze(-1)).squeeze(-1)
            
            outputs.append(h_t.unsqueeze(1))
        
        # Concatenate the predictions and apply layer norm
        outputs = torch.cat(outputs, dim=1)
        outputs = linearRNN.layer_norm(outputs)
        
        # Decode
        x_pred = block.decoder(outputs)
        
        # Reshape
        if block.CI:
            x_pred = rearrange(x_pred, '(b c) s p -> b (s p) c', c=C)
        else:
            x_pred = rearrange(x_pred, 'b s (p c) -> b (s p) c', c=C)
        
        x_pred_list.append(x_pred)
    
    # Combine the isolated mode predictions
    if len(x_pred_list) == 1:
        combined_x_pred = x_pred_list[0]
        # Apply the inverse normalization
        pred = combined_x_pred * std_enc + mean_enc/2
    else:
        combined_x_pred = sum(x_pred_list)
        # Apply the inverse normalization
        pred = combined_x_pred * std_enc + mean_enc
    
    return pred


# def visualize_modal_contributions(model, x_enc, num_modes=5, block_idx=0, save_path='eigenvalue/isolate'):
#     """
#     Visualize contributions of different Koopman modes to the forecast.
    
#     Parameters:
#     -----------
#     model : Model
#         The trained Koopman model
#     x_enc : torch.Tensor
#         Input tensor of shape [batch_size, seq_len, n_features]
#     num_modes : int
#         Number of top modes to visualize
#     block_idx : int
#         Which block to analyze
        
#     Returns:
#     --------
#     None, displays plots
#     """

#     os.makedirs('eigenvalue/isolate', exist_ok=True)

#     # Get the block
#     block = model.blocks[block_idx]
#     linearRNN = block.block.linearRNN
    
#     # Get the Koopman operator matrix (K)
#     K = linearRNN.Whh.weight.data
    
#     # Perform eigendecomposition
#     eigenvalues, eigenvectors = torch.linalg.eig(K)
    
#     # Convert to numpy for easier handling
#     eigenvalues_np = eigenvalues.cpu().numpy()
    
#     # Sort by magnitude
#     sorted_indices = np.argsort(np.abs(eigenvalues_np))[::-1]
    
#     # Get the full prediction (assuming model returns the prediction directly)
#     with torch.no_grad():
#         full_pred = model(x_enc, None, None, None)[0]
    
#     # Generate predictions with individual modes
#     predictions = []
#     for i in range(min(num_modes, len(sorted_indices))):
#         mode_idx = sorted_indices[i].item()
#         with torch.no_grad():
#             mode_pred = isolate_koopman_mode(model, x_enc, mode_idx=mode_idx, conjugate_pair=True, block_idx=block_idx)
#         predictions.append(mode_pred)
    
#     # Create visualization
#     B, L, C = x_enc.shape
#     future_len = predictions[0].shape[1]
    
#     # Prepare time steps
#     time_steps = np.arange(L + future_len)
    
#     # Plot for each variable
#     num_vars = min(C, 1)  # Plot up to 1 variables
#     fig, axes = plt.subplots(num_vars, 1, figsize=(12, 3*num_vars))
#     if num_vars == 1:
#         axes = [axes]
    
#     for var_idx in range(num_vars):
#         ax = axes[var_idx]
        
#         # Plot input
#         ax.plot(time_steps[:L], x_enc[0, :, var_idx].cpu().numpy(), 'k-', label='Input')
        
#         # Plot full prediction - concatenate the last input value with the prediction
#         pred_data = torch.cat([x_enc[0, -1:, var_idx], full_pred[0, :, var_idx]]).cpu().numpy()
#         ax.plot(time_steps[L-1:], pred_data, 'b-', label='Full Prediction')
        
#         # Plot individual mode contributions
#         for i, pred in enumerate(predictions):
#             mode_idx = sorted_indices[i].item()
#             lambda_i = eigenvalues_np[mode_idx]
#             mag = np.abs(lambda_i)
#             phase = np.angle(lambda_i)
#             freq = phase / (2 * np.pi)
            
#             # Concatenate the last input value with the mode prediction
#             mode_data = torch.cat([x_enc[0, -1:, var_idx], pred[0, :, var_idx]]).cpu().numpy()
#             ax.plot(time_steps[L-1:], mode_data, '--', 
#                     label=f'Mode {mode_idx} (|λ|={mag:.3f}, freq={freq:.3f})')
        
#         ax.set_title(f'Variable {var_idx+1}')
#         ax.legend()
#         ax.grid(True)
#     plt.set_title('Full Prediction vs Selected Koopman Mode Prediction ')
    
#     plt.tight_layout()
#     plt.savefig(f'eigenvalue/isolate/block{block_idx}_time_series.pdf', bbox_inches='tight', dpi=300)    
#     plt.show()
    
#     # Plot eigenvalues in complex plane
#     plt.figure(figsize=(8, 8))
#     plt.scatter(eigenvalues_np.real, eigenvalues_np.imag, c=np.abs(eigenvalues_np), cmap='viridis')
#     plt.colorbar(label='Magnitude')

#         # Then highlight the selected eigenvalues
#     for i in range(min(num_modes, len(sorted_indices))):
#         idx = sorted_indices[i]
#         ev = eigenvalues_np[idx]
#         plt.scatter(ev.real, ev.imag, s=100, c='red', edgecolor='black', zorder=10)
#         plt.text(ev.real, ev.imag, str(idx), fontsize=12, fontweight='bold')
    
#     # Add unit circle
#     theta = np.linspace(0, 2*np.pi, 100)
#     plt.plot(np.cos(theta), np.sin(theta), 'k--')
    
#     # Label the top modes
#     for i in range(min(num_modes, len(sorted_indices))):
#         idx = sorted_indices[i]
#         ev = eigenvalues_np[idx]
#         plt.text(ev.real, ev.imag, str(idx), fontsize=12)
    
#     plt.xlabel('Real')
#     plt.ylabel('Imaginary')
#     plt.title('Koopman Eigenvalues in Complex Plane')
#     plt.axis('equal')
#     plt.grid(True)
#     plt.savefig(f'eigenvalue/isolate/block{block_idx}_eigenvalues.pdf', bbox_inches='tight', dpi=300)    

#     plt.show()
# # Example usage:
# # 
# # 1. To get a forecast using only specific modes from a specific block:
# # isolated_forecast = isolate_koopman_mode(model, x_enc, mode_idx=[0, 1], block_idx=0)
# #
# # 2. To visualize modal contributions:
# # visualize_modal_contributions(model, x_enc, num_modes=5, block_idx=0)