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
        # branches = []
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

                        inv_in_out = None ### change it back later, This is for plotting branches
                        if isinstance(outputs, (tuple, list)):
                            outputs, inv_in_out = outputs
                        # branch_pred = None ### This is for plotting branches
                        # if isinstance(outputs, (tuple, tuple)):
                        #     outputs, branch_pred = outputs
                            
                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                # branches.append(branch_pred.detach().cpu().numpy())

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




                if self.args.test_flop:
                    test_params_flop((batch_x.shape[1],batch_x.shape[2]))
                    exit()
            
        # print("Model Num. Parameters:", sum([p.numel() for p in self.model.parameters()]))
        # print("Mem. Info:", torch.cuda.max_memory_allocated())
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0) #b, l, d
        # branches= np.concatenate(branches, axis=1)# n, b, l, d
        
        # branches = branches[0].detach().cpu().numpy()         #if just save one batch
        
        # preds = rearrange(preds, 'b l d -> b d l')
        # trues = rearrange(trues, 'b l d -> b d l')
        # inputx = rearrange(inputx, 'b l d -> b d l')


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
            # np.save(folder_path + 'branches.npy', branches)

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
