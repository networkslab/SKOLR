import math
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat, einsum

# Multihead KoopRNN


class FourierFilter(nn.Module):
    """
    Fourier Filter with Learnable Masks.
    Supports both global (shared across all channels) and channel-wise masks.
    """
    def __init__(self, num_frequencies, num_channels, num_blocks, mask_type='global'):
        """
        Parameters:
            num_frequencies: Number of frequency bins (from rfft).
            num_channels: Number of channels in the input signal.
            num_blocks: Number of blocks to decompose the input into.
            mask_type: 'global' for a shared mask, 'channel-wise' for independent masks per channel.
        """
        super(FourierFilter, self).__init__()
        self.num_frequencies = num_frequencies
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.mask_type = mask_type

        if mask_type == 'global':
            # Learnable global mask: matrix [num_blocks, num_frequencies] shared across all channels
            self.mask_weights = nn.Parameter(torch.rand(num_blocks, num_frequencies))  # Random initialization
        elif mask_type == 'channel':
            # Learnable channel-wise mask: tensor [num_blocks, num_frequencies, num_channels] with per-channel masks
            self.mask_weights = nn.Parameter(torch.rand(num_blocks, num_frequencies, num_channels))  # Random initialization
        elif mask_type == 'FixHalf':
            # Fixed mask with all values set to 1/num_blocks
            self.mask_weights = torch.full((num_blocks, num_frequencies, num_channels), 1/num_blocks)  # Non-trainable tensor
        else:
            raise ValueError("mask_type must be 'global', 'channel', or 'FixHalf'")

    def forward(self, x):
        """
        Forward pass of the Fourier Filter.
        x: Input time-series signal of shape [batch_size, sequence_length, num_channels].
        Returns:
            ifft_results: List of IFFT results for each block.
        """
        # Perform FFT on the input signal
        xf = torch.fft.rfft(x, dim=1)  # Shape: [batch_size, freq_bins, num_channels]

        # Create a soft mask from the learnable weights
        if self.mask_type == 'global':
            mask = torch.sigmoid(self.mask_weights).unsqueeze(1).unsqueeze(-1)  # Shape: [num_blocks, 1, num_frequencies, 1]
        elif self.mask_type == 'channel':
            mask = torch.sigmoid(self.mask_weights).unsqueeze(1)  # Shape: [num_blocks, 1, num_frequencies, num_channels]
        elif self.mask_type == 'FixHalf':
            mask = self.mask_weights.unsqueeze(1)  # Shape: [num_blocks, 1, num_frequencies, num_channels]
            mask = mask.to(xf.device)

        # Apply the mask to the frequency representation
        masked_xf = xf.unsqueeze(0) * mask  # Shape: [num_blocks, batch_size, num_frequencies, num_channels]

        # Inverse FFT to reconstruct the time-domain signal for each block
        ifft_results = [torch.fft.irfft(masked_xf[i], dim=1) for i in range(self.num_blocks)]

        return ifft_results


class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data

    '''
    def __init__(self, 
                 f_in, 
                 f_out, 
                 hidden_dim=128, 
                 hidden_layers=2, 
                 dropout=0.05,
                 activation='tanh'): 
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim), 
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers-2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]
        
        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y

class LinearRNN(nn.Module):
    def __init__(self, hidden_dim):
        super(LinearRNN, self).__init__()

        self.hidden_dim = hidden_dim

        # Linear transformation to update hidden state
        self.Wxh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Whh = nn.Linear(hidden_dim, hidden_dim, bias=False) #This is K

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, pred_len=1):
        B, L, H = x.shape

        # Initialize hidden state (assuming zero initial hidden state)
        h_t = torch.zeros(B, H, device=x.device)

        # Process each time step
        rec = []
        for t in range(L):
            h_t = self.Wxh(x[:, t, :]) + self.Whh(h_t)
            # h_t = x[:, t, :] + self.Whh(h_t)

            # h_t = self.layer_norm(h_t)  # Apply Layer Normalization
            rec.append(h_t.unsqueeze(1))
        # Concatenate the reconstructions along the sequence dimension
        rec = torch.cat(rec, dim=1)  # Shape: (B, L, H)

        # Use the last hidden state to generate predictions
        outputs = []
        for _ in range(pred_len):
            h_t = self.Whh(h_t)  # Predict next step based on the last hidden state
            # h_t = self.layer_norm(h_t)  # Apply Layer Normalization
            outputs.append(h_t.unsqueeze(1))

        # Concatenate the predictions along the sequence dimension
        outputs = torch.cat(outputs, dim=1)  # Shape: (B, P, H)
        outputs = self.layer_norm(outputs)
        return rec, outputs



class TimeVarKP(nn.Module):
    """
    Koopman Predictor with DMD (analysitical solution of Koopman operator)
    Utilize local variations within individual sliding window to predict the future of time-variant term
    """
    def __init__(self,
                 enc_in=8,
                 input_len=96,
                 pred_len=96,
                 seg_len=24,
                 dynamic_dim=128,
                 encoder=None,
                 decoder=None,
                 CI=False,
                 inv_loss=False,
                ):
        super(TimeVarKP, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.seg_len = seg_len
        self.dynamic_dim = dynamic_dim
        self.encoder, self.decoder = encoder, decoder
        self.freq = math.ceil(self.input_len / self.seg_len)  # segment number of input
        self.step = math.ceil(self.pred_len / self.seg_len)   # segment number of output
        self.padding_len = self.seg_len * self.freq - self.input_len
        # Approximate mulitstep K by KPLayerApprox when pred_len is large
        # self.dynamics = KPLayerApprox() if self.multistep else KPLayer()
        self.linearRNN = LinearRNN(self.dynamic_dim)
        self.CI = CI
        self.inv_loss = inv_loss

    def forward(self, x):
        # x: B L C = [32, 96, 7]
        B, L, C = x.shape

        res = torch.cat((x[:, L-self.padding_len:, :], x) ,dim=1)

        res = res.chunk(self.freq, dim=1)     # F x B P C, P means seg_len [2, 32, 48, 7]
        if self.CI:
            res = rearrange(torch.stack(res, dim=1), 'b f p c -> (b c) f p')  # BC F P [32 * 7, 2, 48]
        else:
            res = rearrange(torch.stack(res, dim=1), 'b f p c -> b f (p c)')  # BC F P [32, 2, 48 * 7]

        inv_in = res
        x_enc = self.encoder(res) # BC F H [32* 7, 2, 64]
        x_rec, x_pred = self.linearRNN(x_enc, self.step) # B F H, B S H [32 * 7, 1, 64]

        # x_rec = self.decoder(x_rec)    # BC S P [32*7, 1, 48]
        # x_rec = rearrange(x_rec, '(b c) f p -> b (f p) c', c=C)  # [32, 48, 7]
        x_rec = None #save computation

        x_pred = self.decoder(x_pred)     # B S PC [32, 1, 336(48*7)]
        if self.CI:
            x_pred = rearrange(x_pred, '(b c) s p -> b (s p) c', c=C)  # [32, 48, 7]
        else:
            x_pred = rearrange(x_pred, 'b s (p c) -> b (s p) c', c=C)

        if self.inv_loss:
            inv_out = self.decoder(x_enc)

            return x_rec, x_pred, (inv_in, inv_out)

        return x_rec, x_pred

class KoopRNN_Block(nn.Module):
    def __init__(self, num_blocks,
                 enc_in=8,
                 input_len=96,
                 pred_len=96,
                 seg_len=16,
                 dynamic_dim=128,
                 hidden_dim=256,
                 hidden_layers=1,
                 dropout=0.05,
                 CI=False,
                 inv_loss=False):
        super(KoopRNN_Block, self).__init__()
        self.num_blocks = num_blocks
        self.CI = CI
        self.inv_loss = inv_loss
        if self.CI:
            self.encoder = MLP(f_in=seg_len, f_out=dynamic_dim, activation='relu', hidden_dim=hidden_dim, hidden_layers=hidden_layers, dropout=dropout)
            self.decoder = MLP(f_in=dynamic_dim, f_out=seg_len, activation='relu', hidden_dim=hidden_dim, hidden_layers=hidden_layers, dropout=dropout) 
        else:
            self.encoder = MLP(f_in=seg_len * enc_in, f_out=dynamic_dim, activation='relu', hidden_dim=hidden_dim, hidden_layers=hidden_layers, dropout=dropout) 
            self.decoder = MLP(f_in=dynamic_dim, f_out=seg_len * enc_in, activation='relu', hidden_dim=hidden_dim, hidden_layers=hidden_layers, dropout=dropout) 

        self.block = TimeVarKP(enc_in, input_len, pred_len, seg_len, dynamic_dim, self.encoder, self.decoder, CI, inv_loss)

    def forward(self, x):
        # Process the input through each block in parallel
        results = self.block(x)
        
        return results


# class TimeInvKP(nn.Module):
#     """
#     Koopman Predictor with learnable Koopman operator
#     Utilize lookback and forecast window snapshots to predict the future of time-invariant term
#     """

#     def __init__(self,
#                  input_len=96,
#                  pred_len=96,
#                  dynamic_dim=128,
#                  encoder=None,
#                  decoder=None):
#         super(TimeInvKP, self).__init__()
#         self.dynamic_dim = dynamic_dim
#         self.input_len = input_len
#         self.pred_len = pred_len
#         self.encoder = encoder
#         self.decoder = decoder

#         K_init = torch.randn(self.dynamic_dim, self.dynamic_dim)
#         U, _, V = torch.svd(K_init)  # stable initialization
#         self.K = nn.Linear(self.dynamic_dim, self.dynamic_dim, bias=False)
#         self.K.weight.data = torch.mm(U, V.t())

#     def forward(self, x):
#         # x: B L C
#         res = x.transpose(1, 2)  # B C L
#         res = self.encoder(res)  # B C H
#         res = self.K(res)  # B C H
#         res = self.decoder(res)  # B C S
#         res = res.transpose(1, 2)  # B S C

#         return res


class Model(nn.Module):
    '''
    Koopman Forecasting Model
    '''
    def __init__(self, configs):
        super(Model, self).__init__()
        self.mask_spectrum = configs.mask_spectrum
        self.enc_in = configs.enc_in
        self.input_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = configs.seg_len
        self.num_blocks = configs.num_blocks
        self.dynamic_dim = configs.dynamic_dim
        self.hidden_dim = configs.hidden_dim
        self.hidden_layers = configs.hidden_layers
        self.multistep = configs.multistep
        self.dropout = configs.dropout
        self.CI = configs.CI
        self.mask_type = configs.mask_type

        self.disentanglement = FourierFilter(int(self.input_len/2)+1, self.enc_in, self.num_blocks,self.mask_type)
        # self.disentanglement = FourierFilter(self.mask_spectrum)

        # # shared encoder/decoder to make koopman embedding consistent
        # self.time_inv_encoder = MLP(f_in=self.input_len, f_out=self.dynamic_dim, activation='relu',
        #             hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers, dropout=self.dropout)
        # self.time_inv_decoder = MLP(f_in=self.dynamic_dim, f_out=self.pred_len, activation='relu',
        #                    hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers, dropout=self.dropout)
        # self.time_inv_kps = TimeInvKP(input_len=self.input_len,
        #                             pred_len=self.pred_len,
        #                             dynamic_dim=self.dynamic_dim,
        #                             encoder=self.time_inv_encoder,
        #                             decoder=self.time_inv_decoder)

        # shared encoder/decoder to make koopman embedding consistent
        # if self.CI:
        #     self.time_var_encoder = MLP(f_in=self.seg_len, f_out=self.dynamic_dim, activation='relu',
        #                    hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers, dropout=self.dropout)
        #     self.time_var_decoder = MLP(f_in=self.dynamic_dim, f_out=self.seg_len, activation='relu',
        #                    hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers, dropout=self.dropout)
        # else:
        #     self.time_var_encoder = MLP(f_in=self.seg_len * self.enc_in, f_out=self.dynamic_dim, activation='relu',
        #                                 hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers,
        #                                 dropout=self.dropout)
        #     self.time_var_decoder = MLP(f_in=self.dynamic_dim, f_out=self.seg_len * self.enc_in, activation='relu',
        #                                 hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers,
        #                                 dropout=self.dropout)

        self.inv_loss = configs.inv_loss > 0
        self.shareEncoder = False #placeholder
        # self.time_var_kps = TimeVarKP(enc_in=configs.enc_in,
        #                 input_len=self.input_len,
        #                 pred_len=self.pred_len,
        #                 seg_len=self.seg_len,
        #                 dynamic_dim=self.dynamic_dim,
        #                 encoder=self.time_var_encoder,
        #                 decoder=self.time_var_decoder,
        #                 multistep=self.multistep,
        #                 CI = self.CI,
        #                 inv_loss=self.inv_loss
        #                               )

        # Initialize KoopRNN blocks
        if not self.shareEncoder:
            self.blocks = nn.ModuleList([
                            KoopRNN_Block(self.num_blocks,
                                enc_in=self.enc_in,
                                input_len=self.input_len,
                                pred_len=self.pred_len,
                                seg_len=self.seg_len,
                                dynamic_dim=self.dynamic_dim,
                                hidden_dim=self.hidden_dim,
                                hidden_layers=self.hidden_layers,
                                dropout=self.dropout,
                                CI = self.CI,
                                inv_loss=self.inv_loss
                                              )
                            for _ in range(self.num_blocks)
            ])

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: B L C
        # seq_last = x_enc[:, -1:, :].detach()
        # x_enc = x_enc - seq_last
        # Series Stationarization adopted from NSformer
        mean_enc = x_enc.mean(1, keepdim=True).detach() # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        # Koopman Forecasting
        residual, forecast = x_enc, None
        ifft_results = self.disentanglement(residual) # list of input for each block
        results = [self.blocks[i](ifft_results[i]) for i in range(self.num_blocks)]
        
        # Separate the results into x_rec and x_pred
        x_rec_list, x_pred_list = zip(*[(res[0], res[1]) for res in results])

        combined_x_pred = sum(x_pred_list)
        # combine_res = sum(x_rec_list) #placeholder, now res is None

        if self.blocks[0].inv_loss:
            inv_in_list, inv_out_list = zip(*[res[2] for res in results])
            combined_inv_in = sum(inv_in_list) 
            combined_inv_out = sum(inv_out_list) 
            return x_rec_list, combined_x_pred, (combined_inv_in, combined_inv_out)

        # Series Stationarization adopted from NSformer

        
        
        pred = combined_x_pred * std_enc + mean_enc
        
        return pred

