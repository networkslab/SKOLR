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
            # mask = torch.softmax(self.mask_weights, dim=0).unsqueeze(1).unsqueeze(-1)  # Shape: [num_blocks, 1, num_frequencies, 1]
            mask = torch.sigmoid(self.mask_weights).unsqueeze(1).unsqueeze(-1)  # Shape: [num_blocks, 1, num_frequencies, 1]
        elif self.mask_type == 'channel':
            mask = torch.sigmoid(self.mask_weights).unsqueeze(1)  # Shape: [num_blocks, 1, num_frequencies, num_channels]
            # mask = torch.softmax(self.mask_weights, dim=0).unsqueeze(1).unsqueeze(-1)  # Shape: [num_blocks, 1, num_frequencies, 1]
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
        for i in range(self.hidden_layers-1):
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
        N, B, L, H = x.shape

        # Initialize hidden state (assuming zero initial hidden state)
        h_t = torch.zeros(N, B, H, device=x.device)

        # Process each time step
        rec = []
        for t in range(L):
            h_t = self.Wxh(x[:, :, t, :]) + self.Whh(h_t)
            # h_t = x[:, t, :] + self.Whh(h_t)

            # h_t = self.layer_norm(h_t)  # Apply Layer Normalization
            rec.append(h_t.unsqueeze(2))
        # Concatenate the reconstructions along the sequence dimension
        rec = torch.cat(rec, dim=2)  # Shape: (B, L, H)

        # Use the last hidden state to generate predictions
        outputs = []
        for _ in range(pred_len):
            h_t = self.Whh(h_t)  # Predict next step based on the last hidden state
            # h_t = self.layer_norm(h_t)  # Apply Layer Normalization
            outputs.append(h_t.unsqueeze(2))

        # Concatenate the predictions along the sequence dimension
        outputs = torch.cat(outputs, dim=2)  # Shape: (B, P, H)
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
                 linearRNN=None,
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

        # self.linearRNN = LinearRNN(self.dynamic_dim)
        self.linearRNN = linearRNN
        self.CI = CI
        self.inv_loss = inv_loss

    def forward(self, x):
        # x: B L C = [32, 96, 7] plus block number N at the first dimension
        N, B, L, C = x.shape

        res = torch.cat((x[:,:, L-self.padding_len:, :], x) ,dim=2)

        res = res.chunk(self.freq, dim=2)     # F x N B P C, P means seg_len [2, N, 32, 48, 7]
        if self.CI:
            res = rearrange(torch.stack(res, dim=2), 'n b f p c -> n (b c) f p')  # BC F P [32 * 7, 2, 48]
        else:
            res = rearrange(torch.stack(res, dim=2), 'n b f p c -> n b f (p c)')  # BC F P [32, 2, 48 * 7]

        inv_in = res
        x_enc = self.encoder(res) # BC F H [32* 7, 2, 64]
        x_rec, x_pred = self.linearRNN(x_enc, self.step) # B F H, B S H [32 * 7, 1, 64]

        # x_rec = self.decoder(x_rec)    # BC S P [32*7, 1, 48]
        # x_rec = rearrange(x_rec, '(b c) f p -> b (f p) c', c=C)  # [32, 48, 7]
        x_rec = None #save computation

        x_pred = self.decoder(x_pred)     # B S PC [32, 1, 336(48*7)]
        if self.CI:
            x_pred = rearrange(x_pred, 'n (b c) s p -> n b (s p) c', c=C)  # [32, 48, 7]
        else:
            x_pred = rearrange(x_pred, 'n b s (p c) -> n b (s p) c', c=C)

        if self.inv_loss:
            inv_out = self.decoder(x_enc)

            return x_rec, x_pred, (inv_in, inv_out)

        return x_rec, x_pred
        

# class KoopRNN_Block(nn.Module):
#     def __init__(self, num_blocks,
#                  enc_in=8,
#                  input_len=96,
#                  pred_len=96,
#                  seg_len=16,
#                  dynamic_dim=128,
#                  hidden_dim=256,
#                  hidden_layers=1,
#                  dropout=0.05,
#                  CI=False,
#                  inv_loss=False):
#         super(KoopRNN_Block, self).__init__()
#         self.num_blocks = num_blocks
#         self.CI = CI
#         self.inv_loss = inv_loss
        
#         if self.CI:
#             input_dim = self.seg_len
#             output_dim = self.seg_len
#         else:
#             input_dim = self.seg_len * self.enc_in
#             output_dim = self.seg_len * self.enc_in
            
#         self.encoder = MLP(f_in=input_dim, f_out=dynamic_dim, activation='relu', hidden_dim=hidden_dim, hidden_layers=hidden_layers, dropout=dropout) 
#         self.decoder = MLP(f_in=dynamic_dim, f_out=output_dim, activation='relu', hidden_dim=hidden_dim, hidden_layers=hidden_layers, dropout=dropout) 

#         self.block = TimeVarKP(enc_in, input_len, pred_len, seg_len, dynamic_dim, self.encoder, self.decoder, CI, inv_loss)

#     def forward(self, x):
#         # Process the input through each block in parallel
#         results = self.block(x)
        
#         return results

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
                 shared_mlp=True,
                 CI=False,
                 inv_loss=False):
        super(KoopRNN_Block, self).__init__()
        self.num_blocks = num_blocks
        self.CI = CI
        self.inv_loss = inv_loss
        self.shared_mlp = shared_mlp
        
        if self.CI:
            input_dim = seg_len
            output_dim = seg_len
        else:
            input_dim = seg_len * enc_in
            output_dim = seg_len * enc_in
            

        # Create components based on configuration
        if shared_mlp:
            # Config 1: Shared MLP + Separate RNNs
            self.shared_encoder = MLP(f_in=input_dim, f_out=dynamic_dim, activation='relu', 
                                    hidden_dim=hidden_dim, hidden_layers=hidden_layers, dropout=dropout)
            self.shared_decoder = MLP(f_in=dynamic_dim, f_out=output_dim, activation='relu', 
                                    hidden_dim=hidden_dim, hidden_layers=hidden_layers, dropout=dropout)
            # Create separate RNNs for each block
            self.linearRNNs = nn.ModuleList([
                LinearRNN(dynamic_dim) for _ in range(num_blocks)
            ])
            # Create blocks with shared MLPs but separate RNNs
            self.blocks = nn.ModuleList([
                TimeVarKP(enc_in, input_len, pred_len, seg_len, dynamic_dim, 
                        self.shared_encoder, self.shared_decoder, self.linearRNNs[i], CI, inv_loss)
                for i in range(num_blocks)
            ])
        else:
            # Config 2: Separate MLP + Separate RNNs
            # Create separate encoders, decoders, and RNNs for each block
            self.blocks = nn.ModuleList([
                TimeVarKP(
                    enc_in, input_len, pred_len, seg_len, dynamic_dim,
                    MLP(f_in=input_dim, f_out=dynamic_dim, activation='relu', 
                        hidden_dim=hidden_dim, hidden_layers=hidden_layers, dropout=dropout),
                    MLP(f_in=dynamic_dim, f_out=output_dim, activation='relu', 
                        hidden_dim=hidden_dim, hidden_layers=hidden_layers, dropout=dropout),
                    LinearRNN(dynamic_dim),
                    CI, inv_loss
                )
                for _ in range(num_blocks)
            ])

    def forward(self, x):
        # x: [num_blocks, batch_size, seq_len, channels]
        assert x.shape[0] == self.num_blocks, f"Expected {self.num_blocks} blocks in dimension 0, got {x.shape[0]}"
        
        # Process each block in parallel without loops by using a batched approach
        if self.shared_mlp:
            # For shared MLP, we need to handle each block separately for the RNN part,
            # but we can process the encoder and decoder in parallel
            
            # Split x into individual blocks
            x_blocks = x.unbind(0)  # Returns a tuple of tensors along dimension 0
            breakpoint()
            # Process through shared encoder in parallel
            # First reshape to batch all blocks together for parallel processing
            if self.CI:
                enc_input = torch.cat([rearrange(
                    self._prepare_input(x_block.unsqueeze(0)), 'n b f p c -> n (b c) f p') 
                    for x_block in x_blocks], dim=0)
            else:
                enc_input = torch.cat([rearrange(
                    self._prepare_input(x_block.unsqueeze(0)), 'n b f p c -> n b f (p c)') 
                    for x_block in x_blocks], dim=0)
                
            # Store for inv_loss if needed
            inv_in = enc_input
            
            # Forward through shared encoder (processes all blocks in parallel)
            x_enc = self.shared_encoder(enc_input)
            
            # Now we need to separate the encodings for each block's RNN
            x_enc_split = x_enc.chunk(self.num_blocks, dim=0)
            
            # Process through separate RNNs (this is hard to parallelize completely)
            # Use torch.jit.fork/wait for parallel execution
            futures = []
            for i in range(self.num_blocks):
                futures.append(torch.jit.fork(self._forward_rnn, self.linearRNNs[i], x_enc_split[i], self.step))
                
            # Wait for all RNN computations
            rnn_results = [torch.jit.wait(future) for future in futures]
            
            # Unpack RNN results
            x_rec_list = [res[0] for res in rnn_results]
            x_pred_list = [res[1] for res in rnn_results]
            
            # Skip x_rec to save computation
            x_rec = None
            
            # Concatenate predictions for parallel decoder
            x_pred_concat = torch.cat(x_pred_list, dim=0)
            
            # Forward through shared decoder (processes all blocks in parallel)
            x_pred_decoded = self.shared_decoder(x_pred_concat)
            
            # Reshape back to proper format
            N, B, L, C = x.shape
            if self.CI:
                x_pred = rearrange(x_pred_decoded, 'n (b c) s p -> n b (s p) c', n=self.num_blocks, c=C)
            else:
                x_pred = rearrange(x_pred_decoded, 'n b s (p c) -> n b (s p) c', n=self.num_blocks, c=C)
            
            # Handle inv_loss case
            if self.inv_loss:
                inv_out = self.shared_decoder(x_enc)
                return x_rec, x_pred, (inv_in, inv_out)
            
            return x_rec, x_pred
            
        else:
            # For separate MLPs, we can use vmap to truly parallelize across blocks
            # Use torch.vmap if available (requires PyTorch 1.12+)
            # This processes all blocks completely in parallel
            try:
                return self._vmap_forward(x)
            except (AttributeError, RuntimeError):
                # Fallback to jit.fork/wait parallelism if vmap isn't available
                return self._parallel_jit_forward(x)
    
    def _prepare_input(self, x):
        # Helper to prepare padded input for a single block
        # x: [1, batch_size, seq_len, channels]
        _, B, L, C = x.shape
        padding_data = x[:, :, L-self.padding_len:, :]
        padded_x = torch.cat((padding_data, x), dim=2)
        return padded_x.chunk(self.freq, dim=2)
    
    def _forward_rnn(self, rnn, x_enc, step):
        # Helper for parallel RNN computation
        return rnn(x_enc, step)
    
    def _vmap_forward(self, x):
        # Use torch.vmap for true parallelism across blocks
        # This requires PyTorch 1.12+
        
        # Define a function that processes a single block
        def single_block_forward(block_idx, x_block):
            # Process a single block
            return self.blocks[block_idx](x_block.unsqueeze(0))
        
        # Use vmap to parallelize across blocks
        results = torch.vmap(single_block_forward, in_dims=(0, 0))(
            torch.arange(self.num_blocks, device=x.device), 
            x
        )
        
        # Extract and process results
        if self.inv_loss:
            x_rec, x_pred, inv_tuple = results
            inv_in, inv_out = inv_tuple
            return x_rec, x_pred, (inv_in, inv_out)
        else:
            x_rec, x_pred = results
            return x_rec, x_pred
    
    def _parallel_jit_forward(self, x):
        # Fallback parallelism using torch.jit.fork/wait
        # Split x into individual blocks
        x_blocks = x.unbind(0)
        
        # Launch parallel computations
        futures = []
        for i in range(self.num_blocks):
            futures.append(torch.jit.fork(self.blocks[i], x_blocks[i].unsqueeze(0)))
        
        # Wait for all results
        results = [torch.jit.wait(future) for future in futures]
        
        # Process results based on inv_loss
        if self.inv_loss:
            # If we have inv_loss, each result is a tuple (x_rec, x_pred, (inv_in, inv_out))
            x_rec_list = [res[0] for res in results]
            x_pred_list = [res[1] for res in results]
            inv_in_list = [res[2][0] for res in results]
            inv_out_list = [res[2][1] for res in results]
            
            # Stack everything
            x_rec = torch.stack(x_rec_list, dim=0) if x_rec_list[0] is not None else None
            x_pred = torch.stack(x_pred_list, dim=0)
            inv_in = torch.stack(inv_in_list, dim=0)
            inv_out = torch.stack(inv_out_list, dim=0)
            
            return x_rec, x_pred, (inv_in, inv_out)
        else:
            # If no inv_loss, each result is a tuple (x_rec, x_pred)
            x_rec_list = [res[0] for res in results]
            x_pred_list = [res[1] for res in results]
            
            # Stack everything
            x_rec = torch.stack(x_rec_list, dim=0) if x_rec_list[0] is not None else None
            x_pred = torch.stack(x_pred_list, dim=0)
            
            return x_rec, x_pred


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
        self.shareEncoder = configs.shareEncoder
        self.disentanglement = FourierFilter(int(self.input_len/2)+1, self.enc_in, self.num_blocks, self.mask_type)
  
        self.inv_loss = configs.inv_loss > 0
        
        # Initialize KoopRNN blocks based on sharing configuration
        self.blocks = KoopRNN_Block(
                num_blocks=self.num_blocks,
                enc_in=self.enc_in,
                input_len=self.input_len,
                pred_len=self.pred_len,
                seg_len=self.seg_len,
                dynamic_dim=self.dynamic_dim,
                hidden_dim=self.hidden_dim,
                hidden_layers=self.hidden_layers,
                dropout=self.dropout,
                shared_mlp=self.shareEncoder,  # Share/Sep MLPs across blocks
                CI=self.CI,
                inv_loss=self.inv_loss
            )
            
        # elif self.shareEncoder == 'part':
        #     # For future extension - partial sharing
        #     print('On construction. Come back later...')
        #     exit()
        # else:
        #     raise ValueError(f"Unknown shareEncoder mode: {self.shareEncoder}")


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
        # to tensor, to acheive parallel computation
        ifft_results = torch.stack(ifft_results, dim=0) # N B L C
        results = self.blocks(ifft_results)  # [num_blocks, batch_size, seq_len, channels]
        
        # Get x_rec and x_pred from results tensor
        x_rec, x_pred = results[0], results[1]  # Each is [num_blocks, batch_size, seq_len, channels]
        
        # Sum along block dimension
        combined_x_pred = x_pred.sum(dim=0)  # [batch_size, seq_len, channels]
        
        if self.blocks.inv_loss:
            inv_in, inv_out = results[2]  # Get inv tensors [num_blocks, batch_size, seq_len, channels]
            combined_inv_in = inv_in.sum(dim=0)
            combined_inv_out = inv_out.sum(dim=0)
            return x_rec, combined_x_pred, (combined_inv_in, combined_inv_out)
        
        # Denormalize predictions
        pred = combined_x_pred * std_enc + mean_enc
        # block_pred = torch.stack(x_pred_list, dim=0)  # [num_block, batch, seq_len, channels]
        x_pred = x_pred * std_enc + mean_enc

        breakpoint()
        return pred
