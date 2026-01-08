import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# class Model(nn.Module):
#     def __init__(self, configs, activation=torch.relu, r_min=0.9, r_max=0.999, use_bias=True, #in_features,
#                  unroll=True):
#         super(Model, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         in_features = 2
#         self.hidden_size = in_features
#         self.activation = activation
#         self.use_bias = use_bias
#         self.unroll = unroll  # The parallel algorithm will divide and conquer more if True

#         self.i_dense = nn.Linear(in_features, in_features * 2, bias=use_bias)  # Extend to the complex C
#         self.o_dense = nn.Linear(in_features * 2, in_features, bias=use_bias)  # Back to real R

#         # Initialize parameters
#         u1 = np.random.random(size=in_features)
#         u2 = np.random.random(size=in_features)
#         v_log = np.log(-0.5 * np.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))  # defined in [arxiv] lemma 3.2
#         theta_log = np.log(u2 * np.pi * 2)  # defined in [arxiv] lemma 3.2
#         gamma_log = np.log(np.sqrt(1 - np.exp(-np.exp(v_log)) ** 2))  # defined above eq.7 of [arxiv]

#         # defined in Optimization under exponential parameterization of [arxiv] 3.3
#         self.params_log = nn.Parameter(torch.tensor([v_log, theta_log, gamma_log], dtype=torch.float32))

#     def lru_parallel(self, i, x, v, theta, B, L, D):
#         # Upper/low parallel algorithm, see: https://kexue.fm/archives/9554#%E5%B9%B6%E8%A1%8C%E5%8C%96
#         l = 2 ** i
#         x = x.reshape(B * L // l, l, D)  # (B, L, D) -> (B * L // 2, 2, D)
#         x1, x2 = x[:, :l // 2], x[:, l // 2:]  # Divide the data in half

#         pos = torch.arange(1, l // 2 + 1, dtype=torch.float, device=x.device)  # t=k+1 ~ T
#         vs = torch.einsum('n,d->nd', pos, v)
#         thetas = torch.einsum('n,d->nd', pos, theta)
#         lambs = torch.exp(
#             torch.complex(-vs, thetas))  # defined in Optimization under exponential parameterization of [arxiv] 3.3

#         x2 = x2 + (lambs * x1[:, -1:])  # Add the last element of the half to the second half
#         x = torch.cat([x1, x2], axis=1)
#         if (not self.unroll) and x.shape[1] is not None:
#             x = x.reshape(B, L, D)

#         return i + 1, x, v, theta, B, L, D

#     def while_loop(self, cond, body, loop_vars):
#         while cond(*loop_vars[:2]):
#             loop_vars = body(*loop_vars)
#         return loop_vars

#     def forward(self, inputs):
#         print(inputs.shape)
#         exit()
#         u = self.i_dense(inputs)
#         params = torch.exp(self.params_log)
#         v, theta, gamma = params[0], params[1], params[2]

#         len_seq_in = u.size(1)
#         log2_L = int(np.ceil(np.log2(len_seq_in)))

#         u = torch.view_as_complex(u.view(u.size(0), u.size(1), u.size(2) // 2, 2))
#         u = F.pad(u,
#                   (0, 0, 0, 2 ** log2_L - u.size(1), 0, 0))  # pad the sequence length to the power of 2 (for algorithm)
#         B, L, D = u.size(0), u.size(1), u.size(2)

#         if self.unroll:
#             x = u  # init hidden states as inputs
#             for i in range(log2_L):
#                 _, x, *_ = self.lru_parallel(i + 1, x, v, theta, B, L, D)
#         else:
#             _, x, *_ = self.while_loop(lambda i, x: i <= log2_L, self.lru_parallel, [1, u, v, theta, B, L, D])

#         x = x[:, :len_seq_in] * (gamma.to(torch.complex64) + 0j)  # Element-wise parameter defined in [arxiv] eq.(7)
#         x = self.complex_to_real_imag(x)
#         output = self.o_dense(x)
#         if self.activation is not None:
#             output = self.activation(output)

#         return output

#     def complex_to_real_imag(self, x):
#         real_x = torch.real(x)
#         imag_x = torch.imag(x)
#         return torch.cat((real_x, imag_x), dim=-1)

#     @torch.no_grad()
#     def infer_step(self, input_t, hidden_state_t_1=None):
#         u_t = self.i_dense(input_t)
#         u_t = torch.view_as_complex(u_t.view(u_t.size(0), u_t.size(1), u_t.size(2) // 2, 2))
#         params = torch.exp(self.params_log)
#         v, theta, gamma = params[0], params[1], params[2]

#         if hidden_state_t_1 is None:
#             x_t = u_t
#         else:
#             x_t_1 = hidden_state_t_1
#             lamb = torch.exp(torch.complex(-v, theta))
#             x_t = lamb * x_t_1 + u_t
#         y_t = x_t * (gamma.to(torch.complex64) + 0j)  # Element-wise parameter defined in [arxiv] eq.(7)
#         y_t = self.complex_to_real_imag(y_t)
#         y_t = self.o_dense(y_t)
#         if self.activation is not None:
#             y_t = self.activation(y_t)
#         return x_t, y_t

#     @torch.no_grad()
#     def infer_steps(self, input_t, hidden_state_t_1=None):
#         u_t = self.i_dense(input_t)
#         u_t = torch.view_as_complex(u_t.view(u_t.size(0), u_t.size(1), u_t.size(2) // 2, 2))
#         params = torch.exp(self.params_log)
#         v, theta, gamma = params[0], params[1], params[2]

#         x_t_1 = torch.zeros((u_t.shape[0], u_t.shape[2])) if hidden_state_t_1 is None else hidden_state_t_1[:, -1]
#         lamb = torch.exp(torch.complex(-v, theta))
#         x_t = torch.zeros_like(u_t)
#         for i in range(u_t.shape[1]):
#             x_t_t = lamb * x_t_1 + u_t[:, i]
#             x_t_1 = x_t_t

#             x_t[:, i, :] = x_t_t

#         y_t = x_t * (gamma.to(torch.complex64) * 1j)  # Element-wise parameter defined in [arxiv] eq.(7)
#         y_t = self.complex_to_real_imag(y_t)
#         y_t = self.o_dense(y_t)
#         if self.activation is not None:
#             y_t = self.activation(y_t)
#         return x_t, y_t

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, configs, input_size=2, hidden_size=128, output_size=2, dtype=torch.complex64):
        """
        LRU block for long sequence modeling.
        
        Args:
            input_size (int): Dimension of input u_k.
            hidden_size (int): Dimension of the recurrent state x_k.
            output_size (int): Dimension of the output y_k.
            dtype: Complex dtype for recurrent computations.
        """
        super(Model, self).__init__()
        self.input_size = configs.enc_in
        self.hidden_size = hidden_size
        self.output_size = configs.c_out
        self.dtype = dtype

        # Learnable parameters for the recurrent eigenvalues (in log domain)
        # nu_log controls the magnitude (decay) and theta_log controls the phase (oscillation)
        self.nu_log = nn.Parameter(torch.zeros(self.hidden_size))
        self.theta_log = nn.Parameter(torch.zeros(self.hidden_size))

        # Real-valued input-to-hidden matrix B, hidden-to-output matrix C, and skip connection D
        self.B = nn.Parameter(torch.Tensor(hidden_size, self.input_size))
        self.C = nn.Parameter(torch.Tensor(self.output_size, self.hidden_size))
        self.D = nn.Parameter(torch.Tensor(self.output_size, self.input_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        nn.init.xavier_uniform_(self.D)
        # Initialize nu_log to a negative value for a modest decay (stability)
        nn.init.constant_(self.nu_log, -1.0)
        nn.init.constant_(self.theta_log, 0.0)

    # def forward(self, u):
    #     """
    #     Processes an input sequence u through the LRU block.
        
    #     Args:
    #         u: Tensor of shape (batch_size, seq_length, input_size)
    #            containing the input sequence (real-valued).
        
    #     Returns:
    #         outputs: Tensor of shape (batch_size, seq_length, output_size)
    #                  containing the output at each time step.
    #     """
    #     batch_size, seq_length, _ = u.shape

    #     # Compute complex eigenvalues lambda:
    #     # lambda_j = exp(-exp(nu_log_j) + i * exp(theta_log_j))
    #     mag = torch.exp(-torch.exp(self.nu_log))   # Magnitude part, shape: (hidden_size,)
    #     phase = torch.exp(self.theta_log)            # Phase part, shape: (hidden_size,)
    #     lambda_complex = mag * torch.exp(1j * phase)   # Complex eigenvalues

    #     # Compute normalization gamma = sqrt(1 - |lambda|^2) to ensure stability.
    #     gamma = torch.sqrt(1 - (mag ** 2) + 1e-8)      # (hidden_size,)

    #     # Initialize hidden state x (complex-valued) as zero.
    #     x = torch.zeros(batch_size, self.hidden_size, device=u.device, dtype=self.dtype)
    #     outputs = []

    #     # Process the sequence one time step at a time.
    #     for t in range(seq_length):
    #         u_t = u[:, t, :]  # shape: (batch_size, input_size)
    #         # Compute B * u_t; then convert to complex for recurrent arithmetic.
    #         Bu = F.linear(u_t, self.B)
    #         Bu = Bu.to(self.dtype)
    #         # Update hidden state: x = gamma * (lambda * x) + B * u_t
    #         x = gamma * (lambda_complex * x) + Bu
    #         # Compute output: y_t = C * Re(x) + D * u_t.
    #         y_t = F.linear(x.real, self.C) + F.linear(u_t, self.D)
    #         outputs.append(y_t.unsqueeze(1))
        
    #     outputs = torch.cat(outputs, dim=1)  # shape: (batch_size, seq_length, output_size)
    #     return outputs

    def forward_parallel(self, u):
        """
        Parallel (vectorized) implementation of the LRU recurrence.
        
        Args:
            u: Real-valued input tensor of shape (batch_size, seq_length, input_size)
        
        Returns:
            outputs: Tensor of shape (batch_size, seq_length, output_size)
        """
        batch_size, seq_length, _ = u.shape
    
        # Compute the complex eigenvalues:
        # lambda_j = exp(-exp(nu_log_j) + i * exp(theta_log_j))
        mag = torch.exp(-torch.exp(self.nu_log))   # shape: (hidden_size,)
        phase = torch.exp(self.theta_log)            # shape: (hidden_size,)
        lambda_complex = mag * torch.exp(1j * phase)   # shape: (hidden_size,)
    
        # Normalization factor gamma = sqrt(1 - |lambda|^2)
        gamma = torch.sqrt(1 - (mag ** 2) + 1e-8)      # shape: (hidden_size,)
    
        # Compute F for each time step: F_t = gamma ⊙ (B * u_t)
        # F_t shape: (batch_size, seq_length, hidden_size)
        F_t = F.linear(u, self.B)  
        F_t = F_t.to(self.dtype) * gamma.unsqueeze(0).unsqueeze(0)
    
        # Create a time index vector: shape (seq_length, 1)
        t_vec = torch.arange(seq_length, device=u.device, dtype=self.B.dtype).unsqueeze(1)
        # Compute p_t = lambda_complex^t; shape: (seq_length, hidden_size)
        p = torch.pow(lambda_complex.unsqueeze(0), t_vec)
        # Expand p to shape: (batch_size, seq_length, hidden_size)
        p = p.unsqueeze(0).expand(batch_size, -1, -1)
        # Clamp p to avoid division by extremely small values.
        p = p + 1e-8
    
        # Compute F_div = F_t / p elementwise
        F_div = F_t / p
    
        # Compute cumulative sum over time: S_t = sum_{j=0}^{t} F_div[j]
        S = torch.cumsum(F_div, dim=1)
    
        # Reconstruct the hidden state: x_t = p_t ⊙ S_t
        x = p * S
    
        # Compute outputs: y_t = C * Re(x_t) + D * u_t.
        x_real = x.real  # take the real part of the complex hidden state
        y = F.linear(x_real, self.C) + F.linear(u, self.D)
        return y


    def forward(self, u):
        """
        By default, we use the parallel scan version for efficiency.
        
        Args:
            u: Real-valued input tensor of shape (batch_size, seq_length, input_size)
        
        Returns:
            outputs: Tensor of shape (batch_size, seq_length, output_size)
        """
        return self.forward_parallel(u)
