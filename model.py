import torch
from torch import nn
from torch.nn import init
import numpy as np

class SlotAttention(nn.Module):
    """refer to https://github.com/lucidrains/slot-attention"""
    def __init__(self, num_slots, slotdim, iters = 3, eps = 1e-8, mlp_hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = slotdim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, slotdim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slotdim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(slotdim, slotdim)
        self.to_k = nn.Linear(slotdim, slotdim)
        self.to_v = nn.Linear(slotdim, slotdim)

        self.gru = nn.GRUCell(slotdim, slotdim)

        mlp_hidden_dim = max(slotdim, mlp_hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(slotdim, mlp_hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(mlp_hidden_dim, slotdim)
        )

        self.norm_input  = nn.LayerNorm(slotdim)
        self.norm_slots  = nn.LayerNorm(slotdim)
        self.norm_pre_ff = nn.LayerNorm(slotdim)

    def forward(self, inputs, num_slots = None):
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device, dtype = dtype)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps

            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots
    
class SlotAttentionAutoEncoder(nn.Module):
  """Slot Attention-based auto-encoder for object discovery."""

  def __init__(self, resolution, num_slots, num_iterations):
    """Builds the Slot Attention-based auto-encoder.

    Args:
      resolution: Tuple of integers specifying width and height of input image.
      num_slots: Number of slots in Slot Attention.
      num_iterations: Number of iterations in Slot Attention.
    """
    super(SlotAttentionAutoEncoder,self).__init__()
    self.resolution = resolution #(128,128)
    self.num_slots = num_slots
    self.num_iterations = num_iterations

    encoder_block = [nn.Conv2d(in_channels=3,out_channels=64, kernel_size=5, stride=1, padding=2,),nn.ReLU()]
    for _ in range(3):
       encoder_block += [nn.Conv2d(in_channels=64,out_channels=64, kernel_size=5, stride=1, padding=2,),nn.ReLU()] 
    self.encoder_cnn = nn.Sequential(*encoder_block)

    self.decoder_initial_size = (8, 8)
    self.decoder_cnn = nn.Sequential(
        nn.ConvTranspose2d(64,64, 4, stride=(2, 2), padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64,64, 4, stride=(2, 2), padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64,64, 4, stride=(2, 2), padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64,64, 4, stride=(2, 2), padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64,64, 5, stride=(1, 1), padding=2),
        nn.ReLU(),
        nn.ConvTranspose2d(64,4, 3, stride=(1, 1), padding=1))

    self.encoder_pos = SoftPositionEmbed(64, self.resolution)
    self.decoder_pos = SoftPositionEmbed(64, self.decoder_initial_size)

    self.layer_norm = nn.LayerNorm(normalized_shape=(self.resolution[0]*self.resolution[1]))
    self.ch_conv = nn.Sequential(nn.Conv2d(64,64,1,stride=1,padding=0),
        nn.ReLU(),
        nn.Conv2d(64,64,1,stride=1,padding=0))

    self.slot_attention = SlotAttention(
        iters=self.num_iterations,
        num_slots=self.num_slots,
        slotdim=64,
        mlp_hidden_dim=128)
    
  def forward(self, image):
      # `image` has shape: [batch_size,num_channels, width, height ].
      # Convolutional encoder with position embedding.
      x = self.encoder_cnn(image)  # CNN Backbone.
      x = self.encoder_pos(x)  # Position embedding.
      x = self.spatial_flatten(x)  # Flatten spatial dimensions (treat image as set).
      x = self.ch_conv(self.layer_norm(x))  # Feedforward network on set.
      # `x` has shape: [batch_size,64, width*height ] 
      #注意送進 slot attention module的(B,N, dim)為(batch, N==每一個像素, dim==每一個像素用幾個channel去表示)所以需要permute軸
      x = x.permute(0,2,1)
      # Slot Attention module.
      slots = self.slot_attention(x)
      # `slots` has shape: [batch_size, num_slots, slot_size].

      # Spatial broadcast decoder.
      x = self.spatial_broadcast(slots, self.decoder_initial_size)
      # `x` has shape: [batch_size*num_slots,slot_size, width_init, height_init].
      x = self.decoder_pos(x)
      x = self.decoder_cnn(x)
      # `x` has shape: [batch_size*num_slots,num_channels+1, width, height].

      # Undo combination of slot and batch dimension; split alpha masks.
      recons, masks = self.unstack_and_split(x, batch_size=image.shape[0])
      # `recons` has shape: [batch_size, num_slots, num_channels, width, height].
      # `masks` has shape: [batch_size, num_slots, 1, width, height].

      # Normalize alpha masks over slots.
      masks = torch.softmax(masks, axis=1)
      recon_combined = torch.sum(recons * masks, axis=1)  # Recombine image.
      # `recon_combined` has shape: [batch_size,num_channels, width, height ].

      return recon_combined, recons, masks, slots

  def spatial_flatten(self,x):
      return torch.reshape(x, [-1, x.shape[1], x.shape[2] * x.shape[3]]) # B,C, H*W

  def spatial_broadcast(self,slots, resolution):
      """Broadcast slot features to a 2D grid and collapse slot dimension."""
      # `slots` has shape: [batch_size, num_slots, slot_size].
      slots = torch.reshape(slots, [-1, slots.shape[-1]]).unsqueeze(2).unsqueeze(3)
      grid = torch.tile(slots, [1, 1,resolution[0], resolution[1]])
      # `grid` has shape: [batch_size*num_slots,slot_size, width, height ].
      return grid

  def unstack_and_split(self,x, batch_size, num_channels=3):
    """Unstack batch dimension and split into channels and alpha mask."""
    unstacked = torch.reshape(x, [batch_size, -1] + list(x.shape)[1:]) # shape: [batch_size, num_slots, num_channels+1, width, height].
    channels, masks = torch.split(unstacked, [num_channels, 1], dim=2)
    return channels, masks


class SoftPositionEmbed(nn.Module):
  """Adds soft positional embedding with learnable projection."""

  def __init__(self, input_channel, resolution):
    """Builds the soft position embedding layer.

    Args:
      input_channel: Size of input feature dimension.
      resolution: Tuple of integers specifying width and height of grid.
    """
    super(SoftPositionEmbed,self).__init__()
    self.ch_embedding = nn.Conv2d(4,input_channel,kernel_size=1,stride=1,bias=True)
    self.grid = build_grid(resolution) # embedding 初始化
    # print("shape of grid: ",self.grid.shape)

  def forward(self, x):
    # print(self.ch_embedding(self.grid).shape)
    # print(x.shape)
    return x + self.ch_embedding(self.grid)
  

def build_grid(resolution):
  ranges = [np.linspace(0., 1., num=res) for res in resolution] # 兩個array 分別介於0~1之間，元素分別為看有多少解析度
  grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
  grid = np.stack(grid, axis=-1)
  grid = np.reshape(grid, [resolution[0], resolution[1], -1])
  grid = np.expand_dims(grid, axis=0)
  grid = grid.astype(np.float32)
  grid_t = torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)) #4個channel軸?
  return grid_t.permute(0,3,1,2)