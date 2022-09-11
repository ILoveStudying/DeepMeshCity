from Model.layers.modules import *
from einops import rearrange


class DeepMeshCity(nn.Module):
    def __init__(self, configs):
        super(DeepMeshCity, self).__init__()
        # hyperparrams
        self.meta_in = configs.meta_dim
        self.map_width = configs.map_width
        self.patch_size = configs.patch_size
        self.resize_ratio = self.map_width // self.patch_size
        self.frame_channel = configs.map_channel
        self.input_length = configs.input_length
        self.device = configs.device

        # Parameter for External Meta data
        self.meta_out = self.input_length * self.map_width * self.map_width * self.frame_channel
        self.is_metadate = configs.is_metadate

        # Parameter for Model
        self.Meta_frame_channel = self.frame_channel * 2 if self.is_metadate else self.frame_channel
        self.num_layers = configs.num_layers
        self.num_hidden = configs.num_hidden
        self.filter_size = configs.filter_size
        self.stride = configs.stride
        self.layer_norm = configs.layer_norm

        # Early Fusion
        if self.is_metadate:
            self.meta_learn = nn.Sequential(nn.Linear(self.meta_in, 10),
                                            nn.ReLU(),
                                            nn.Linear(10, self.meta_out),
                                            nn.ReLU())

        # Stacked SA-CGL Blocks
        SACGL_blocks = []
        for i in range(self.num_layers):
            in_channel = self.Meta_frame_channel if i == 0 else self.num_hidden[i - 1]
            attn_channel = in_channel * self.resize_ratio * self.resize_ratio
            SACGL_blocks.append(nn.ModuleList([
                self_attention(attn_channel, self.num_hidden[i]),
                CGL(in_channel, self.num_hidden[i], self.filter_size, self.stride, self.layer_norm)
            ]))

        self.SACGL_blocks = nn.ModuleList(SACGL_blocks)

        # Output Module
        self.Output_module = nn.Sequential(
            nn.Conv2d(self.num_hidden[self.num_layers - 1], self.num_hidden[-1],
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.num_hidden[-1], self.frame_channel,
                      kernel_size=1, stride=1, padding=0, bias=False)
        )

    def Attention_module(self, net, i):
        net = rearrange(net, 'b c (p1 h) (p2 w) -> b (h w c) p1 p2', p1=self.patch_size, p2=self.patch_size)
        net, _ = self.SACGL_blocks[i][0](net, net, net)
        net = rearrange(net, 'b (h w c) p1 p2 -> b c (p1 h) (p2 w)', h=self.resize_ratio, w=self.resize_ratio)

        return net

    def forward(self, xc, xp, xt, yd):
        B, _, C, H, W = xc.shape

        if self.is_metadate:
            yd = rearrange(self.meta_learn(yd), 'b (f c h w) -> b f c h w', c=self.frame_channel,
                           h=self.map_width, w=self.map_width)
            xp = torch.cat((xp, yd), dim=2)
            xt = torch.cat((xt, yd), dim=2)
            xc = torch.cat((xc, yd), dim=2)

        # in the process, the time dimension is along the time [0] is the farest
        frames = torch.cat((xt, xp, xc), dim=1)

        batch = frames.shape[0]
        height = frames.shape[-2]
        width = frames.shape[-1]

        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.device)

        for t in range(self.input_length * 3):
            net = frames[:, t]

            net = self.Attention_module(net, 0)

            h_t[0], c_t[0], memory = self.SACGL_blocks[0][1](net, c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i - 1] = self.Attention_module(h_t[i - 1], i)
                h_t[i], c_t[i], memory = self.SACGL_blocks[i][1](h_t[i - 1], c_t[i], memory)

        x_gen = self.Output_module(h_t[self.num_layers - 1])

        return x_gen
