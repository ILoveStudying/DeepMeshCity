import torch
import torch.nn as nn


class CGL(nn.Module):
    def __init__(self, in_channel, num_hidden, filter_size, stride, is_norm):
        super(CGL, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self.forget_bias = -1.0

        if is_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.GroupNorm(num_hidden // 2, 4 * num_hidden)
            )
            self.conv_c = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.GroupNorm(num_hidden // 2, num_hidden * 2)
            )

            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.GroupNorm(num_hidden // 2, num_hidden * 2)
            )
            self.conv_o_c = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0),
                nn.GroupNorm(num_hidden // 2, num_hidden)
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            )
            self.conv_c = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding),
            )
            self.conv_o_c = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            )

        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

    def forward(self, x_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        m_concat = self.conv_m(m_t)
        c_concat = self.conv_c(c_t)

        x_p, f_x, x_m_p, f_m_x = torch.split(x_concat, self.num_hidden, dim=1)
        f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)
        f_c, g_c = torch.split(c_concat, self.num_hidden, dim=1)

        f_t = torch.sigmoid(f_x + f_c + self.forget_bias)
        x_p_t = torch.tanh(x_p + g_c)

        c = f_t * c_t + (1 - f_t) * x_p_t

        f_t_prime = torch.sigmoid(f_m_x + f_m + self.forget_bias)
        g_t_prime = torch.tanh(x_m_p + g_m)

        m = f_t_prime * m_t + (1 - f_t_prime) * g_t_prime

        mem = torch.cat((c, m), 1)

        o_t = torch.sigmoid(self.conv_o_c(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c, m


class self_attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_szie=1):
        super(self_attention, self).__init__()
        self.layer_q = nn.Conv2d(input_dim, hidden_dim, kernel_szie)
        self.layer_k = nn.Conv2d(input_dim, hidden_dim, kernel_szie)
        self.layer_v = nn.Conv2d(input_dim, hidden_dim, kernel_szie)
        self.layer_f = nn.Conv2d(hidden_dim, input_dim, kernel_szie)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(self, q, k, v):
        batch_size, channel, H, W = q.shape
        h = q  # short connection

        q = self.layer_q(q)
        k = self.layer_k(k)
        v = self.layer_v(v)

        q = q.view(batch_size, self.hidden_dim, H * W)
        k = k.view(batch_size, self.hidden_dim, H * W)
        v = v.view(batch_size, self.hidden_dim, H * W)

        attn = torch.matmul(q.transpose(1, 2), k)
        # print('e shape is', e.shape)
        attention = torch.softmax(attn, dim=-1)  # attention
        z = torch.matmul(attention, v.permute(0, 2, 1))
        z = z.view(batch_size, self.hidden_dim, H, W)
        out = self.layer_f(z) + h

        return out, attention
