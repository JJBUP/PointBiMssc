import torch
import torch.nn as nn
import einops
from libs import pointops

from ..builder import MODELS
from .utils import LayerNorm1d


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16, p_inter_dim=6, spa_preprocess=True, spa_add_fea=True):
        super().__init__()
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.spa_add_fea = spa_add_fea
        self.spa_preprocess = spa_preprocess
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, out_planes)
        self.linear_k = nn.Linear(in_planes, out_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p_q = nn.Linear(out_planes, out_planes)
        self.linear_p_k = nn.Linear(out_planes, out_planes)
        self.linear_p_v = nn.Linear(out_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(6, p_inter_dim),
                                      LayerNorm1d(p_inter_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(p_inter_dim, out_planes)
                                      )
        if self.spa_preprocess:
            self.linear_x = nn.Linear(in_planes, out_planes)
            self.linear_p2 = nn.Sequential(LayerNorm1d(out_planes),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(out_planes, out_planes),
                                        LayerNorm1d(out_planes),
                                        nn.ReLU(inplace=True),
                                        )
        self.linear_p_w = nn.Sequential(LayerNorm1d(out_planes),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(out_planes, out_planes // share_planes),
                                        LayerNorm1d(out_planes // share_planes),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.linear_f_w = nn.Sequential(LayerNorm1d(out_planes),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(out_planes, out_planes // share_planes),
                                        LayerNorm1d(out_planes // share_planes),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
        self.br_f = nn.Sequential(nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
        self.br_p = nn.Sequential(nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
        self.linear_fp = nn.Linear(2 * out_planes, out_planes)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        q, k, v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        k, idx = pointops.knn_query_and_group(k, p, o, new_xyz=p, new_offset=o,
                                              nsample=self.nsample, with_xyz=True)
        v, idx = pointops.knn_query_and_group(v, p, o, new_xyz=p, new_offset=o,
                                              idx=idx, nsample=self.nsample, with_xyz=False)
        p_r, k = k[:, :, 0:3], k[:, :, 3:]
        p_r = self.linear_p(torch.cat([p_r, self.xyz2sphere(p_r)], dim=-1))  # N G D  [best]
        w = k - q.unsqueeze(1) + p_r  # N G D
        w = self.softmax(self.linear_f_w(w))  # (n, nsample, c)
        feat = torch.einsum("n t s i, n t i -> n s i", einops.rearrange(v + p_r, "n ns (s i) -> n ns s i", s=self.share_planes), w)
        feat = self.br_f(einops.rearrange(feat, "n s i -> n (s i)"))
        
        if self.spa_preprocess:
            x = self.linear_x(x)
            p_r = self.linear_p2(p_r)
        
        q, k, v = self.linear_p_q(p_r).mean(dim=-1, keepdim=True), self.linear_p_k(p_r), self.linear_p_v(p_r)
        x, idx = pointops.knn_query_and_group(x, p, o, new_xyz=p, new_offset=o, idx=idx, nsample=self.nsample,
                                              with_xyz=False)
        if self.spa_add_fea :
            w = k - q + x
        else:
            w = k - q
        w = self.softmax(self.linear_p_w(w))
        post = torch.einsum("n t s i, n t i -> n s i", einops.rearrange(v + x, "n ns (s i) -> n ns s i", s=self.share_planes), w)
        post = self.br_p(einops.rearrange(post, "n s i -> n (s i)"))
        return self.linear_fp(torch.cat([feat, post], dim=1))

    def xyz2sphere(self, xyz, normalize=True):
        """
        Convert XYZ to Spherical Coordinate
        """
        rho = torch.sqrt(torch.sum(torch.pow(xyz, 2), dim=-1, keepdim=True))
        rho = torch.clamp(rho, min=0)
        theta = torch.acos(xyz[..., 2, None] / rho)
        phi = torch.atan2(xyz[..., 1, None], xyz[..., 0, None])
        # check nan
        idx = rho == 0
        theta[idx] = 0
        if normalize:
            theta = theta / torch.pi
            phi = phi / (2 * torch.pi) + .5
        out = torch.cat([rho, theta, phi], dim=-1)
        return out  # [N, 3] / [N, G, 3]


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, mssc_scale, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample, self.mssc_scale = stride, nsample, mssc_scale
        self.out_planes = out_planes
        self.nsample = 10
        if stride != 1:
            self.linear1 = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.linear2 = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.linear3 = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.linear = nn.Linear(3 * out_planes, out_planes)
            self.pool1 = nn.MaxPool1d(self.mssc_scale[0] * self.nsample)
            self.pool2 = nn.MaxPool1d(self.mssc_scale[1] * self.nsample)
            self.pool3 = nn.MaxPool1d(self.mssc_scale[2] * self.nsample)
            self.bn1 = nn.BatchNorm1d(out_planes)
            self.bn2 = nn.BatchNorm1d(out_planes)
            self.bn3 = nn.BatchNorm1d(out_planes)
            self.bn = nn.BatchNorm1d(out_planes)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
            self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.farthest_point_sampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)

            x, _ = pointops.knn_query_and_group(x, p, offset=o, new_xyz=n_p, new_offset=n_o, nsample=self.mssc_scale[2] * self.nsample, with_xyz=True)
            x = torch.concat([self.pool1(
                self.relu(self.bn1(self.linear1(x[:, :self.mssc_scale[0] * self.nsample]).transpose(1, 2).contiguous()))).squeeze(-1),
                              self.pool2(self.relu(self.bn2(
                                  self.linear2(x[:, :self.mssc_scale[1] * self.nsample]).transpose(1, 2).contiguous()))).squeeze(-1),
                              self.pool3(self.relu(self.bn3(self.linear3(x).transpose(1, 2).contiguous()))).squeeze(
                                  -1)], dim=-1)  # (m,3*c)
            x = self.relu(self.bn(self.linear(x)))  # (m, c, nsample)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]

class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2 * in_planes, in_planes),
                                         nn.BatchNorm1d(in_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes),
                                         nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes),
                                         nn.BatchNorm1d(out_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes),
                                         nn.BatchNorm1d(out_planes),
                                         nn.ReLU(inplace=True))

    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x

class Bottleneck(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, share_planes=8, nsample=16, p_inter_dim=6, spa_preprocess=True, spa_add_fea=True):
        super(Bottleneck, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer = PointTransformerLayer(planes, planes, share_planes, nsample, p_inter_dim, spa_preprocess, spa_add_fea)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]

@MODELS.register_module("PointBiMssc")
class PointBiMssc(nn.Module):
    def __init__(self, in_channels=None, num_classes=None, block=Bottleneck,
                blocks=(1, 2, 3, 5, 2), Mssc_scale=(1,3,8),
                p_inter_dim=6, spa_preprocess=True, spa_add_fea=True
                ):
        super().__init__()
        self.in_channels = in_channels
        self.in_planes, planes = in_channels, [32, 64, 128, 256, 512]
        share_planes = 8
        stride, nsample = [1, 4, 4, 4, 4], [10, 10, 10, 10, 10]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], Mssc_scale, share_planes, stride=stride[0], nsample=nsample[0],
                                   p_inter_dim = p_inter_dim, spa_preprocess = spa_preprocess, spa_add_fea = spa_add_fea)  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], Mssc_scale, share_planes, stride=stride[1], nsample=nsample[1],
                                   p_inter_dim = p_inter_dim, spa_preprocess = spa_preprocess, spa_add_fea = spa_add_fea)  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], Mssc_scale, share_planes, stride=stride[2], nsample=nsample[2],
                                   p_inter_dim = p_inter_dim, spa_preprocess = spa_preprocess, spa_add_fea = spa_add_fea)  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], Mssc_scale, share_planes, stride=stride[3], nsample=nsample[3],
                                   p_inter_dim = p_inter_dim, spa_preprocess = spa_preprocess, spa_add_fea = spa_add_fea)  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], Mssc_scale, share_planes, stride=stride[4], nsample=nsample[4],
                                   p_inter_dim = p_inter_dim, spa_preprocess = spa_preprocess, spa_add_fea = spa_add_fea)  # N/256
        self.dec5 = self._make_dec(block, planes[4], 1, share_planes, nsample=nsample[4], is_head=True,
                                   p_inter_dim = p_inter_dim, spa_preprocess = spa_preprocess, spa_add_fea = spa_add_fea)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 1, share_planes, nsample=nsample[3],
                                   p_inter_dim = p_inter_dim, spa_preprocess = spa_preprocess, spa_add_fea = spa_add_fea)  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 1, share_planes, nsample=nsample[2],
                                   p_inter_dim = p_inter_dim, spa_preprocess = spa_preprocess, spa_add_fea = spa_add_fea)  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 1, share_planes, nsample=nsample[1],
                                   p_inter_dim = p_inter_dim, spa_preprocess = spa_preprocess, spa_add_fea = spa_add_fea)  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 1, share_planes, nsample=nsample[0],
                                   p_inter_dim = p_inter_dim, spa_preprocess = spa_preprocess, spa_add_fea = spa_add_fea)  # fusion p2 and p1
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]),
                                 nn.BatchNorm1d(planes[0]),
                                 nn.ReLU(inplace=True), nn.Linear(planes[0], num_classes))

    def _make_enc(self, block, planes, blocks, Mssc_scale ,share_planes=8, stride=1, nsample=16,
                  p_inter_dim=6, spa_preprocess=True, spa_add_fea=True):
        layers = [TransitionDown(self.in_planes, planes * block.expansion, Mssc_scale, stride, nsample)]
        self.in_planes = planes * block.expansion
        for _ in range(blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample, p_inter_dim, spa_preprocess, spa_add_fea))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False,
                  p_inter_dim=6, spa_preprocess=True, spa_add_fea=True):
        layers = [TransitionUp(self.in_planes, None if is_head else planes * block.expansion)]
        self.in_planes = planes * block.expansion
        for _ in range(blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample, p_inter_dim, spa_preprocess, spa_add_fea))
        return nn.Sequential(*layers)

    def forward(self, input_dict):
        p0 = input_dict["coord"]
        x0 = input_dict["feat"]
        o0 = input_dict["offset"].int()
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        x = self.cls(x1)
        return x
