import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from .utils import _sce_loss, _get_logger
from tensorboardX import SummaryWriter
from functools import partial
from .dataset import SteamboatDataset
import os
from typing import Literal


class NonNegLinear(nn.Module):
    def __init__(self, d_in, d_out, bias) -> None:
        """Nonegative linear layer

        :param d_in: number of input features
        :param d_out: number of output features
        :param bias: umimplemented
        :raises NotImplementedError: when bias is True
        """
        super().__init__()
        self._weight = torch.nn.Parameter(torch.randn(d_out, d_in) / 10 - 2)
        self.elu = nn.ELU()
        if bias:
            raise NotImplementedError()

    @property
    def weight(self):
        """transform weight matrix to be non-negative

        :return: transformed weight matrix
        """
        return self.elu(self._weight) + 1

    def forward(self, x):
        return x @ self.weight.T
    
class TransposedNonNegLinear(nn.Module):
    def __init__(self, orig_layer):
        super().__init__()
        self.orig_layer = orig_layer

    @property
    def weight(self):
        return self.orig_layer.weight.T
    
    def forward(self, x):
        return x @ self.weight.T

class NormNonNegLinear(nn.Module):
    def __init__(self, d_in, d_out, bias) -> None:
        """_summary_

        :param d_in: number of input features
        :param d_out: number of output features
        :param bias: umimplemented
        :raises NotImplementedError: when bias is True
        """
        super().__init__()
        self._weight = torch.nn.Parameter(torch.randn(d_out, d_in) / 10 - 2)
        self.sigmoid = nn.Sigmoid()
        if bias:
            raise NotImplementedError()

    @property
    def weight(self):
        """transform weight matrix to be non-negative

        :return: transformed weight matrix
        """
        temp =  self.sigmoid(self._weight)
        return temp / temp.sum()

    def forward(self, x):
        return x @ self.weight.T

    
class NonNegBias(nn.Module):
    def __init__(self, d) -> None:
        """Non-negative bias layer (i.e., add a non-negative vector to the output)

        :param d: number of input/output features
        """
        super().__init__()
        self._bias = torch.nn.Parameter(torch.zeros(1, d))
        self.elu = nn.ELU()

    @property
    def bias(self):
        """Transform bias to be non-negative

        :return: non-negative bias
        """
        return self.elu(self._bias) + 1

    def forward(self, x):
        return x + self.bias


class ScaleSwtich(nn.Module):
    def __init__(self, n_heads, n_scales) -> None:
        """Switch scales for heads

        :param n_heads: number of heads
        :param n_scales: number of scales
        """
        super().__init__()
        self._switch = torch.nn.Parameter(torch.zeros(n_heads, n_scales))
        self._softmax = nn.Softmax(dim=-1)

    def forward(self):
        """Switch matrix

        :return: switch matrix
        """
        return self._softmax(self._switch)
    
    def l2_reg(self):
        return self._switch.square().mean()

class BilinearAttention(nn.Module):
    def __init__(self, d_in, n_heads, n_scales=2, d_out=None):
        """Bilinear attention layer

        :param d_in: number of input features
        :param d_factors: number of factors
        :param d_out: _description_, defaults to None (meaning d_out = d_in)
        """
        super(BilinearAttention, self).__init__()
        if d_out is None:
            d_out = d_in
        self.d_in = d_in
        self.n_heads = n_heads
        self.n_scales = n_scales

        # self.switch = ScaleSwtich(n_heads, n_scales=2)

        # A bias layer for the output to account for any "DC" component
        self.bias = NonNegBias(d_out)

        # The transforms are shared by all scales
        # n * g -> n * d
        self.q = NonNegLinear(d_in, n_heads, bias=False) # each row of the weight matrix is a metagene (x -> x @ w.T)
        self.k = NonNegLinear(d_in, n_heads, bias=False) # each row ...
        self.v = NonNegLinear(n_heads, d_out, bias=False) # each column ..
        # self.v = TransposedNonNegLinear(self.q)

        # remember some variables during forward
        # Note: with gradient; detach before use when gradient is not needed
        self.q_emb = None
        self.k_local_emb = None
        self.k_regional_embs = None

        self.cosine_similarity = nn.CosineSimilarity(dim=-2)

    def score_intrinsic(self, q_emb, k_emb):
        """Score intrinsic factors. No attention to other cells/environment.

        :param q_emb: query scores
        :return: ego scores
        """
        scores = q_emb * k_emb
        return scores

    def score_interactive(self, q_emb, k_emb, adj_list):
        """Score interactive factors. Attention to other cells/environment.

        :param q_emb: query scores
        :param k_emb: key scores
        :param adj_list: adjacency list
        :return: interactive scores for short or long range interaction
        """
        q = q_emb[adj_list[1, :], :] # n * g ---v-> kn * d
        k = k_emb[adj_list[0, :], :] # n * g ---u-> kn * d
        scores = q * k # nk * d
        nominal_k = scores.shape[0] // q_emb.shape[0]
        if adj_list.shape[0] == 3: # masked for unequal neighbors
            scores.masked_fill_((adj_list[2, :] == 0).reshape([-1, 1]), 0.)

        # reshape
        scores = scores.reshape([q_emb.shape[0], nominal_k, self.n_heads]) # n * k * d 
        scores = scores.transpose(-1, -2)

        # Normalize by the actual number of neighbors
        if adj_list.shape[0] == 3:
            actual_k = adj_list[2, :].reshape(q_emb.shape[0], nominal_k).sum(axis=1) # TODO: memorize this
            scores /= actual_k[:, None, None] 
        else:
            scores /= nominal_k

        return scores

    def flat_k_penalty(self, kind: Literal['entropy', 'cosine', 'variance']):
        """Scoring how homogeneous the k_emb is over all cells.
        The score is the highest when the k_emb is the same for all cells. 
        This will mean no difference between the local and ego attention.

        :return: entropy penalty
        """
        if kind == 'entropy':
            # Considering the scores of all cells, a higher entropy means more flat distribution
            probs = self.k_local_emb / self.k_local_emb.sum(dim=-2, keepdim=True)
            penalty = -(probs * torch.log(probs + 1e-9)).sum(dim=-2).mean()
        if kind == 'cosine':
            # More similar to an all-1 vector means more flat distribution
            penalty = self.cosine_similarity(self.k_local_emb, torch.ones_like(self.k_local_emb)).mean()
        if kind == 'variance':
            # LESS variance means more flat distribution
            probs = self.k_local_emb / self.k_local_emb.sum(dim=-2, keepdim=True)
            penalty = -probs.var(dim=-2).mean()
        return penalty

    def l2_reg(self):
        # No penalty on bias. Can't do this with weight_decay in optimizer
        return self.q.weight.square().sum(dim=-2).mean() + self.k.weight.square().sum(dim=-2).mean() + self.v.weight.square().sum(dim=-2).mean()

    def forward(self, adj_list, x, masked_x=None, regional_adj_lists=None, regional_xs=None, get_details=False):
        assert isinstance(regional_xs, list), "regional_xs should be a list of regional features."
        if regional_adj_lists is None:
            regional_adj_lists = []
        if regional_xs is None:
            regional_xs = []
        assert len(regional_adj_lists) == len(regional_xs)
        assert self.n_scales == len(regional_xs) + 2

        if masked_x is None:
            masked_x = x

        # Get embeddings for all cells and regions
        q_emb = self.q(masked_x) / (x.shape[1] ** .5)
        k_local_emb = self.k(x) / (x.shape[1] ** .5)
        k_regional_embs = [self.k(regional_x) / (x.shape[1] ** .5) for regional_x in regional_xs]

        # Get raw attention scores
        # scale_switch = self.switch() # h * s
        ego_score = self.score_intrinsic(q_emb, q_emb) # * scale_switch[:, 0].reshape([1, self.n_heads])
        local_score = self.score_interactive(q_emb, k_local_emb, adj_list) #  * scale_switch[:, 1].reshape([1, self.n_heads, 1]) # n * h * m
        regional_scores = [self.score_interactive(q_emb, k_regional_emb, regional_adj_list) 
                           for i, (k_regional_emb, regional_adj_list) in enumerate(zip(k_regional_embs, regional_adj_lists))]
        # regional_scores = [self.score_interactive(q_emb, k_regional_emb, adj_list) * scale_switch[:, i + 2].reshape([1, self.n_heads, 1]) for i, k_regional_emb in enumerate(k_regional_embs)]

        # Normalize attention scores
        sum_local_score = torch.sum(local_score, dim=-1)
        sum_regional_scores = [torch.sum(regional_score, dim=-1) for regional_score in regional_scores]
        sum_score = ego_score + sum_local_score + sum(sum_regional_scores) # n * h
        normalization_factor = sum_score.sum(axis=-1, keepdim=True) + 1e-9 # n * 1

        sum_attn = sum_score / normalization_factor
        res = self.bias(self.v(sum_attn))

        self.q_emb = q_emb
        self.k_local_emb = k_local_emb
        self.k_regional_embs = k_regional_embs

        if get_details:
            ego_attnp = ego_score / normalization_factor
            local_attnp = local_score / normalization_factor[:, :, None]
            regional_attnps = [regional_score / normalization_factor[:, :, None] for regional_score in regional_scores]

            ego_attnm = ego_attnp
            local_attnm = local_attnp.sum(axis=-1)
            regional_attnms = [regional_attnp.sum(axis=-1) for regional_attnp in regional_attnps]

            return res, {
                'embq': q_emb,
                'embk': (k_local_emb, k_regional_embs),
                'attnp': (ego_attnp, local_attnp, regional_attnps),
                'attnm': (ego_attnm, local_attnm, regional_attnms)}
        else:
            return res

    
class Steamboat(nn.Module):
    def __init__(self, features: list[str] | int, n_heads: int, n_scales: int = 2):
        """Steamboat model

        :param features: feature names (usuall `adata.var_names` or a column in `adata.var` for gene symbols)
        :param n_heads: number of heads
        :param n_scales: number of scales
        """
        super(Steamboat, self).__init__()

        if isinstance(features, list):
            self.features = features
        else:
            self.features = [f'feature_{i}' for i in range(features)]

        d_in = len(self.features)
        self.spatial_gather = BilinearAttention(d_in, n_heads, n_scales)

    def masking(self, x: torch.Tensor, entry_masking_rate: float, feature_masking_rate: float):
        """Masking the dataset

        :param x: input data
        :param mask_rate: masking rate
        :param masking_method: full matrix or feature-wise masking
        :return: masked data
        """
        out_x = x.clone()
        if entry_masking_rate > 0.:
            random_mask = torch.rand(x.shape, device=x.get_device()) < entry_masking_rate
            out_x.masked_fill_(random_mask, 0.)
        if feature_masking_rate > 0.:
            random_mask = torch.rand([1, x.shape[1]], device=x.get_device()) < feature_masking_rate
            out_x.masked_fill_(random_mask, 0.)
        return out_x

    def forward(self, adj_list, x, masked_x, regional_adj_lists, regional_xs, get_details=False):
        return self.spatial_gather(adj_list, x, masked_x, regional_adj_lists, regional_xs, get_details)

    def fit(self, dataset: SteamboatDataset, 
            entry_masking_rate: float = 0.0, feature_masking_rate: float = 0.0,
            device:str = 'cuda', 
            *, 
            flat_k_penalty: float = 0.0, flat_k_penalty_args=None, switch_l2_penalty: float = 0.0, weight_l2_penalty: float = 0.0,
            opt=None, opt_args=None, 
            loss_fun=None,
            max_epoch: int = 100, stop_eps: float = 1e-4, stop_tol: int = 10, 
            log_dir: str = 'log/', report_per: int = 10):
        """Create a PyTorch Dataset from a list of adata

        :param dataset: Dataset to be trained on
        :param entry_masking_rate: Rate of masking a random entries, default 0.0
        :param feature_masking_rate: Rate of masking a full feature (can overlap with entry masking), default 0.0
        :param device: Device to be used ("cpu" or "cuda")
        :param local_entropy_penalty: entropy penalty to make the local attention more diverse
        :param opt: Optimizer for fitting
        :param opt_args: Arguments for optimizer (e.g., {'lr': 0.01})
        :param loss_fun: Loss function: Default is MSE (`nn.MSELoss`). 
        You may use MAE `nn.L1Loss`, Huber 'nn.HuberLoss`, SmoothL1 `nn.SmoothL1Loss`, or a customized loss function.
        :param max_epoch: maximum number of epochs
        :param stop_eps: Stopping criterion: minimum change (see also `stop_tol`)
        :param stop_tol: Stopping criterion: number of epochs that don't meet `stop_eps` before stopping
        :param log_dir: Directory to save logs
        :param report_per: report per how many epoch. 0 to only report before termination. negative number to never report.

        :return: self
        """
        self.train()

        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        parameters = self.parameters()

        if flat_k_penalty_args is None:
            flat_k_penalty_args = {}

        if loss_fun is None:
            criterion = nn.MSELoss()
        else:
            criterion = loss_fun

        if opt_args is None:
            opt_args = {}
        if opt is None:
            optimizer = optim.Adam(parameters, **opt_args)
        else:
            optimizer = opt(parameters, **opt_args)

        os.makedirs(log_dir, exist_ok=True)
        logger = _get_logger('train', log_dir)
        # writer = SummaryWriter(logdir=log_dir)

        cnt = 0
        best_loss = np.inf
        for epoch in range(max_epoch):
            total_loss = 0.
            total_penalty = 0.
            for x, adj_list, regional_xs, regional_adj_lists in loader:
                # Send everything to required device
                adj_list = adj_list.squeeze(0).to(device)
                x = x.squeeze(0).to(device)
                regional_adj_lists = [regional_adj_list.squeeze(0).to(device) for regional_adj_list in regional_adj_lists]
                regional_xs = [regional_x.squeeze(0).to(device) for regional_x in regional_xs]

                masked_x = self.masking(x, entry_masking_rate, feature_masking_rate)

                x_recon = self.forward(adj_list, x, masked_x, regional_adj_lists, regional_xs, get_details=False)
                
                loss = criterion(x_recon, x)
                total_loss += loss.item()
                # loss = loss * x.shape[0] / 10000 # handle size differences among datasets; larger dataset has higher weight

                reg = 0.
                if flat_k_penalty > 0.:
                    reg += self.spatial_gather.flat_k_penalty(**flat_k_penalty_args) * flat_k_penalty
                    total_penalty += reg.item()
                # if switch_l2_penalty > 0.:
                #     reg += self.spatial_gather.switch.l2_reg() * switch_l2_penalty
                #     total_penalty += reg.item()
                if weight_l2_penalty > 0.:
                    reg += self.spatial_gather.l2_reg() * weight_l2_penalty
                    total_penalty += reg.item()

                optimizer.zero_grad()
                (loss + reg).backward()
                optimizer.step()

            avg_loss = total_loss / len(loader)
            avg_penalty = total_penalty / len(loader)

            if best_loss - (avg_loss + avg_penalty) < stop_eps:
                cnt += 1
            else:
                cnt = 0
            if report_per >= 0 and cnt >= stop_tol:
                logger.info(f"Epoch {epoch + 1}: train_loss {avg_loss:.5f}, reg {avg_penalty:.6f}")
                logger.info(f"Stopping criterion met.")
                break
            elif report_per > 0 and (epoch % report_per) == 0:
                logger.info(f"Epoch {epoch + 1}: train_loss {avg_loss:.5f}, reg {avg_penalty:.6f}")
            best_loss = min(best_loss, avg_loss + avg_penalty)

            # writer.add_scalar('Train_Loss', train_loss / len(loader), epoch)
            # writer.add_scalar('Learning_Rate', optimizer.state_dict()["param_groups"][0]["lr"], epoch)
            # scheduler.step()
        else:
            logger.info(f"Maximum iterations reached.")
        self.fit_loss = avg_loss
        self.eval()
        return self

    def transform(self, x, adj_matrix):
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.Tensor(x)
            if not isinstance(adj_matrix, torch.Tensor):
                adj_matrix = torch.Tensor(adj_matrix)
            
            return self(adj_matrix, x, get_details=True)


    def get_bias(self) -> np.array:
        b = self.spatial_gather.bias.bias.detach().cpu().numpy()
        return b.T

    def get_ego_transform(self) -> np.array:
        """Get gene attention matrix
        
        :param separate_q_k: If True, return Q and K. Otherwise, return Q.T @ K.

        :return: Gene attention vectors
        """
        # qk = self.spatial_gather.qk_ego.weight.detach().cpu().numpy()
        qk = self.spatial_gather.qk_ego.weight.detach().cpu().numpy()
        v = self.spatial_gather.v_ego.weight.detach().cpu().numpy()
        return qk, v.T

    def get_local_transform(self) -> np.array:
        """Get gene attention matrix
        
        :param separate_q_k: If True, return Q and K. Otherwise, return Q.T @ K.

        :return: Gene attention vectors
        """
        q = self.spatial_gather.q_local.weight.detach().cpu().numpy()
        k = self.spatial_gather.k_local.weight.detach().cpu().numpy()
        v = self.spatial_gather.v_local.weight.detach().cpu().numpy()
        return q, k, v.T
        
    def get_global_transform(self) -> np.array:
        """Get gene attention matrix
        
        :param separate_q_k: If True, return Q and K. Otherwise, return Q.T @ K.

        :return: Gene attention matrix
        """
        q = self.spatial_gather.q_global.weight.detach().cpu().numpy()
        k = self.spatial_gather.k_global.weight.detach().cpu().numpy()
        v = self.spatial_gather.v_global.weight.detach().cpu().numpy()
        return q, k, v.T
       
    def score_cells(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        res = {}
        qk_ego, v_ego = self.get_ego_transform()
        for i in range(qk_ego.shape[0]):
            res[f'u_ego_{i}'] = x @ qk_ego[i, :]
        q_local, k_local, v_local = self.get_local_transform()
        for i in range(q_local.shape[0]):
            res[f'q_local_{i}'] = x @ q_local[i, :]
            res[f'k_local_{i}'] = x @ k_local[i, :]
        if self.spatial_gather.d_global > 0:
            q_global, k_global, v_global = self.get_global_transform()
        for i in range(q_global.shape[0]):
            res[f'q_global_{i}'] = x @ q_global[i, :]
        return res

    def get_top_features(self, top_k=5):
        res = {}
        features = np.array(self.features)
        qk_ego, v_ego = self.get_ego_transform()
        for i in range(qk_ego.shape[0]):
            res[f'U_ego_{i}'] = features[np.argsort(-qk_ego[i, :])[:top_k]].tolist()
            # res[f'V_ego_{i}'] = features[np.argsort(-v_ego[i, :])[:top_k]].tolist()
        q_local, k_local, v_local = self.get_local_transform()
        for i in range(q_local.shape[0]):
            res[f'Q_local_{i}'] = features[np.argsort(-q_local[i, :])[:top_k]].tolist()
            res[f'K_local_{i}'] = features[np.argsort(-k_local[i, :])[:top_k]].tolist()
            res[f'V_local_{i}'] = features[np.argsort(-v_local[i, :])[:top_k]].tolist()
        if self.spatial_gather.d_global > 0:
            q_global, k_global, v_global = self.get_global_transform()
            for i in range(q_global.shape[0]):
                res[f'Q_global_{i}'] = features[np.argsort(-q_global[i, :])[:top_k]].tolist()
                res[f'K_global_{i}'] = features[np.argsort(-k_global[i, :])[:top_k]].tolist()
                res[f'V_global_{i}'] = features[np.argsort(-v_global[i, :])[:top_k]].tolist()
        return res
    
    def score_local(self, x, adj_matrix):
        with torch.no_grad():
            return self.spatial_gather.score_local(x, adj_matrix).cpu().numpy()
    
    def score_global(self, x, x_bar=None):
        with torch.no_grad():
            return self.spatial_gather.score_global(x, x_bar=x_bar).cpu().numpy()