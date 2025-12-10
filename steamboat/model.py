import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from .utils import _get_logger
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
        self._weight = torch.nn.Parameter(torch.randn(d_out, d_in) - 3)
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
    
class NonNegScale(nn.Module):
    def __init__(self, d) -> None:
        """Non-negative bias layer (i.e., add a non-negative vector to the output)

        :param d: number of input/output features
        """
        super().__init__()
        self._scale = torch.nn.Parameter(torch.zeros(1, d))
        self.elu = nn.ELU()

    @property
    def scale(self):
        """Transform bias to be non-negative

        :return: non-negative bias
        """
        return self.elu(self._scale) + 1

    def forward(self, x):
        return x * self.scale


class BilinearAttention(nn.Module):
    def __init__(self, d_in: int, n_heads: int, n_scales: int = 2, d_out: int = None):
        """Bilinear attention layer

        :param d_in: number of input features
        :param n_heads: number of heads
        :param n_scales: number of scales (default 2, i.e., ego and local; 3 will add global)
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
        # self.k = NonNegLinear(d_in, n_heads, bias=False) # each row ...
        
        self.k_local = NonNegLinear(d_in, n_heads, bias=False)
        self.k_regionals = nn.ModuleList(NonNegLinear(d_in, n_heads, bias=False)
                                         for i in range(n_scales - 2))
            
        self.w_ego = NonNegScale(n_heads)

        self.tanh = nn.Tanh() # for clamping of the values

        self.v = NonNegLinear(n_heads, d_out, bias=False) # each column ..
        # self.v = TransposedNonNegLinear(self.q)

        # remember some variables during forward
        # Note: with gradient; detach before use when gradient is not needed
        self.q_emb = None
        self.k_local_emb = None
        self.k_regional_embs = None

        # For debugging
        # self.attn_shortcut = torch.nn.Sequential(torch.nn.Linear(d_in, n_heads), 
        #                                          torch.nn.Tanh(), 
        #                                          torch.nn.Linear(n_heads, n_heads),
        #                                          torch.nn.Tanh())
        # self.v_shortcut = torch.nn.Linear(n_heads, d_out)

        self.cosine_similarity = nn.CosineSimilarity(dim=-2)

    def score_intrinsic(self, q_emb, k_emb, activation=None):
        """Score intrinsic factors. No attention to other cells/environment.

        :param q_emb: query scores
        :param k_emb: key scores
        :param activation: activation function
        :return: ego scores
        """
        scores = q_emb * k_emb
        if activation is not None:
            scores = activation(scores)
        return scores

    def score_interactive(self, q_emb, k_emb, adj_list, activation=None):
        """Score interactive factors. Attention to other cells/environment.

        :param q_emb: query scores
        :param k_emb: key scores
        :param adj_list: adjacency list
        :return: interactive scores for short or long range interaction
        """
        q = q_emb[adj_list[1, :], :] # n * g ---v-> kn * d
        k = k_emb[adj_list[0, :], :] # n * g ---u-> kn * d
        scores = q * k # nk * d
        if activation is not None:
            scores = activation(scores)
        nominal_k = scores.shape[0] // q_emb.shape[0]
        if adj_list.shape[0] == 3: # masked for unequal neighbors
            scores.masked_fill_((adj_list[2, :] == 0).reshape([-1, 1]), 0.)

        # reshape
        scores = scores.reshape([q_emb.shape[0], nominal_k, self.n_heads]) # n * k * d 
        scores = scores.transpose(-1, -2)

        # Normalize by the actual number of neighbors
        if adj_list.shape[0] == 3:
            actual_k = adj_list[2, :].reshape(q_emb.shape[0], nominal_k).sum(axis=1) # TODO: memorize this
            scores = scores / (actual_k[:, None, None] + 1e-6)
        else:
            scores = scores / nominal_k

        return scores

    def forward(self, adj_list, x, masked_x=None, regional_adj_lists=None, regional_xs=None, get_details=False,
                explained_variance_mask=None):
        """Forward pass

        :param adj_list: adjacency list for spatial graph
        :param x: input data
        :param masked_x: masked input data, defaults to None (i.e, using x)
        :param regional_adj_lists: list of adjacency list for bipartite graph of cells - regions, defaults to None
        :param regional_xs: list of mean expression of regions, defaults to None
        :param get_details: whether to return details, defaults to False
        :return: reconstructed gene expression
        """
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
        q_emb = self.q(masked_x) / x.shape[1]
        k_local_emb = self.k_local(x) / x.shape[1]
        k_regional_embs = [self.k_regionals[i](regional_x) / x.shape[1] 
                           for i, regional_x in enumerate(regional_xs)]

        # Get raw attention scores
        # scale_switch = self.switch() # h * s
        ego_score = self.w_ego(self.score_intrinsic(q_emb, q_emb)) # * scale_switch[:, 0].reshape([1, self.n_heads])
        local_score = (self.score_interactive(q_emb, k_local_emb, adj_list)) #  * scale_switch[:, 1].reshape([1, self.n_heads, 1]) # n * h * m
        regional_scores = [(self.score_interactive(q_emb, k_regional_emb, regional_adj_list))
                           for i, (k_regional_emb, regional_adj_list) in enumerate(zip(k_regional_embs, regional_adj_lists))]
        # regional_scores = [self.score_interactive(q_emb, k_regional_emb, adj_list) * scale_switch[:, i + 2].reshape([1, self.n_heads, 1]) for i, k_regional_emb in enumerate(k_regional_embs)]

        # Normalize attention scores
        sum_local_score = torch.sum(local_score, dim=-1)
        sum_regional_scores = [torch.sum(regional_score, dim=-1) for regional_score in regional_scores]
        sum_score = ego_score + sum_local_score + sum(sum_regional_scores) # n * h
        normalization_factor = sum_score.sum(axis=-1, keepdim=True) + 1e-9 # n * 1

        if explained_variance_mask is None:
            sum_attn = sum_score / normalization_factor
        else:
            if explained_variance_mask == 'ego':
                sum_attn = ego_score / normalization_factor
            elif explained_variance_mask == 'local':
                sum_attn = sum_local_score / normalization_factor
            elif explained_variance_mask == 'ego+local':
                sum_attn = (ego_score + sum_local_score) / normalization_factor
            elif explained_variance_mask == 'global':
                sum_attn = sum(sum_regional_scores) / normalization_factor
            elif explained_variance_mask == 'no ego':
                sum_attn = (sum_local_score + sum(sum_regional_scores)) / normalization_factor
            elif explained_variance_mask == 'no local':
                sum_attn = (ego_score + sum(sum_regional_scores)) / normalization_factor
            elif explained_variance_mask == 'no global':
                sum_attn = (ego_score + sum_local_score) / normalization_factor
            elif explained_variance_mask == 'full':
                sum_attn = sum_score / normalization_factor
            else:
                raise ValueError(f"Unknown explained_variance_mask: {explained_variance_mask}")
        
        # Reconstruct
        # res = self.bias(self.v(sum_attn))
        # sum_attn = self.attn_shortcut(x)
        res = self.v(sum_attn)
        # res = self.v_shortcut(sum_attn)

        # Remember variables for later inspection
        self.q_emb = q_emb
        self.k_local_emb = k_local_emb
        self.k_regional_embs = k_regional_embs

        if get_details:
            ego_attnp = ego_score / normalization_factor
            local_attnp = local_score / normalization_factor[:, :, None]
            regional_attnps = [regional_score / normalization_factor[:, :, None] for regional_score in regional_scores]
            # regional_attnps = [regional_score for regional_score in regional_scores]

            ego_attnm = ego_attnp
            local_attnm = local_attnp.sum(axis=-1)
            regional_attnms = [regional_attnp.sum(axis=-1) for regional_attnp in regional_attnps]

            return res, {
                'attn': sum_attn,
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
        :param n_scales: number of scales (default 2, i.e., ego and local; 3 will add global)
        """
        super(Steamboat, self).__init__()

        if isinstance(features, list):
            self.features = features
        else:
            self.features = [f'feature_{i}' for i in range(features)]

        d_in = len(self.features)
        self.spatial_gather = BilinearAttention(d_in, n_heads, n_scales)

    def masking(self, x: torch.Tensor, xs, entry_masking_rate: float, feature_masking_rate: float):
        """Masking the dataset

        :param x: input data
        :param mask_rate: masking rate
        :param masking_method: full matrix or feature-wise masking
        :return: masked data
        """
        out_x = x.clone()
        out_xs = []
        if entry_masking_rate > 0.:
            random_mask = torch.rand(x.shape, device=x.device) < entry_masking_rate
            out_x.masked_fill_(random_mask, 0.)
        if feature_masking_rate > 0.:
            random_mask = torch.rand([1, x.shape[1]], device=x.device) < feature_masking_rate
            out_x.masked_fill_(random_mask, 0.)
            for x in xs:
                x = x.clone()
                x.masked_fill_(random_mask, 0.)
                out_xs.append(x)
        else:
            for x in xs:
                x = x.clone()
                out_xs.append(x)
        return out_x, out_xs

    def forward(self, adj_list, x, masked_x, regional_adj_lists, regional_xs, get_details=False, explained_variance_mask=None):
        return self.spatial_gather(adj_list, x, masked_x, regional_adj_lists, regional_xs, get_details, explained_variance_mask)

    def fit(self, dataset: SteamboatDataset, 
            entry_masking_rate: float = 0.1, feature_masking_rate: float = 0.1,
            device:str = 'cuda', 
            *, 
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

        if loss_fun is None:
            criterion = nn.MSELoss(reduction='sum')
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
        n_cells = 0
        n_genes = 0
        for x, adj_list, regional_xs, regional_adj_lists in loader:
            n_cells += x.shape[0]
            n_genes = x.shape[1]
        for epoch in range(max_epoch):
            avg_loss = 0.
            optimizer.zero_grad()
            for x, adj_list, regional_xs, regional_adj_lists in loader:
                # Send everything to required device
                adj_list = adj_list.squeeze(0).to(device)
                x = x.squeeze(0).to(device)
                regional_adj_lists = [regional_adj_list.squeeze(0).to(device) for regional_adj_list in regional_adj_lists]
                regional_xs = [regional_x.squeeze(0).to(device) for regional_x in regional_xs]

                masked_x, masked_xs = self.masking(x, regional_xs, entry_masking_rate, feature_masking_rate)

                x_recon = self.forward(adj_list, masked_x, masked_x, 
                                       regional_adj_lists, masked_xs, get_details=False)
                
                # loss = criterion(x_recon, x)
                # total_loss = loss + total_loss
                loss = criterion(x_recon, x) / n_cells / n_genes
                avg_loss += loss.item()
                loss.backward()

                # n_cells += x.shape[0]
                # loss = loss * x.shape[0] / 10000 # handle size differences among datasets; larger dataset has higher weight

                # reg = 0.
                # if flat_k_penalty > 0.:
                #     reg += self.spatial_gather.flat_k_penalty(**flat_k_penalty_args) * flat_k_penalty
                #     total_penalty += reg.item()
                # if switch_l2_penalty > 0.:
                #     reg += self.spatial_gather.switch.l2_reg() * switch_l2_penalty
                #     total_penalty += reg.item()
                # if weight_l2_penalty > 0.:
                #     reg += self.spatial_gather.l2_reg() * weight_l2_penalty
                #     total_penalty += reg.item()
            optimizer.step()

            if best_loss - avg_loss < stop_eps:
                cnt += 1
            else:
                cnt = 0
            if report_per >= 0 and cnt >= stop_tol:
                logger.info(f"Epoch {epoch + 1}: train_loss {avg_loss:.5f}")
                logger.info(f"Stopping criterion met.")
                break
            elif report_per > 0 and (epoch % report_per) == 0:
                logger.info(f"Epoch {epoch + 1}: train_loss {avg_loss:.5f}")
            best_loss = min(best_loss, avg_loss)

            self.training_loss_ = avg_loss

            # writer.add_scalar('Train_Loss', train_loss / len(loader), epoch)
            # writer.add_scalar('Learning_Rate', optimizer.state_dict()["param_groups"][0]["lr"], epoch)
            # scheduler.step()
        else:
            logger.info(f"Maximum iterations reached at epoch {epoch + 1}. train_loss {avg_loss:.5f}")
            
        self.eval()
        return self

    def transform(self, x, adj_matrix, get_details=True, explained_variance_mask=None):
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.Tensor(x)
            if not isinstance(adj_matrix, torch.Tensor):
                adj_matrix = torch.Tensor(adj_matrix)
            
            return self(adj_matrix, x, get_details=True, explained_variance_mask=explained_variance_mask)


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