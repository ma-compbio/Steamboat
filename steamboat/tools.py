import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .model import Steamboat
# from .integrated_model # import IntegratedSteamboat
from typing import Literal
from torch import nn
import scanpy as sc
import numpy as np
import torch
from .dataset import SteamboatDataset
import scipy as sp
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import squidpy as sq

palettes = {
    'ncr10': ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', 
              '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85'],
    'npg3': ['#952522', '#0a4b84', '#98752b']
}

def rank(x, axis=1):
    return np.argsort(np.argsort(x, axis=axis), axis=axis)

def plot_transforms(model: Steamboat, top: int = 3, reorder: bool = False, 
                    figsize: str | tuple[float, float] = 'auto', 
                    qkv_colors: list[str] = palettes['npg3'],
                    vmin: float = 0., vmax: float = 1.):
    """Plot all metagenes

    :param model: Steamboat model
    :param top: Number of top genes per metagene to plot, defaults to 3
    :param reorder: Reorder the genes by metagene, or keep the orginal ordering, defaults to False
    :param figsize: Size of the figure, defaults to 'auto'
    :param qkv_colors: Colors for the bar plot showing the magnitude of each metagene before normalization, defaults to palettes['npg3']
    :param vmin: minimum value in the color bar, defaults to 0.
    :param vmax: maximum value in the color bar, defaults to 1.
    """
    assert len(qkv_colors) == 3, f"Expect a color palette with at 3 colors, get {len(qkv_colors)}."
    d_ego : int = model.spatial_gather.d_ego
    d_loc : int = model.spatial_gather.d_local
    d_glb : int = model.spatial_gather.d_global
    d : int = d_ego + d_loc + d_glb

    qk_ego, v_ego = model.get_ego_transform()
    q_local, k_local, v_local = model.get_local_transform()
    q_global, k_global, v_global = model.get_global_transform()

    if top > 0:
        if reorder:
            rank_v_ego = np.argsort(-v_ego, axis=1)[:, :top]
            rank_q_local = np.argsort(-np.abs(q_local), axis=1)[:, :top]
            rank_k_local = np.argsort(-k_local, axis=1)[:, :top]
            rank_v_local = np.argsort(-v_local, axis=1)[:, :top]
            rank_q_global = np.argsort(-np.abs(q_global), axis=1)[:, :top]
            rank_k_global = np.argsort(-k_global, axis=1)[:, :top]
            rank_v_global = np.argsort(-v_global, axis=1)[:, :top]
            feature_mask = {}
            for i in rank_v_ego:
                for j in i:
                    feature_mask[j] = None
            for i in range(d_loc):
                for j in rank_k_local[i, :]:
                    feature_mask[j] = None
                for j in rank_q_local[i, :]:
                    feature_mask[j] = None
                for j in rank_v_local[i, :]:
                    feature_mask[j] = None
            for i in range(d_glb):
                for j in rank_k_global[i, :]:
                    feature_mask[j] = None
                for j in rank_q_global[i, :]:
                    feature_mask[j] = None
                for j in rank_v_global[i, :]:
                    feature_mask[j] = None
            feature_mask = list(feature_mask.keys())
        else:
            rank_v_ego = rank(v_ego)
            rank_q_local = rank(np.abs(q_local))
            rank_k_local = rank(k_local)
            rank_v_local = rank(v_local)
            rank_q_global = rank(np.abs(q_global))
            rank_k_global = rank(k_global)
            rank_v_global = rank(v_global)
            max_rank = np.max(np.vstack([rank_v_ego, 
                                        rank_q_local, 
                                        rank_k_local, 
                                        rank_v_local, 
                                        rank_q_global, 
                                        rank_k_global, 
                                        rank_v_global]), axis=0)
            feature_mask = (max_rank > (max_rank.max() - 3))
            
        chosen_features = np.array(model.features)[feature_mask]
    else:
        feature_mask = list(range(len(model.features)))
        chosen_features = np.array(model.features)

    if figsize == 'auto':
        figsize = (d_ego * 0.36 + (d_loc + d_glb) * 0.49 + 2 + .5, len(chosen_features) * 0.15 + .25 + .75)
    # print(figsize)
    fig, axes = plt.subplots(2, d + 1, sharey='row', sharex='col',
                                          figsize=figsize, 
                                          height_ratios=(.75, len(chosen_features) * .15 + .25),
                                          width_ratios=[2] * d_ego + [3] * (d_loc + d_glb) + [.5])
    plot_axes = axes[1]
    bar_axes = axes[0]
    cbar_ax = plot_axes[-1].inset_axes([0.0, 0.1, 1.0, .8])
    common_params = {'linewidths': .05, 'linecolor': 'gray', 'yticklabels': chosen_features, 
                     'cmap': 'Reds', 'cbar_kws': {"orientation": "vertical"}, 'square': True,
                     'vmax': vmax, 'vmin': vmin}

    # Local
    #
    for i in range(0, d_loc + d_glb + d_ego):
        title = ''
        if i < d_ego:
            what = f'{i}'
            if i == (d_ego - 1) // 2:
                if d_ego % 2 == 0:
                    title += '          '
                title += 'Ego'
            labels = ('u', 'v')
            to_plot = np.vstack((qk_ego[i, feature_mask],
                                 v_ego[i, feature_mask])).T
            color = qkv_colors[1:]
        elif i < d_loc + d_ego:
            j = i - d_ego
            what = f'{j}'
            if i == (d_loc - 1) // 2 + d_ego:
                if d_loc % 2 == 0:
                    title += '          '
                title += 'Local'
            labels = ('k', 'q', 'v')
            to_plot = np.vstack((k_local[j, feature_mask],
                                 q_local[j, feature_mask],
                                 v_local[j, feature_mask])).T
            color = qkv_colors
        else:
            j = i - d_ego - d_loc
            what = f'{j}'
            if i == (d_glb - 1) // 2 + d_ego + d_loc:
                if d_glb % 2 == 0:
                    title += '          '
                title += 'Global'
            labels = ('k', 'q', 'v')
            to_plot = np.vstack((k_global[j, feature_mask],
                                 q_global[j, feature_mask],
                                 v_global[j, feature_mask])).T
            color = qkv_colors
        
        true_vmax = to_plot.max(axis=0)
        # print(true_vmax)
        to_plot /= true_vmax
 
        bar_axes[i].bar(np.arange(len(true_vmax)) + .5, true_vmax, color=color)
        bar_axes[i].set_xticks(np.arange(len(true_vmax)) + .5, [''] * len(true_vmax))
        bar_axes[i].set_yscale('log')
        bar_axes[i].set_title(title, size=10, fontweight='bold')
        if i != 0:
            bar_axes[i].get_yaxis().set_visible(False)
        for pos in ['right', 'top', 'left']:
            if pos == 'left' and i == 0:
                continue
            else:
                bar_axes[i].spines[pos].set_visible(False)
        sns.heatmap(to_plot, xticklabels=labels, ax=plot_axes[i], 
                    **common_params, cbar_ax=cbar_ax)
        plot_axes[i].set_xlabel(f"{what}")
        
    # All text straight up
    for i in range(d_ego + d_loc + d_glb):
        plot_axes[i].set_xticklabels(plot_axes[i].get_xticklabels(), rotation=0)

    for i in range(1, d_ego + d_loc + d_glb):
        plot_axes[i].get_yaxis().set_visible(False)

    # Remove duplicate cbars
    bar_axes[-1].set_visible(False)

    plot_axes[-1].get_yaxis().set_visible(False)
    plot_axes[-1].get_xaxis().set_visible(False)
    for pos in ['right', 'top', 'bottom', 'left']:
        plot_axes[-1].spines[pos].set_visible(False)
    # axes[-1].set_visible(False)

    fig.align_xlabels()
    plt.tight_layout()

    # Subplots sep lines [credit: https://stackoverflow.com/a/55465138]
    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bboxes = np.array(list(map(get_bbox, axes[1,:-1].flat)), mpl.transforms.Bbox)

    #Get the minimum and maximum extent, get the coordinate half-way between those
    xmax = bboxes[:, 1, 0]
    xmin = bboxes[:, 0, 0]

    # print(xmax, xmin)

    xs = np.c_[xmax[:-1], xmin[1:]].mean(axis=1)

    # for x in xmax:
    #     line = plt.Line2D([x, x],[0, 1], transform=fig.transFigure, color="red", linewidth=1.)
    #     fig.add_artist(line)

    # for x in xmin:
    #     line = plt.Line2D([x, x],[0, 1], transform=fig.transFigure, color="blue", linewidth=1.)
    #     fig.add_artist(line)

    for i, x in enumerate(xs):
        if i in (d_ego - 1, d_ego + d_loc - 1):
            line = plt.Line2D([x, x],[0, 1], transform=fig.transFigure, color="black", linewidth=.5)
            fig.add_artist(line)


def plot_regional_transforms(model: Steamboat, top: int = 3, reorder: bool = False, 
                    figsize: str | tuple[float, float] = 'auto', 
                    vmin: float = 0., vmax: float = 1.,
                    xticklabels: tuple[str, str, str] = ['k', 'q', 'v']):
    """Plot all metagenes

    :param model: Steamboat model
    :param top: Number of top genes per metagene to plot, defaults to 3
    :param reorder: Reorder the genes by metagene, or keep the orginal ordering, defaults to False
    :param figsize: Size of the figure, defaults to 'auto'
    :param vmin: minimum value in the color bar, defaults to 0.
    :param vmax: maximum value in the color bar, defaults to 1.
    """
    n_heads = model.spatial_gather.n_heads
    n_scales = model.spatial_gather.n_scales

    q = model.spatial_gather.q.weight.detach().cpu()
    k = model.spatial_gather.k.weight.detach().cpu()
    v = model.spatial_gather.v.weight.detach().cpu().T
    switch = model.spatial_gather.switch().detach().cpu()

    if top > 0:
        if reorder:
            rank_v = np.argsort(-v, axis=1)[:, :top]
            rank_q = np.argsort(-q, axis=1)[:, :top]
            rank_k = np.argsort(-k, axis=1)[:, :top]
            feature_mask = {}
            for i in range(n_heads):
                for j in rank_k[i, :]:
                    feature_mask[j] = None
                for j in rank_q[i, :]:
                    feature_mask[j] = None
                for j in rank_v[i, :]:
                    feature_mask[j] = None
            feature_mask = list(feature_mask.keys())
        else:
            rank_v = rank(v)
            rank_q = rank(q)
            rank_k = rank(k)
            max_rank = np.max(np.vstack([rank_v, rank_q, rank_k]), axis=0)
            feature_mask = (max_rank > (max_rank.max() - 3))
            
        chosen_features = np.array(model.features)[feature_mask]
    else:
        feature_mask = list(range(len(model.features)))
        chosen_features = np.array(model.features)

    if figsize == 'auto':
        figsize = (n_heads * 0.49 + 2 + .5, len(chosen_features) * 0.15 + .25 + .75)
    # print(figsize)
    fig, axes = plt.subplots(2, n_heads + 1, sharey='row', sharex='col',
                                          figsize=figsize, 
                                          height_ratios=(.75, len(chosen_features) * .15 + .25))
    plot_axes = axes[1]
    bar_axes = axes[0]
    cbar_ax = plot_axes[-1].inset_axes([0.0, 0.1, 1.0, .8])
    common_params = {'linewidths': .05, 'linecolor': 'gray', 'yticklabels': chosen_features, 
                     'cmap': 'Reds', 'cbar_kws': {"orientation": "vertical"}, 'square': True,
                     'vmax': vmax, 'vmin': vmin}

    for i in range(0, n_heads):
        title = ''
        what = f'{i}'
        to_plot = np.vstack((k[i, feature_mask],
                             q[i, feature_mask],
                             v[i, feature_mask])).T
        
        true_vmax = to_plot.max(axis=0)
        # print(true_vmax)
        to_plot /= true_vmax
 
        bar_axes[i].bar(np.arange(len(true_vmax)) + .5, true_vmax)
        bar_axes[i].set_xticks(np.arange(len(true_vmax)) + .5, [''] * len(true_vmax))
        bar_axes[i].set_yscale('log')
        bar_axes[i].set_title(title, size=10, fontweight='bold')
        if i != 0:
            bar_axes[i].get_yaxis().set_visible(False)
        for pos in ['right', 'top', 'left']:
            if pos == 'left' and i == 0:
                continue
            else:
                bar_axes[i].spines[pos].set_visible(False)
        sns.heatmap(to_plot, xticklabels=xticklabels, ax=plot_axes[i], 
                    **common_params, cbar_ax=cbar_ax)
        plot_axes[i].set_xlabel(f"{what}")
        
    # All text straight up
    for i in range(n_heads):
        plot_axes[i].set_xticklabels(plot_axes[i].get_xticklabels(), rotation=0)

    for i in range(1, n_heads):
        plot_axes[i].get_yaxis().set_visible(False)

    # Remove duplicate cbars
    bar_axes[-1].set_visible(False)

    plot_axes[-1].get_yaxis().set_visible(False)
    plot_axes[-1].get_xaxis().set_visible(False)
    for pos in ['right', 'top', 'bottom', 'left']:
        plot_axes[-1].spines[pos].set_visible(False)
    # axes[-1].set_visible(False)

    fig.align_xlabels()
    plt.tight_layout()


def plot_transform(model, scope: Literal['ego', 'local', 'global'], d, 
                   top: int = 3, reorder: bool = False, 
                   figsize: str | tuple[float, float] = 'auto'):
    """Plot a single set of metagenes (q, k, v)

    :param model: Steamboat model
    :param scope: type of the factor: 'ego', 'local', 'global'
    :param d: Number of the head to be plotted
    :param top: Top genes to plot, defaults to 3
    :param reorder: Reorder the genes or use the orginal ordering, defaults to False
    :param figsize: Size of the figure, defaults to 'auto'
    """
    if scope == 'ego':
        qk_ego, v_ego = model.get_ego_transform()
        assert False, "Not implemented for ego."
    elif scope == 'local':
        q, k, v = model.get_local_transform()
    elif scope == 'global':
        q, k, v = model.get_global_transform()
    else:
        assert False, "scope must be local or global."

    q = q[d, :]
    k = k[d, :]
    v = v[d, :]
    
    if top > 0:
        rank_q = np.argsort(-q)[:top]
        rank_k = np.argsort(-k)[:top]
        rank_v = np.argsort(-v)[:top]
        feature_mask = {}
        for j in rank_k:
            feature_mask[j] = None
        for j in rank_q:
            feature_mask[j] = None
        for j in rank_v:
            feature_mask[j] = None
        feature_mask = list(feature_mask.keys())
        chosen_features = np.array(model.features)[feature_mask]
    else:
        feature_mask = list(range(len(model.features)))
        chosen_features = np.array(model.features)

    if figsize == 'auto':
        figsize = (.65, len(chosen_features) * 0.15 + .75)
    # print(figsize)
    fig, ax = plt.subplots(figsize=figsize)
    common_params = {'linewidths': .05, 'linecolor': 'gray', 'yticklabels': chosen_features, 
                     'cmap': 'Reds'}

    to_plot = np.vstack((k[feature_mask],
                         q[feature_mask],
                         v[feature_mask])).T
    true_vmax = to_plot.max(axis=0)
    # print(true_vmax)
    to_plot /= true_vmax

    sns.heatmap(to_plot, xticklabels=['k', 'q', 'v'], ax=ax, **common_params)
    
    # ax.set_xticklabels(plot_axes[i].get_xticklabels(), rotation=0)
    # ax.get_yaxis().set_visible(False)

    plt.tight_layout()


def plot_transforms(model: Steamboat, top: int = 3, reorder: bool = False, 
                    figsize: str | tuple[float, float] = 'auto', 
                    qkv_colors: list[str] = palettes['npg3'],
                    vmin: float = 0., vmax: float = 1.):
    """Plot all metagenes

    :param model: Steamboat model
    :param top: Number of top genes per metagene to plot, defaults to 3
    :param reorder: Reorder the genes by metagene, or keep the orginal ordering, defaults to False
    :param figsize: Size of the figure, defaults to 'auto'
    :param qkv_colors: Colors for the bar plot showing the magnitude of each metagene before normalization, defaults to palettes['npg3']
    :param vmin: minimum value in the color bar, defaults to 0.
    :param vmax: maximum value in the color bar, defaults to 1.
    """
    assert len(qkv_colors) == 3, f"Expect a color palette with at 3 colors, get {len(qkv_colors)}."
    d_ego : int = model.spatial_gather.d_ego
    d_loc : int = model.spatial_gather.d_local
    d_glb : int = model.spatial_gather.d_global
    d : int = d_ego + d_loc + d_glb

    qk_ego, v_ego = model.get_ego_transform()
    q_local, k_local, v_local = model.get_local_transform()
    q_global, k_global, v_global = model.get_global_transform()

    if top > 0:
        if reorder:
            rank_v_ego = np.argsort(-v_ego, axis=1)[:, :top]
            rank_q_local = np.argsort(-np.abs(q_local), axis=1)[:, :top]
            rank_k_local = np.argsort(-k_local, axis=1)[:, :top]
            rank_v_local = np.argsort(-v_local, axis=1)[:, :top]
            rank_q_global = np.argsort(-np.abs(q_global), axis=1)[:, :top]
            rank_k_global = np.argsort(-k_global, axis=1)[:, :top]
            rank_v_global = np.argsort(-v_global, axis=1)[:, :top]
            feature_mask = {}
            for i in rank_v_ego:
                for j in i:
                    feature_mask[j] = None
            for i in range(d_loc):
                for j in rank_k_local[i, :]:
                    feature_mask[j] = None
                for j in rank_q_local[i, :]:
                    feature_mask[j] = None
                for j in rank_v_local[i, :]:
                    feature_mask[j] = None
            for i in range(d_glb):
                for j in rank_k_global[i, :]:
                    feature_mask[j] = None
                for j in rank_q_global[i, :]:
                    feature_mask[j] = None
                for j in rank_v_global[i, :]:
                    feature_mask[j] = None
            feature_mask = list(feature_mask.keys())
        else:
            rank_v_ego = rank(v_ego)
            rank_q_local = rank(np.abs(q_local))
            rank_k_local = rank(k_local)
            rank_v_local = rank(v_local)
            rank_q_global = rank(np.abs(q_global))
            rank_k_global = rank(k_global)
            rank_v_global = rank(v_global)
            max_rank = np.max(np.vstack([rank_v_ego, 
                                        rank_q_local, 
                                        rank_k_local, 
                                        rank_v_local, 
                                        rank_q_global, 
                                        rank_k_global, 
                                        rank_v_global]), axis=0)
            feature_mask = (max_rank > (max_rank.max() - 3))
            
        chosen_features = np.array(model.features)[feature_mask]
    else:
        feature_mask = list(range(len(model.features)))
        chosen_features = np.array(model.features)

    if figsize == 'auto':
        figsize = (d_ego * 0.36 + (d_loc + d_glb) * 0.49 + 2 + .5, len(chosen_features) * 0.15 + .25 + .75)
    # print(figsize)
    fig, axes = plt.subplots(2, d + 1, sharey='row', sharex='col',
                                          figsize=figsize, 
                                          height_ratios=(.75, len(chosen_features) * .15 + .25),
                                          width_ratios=[2] * d_ego + [3] * (d_loc + d_glb) + [.5])
    plot_axes = axes[1]
    bar_axes = axes[0]
    cbar_ax = plot_axes[-1].inset_axes([0.0, 0.1, 1.0, .8])
    common_params = {'linewidths': .05, 'linecolor': 'gray', 'yticklabels': chosen_features, 
                     'cmap': 'Reds', 'cbar_kws': {"orientation": "vertical"}, 'square': True,
                     'vmax': vmax, 'vmin': vmin}

    # Local
    #
    for i in range(0, d_loc + d_glb + d_ego):
        title = ''
        if i < d_ego:
            what = f'{i}'
            if i == (d_ego - 1) // 2:
                if d_ego % 2 == 0:
                    title += '          '
                title += 'Ego'
            labels = ('u', 'v')
            to_plot = np.vstack((qk_ego[i, feature_mask],
                                 v_ego[i, feature_mask])).T
            color = qkv_colors[1:]
        elif i < d_loc + d_ego:
            j = i - d_ego
            what = f'{j}'
            if i == (d_loc - 1) // 2 + d_ego:
                if d_loc % 2 == 0:
                    title += '          '
                title += 'Local'
            labels = ('k', 'q', 'v')
            to_plot = np.vstack((k_local[j, feature_mask],
                                 q_local[j, feature_mask],
                                 v_local[j, feature_mask])).T
            color = qkv_colors
        else:
            j = i - d_ego - d_loc
            what = f'{j}'
            if i == (d_glb - 1) // 2 + d_ego + d_loc:
                if d_glb % 2 == 0:
                    title += '          '
                title += 'Global'
            labels = ('k', 'q', 'v')
            to_plot = np.vstack((k_global[j, feature_mask],
                                 q_global[j, feature_mask],
                                 v_global[j, feature_mask])).T
            color = qkv_colors
        
        true_vmax = to_plot.max(axis=0)
        # print(true_vmax)
        to_plot /= true_vmax
 
        bar_axes[i].bar(np.arange(len(true_vmax)) + .5, true_vmax, color=color)
        bar_axes[i].set_xticks(np.arange(len(true_vmax)) + .5, [''] * len(true_vmax))
        bar_axes[i].set_yscale('log')
        bar_axes[i].set_title(title, size=10, fontweight='bold')
        if i != 0:
            bar_axes[i].get_yaxis().set_visible(False)
        for pos in ['right', 'top', 'left']:
            if pos == 'left' and i == 0:
                continue
            else:
                bar_axes[i].spines[pos].set_visible(False)
        sns.heatmap(to_plot, xticklabels=labels, ax=plot_axes[i], 
                    **common_params, cbar_ax=cbar_ax)
        plot_axes[i].set_xlabel(f"{what}")
        
    # All text straight up
    for i in range(d_ego + d_loc + d_glb):
        plot_axes[i].set_xticklabels(plot_axes[i].get_xticklabels(), rotation=0)

    for i in range(1, d_ego + d_loc + d_glb):
        plot_axes[i].get_yaxis().set_visible(False)

    # Remove duplicate cbars
    bar_axes[-1].set_visible(False)

    plot_axes[-1].get_yaxis().set_visible(False)
    plot_axes[-1].get_xaxis().set_visible(False)
    for pos in ['right', 'top', 'bottom', 'left']:
        plot_axes[-1].spines[pos].set_visible(False)
    # axes[-1].set_visible(False)

    fig.align_xlabels()
    plt.tight_layout()

    # Subplots sep lines [credit: https://stackoverflow.com/a/55465138]
    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bboxes = np.array(list(map(get_bbox, axes[1,:-1].flat)), mpl.transforms.Bbox)

    #Get the minimum and maximum extent, get the coordinate half-way between those
    xmax = bboxes[:, 1, 0]
    xmin = bboxes[:, 0, 0]

    # print(xmax, xmin)

    xs = np.c_[xmax[:-1], xmin[1:]].mean(axis=1)

    # for x in xmax:
    #     line = plt.Line2D([x, x],[0, 1], transform=fig.transFigure, color="red", linewidth=1.)
    #     fig.add_artist(line)

    # for x in xmin:
    #     line = plt.Line2D([x, x],[0, 1], transform=fig.transFigure, color="blue", linewidth=1.)
    #     fig.add_artist(line)

    for i, x in enumerate(xs):
        if i in (d_ego - 1, d_ego + d_loc - 1):
            line = plt.Line2D([x, x],[0, 1], transform=fig.transFigure, color="black", linewidth=.5)
            fig.add_artist(line)


def calc_v_weights(model: Steamboat, normalize: bool = True):
    v_weights = model.spatial_gather.v.weight.detach().cpu().numpy().sum(axis=0)
    if normalize:
        v_weights = v_weights / sum(v_weights)
    return v_weights

def calc_head_weights(adatas, model: Steamboat):
    ego = 0
    local = 0
    regional = 0

    for i in range(len(adatas)):
        ego += np.mean(adatas[i].obsm['ego_attn'], axis=0)
        local += np.mean(adatas[i].obsm['local_attn'], axis=0)
        regional += np.mean(adatas[i].obsm['global_attn_0'], axis=0)

    matrix = np.vstack([ego, local, regional]) * calc_v_weights(model)

    return matrix


def calc_interaction(adatas, model: Steamboat, sample_key: str, cell_type_key: str, pseudocount: float = 20.):
    v_weights = calc_v_weights(model)
    celltype_attnp_df_dict = {}
    for i in range(len(adatas)):    
        total_attnp = None
        for j in range(model.spatial_gather.n_heads):
            if total_attnp is None:
                total_attnp = adatas[i].obsp[f'local_attn_{j}'] * v_weights[j] * 25
            else:
                total_attnp += adatas[i].obsp[f'local_attn_{j}'] * v_weights[j] * 25
        
        celltypes = sorted(adatas[i].obs['cell.types.nolc'].unique())
        celltype_attnp_df = pd.DataFrame(-1., index=celltypes, columns=celltypes)
        
        actual_min = float("inf")
        for celltype0 in celltype_attnp_df.index:
            mask0 = (adatas[i].obs[cell_type_key] == celltype0)
            for celltype1 in celltype_attnp_df.columns:
                mask1 = (adatas[i].obs[cell_type_key] == celltype1)
                sub_attnp = total_attnp[mask0, :][:, mask1]
                normalization_factor = sub_attnp.nnz + 20
                # normalization_factor = np.prod(sub_attnp.shape)
                if normalization_factor >= 1:
                    celltype_attnp_df.loc[celltype0, celltype1] = sub_attnp.sum() / normalization_factor
                    actual_min = min(actual_min, celltype_attnp_df.loc[celltype0, celltype1])
                else:
                    celltype_attnp_df.loc[celltype0, celltype1] = 0.

        celltype_attnp_df_dict[adatas[i].obs[sample_key].unique().astype(str).item()] = celltype_attnp_df + celltype_attnp_df.T
    return celltype_attnp_df_dict


def calc_adjacency_freq(adatas, sample_key: str, cell_type_key: str):
    adjacency_freq = {}
    for i in range(len(adatas)):
        k = adatas[i].obs[sample_key].unique().astype(str).item()
        adatas[i].obs[cell_type_key] = adatas[i].obs[cell_type_key].astype("category")
        sq.gr.interaction_matrix(adatas[i], cell_type_key, normalized=False)
        temp = pd.DataFrame(adatas[i].uns['cell.types.nolc_interactions'], 
                                            index=adatas[i].obs[cell_type_key].cat.categories, 
                                            columns=adatas[i].obs[cell_type_key].cat.categories)
        normalization_factor = adatas[i].obs[cell_type_key].value_counts().sort_index() + 20
        adjacency_freq[k] = temp.div(normalization_factor, axis=0).div(normalization_factor, axis=1)
    return adjacency_freq


def plot_head_weights(head_weights, multiplier: float = 100, order=None, figsize=(7, 0.8), heatmap_kwargs=None, save: str = None):
    matrix = head_weights.copy()
    matrix /= matrix.sum()
    fig, ax = plt.subplots(figsize=figsize)

    heatmap_kwargs0 = dict(vmax=10, linewidths=0.2, linecolor='grey', cmap='Reds', annot=True, fmt='.0f', square=True)
    if heatmap_kwargs is not None:
        for key, value in heatmap_kwargs.items():
            heatmap_kwargs0[key] = value
    
    if order is None:
        sns.heatmap(matrix * 100, ax=ax, **heatmap_kwargs0)
    else:
        sns.heatmap(matrix[:, order] * 100, vmax=10, ax=ax, linewidths=0.2, linecolor='grey', cmap='Reds', annot=True, fmt='.0f', square=True)
        ax.set_xticklabels(order, rotation=0)
    
    ax.set_yticklabels(['ego', 'local', 'regional'], rotation=0)

    if save is not None and save != False:
        assert isinstance(save, str), "save must be a string."
        fig.savefig(save, bbox_inches='tight', transparent=True)


def calc_var(model: Steamboat):
    n_heads = model.spatial_gather.n_heads
    q = model.spatial_gather.q.weight.detach().cpu().numpy()
    k_local = model.spatial_gather.k_local.weight.detach().cpu().numpy()
    k_global = model.spatial_gather.k_regionals[0].weight.detach().cpu().numpy()
    v = model.spatial_gather.v.weight.detach().cpu().numpy().T

    index = ([f'q_{i}' for i in range(n_heads)] + 
            [f'k_local_{i}' for i in range(n_heads)] + 
            [f'k_global_{i}' for i in range(n_heads)] + 
            [f'v_{i}' for i in range(n_heads)])

    return pd.DataFrame(np.vstack([q, k_local, k_global, v]), 
                        index=index, columns=model.features).T


def calc_geneset_auroc(metagenes, genesets):
    gene_df = metagenes.copy()
    df = pd.DataFrame(index=gene_df.columns)
    for k, v in genesets.items():
        aurocs = []
        pvals = []
        for i in gene_df.columns:
            aurocs.append(roc_auc_score(gene_df.index.isin(v), gene_df[i]))
            pvals.append(sp.stats.mannwhitneyu(gene_df.loc[gene_df.index.isin(v), i],
                                                           gene_df.loc[~gene_df.index.isin(v), i]).pvalue)
        df[k + '_auroc'] = aurocs
        # df[k + '_p'] = pvals
    return df


def calc_geneset_auroc_order(sig_df, by='q'):
    plt_df = sig_df[sig_df.index.str.contains(by + '_')]
    order = np.argsort(np.argmax(plt_df, axis=1) - np.max(plt_df, axis=1) / (np.max(plt_df) + 1)).tolist()
    return order


def plot_geneset_auroc(sig_df, order, figsize=(8, 5)):
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    ax = axes[0]
    plt_df = sig_df[sig_df.index.str.contains('q_')]
    sns.heatmap(plt_df.T.iloc[:, order], vmin=.2, vmax=.8, cmap='vlag', 
            linewidths=.5, linecolor='grey', ax=ax, square=True)
    ax.set_xticklabels(order, rotation=0)
    ax.set_xlabel('Center cell metagenes')

    ax = axes[1]
    plt_df = sig_df[sig_df.index.str.contains('k_local')]
    sns.heatmap(plt_df.T.iloc[:, order], vmin=.2, vmax=.8, cmap='vlag', 
            linewidths=.5, linecolor='grey', ax=ax, square=True)
    ax.set_xticklabels(order, rotation=0)
    ax.set_xlabel('Local environment metagenes')

    ax = axes[2]
    plt_df = sig_df[sig_df.index.str.contains('k_global')]
    sns.heatmap(plt_df.T.iloc[:, order], vmin=.2, vmax=.8, cmap='vlag', 
            linewidths=.5, linecolor='grey', ax=ax, square=True)
    ax.set_xticklabels(order, rotation=0)
    ax.set_xlabel('Global environment metagenes')

    fig.tight_layout()
    return fig, ax

def calc_obs(adatas: list[sc.AnnData], dataset: SteamboatDataset, model: Steamboat, 
                    device='cuda', get_recon: bool = False):
    """Calculate and store the embeddings and attention scores in the AnnData objects
    
    :param adatas: List of AnnData objects to store the embeddings and attention scores
    :param dataset: SteamboatDataset object to be processed
    :param model: Steamboat model
    :param device: Device to run the model, defaults to 'cuda'
    :param get_recon: Whether to store the reconstructed data, defaults to False
    """
    # Safeguards
    assert len(adatas) == len(dataset), "mismatch in lenghths of adatas and dataset"
    for adata, data in zip(adatas, dataset):
        assert adata.shape[0] == data[0].shape[0], f"adata[{i}] has {adata.shape[0]} cells but dataset[{i}] has {data[0].shape[0]}."

    # Calculate embeddings and attention scores for each slide
    for i, (x, adj_list, regional_xs, regional_adj_lists) in tqdm(enumerate(dataset), total=len(dataset)):
        adj_list = adj_list.squeeze(0).to(device)
        x = x.squeeze(0).to(device)
        regional_adj_lists = [regional_adj_list.to(device) for regional_adj_list in regional_adj_lists]
        regional_xs = [regional_x.to(device) for regional_x in regional_xs]
        
        with torch.no_grad():
            res, details = model(adj_list, x, x, regional_adj_lists, regional_xs, get_details=True)
            
            if get_recon:
                adatas[i].obsm['X_recon'] = res.cpu().numpy()

            adatas[i].obsm['q'] = details['embq'].cpu().numpy()
            adatas[i].obsm['local_k'] = details['embk'][0].cpu().numpy()

            for j in range(model.spatial_gather.n_scales - 2):
                adatas[i].obsm[f'global_k_{j}'] = model.spatial_gather.k_regionals[j](x).cpu().numpy()

            for j, emb in enumerate(details['embk'][1]):
                adatas[i].uns[f'global_k_{j}'] = emb.cpu().numpy()
                
            adatas[i].obsm['attn'] = details['attn'].cpu().numpy()
            adatas[i].obsm['ego_attn'] = details['attnm'][0].cpu().numpy()
            adatas[i].obsm['local_attn'] = details['attnm'][1].cpu().numpy()

            for j, matrix in enumerate(details['attnm'][2]):
                adatas[i].obsm[f'global_attn_{j}'] = matrix.cpu().numpy()

            # local attention (as graph)
            for j in range(model.spatial_gather.n_heads):
                w = details['attnp'][1].cpu().numpy()[:, j, :].flatten()
                uv = adj_list.cpu().numpy()
                u = uv[0]
                v = uv[1]
                if uv.shape[0] == 3: # masked for unequal neighbors
                    m = (uv[2] > 0)
                    w, u, v = w[m], u[m], v[m]
                adatas[i].obsp[f'local_attn_{j}'] = sp.sparse.csr_matrix((w, (u, v)), 
                                                                            shape=(adatas[i].shape[0], 
                                                                                adatas[i].shape[0]))


def gather_obs(adata: sc.AnnData, adatas: list[sc.AnnData]):
    """Gather obs/obsm/uns from a list of AnnData objects to a single AnnData object

    """
    all_embq = []
    all_embk = []
    all_embk_glb = []
    all_ego_attn = []
    all_local_attn = []
    all_global_attn = []
    all_attn = []
    
    for i in range(len(adatas)):
        all_embq.append(adatas[i].obsm['q'])
        all_embk.append(adatas[i].obsm['local_k'])
        all_ego_attn.append(adatas[i].obsm['ego_attn'])
        all_local_attn.append(adatas[i].obsm['local_attn'])
        all_global_attn.append(adatas[i].obsm['global_attn_0'])
        all_attn.append(adatas[i].obsm['attn'])
        all_embk_glb.append(adatas[i].obsm['global_k_0'])

    adata.obsm['q'] = np.vstack(all_embq)
    adata.obsm['local_k'] = np.vstack(all_embk)
    adata.obsm['ego_attn'] = np.vstack(all_ego_attn)
    adata.obsm['local_attn'] = np.vstack(all_local_attn)
    adata.obsm['global_attn'] = np.vstack(all_global_attn)
    
    adata.obsm['attn'] = np.vstack(all_attn)
    adata.obsm['global_k_0'] = np.vstack(all_embk_glb)

    if 'X_recon' in adatas[0].obsm:
        all_recon = []
        for i in range(len(adatas)):
            all_recon.append(adatas[i].obsm['X_recon'])
        adata.obsm['X_recon'] = np.vstack(all_recon)

    return adata


def neighbors(adata: sc.AnnData,
              use_rep: str = 'X_local_q', 
              key_added: str = 'steamboat_emb',
              metric='cosine', 
              neighbors_kwargs: dict = None):
    """A thin wrapper for scanpy.pp.neighbors for Steamboat functionalities

    :param adata: AnnData object to be processed
    :param use_rep: embedding to be used, 'X_local_q' or 'X_local_attn' (if very noisy data), defaults to 'X_local_q'
    :param key_added: key in obsp to store the resulting similarity graph, defaults to 'steamboat_emb'
    :param metric: metric for similarity graph, defaults to 'cosine'
    :param neighbors_kwargs: Other parameters for scanpy.pp.neighbors if desired, defaults to None
    :return: hands over what scanpy.pp.neighbors returns
    """
    if neighbors_kwargs is None:
        neighbors_kwargs = {}
    return sc.pp.neighbors(adata, use_rep=use_rep, key_added=key_added, metric=metric, **neighbors_kwargs)


def leiden(adata: sc.AnnData, resolution: float = 1., *,
            obsp='steamboat_emb_connectivities',
            key_added='steamboat_clusters',
            leiden_kwargs: dict = None):
    """A thin wrapper for scanpy.tl.leiden to cluster for cell types (for spatial domain segmentation, use `segment`).

    :param adata: AnnData object to be processed
    :param resolution: resolution for Leiden clustering, defaults to 1.
    :param obsp: obsp key to be used, defaults to 'steamboat_emb_connectivities'
    :param key_added: obs key to be added for resulting clusters, defaults to 'steamboat_clusters'
    :param leiden_kwargs: Other parameters for scanpy.tl.leiden if desired, defaults to None
    :return: hands over what scanpy.tl.leiden returns
    """
    if leiden_kwargs is None:
        leiden_kwargs = {}
    return sc.tl.leiden(adata, obsp=obsp, key_added=key_added, resolution=resolution, **leiden_kwargs)
    

def segment(adata: sc.AnnData, resolution: float = 1., *,
            key_added: str = 'steamboat_spatial_domain',
            key_added_pairwise: str = 'pairwise',
            key_added_similarity: str = 'similarity', 
            key_added_combined: str = 'combined', 
            n_prop: int = 3,
            spatial_graph_threshold: float = 0.0,
            leiden_kwargs: dict = None):
    """Spatial domain segmentation using Steamboat embeddings and graphs

    :param adata: AnnData object to be processed
    :param resolution: resolution for Leiden clustering, defaults to 1.
    :param embedding_key: key in obsp for similarity graph (by running `neighbors`), defaults to 'steamboat_emb_connectivities'
    :param key_added: obs key for semgentaiton result, defaults to 'steamboat_spatial_domain'
    :param obsp_summary: obsp key for summary spatial graph, defaults to 'steamboat_summary_connectivities'
    :param obsp_combined: obsp key for combined spatial and similarity graphs, defaults to 'steamboat_combined_connectivities'
    :param spatial_graph_threshold: threshold to include/exclude an edge, a larger number will make the program run faster but potentially less accurate, defaults to 0.0
    :param leiden_kwargs: Other parameters for scanpy.tl.leiden if desired, defaults to None
    :return: _descripthands over what scanpy.tl.leiden returnsion_
    """
    if leiden_kwargs is None:
        leiden_kwargs = {}

    adata.obsm['local_attn_std'] = adata.obsm['local_attn'] / adata.obsm['local_attn'].std(axis=0, keepdims=True)
    sc.pp.neighbors(adata, use_rep='local_attn_std', key_added=key_added_similarity, metric='euclidean')

    temp = 0
    j = 0
    while f'local_attn_{j}' in adata.obsp:
        temp += adata.obsp[f'local_attn_{j}']
        j += 1

    temp = temp ** n_prop
    temp = temp.power(1/n_prop)

    temp.data /= temp.data.max()
    temp.data[temp.data < spatial_graph_threshold] = 0
    temp.eliminate_zeros()

    adata.obsp[key_added_pairwise + '_connectivities'] = temp
    adata.obsp[key_added_combined + '_connectivities'] = (adata.obsp[key_added_pairwise + '_connectivities'] + 
                                                          adata.obsp[key_added_similarity + '_connectivities'])
    adata.obsp[key_added_combined + '_connectivities'].eliminate_zeros() 
    return sc.tl.leiden(adata, obsp=key_added_combined + '_connectivities', 
                        key_added=key_added, resolution=resolution, **leiden_kwargs)


def plot_vq(model, chosen_features):
    features_mask = [model.features.index(i) for i in chosen_features]

    q = model.spatial_gather.q.weight.detach().cpu().numpy().T
    q = q[features_mask, :]
    q = q / q.max(axis=0)
    head_order = np.argsort(np.argmax(q.T, axis=1) - np.max(q.T, axis=1) / (np.max(q.T) + 1)).tolist()

    common_params = {'linewidths': .05, 'linecolor': 'gray', 'cmap': 'Reds'}
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.heatmap(q[:, head_order], yticklabels=chosen_features, xticklabels=head_order, square=True, **common_params, ax=ax)
    return fig, ax


def plot_all_transforms(model, 
                   top: int = 3, head_order=None,
                   figsize: str | tuple[float, float] = 'auto',
                    chosen_features=None):
    if chosen_features is None:
        feature_mask = {}
        for d in head_order if head_order is not None else range(model.spatial_gather.n_heads):
            k1 = model.spatial_gather.k_local.weight[d, :].detach().cpu().numpy()
            k2 = model.spatial_gather.k_regionals[0].weight[d, :].detach().cpu().numpy()
            q = model.spatial_gather.q.weight[d, :].detach().cpu().numpy()
            v = model.spatial_gather.v.weight[:, d].detach().cpu().numpy()
        
            rank_q = np.argsort(-q)[:top]
            rank_k1 = np.argsort(-k1)[:top]
            rank_k2 = np.argsort(-k2)[:top]
            rank_v = np.argsort(-v)[:top]
    
            for j in rank_k1:
                feature_mask[j] = None
            for j in rank_k2:
                feature_mask[j] = None
            for j in rank_q:
                feature_mask[j] = None
            for j in rank_v:
                feature_mask[j] = None
        feature_mask = list(feature_mask.keys())
        chosen_features = np.array(model.features)[feature_mask]
    else:
        feature_mask = []
        for i in chosen_features:
            feature_mask.append(model.features.index(i))
    print(chosen_features)
    
    if figsize == 'auto':
        figsize = (.7 * (1 + model.spatial_gather.n_heads), len(chosen_features) * 0.15 + .75)
    fig, axes = plt.subplots(1, model.spatial_gather.n_heads + 1, figsize=figsize, sharey='row')

    cbar_ax = axes[-1].inset_axes([0.0, 0.1, .2, .8])
    axes[-1].get_xaxis().set_visible(False)
    for ax in axes[1:]:
        ax.get_yaxis().set_visible(False)
    for pos in ['right', 'top', 'bottom', 'left']:
        axes[-1].spines[pos].set_visible(False)
        
    for i_ax, d in enumerate(head_order if head_order is not None else range(model.spatial_gather.n_heads)):
        k1 = model.spatial_gather.k_local.weight[d, :].detach().cpu().numpy()
        k2 = model.spatial_gather.k_regionals[0].weight[d, :].detach().cpu().numpy()
        q = model.spatial_gather.q.weight[d, :].detach().cpu().numpy()
        v = model.spatial_gather.v.weight[:, d].detach().cpu().numpy()
        
        common_params = {'linewidths': .05, 'linecolor': 'gray', 'yticklabels': chosen_features, 
                         'cmap': 'Reds'}

        to_plot = np.vstack((k2[feature_mask],
                                 k1[feature_mask],
                                 q[feature_mask],
                                 v[feature_mask])).T
        true_vmax = to_plot.max(axis=0)
        # print(true_vmax)
        to_plot /= true_vmax
        
        sns.heatmap(to_plot, xticklabels=['global env', 'local env', 'ego env / center', 'reconstruction'], square=True, 
                    ax=axes[i_ax], **common_params, cbar_ax=cbar_ax)
        axes[i_ax].set_title(d)
        # axes[i_ax].set_xticklabels(['global env', 'local env', 'ego env / center', 'reconstruction'], rotation=45, ha='right', va='center', rotation_mode='anchor')
        # ax.set_xticklabels(plot_axes[i].get_xticklabels(), rotation=0)
        # ax.get_yaxis().set_visible(False)
    
    plt.tight_layout()


def plot_cell_type_enrichment(all_adata, adatas, score_dim, label_key, select_labels=None,
                              figsize=(.75, 4)):
    all_adata.obsm[f'q_{score_dim}'] = np.vstack([i.obsm['q'][:, None, score_dim] for i in adatas])
    all_adata.obsm[f'global_attn_{score_dim}'] = np.vstack([i.obsm['global_attn_0'][:, None, score_dim] for i in adatas])
    all_adata.obsm[f'global_k_0_{score_dim}'] = np.vstack([i.obsm['global_k_0'][:, None, score_dim] for i in adatas])

    cols = [f'q_{score_dim}']
    global_attn_df = pd.DataFrame(all_adata.obsm[f'q_{score_dim}'], 
                                index=all_adata.obs_names, 
                                columns=cols)
    global_attn_df[label_key] = all_adata.obs[label_key]

    cell_median_df = global_attn_df[[label_key] + cols].groupby(label_key).median().astype('float')
    cell_p_df = cell_median_df.copy()
    cell_p_df[:] = 0.
    cell_f_df = cell_p_df.copy()

    for i in cell_p_df.columns:
        for j in cell_p_df.index:
            x = global_attn_df.loc[global_attn_df[label_key] == j, i]
            y = global_attn_df.loc[global_attn_df[label_key] != j, i]
            test_res = sp.stats.mannwhitneyu(x, y)
            cell_p_df.loc[j, i] = test_res.pvalue
            cell_f_df.loc[j, i] = test_res.statistic / len(x) / len(y)

    selected_celltypes = {}
    for i in cell_f_df.columns:
        for j in cell_f_df.sort_values(i, ascending=False).index[:len(cell_f_df)]:
            if j not in ['dirt', 'undefined']:
                selected_celltypes[j] = None
    selected_celltypes = list(selected_celltypes.keys())
    if select_labels is not None:
        selected_celltypes = [i for i in selected_celltypes if i in select_labels]
    
    cols = [f'global_attn_{score_dim}']
    global_attn_df = pd.DataFrame(all_adata.obsm[f'global_attn_{score_dim}'], 
                                index=all_adata.obs_names, 
                                columns=cols)
    global_attn_df[label_key] = all_adata.obs[label_key]

    for i in cols:
        for j in cell_p_df.index:
            x = global_attn_df.loc[global_attn_df[label_key] == j, i]
            y = global_attn_df.loc[global_attn_df[label_key] != j, i]
            test_res = sp.stats.mannwhitneyu(x, y)
            cell_p_df.loc[j, i] = test_res.pvalue
            cell_f_df.loc[j, i] = test_res.statistic / len(x) / len(y)


    cols = [f'global_k_0_{score_dim}']
    global_attn_df = pd.DataFrame(all_adata.obsm[f'global_k_0_{score_dim}'], 
                                index=all_adata.obs_names, 
                                columns=cols)
    global_attn_df[label_key] = all_adata.obs[label_key]

    for i in cols:
        for j in cell_p_df.index:
            x = global_attn_df.loc[global_attn_df[label_key] == j, i]
            y = global_attn_df.loc[global_attn_df[label_key] != j, i]
            test_res = sp.stats.mannwhitneyu(x, y)
            cell_p_df.loc[j, i] = test_res.pvalue
            cell_f_df.loc[j, i] = test_res.statistic / len(x) / len(y)
            
    cell_p_df *= cell_p_df.shape[0]

    common_params = {'linewidths': .05, 'linecolor': 'gray', 'cmap': 'vlag', 'center': .5, 'square': True}

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cell_f_df.loc[selected_celltypes], ax=ax, **common_params)
    # selected_cell_p_df = cell_p_df.loc[selected_celltypes]
    # for i, iv in enumerate(selected_cell_p_df.index):
    #     for j, jv in enumerate(selected_cell_p_df.columns):
    #         text = p2stars(selected_cell_p_df.loc[iv, jv])
    #         ax.text(j + .5, i + .5, text,
    #                 horizontalalignment='center',
    #                 verticalalignment='center',
    #                 c='white', size=8)
    ax.set_xticks([])

    return fig, ax