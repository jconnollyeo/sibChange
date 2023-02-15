# %load simulationFuncs.py

def plotTS(data, mode, norm=True, ax=None):
    """
    Plot the time series of a pixel and all of its siblings. 
    
    data must be ion the output format of flatWrapper. 
    
    mode: "phase", "amp"
    
    """
    import numpy as np
    import sys
    import matplotlib.pyplot as plt
    
    if isinstance(ax, type(None)):
        fig, ax = plt.subplots(figsize=(12, 5))
    else: pass
    
    if mode == "phase":
        if norm:
            # data = data/abs(data)
#             data = np.angle(data/data[:, 0, 0][:, None, None])
            data = np.angle(data*data[:, 0, 0][:, None, None].conjugate())
        else:
            data = np.angle(data)
        ax.set_ylim([-np.pi, np.pi])
    elif mode == "amp":
        if norm:
            data = abs(data) - abs(data[:, 0, 0][:, None, None])
        else:
            data = abs(data)

    else:
        print ("mode unrecognised.")
        sys.exit(1)

    for p in np.arange(1, data.shape[1]):
        ax.plot(data[:, p, 0], color="black", alpha=0.2)
    ax.plot(data[:, 0, 0], color="red")

    return ax

def pixelCoherence(im1, im2, sibling_i, sibling_j):
    import numpy as np

    coherence_num = np.nansum(im1[sibling_i, sibling_j] *
                              im2[sibling_i, sibling_j].conjugate())
    coherence_den = np.sqrt(np.nansum(im1[sibling_i, sibling_j] *
                                      im1[sibling_i, sibling_j].conjugate()) *
                            np.nansum(im2[sibling_i, sibling_j] * 
                                      im2[sibling_i, sibling_j].conjugate()))

    return coherence_num / coherence_den

def coherenceMatrix(i, j, SHP_i, SHP_j, arr):
    import numpy as np
    mat = np.zeros((arr.shape[0], arr.shape[0]), dtype=np.complex64)
    
    i_, j_ = np.mgrid[0:arr.shape[0], 0:arr.shape[0]]
    
    mask = i_ <= j_
    
    i_ = i_[mask].flatten()
    j_ = j_[mask].flatten()
    
    sibling_i = SHP_i[:, i, j]
    sibling_j = SHP_j[:, i, j]
    
    mask = ~np.isnan(sibling_i)*~np.isnan(sibling_j)
    
    sibling_i = sibling_i[mask]
    sibling_j = sibling_j[mask]
    
    for i_ix, j_ix in zip(i_, j_):
        mat[i_ix, j_ix] = pixelCoherence(arr[i_ix, :, :], arr[j_ix, :, :], sibling_i.astype(int), sibling_j.astype(int))
    return mat

def bootstrapCoherence(im1, im2, SHP_i, SHP_j, max_iter=1000, n_siblings = None, progress=True, track=False):
    import random
    import numpy as np
    from tqdm.notebook import tqdm
    
    coherences = np.empty((max_iter), dtype=np.complex64)
    mask = ~np.isnan(SHP_i) * ~np.isnan(SHP_j)
    SHP_i = SHP_i[mask].astype(int)
    SHP_j = SHP_j[mask].astype(int)
    
    if isinstance(n_siblings, type(None)):
        n_siblings = np.nansum(mask)
        n_siblings_ix = np.arange(n_siblings)
    else: 
        n_siblings = n_siblings
        n_siblings_ix = np.arange(np.nansum(mask))
    
    random_ixs = np.array(random.choices(np.arange(n_siblings), k=n_siblings*max_iter)).reshape((max_iter, n_siblings))
    
    for q in tqdm(np.arange(max_iter), disable=~progress):
        sib_i = SHP_i[random_ixs[q]]
        sib_j = SHP_j[random_ixs[q]]
        coherences[q] = pixelCoherence(im1, im2, sib_i, sib_j)
    
    if track:
        return coherences, random_ixs
    else:
        return coherences

def bootstrapTimeseries(i, j, sib_i, sib_j, ifgs, baseline=1, max_iter=1000, track=False, disable=False):
    import numpy as np
    from tqdm import tqdm
    from math import ceil

    if baseline == 0:
        BS_timeseries = np.empty((ifgs.shape[0], max_iter), dtype=np.complex64)
    else:
        BS_timeseries = np.empty((ceil(ifgs.shape[0]/baseline) - 1, max_iter), dtype=np.complex64)
    sibi = sib_i[:, i, j]
    sibj = sib_j[:, i, j]
    m = ~np.isnan(sibi)*~np.isnan(sibj)
    sibi = sibi[m].astype(int)
    sibj = sibj[m].astype(int)

    tracked_siblings = np.empty((ifgs.shape[0]-1, max_iter, np.sum(m)), dtype=int)
            
    for ix, d_ix in enumerate(tqdm(range(ifgs.shape[0])[:-baseline:int(baseline)], disable=disable)):
        im1 = ifgs[d_ix]
        im2 = ifgs[d_ix+int(baseline)]
        
        if track:
            coherences, random_ixs = bootstrapCoherence(im1, im2, sibi.flatten(),
                                            sibj.flatten(), progress=False, max_iter=max_iter, track=track)
            BS_timeseries[ix] = coherences
            tracked_siblings[ix] = random_ixs
            
        else:
            coherences = bootstrapCoherence(im1, im2, sibi.flatten(),
                                            sibj.flatten(), progress=False, max_iter=max_iter, track=track)

            BS_timeseries[ix] = coherences
        
    return BS_timeseries, tracked_siblings

def coherenceTimeseries(i, j, sib_i, sib_j, ifgs, baseline=1, progress=False):
    import numpy as np
    from math import ceil
    from tqdm.notebook import tqdm

    if baseline == 0:
        out = np.empty((ifgs.shape[0]), dtype=np.complex64)
    else:
        out = np.empty((ceil(ifgs.shape[0]/baseline) - 1), dtype=np.complex64)
        
    out = np.empty((ifgs.shape[0] - 1), dtype=np.complex64)
    sibi = sib_i[:, i, j]
    sibj = sib_j[:, i, j]
    m = ~np.isnan(sibi)*~np.isnan(sibj)
    sibi = sibi[m].astype(int)
    sibj = sibj[m].astype(int)

    for d_ix in tqdm(range(ifgs.shape[0])[:-baseline:int(baseline)], disable=~progress):
        im1 = ifgs[d_ix]
        im2 = ifgs[d_ix+int(baseline)]
        coherence = pixelCoherence(im1, im2, sibi.flatten(),
                                        sibj.flatten())

        out[d_ix] = coherence
        
    return out

def flatBSWrapper(ifgs, ij, SHP_i, SHP_j, start=140, end=None, n=5, central=False, seed=None):
    import numpy as np
    i, j = ij
    
    N = np.sum(~np.isnan(SHP_i[:, i, j])*~np.isnan(SHP_j[:, i, j]))
    
    if isinstance(n, float):
        n = round(N*n)
    else:
        pass
    
    ph = addNoise((i, j), ifgs, SHP_i, SHP_j, start=start, end=end, n=n, central=central, seed=seed).T
    
    sib_i = SHP_i[~np.isnan(SHP_i[:, i, j]), i, j].astype(int)
    sib_j = SHP_j[~np.isnan(SHP_j[:, i, j]), i, j].astype(int)
    
    ph = abs(ifgs[:, sib_i, sib_j])*np.exp(1j*ph)
    
    n = ph.shape[1]
    
    sib_j = np.zeros(n)[:, None, None]
    sib_i = np.arange(n)[:, None, None]
    
    return 0, 0, sib_i, sib_j, ph[:, :, None]

def flatWrapper(ifgs, ij, SHP_i, SHP_j):
    import numpy as np
    i, j = ij
        
    sib_i = SHP_i[~np.isnan(SHP_i[:, i, j])*~np.isnan(SHP_j[:, i, j]), i, j].astype(int)
    sib_j = SHP_j[~np.isnan(SHP_i[:, i, j])*~np.isnan(SHP_j[:, i, j]), i, j].astype(int)
    
    ph = ifgs[:, sib_i, sib_j] #abs(ifgs[:, sib_i, sib_j])*np.exp(1j*ph)
    
    n = ph.shape[1]
    
    sib_j = np.zeros(n)[:, None, None]
    sib_i = np.arange(n)[:, None, None]
    
    return 0, 0, sib_i, sib_j, ph[:, :, None]


def AICdiff(X):
    """
    This fits a GMM with one component, and another with two 
    components to the data. Then finds the AIC diff between 
    them to determine which is the better fit. 
    """
    from sklearn.mixture import GaussianMixture
    
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    else:
        pass
    
    model1 = GaussianMixture(n_components=1, init_params='kmeans')
    model1.fit(X)
    AIC1 = model1.aic(X)

    model2 = GaussianMixture(n_components=3, init_params='kmeans')
    model2.fit(X)
    AIC2 = model2.aic(X)
    
    AICdiff = AIC2 - AIC1
    
    return AICdiff

def AICdiffTS(X):
    import numpy as np
    AICs = np.zeros(X.shape[0], dtype=np.float16)
    
    for d in np.arange(X.shape[0]):
        X_ = abs(X[d])
        X_[np.isnan(X_)] = 0.
        AICs[d] = AICdiff(abs(X_))
    
    return AICs

def plotBootstrapTimeseries(BS, overlay=None, overlay_label=None, overlay_lim=[None, None], save=False, title=None, ax=None, cbar_max=500, log=True, bins=None, setbad=0):
    import matplotlib.pyplot as plt
    from copy import copy
    import numpy as np
    import matplotlib as mpl

    if isinstance(bins, type(None)):
        bins= np.linspace(0, 1, 30)
    else:
        pass
    if isinstance(ax, type(None)):
        fig, ax = plt.subplots(figsize=(15, 3))
    else:
        pass
    cmap = copy(plt.cm.plasma)
    cmap.set_bad(cmap(setbad))
    if log:
        h = ax.hist2d((np.resize(np.arange(BS.shape[0]).flatten(), BS.shape[::-1]).T).flatten(), BS.flatten(), bins=[np.arange(BS.shape[0]), bins], norm=mpl.colors.LogNorm(vmax=cbar_max), cmap=cmap)
    else:
        h = ax.hist2d((np.resize(np.arange(BS.shape[0]).flatten(), BS.shape[::-1]).T).flatten(), BS.flatten(), bins=[np.arange(BS.shape[0]), bins], vmax=cbar_max, cmap=cmap)

    if isinstance(overlay, type(None)):
        pass
    else:
        ax_overlay=ax.twinx()
        ax_overlay.plot(np.arange(len(overlay))+0.5, overlay, 'w')
        ax_overlay.set_ylabel(overlay_label)
        ax_overlay.set_ylim(overlay_lim)
    cbar = plt.colorbar(h[3], extend="max", ax=ax)
    cbar.ax.set_ylabel("# of values")
    ax.set_ylabel("Coherence Magnitude")
    ax.set_xlabel("Interferogram index")
    ax.set_title(title)
    if save:
        plt.savefig(title + ".jpg", dpi=300)
        
    return ax, h

def plotBootstrapTimeseriesBox(BS, overlay=None, overlay_label=None, overlay_lim=[None, None], save=False, title=None, ax=None, cbar_max=500, log=True, bins=None, setbad=0):
    import matplotlib.pyplot as plt
    from copy import copy
    import numpy as np
    
    if isinstance(bins, type(None)):
        bins= np.linspace(0, 1, 30)
    else:
        pass

    if isinstance(ax, type(None)):
        fig, ax = plt.subplots(figsize=(15, 3))
    else:
        pass
    cmap = copy(plt.cm.plasma)
    cmap.set_bad(cmap(setbad))
    if log:
        h = ax.hist2d((np.resize(np.arange(BS.shape[0]).flatten(), BS.shape[::-1]).T).flatten(), abs(BS).flatten(), bins=[np.arange(BS.shape[0]), bins], norm=mpl.colors.LogNorm(vmax=cbar_max), cmap=cmap)
    else:
        h = ax.hist2d((np.resize(np.arange(BS.shape[0]).flatten(), BS.shape[::-1]).T).flatten(), abs(BS).flatten(), bins=[np.arange(BS.shape[0]), bins], vmax=cbar_max, cmap=cmap)

    # print (len(h))
    if isinstance(overlay, type(None)):
        pass
    else:
        ax_overlay=ax.twinx()
        ax_overlay.plot(np.arange(len(overlay))+0.5, overlay, 'w')
        ax_overlay.set_ylabel(overlay_label)
        ax_overlay.set_ylim(overlay_lim)
    cbar = plt.colorbar(h[3], extend="max", ax=ax)
    cbar.ax.set_ylabel("# of values")
    ax.set_ylabel("Coherence Magnitude")
    ax.set_xlabel("Interferogram index")
    ax.set_title(title)
    if save:
        plt.savefig(title + ".jpg", dpi=300)
        
    return ax, h

def box(data, ax=None):
    import numpy as np
    import matplotlib.pyplot as plt
    
    med = np.median(data, axis=1)
    lq = np.quantile(data, 0.25, axis=1)
    uq = np.quantile(data, 0.75, axis=1)
    
    iqr = uq-lq
    
    up = uq + 1.5*iqr #np.max(data, axis=1)
    lo = lq - 1.5*iqr #np.min(data, axis=1)
    
    outliers = ~(data < up[:, np.newaxis]) & ~(data > lo[:, np.newaxis])
    
    if isinstance(ax, type(None)):
        fig, ax = plt.subplots(figsize=(14, 5))
    else:
        pass
    
    for d in np.arange(data.shape[0]):
        ax.plot([d, d], [up[d], lo[d]], color="black", ls="-", marker="_", markersize=10)
        ax.scatter(np.ones(np.sum(outliers[d]))*d, data[d][outliers[d]], marker="o", facecolor='none', color="black")
    ax.plot(np.arange(data.shape[0]), lq, marker=10, markersize=10, color="black")
    ax.plot(np.arange(data.shape[0]), uq, marker=11, markersize=10, color="black")
    ax.set_xlim([0, data.shape[0]])
    
    return ax, [up, uq, med, lq, lo, outliers]

def formatComplex(X):
    import numpy as np
    
    return np.dstack((X.real, X.imag))

def formatComplexPolar(X):
    import numpy as np
    
    return np.dstack((np.angle(X), abs(X)))[0]

def polar2Dscat(X, ax=None, **kwargs):
    import numpy as np
    import matplotlib.pyplot as plt
    # theta, r = formatComplexPolar(BS[175, :]).T
    theta, r = formatComplexPolar(X).T
    
    if isinstance(ax, type(None)):
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), subplot_kw={'projection': 'polar'})
    else: 
        pass

    # print (Theta.shape, R.shape)
    # ax.contour(H.T,extent=[r_edges[0],r_edges[-1],theta_edges[0],theta_edges[-1]])
    # ax.contour(H.T)
#     print (type(kwargs['s']))

    p = ax.scatter(theta, r, **kwargs)
    if "c" in kwargs:
        cbar = plt.colorbar(p)
    else:
        pass
    
    # ax.grid()
#     ax.set_ylim([0, ylim])
    plt.tight_layout()
    return ax


def interactiveCoherence(BS, animate=False, save_fn = "movie", overlay=None, overlay_label=None, overlay_lim=[None, None], cbar_max=500, hist_max=300):
    import matplotlib.pyplot as plt
    import numpy as np
    import ipywidgets as widgets

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(223, projection="polar")
    ax3 = fig.add_subplot(224)

    plotBootstrapTimeseries(abs(BS), ax=ax1, overlay=overlay, overlay_label=overlay_label, overlay_lim=overlay_lim, cbar_max=cbar_max);

    vline = ax1.axvline(x=0, color="white", alpha=0.75)
    # ax1.set_xlim([0, 279])
    theta, r = formatComplexPolar(BS[0]).T
    line, = ax2.plot(theta, r, 'r.', markersize=3)
    ax2.set_ylim([0, 1])
    ax2.set_title("Complex Coherence")
    HIST_BINS = np.linspace(0, 1, 30)
    data = abs(BS[0])
    n, _ = np.histogram(data, HIST_BINS)
    _, _, bar_container = ax3.hist(data, bins=HIST_BINS, lw=1,
                                  ec="white", fc="green", alpha=0.5)
    ax3.set_ylim([0, hist_max])
    plt.tight_layout()
    style = {'description_width': 'initial'}
    ax3.set_title("Coherence magnitude")

    def update(epoch=widgets.IntSlider(min=0, max=BS.shape[0], step=1, value=0)):
        vline.set_xdata(int(epoch))

        # Update the polar plot
        theta, r = formatComplexPolar(BS[int(epoch)]).T
        line.set_data((theta, r))
        fig.suptitle(f"{epoch = }")

        # Update the histogram
        data = abs(BS[int(epoch)])
        n, _ = np.histogram(data, HIST_BINS)
        for count, rect in zip(n, bar_container.patches):
            rect.set_height(count)

        fig.canvas.draw_idle()

    plt.subplots_adjust(top=0.9, left=0.1, right=0.9)

    if animate:
        ani = animation.FuncAnimation(
                fig, update, interval=500, blit=True, save_count=278)
        if isinstance(save_fn, str):
            ani.save(f"{save_fn}.mp4", dpi=400)
        else:
            pass
    else:
        widgets.interact(update)#, limit=limit_case);
        
        
def kurtskew(BS, plot_kurt=True, plot_skew=True, plot_std=False):
    import matplotlib.pyplot as plt
    from scipy.stats import kurtosis, skew
    import numpy as np
    fig, ax1 = plt.subplots(figsize=(14, 6))
    fig.subplots_adjust(right=0.75)
    s = skew(abs(BS), axis=1)
    k = kurtosis(abs(BS), axis=1)
    ax = [ax1]
    
    if plot_kurt:
        ax1.plot(k, label="Kurtosis", alpha=0.7, color="navy")
        ax1.set_xlabel("Interferogram index", fontsize=14)
        ax1.set_ylabel("Kurtosis", color="navy", fontsize=14, alpha=0.7)
    
    if plot_skew:
        ax2 = ax1.twinx()
        ax2.plot(s, label="Skewness", color="red", ls=':')
        ax2.set_ylabel("Skewness", color="red", fontsize=14, alpha=0.7)
        ax.append(ax2)
        
    # ax1.legend(loc="upper left")
    # ax2.legend(loc="upper right")

    if plot_std:
        ax3 = ax1.twinx()
        ax3.spines.right.set_position(("axes", 1.2))
        ax3.plot(np.std(abs(BS), axis=1), color="green", alpha=0.7)
        ax3.set_ylabel("STD", color="green", fontsize=14, alpha=0.7)
        ax.append(ax3)
    
    return fig, ax

def simulate(ij1, ij2, n_change, phase, SHP_i, SHP_j, d):
    import numpy as np
    
    i1, j1 = ij1 
    i2, j2 = ij2
    
    i_, j_, sib_i, sib_j, ph_change = flatWrapper(phase, (i1, j1), SHP_i, SHP_j)
    a = abs(ph_change)
    _, _, _, _, ph_change_ = flatWrapper(phase, (i2, j2), SHP_i, SHP_j)
    if isinstance(n_change, int):
        ph_change[d[0]:d[1], -n_change:, 0] = ph_change_[d[0]:d[1], -n_change:, 0]
    elif isinstance(n_change, type(np.arange(2))):
        for n in n_change:
            ph_change[d[0]:d[1], n, 0] = ph_change_[d[0]:d[1], n, 0]

#     ph_change = (ph_change/abs(ph_change)) * a
    
#     ph_change = ph_change * 
    
    return i_, j_, sib_i, sib_j, ph_change

def damped(t, A, gamma, ohm, phi):
    return A*np.exp(-gamma * t)*np.cos(ohm*t + phi)

def poly(args):
    if len(args) == 2:
        m, c = args
        return lambda x: m*x + c
    elif len(args) == 3:
        a, b, c = args
        return lambda x: a*(x**2) + b*x + c
    elif len(args) == 4:
        a, b, c, d = args
        return lambda x: a*(x**3) + b*(x**2) + c*x + d
    elif len(args) == 5:
        a, b, c, d, e = args
        return lambda x: a*(x**4) + b*(x**3) + c*(x**2) + d*x + e
        
def doPCA(data, n_components, plot=False, scale=True):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    if scale:
        data = StandardScaler().fit_transform(data)

        pca = PCA(n_components=n_components)

        PC = pca.fit_transform(data)
    else:
        pca = PCA(n_components=n_components)

        PC = pca.fit_transform(data)
    
    if plot:
        for d in PC.T:
            plt.figure(figsize=(10, 6))
            plt.plot(d)
        
    return PC

def sarles(BS, axis=1):
    from scipy.stats import kurtosis, skew

    k = kurtosis(abs(BS), axis=axis, fisher=False, bias=False)
    s = skew(abs(BS), axis=axis)
    return (s**2 + 1)/k#, k, s

def AICdiff(X):
    """
    This fits a GMM with one component, and another with two 
    components to the data. Then finds the AIC diff between 
    them to determine which is the better fit. 
    """
    from sklearn.mixture import GaussianMixture
    
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    else:
        pass
    
    model1 = GaussianMixture(n_components=1, init_params='kmeans')
    model1.fit(X)
    AIC1 = model1.aic(X)

    model2 = GaussianMixture(n_components=2, init_params='kmeans')
    model2.fit(X)
    AIC2 = model2.aic(X)
    
    AICdiff = AIC2 - AIC1
    
    return AICdiff

def complexCoherenceAIC_TS(BS):
    from tqdm import tqdm
    from sklearn.mixture import GaussianMixture
    import numpy as np
    AICs = []
    for coherences in tqdm(BS):
#         print (coherences.shape)
        X = formatComplex(coherences)[0]
#         print (X.shape)
#         print (X.shape)
        GMM1 = GaussianMixture(n_components=1).fit(X)
        GMMs = [GaussianMixture(n_components=n).fit(X) for n in np.arange(2, 4, 1)]
        
        AICs.append(np.min([GMM.aic(X)-GMM1.aic(X) for GMM in GMMs]))# - GMM1.aic(X))
        # BICs.append(GMM2.bic(X) - GMM1.bic(X))
        
    return AICs

def magnitudeCoherenceAIC_TS(BS):
    import numpy as np
    from sklearn.mixture import GaussianMixture
    
    AICs = []
    BICs = []
    for coherences in BS:
        coherences_ = abs(coherences.reshape(-1, 1))
        GMM1 = GaussianMixture(n_components=1).fit(coherences_)
        GMMs = [GaussianMixture(n_components=n).fit(coherences_) for n in np.arange(2, 6, 1)]
        AICs.append(np.min([GMM.aic(coherences_) for GMM in GMMs]) - GMM1.aic(coherences_))
        
        # AICs.append(GMM2.aic(coherences_) - GMM1.aic(coherences_))
        # BICs.append(GMM2.bic(coherences_) - GMM1.bic(coherences_))
        
    return AICs

def polarInteractive(X, ylim=1):
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    
    theta, r = formatComplexPolar(X[0, :]).T
    line, = ax.plot(theta, r, 'r.', markersize=3, alpha=0.2)
    ax.set_ylim([0, ylim])

    def update(epoch=widgets.IntSlider(min=0, max=X.shape[0]-1, step=1, value=0)):
        # Update the polar plot
        theta, r = formatComplexPolar(X[int(epoch), :]).T
        line.set_data((theta, r))
        fig.suptitle(f"{epoch = }")

        fig.canvas.draw_idle()

    widgets.interact(update)
    
def polarInteractiveClusters(X, ylim=1, clusters=None):
    """
    
    Plot bootstrapped coherence values in polar coordinates by their cluster value (based on the
    Kmeans clustering with n corresponding to the AIC value). 
    
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    
    if isinstance(clusters, type(None)):
        clusters = np.zeros(X.shape)
    else:
        pass
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    
    
    theta, r = formatComplexPolar(X[0, :]).T

    lines = [ax.plot(theta[clusters == c], r[clusters == c], '.', markersize=3)[0] for c in set(clusters)]
#     line, = ax.plot(theta, r, 'r.', markersize=3)
    ax.set_ylim([0, ylim])

    def update(epoch=widgets.IntSlider(min=0, max=X.shape[0]-1, step=1, value=0)):
        # Update the polar plot
        theta, r = formatComplexPolar(X[int(epoch), :]).T
        line.set_data((theta, r))
        fig.suptitle(f"{epoch = }")

        fig.canvas.draw_idle()

    widgets.interact(update)
    
def transformPhaseCentre(ph_1, ph_2):
    """
    Returns ph_2 but with same phase centre as ph_1
    """
    import numpy as np
    ph_diff = np.sum(ph_1, axis=1) * np.sum(ph_2, axis=1).conjugate()
    ph_diff = ph_diff/abs(ph_diff)

    return ph_2 * ph_diff[:, :, None]

def hdfRead(file, header=False):
    """Short function to extract data from the file. Returns False if no data is found
    for that header. """
    import h5py as h5
    import numpy as np
    
    with h5.File(file) as f:
        if header:
            try:
                data = np.asarray(f[header])

            except KeyError:
                print (f'No dataset for {header} found. Set to False. \n')
                headers = list(f.keys())
                print ('Possible headers: \n', headers)
                data = False
        else:
            data = [np.asarray(f[str(h)]) for h in list(f.keys())]
    return data


def quickCoherence(im1,
                   im2,
                   sibling_i,
                   sibling_j,
                   true_index=True,
                   subset_size=100,
                   desc="Estimating coherence",
                   progress = True):

    import numpy as np
#     import numexpr as ne
    from tqdm import tqdm

    assert len(im1.shape) == 2, "Image must be 2 dimensions."
    assert len(im2.shape) == 2, "Image must be 2 dimensions."
    assert im1.shape == im2.shape, "Images must be the same shape."

    out = np.zeros(im1.shape, dtype=np.complex64)

    n = (im1.shape[0] // subset_size + 1)
    m = (im1.shape[1] // subset_size + 1)

    subset_i = np.arange(0, n * subset_size, subset_size)
    subset_i[-1] = im1.shape[0]

    subset_j = np.arange(0, m * subset_size, subset_size)
    subset_j[-1] = im1.shape[1]

    subset_i_ix = np.repeat(np.arange(0, n - 1, 1), m - 1)
    subset_j_ix = np.repeat(np.arange(0, m - 1, 1)[:, np.newaxis],
                            n - 1,
                            axis=1).T.flatten()
    
    for pq in tqdm(np.dstack((subset_i_ix, subset_j_ix))[0], desc=desc, disable=~progress):
        p, q = pq
        i_start = subset_i[p]
        i_end = subset_i[p + 1]
        j_start = subset_j[q]
        j_end = subset_j[q + 1]

        subset_sib_i = sibling_i[:, i_start:i_end, j_start:j_end]
        subset_sib_j = sibling_j[:, i_start:i_end, j_start:j_end]

        im1_subset, im2_subset = np.full(subset_sib_j.shape,
                                         np.nan,
                                         dtype=np.complex64), np.full(
                                             subset_sib_j.shape,
                                             np.nan,
                                             dtype=np.complex64)

        mask = ~np.isnan(subset_sib_i) * ~np.isnan(subset_sib_j)

        im1_subset[mask] = im1[subset_sib_i[mask].astype(int),
                               subset_sib_j[mask].astype(int)]
        im2_subset[mask] = im2[subset_sib_i[mask].astype(int),
                               subset_sib_j[mask].astype(int)]

        num = np.nansum(im1_subset * im2_subset.conjugate(), axis=0)
        den = np.sqrt(
            np.nansum(abs(im1_subset)**2, axis=0) *
            np.nansum(abs(im2_subset)**2, axis=0))

        out[i_start:i_end, j_start:j_end] = num / den
    return out

def circGaussian(n_siblings, locr, loci, cohs):
    import numpy as np
#     shape=(280, 60, 1)
    sigma_CR = lambda coherence: (1 - abs(coherence)**2)/(2*abs(coherence**2))
    signal = np.empty((len(locr), n_siblings, 1), dtype=np.complex64)
    
    for i, locr_, loci_, coh in zip(np.arange(len(locr)), locr, loci, cohs):
        
        scale = np.sqrt(sigma_CR(coh))
        
        Re = np.random.normal(loc=locr_, scale=scale, size=n_siblings)
        Im = np.random.normal(loc=loci_, scale=scale, size=n_siblings)
        
        signal[i, :, 0] = Re + 1j*Im
        
        signal[i, 0, 0] = np.mean(signal[i, :, 0])
        
#     Re[:, 0, 0] = np.mean(Re, axis=1)[:, 0]
#     Im[:, 0, 0] = np.mean(Im, axis=1)[:, 0]
        
#     signal = Re + 1j*Im
    
    return signal

def confusion_matrix_display(test_labels, predictions, vmin=None, vmax=None, ax=None):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    
    cm = confusion_matrix(test_labels, predictions, labels=[0.0, 1.0])
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        pass
    
    p = ax.imshow(cm, vmin=vmin, vmax=vmax)
    
    ax.set_xticks([0, 1], [0, 1])
    ax.set_yticks([0, 1], [0, 1])    
    
    ax.set_ylabel("True labels", fontsize=16)
    ax.set_xlabel("Predicted labels", fontsize=16)
    
    ax.text(0, 0, cm[0, 0], backgroundcolor="White", color="Black", fontsize=18, ha="center", weight="bold")
    ax.text(0, 1, cm[1, 0], backgroundcolor="White", color="Black", fontsize=18, ha="center", weight="bold")
    ax.text(1, 0, cm[0, 1], backgroundcolor="White", color="Black", fontsize=18, ha="center", weight="bold")
    ax.text(1, 1, cm[1, 1], backgroundcolor="White", color="Black", fontsize=18, ha="center", weight="bold")
    
    plt.colorbar(p, ax=ax)
    
    
def runRF(train_features, train_labels, test_features, test_labels, vmin=None, vmax=None, save_fn=False, grid_search=False):
    from sklearn.ensemble import RandomForestClassifier    
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay
    import pandas as pd
    import numpy as np
    
    rf = RandomForestClassifier(max_depth=8, n_estimators=250, random_state=1)
    rf.fit(train_features, train_labels)
    
    predictions = rf.predict(test_features)
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    confusion_matrix_display(test_labels, predictions, vmin=vmin, vmax=vmax, ax=ax[0])
    
    a = RocCurveDisplay.from_estimator(rf, test_features, test_labels, ax=ax[1])
    
    ax[1].plot([0, 1], [0, 1], 'r-')
    ax[1].set_aspect(1)
    ax[1].set_ylim([0, 1])
    ax[1].set_xlim([0, 1])    

    importance = pd.DataFrame(np.dstack((np.array(list(train_features)), rf.feature_importances_))[0], columns=["Feature name", "Feature importance"])
    
    if save_fn:
        import joblib

        joblib.dump(rf, f"{save_fn}.joblib")
        
    return rf, predictions, (a.fpr, a.tpr), importance

def assignWeights(data, mask=None, bins=50):
    """
    Generate weightings based on the coherence at that point (can be done with 2d or 3d data array.)
    
    data: 2d or 3d array to assign weights on.
    mask: 2d or 2d boolean array. If element is False the weight is set to 0.
    bins: int or 1d numpy array to partition input to be able to assign weights.
    
    """    
    import numpy as np
    
    if isinstance(type(bins), int):
        bins = np.linspace(0, 1, bins)
    else:
        pass
    
    if mask is None:
        mask = np.ones(data.shape, dtype=bool)
    else:
        pass
    
    if len(data.shape) == 2:
        indices = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    elif len(data.shape) == 3:
        indices = np.mgrid[0:data.shape[0], 0:data.shape[1], 0:data.shape[2]]
        
    dens, bins = np.histogram(data, bins, density=True)
    
    weights = dens.copy()
    
    weights[weights > 0] = (np.max(weights[weights > 0])/weights[weights > 0])/np.sum(np.max(weights[weights > 0])/weights[weights > 0])

    bin_numbers = np.digitize(data.flatten(), bins[1:-1])
    
    value_weights = weights[bin_numbers].reshape(data.shape)
    
    value_weights[~mask] = 0
    
    value_weights = value_weights/np.sum(value_weights)
    
    return value_weights

def manipulateSiblings(coord1, coord2, d1, d2, data, sib_i, sib_j, n_change, smooth=True, seed=None):
    """
    coord1:  (1, 2) np.array, coordinated for pixel's siblings to be changed
    coord2:  (1, 2) np.array, coordinates for pixel's siblings to be changed to
    
    d1:       int or None, date index 1
    d2:       int or None, date index 2
    
    data:     3d np.array, Interferogram dataset
    sib_i:    3d np.array, siblings in axis 1 of data
    sib_j:    3d np.array, siblings in axis 2 of data
    
    n_change: int or None, number of siblings to be manipulated
    smooth:   bool, whether to smooth the data in time
    
    """
    import random
    import simulationFuncs as SF
    import numpy as np

    # Get phase of pixels 1 and 2
    _, _, sib_i_, sib_j_, ph_1 = flatWrapper(data, coord1, sib_i, sib_j)
    _, _, _, _, ph_2 = flatWrapper(data, coord2, sib_i, sib_j)
    
    # Transform the phase centre
    ph_2 = transformPhaseCentre(ph_1, ph_2)
    
    # Check if n_change is None, if so generate random number
    if n_change is None:
        n_change = np.random.randint(low=np.round(np.min(np.array([ph_1.shape[1], ph_2.shape[1]]))*0.25), high=np.min(np.array([ph_1.shape[1], ph_2.shape[1]])), size=1)[0]
        # n_change = np.random.randint(low=1, high=np.min(np.array([ph_1.shape[1], ph_2.shape[1]])), size=1)[0]
    else:
        pass
    
    np.random.seed(seed)
    
    indices1 = np.array(random.sample(range(ph_1.shape[1]), n_change)) # Indices to be changed in ph_1
    
    if np.isclose(coord1, coord2).all():
        indices2 = indices1.copy()
    else:
        indices2 = np.array(random.sample(range(ph_2.shape[1]), n_change)) # Indices to be changed into of ph_2
        
    # Check if date indices are passed or None
    if d1 is None and d2 is None:
        d1 = np.random.randint(low=1, high=data.shape[0]-1, size=1)[0]
        d2 = np.random.randint(low=1, high=data.shape[0]-1, size=1)[0]
    else:
        pass
    
    ph_change = ph_1.copy()[d1-1:d1+1]
    # print (ph_change.shape)
    # print (ph_1.shape, ph_2.shape, d1, d2)
    actual_coherence = coherenceTimeseries(0, 0, sib_i_, sib_j_, ph_change)
    
    ph_change[np.ix_(np.arange(ph_change.shape[0]), indices1)] = ph_2[np.ix_(np.array([d2-1, d2]), indices2)]
    apparent_coherence = coherenceTimeseries(0, 0, sib_i_, sib_j_, ph_change)
    
    return ph_change, actual_coherence, apparent_coherence

def jackknife(i, j, sib_i, sib_j, data, bias = False):
    """
    i, j: coordinates of the pixels
    sib_i, sib_j: arrays of the coordinates of the siblings
    data: complex interferogram values
    
    """
    import numpy as np
    
    n_siblings = sib_i.shape[0]
    coherences = np.empty(n_siblings, dtype=np.complex64)
    
    for exclude_ix in np.arange(n_siblings):
        ix = np.r_[:exclude_ix, exclude_ix+1:n_siblings]
        coherence = coherenceTimeseries(i, j, sib_i[ix, :, :], sib_j[ix, :, :], data)
        # print (coherence)
        coherences[exclude_ix] = coherence
        
    if bias:
        actual = coherenceTimeseries(i, j, sib_i, sib_j, data)
        jack_av = np.mean(abs(coherences))
        bias = (n_siblings - 1)*(jack_av - abs(actual))
        
        return coherences, bias[0]
    else:
        return coherences, None
    
def generateSims(coords1, coords2, d1s, d2s, data, sib_i, sib_j, n_changes):
    """
    
    Run through all the input to generate a series of simulations based on the real data. 
    
    coords1: (n, 2) np.array, list of coordinates
    coords2: (n, 2) np.array, list of coordinates
    
    d1s: (n,) np.array, list of dates
    d2s: (n,) np.array, list of dates
    
    data: 3d np.array, stack of interferograms
    sib_i: 3d np.array, coordinates of siblings
    sib_j: 3d np.array, coordinates of siblings
    
    n_change: (n,) np.array, list of numbers of siblings to be changed
    """
    import numpy as np
    from tqdm import tqdm
    
    abs_data = abs(data)
    
    if d1s is None:
        d1s = np.full(coords1.shape[0], fill_value=None)
    else: pass

    if d2s is None:
        d2s = np.full(coords2.shape[0], fill_value=None)
    else: pass

    if n_changes is None:
        n_changes = np.full(coords2.shape[0], fill_value=None)
    else: pass
    
    out = np.empty((coords1.shape[0], 12), dtype=object)
    for n, (coord1, coord2, d1, d2, n_change) in enumerate(tqdm(zip(coords1, coords2, d1s, d2s, n_changes), total=coords1.shape[0])):
        coh_diff = 0
        if (coord1 == coord2).all():
            ph_change, actual_coherence, apparent_coherence = manipulateSiblings(coord1, coord2, d1, d2, data, sib_i, sib_j, n_change, smooth=True, seed=None)
        else:
            count = 0
            while coh_diff < 0.2:
                ph_change, actual_coherence, apparent_coherence = manipulateSiblings(coord1, coord2, d1, d2, data, sib_i, sib_j, n_change, smooth=True, seed=None)
                coh_diff = abs(actual_coherence - apparent_coherence)
                if count == 20:
                    break
                else:
                    count += 1
        
        sib_i_ = np.arange(ph_change.shape[1])[:, None, None]
        sib_j_ = np.zeros(ph_change.shape[1])[:, None, None]
        
        jackknifed, jk_bias = jackknife(0, 0, sib_i_, sib_j_, ph_change, bias=True)
        
        assert d1 > 5, "d1 must be greater than 5"
        
        # amp_mean_temporal = np.mean(abs_data[d1 - 5:d1+1, coord1[0], coord1[1]])
        # amp_mean = np.mean(abs(np.product(ph_change, axis=0)))
        amp_mean = np.mean(abs(ph_change[0]))
        amp_std  = np.std(abs(np.product(ph_change, axis=0)))
        amp_px   = abs(np.product(ph_change, axis=0))[0, 0]
        # int_amp = abs(ph_change)[0, :, 0]*abs(ph_change)[1, :, 0]
        # max_amp_diff = np.nanmax(abs(image_with_siblings[0, 1:]) - abs(image_with_siblings[1, 1:]), axis=0).flatten()
        
        max_amp_diff = np.nanmax(abs(ph_change)[1, 1:, 0] - abs(ph_change)[0, 1:, 0])
        poi_diff = abs(ph_change)[1, 0, 0] - abs(ph_change)[0, 0, 0]
        
        out[n, :] = np.array([coord1[0], coord1[1], ph_change.shape[1], np.nanstd(jackknifed), jk_bias, amp_mean, amp_std, amp_px, poi_diff, max_amp_diff, actual_coherence, apparent_coherence], dtype=object)
        
    return out

def fillresize(arr, ax1=100, dtype="np.complex64"):
    """
    For an object array with odd sized dimensions, fill it out with nan values
    """
    import numpy as np
    dtypes = {"np.complex64":np.complex64, "int":int, "float":float}
    dtype = dtypes[dtype]
    
    arr_out = np.full((arr.shape[0], ax1), dtype=dtype, fill_value=np.nan)
    for i, a in enumerate(arr):
        arr_out[i, 0:a.size] = a
        
    return arr_out

def fillresize1d(arr, ax0=100, dtype="np.complex64"):
    """
    For an object array with odd sized dimensions, fill it out with nan values
    """
    import numpy as np
    
    dtypes = {"np.complex64":np.complex64, "int":int, "float":float}
    
    dtype = dtypes[dtype]
    
    arr_out = np.full((ax0), dtype=dtype, fill_value=np.nan)
    
    arr_out[0:arr.size] = arr
        
    return arr_out

def meanDiff(arr, kernel_size=11):
    
    from scipy import signal
    import numpy as np
    
    smoothed = signal.fftconvolve(arr, (np.ones(kernel_size)/kernel_size)[:, None, None], mode='same')
    
    meandiff = arr - smoothed
    
    return meandiff

def sampleUniform(arr, mask=None, bins=50, N=1000):
    import random
    import numpy as np
    from tqdm import tqdm
    
    arr_flat = arr.flatten()
    
    p = np.arange(arr.size,dtype=float)
    
    if mask is None:
        pass
    else:
        mask_flat = mask.flatten()
        p = p[mask_flat]
    
    bins = np.linspace(np.min(arr[mask]), np.max(arr[mask]), bins+1)
    
    N_each_bin = int(np.ceil(N/(bins.size-2)))
    
    bin_numbers = np.digitize(arr[mask].flatten(), bins, right=True)
    
    selected_values = np.full((np.max(bin_numbers), N_each_bin), dtype=float, fill_value=np.nan)
    
    for bin_number in tqdm(np.arange(1, np.max(bin_numbers)+1), desc=f"Looping through bins ({N_each_bin = })"):
        if np.sum((bin_numbers == bin_number)) > N_each_bin:
            picked_p = random.sample(list(p[(bin_numbers == bin_number)]), N_each_bin)
        else:
            picked_p = p[(bin_numbers == bin_number)]
            
        selected_values[bin_number - 1, :np.array(picked_p).size] = picked_p
        
    selected_values = selected_values[(~np.isnan(selected_values) * (selected_values > 0))].flatten()
    
    np.random.shuffle(selected_values)
    
    d, i, j = np.unravel_index(selected_values[:int(N)].astype(int), shape=arr.shape)
    
    return d, i, j

def test_training_pie(train_labels, test_labels):
    import matplotlib.pyplot as plt
    import numpy as np
    
    def func(pct, N):
        return f"{pct:.1f}%\n{pct*N/100:.0f}"
    
    cmap = plt.get_cmap("tab20c")
    colors = cmap(np.array([1, 2, 5, 6]))
    
    plt.figure(figsize=(12, 12))

    wedges, texts, autotexts = plt.pie(x = [np.sum(train_labels == 1), np.sum(train_labels == 0), 
                                            np.sum(test_labels == 1), np.sum(test_labels == 0)],
                                       labels = ["True (training)", "False (training)", 
                                                 "True (test)", "False (test)"], colors=colors, 
                                       autopct=lambda pct: func(pct, train_labels.shape[0]), 
                                       textprops = dict(fontsize=20))
    
def falseNaN(arr, keep=True):
    import numpy as np
    
    arrnan = np.ones(arr.shape)
    arrnan[~(arr == keep)] = np.nan
    return arrnan

def jackknifeImage(ifg1, ifg2, sib_i, sib_j):
    from multiprocessing import Pool
    import numpy as np
    from tqdm import tqdm
    from functools import partial

    def jack(s):
        mask = np.ones(sib_i.shape, dtype=bool)
        mask[s] = False
        out = quickCoherence(ifg1, ifg2, sib_i[mask].reshape((sib_i.shape[0]-1, *sib_i.shape[1:])), sib_j[mask].reshape((sib_j.shape[0]-1, *sib_j.shape[1:])), progress=False)
        
        return s, out
    
    # out = np.empty(sib_i.shape, dtype=np.complex64)
    # for s in tqdm(np.arange(out.shape[0]), desc="Jackknife resampling"):
    
    pool = Pool(6)
    outputs = pool.map(jack, range(sib_i.shape[0]))
    
    print (outputs)

        # mask = np.ones(out.shape[0], dtype=bool)
        # mask[s] = False
        # out[s] = quickCoherence(ifg1, ifg2, sib_i[mask].reshape((sib_i.shape[0]-1, *sib_i.shape[1:])), sib_j[mask].reshape((sib_j.shape[0]-1, *sib_j.shape[1:])), progress=False)
        
    out[~(~np.isnan(sib_i)*~np.isnan(sib_j))] = np.nan + 1j*np.nan
    
    return out

def coherence_exclude(ifg1, ifg2, sib_i, sib_j, ix):
    """
    Function to calculate the coherence between two images using while 
    excluding a slice of the sibling arrays (used for jacknknife resampling).
    """
    import numpy as np
    mask = np.ones(sib_i.shape, dtype=bool)
    mask[ix] = False
    
    coh = quickCoherence(ifg1, ifg2, sib_i[mask].reshape((sib_i.shape[0]-1, *sib_i.shape[1:])), sib_j[mask].reshape((sib_j.shape[0]-1, *sib_j.shape[1:])), progress=False)

    return ix, coh

def jackknifeMulticore(ifg1, ifg2, sib_i, sib_j, n_cores=6):
    """ 
    Function to do multicore jackknife resampling of coherence estimation. 
    """
    from functools import partial
    from multiprocessing import Pool
    import numpy as np
    
    print ("Starting jackknife multicore. ")
    pool = Pool(n_cores)
    
    results = pool.map(partial(coherence_exclude, ifg1, ifg2, sib_i, sib_j), np.arange(sib_i.shape[0]))
    
    out = np.empty((sib_i.shape), dtype=np.complex64)

    for result in results:
        s, arr = result
        out[s] = arr

    out[np.logical_or(np.isnan(sib_i), np.isnan(sib_j))] = np.nan

    return out    

def generateMetricsIFG(ifg1, ifg2, sib_i, sib_j):
    import numpy as np
    import sys
    # out = np.empty((ifg1.size, 10), dtype=object)
    
    mask = (~np.isnan(sib_i) * ~np.isnan(sib_j))
    
    n_siblings = np.sum(mask, axis=0)
    
    image_with_siblings = np.full((2, *sib_i.shape), fill_value=np.nan+1j*np.nan)

    # print (ifg1[sib_i[mask].astype(int), sib_j[mask].astype(int)].shape)

    image_with_siblings[0, mask] = ifg1[sib_i[mask].astype(int), sib_j[mask].astype(int)]
    image_with_siblings[1, mask] = ifg2[sib_i[mask].astype(int), sib_j[mask].astype(int)]
    
    # intAmp = abs(image_with_siblings[0])*abs(image_with_siblings[1])
    intAmp = abs(image_with_siblings[0])

    # max_amp_diff = np.nanmax(intAmp[1:] - intAmp[0], axis=0).flatten() # NEW
    # print (image_with_siblings.shape)
    print ("max_amp_diff")
    max_amp_diff = np.nanmax(abs(image_with_siblings[0, 1:]) - abs(image_with_siblings[1, 1:]), axis=0).flatten()
    print ("poi_diff")
    poi_diff = abs(image_with_siblings[0, 0]) - abs(image_with_siblings[1, 0])

    print ("Starting jackknife")
    # jackknifed = jackknifeImage(ifg1, ifg2, sib_i, sib_j)
    jackknifed = jackknifeMulticore(ifg1, ifg2, sib_i, sib_j)    
    print ("Completed jackknife")

    print ("mean_jackknifed")
    mean_jackknifed = np.nanmean(abs(jackknifed), axis=0)
    print ("coh")
    coh = quickCoherence(ifg1, ifg2, sib_i, sib_j)
    print ("bias")
    bias = (n_siblings-1)*(abs(mean_jackknifed) - abs(coh))
    
    i_, j_ = np.mgrid[0:ifg1.shape[0], 0:ifg2.shape[1]]
    
    # out = np.vstack((i_.flatten(), j_.flatten(), n_siblings.flatten(), np.nanstd(jackknifed, axis=0).flatten(), bias.flatten(), 
    #                  np.nanmean(intAmp, axis=0).flatten(), np.nanstd(intAmp, axis=0).flatten(), 
    #                  (abs(ifg1)*abs(ifg2)).flatten(), abs(coh).flatten(), abs(coh).flatten())).T
    print ("Making output metrics")
    out = np.vstack((i_.flatten(), 
                     j_.flatten(), 
                     n_siblings.flatten(), 
                     np.nanstd(jackknifed, axis=0).flatten(), 
                     bias.flatten(), 
                     np.nanmean(intAmp, axis=0).flatten(), 
                     np.nanstd(intAmp, axis=0).flatten(), 
                     max_amp_diff,
                     intAmp[0].flatten(),
                     poi_diff.flatten(),
                     abs(coh).flatten(), 
                     abs(coh).flatten())).T # NEW, -3 
    
    return out

def plotSibs(i, j, SHP_i, SHP_j, ax):
    
    mask = ~np.isnan(SHP_i[:, i, j])*~np.isnan(SHP_j[:, i, j])
    
    sib_i = SHP_i[mask, i, j]
    sib_j = SHP_j[mask, i, j]
    
    a = np.full(SHP_i.shape[1:], fill_value=np.nan)
    a[sib_i.astype(int), sib_j.astype(int)] = 1
    
    plt.scatter(sib_j[1:]+0.5, sib_i[1:]+0.5, color='magenta', s=3)
    plt.scatter(j+0.5, i+0.5, color='black', marker='*', s=30, ec="magenta")
    
    # ax.pcolormesh(a)
    
    return ax, a

def sampleUniform2D(arr1, arr2, mask=None, bins=30, N=1000):
    import random
    import numpy as np
    from tqdm import tqdm
    
    if mask is not None:
        mask_flatten = mask.flatten()
    else:
        pass
    
    assert arr1.shape == arr2.shape, "arr1 and arr2 must be the same shape."
    
    arr1_flat = arr1.flatten()
    arr2_flat = arr2.flatten()
    
    min1, max1 = np.nanmin(arr1), np.nanmax(arr1)
    min2, max2 = np.nanmin(arr2), np.nanmax(arr2)
    
    bins1 = np.linspace(min1, max1, bins)
    bins2 = np.linspace(min2, max2, bins)
    
    bin_numbers1 = np.digitize(arr1_flat, bins=bins1, right=True)
    bin_numbers2 = np.digitize(arr2_flat, bins=bins2, right=True)
    
    N_each_bin = int(np.ceil(N/((bins1.size - 2)*(bins2.size - 2))))
    
    out = np.empty((np.max(bin_numbers1+1)*np.max(bin_numbers2+1), N_each_bin))
    
    p = np.arange(arr1.size,dtype=float)
    
    i = 0
    
    for b1 in tqdm(np.arange(np.max(bin_numbers1)+1)):
        for b2 in np.arange(np.max(bin_numbers2)+1):
            mask_bin = (bin_numbers1 == b1) & (bin_numbers2 == b2) & (mask_flatten)
            if np.sum(mask_bin) > N_each_bin:
                picked_p = random.sample(list(p[mask_bin]), N_each_bin)
            else:
                picked_p = random.sample(list(p[mask_bin]), np.sum(mask_bin))
                picked_p = fillresize1d(np.array(picked_p), ax0=N_each_bin, dtype="float")
                
            out[i, :] = picked_p
            
            i += 1
            
    out_flat = out.flatten()
    
    np.random.shuffle(out_flat)
    
    d, i, j = np.unravel_index(out_flat[~np.isnan(out_flat)][:int(N)].astype(int), shape=arr1.shape)
    
    return d, i, j

def postProcess_df(df_fn):
    df = pd.read_csv(df_fn)

    plt.figure(1)
    plt.hist(df["n_siblings"], bins=np.linspace(-0.5, 100.5, 101))
    plt.title("Number of siblings of each pixel used in test/training. ")

    plt.figure(2)
    plt.scatter(df["n_siblings"], abs(df["actual_coherence"] - df["apparent_coherence"]), s=1)
    plt.xlabel("Number of siblings of each pixel")
    plt.ylabel("Difference in coherence (from simulated change)")

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].scatter()

def filterPredictions(predictions, SHP_i, SHP_j, threshold):
    import numpy as np

    mask = ~np.isnan(SHP_i) & ~np.isnan(SHP_j)

    mask_siblings = np.empty_like(SHP_i)

    mask_siblings[mask] = predictions[(SHP_i[mask]).astype(int), (SHP_j[mask]).astype(int)]

    if threshold % 1 == 0:
        return mask_siblings, (np.nansum(mask_siblings, axis=0) > threshold) & predictions
    else:
        return mask_siblings, (np.nansum(mask_siblings, axis=0)/np.sum(mask, axis=0) > threshold) & predictions

def main():
    return None

if __name__ == "__main__":
    sys.exit(main())