import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
import glob

import torch, os



def plot_samples(testdata, s=None):
    m = []
    n = []
    for i, (x, y, data) in enumerate(testdata):
        m.append(x.numpy())
        n.append(y.numpy())
    m = np.concatenate(m, axis=0)
    n = np.concatenate(n, axis=0)
    # print("m:",m)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(26, 12))
    ax[0].axis('off');
    ax[1].axis('off')
    ax[0].set_title('Data', fontsize=24)
    # ax[0].hist2d(np.array(m), np.array(n), bins=256, range=[[-1.5, 2.5], [-2, 2]])
    ax[0].scatter(np.array(m), np.array(n))
    if s is not None:
        s = s.detach().numpy()
        ax[1].set_title('Samples', fontsize=24)
        # ax[1].hist2d(s[...,0], s[...,1], bins=256, range=[[-4, 4], [-4, 4]])
        ax[1].scatter(s[..., 0], s[..., 1])
    plt.show()

def plot_samples2(data, sample):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(26, 12))
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_title('x_dash', fontsize=24)
    ax[1].set_title('z_dash', fontsize=24)
    data = data.detach().numpy()
    sample = sample.detach().numpy()
    ax[0].scatter(data[..., 0], data[..., 1])
    ax[1].scatter(sample[..., 0], sample[..., 1])
    plt.show()


def plot_s(data, s=None, a=None, b=None):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(26, 12))
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_title(a, fontsize=24)

    data = data.detach().numpy()
    ax[0].scatter(data[..., 0], data[..., 1])
    if s is not None:
        s = s.detach().numpy()
        ax[1].set_title(b, fontsize=24)
        # ax[1].hist2d(s[...,0], s[...,1], bins=256, range=[[-4, 4], [-4, 4]])
        ax[1].scatter(s[..., 0], s[..., 1])
    plt.show()




def plot_final(data, sample,x_dash,z_dash,fname=None):
        m = []
        n = []
        for i, (x, y, data) in enumerate(data):
            m.append(x.numpy())
            n.append(y.numpy())
        m = np.concatenate(m, axis=0)
        n = np.concatenate(n, axis=0)
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(30, 20))
        ax[0,0].axis('off')
        ax[0,1].axis('off')
        ax[1,0].axis('off')
        ax[1,1].axis('off')
        ax[0, 0].set_title('z = f(X) (inverse transform)', fontsize=24)
        ax[0,1].set_title('z ~ p(z) (Samples from prior) ', fontsize=24)
        ax[1,0].set_title('X ~ p(X) (dataset)', fontsize=24)
        ax[1,1].set_title('X = g(z) (forward transform)', fontsize=24)
        data = data.detach().numpy()
        sample = sample.detach().numpy()
        x_dash = x_dash.detach().numpy()
        z_dash = z_dash.detach().numpy()
        ax[0,0].scatter(z_dash[..., 0], z_dash[..., 1])

        ax[0,1].scatter(sample[..., 0], sample[..., 1])
        ax[1, 0].scatter(np.array(m), np.array(n),color='red')
        ax[1, 1].scatter(x_dash[..., 0], x_dash[..., 1],color='red')
        if fname!= None:
           plt.savefig(fname,facecolor='white', edgecolor='none')
        plt.show()

def plot_cnf_animation(target_sample, t0, t1, viz_timesteps, p_z0, z_t1, z_t_samples, z_t_density, logp_diff_t):
    img_path = os.path.join('etc', 'cnf')

    for (t, z_sample, z_density, logp_diff) in zip(
            np.linspace(t0, t1, viz_timesteps),
            z_t_samples, z_t_density, logp_diff_t):
        fig = plt.figure(figsize=(12, 4), dpi=200)
        plt.tight_layout()
        plt.axis('off')
        plt.margins(0, 0)
        fig.suptitle(f'{t:.2f}s')

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_title('Target')
        ax1.get_xaxis().set_ticks([])
        ax1.get_yaxis().set_ticks([])
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title('Samples')
        ax2.get_xaxis().set_ticks([])
        ax2.get_yaxis().set_ticks([])
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_title('Log Probability')
        ax3.get_xaxis().set_ticks([])
        ax3.get_yaxis().set_ticks([])



        ax1.hist2d(target_sample[:,0],target_sample[:,1], bins=300, density=True,
                   range=[[-1.5, 1.5], [-1.5, 1.5]])

        ax2.hist2d(*z_sample.detach().cpu().numpy().T, bins=300, density=True,
                   range=[[-1.5, 1.5], [-1.5, 1.5]])

        logp = p_z0.log_prob(z_density) - logp_diff.view(-1)
        ax3.tricontourf(*z_t1.detach().cpu().numpy().T,
                        np.exp(logp.detach().cpu().numpy()), 200)

        plt.savefig(os.path.join(img_path, f"cnf-viz-{int(t*1000):05d}.jpg"),
                   pad_inches=0.2, bbox_inches='tight')
        plt.close()

    imgs = [Image.open(f) for f in sorted(glob.glob(os.path.join(img_path, f"cnf-viz-*.jpg")))]

    fig = plt.figure(figsize=(18,6))
    ax  = fig.gca()
    img = ax.imshow(imgs[0])

    def animate(i):
        img.set_data(imgs[i])
        return img,

    anim = animation.FuncAnimation(fig, animate, frames=41, interval=200)
    plt.close()
    return anim
