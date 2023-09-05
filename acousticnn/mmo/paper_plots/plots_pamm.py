import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
from acousticnn.mmo import get_dataloader, Iter_Dataset
from acousticnn.mmo import train
from acousticnn.utils.builder import build_opti_sche, build_model
from acousticnn.utils.logger import init_train_logger, print_log
from acousticnn.utils.argparser import get_args, get_config
from torchinfo import summary
import wandb 
import time
import os
import torch
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
# rcParams['font.sans-serif'] = ['Arial']

rcParams['axes.labelsize'] = 9
rcParams['axes.titlesize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams["figure.figsize"] = (10 / 2.54, 8 / 2.54)
rcParams["text.usetex"] = True  

def plot_loss(losses_per_f, f, ax=None, quantile=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    losses_per_f = losses_per_f.transpose(0, 2, 1).reshape(-1, 200)
    mean = np.mean(losses_per_f, axis=0)
    std = np.std(losses_per_f, axis=0)
    ax.semilogx(f, mean)
    if quantile is not None:
        quantiles = np.quantile(losses_per_f, [0+quantile, 1-quantile], axis=0)
        ax.fill_between(f, quantiles[0], quantiles[1], alpha=0.2)

    ax.set_xlabel('Angular frequency in rad/s')
    ax.set_ylabel('rmse dB re 1m')
    return ax

def plot_results(prediction, amplitude, f, ax=None, quantile=None, linestyle_ref = "-", linestyle_pred = "--"):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 12))

    ax.semilogx(f, amplitude, c='black', alpha = 0.8, linestyle = linestyle_ref)
    ax.semilogx(f, prediction, c='red', linestyle = linestyle_pred)
    
    ax.grid(True)
    ax.set_xlabel('Angular frequency in rad/s')
    ax.set_ylabel('Amplitude in dB re 1m')
    
    return ax

args = get_args(["--config", r"C:\Users\Schultz\NextCloud\WorkingDir\Projects\D2A\Code\spp_ai_acoustics\acousticnn\mmo\configs\explicit_mlp.yaml"])
args.encoding = "none"
config = get_config(args.config)

def get_dataloader_helper(n_masses, n_samples, test, batch_size):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    parameters_dataset = {"n_masses": n_masses, 
            "sample_f": False,
            "f_per_sample": 200,
            "sample_m": True,
            "m_range": (1, 25),
            "sample_d": True,
            "d_range": (0.1, 1),
            "sample_k": True,
            "k_range": (1, 30),
            "normalize": False,
            "normalize_factor": 10,
            "f_range": (-1.5, 1)
    }    
    if args.encoding == "none":
        parameters_dataset["normalize"] = True
        parameters_dataset["normalize_factor"] = 10
        config.model.input_encoding = "none"
    elif args.encoding == "random":
        parameters_dataset["normalize"] = True
        parameters_dataset["normalize_factor"] = 10
        config.model.input_encoding = "random"
    elif args.encoding == "sin":
        parameters_dataset["normalize"] = True
        parameters_dataset["normalize_factor"] = 100
        config.model.input_encoding = "sin"
    else:
        raise NotImplementedError
    return get_dataloader(args, config, test=test, n_samples=n_samples, parameters=parameters_dataset, batch_size=batch_size)
valloader = get_dataloader_helper(4, 1000, True, 100)
f = np.logspace(-1.5, 1, 200)

ref = torch.mean(valloader.dataset.amplitude, dim=0).unsqueeze(0)
plt.semilogx(f, ref[0])

def generate_prediction():
    pred, amp, losses, losses_per_f = [], [], [], []
    with torch.no_grad():
        for frequency, parameters, amplitude in valloader: 
            frequency, parameters, amplitude = frequency.to(args.device), [par.to(args.device) for par in parameters], amplitude.to(args.device)
            prediction = net(frequency, parameters) # B x num_masses x num_frequencies  
            prediction = prediction * 10
            amplitude = amplitude * 10  
            loss = torch.nn.functional.mse_loss(prediction, amplitude)
            amplitude, prediction = amplitude.cpu(), prediction.cpu()
            pred.append(prediction), amp.append(amplitude), losses.append(loss.cpu())
            losses_per_f.append(torch.nn.functional.mse_loss(prediction, amplitude, reduction="none").detach().cpu().numpy())

    losses_per_f = np.vstack(losses_per_f)
    rmse_per_f = np.sqrt(losses_per_f)
    # prediction, amplitude = np.vstack(pred)- ref.numpy(), np.vstack(amp)- ref.numpy()
    prediction, amplitude = np.vstack(pred), np.vstack(amp)
    print(np.mean(losses_per_f)) 
    return prediction, amplitude, rmse_per_f

# fixed grid
config = get_config(r'C:\Users\Schultz\NextCloud\WorkingDir\Projects\D2A\Code\spp_ai_acoustics\acousticnn\mmo\configs\explicit_mlp.yaml')
config.model.input_encoding = "none"
args.device = "cpu"
path = r"C:\Users\Schultz\NextCloud\WorkingDir\Projects\D2A\Code\spp_ai_acoustics\acousticnn\mmo\experiments\arch\no_encoding\explicitmlp\checkpoint_best"
net = build_model(valloader, args, config)
net.load_state_dict(torch.load(path, map_location= "cpu")["model_state_dict"])
prediction_cond1, amplitude_cond1, losses_per_f_cond1 = generate_prediction()

loss_per_sample_1 = np.mean(np.mean(losses_per_f_cond1, axis=1), axis=1)
print(np.argmin(loss_per_sample_1), np.argmax(loss_per_sample_1))
print(loss_per_sample_1[np.argmin(loss_per_sample_1)], loss_per_sample_1[np.argmax(loss_per_sample_1)])

# query frequency
config = get_config(r'C:\Users\Schultz\NextCloud\WorkingDir\Projects\D2A\Code\spp_ai_acoustics\acousticnn\mmo\configs\implicit_mlp.yaml')
config.model.input_encoding = "none"
path = r"C:\Users\Schultz\NextCloud\WorkingDir\Projects\D2A\Code\spp_ai_acoustics\acousticnn\mmo\experiments\arch\no_encoding\implicitmlp\checkpoint_best"
net = build_model(valloader, args, config)
net.load_state_dict(torch.load(path, map_location= "cpu")["model_state_dict"])
prediction_cond2, amplitude_cond2, losses_per_f_cond2 = generate_prediction()

loss_per_sample_2 = np.mean(np.mean(losses_per_f_cond2, axis=1), axis=1)
print(np.argmin(loss_per_sample_2), np.argmax(loss_per_sample_2))
print(loss_per_sample_2[np.argmin(loss_per_sample_2)], loss_per_sample_2[np.argmax(loss_per_sample_2)])

width = 5.4
height = 5.4

# Loss plots
fig, ax = plt.subplots(1, 1, figsize=(width / 2.54, height / 2.54))
plot = plot_loss(losses_per_f_cond1, f, ax, quantile=0.1)
plot = plot_loss(losses_per_f_cond2, f, ax, quantile=0.1)
plot.set_title("Error")
plot.legend(["mean - Grid appr.","90 \% interval", "mean - Query appr.", " 90 \% interval"], ncol = 1, prop = {'size': 6}, loc = 'upper left')
plot.set_ylim([0.0, 5.0])
plt.tight_layout()
fig.savefig("losses_query_fixed.png", format='png', dpi = 600)
fig.savefig("losses_query_fixed.pdf", format='pdf')

# Fixed grid plots
idx_min_loss_1 = np.argmin(loss_per_sample_1)
idx_max_loss_1 = np.argmax(loss_per_sample_1)

fig, ax = plt.subplots(1, 1, figsize=(width / 2.54, height / 2.54))
i=[0]
num= idx_min_loss_1
plot = plot_results(prediction_cond1[num][:,i], amplitude_cond1[num][:,i], f, ax)
plot.set_title("Grid appr. - Best prediction")
plot.legend(["Reference $x_1$","Prediction $x_1$"], prop = {'size': 6}, loc = 'lower left')
plot.set_ylim([-35,15])
plt.tight_layout()
fig.savefig("samples_fixed_best.png", format='png', dpi = 600)
fig.savefig("samples_fixed_best.pdf", format='pdf')

fig, ax = plt.subplots(1, 1, figsize=(width / 2.54, height / 2.54))
i=[0]
num= idx_max_loss_1
plot = plot_results(prediction_cond1[num][:,i], amplitude_cond1[num][:,i], f, ax)
plot.set_title("Grid appr.")
plot.legend(["Reference $x_1$","Prediction $x_1$"], prop = {'size': 6}, loc = 'lower left')
plot.set_ylim([-35,15])
plt.tight_layout()
fig.savefig("samples_fixed_worst.png", format='png', dpi = 600)
fig.savefig("samples_fixed_worst.pdf", format='pdf')

fig, ax = plt.subplots(1, 1, figsize=(width / 2.54, height / 2.54))
i=[1,2,3]
num= idx_max_loss_1
plot = plot_results(prediction_cond1[num][:,i], amplitude_cond1[num][:,i], f, ax)
plot.set_title("Grid appr.")
plot.legend(["Ref. $x_2$","Ref. $x_3$","Ref. $x_4$","Pred. $x_2$","Pred. $x_3$","Pred. $x_4$"], prop = {'size': 6}, loc = 'lower left')
plot.set_ylim([-90,15])
plt.tight_layout()
fig.savefig("samples_fixed_random.png", format='png', dpi = 600)
fig.savefig("samples_fixed_random.pdf", format='pdf')


# Querry grid plots
idx_min_loss_2 = np.argmin(loss_per_sample_2)
idx_max_loss_2 = np.argmax(loss_per_sample_2)

fig, ax = plt.subplots(1, 1, figsize=(width / 2.54, height / 2.54))
i=[0]
num= idx_min_loss_2
plot = plot_results(prediction_cond2[num][:,i], amplitude_cond2[num][:,i], f, ax)
plot.set_title("Query appr. - Best prediction")
plot.legend(["Reference $x_1$","Prediction $x_1$"], prop = {'size': 6}, loc = 'lower left')
plot.set_ylim([-35,15])
plt.tight_layout()
fig.savefig("samples_query_best.png", format='png', dpi = 600)
fig.savefig("samples_query_best.pdf", format='pdf')

fig, ax = plt.subplots(1, 1, figsize=(width / 2.54, height / 2.54))
i=[0]
num= idx_max_loss_2
plot = plot_results(prediction_cond2[num][:,i], amplitude_cond2[num][:,i], f, ax)
plot.set_title("Query appr.")
plot.legend(["Reference $x_1$","Prediction $x_1$"], prop = {'size': 6}, loc = 'lower left')
plot.set_ylim([-35,15])
plt.tight_layout()
fig.savefig("samples_query_worst.png", format='png', dpi = 600)
fig.savefig("samples_query_worst.pdf", format='pdf')

fig, ax = plt.subplots(1, 1, figsize=(width / 2.54, height / 2.54))
i=[1,2,3]
num= idx_max_loss_1
plot = plot_results(prediction_cond2[num][:,i], amplitude_cond2[num][:,i], f, ax)
plot.set_title("Query appr.")
plot.legend(["Ref. $x_2$","Ref. $x_3$","Ref. $x_4$","Pred. $x_2$","Pred. $x_3$","Pred. $x_4$"], prop = {'size': 6}, loc = 'lower left')
plot.set_ylim([-90,15])
plt.tight_layout()
fig.savefig("samples_query_random.png", format='png', dpi = 600)
fig.savefig("samples_query_random.pdf", format='pdf')


fig, ax = plt.subplots(1, 1, figsize=(width / 2.54, height / 2.54))
fig.savefig("empty.pdf", format='pdf')

plt.show()

