# %%
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
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 10
rcParams["figure.figsize"] = (10 / 2.54, 8 / 2.54)

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

def plot_results(prediction, amplitude, f, ax=None, quantile=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 12))

    ax.semilogx(f, amplitude, c='black', alpha = 0.8)
    ax.semilogx(f, prediction, c='red', linestyle = 'dashed')
    
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

# mlp
config = get_config(r'C:\Users\Schultz\NextCloud\WorkingDir\Projects\D2A\Code\spp_ai_acoustics\acousticnn\mmo\configs\implicit_mlp.yaml')
config.model.input_encoding = "none"
args.device = "cpu"
path = r"C:\Users\Schultz\NextCloud\WorkingDir\Projects\D2A\Code\spp_ai_acoustics\acousticnn\mmo\experiments\arch\no_encoding\implicitmlp\checkpoint_best"
net = build_model(valloader, args, config)
net.load_state_dict(torch.load(path, map_location= "cpu")["model_state_dict"])
prediction_cond1, amplitude_cond1, losses_per_f_cond1 = generate_prediction()

# %%
loss_per_sample_1 = np.mean(np.mean(losses_per_f_cond1, axis=1), axis=1)
print(np.argmin(loss_per_sample_1), np.argmax(loss_per_sample_1))
print(loss_per_sample_1[np.argmin(loss_per_sample_1)], loss_per_sample_1[np.argmax(loss_per_sample_1)])

# %%
# transformer
config = get_config(r'C:\Users\Schultz\NextCloud\WorkingDir\Projects\D2A\Code\spp_ai_acoustics\acousticnn\mmo\configs\implicit_transformer.yaml')
config.model.input_encoding = "none"
path = r"C:\Users\Schultz\NextCloud\WorkingDir\Projects\D2A\Code\spp_ai_acoustics\acousticnn\mmo\experiments\arch\no_encoding\implicit_transformer\checkpoint_best"
net = build_model(valloader, args, config)
net.load_state_dict(torch.load(path, map_location= "cpu")["model_state_dict"])
prediction_cond2, amplitude_cond2, losses_per_f_cond2 = generate_prediction()

# %%
loss_per_sample_2 = np.mean(np.mean(losses_per_f_cond2, axis=1), axis=1)
print(np.argmin(loss_per_sample_2), np.argmax(loss_per_sample_2))
print(loss_per_sample_2[np.argmin(loss_per_sample_2)], loss_per_sample_2[np.argmax(loss_per_sample_2)])

# Loss plots
fig, ax = plt.subplots(1, 1, figsize=(8 / 2.54, 7.5 / 2.54))
plot = plot_loss(losses_per_f_cond1, f, ax, quantile=0.1)
plot = plot_loss(losses_per_f_cond2, f, ax, quantile=0.1)
plot.set_title("Error")
plot.legend(["mean - MLP","90 % quant.", "mean - Transformer", " 90 % quant."], ncol = 1, prop = {'size': 9}, loc = 'upper left')
plot.set_ylim([0.0, 5.0])
plt.tight_layout()
fig.savefig("losses_MLP_Trans.png", format='png', dpi = 600)

# MLP plots
idx_min_loss_1 = np.argmin(loss_per_sample_1)
idx_max_loss_1 = np.argmax(loss_per_sample_1)

fig, ax = plt.subplots(1, 1, figsize=(8 / 2.54, 7.5 / 2.54))
i=[0]
num= idx_min_loss_1
plot = plot_results(prediction_cond1[num][:,i], amplitude_cond1[num][:,i], f, ax)
plot.set_title("MLP - Best prediction")
plot.legend(["Reference $x_1$","Prediction $x_1$"], prop = {'size': 9}, loc = 'lower left')
plt.tight_layout()
fig.savefig("samples_MLP_Best.png", format='png', dpi = 600)

fig, ax = plt.subplots(1, 1, figsize=(8 / 2.54, 7.5 / 2.54))
i=[0]
num= idx_max_loss_1
plot = plot_results(prediction_cond1[num][:,i], amplitude_cond1[num][:,i], f, ax)
plot.set_title("MLP - Worst prediction")
plot.legend(["Reference $x_1$","Prediction $x_1$"], prop = {'size': 9}, loc = 'lower left')
plt.tight_layout()
fig.savefig("samples_MLP_Worst.png", format='png', dpi = 600)

fig, ax = plt.subplots(1, 1, figsize=(8 / 2.54, 7.5 / 2.54))
i=[1,2,3]
num= idx_max_loss_1
plot = plot_results(prediction_cond1[num][:,i], amplitude_cond1[num][:,i], f, ax)
plot.set_title("MLP - Dof 2 - 4")
plot.legend(["Reference $x_2$","Reference $x_3$","Reference $x_4$","Prediction $x_2$","Prediction $x_3$","Prediction $x_4$"], prop = {'size': 9}, loc = 'lower left')
plt.tight_layout()
fig.savefig("samples_MLP_Samples.png", format='png', dpi = 600)


# Transformer plots
idx_min_loss_2 = np.argmin(loss_per_sample_2)
idx_max_loss_2 = np.argmax(loss_per_sample_2)

fig, ax = plt.subplots(1, 1, figsize=(8 / 2.54, 7.5 / 2.54))
i=[0]
num= idx_min_loss_2
plot = plot_results(prediction_cond2[num][:,i], amplitude_cond2[num][:,i], f, ax)
plot.set_title("Transformer - Best prediction")
plot.legend(["Reference $x_1$","Prediction $x_1$"], prop = {'size': 9}, loc = 'lower left')
plt.tight_layout()
fig.savefig("samples_trans_best.png", format='png', dpi = 600)

fig, ax = plt.subplots(1, 1, figsize=(8 / 2.54, 7.5 / 2.54))
i=[0]
num= idx_max_loss_2
plot = plot_results(prediction_cond2[num][:,i], amplitude_cond2[num][:,i], f, ax)
plot.set_title("Transformer - Worst prediction")
plot.legend(["Reference $x_1$","Prediction $x_1$"], prop = {'size': 9}, loc = 'lower left')
plt.tight_layout()
fig.savefig("samples_trans_worst.png", format='png', dpi = 600)

fig, ax = plt.subplots(1, 1, figsize=(8 / 2.54, 7.5 / 2.54))
i=[1,2,3]
num= idx_max_loss_1
plot = plot_results(prediction_cond2[num][:,i], amplitude_cond2[num][:,i], f, ax)
plot.set_title("Transformer - Dof 2 - 4")
plot.legend(["Reference $x_2$","Reference $x_3$","Reference $x_4$","Prediction $x_2$","Prediction $x_3$","Prediction $x_4$"], prop = {'size': 9}, loc = 'lower left')
plt.tight_layout()
fig.savefig("samples_trans_random.png", format='png', dpi = 600)

plt.show()

