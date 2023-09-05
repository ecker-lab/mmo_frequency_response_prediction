import numpy as np
from torch import nn
import torch
import matplotlib.pyplot as plt
from acousticnn.utils.logger import print_log
import os
import wandb

# Training functions
def train(args, config, net, dataloader, optimizer, valloader, scheduler, logger=None):
    lowest=np.inf
    for i in range(config.epochs):
        losses = []
        if i % config.generation_frequency == 0 and i != 0:
            dataloader.dataset.generate_samples()
            print_log(f"Generate new samples in epoch {i}", logger=logger)
        for frequency, parameters, amplitude in dataloader:
            frequency, parameters, amplitude = frequency.to(args.device), [par.to(args.device) for par in parameters], amplitude.to(args.device)
            optimizer.zero_grad()
            prediction = net(frequency, parameters) # B x num_masses x num_frequencies
            loss = nn.functional.mse_loss(prediction, amplitude)
            losses.append(loss.detach().item()*1000)
            loss.backward()
            if config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.gradient_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step(i)

        print_log(f"Epoch {i} training loss = {(np.mean(losses)):4.4}", logger=logger)
        if logger is not None:
            wandb.log({'Loss / Training': np.mean(losses), 'LR': optimizer.param_groups[0]['lr'], 'Epoch': i})

        if i % config.validation_frequency == 0 or i % int(config.epochs/10) == 0: 
            save_model(args.dir, i, net, optimizer, loss, "checkpoint_last")
            loss = evaluate(args, config, net, valloader, logger)
            if loss < lowest:
                print_log("best model", logger=logger)
                save_model(args.dir, i, net, optimizer, loss)
                lowest = loss
        if i == (config.epochs - 1):
            evaluate(args, config, net, valloader, logger, True)
            


def evaluate(args, config, net, dataloader, logger=None, plot=False):
    net.eval()
    pred, amp, losses = [], [], []
    with torch.no_grad():
        for frequency, parameters, amplitude in dataloader: 
            frequency, parameters, amplitude = frequency.to(args.device), [par.to(args.device) for par in parameters], amplitude.to(args.device)
            prediction = net(frequency, parameters) # B x num_masses x num_frequencies    
            loss = nn.functional.mse_loss(prediction, amplitude)
            amplitude, prediction = amplitude.cpu(), prediction.cpu()
            pred.append(prediction), amp.append(amplitude), losses.append(loss.cpu())
    loss = np.mean(losses)*1000
    print_log(f"Validation loss = {loss:4.4}", logger=logger)
    if logger is not None:
        wandb.log({"Loss / Validation": loss})
    if plot: 
        prediction, amplitude = pred[0], amp[0]
        fig, ax = plt.subplots(4, 1, figsize=(7, 10))
        f = np.logspace(*dataloader.dataset.parameters.f_range, dataloader.dataset.parameters.f_per_sample)
        for i in range(4):
            ax[i].grid(True)
            ax[i].semilogx(f, prediction[i], c='black')
            ax[i].semilogx(f, amplitude[i], c='green')
        #plt.show()
        if logger is not None:
            wandb.log({"Results": wandb.Image(fig)})
        plt.savefig(os.path.join(args.dir, "prediction.png"))
    return loss

def save_model(savepath, epoch, model, optimizer, loss, name="checkpoint_best"):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, 
            os.path.join(savepath, name))