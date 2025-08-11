#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:59:53 2024

@author: turkaygencer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import griddata
from timeit import default_timer as timer

###############################################################################
### MODE SELECTION ############################################################
###############################################################################
trainMode = False
continueTraining = False
savePost = False
isLeveled = True

###############################################################################
### HYPERPARAMETERS ###########################################################
###############################################################################
hidden_layers = [512, 256, 128, 64] 
batch_size = 256
learning_rate = 0.001
epochs = 4000
model_name = 'model-c2'
directory = 'models/post/'
training_data = [5, 10, 12, 14, 16, 18, 22, 24, 26, 28, 30, 32, 34, 36, 38]
activation_fn = 'Sigmoid' # Chose between 'ReLU', 'Leaky ReLU', 'Sigmoid', 'Tanh', 'Softsign'

###############################################################################
### POST PARAMETERS ###########################################################
###############################################################################
phi_post = 20
x_post_min, x_post_max = -1, 3
y_post_min, y_post_max = -1, 1

level_diff_u = np.linspace(0, 100, 11)
level_diff_v = np.linspace(0, 100, 11)
level_diff_p = np.linspace(0, 100, 11)

level_u_20 = np.linspace(-0.03, 0.24, 9)
level_v_20 = np.linspace(-0.1, 0.1, 10)
level_p_20 = np.linspace(-0.024, 0.03, 10)
###############################################################################
x_lim_min, x_lim_max = -5, 5
y_lim_min, y_lim_max = -5, 5
# Check that MPS is available
device="cpu"

# Fix random initial values
seed = 13
torch.manual_seed(seed)
np.random.seed(seed)

# 1. Read dataset from a CSV file
df = pd.read_csv(f'training_data/re{training_data[0]}.csv')
for i in range(1, len(training_data)):
    df0 = pd.read_csv(f'training_data/re{training_data[i]}.csv')
    df = pd.concat([df, df0], ignore_index=True)

# Convert data to numpy arrays
df = df[(df['x-coordinate'].between(x_lim_min, x_lim_max)) 
                  & (df['y-coordinate'].between(y_lim_min, y_lim_max))]
phi = df['re'].values
x_coord = df['x-coordinate'].values
y_coord = df['y-coordinate'].values
x_vel = df['x-velocity'].values
y_vel = df['y-velocity'].values
pressure = df['pressure'].values

# Normaliaze data
phi_mean, phi_std = np.mean(phi), np.std(phi)
x_mean, x_std = np.mean(x_coord), np.std(x_coord)
y_mean, y_std = np.mean(y_coord), np.std(y_coord)
u_mean, u_std = np.mean(x_vel), np.std(x_vel)
v_mean, v_std = np.mean(y_vel), np.std(y_vel)
p_mean, p_std = np.mean(pressure), np.std(pressure)

umin, umax = min(x_vel), max(x_vel)
vmin, vmax = min(y_vel), max(y_vel)
pmin, pmax = min(pressure), max(pressure)

phi_norm_factor = 40

def scale(value, type):
    if type == 'x':
        return value/10
    elif type == 'y':
        return value/10
    elif type == 'u':
        return (value - umin)/(umax - umin)
    elif type == 'v':
        return (value - vmin)/(vmax - vmin)
    elif type == 'p':
        return (value - pmin)/(pmax - pmin)
    else:
        return value/phi_norm_factor

def unscale(value, type):
    if type == 'x':
        return value*10
    elif type == 'y':
        return value*10
    elif type == 'u':
        return (value * (umax - umin)) + umin
    elif type == 'v':
        return (value * (vmax - vmin)) + vmin
    elif type == 'p':
        return (value * (pmax - pmin)) + pmin
    else:
        return value*phi_norm_factor
        
    
phi_norm = scale(phi, 'phi')
x_coord_norm = scale(x_coord, 'x')
y_coord_norm = scale(y_coord, 'y')
x_vel_norm = scale(x_vel, 'u')
y_vel_norm = scale(y_vel, 'v')
pressure_norm = scale(pressure, 'p')

# Convert data to PyTorch tensors
inputs = torch.tensor(np.vstack((phi_norm, x_coord_norm, y_coord_norm)).T, 
                      dtype=torch.float32, device=device)
outputs = torch.tensor(np.vstack((x_vel_norm, y_vel_norm, pressure_norm)).T, 
                       dtype=torch.float32, device=device)

# Create dataset and dataloader
dataset = TensorDataset(inputs, outputs)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 2. Define neural network model
class FlexibleNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(FlexibleNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        # First hidden layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_layers[0]))
        
        # Other hidden layers
        for i in range(1, len(hidden_layers)):
            self.hidden_layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_layers[-1], output_size)
    
    def forward(self, phi):
        for layer in self.hidden_layers:
            if activation_fn == 'ReLU':
                phi = F.relu(layer(phi)) # ACTIVATION FUNCTION #
            elif activation_fn == 'Leaky ReLU':
                phi = F.leaky_relu(layer(phi))
            elif activation_fn == 'Sigmoid':
                phi = F.sigmoid(layer(phi))
            elif activation_fn == 'Tanh':
                phi = F.tanh(layer(phi))
            elif activation_fn == 'Softsign':
                phi = F.softsign(layer(phi))
            else:
                print('Custom activation function used')
                # phi = 2*F.sigmoid(layer(phi))-1
        phi = self.output_layer(phi)
        return phi

model = FlexibleNN(input_size=3, hidden_layers=hidden_layers, output_size=3).to(device)

# 3. Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if continueTraining:
    model.load_state_dict(torch.load(f"{directory}{model_name}.pth"))
    
# 4. Train model
if trainMode:
    lossLog=np.zeros(epochs)
    ulog = np.zeros(epochs)
    vlog = np.zeros(epochs)
    plog = np.zeros(epochs)
    start = timer()
    for epoch in range(epochs):
        size = len(data_loader.dataset)
        model.train()
        for batch, (inputs, targets) in enumerate(data_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # if batch % 20 == 0:
            #     loss, current = loss.item(), (batch + 1) * len(inputs)
            #     print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
        monitor_inputs = torch.tensor([scale(phi_post, 'phi'), 0.1, 0.1], dtype=torch.float32, device=device)
        with torch.no_grad():
            monitor_predicted = model(monitor_inputs)
            ulog[epoch], vlog[epoch], plog[epoch] = monitor_predicted.to("cpu").numpy()
        
        if (epoch+1) % 100 == 0:  # Print loss per 100 epochs
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.7f}')
        lossLog[epoch] = loss.item()
    end = timer()
    training_time = end - start
    print(f'Training Time: {training_time:.2f}s')
    
    # Write model information
    with open(f'{directory}/{model_name}-index.txt', 'w') as file:
        file.write(f'Hidden Layers: {hidden_layers}\n')
        file.write(f'Batch Size: {batch_size}\n')
        file.write(f'Learning Rate: {learning_rate}\n')
        file.write(f'Epochs: {epochs}\n')
        file.write(f'Trained at Reynolds: {training_data}\n')
        file.write(f'Activation Function: {activation_fn}\n')
        file.write(f'Mean Loss: {lossLog[epochs-1]}\n')
        file.write(f'Training Time: {training_time:.2f}\n')
        
    # Plot loss
    plt.figure(1)
    plt.plot(np.arange(epochs), lossLog)
    plt.yscale("log")
    plt.ylim(1e-8,1e-2)
    plt.xlabel("Eopchs")
    plt.ylabel("Loss")
    plt.grid(color='0.95')
    plt.savefig(fname=f"{directory}{model_name}-loss.png")

    # Plot monitors
    plt.figure(2)
    plt.plot(np.arange(epochs), ulog)
    plt.xlabel("Eopchs")
    plt.ylabel("u_log")
    plt.grid(color='0.95')
    plt.savefig(fname=f"{directory}{model_name}-ulog.png")

    plt.figure(3)
    plt.plot(np.arange(epochs), vlog)
    plt.xlabel("Eopchs")
    plt.ylabel("v_log")
    plt.grid(color='0.95')
    plt.savefig(fname=f"{directory}{model_name}-vlog.png")

    plt.figure(4)
    plt.plot(np.arange(epochs), plog)
    plt.xlabel("Eopchs")
    plt.ylabel("p_log")
    plt.grid(color='0.95')
    plt.savefig(fname=f"{directory}/{model_name}-plog.png")
     
# Save model
    torch.save(model.state_dict(), f"{directory}{model_name}.pth")
    print(f"Saved PyTorch Model State to {directory}{model_name}.pth")
else:
# Load model
    model.load_state_dict(torch.load(f"{directory}{model_name}.pth"))

# 5. Evaluate model
model.eval()

# Generate grid
sample_x = 401
sample_y = 101
x_unique = np.linspace(scale(x_post_min, 'x'), scale(x_post_max, 'x'), sample_x)
y_unique = np.linspace(scale(y_post_min, 'y'), scale(y_post_max, 'y'), sample_y)
x_grid, y_grid = np.meshgrid(x_unique, y_unique)
x_grid_flatten = x_grid.flatten()
y_grid_flatten = y_grid.flatten()

# Prediction using post-process coordinates 
phi_input = torch.tensor((np.ones(sample_x*sample_y)*scale(phi_post, 'phi')), dtype=torch.float32, device=device).view(-1, 1)
x_input = torch.tensor(x_grid_flatten, dtype=torch.float32, device=device).view(-1, 1)
y_input = torch.tensor(y_grid_flatten, dtype=torch.float32, device=device).view(-1, 1)
inputs = torch.cat((phi_input, x_input, y_input), dim=1)
predicted = model(inputs).detach().numpy()

# 6. Visualization

df_post = pd.read_csv(f'training_data/re{phi_post}.csv')
# Filter original data
df_filtered = df_post[(df_post['x-coordinate'].between(x_post_min-0.5, x_post_max+0.5)) 
                 & (df_post['y-coordinate'].between(y_post_min-0.5, y_post_max+0.5))]
x_filtered = df_filtered['x-coordinate'].values
y_filtered = df_filtered['y-coordinate'].values
u_filtered = df_filtered['x-velocity'].values
v_filtered = df_filtered['y-velocity'].values
p_filtered = df_filtered['pressure'].values


# Calculate Z


z_predicted_u = griddata((x_grid_flatten, y_grid_flatten), predicted[:, 0], (x_grid, y_grid), method='linear')
z_predicted_v = griddata((x_grid_flatten, y_grid_flatten), predicted[:, 1], (x_grid, y_grid), method='linear')
z_predicted_p = griddata((x_grid_flatten, y_grid_flatten), predicted[:, 2], (x_grid, y_grid), method='linear')

x_grid = unscale(x_grid, 'x')
y_grid = unscale(y_grid, 'y')
z_predicted_u = unscale(z_predicted_u, 'u')
z_predicted_v = unscale(z_predicted_v, 'v')
z_predicted_p = unscale(z_predicted_p, 'p')

z_original_u = griddata((x_filtered, y_filtered), u_filtered, (x_grid, y_grid), method='linear')
z_original_v = griddata((x_filtered, y_filtered), v_filtered, (x_grid, y_grid), method='linear')
z_original_p = griddata((x_filtered, y_filtered), p_filtered, (x_grid, y_grid), method='linear')

temp_loss = []
res = 500
loss_u = abs((z_predicted_u - z_original_u)*100/(z_original_u+1e-10))
for i in range(sample_y):
    for j in range(sample_x):
        if loss_u[i][j] < res:
            temp_loss.append(loss_u[i][j])
mean_loss_u = np.mean(temp_loss)
print(f'Mean u loss: {mean_loss_u}')

temp_loss = []
loss_v = abs((z_predicted_v - z_original_v)*100/(z_original_v+1e-10))
for i in range(sample_y):
    for j in range(sample_x):
        if loss_v[i][j] < res:
            temp_loss.append(loss_v[i][j])
mean_loss_v = np.mean(temp_loss)
print(f'Mean v loss: {mean_loss_v}')

temp_loss = []
loss_p = abs((z_predicted_p - z_original_p)*100/(z_original_p+1e-10))
for i in range(sample_y):
    for j in range(sample_x):
        if loss_p[i][j] < res:
            temp_loss.append(loss_p[i][j])
mean_loss_p = np.mean(temp_loss)
print(f'Mean p loss: {mean_loss_p}')

u_min_o, u_max_o = np.min(z_original_u), np.max(z_original_u)
u_min_p, u_max_p = np.min(z_predicted_u), np.max(z_predicted_u)
u_min, u_max = min(u_min_o,u_min_p), max(u_max_o,u_max_p)

v_min_o, v_max_o = np.min(z_original_v), np.max(z_original_v)
v_min_p, v_max_p = np.min(z_predicted_v), np.max(z_predicted_v)
v_min, v_max = min(v_min_o,v_min_p), max(v_max_o,v_max_p)

p_min_o, p_max_o = np.min(z_original_p), np.max(z_original_p)
p_min_p, p_max_p = np.min(z_predicted_p), np.max(z_predicted_p)
p_min, p_max = min(p_min_o,p_min_p), max(p_max_o,p_max_p)

level_u = np.linspace(u_min, u_max, 11)
level_v = np.linspace(v_min, v_max, 11)
level_p = np.linspace(p_min, p_max, 11)

# Cylinder
circles = []
for i in range(13):
    circles.append(Circle((0, 0), 0.5, color='white', fill=True, zorder=10))
    
# Load geometry from data file
with open('naca0012.txt', 'r') as file:
    x, y = np.loadtxt(file, dtype=float, unpack=True)

plt.figure(figsize=(10, 8))
plt.figure(5) if trainMode else plt.figure(1)

# Original x-velocity
plt.subplot(3, 3, 1).set_aspect('equal', 'box')
if isLeveled:
    contour_original_u = plt.contourf(x_grid, y_grid, z_original_u, levels=level_u, cmap='viridis')
else:
    contour_original_u = plt.contourf(x_grid, y_grid, z_original_u, cmap='viridis')

plt.colorbar(contour_original_u)
plt.title('HAD: X-Hızı [m/s]')
# plt.fill(x, y, color='w', zorder=3)
plt.gca().add_patch(circles[0])

# Predicted x-velocity
plt.subplot(3, 3, 2).set_aspect('equal', 'box')
if isLeveled:
    contour_predicted_u = plt.contourf(x_grid, y_grid, z_predicted_u, levels=level_u, cmap='viridis')
else:
    contour_predicted_u = plt.contourf(x_grid, y_grid, z_predicted_u, cmap='viridis')
plt.colorbar(contour_predicted_u)
plt.title('Tahmin: X-Hızı [m/s]')
# plt.fill(x, y, color='w', zorder=3)
plt.gca().add_patch(circles[1])

# Difference between original and predicted
plt.subplot(3, 3, 3).set_aspect('equal', 'box')
contour_difference_u = plt.contourf(x_grid, y_grid, 
                                   abs((z_predicted_u - z_original_u)*100/(z_original_u+1e-10)), 
                                   cmap='viridis',
                                   levels=level_diff_u)
plt.colorbar(contour_difference_u)
plt.title('Fark: X-Hızı [%]')
# plt.fill(x, y, color='w', zorder=3)
plt.gca().add_patch(circles[2])

# Original y-velocity
plt.subplot(3, 3, 4).set_aspect('equal', 'box')
if isLeveled:
    contour_original_v = plt.contourf(x_grid, y_grid, z_original_v,levels=level_v, cmap='viridis')
else:
    contour_original_v = plt.contourf(x_grid, y_grid, z_original_v, cmap='viridis')
plt.colorbar(contour_original_v)
plt.title('HAD: Y-Hızı [m/s]')
# plt.fill(x, y, color='w', zorder=3)
plt.gca().add_patch(circles[3])

# Predicted y-velocity
plt.subplot(3, 3, 5).set_aspect('equal', 'box')
if isLeveled:
    contour_predicted_v = plt.contourf(x_grid, y_grid, z_predicted_v,levels=level_v, cmap='viridis')
else:
    contour_predicted_v = plt.contourf(x_grid, y_grid, z_predicted_v, cmap='viridis') 
plt.colorbar(contour_predicted_v)
plt.title('Tahmin: Y-Hızı [m/s]')
# plt.fill(x, y, color='w', zorder=3)
plt.gca().add_patch(circles[4])

# Difference between original and predicted
plt.subplot(3, 3, 6).set_aspect('equal', 'box')
contour_difference_v = plt.contourf(x_grid, y_grid, 
                                   abs((z_predicted_v - z_original_v)*100/(z_original_v+1e-10)), 
                                   cmap='viridis',
                                   levels=level_diff_v)
plt.colorbar(contour_difference_v)
plt.title('Fark: Y-Hızı [%]')
# plt.fill(x, y, color='w', zorder=3)
plt.gca().add_patch(circles[5])

# Original pressure
plt.subplot(3, 3, 7).set_aspect('equal', 'box')
if isLeveled:
    contour_original_p = plt.contourf(x_grid, y_grid, z_original_p, cmap='viridis', levels=level_p)
else:
    contour_original_p = plt.contourf(x_grid, y_grid, z_original_p, cmap='viridis')
plt.colorbar(contour_original_p)
plt.title('HAD: Basınç [Pa]')
# plt.fill(x, y, color='w', zorder=3)
plt.gca().add_patch(circles[6])

# Predicted pressure
plt.subplot(3, 3, 8).set_aspect('equal', 'box')
if isLeveled:
    contour_predicted_p = plt.contourf(x_grid, y_grid, z_predicted_p, cmap='viridis', levels=level_p)
else:
    contour_predicted_p = plt.contourf(x_grid, y_grid, z_predicted_p, cmap='viridis')
plt.colorbar(contour_predicted_p)
plt.title('Tahmin: Basınç [Pa]')
# plt.fill(x, y, color='w', zorder=3)
plt.gca().add_patch(circles[7])

# Difference between original and predicted
plt.subplot(3, 3, 9).set_aspect('equal', 'box')
contour_difference_p = plt.contourf(x_grid, y_grid, 
                                   abs((z_predicted_p - z_original_p)*100/(z_original_p+1e-10)), 
                                   cmap='viridis',
                                   levels=level_diff_p)
plt.colorbar(contour_difference_p)
plt.title('Fark: Basınç [%]')
# plt.fill(x, y, color='w', zorder=3)
plt.gca().add_patch(circles[8])

plt.tight_layout()
if savePost:
    plt.savefig(f"{directory}{model_name}-contour-re{phi_post}.png")
    
plt.figure(figsize=(9, 5))
plt.figure(6) if trainMode else plt.figure(2)
# Original streamline
plt.subplot(1, 2, 1).set_aspect('equal', 'box')
vel_original = np.sqrt(z_original_u**2 + z_original_v**2)
# plt.fill(x, y, color='k', zorder=3)
strm = plt.streamplot(x_grid, y_grid, z_original_u, z_original_v, density=3,
                      color=vel_original, broken_streamlines=False, linewidth=1, cmap='viridis')
# plt.colorbar(strm.lines)
plt.title('HAD Akış Çizgisi')
plt.gca().add_patch(circles[9])
# Predicted streamline
plt.subplot(1, 2, 2).set_aspect('equal', 'box')
# plt.fill(x, y, color='k', zorder=3)
vel_predicted = np.sqrt(z_predicted_u**2 + z_predicted_v**2)
strm = plt.streamplot(x_grid, y_grid, z_predicted_u, z_predicted_v, density=3,
                      color=vel_predicted, broken_streamlines=False, linewidth=1, cmap='viridis')
# plt.colorbar(strm.lines)
plt.title('Tahmin Akış Çizgisi')
plt.gca().add_patch(circles[10])
plt.tight_layout()
if savePost:
    plt.savefig(f"{directory}{model_name}-streamline-re{phi_post}.png")
    
# Original streamline
plt.figure(figsize=(7, 5))
plt.figure(7) if trainMode else plt.figure(3)
plt.axes().set_aspect('equal', 'box')
vel_original = np.sqrt(z_original_u**2 + z_original_v**2)
# plt.fill(x, y, color='k', zorder=3)
strm = plt.streamplot(x_grid, y_grid, z_original_u, z_original_v, density=(3,1), arrowstyle='-',
                      color='k', broken_streamlines=False, linewidth=1, cmap='viridis')
# plt.colorbar(strm.lines)
plt.title('HAD Akış Çizgisi')
plt.gca().add_patch(circles[11])
if savePost:
    plt.savefig(f"{directory}{model_name}-stro-re{phi_post}.png")

# Predicted streamline
plt.figure(figsize=(7, 5))
plt.figure(8) if trainMode else plt.figure(4)
plt.axes().set_aspect('equal', 'box')
vel_original = np.sqrt(z_original_u**2 + z_original_v**2)
# plt.fill(x, y, color='k', zorder=3)
vel_predicted = np.sqrt(z_predicted_u**2 + z_predicted_v**2)
strm = plt.streamplot(x_grid, y_grid, z_predicted_u, z_predicted_v, density=(3,1), arrowstyle='-',
                      color='k', broken_streamlines=False, linewidth=1, cmap='viridis')
# plt.colorbar(strm.lines)
plt.title('Tahmin Akış Çizgisi')
plt.gca().add_patch(circles[12])
if savePost:
    plt.savefig(f"{directory}{model_name}-strp-re{phi_post}.png")

plt.show()
