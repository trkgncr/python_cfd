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
# from matplotlib.patches import Circle
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import griddata
from timeit import default_timer as timer

###############################################################################
### MODE SELECTION ############################################################
###############################################################################
trainMode = True
continueTraining = False
savePost = True
isLeveled = True

###############################################################################
### HYPERPARAMETERS ###########################################################
###############################################################################
hidden_layers = [1024,512,512,512] 
batch_size = 1024
learning_rate = 0.001
epochs = 10
model_name = 'model-2'
directory = 'models/naca0012_final/'
activation_fn = 'Sigmoid' # Chose between 'ReLU', 'Leaky ReLU', 'Sigmoid', 'Tanh', 'Softsign'
stop_criteria = 3e-6
###############################################################################
### POST PARAMETERS ###########################################################
###############################################################################
aoa_post_index = 0
x_post_min, x_post_max = -1, 2
y_post_min, y_post_max = -1, 1

level_diff_u = np.linspace(0, 100, 11)
level_diff_v = np.linspace(0, 100, 11)
level_diff_p = np.linspace(0, 100, 11)
###############################################################################
x_lim_min, x_lim_max = -2, 3
y_lim_min, y_lim_max = -2.5, 2.5
# Check that MPS is available
device="cuda"

seed = 13
torch.manual_seed(seed)
np.random.seed(seed)

data = np.linspace(0, 8, 41)
np.random.shuffle(data)
training_data, validation_data, test_data = np.split(data, [29, 35])

###############################################################################
### TRAINING DATA #############################################################
###############################################################################
# 1. Veri setini CSV dosyasından okuma
df_training = pd.read_csv(f'training_data/aoa{training_data[0]:.1f}.csv')
for i in range(1, len(training_data)):
    df0 = pd.read_csv(f'training_data/aoa{training_data[i]:.1f}.csv')
    df_training = pd.concat([df_training, df0], ignore_index=True)

# Verileri numpy dizilerine dönüştürme
df_training = df_training[(df_training['x-coordinate'].between(x_lim_min, x_lim_max)) 
                  & (df_training['y-coordinate'].between(y_lim_min, y_lim_max))]
aoa = df_training['aoa'].values
x_coord = df_training['x-coordinate'].values
y_coord = df_training['y-coordinate'].values
x_vel = df_training['x-velocity'].values
y_vel = df_training['y-velocity'].values
pressure = df_training['pressure'].values

# Veriyi normalize etme
# aoa_mean, aoa_std = np.mean(aoa), np.std(aoa)
# x_mean, x_std = np.mean(x_coord), np.std(x_coord)
# y_mean, y_std = np.mean(y_coord), np.std(y_coord)
# u_mean, u_std = np.mean(x_vel), np.std(x_vel)
# v_mean, v_std = np.mean(y_vel), np.std(y_vel)
# p_mean, p_std = np.mean(pressure), np.std(pressure)

umin, umax = min(x_vel), max(x_vel)
vmin, vmax = min(y_vel), max(y_vel)
pmin, pmax = min(pressure), max(pressure)

aoa_norm_factor = 10

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
        return value/aoa_norm_factor

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
        return value*aoa_norm_factor
        
    
aoa_norm = scale(aoa, 'aoa')
x_coord_norm = scale(x_coord, 'x')
y_coord_norm = scale(y_coord, 'y')
x_vel_norm = scale(x_vel, 'u')
y_vel_norm = scale(y_vel, 'v')
pressure_norm = scale(pressure, 'p')

# Veriyi PyTorch tensörlerine dönüştürme
inputs = torch.tensor(np.vstack((aoa_norm, x_coord_norm, y_coord_norm)).T, 
                      dtype=torch.float32, device=device)
outputs = torch.tensor(np.vstack((x_vel_norm, y_vel_norm, pressure_norm)).T, 
                       dtype=torch.float32, device=device)
training_set = TensorDataset(inputs, outputs)
training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
###############################################################################
### VALIDATION DATA ###########################################################
###############################################################################
# 1. Veri setini CSV dosyasından okuma
df_validation = pd.read_csv(f'training_data/aoa{validation_data[0]:.1f}.csv')
for i in range(1, len(validation_data)):
    df0 = pd.read_csv(f'training_data/aoa{validation_data[i]:.1f}.csv')
    df_validation = pd.concat([df_validation, df0], ignore_index=True)

# Verileri numpy dizilerine dönüştürme
df_validation = df_validation[(df_validation['x-coordinate'].between(x_lim_min, x_lim_max)) 
                  & (df_validation['y-coordinate'].between(y_lim_min, y_lim_max))]
aoa = df_validation['aoa'].values
x_coord = df_validation['x-coordinate'].values
y_coord = df_validation['y-coordinate'].values
x_vel = df_validation['x-velocity'].values
y_vel = df_validation['y-velocity'].values
pressure = df_validation['pressure'].values

# Veriyi normalize etme
aoa_norm = scale(aoa, 'aoa')
x_coord_norm = scale(x_coord, 'x')
y_coord_norm = scale(y_coord, 'y')
x_vel_norm = scale(x_vel, 'u')
y_vel_norm = scale(y_vel, 'v')
pressure_norm = scale(pressure, 'p')

# Veriyi PyTorch tensörlerine dönüştürme
inputs = torch.tensor(np.vstack((aoa_norm, x_coord_norm, y_coord_norm)).T, 
                      dtype=torch.float32, device=device)
outputs = torch.tensor(np.vstack((x_vel_norm, y_vel_norm, pressure_norm)).T, 
                       dtype=torch.float32, device=device)

# Veri seti ve DataLoader oluşturma
validation_set = TensorDataset(inputs, outputs)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

# 2. Sinir ağı modelini tanımlama
class FlexibleNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(FlexibleNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        # İlk gizli katman
        self.hidden_layers.append(nn.Linear(input_size, hidden_layers[0]))
        
        # Diğer gizli katmanlar
        for i in range(1, len(hidden_layers)):
            self.hidden_layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        
        # Çıkış katmanı
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

# 3. Kayıp fonksiyonu ve optimizer'ı tanımlama
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if continueTraining:
    model.load_state_dict(torch.load(f"{directory}{model_name}.pth"))
    
# 4. Modeli eğitme
if trainMode:
    lossLog = np.zeros(0)
    vlossLog = np.zeros(0)
    ulog = np.zeros(0)
    vlog = np.zeros(0)
    plog = np.zeros(0)
    start = timer()
    print('----------------------------------------')
    for epoch in range(epochs):
        model.train(True)
        for batch, (inputs, targets) in enumerate(training_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.7f}')
        lossLog = np.append(lossLog, loss.item())
        
        model.eval()
        with torch.no_grad():
            for vbatch, (vinputs, vtargets) in enumerate(validation_loader):
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vtargets)
            monitor_input = torch.tensor([scale(validation_data[0], 'aoa'), 0.1, 0.1], dtype=torch.float32, device=device)
            monitor_predicted = model(monitor_input)
            utemp, vtemp, ptemp = monitor_predicted.to("cpu").numpy()
            ulog = np.append(ulog, unscale(utemp, 'u'))
            vlog = np.append(vlog, unscale(vtemp, 'v'))
            plog = np.append(plog, unscale(ptemp, 'p'))
        
        if (epoch+1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {vloss.item():.7f}')
            print('----------------------------------------')
        vlossLog = np.append(vlossLog, vloss.item())
        
        if loss.item() <stop_criteria:
            break
    end = timer()
    training_time = end - start
    print(f'Training Time: {training_time:.2f}s')
    
    # Write model information
    with open(f'{directory}{model_name}-index.txt', 'w') as file:
        file.write(f'Hidden Layers: {hidden_layers}\n')
        file.write(f'Batch Size: {batch_size}\n')
        file.write(f'Learning Rate: {learning_rate}\n')
        file.write(f'Epochs: {epochs}\n')
        file.write(f'Trained at AoAs: {training_data}\n')
        file.write(f'Validated at AoAs: {validation_data}\n')
        file.write(f'Tested at AoAs: {test_data}\n')
        file.write(f'Activation Function: {activation_fn}\n')
        file.write(f'Mean Training Loss: {lossLog[-1]}\n')
        file.write(f'Mean Validation Loss: {vlossLog[-1]}\n')
        file.write(f'Training Time: {training_time:.2f}\n')
        
    # Kaybı çizdir
    plt.figure(1)
    plt.plot(np.arange(epoch+1), lossLog, '-b', label='Eğitim kaybı')
    plt.plot(np.arange(epoch+1), vlossLog, '-r', label='Doğrulama kaybı')
    plt.yscale("log")
    plt.ylim(1e-6,1e0)
    plt.xlabel("Tur Sayısı")
    plt.ylabel("Kayıp")
    plt.grid(color='0.95')
    plt.legend(fontsize='large')
    plt.savefig(fname=f"{directory}{model_name}-loss.png")

    # Monitörleri çizdir
    plt.figure(2)
    plt.plot(np.arange(epoch+1), ulog, label="Tahmin")
    plt.plot(np.arange(epoch+1), np.ones(epoch+1)*1.029, '--', label="Referans")
    plt.xlabel("Tur Sayısı")
    plt.ylabel("u_log")
    plt.grid(color='0.95')
    plt.savefig(fname=f"{directory}{model_name}-ulog.png")

    plt.figure(3)
    plt.plot(np.arange(epoch+1), vlog, label="Tahmin")
    plt.plot(np.arange(epoch+1), np.ones(epoch+1)*0.034, '--', label="Referans")
    plt.xlabel("Tur Sayısı")
    plt.ylabel("v_log")
    plt.grid(color='0.95')
    plt.savefig(fname=f"{directory}{model_name}-vlog.png")

    plt.figure(4)
    plt.plot(np.arange(epoch+1), plog, label="Tahmin")
    plt.plot(np.arange(epoch+1), np.ones(epoch+1)*(-0.028), '--', label="Referans")
    plt.xlabel("Tur Sayısı")
    plt.ylabel("p_log")
    plt.grid(color='0.95')
    plt.savefig(fname=f"{directory}{model_name}-plog.png")
     
# Save model
    torch.save(model.state_dict(), f"{directory}{model_name}.pth")
    print(f"Saved PyTorch Model State to {directory}{model_name}.pth")
else:
# Load model
    model.load_state_dict(torch.load(f"{directory}{model_name}.pth"))

###############################################################################
### TEST ######################################################################
###############################################################################
# 5. Evaluate model
model.eval()

# Generate grid
sample_x = 301
sample_y = 201
x_unique = np.linspace(scale(x_post_min, 'x'), scale(x_post_max, 'x'), sample_x)
y_unique = np.linspace(scale(y_post_min, 'y'), scale(y_post_max, 'y'), sample_y)
x_grid, y_grid = np.meshgrid(x_unique, y_unique)
x_grid_flatten = x_grid.flatten()
y_grid_flatten = y_grid.flatten()

# Post koordinatları kullanarak tahmin 
x_input = torch.tensor(x_grid_flatten, dtype=torch.float32, device=device).view(-1, 1)
y_input = torch.tensor(y_grid_flatten, dtype=torch.float32, device=device).view(-1, 1)

print(f'Test for aoa: {test_data[aoa_post_index]:.1f} deg\n')
aoa_input = torch.tensor((np.ones(sample_x*sample_y)*scale(test_data[aoa_post_index], 'aoa')), 
                         dtype=torch.float32, device=device).view(-1, 1)
inputs = torch.cat((aoa_input, x_input, y_input), dim=1)
predicted = model(inputs).detach().cpu().numpy()

# 6. Sonuçları görselleştirme

df_post = pd.read_csv(f'training_data/aoa{test_data[aoa_post_index]:.1f}.csv')
# Orijinal veriyi filtreleme
df_filtered = df_post[(df_post['x-coordinate'].between(x_post_min-0.5, x_post_max+0.5)) 
    & (df_post['y-coordinate'].between(y_post_min-0.5, y_post_max+0.5))]
x_filtered = df_filtered['x-coordinate'].values
y_filtered = df_filtered['y-coordinate'].values
u_filtered = df_filtered['x-velocity'].values
v_filtered = df_filtered['y-velocity'].values
p_filtered = df_filtered['pressure'].values


# Z değerlerini hesaplama
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
rmse_u = np.sqrt(np.mean((z_predicted_u - z_original_u)**2))
rms_u = np.sqrt(np.mean(z_original_u**2))
rmse_loss_u = rmse_u*100/rms_u
print(f'RMSE u loss: {rmse_loss_u}')
print(f'Mean u loss: {mean_loss_u}')
print()

temp_loss = []
loss_v = abs((z_predicted_v - z_original_v)*100/(z_original_v+1e-10))
for i in range(sample_y):
    for j in range(sample_x):
        if loss_v[i][j] < res:
            temp_loss.append(loss_v[i][j])
mean_loss_v = np.mean(temp_loss)
rmse_v = np.sqrt(np.mean((z_predicted_v - z_original_v)**2))
rms_v = np.sqrt(np.mean(z_original_v**2))
rmse_loss_v = rmse_v*100/rms_v
print(f'RMSE v loss: {rmse_loss_v}')
print(f'Mean v loss: {mean_loss_v}')
print()

temp_loss = []
loss_p = abs((z_predicted_p - z_original_p)*100/(z_original_p+1e-10))
for i in range(sample_y):
    for j in range(sample_x):
        if loss_p[i][j] < res:
            temp_loss.append(loss_p[i][j])
mean_loss_p = np.mean(temp_loss)
rmse_p = np.sqrt(np.mean((z_predicted_p - z_original_p)**2))
rms_p = np.sqrt(np.mean(z_original_p**2))
rmse_loss_p = rmse_p*100/rms_p
print(f'RMSE p loss: {rmse_loss_p}')
print(f'Mean p loss: {mean_loss_p}')
print()

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
    
# load geometry from data file
with open('naca0012.txt', 'r') as file:
    x_naca, y_naca = np.loadtxt(file, dtype=float, unpack=True)

plt.figure(figsize=(10, 6))
plt.figure(5) if trainMode else plt.figure(1)

# x-hızının orijinal değeri
plt.subplot(3, 3, 1).set_aspect('equal', 'box')
if isLeveled:
    contour_original_u = plt.contourf(x_grid, y_grid, z_original_u, levels=level_u, cmap='viridis')
else:
    contour_original_u = plt.contourf(x_grid, y_grid, z_original_u, cmap='viridis')

plt.colorbar(contour_original_u)
plt.title('HAD: X-Hızı [m/s]')
plt.fill(x_naca, y_naca, color='w', zorder=3)
# plt.gca().add_patch(circles[0])

# x-hızının tahmin edilen değeri
plt.subplot(3, 3, 2).set_aspect('equal', 'box')
if isLeveled:
    contour_predicted_u = plt.contourf(x_grid, y_grid, z_predicted_u, levels=level_u, cmap='viridis')
else:
    contour_predicted_u = plt.contourf(x_grid, y_grid, z_predicted_u, cmap='viridis')
plt.colorbar(contour_predicted_u)
plt.title('Tahmin: X-Hızı [m/s]')
plt.fill(x_naca, y_naca, color='w', zorder=3)
# plt.gca().add_patch(circles[1])

# x-hızının farkı
plt.subplot(3, 3, 3).set_aspect('equal', 'box')
contour_difference_u = plt.contourf(x_grid, y_grid, 
                                    abs((z_predicted_u - z_original_u)*100/(z_original_u+1e-10)), 
                                    cmap='viridis',
                                    levels=level_diff_u)
plt.colorbar(contour_difference_u)
plt.title('Fark: X-Hızı [%]')
plt.fill(x_naca, y_naca, color='w', zorder=3)
# plt.gca().add_patch(circles[2])

# y-hızının orijinal değeri
plt.subplot(3, 3, 4).set_aspect('equal', 'box')
if isLeveled:
    contour_original_v = plt.contourf(x_grid, y_grid, z_original_v,levels=level_v, cmap='viridis')
else:
    contour_original_v = plt.contourf(x_grid, y_grid, z_original_v, cmap='viridis')
plt.colorbar(contour_original_v)
plt.title('HAD: Y-Hızı [m/s]')
plt.fill(x_naca, y_naca, color='w', zorder=3)
# plt.gca().add_patch(circles[3])

# y-hızının tahmin edilen değeri
plt.subplot(3, 3, 5).set_aspect('equal', 'box')
if isLeveled:
    contour_predicted_v = plt.contourf(x_grid, y_grid, z_predicted_v,levels=level_v, cmap='viridis')
else:
    contour_predicted_v = plt.contourf(x_grid, y_grid, z_predicted_v, cmap='viridis') 
plt.colorbar(contour_predicted_v)
plt.title('Tahmin: Y-Hızı [m/s]')
plt.fill(x_naca, y_naca, color='w', zorder=3)
# plt.gca().add_patch(circles[4])

# y-hızının farkı
plt.subplot(3, 3, 6).set_aspect('equal', 'box')
contour_difference_v = plt.contourf(x_grid, y_grid, 
                                    abs((z_predicted_v - z_original_v)*100/(z_original_v+1e-10)), 
                                    cmap='viridis',
                                    levels=level_diff_v)
plt.colorbar(contour_difference_v)
plt.title('Fark: Y-Hızı [%]')
plt.fill(x_naca, y_naca, color='w', zorder=3)
# plt.gca().add_patch(circles[5])

# Basıncın orijinal değeri
plt.subplot(3, 3, 7).set_aspect('equal', 'box')
if isLeveled:
    contour_original_p = plt.contourf(x_grid, y_grid, z_original_p, cmap='viridis', levels=level_p)
else:
    contour_original_p = plt.contourf(x_grid, y_grid, z_original_p, cmap='viridis')
plt.colorbar(contour_original_p)
plt.title('HAD: Basınç [Pa]')
plt.fill(x_naca, y_naca, color='w', zorder=3)
# plt.gca().add_patch(circles[6])

# Basıncın tahmin edilen değeri
plt.subplot(3, 3, 8).set_aspect('equal', 'box')
if isLeveled:
    contour_predicted_p = plt.contourf(x_grid, y_grid, z_predicted_p, cmap='viridis', levels=level_p)
else:
    contour_predicted_p = plt.contourf(x_grid, y_grid, z_predicted_p, cmap='viridis')
plt.colorbar(contour_predicted_p)
plt.title('Tahmin: Basınç [Pa]')
plt.fill(x_naca, y_naca, color='w', zorder=3)
# plt.gca().add_patch(circles[7])

# Basıncın farkı
plt.subplot(3, 3, 9).set_aspect('equal', 'box')
contour_difference_p = plt.contourf(x_grid, y_grid, 
                                    abs((z_predicted_p - z_original_p)*100/(z_original_p+1e-10)), 
                                    cmap='viridis',
                                    levels=level_diff_p)
plt.colorbar(contour_difference_p)
plt.title('Fark: Basınç [%]')
plt.fill(x_naca, y_naca, color='w', zorder=3)
# plt.gca().add_patch(circles[8])

plt.tight_layout()
if savePost:
    plt.savefig(f"{directory}{model_name}-contour-aoa{test_data[aoa_post_index]:.1f}.png")
    
# plt.figure(figsize=(9, 2))
# plt.figure(6) if trainMode else plt.figure(2)
# # Original streamline
# plt.subplot(1, 2, 1).set_aspect('equal', 'box')
# vel_original = np.sqrt(z_original_u**2 + z_original_v**2)
# plt.fill(x_naca, y_naca, color='k', zorder=3)
# strm = plt.streamplot(x_grid, y_grid, z_original_u, z_original_v, density=(1,0.5), arrowstyle='-',
#                       color='k', broken_streamlines=False, linewidth=0.2, cmap='viridis')
# # plt.colorbar(strm.lines)
# plt.title('HAD Akış Çizgisi')
# # plt.gca().add_patch(circles[9])
# # Predicted streamline
# plt.subplot(1, 2, 2).set_aspect('equal', 'box')
# plt.fill(x_naca, y_naca, color='k', zorder=3)
# vel_predicted = np.sqrt(z_predicted_u**2 + z_predicted_v**2)
# strm = plt.streamplot(x_grid, y_grid, z_predicted_u, z_predicted_v, density=(1,0.5), arrowstyle='-',
#                       color='k', broken_streamlines=False, linewidth=0.2, cmap='viridis')
# # plt.colorbar(strm.lines)
# plt.title('Tahmin Akış Çizgisi')
# # plt.gca().add_patch(circles[10])
# plt.tight_layout()
# if savePost:
#     plt.savefig(f"{directory}{model_name}-streamline-aoa{test_data[aoa_post_index]:.1f}.png")
    
##################
### Extra-Post ###
##################

plt.figure(figsize=(9,6))
plt.figure(6) if trainMode else plt.figure(2)

# Original streamline
plt.subplot(2, 2, 1).set_aspect('equal', 'box')
vel_original = np.sqrt(z_original_u**2 + z_original_v**2)
plt.fill(x_naca, y_naca, color='k', zorder=3)
strm = plt.streamplot(x_grid, y_grid, z_original_u, z_original_v, density=(1,0.5), arrowstyle='-',
                      color='k', broken_streamlines=False, linewidth=0.2, cmap='viridis')
plt.title('HAD Akış Çizgisi')

# Predicted streamline
plt.subplot(2, 2, 2).set_aspect('equal', 'box')
plt.fill(x_naca, y_naca, color='k', zorder=3)
vel_predicted = np.sqrt(z_predicted_u**2 + z_predicted_v**2)
strm = plt.streamplot(x_grid, y_grid, z_predicted_u, z_predicted_v, density=(1,0.5), arrowstyle='-',
                      color='k', broken_streamlines=False, linewidth=0.2, cmap='viridis')
plt.title('Tahmin Akış Çizgisi')

# Original te
plt.subplot(2, 2, 3).set_aspect('equal', 'box')
vel_original = np.sqrt(z_original_u**2 + z_original_v**2)
plt.fill(x_naca, y_naca, color='k', zorder=3)
strm = plt.streamplot(x_grid, y_grid, z_original_u, z_original_v, density=(1,0.5), arrowstyle='-',
                      color='k', broken_streamlines=False, linewidth=0.2, cmap='viridis')
plt.title('HAD Akış Çizgisi')
plt.gca().set_xlim([0.55, 1.15])
plt.gca().set_ylim([-0.15, 0.25])

# Predicted te
plt.subplot(2, 2, 4).set_aspect('equal', 'box')
plt.fill(x_naca, y_naca, color='k', zorder=3)
vel_predicted = np.sqrt(z_predicted_u**2 + z_predicted_v**2)
strm = plt.streamplot(x_grid, y_grid, z_predicted_u, z_predicted_v, density=(2,0.5), arrowstyle='-',
                      color='k', broken_streamlines=False, linewidth=0.2, cmap='viridis')
plt.title('Tahmin Akış Çizgisi')
plt.gca().set_xlim([0.55, 1.15])
plt.gca().set_ylim([-0.15, 0.25])

plt.tight_layout()
if savePost:
    plt.savefig(f"{directory}{model_name}-streamline4-aoa{test_data[aoa_post_index]:.1f}.png")
    
############################
### Pressure-Coefficient ###
############################
dfp = pd.read_csv(f'test_data/cp-{test_data[aoa_post_index]:.1f}deg.csv')
x = dfp['x-coordinate'].values
y = dfp['y-coordinate'].values
p = dfp['pressure'].values

aoa_input = torch.tensor((np.ones(len(x))*scale(test_data[aoa_post_index], 'aoa')), dtype=torch.float32, device=device).view(-1, 1)
x_input = torch.tensor(scale(x, 'x'), dtype=torch.float32, device=device).view(-1, 1)
y_input = torch.tensor(scale(y, 'y'), dtype=torch.float32, device=device).view(-1, 1)
inputs = torch.cat((aoa_input, x_input, y_input), dim=1)
predicted = model(inputs).detach().cpu().numpy()

plt.figure(7) if trainMode else plt.figure(3)
plt.plot(x/1.07, 2*unscale(predicted[:, 2], 'p'), 'r^', label='Tahmin')
plt.plot(x/1.07, 2*p, '.', label='HAD')
plt.legend(fontsize='large')
plt.xlabel('X/C', fontsize='large')
plt.ylabel('Cp', fontsize='large')
plt.title(f'Hücum açısı = {test_data[aoa_post_index]:.1f} derece', fontsize='large')
plt.gca().set_xlim([-0.1, 1.1])
plt.gca().set_ylim([-1.2, 1.2])
if savePost:
    plt.savefig(f"{directory}{model_name}-cp-aoa{test_data[aoa_post_index]:.1f}.png")

dfcp = pd.DataFrame({'x-coordinate': x,
                    'y-coordinate': y,
                    'pressure': unscale(predicted[:, 2], 'p')})

dfcp.to_csv(f'{directory}{model_name}-predicted-cp-{test_data[aoa_post_index]:.1f}deg.csv', index=False)
    
# plt.figure(figsize=(9, 5))
# plt.figure(7) if trainMode else plt.figure(3)
# # Original vector
# plt.subplot(1, 2, 1).set_aspect('equal', 'box')
# vel_original = np.sqrt(z_original_u**2 + z_original_v**2)
# plt.fill(x, y, color='0.8', zorder=3)
# vect = plt.quiver(x_grid, y_grid, z_original_u/vel_original, z_original_v/vel_original, units='width')
# # plt.colorbar(strm.lines)
# plt.title('HAD Hız Vektörleri')
# # plt.gca().add_patch(circles[9])
# # Predicted streamline
# plt.subplot(1, 2, 2).set_aspect('equal', 'box')
# plt.fill(x, y, color='0.8', zorder=3)
# vel_predicted = np.sqrt(z_predicted_u**2 + z_predicted_v**2)
# vect = plt.quiver(x_grid, y_grid, z_predicted_u/vel_predicted, z_predicted_v/vel_predicted, units='width')
# # plt.colorbar(strm.lines)
# plt.title('Tahmin Hız Vektörleri')
# # plt.gca().add_patch(circles[10])
# plt.tight_layout()
# if savePost:
#     plt.savefig(f"{directory}{model_name}-vector-aoa{test_data[aoa_post_index]:.1f}.png")
    
# # Original streamline
# plt.figure(figsize=(7, 5))
# plt.figure(7) if trainMode else plt.figure(3)
# plt.axes().set_aspect('equal', 'box')
# vel_original = np.sqrt(z_original_u**2 + z_original_v**2)
# plt.fill(x, y, color='k', zorder=3)
# strm = plt.streamplot(x_grid, y_grid, z_original_u, z_original_v, density=3,
#                       color=vel_original, broken_streamlines=False, linewidth=1, cmap='viridis')
# # plt.colorbar(strm.lines)
# plt.title('HAD Akış Çizgisi')
# # plt.gca().add_patch(circles[11])
# if savePost:
#     plt.savefig(f"{directory}{model_name}-stro-aoa{test_data[aoa_post_index]:.1f}.png")

# # Predicted streamline
# plt.figure(figsize=(7, 5))
# plt.figure(8) if trainMode else plt.figure(4)
# plt.axes().set_aspect('equal', 'box')
# vel_original = np.sqrt(z_original_u**2 + z_original_v**2)
# plt.fill(x, y, color='k', zorder=3)
# vel_predicted = np.sqrt(z_predicted_u**2 + z_predicted_v**2)
# strm = plt.streamplot(x_grid, y_grid, z_predicted_u, z_predicted_v, density=3,
#                       color=vel_predicted, broken_streamlines=False, linewidth=1, cmap='viridis')
# # plt.colorbar(strm.lines)
# plt.title('Tahmin Akış Çizgisi')
# # plt.gca().add_patch(circles[12])
# if savePost:
#     plt.savefig(f"{directory}{model_name}-strp-aoa{test_data[aoa_post_index]:.1f}.png")


plt.show()
