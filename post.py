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

###########################
### MODEL PARAMETRELERİ ###
###########################
hidden_layers = [512, 256, 128, 64] 
batch_size = 256
learning_rate = 0.001
epochs = 4000
model_name = 'model17'
directory = 'models/activation_functions/'
###########################

# Check that MPS is available
device="cpu"

# Rastgele başlangıçların her seferinde aynı olmasını sağlamak için seed değerlerini ayarlayalım
seed = 13
torch.manual_seed(seed)
np.random.seed(seed)

# 1. Veri setini CSV dosyasından okuma
df = pd.read_csv('train_data/cylstd_2.csv')

# Verileri numpy dizilerine dönüştürme
re = df['re'].values
x_coord = df['x-coordinate'].values
y_coord = df['y-coordinate'].values
x_vel = df['x-velocity'].values
y_vel = df['y-velocity'].values
pressure = df['pressure'].values
vel_mag = np.sqrt(x_vel**2 + y_vel**2) 

# Veriyi normalize etme
re_norm = re / 20
x_coord_norm = x_coord / 10
y_coord_norm = y_coord / 10
# x_vel_norm = (x_vel - np.mean(x_vel)) / np.std(x_vel)
# y_vel_norm = (y_vel - np.mean(y_vel)) / np.std(y_vel)
# pressure_norm = (pressure - np.mean(pressure)) / np.std(pressure)

# mean_re, std_re = np.mean(re), np.std(re)
# mean_x, std_x = np.mean(x_coord), np.std(x_coord)
# mean_y, std_y = np.mean(y_coord), np.std(y_coord)
# mean_u, std_u = np.mean(x_vel), np.std(x_vel)
# mean_v, std_v = np.mean(y_vel), np.std(y_vel)
# mean_p, std_p = np.mean(pressure), np.std(pressure)
# x_norm_min = np.min(x_coord_norm)
# x_norm_max = np.max(x_coord_norm)

# Veriyi PyTorch tensörlerine dönüştürme
# (inputlar normalize, outputlar gerçek değerleri)
inputs = torch.tensor(np.vstack((re_norm, x_coord_norm, y_coord_norm)).T, dtype=torch.float32, device=device)
outputs = torch.tensor(np.vstack((x_vel, y_vel, pressure)).T, dtype=torch.float32, device=device)

# Veri seti ve DataLoader oluşturma
dataset = TensorDataset(inputs, outputs)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
            phi = F.sigmoid(layer(phi))
        phi = self.output_layer(phi)
        return phi

model = FlexibleNN(input_size=3, hidden_layers=hidden_layers, output_size=3).to(device)

# 3. Kayıp fonksiyonu ve optimizer'ı tanımlama
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer kullanımı

# 4. Modeli eğitme
# lossLog=np.zeros(epochs)
# ulog = np.zeros(epochs)
# vlog = np.zeros(epochs)
# plog = np.zeros(epochs)
# for epoch in range(epochs):
#     size = len(data_loader.dataset)
#     model.train()
#     for batch, (inputs, targets) in enumerate(data_loader):
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = loss_fn(outputs, targets)
#         loss.backward()
#         optimizer.step()
        
#         # if batch % 20 == 0:
#         #     loss, current = loss.item(), (batch + 1) * len(inputs)
#         #     print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
#     monitor_inputs = torch.tensor([(20-10)/20, 0.1, 0.1], dtype=torch.float32, device=device)
#     with torch.no_grad():
#         monitor_predicted = model(monitor_inputs)
#         ulog[epoch], vlog[epoch], plog[epoch] = monitor_predicted.to("cpu").numpy()
    
#     if (epoch+1) % 100 == 0:  # Daha az sıklıkla çıktı veriyoruz
#         print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.7f}')
#     lossLog[epoch] = loss.item()

# # Kaybı çizdir
# plt.figure(1)
# plt.plot(np.arange(epochs), lossLog)
# plt.yscale("log")
# plt.ylim(1e-7,1e-2)
# plt.xlabel("Eopchs")
# plt.ylabel("Loss")
# plt.grid(color='0.95')
# plt.savefig(fname=f"models/hidden_layers/{model_name}-loss.png")

# # Monitörleri çizdir
# plt.figure(2)
# plt.plot(np.arange(epochs), ulog)
# plt.xlabel("Eopchs")
# plt.ylabel("u_log")
# plt.grid(color='0.95')
# plt.savefig(fname=f"models/hidden_layers/{model_name}-ulog.png")

# plt.figure(3)
# plt.plot(np.arange(epochs), vlog)
# plt.xlabel("Eopchs")
# plt.ylabel("v_log")
# plt.grid(color='0.95')
# plt.savefig(fname=f"models/hidden_layers/{model_name}-vlog.png")

# plt.figure(4)
# plt.plot(np.arange(epochs), plog)
# plt.xlabel("Eopchs")
# plt.ylabel("p_log")
# plt.grid(color='0.95')
# plt.savefig(fname=f"models/hidden_layers/{model_name}-plog.png")
 
# # Modeli kaydetme       
# torch.save(model.state_dict(), f"models/hidden_layers/{model_name}.pth")
# print(f"Saved PyTorch Model State to {model_name}.pth")

# Modeli yükleme
model.load_state_dict(torch.load(f"{directory}{model_name}.pth"))

# 5. Modelin performansını değerlendirme
model.eval()

# Post-işlem koordinatları
re_post = 20
x_post_min = -5
x_post_max = 5
y_post_min = -5
y_post_max = 5

# Koordinatları grid formatına dönüştürme
x_unique = np.linspace(x_post_min/10, x_post_max/10, 101)
y_unique = np.linspace(y_post_min/10, y_post_max/10, 101)
x_grid, y_grid = np.meshgrid(x_unique, y_unique)
x_grid_flatten = x_grid.flatten()
y_grid_flatten = y_grid.flatten()

# Post koordinatları kullanarak tahmin 
re_input = torch.tensor((np.ones(101*101)*re_post)/20, dtype=torch.float32, device=device).view(-1, 1)
x_input = torch.tensor(x_grid_flatten, dtype=torch.float32, device=device).view(-1, 1)
y_input = torch.tensor(y_grid_flatten, dtype=torch.float32, device=device).view(-1, 1)
inputs = torch.cat((re_input, x_input, y_input), dim=1)
predicted = model(inputs).detach().numpy()
predicted_u = predicted[:, 0]
predicted_v = predicted[:, 1]
predicted_p = predicted[:, 2]

# 6. Sonuçları görselleştirme
plt.figure(figsize=(12, 12))

# Orijinal veriyi filtreleme
df_filtered = df[(df['re'] == re_post) & (df['x-coordinate'].between(-5, 5)) & (df['y-coordinate'].between(-5, 5))]
x_filtered = df_filtered['x-coordinate'].values
y_filtered = df_filtered['y-coordinate'].values
u_filtered = df_filtered['x-velocity'].values
v_filtered = df_filtered['y-velocity'].values
p_filtered = df_filtered['pressure'].values


# Z değerlerini hesaplama
z_original_u = griddata((x_filtered, y_filtered), u_filtered, (x_grid*10, y_grid*10), method='linear')
z_original_v = griddata((x_filtered, y_filtered), v_filtered, (x_grid*10, y_grid*10), method='linear')
z_original_p = griddata((x_filtered, y_filtered), p_filtered, (x_grid*10, y_grid*10), method='linear')

z_predicted_u = griddata((x_grid_flatten, y_grid_flatten), predicted[:, 0], (x_grid, y_grid), method='linear')
z_predicted_v = griddata((x_grid_flatten, y_grid_flatten), predicted[:, 1], (x_grid, y_grid), method='linear')
z_predicted_p = griddata((x_grid_flatten, y_grid_flatten), predicted[:, 2], (x_grid, y_grid), method='linear')

# Silindir
circle1 = Circle((0, 0), 0.5, color='white', fill=True, zorder=10)
circle2 = Circle((0, 0), 0.5, color='white', fill=True, zorder=10)
circle3 = Circle((0, 0), 0.5, color='white', fill=True, zorder=10)
circle4 = Circle((0, 0), 0.5, color='white', fill=True, zorder=10)
circle5 = Circle((0, 0), 0.5, color='white', fill=True, zorder=10)
circle6 = Circle((0, 0), 0.5, color='white', fill=True, zorder=10)

plt.figure(1)
# x-hızının orijinal değeri
level_u = np.linspace(-0.03, 0.24, 10)
plt.subplot(3, 2, 1)
contour_original_u = plt.contourf(x_grid*10, y_grid*10, z_original_u, cmap='viridis')
# contour_original_u = plt.contourf(x_grid*10, y_grid*10, z_original_u, 
#                                   levels=level_u, 
#                                   cmap='viridis',
#                                   extend='both')
plt.colorbar(contour_original_u)
plt.title('Original x-velocity')
plt.gca().add_patch(circle1)
# plt.clim(-0.05, 0.4)

# x-hızının tahmin edilen değeri
plt.subplot(3, 2, 2)
# contour_predicted_u = plt.contourf(x_grid*10, y_grid*10, z_predicted_u, cmap='viridis')
contour_predicted_u = plt.contourf(x_grid*10, y_grid*10, z_predicted_u, 
                                   levels=level_u, 
                                   cmap='viridis',
                                   extend='neither')
plt.colorbar(contour_predicted_u)
plt.title('Predicted x-velocity')
plt.gca().add_patch(circle2)
# plt.clim(-0.05, 0.4)

# y-hızının orijinal değeri
level_v = np.linspace(-0.2, 0.2, 9)
plt.subplot(3, 2, 3)
contour_original_v = plt.contourf(x_grid*10, y_grid*10, z_original_v, cmap='viridis')
# contour_original_v = plt.contourf(x_grid*10, y_grid*10, z_original_v,levels=level_v, cmap='viridis')
plt.colorbar(contour_original_v)
plt.title('Original y-velocity')
plt.gca().add_patch(circle3)
# plt.clim(-0.16, 0.16)

# y-hızının tahmin edilen değeri
plt.subplot(3, 2, 4)
contour_predicted_v = plt.contourf(x_grid*10, y_grid*10, z_predicted_v, cmap='viridis')
# contour_predicted_v = plt.contourf(x_grid*10, y_grid*10, z_predicted_v,levels=level_v, cmap='viridis')
plt.colorbar(contour_predicted_v)
plt.title('Predicted y-velocity')
plt.gca().add_patch(circle4)
# plt.clim(-0.16, 0.16)

# Basıncın orijinal değeri
level_p = np.linspace(-0.045, 0.06, 9)
plt.subplot(3, 2, 5)
contour_original_p = plt.contourf(x_grid*10, y_grid*10, z_original_p, cmap='viridis')
plt.colorbar(contour_original_p)
plt.title('Original pressure')
plt.gca().add_patch(circle5)
# plt.clim(-0.05, 0.4)

# Basıncın tahmin edilen değeri
plt.subplot(3, 2, 6)
contour_predicted_p = plt.contourf(x_grid*10, y_grid*10, z_predicted_p, cmap='viridis')
plt.colorbar(contour_predicted_p)
plt.title('Predicted pressure')
plt.gca().add_patch(circle6)

plt.tight_layout()
# plt.savefig(f"{directory}{model_name}-contour.png")
# plt.show()