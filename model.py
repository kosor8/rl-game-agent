import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sistemin beyni
# amaç : verilen state vektörünü alıp, olası aksiyonlar için Q değerlerini(kalite değerlerini) hesaplamak.
# çıktı: olası hamlelerin (örn: sağa dön, sola dön, düz git) her biri için bir skor tahmini üretir.
#        ajan genelde en yüksek Q skoruna sahip hamleyi seçer.
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Dense -> ReLU -> Dense -> ReLU -> Çıktı yapısı
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


# Öğrenme Algoritması (Optimizasyon)
# modelin hatalarından ders çıkarmasını sağlayan matematiksel güncellemeleri yapar
# yaptığı tahmin ile yapması gereken ideal tahmin arasındaki farkı (Loss) hesaplar 
# ağın ağırlıklarını (W ve b) geriye dönük olarak günceller (Backpropagation)
class QTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # PyTorch uyarısını önlemek için listeleri önce numpy array'e sonra tensöre çeviriyoruz
        state = torch.tensor(np.array(state), dtype=torch.float).to(device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(device)
        action = torch.tensor(np.array(action), dtype=torch.long).to(device)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(device)
        
        # Eğer veri 1 boyutlu geldiyse (tekil state), batch boyutu (1, x) yap
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, ) # Tuple içine al
            
        # 1: Mevcut state üzerinden Q değerleri (Prediction)
        pred = self.model(state)
        
        # 2: Q_new = R + gamma * max(next_predicted_Q) (Target Network kullanarak)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Hedef ağını kullanarak sonraki durumun maksimum Q değerini al
                Q_new = reward[idx] + self.gamma * torch.max(self.target_model(next_state[idx]))
                
            # Yalnızca seçilen aksiyonun olduğu indeksi güncelle
            target[idx][torch.argmax(action[idx]).item()] = Q_new
            
        # 3: Kayıp hesapla ve ağ ağırlıklarını güncelle (Backpropagation)
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
