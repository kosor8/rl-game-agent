import torch
import random
import numpy as np
from collections import deque
from snake_game import Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 64
LR = 0.001

import os # Model yükleme yolunu kontrol etmek için

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0 # Başlangıçta tam rastgelelik (Exploration)
        self.gamma = 0.9 # İndirim oranı (discount rate)
        self.memory = deque(maxlen=MAX_MEMORY) # Replay Memory
        
        # Model (Policy Network)
        # 11 Girdi(state vektörü), 256 Gizli Katman, 3 Çıktı(Q-value tahmini)
        self.model = Linear_QNet(11, 256, 3)
        
        # Eğitime kalındığı yerden devam etmek için kayıtlı modeli yükle
        model_path = './model/model.pth'
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print(">>> Kaydedilmiş model başarıyla yüklendi! Eğitim kaldığı yerden devam ediyor... <<<")
            
            # Eğer daha önceden eğitilmiş model varsa çok fazla rastgele hamle yapmasına gerek yok
            # Yarı keşif yarı bildiğini okuma modunda başlasın (isteğe göre değiştirilebilir)
            self.epsilon = 0.10
        
        # Target Network (Hedef Ağ)
        self.target_model = Linear_QNet(11, 256, 3)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval() # Target network sadece tahmin içindir
        
        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma)
        
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        

    def get_state(self, game):
        """
        Oyunun mevcut durumunu (state) 11 elemanlı bir vektör olarak döner.
        Vektör içeriği:
        [
            Tehlike Düz, Tehlike Sağ, Tehlike Sol,
            Yön Sol, Yön Sağ, Yön Yukarı, Yön Aşağı,
            Yemek Sol, Yemek Sağ, Yemek Yukarı, Yemek Aşağı
        ]

        Bu vektör, ajanın çevresini ve hedefini anlaması için sadeleştirilmiş bir "gözlem" sunar. 
        Ajan, bu vektörü kullanarak hangi aksiyonu seçeceğine karar verir.
        
        Tehlike Düz: Yılan mevcut yönünde bir adım giderse çarpışır mı? (duvar veya kendine)
        Tehlike Sağ: Yılan sağa dönerse çarpışır mı?
        Tehlike Sol: Yılan sola dönerse çarpışır mı?
        Yön Sol: Yılan sola mı gidiyor?
        Yön Sağ: Yılan sağa mı gidiyor?
        Yön Yukarı: Yılan yukarı mı gidiyor?
        Yön Aşağı: Yılan aşağı mı gidiyor?
        Yemek Sol: Yem yılanın başının solunda mı?
        Yemek Sağ: Yem yılanın başının sağında mı?
        Yemek Yukarı: Yem yılanın başının yukarısında mı?
        Yemek Aşağı: Yem yılanın başının aşağısında mı?
        """
        head = game.snake[0]
        
        # Yılanın başının 1 blok etrafındaki noktalar
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        # Mevcut yön bilgisi (Sadece biri True olacak)
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        # Durumları belirleme
        state = [
            # 1. Tehlike (Düz)
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),
            
            # 2. Tehlike (Sağ)
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),
            
            # 3. Tehlike (Sol)
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # 4. Yön (Sol)
            dir_l,
            # 5. Yön (Sağ)
            dir_r,
            # 6. Yön (Yukarı)
            dir_u,
            # 7. Yön (Aşağı)
            dir_d,
            
            # 8. Yemek Solumuzda mı?
            game.food.x < game.head.x,
            # 9. Yemek Sağımızda mı?
            game.food.x > game.head.x,
            # 10. Yemek Yukarımızda mı?
            game.food.y < game.head.y,
            # 11. Yemek Aşağıda mı?
            game.food.y > game.head.y
        ]
        
        # True/False değerlerini 1 ve 0'a çevirerek Numpy array olarak döndür
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # deque sağdan ekler, maxlen aşılırsa soldan siler
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # Rastgele batch
        else:
            mini_sample = self.memory

        # Zip ile mini_sample içindeki tuple'ları ayrı listelere (batch) ayır
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Epsilon-Greedy
        final_move = [0, 0, 0]

        # random.random() 0.0 ile 1.0 arasında bir değer üretir.
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Tahmin (Prediction)
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
