from agent import Agent
from snake_game import SnakeGameAI
from helper import plot

TARGET_UPDATE_FREQUENCY = 10 # Her 10 oyunda bir Target Network'ü güncelle

import os

def train():
    record = 0
    agent = Agent()
    game = SnakeGameAI(render_mode=True) # Eğitim sürecini görsel olarak izlemek için açtık
    
    # Eğer daha önce rekor kırılmışsa ve kaydedilmişse oku
    if os.path.exists('./model/record.txt'):
        try:
            with open('./model/record.txt', 'r') as f:
                record = int(f.read().strip())
                print(f"*** Önceki Rekor ({record}) Yüklendi! Artık model sadece bu skoru geçerse güncellenecek. ***")
        except:
            pass
    
    # Epsilon decay ayarları
    epsilon_min = 0.01
    epsilon_decay = 0.99
    
    # Grafikler için listeler
    plot_scores = []
    plot_mean_scores = []
    plot_epsilons = []
    total_score = 0
    
    print("Eğitim Başlıyor...")
    plot(plot_scores, plot_mean_scores, plot_epsilons) # Grafiği en başta boş olarak aç
    
    while True:
        # 1. Mevcut durumu (state) al
        state_old = agent.get_state(game)
        
        # 2. Ajanın bu duruma göre bir aksiyon (action) seçmesi (Epsilon-Greedy)
        final_move = agent.get_action(state_old)
        
        # 3. Aksiyonu oyunda uygula ve yeni durumu, ödülü, oyunun bitip bitmediğini al
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # 4. Ağı sadece o anki adım için eğit (Kısa Süreli Hafıza)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # 5. Deneyimi daha sonra kullanmak üzere belleğe kaydet (Replay Memory)
        agent.remember(state_old, final_move, reward, state_new, done)
        
        # Oyun Bittiğinde:
        if done:
            game.reset()
            agent.n_games += 1
            
            # Epsilon Decay: Rastgeleliği her oyun bitişinde yavaşça azalt
            if agent.epsilon > epsilon_min:
                agent.epsilon *= epsilon_decay
            else:
                agent.epsilon = epsilon_min
                
            # Uzun Süreli Hafızayı (Replay Memory) kullanarak mini-batch eğitimi yap
            agent.train_long_memory()
            
            # Belirli periyotlarda (örn: 10 oyun) Target Network ağırlıklarını güncelle
            if agent.n_games % TARGET_UPDATE_FREQUENCY == 0:
                agent.update_target_network()
            
            # Rekor kırılırsa modeli kaydet
            if score > record:
                record = score
                agent.model.save()
                
                # Yeni rekoru text dosyasına da yaz
                with open('./model/record.txt', 'w') as f:
                    f.write(str(record))
                
            total_score += score
            mean_score = total_score / agent.n_games
            
            # Grafiğe eklenecek değerleri kaydet
            plot_scores.append(score)
            plot_mean_scores.append(mean_score)
            plot_epsilons.append(agent.epsilon)
            
            print(f"Oyun: {agent.n_games} | Skor: {score} | Rekor: {record} | Ortalama Skor: {mean_score:.2f} | Epsilon: {agent.epsilon:.3f}")
            
            # Grafiği güncelle
            plot(plot_scores, plot_mean_scores, plot_epsilons)

if __name__ == '__main__':
    train()
