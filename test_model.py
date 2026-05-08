import pygame
from agent import Agent
from snake_game import SnakeGameAI

def test():
    # 1. Ajanı oluştur
    agent = Agent()
    
    # 2. Epsilon'u 0 yapıyoruz! 
    # Bu sayede ajan HİÇBİR ŞEKİLDE rastgele hamle yapmayacak (Keşif/Exploration kapalı).
    # Sadece ve sadece modelin (beynin) ona dikte ettiği en mantıklı hamleyi seçecek.
    agent.epsilon = 0
    
    # 3. Oyunu görselleştirme açık şekilde başlatıyoruz
    game = SnakeGameAI(render_mode=True)
    
    print("\n" + "="*50)
    print(">>> TEST MODU BAŞLADI <<<")
    print("Yılan tamamen öğrendiklerini (Exploitation) uyguluyor...")
    print("="*50 + "\n")
    
    while True:
        # Mevcut durumu al
        state = agent.get_state(game)
        
        # Modelden en iyi hamleyi iste
        final_move = agent.get_action(state)
        
        # Hamleyi oyunda uygula
        reward, done, score = game.play_step(final_move)
        
        # Hareketleri rahat izleyebilmek için araya ufak bir gecikme ekliyoruz
        # (İsterseniz bu satırı silebilir veya süreyi değiştirebilirsiniz)
        pygame.time.delay(30)
        
        if done:
            print(f"Oyun Bitti! Ulaşılan Skor: {score}")
            game.reset()
            # Oyun bittiğinde baştan başlar, ajan yeniden yeteneklerini sergiler.

if __name__ == '__main__':
    test()
