from snake_game import SnakeGameAI
import random
import pygame

def test():
    # Oyunu başlat (Hızı test için biraz daha yavaş görebilmek adına)
    game = SnakeGameAI()
    
    while True:
        # Rastgele bir aksiyon seç:
        # [1, 0, 0] -> Düz devam et
        # [0, 1, 0] -> Sağa dön
        # [0, 0, 1] -> Sola dön
        action = [0, 0, 0]
        move_idx = random.randint(0, 2)
        action[move_idx] = 1
        
        # Oyunda adımı uygula
        reward, game_over, score = game.play_step(action)
        
        # Görsel olarak takip edebilmek için biraz gecikme ekleyelim
        if getattr(game, 'render_mode', True):
            pygame.time.delay(50) 
        
        if game_over:
            print(f"Oyun Bitti! Skor: {score}")
            game.reset()

if __name__ == '__main__':
    test()
