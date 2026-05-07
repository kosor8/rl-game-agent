import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font(pygame.font.get_default_font(), 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Renkler
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

# Yeni Tema Renkleri
LIGHT_GREEN = (170, 215, 81)
DARK_GREEN = (162, 209, 73)
HEAD_COLOR_OUTER = (0, 50, 150)
HEAD_COLOR_INNER = (0, 200, 255)

BLOCK_SIZE = 30 # Orta karar büyüklük (Orijinali 20'ydi)
SPEED = 40 # İnsan için 10-15 ideal, yapay zeka eğitirken hızlandırmak için 40+ yapabiliriz

class SnakeGameAI:
    def __init__(self, w=960, h=720, render_mode=True):
        self.w = w
        self.h = h
        self.render_mode = render_mode
        # Ekranı oluştur
        if self.render_mode:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake RL Environment')
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self):
        # Oyun başlangıç durumu
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
            
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. Kullanıcı girdilerini kontrol et (Pencereyi kapatma vs.)
        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        
        # 2. Hareket et
        self._move(action) # Aksiyonu uygula
        self.snake.insert(0, self.head)
        
        # 3. Oyun bitti mi kontrol et
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10 # Ceza: Oyunu kaybetti
            return reward, game_over, self.score
            
        # 4. Yem yendi mi?
        if self.head == self.food:
            self.score += 1
            reward = 10 # Ödül: Yem yedi
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. Arayüzü güncelle ve saati ilerlet
        if self.render_mode:
            self._update_ui()
            self.clock.tick(SPEED)
        
        # 6. Ödülü ve durumu döndür
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Sınır (Duvar) çarpışması
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Kendine çarpma
        if pt in self.snake[1:]:
            return True
        return False
        
    def _update_ui(self):
        # Satranç tahtası (Zemin)
        for row in range(int(self.h / BLOCK_SIZE)):
            for col in range(int(self.w / BLOCK_SIZE)):
                color = LIGHT_GREEN if (row + col) % 2 == 0 else DARK_GREEN
                pygame.draw.rect(self.display, color, pygame.Rect(col * BLOCK_SIZE, row * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                
        # Yılanı çizdirme
        inner_offset = BLOCK_SIZE // 5
        inner_size = BLOCK_SIZE - (2 * inner_offset)
        
        for idx, pt in enumerate(self.snake):
            if idx == 0:
                # Baş kısmı
                pygame.draw.rect(self.display, HEAD_COLOR_OUTER, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, HEAD_COLOR_INNER, pygame.Rect(pt.x+inner_offset, pt.y+inner_offset, inner_size, inner_size))
            else:
                # Gövde kısmı
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+inner_offset, pt.y+inner_offset, inner_size, inner_size))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Skor: " + str(self.score), True, BLACK) # Yazı rengini zemine uyumlu siyah yaptık
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        # Aksiyon listesi: [Düz devam et, Sağa dön, Sola dön]
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # Değişim yok, düz devam
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # Sağa dön (Saat yönü)
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # Sola dön (Saat yönünün tersi)
            
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
