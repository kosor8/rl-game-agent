# Hiperparametre Optimizasyonu (Tuning Rehberi) 

# agent.py içerisindeki ayarlar:

# LR = 0.001 (Öğrenme Oranı - alpha): Hatalarından ne kadar hızlı ders çıkaracağını belirler. 
# 0.01 yaparsanız çok hızlı ama dengesiz öğrenir (hemen unutur),
# 0.0001 yaparsanız çok yavaş ama sağlam öğrenir.

# gamma = 0.9 (İndirim Faktörü - gamma): Gelecek planlamasıdır. 
# 0.99 yaparsanız gelecekteki çok uzak bir yeme gitmeyi bile şimdiki ufak bir riske tercih eder. 
# 0.1 yaparsanız sadece "burnunun ucunu" düşünür.

# BATCH_SIZE = 64: Replay Memory'den rastgele çekilip eğitilecek anı sayısı. 
# 128 veya 256 yaparak daha dengeli bir eğitim deneyebilirsiniz (ama işlemci/GPU'yu biraz daha yorar).
# train.py içerisindeki ayarlar:

# epsilon_decay = 0.99: Rastgeleliğin düşüş hızı. 0.995 yaparsanız ajana daha çok keşif yapma şansı tanırsınız (Ajan 300 oyun boyunca etrafı kurcalar).
# TARGET_UPDATE_FREQUENCY = 10: Hedef ağı kaç oyunda bir güncelleyeceğiniz. Eğer yılan olduğu yerde anlamsızca takılıyorsa, bu sayıyı 20 veya 30 yaparak hedeflerin daha sabit kalmasını sağlayabilirsiniz.

# train.py içerisindeki ayarlar:

# epsilon_decay = 0.99: Rastgeleliğin düşüş hızı.
# 0.995 yaparsanız ajana daha çok keşif yapma şansı tanırsınız (Ajan 300 oyun boyunca etrafı kurcalar).

# TARGET_UPDATE_FREQUENCY = 10: Hedef ağı kaç oyunda bir güncelleyeceğiniz.
# Eğer yılan olduğu yerde anlamsızca takılıyorsa, bu sayıyı 20 veya 30 yaparak hedeflerin daha sabit kalmasını sağlayabilirsiniz.



import matplotlib.pyplot as plt

plt.ion() # İnteraktif modu aç, böylece grafik program akışını durdurmaz

def plot(scores, mean_scores, epsilons):
    plt.clf()
    
    fig, ax1 = plt.subplots(num=1) # 1 numaralı figürü sürekli günceller
    
    # 1. Eksen: Skor ve Ortalama Skor
    ax1.set_title('Snake DQN Eğitim Süreci')
    ax1.set_xlabel('Oyun Sayısı')
    ax1.set_ylabel('Skor', color='tab:blue')
    ax1.plot(scores, label='Anlık Skor', color='lightblue')
    ax1.plot(mean_scores, label='Ortalama Skor', color='blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(ymin=0)
    
    # Son değerleri grafiğin sağına text olarak yazdır
    if len(scores) > 0:
        ax1.text(len(scores)-1, scores[-1], str(scores[-1]))
        ax1.text(len(mean_scores)-1, mean_scores[-1], str(round(mean_scores[-1], 2)))
        
    # 2. Eksen: Epsilon Değeri
    ax2 = ax1.twinx() # Ortak x ekseni kullanan ikinci y ekseni
    ax2.set_ylabel('Epsilon (\u03B5)', color='tab:red')
    ax2.plot(epsilons, label='Epsilon', color='red', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0, 1.05)
    
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
