# Deep Q-Learning Snake Game Agent 🐍🧠

Bu proje, klasik Snake (Yılan) oyununu deneyimleyerek kendi kendine öğrenen bir **Pekiştirmeli Öğrenme (Reinforcement Learning)** ajanıdır. Ajan, **Deep Q-Network (DQN)** algoritmasını kullanarak PyTorch ile sıfırdan geliştirilmiştir.

## Özellikler ✨

- **Derin Sinir Ağı (DQN):** Çevre durumunu (state) 11 elemanlı bir vektör olarak alıp, hareket (action) kalitesini (Q-Value) tahmin eden Feed-Forward ağ.
- **Replay Memory:** Ajanın geçmiş deneyimlerini kaydeden ve eğitim sırasındaki korelasyonu kırmak için rastgele (mini-batch) örneklemeler yapan bellek.
- **İki Ağlı Yapı (Target Network):** Öğrenme stabilitesini artırmak ve ajanın "kendi kuyruğunu kovalamasını" önlemek için belirli periyotlarda güncellenen donuk hedef sinir ağı.
- **Epsilon-Greedy Stratejisi:** Rastgelelik oranı ($\epsilon$) yavaşça düşürülerek keşif (Exploration) ve sömürü (Exploitation) dengesinin kurulması.
- **Canlı Grafik Loglama:** Eğitim devam ederken skorları, ortalama skoru ve Epsilon düşüşünü anlık olarak ekrana çizen Matplotlib destekli görselleştirme.
- **Kaldığı Yerden Devam Etme:** Rekor kırıldığında ağırlıklar (`model.pth`) ve rekor skoru (`record.txt`) senkronize olarak kaydedilir. Eğitimi durdurup tekrar başlattığınızda ajan önceki deneyimlerini (weights) hatırlayarak şapşal hamleleri atlar ve eğitimine kaldığı yerden devam eder.
- **Gelişmiş Arayüz:** 960x720 çözünürlüğünde, yılanın başını gövdesinden ayırt eden, satranç tahtası desenli (checkerboard) şık ve okunaklı bir Pygame oyun ekranı.

## Kurulum 🛠️

Projeyi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin.

1. Projeyi bilgisayarınıza klonlayın:
   ```bash
   git clone <REPO-URL>
   cd rl-game-agent
   ```

2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

## Kullanım 🚀

### 1. Eğitim Modu (Train)
Ajanı eğitmeye başlamak ve canlı öğrenme grafiğiyle birlikte görsel şöleni izlemek için ana dosyayı çalıştırın:

```bash
python train.py
```

*Not:* Eğitimi ekran çizimleri olmadan donanımsal olarak en yüksek hızda (ışık hızında) yapmak isterseniz, `train.py` dosyası içerisindeki `SnakeGameAI(render_mode=True)` değerini `False` olarak değiştirebilirsiniz.

### 2. Test Modu (Sadece Bildiklerini Oynatma)
Eğitimleri tamamladınız ve rekor kıran süper zeki yılanınızın sadece yeteneklerini sergilemesini mi istiyorsunuz? Rastgeleliğin (Exploration) tamamen kapalı olduğu ve ajanın sadece güvendiği (en yüksek Q-Value) kararları verdiği test modunu çalıştırın:

```bash
python test_model.py
```

## Hiperparametre Optimizasyonu (Tuning) ⚙️

Eğitimin seyrini ve ajanın karakterini değiştirmek için projedeki şu parametrelerle oynayabilirsiniz:

- **`agent.py` içerisindeki ayarlar:**
  - **`LR = 0.001` (Öğrenme Oranı - $\alpha$):** Ajanın hatalarından ne kadar hızlı ders çıkaracağını belirler. 
  - **`gamma = 0.9` (İndirim Faktörü - $\gamma$):** Gelecek planlamasını etkiler. `0.99` gelecekteki ödüllere odaklanırken, `0.1` kısa vadeli (anlık) ödülleri tercih etmesini sağlar.
  - **`BATCH_SIZE = 64`:** Replay Memory'den rastgele çekilip eğitilecek anı kümesinin büyüklüğü.

- **`train.py` içerisindeki ayarlar:**
  - **`epsilon_decay = 0.99`:** Rastgeleliğin ($\epsilon$) düşüş hızı. Ajanın dünyayı keşfetme süresini ayarlar.
  - **`TARGET_UPDATE_FREQUENCY = 10`:** Target Network'ün kaç oyunda bir güncelleneceği. Stabilitenin sağlanmasındaki anahtardır.

## Lisans 📝
Bu proje açık kaynaklıdır ve eğitim/geliştirme amacıyla serbestçe kullanılabilir.
