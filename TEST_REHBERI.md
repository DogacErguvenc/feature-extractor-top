# 🧪 Terazi AI Sistemi - Detaylı Test Rehberi

## 📋 Test Öncesi Hazırlık

### ✅ Kontrol Listesi
- [ ] Terazi Windows açık ve çalışıyor
- [ ] İnternet bağlantısı var (OpenAI API için)
- [ ] Python 3.11+ kurulu
- [ ] Node.js 18+ kurulu
- [ ] Yarn kurulu
- [ ] MongoDB kurulu ve çalışıyor

---

## 📦 ADIM 1: Projeyi Terازiye Aktarma

### 1.1 GitHub'dan İndirme (Önerilen)

```cmd
# Desktop'a indir
cd C:\Users\%USERNAME%\Desktop

# GitHub reposunu clone'la
git clone [GITHUB_REPO_URL] terazi-ai

# Klasöre gir
cd terazi-ai
```

**Git yoksa:** Projeyi ZIP olarak indirin ve `C:\Users\[Kullanıcı]\Desktop\terazi-ai` klasörüne çıkartın.

### 1.2 Sistem Kontrolü

```cmd
# Test script'ini çalıştır
test_terazi.bat
```

**Beklenen Çıktı:**
```
========================================
    TERAZI AI SISTEMI TESTI
========================================

[1/5] MongoDB testi...
  ✓ BAŞARILI: MongoDB çalışıyor

[2/5] Python testi...
  Python 3.11.x
  ✓ BAŞARILI: Python bulundu

[3/5] Node.js testi...
  v18.x.x
  ✓ BAŞARILI: Node.js bulundu

[4/5] Yarn testi...
  1.22.x
  ✓ BAŞARILI: Yarn bulundu

[5/5] Backend API testi...
  ⚠ UYARI: Backend çalışmıyor (henüz başlatılmamış)

========================================
    TEST TAMAMLANDI
========================================
```

**Hata varsa:** `TERAZI_KURULUM.md` dosyasını okuyun ve eksik yazılımları kurun.

---

## 🔧 ADIM 2: Kurulum

### 2.1 Backend Kurulumu

```cmd
cd backend
pip install -r requirements.txt
```

**Beklenen çıktı:**
```
```

**Hata alırsanız:**
```cmd
# Python versiyonunu kontrol edin
python --version

# Pip'i güncelleyin
python -m pip install --upgrade pip

# Tekrar deneyin
pip install -r requirements.txt
```

### 2.2 Frontend Kurulumu

```cmd
cd ..\frontend
yarn install
```

**Beklenen çıktı:**
```
[1/4] Resolving packages...
[2/4] Fetching packages...
[3/4] Linking dependencies...
[4/4] Building fresh packages...
✨ Done in X.XXs.
```

### 2.3 Yapılandırma Kontrolü

Backend `.env` kontrol:
```cmd
cd ..\backend
type .env
```

**Olması gereken:**
```
MONGO_URL=mongodb://localhost:27017
DB_NAME=terazi_production
API_KEY=YOUR_API_KEY
ALLOWED_IMAGE_DIR=backend\incoming
DISABLE_DOCS=true
CORS_ORIGINS=http://localhost:3000
```

Frontend `.env` kontrol:
```cmd
cd ..\frontend
type .env
```

**Olması gereken:**
```
REACT_APP_BACKEND_URL=http://localhost:8001
REACT_APP_API_KEY=YOUR_API_KEY
WDS_SOCKET_PORT=3000
```

**Not:** IP üzerinden erişim için `REACT_APP_BACKEND_URL=http://[TERAZI_IP]:8001`

---

## 🚀 ADIM 3: Sistemi Başlatma

### 3.1 Otomatik Başlatma (Önerilen)

```cmd
cd C:\Users\%USERNAME%\Desktop\terazi-ai
start_terazi.bat
```

**Ne olacak:**
1. MongoDB başlatılacak
2. Backend penceresi açılacak (siyah terminal)
3. 5 saniye bekleme
4. Frontend penceresi açılacak (siyah terminal)
5. Tarayıcı otomatik açılacak: `http://localhost:3000`

**Her iki pencereyi de açık tutun!**

### 3.2 Manuel Başlatma (Alternatif)

**Terminal 1: Backend**
```cmd
cd C:\Users\%USERNAME%\Desktop\terazi-ai\backend
python -m uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

**Beklenen çıktı:**
```
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Terminal 2: Frontend** (yeni terminal açın)
```cmd
cd C:\Users\%USERNAME%\Desktop\terazi-ai\frontend
yarn start
```

**Beklenen çıktı:**
```
Compiled successfully!

You can now view frontend in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.x.x:3000
```

Tarayıcı otomatik açılacak.

---

## 🎯 ADIM 4: İlk PLU Oluşturma

### 4.1 Web Arayüzünü Açın

Tarayıcıda: `http://localhost:3000`

**Göreceğiniz ekran:**
- ✅ Üstte navigasyon: "⚖️ Terazi AI | 🏪 Terazi Ekranı | 📊 Dashboard | 🏷️ PLU Yönetimi"
- ✅ Ana sayfa: "Henüz PLU eklenmemiş" mesajı

### 4.2 PLU Yönetimi'ne Gidin

Üst menüden **"🏷️ PLU Yönetimi"** → Tıklayın

### 4.3 İlk PLU'yu Oluşturun

**"+ Yeni PLU Ekle"** butonuna tıklayın

**Form açılacak:**

**PLU Kodu:** `101`

**Ürün Adı:** `Dana Kıyma`

**Açıklama:** 
```
Taze dana kıyma, orta yağlı, parlak kırmızı renk, ince çekilmiş, homojen doku, yağ oranı %15-20, taze koku, pembe-kırmızı arası ton
```

**ÖNEMLİ:** Açıklama ne kadar detaylı olursa AI o kadar doğru çalışır!

**"✓ PLU Ekle"** butonuna tıklayın

**Sonuç:** Toast mesajı: "✅ PLU başarıyla eklendi!"

### 4.4 Daha Fazla PLU Ekleyin

Test için en az 3 PLU oluşturun:

**PLU 102 - Kuzu Pirzola:**
```
Taze kuzu pirzola, kemikli, açık pembe-beyaz renk, mermer dokusu, ince yağ tabakası, kemik etrafında et oranı yüksek, taze koku
```

**PLU 103 - Tavuk Göğüs:**
```
Taze tavuk göğüs, derisize, açık pembe renk, beyaz-pembe ton, düz yüzey, lifli doku, yağsız, taze koku, nem oranı orta
```

### 4.5 PLU'ları Kontrol Edin

PLU Yönetimi sayfasında 3 PLU card'ı görmelisiniz:
- PLU 101 - Dana Kıyma
- PLU 102 - Kuzu Pirzola  
- PLU 103 - Tavuk Göğüs

---

## 📸 ADIM 5: Kamera Testi

### 5.1 Kamera Bağlantısını Test Edin

Yeni bir **Command Prompt** açın:

```cmd
curl -H "x-api-key: YOUR_API_KEY" -X POST http://localhost:8001/api/camera/test
```

**Başarılı yanıt:**
```json
{
  "message": "Camera test successful",
  "image_preview": "/9j/4AAQSkZJRg..."
}
```

**Kamera yoksa:** Sistem otomatik mock görüntü oluşturacak (bu normal, devam edin)

### 5.2 Backend Loglarını Kontrol Edin

Backend terminalinde şunu görmelisiniz:
```
WARNING:  Cannot open camera - using mock image for demo
```

veya

```
INFO:  Camera captured successfully
```

---

## 🏪 ADIM 6: Terazi Ekranından Test (Eğitim Modu)

### 6.1 Terazi Ekranına Gidin

Web'de üst menüden **"🏪 Terazi Ekranı"** → Tıklayın

**Göreceğiniz:**
- ✅ Sağ üstte: "🟡 Eğitim Modu" badge'i
- ✅ 3 PLU kartı (101, 102, 103)
- ✅ Her kartta "✓ Seç" butonu

### 6.2 İlk PLU Seçimi

**PLU 101 (Dana Kıyma)** kartında **"✓ Seç"** butonuna tıklayın

**Ne olacak:**
1. Buton "⏳ İşleniyor..." olacak
2. 2-3 saniye sonra toast mesajları:
   - "✅ Dana Kıyma seçildi ve fotoğraf çekildi!"
   - "📸 Toplanan fotoğraf sayısı: 1"
3. Kart kısa süre highlight olacak

### 6.3 Diğer PLU'ları Test Edin

Her PLU için 3-5 kez seçim yapın:
- PLU 101: 5 kez seç
- PLU 102: 5 kez seç
- PLU 103: 5 kez seç

**Her seçimde:** Fotoğraf sayısı artacak (1, 2, 3, 4, 5...)

---

## 📊 ADIM 7: Dashboard Kontrolü

### 7.1 Dashboard'a Gidin

Üst menüden **"📊 Dashboard"** → Tıklayın

**Göreceğiniz istatistikler:**

```
┌─────────────────┬─────────────────┬─────────────┬─────────────┐
│ TOPLAM FOTOĞRAF │ AI KONTROL SAYISI│   UYUMLU   │  UYUMSUZ   │
│       15        │        0         │     0      │     0      │
└─────────────────┴─────────────────┴─────────────┴─────────────┘
```

**Mod Badge:** "🟡 Eğitim Modu"

### 7.2 Fotoğraflar Sekmesi

**"Fotoğraflar (15)"** sekmesine tıklayın

**Göreceğiniz:**
- 15 fotoğraf kartı
- Her birinde: PLU kodu, tarih-saat, "📚 Eğitim" badge'i

### 7.3 PLU İstatistikleri Sekmesi

**"PLU İstatistikleri"** sekmesine tıklayın

**Göreceğiniz:**
```
PLU 101: 5 Fotoğraf
PLU 102: 5 Fotoğraf
PLU 103: 5 Fotoğraf
```

---

## 📸 ADIM 8: Hazır Fotoğrafları Yükleme

### 8.1 Fotoğrafları Hazırlayın

Elinizde 200 fotoğraf varsa, şu şekilde organize edin:

```
C:\terazi-photos\
  ├── 101\
  │   ├── foto001.jpg
  │   ├── foto002.jpg
  │   ├── ...
  │   └── foto200.jpg  (toplam 200)
  ├── 102\
  │   └── ... (200 foto)
  └── 103\
      └── ... (200 foto)
```

**Önemli:**
- Klasör adı = PLU kodu (101, 102, 103)
- Desteklenen formatlar: .jpg, .jpeg, .png, .webp
- Dosya adı önemli değil

### 8.2 Import Scriptini Çalıştırın

Yeni bir **Command Prompt** açın:

```cmd
cd C:\Users\%USERNAME%\Desktop\terazi-ai\backend
python import_photos.py
```

**Script soracak:**

**Soru 1:** 
```
Fotoğrafların bulunduğu klasörü girin
Varsayılan: C:\terazi-photos

Klasör yapısı:
  klasor/
    ├── 101/  (PLU kodu)
    │   ├── img001.jpg
    │   └── img002.jpg
    ├── 102/
    └── ...

Klasör yolu (Enter = varsayılan): _
```

**Cevap:** `C:\terazi-photos` (veya Enter)

**Soru 2:**
```
Fotoğraflar hangi faza aktarılsın?
  1. training (Eğitim - varsayılan)
  2. production (Üretim)
Seçim (1/2): _
```

**Cevap:** `1` (veya Enter)

**Soru 3:**
```
📂 Klasör: C:\terazi-photos
🎯 Faz: training

Devam etmek istiyor musunuz? (E/H): _
```

**Cevap:** `E`

### 8.3 İçe Aktarma İzleme

**Script çalışırken göreceğiniz:**

```
🚀 İçe aktarma başlıyor...

📂 Toplam 3 PLU klasörü bulundu

📸 PLU 101: 200 fotoğraf bulundu, yükleniyor...
  ✅ 10/200 yüklendi...
  ✅ 20/200 yüklendi...
  ✅ 30/200 yüklendi...
  ...
  ✅ 200/200 yüklendi...
  ✅ PLU 101 tamamlandı: 200 fotoğraf

📸 PLU 102: 200 fotoğraf bulundu, yükleniyor...
  ...

📸 PLU 103: 200 fotoğraf bulundu, yükleniyor...
  ...

====================================================
📊 İÇE AKTARMA İSTATİSTİKLERİ
====================================================
Toplam Dosya:     600
✅ Başarılı:      600
❌ Başarısız:     0
⚠️ Atlanan:       0
====================================================

📦 VERİTABANI DURUMU
Toplam Fotoğraf:  615  (15 + 600)
Eğitim Fazı:      615
====================================================
```

**Süre:** ~2-5 dakika (600 fotoğraf için)

### 8.4 Dashboard'dan Kontrol Edin

Web'de Dashboard → Fotoğraflar sekmesi

**Görmelisiniz:**
- Toplam Fotoğraf: 615
- PLU 101: 205 fotoğraf
- PLU 102: 205 fotoğraf
- PLU 103: 205 fotoğraf

---

## 🚀 ADIM 9: Üretim Moduna Geçiş

### 9.1 Dashboard'dan Mod Değiştir

Dashboard sayfasında, sağ üstte **"🚀 Üretim Moduna Geç"** butonuna tıklayın

**Onay:**
- Toast mesajı: "✅ Sistem modu Üretim moduna geçti"
- Badge değişecek: "🟢 Üretim Modu"

### 9.2 Mod Değişikliğini Doğrulayın

**Terazi Ekranı** sayfasına gidin

**Sağ üstte görmelisiniz:** "🟢 Üretim Modu"

**Alt kısımdaki bilgi kartı kaybolacak** (artık eğitim değil)

---

## 🤖 ADIM 10: AI Analizini Test Etme

### 10.1 İlk AI Testi

**Terazi Ekranı** → **PLU 101 (Dana Kıyma)** → **"✓ Seç"** butonuna tıklayın

**Ne olacak:**
1. Buton "⏳ İşleniyor..." olacak
2. Kamera fotoğraf çekecek
3. Toast mesajları:
   - "✅ Dana Kıyma seçildi ve fotoğraf çekildi!"
   - "🤖 AI analizi başlatıldı..."
4. **Arka planda AI çalışıyor** (2-3 saniye)

### 10.2 Backend Loglarını İzleyin

Backend terminalinde şunu görmelisiniz:

```
INFO: POST /api/plu/select
INFO: Captured image for PLU 101
INFO: Starting AI analysis...
INFO: AI analysis completed: Match=True, Confidence=85%
```

### 10.3 AI Sonuçlarını Kontrol Edin

**Dashboard** → **"AI Kontrolleri (1)"** sekmesine tıklayın

**İlk AI kontrol kartını göreceksiniz:**

```
┌─────────────────────────────────────────┐
│ Dana Kıyma                    ✅ Uyumlu │
│ 01.12.2025 14:30:25                    │
│                                         │
│ Güven: 85%                              │
│ [████████████████████░░░░░░] 85%       │
└─────────────────────────────────────────┘
```

**Karta tıklarsanız** detaylı analiz göreceksiniz (gelecek feature için).

### 10.4 Farklı PLU'ları Test Edin

Sırayla test edin:
- PLU 101 → 3 kez seç
- PLU 102 → 3 kez seç
- PLU 103 → 3 kez seç

**Dashboard → AI Kontrolleri** → 9 sonuç görmelisiniz

### 10.5 Yanlış PLU Testi (Uyumsuzluk)

**Senaryo:** Kasap yanlış PLU seçerse?

**Test:**
1. Gerçek bir **kuzu pirzola** fotoğrafınız varsa kullanın
2. Veya hayal edin: terazide kuzu var ama...
3. **PLU 101 (Dana Kıyma)** seçin → AI "❌ Uyumsuz" demeli

**Şu an mock görüntü kullanıldığı için:** AI rastgele sonuç verebilir. Gerçek kamera ile test gerekli.

---

## 📊 ADIM 11: Sonuçları Değerlendirme

### 11.1 Dashboard İstatistikleri

Dashboard'da şunları kontrol edin:

```
┌─────────────────┬─────────────────┬─────────────┬─────────────┐
│ TOPLAM FOTOĞRAF │ AI KONTROL SAYISI│   UYUMLU   │  UYUMSUZ   │
│      624        │        9         │     7      │     2      │
└─────────────────┴─────────────────┴─────────────┴─────────────┘
```

**Doğruluk Oranı Card'ı:**
```
┌─────────────────────────┐
│   Doğruluk Oranı        │
│                         │
│      ⭕ 78%            │
│                         │
│ Toplam 9 kontrol yapıldı│
│ ✅ 7 uyumlu tespit      │
│ ❌ 2 uyumsuz tespit     │
└─────────────────────────┘
```

### 11.2 Doğruluk Değerlendirmesi

**%80+ doğruluk:** ✅ Sistem iyi çalışıyor  
**%60-80 doğruluk:** ⚠️ Kabul edilebilir, izlenebilir  
**%60 altı:** ❌ Sorun var, inceleme gerekli

**Not:** Şu an mock görüntü kullanıldığı için gerçek doğruluk ölçülemez.

### 11.3 Gerçek Kamera ile Final Test

**Gerçek terazi kamerası bağlandığında:**

1. Sistemi yeniden başlatın
2. Gerçek ürünleri tartın
3. PLU seçin
4. AI sonuçlarını izleyin
5. 20-30 işlem yapın
6. Dashboard'dan doğruluk oranına bakın

**Beklenen:** %75-90 doğruluk

---

## ✅ ADIM 12: Test Sonuç Raporu

### 12.1 Başarı Kriterleri

Testi başarılı saymak için:

- [x] Backend başladı ve çalışıyor
- [x] Frontend açıldı
- [x] PLU'lar oluşturuldu (en az 3)
- [x] Kamera testi yapıldı (mock da olsa)
- [x] Eğitim modunda fotoğraf çekildi
- [x] Hazır fotoğraflar yüklendi (varsa)
- [x] Üretim moduna geçildi
- [x] AI analizi çalıştı
- [x] Dashboard'da sonuçlar görüldü
- [x] İstatistikler doğru gösteriliyor

**Hepsi ✅ ise:** Sistem çalışıyor! 🎉

### 12.2 Test Raporu Şablonu

Aşağıdaki bilgileri kaydedin:

```
====================================================
TERAZI AI SİSTEMİ TEST RAPORU
====================================================
Tarih: ______________
Tester: ______________

SISTEM DURUMU:
✅/❌ Backend çalışıyor
✅/❌ Frontend çalışıyor
✅/❌ MongoDB çalışıyor
✅/❌ Kamera erişimi (gerçek/mock)

PLU'LAR:
- PLU 101: Dana Kıyma
- PLU 102: Kuzu Pirzola
- PLU 103: Tavuk Göğüs

FOTOĞRAF İSTATİSTİKLERİ:
- Toplam fotoğraf: ____
- Eğitim fazı: ____
- Üretim fazı: ____

AI PERFORMANSI:
- Toplam kontrol: ____
- Uyumlu: ____
- Uyumsuz: ____
- Doğruluk oranı: ____%

NOTLAR:
__________________________________________________
__________________________________________________
__________________________________________________

KARAR:
✅ Sistemi kullanıma hazır
⚠️ İyileştirme gerekiyor
❌ Sorunlar var, tekrar test
====================================================
```

---

## 🔧 ADIM 13: Sorun Giderme

### Sorun 1: Backend Başlamıyor

**Hata:**
```
Error: Cannot connect to MongoDB
```

**Çözüm:**
```cmd
net start MongoDB
```

### Sorun 2: Frontend Açılmıyor

**Hata:**
```
Error: Cannot find module 'react'
```

**Çözüm:**
```cmd
cd frontend
rmdir /s /q node_modules
yarn install
```

### Sorun 3: Kamera Erişim Hatası

**Hata:**
```
Cannot open camera
```

**Çözüm:**
- Başka program kamerayı kullanıyor olabilir (kapatın)
- Kamera izinlerini Windows Ayarlar → Gizlilik → Kamera'dan kontrol edin
- Sistem otomatik mock görüntüye geçecek (devam edin)

### Sorun 4: AI Yanıt Vermiyor

**Hata:**
```
AI analysis timeout
```

**Çözüm:**
- İnternet bağlantısını kontrol edin
- Backend loglarını inceleyin

### Sorun 5: Port Meşgul

**Hata:**
```
Address already in use: 8001
```

**Çözüm:**
```cmd
# Port'u temizle
netstat -ano | findstr :8001
taskkill /PID [PID] /F
```

---

## 📞 Test Sonrası

### Başarılıysa:

1. ✅ Sistemi terazi üzerinde bırakın
2. ✅ `start_terazi.bat` ile her gün başlatın
3. ✅ Günlük kullanıma başlayın
4. ✅ İlk hafta her gün Dashboard'u kontrol edin
5. ✅ Doğruluk oranını izleyin

### İyileştirme Gerekiyorsa:

1. ⚠️ Doğruluk oranı düşükse → Bana bildirin (RAG/Few-Shot ekleyelim)
2. ⚠️ Yavaş çalışıyorsa → İnternet hızını kontrol edin
3. ⚠️ PLU açıklamaları yetersizse → Daha detaylı yazın

### Kritik Sorun Varsa:

1. ❌ Test sonuç raporunu paylaşın
2. ❌ Backend loglarını gönderin
3. ❌ Hata mesajlarının screenshot'ını alın

---

## 🎉 Test Tamamlandı!

Tebrikler! Terazi AI sistemini başarıyla test ettiniz.

**Sonraki adımlar:**
- Gerçek kamera ile test
- Günlük kullanıma başlama
- İstatistikleri izleme
- Gerekirse optimizasyon

**Sorularınız için:** Test sonuç raporunu benimle paylaşın.
