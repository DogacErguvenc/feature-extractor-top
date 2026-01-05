# Terazi AI Sistemi - Kullanım Rehberi

## 📋 Proje Özeti

Kasaplar için yapay zeka destekli terazi sistemi. PLU seçimi yapıldığında otomatik fotoğraf çeker ve AI ile ürün-PLU uyumluluğunu kontrol eder.

## 🎯 Sistem Modları

### 1. Eğitim Modu (Training Mode)
- **Süre:** İlk 30 gün
- **Amaç:** Veri toplama
- **İşleyiş:** 
  - Her PLU seçiminde fotoğraf çekilir ve kaydedilir
  - AI analizi yapılmaz
  - Hedef: PLU başına ~200 fotoğraf
  
### 2. Üretim Modu (Production Mode)
- **Başlama:** 30 gün sonra (yeterli veri toplandıktan sonra)
- **İşleyiş:**
  - Her PLU seçiminde fotoğraf çekilir
  - OpenAI GPT-5 Vision ile otomatik analiz yapılır
  - Uyumlu/Uyumsuz raporları üretilir

## 🖥️ Sistem Bileşenleri

### 1. Terazi Operatör Ekranı (Ana Ekran)
- **Kullanıcı:** Kasap çalışanları
- **Fonksiyonlar:**
  - PLU listesini görüntüleme
  - PLU seçimi yapma
  - Otomatik fotoğraf çekimi
  - Sistem modu göstergesi

**Kullanım:**
1. Ürünü terazinin kefesine koyun
2. Uygun PLU'yu ekrandan seçin
3. Sistem otomatik fotoğraf çeker
4. Eğitim modunda: Fotoğraf sayısı gösterilir
5. Üretim modunda: AI analizi başlatılır

### 2. Dashboard (Yönetim Paneli)
- **Kullanıcı:** Yöneticiler
- **Fonksiyonlar:**
  - İstatistikler görüntüleme
  - Mod değiştirme (Eğitim ↔ Üretim)
  - Fotoğraf arşivi görüntüleme
  - AI kontrol sonuçları görüntüleme
  - PLU bazlı istatistikler

**Özellikler:**
- Toplam fotoğraf sayısı
- AI kontrol sayısı
- Uyumlu/Uyumsuz tespit sayıları
- Doğruluk oranı (%)
- PLU bazında fotoğraf dağılımı

### 3. PLU Yönetimi
- **Kullanıcı:** Yöneticiler
- **Fonksiyonlar:**
  - Yeni PLU ekleme
  - PLU listesini görüntüleme
  - PLU silme

**PLU Bilgileri:**
- PLU Kodu (örn: 101, 102)
- Ürün Adı (örn: Dana Kıyma)
- Açıklama (ürün özellikleri: renk, şekil, doku)

## 🔧 Teknik Detaylar

### Kamera Sistemi
- **Gerçek Kamera:** Sistem otomatik algılar (cv2.VideoCapture)
- **Demo Modu:** Kamera yoksa otomatik mock görüntü oluşturur
- **Format:** JPEG, base64 encoding
- **Çözünürlük:** Maks 1920px (otomatik yeniden boyutlandırma)

### AI Analiz Sistemi
- **Model:** OpenAI GPT-5 Vision
- **Analiz Süresi:** ~2-3 saniye
- **Çıktılar:**
  - Görülen ürün açıklaması
  - Uyumluluk durumu (Evet/Hayır)
  - Güven skoru (0-100%)
  - Detaylı açıklama

### Veritabaları (MongoDB)
1. **plu_products** - PLU tanımları
2. **captured_images** - Çekilen fotoğraflar
3. **validation_results** - AI kontrol sonuçları

## 📊 Kullanım Senaryosu

### İlk Kurulum (Gün 1-30)
```
1. PLU'ları tanımlayın (PLU Yönetimi)
   - PLU 101: Dana Kıyma
   - PLU 102: Kuzu Pirzola
   - PLU 103: Tavuk Göğüs
   vb.

2. Sistem otomatik "Eğitim Modu"nda başlar

3. Normal kullanıma devam edin:
   - Ürünü tartın
   - PLU seçin
   - Sistem her seferinde fotoğraf çeker

4. Dashboard'dan ilerlemeyi takip edin
   - Her PLU için toplanan fotoğraf sayısı
   - Hedef: PLU başına 200+ fotoğraf
```

### Üretim Aşaması (Gün 30+)
```
1. Dashboard'dan "Üretim Moduna Geç" butonuna tıklayın

2. Artık her işlemde AI analizi devreye girer:
   - Ürünü tartın
   - PLU seçin
   - Fotoğraf çekilir
   - AI otomatik analiz yapar
   - Sonuç kaydedilir

3. Dashboard'dan sonuçları inceleyin:
   - Uyumlu/Uyumsuz tespit sayıları
   - Doğruluk oranı
   - Detaylı analiz raporları
```

## 🚀 API Endpoints (Gelişmiş Kullanım)

```bash
# PLU İşlemleri
GET  /api/plu/list                    # PLU listesi
POST /api/plu/create                  # Yeni PLU ekle
DELETE /api/plu/delete/{plu_code}     # PLU sil

# Terazi İşlemleri
POST /api/plu/select                  # PLU seç ve fotoğraf çek
POST /api/camera/test                 # Kamera test

# İstatistikler
GET /api/stats/dashboard              # Dashboard istatistikleri
GET /api/images/captured              # Çekilen fotoğraflar
GET /api/validation/results           # AI kontrol sonuçları

# Sistem Yönetimi
GET  /api/system/mode                 # Mevcut mod
POST /api/system/mode                 # Mod değiştir
```

## 🔑 Önemli Notlar

### Kamera Gereksinimleri
- Windows üzerinde USB kamera (terazi entegre kamerası)
- OpenCV uyumlu herhangi bir kamera
- Kamera yoksa sistem otomatik demo moduna geçer

### Ürün Açıklamalarında Dikkat Edilecekler
PLU tanımlarken mümkün olduğunca detaylı açıklama yazın:
```
✅ İYİ: "Taze dana kıyma, orta yağlı, parlak kırmızı renk, ince çekilmiş"
❌ KÖTÜ: "Dana kıyma"
```

### Performans
- Her fotoğraf: ~100-500KB
- AI analizi: ~2-3 saniye
- 1 aylık veri: ~2-3GB (30 PLU × 200 foto)

### Ticari Kullanım
- OpenAI GPT-5 Vision tam ticari lisanslı
- Kendi API anahtarınızla da kullanabilirsiniz

## 🛠️ Sorun Giderme

### "Kamera açılamadı" hatası
- Kamera bağlı mı kontrol edin
- Sistem otomatik demo moduna geçer
- Gerçek kamera için driver'ları kontrol edin

### AI analizi çok uzun sürüyor
- İnternet bağlantısını kontrol edin
- API key limitlerini kontrol edin
- Backend loglarını inceleyin

### Düşük doğruluk oranı
- Daha fazla eğitim fotoğrafı toplayın
- PLU açıklamalarını detaylandırın
- Işıklandırmayı iyileştirin
- Kamera kalitesini artırın

## 📞 Destek



**Loglar:**
```bash
# Backend logs
tail -f /var/log/supervisor/backend.*.log

# Frontend logs
tail -f /var/log/supervisor/frontend.*.log
```
