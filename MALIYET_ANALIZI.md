# 💰 Terazi AI Sistemi - Maliyet Analizi

## ⚠️ ÖNEMLİ: BEDAVA DEĞİL!

Sistem çalıştırmak için **OpenAI API kullanımı** gerekiyor ve bu **ücretli**.

---

## 💳 Ücretli Servisler

### 1. OpenAI GPT-5 Vision API ⚠️ ÜCRETLI

**Kullanım:** Her PLU seçiminde (üretim modunda) bir API çağrısı yapılır.

**Fiyatlandırma (Ocak 2025 itibariyle):**
- **Giriş (Input):** $2.50 per 1M tokens
- **Çıkış (Output):** $10.00 per 1M tokens
- **Görüntü:** ~1000 token (640x480 görüntü için)
- **Metin:** ~200 token (PLU açıklaması + yanıt)

**Tek bir analiz maliyeti:**
```
Input:  1200 tokens × $2.50/1M  = $0.003
Output:  200 tokens × $10.00/1M = $0.002
─────────────────────────────────────────
TOPLAM:                          $0.005
                          (yaklaşık 0.15 TL)
```

**Günlük kullanım senaryoları:**

| Senaryo | Günlük İşlem | Aylık Maliyet | Yıllık Maliyet |
|---------|--------------|---------------|----------------|
| Küçük kasap | 50 işlem/gün | ~$7.50 | ~$90 |
| Orta kasap | 200 işlem/gün | ~$30 | ~$360 |
| Büyük kasap | 500 işlem/gün | ~$75 | ~$900 |



**Maliyet:**
- 💰 **Kullanım başına kredi tüketimi var**
- 💳 Bakiye biterse: Manuel ekleme veya auto-top up gerekli

**Kredi Sistemi:**
- Bakiye yönetimi: `Profile → Universal Key → Add Balance`
- Auto-top up aktif edilebilir

**Alternatif:**
- Kendi OpenAI API anahtarınızı kullanabilirsiniz
- Doğrudan OpenAI'dan ücretlendirilirsiniz

---

## ✅ Ücretsiz Yazılımlar

### 1. MongoDB Community Server ✅ BEDAVA
- Lokal kullanım tamamen ücretsiz
- Ticari kullanım da ücretsiz
- Lisans: Server Side Public License (SSPL)

### 2. Python & Node.js ✅ BEDAVA
- Açık kaynak
- Ticari kullanım serbest

### 3. React, FastAPI, vb. ✅ BEDAVA
- Tüm kütüphaneler açık kaynak
- MIT/Apache lisansları

### 4. Windows İşletim Sistemi 💰 ZATİN VAR
- Terazide zaten Windows yüklü
- Ek maliyet yok

---

## 📊 Toplam Maliyet Tablosu

### İlk Kurulum Maliyeti: **$0**
- Tüm yazılımlar ücretsiz
- Sadece zaman

### Aylık İşletme Maliyeti:

| Bileşen | Maliyet |
|---------|---------|
| OpenAI API (orta kullanım) | ~$30/ay |
| MongoDB (lokal) | $0 |
| Elektrik (terazi 7/24) | ~$5-10/ay |
| İnternet | Zaten var |
| **TOPLAM** | **~$35-40/ay** |

### Yıllık Maliyet: **~$420-480**

---

## 💡 Maliyet Optimizasyonu

### 1. Sadece Gerektiğinde Üretim Modu ✅

**Strateji:**
- İlk 30 gün: **Eğitim Modu** (sadece fotoğraf, AI yok) → $0
- Sonrasında: **Seçici Üretim Modu** (sadece şüpheli durumlarda AI)

**Örnek:**
- Kasap deneyimli, güveniyorsa → AI atla → $0
- Kasap yeni veya şüpheli → AI kontrol et → $0.005

**Tasarruf:** %50-70 maliyet azaltma

### 2. Batch İşleme (Gelecek Feature)

**Strateji:**
- Günlük 200 işlem → Gün sonunda toplu analiz
- 200 görüntü → Tek API çağrısı (daha verimli)

**Tasarruf:** %30-40 maliyet azaltma

### 3. Kendi OpenAI Anahtarı 💳

**Avantaj:**
- Direkt OpenAI fiyatlandırması
- Daha şeffaf maliyet takibi

**Dezavantaj:**
- Ayrı hesap yönetimi
- Kredi kartı gerekli

### 4. Cache/Önbellek Sistemi (Gelecek Feature)

**Strateji:**
- Benzer görüntüler için AI'yı tekrar çağırma
- Veritabanında önbellekleme

**Tasarruf:** %20-30 maliyet azaltma

---

## 🤔 Alternatif Yaklaşımlar

### Seçenek A: Hibrit Model (Önerim)

**İlk 30 gün:**
- Sadece fotoğraf toplama → $0

**31+ gün:**
- %20 rastgele kontrol (audit) → Aylık $6
- Şüpheli durumlar (manuel tetikleme) → Aylık $3
- **Toplam:** ~$9/ay

**Avantaj:** Çok düşük maliyet, yine de kontrol var

### Seçenek B: Tam AI Kontrol

**Her işlem:**
- AI kontrolü → Aylık $30

**Avantaj:** Tam otomatik, maksimum güvenlik

### Seçenek C: Sadece Eğitim Modu

**Hiç AI kullanma:**
- Sadece fotoğraf arşivi → $0
- Manuel kontrol için fotoğrafları incele

**Avantaj:** Tamamen bedava

---


### Bakiyenizi Kontrol Edin

2. **Profile → Universal Key** bölümüne gidin
3. Mevcut bakiyeyi görün
4. Kullanım geçmişini inceleyin

### Bakiye Ekleme

**Manuel Ekleme:**
1. Profile → Universal Key → **Add Balance**
2. Kredi kartı ile istediğiniz miktarı ekleyin

**Auto Top-Up:**
1. Bakiye belirli bir seviyenin altına düşünce otomatik yükleme
2. Ayarlar → Auto Top-Up → Aktif et
3. Minimum bakiye ve yükleme miktarı belirle

### Uyarılar

⚠️ **Bakiye Bitti:**
- AI analizi çalışmaz
- Backend log'da hata görürsünüz: "API quota exceeded"
- Fotoğraf çekimi çalışır, sadece AI durur

⚠️ **Günlük/Aylık Limit:**
- OpenAI bazı hesaplarda günlük/aylık limit koyabilir
- Limit aşımında geçici olarak servis durur

---

## 🔐 Kendi OpenAI Anahtarınızı Kullanma

### Neden Kendi Anahtarınızı Kullanasınız?

- ✅ Daha şeffaf maliyet takibi
- ✅ OpenAI'dan direkt faturalama
- ✅ Kredi limitleri kendiniz kontrol

### Nasıl Yapılır?

**1. OpenAI Hesabı Oluşturun**
- https://platform.openai.com/signup
- Kredi kartı bilgisi ekleyin
- Ödeme yöntemi ayarlayın

**2. API Anahtarı Oluşturun**
- Dashboard → API Keys → Create new secret key
- Anahtarı kopyalayın: `sk-proj-xxxxxxxxxxxxx`

**3. Kodu Güncelleyin**

`backend/server.py` dosyasında değişiklik yapın:

```python
# ÖNCEKİ:

chat = LlmChat(
    ...
)

# YENİ:
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# API çağrısını değiştirin (standart OpenAI SDK kullanın)
```

**4. .env Dosyasını Güncelleyin**

`backend/.env`:
```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx  # Ekle
```

**Not:** Bu değişiklik için kod modifikasyonu gerekir. İsterseniz yapabilirim.

---

## 📊 Maliyet Karşılaştırması

### Manuel Kontrol vs AI Kontrol

**Senaryo:** 200 işlem/gün

| Yöntem | Aylık Maliyet | Avantaj | Dezavantaj |
|--------|---------------|---------|------------|
| Manuel (çalışan kontrol) | ~$300-500 (maaş) | İnsan faktörü | Hata payı yüksek |
| AI (tam otomatik) | ~$30 | Tutarlı, hızlı | API maliyeti |
| Hibrit (AI örnekleme) | ~$6-9 | Ekonomik | Tam koruma yok |

**Sonuç:** AI çok daha ekonomik!

---

## 🎯 Önerilerim

### Aşama 1: Test (Şimdi)
```
✅ İlk testlerde ücretsiz/düşük maliyetli
✅ 1-2 hafta test et, maliyeti ölç
```

### Aşama 2: Değerlendirme
```
📊 Günlük ortalama işlem sayısı?
📊 Aylık tahmini maliyet?
📊 ROI (Return on Investment) hesapla
```

### Aşama 3: Optimizasyon
```
🔧 Hibrit model uygula (rastgele örnekleme)
🔧 Cache sistemi ekle
🔧 Gerekirse kendi OpenAI anahtarı
```

---

## 💸 ROI (Yatırım Getirisi) Hesabı

### Maliyet:
- Yazılım geliştirme: $0 (benim tarafımdan)
- Aylık AI maliyeti: $30 (orta kullanım)
- **Yıllık:** ~$360

### Tasarruf/Kazanç:
- Hatalı etiketleme önleme: ~$500-1000/yıl
- Müşteri memnuniyeti: Paha biçilemez
- İtibar kaybı önleme: Paha biçilemez
- Denetim kolaylığı: Zaman tasarrufu

**ROI:** Pozitif! Maliyet çok düşük, fayda yüksek.

---

## ❓ SSS

C: AI analizi durur, fotoğraf çekimi devam eder. Bakiye ekleyin.

**S: OpenAI fiyatları değişir mi?**
C: Evet, OpenAI zaman zaman fiyat güncellemesi yapar. Genelde düşer.

**S: Offline çalışabilir mi?**
C: Hayır, AI için internet gerekli. Fotoğraf çekimi offline çalışır.

**S: Alternatif daha ucuz AI var mı?**
C: Evet (Claude, Gemini) ama GPT-5 Vision et tanımada en iyi. Değiştirebiliriz.

**S: Kendi modelimi eğitsem bedava olur mu?**
C: Evet ama eğitim maliyeti ($100-500) + GPU gereksinimi. Uzun vadede karlı olabilir.

---

## 📞 Sonuç

### Kısa Özet:
- ❌ **Tamamen bedava değil**
- 💰 **Aylık ~$30-40** (orta kullanım)
- ✅ **Yazılım bedava, sadece AI maliyeti**
- 📊 **ROI pozitif, faydalı**
- 🔧 **Maliyet optimizasyonu mümkün**

### Eylem Planı:
2. 📊 1 hafta kullan, maliyeti ölç
3. 💡 Optimizasyon kararı ver
4. 🚀 Uzun vadeli strateji belirle

**Sorularınız varsa benimle paylaşın!**
