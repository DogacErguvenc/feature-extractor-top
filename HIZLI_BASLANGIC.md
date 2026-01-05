# ⚡ Hızlı Başlangıç - Terazi Kurulum

## 📋 Öncesi Kontrol Listesi

Terazide şunların kurulu olduğundan emin olun:
- ✅ Python 3.9+ (`python --version`)
- ✅ Node.js 18+ (`node --version`)
- ✅ Yarn (`yarn --version`)
- ✅ MongoDB (`mongosh`)

**Kurulu değilse:** `TERAZI_KURULUM.md` dosyasını okuyun.

---

## 🚀 3 Adımda Kurulum

### 1️⃣ Projeyi İndirin
```cmd
cd C:\Users\YourUser\Desktop
git clone [REPO_URL] terazi-ai
cd terazi-ai
```

### 2️⃣ Bağımlılıkları Kurun
```cmd
REM Backend
cd backend
pip install -r requirements.txt

REM Frontend
cd ..\frontend
yarn install
```

### 3️⃣ Başlatın
```cmd
REM Terminal 1: Backend
cd backend
python -m uvicorn server:app --host 0.0.0.0 --port 8001 --reload

REM Terminal 2 (yeni terminal): Frontend  
cd frontend
yarn start
```

**Tarayıcı otomatik açılacak:** `http://localhost:3000`

---

## 📸 Hazır Fotoğrafları Yükleyin

### Adım 1: Fotoğrafları Organize Edin
```
C:\terazi-photos\
  ├── 101\          # PLU kodu = klasör adı
  │   ├── foto1.jpg
  │   ├── foto2.jpg
  │   └── ...       # ~200 fotoğraf
  ├── 102\
  │   └── ...
  └── ...
```

### Adım 2: PLU'ları Oluşturun
Web arayüzünden (`http://localhost:3000`):
1. **PLU Yönetimi** → **Yeni PLU Ekle**
2. Her klasör için bir PLU oluşturun:
   - PLU Kodu: `101`
   - Ürün Adı: `Dana Kıyma`
   - Açıklama: `Taze dana kıyma, orta yağlı, parlak kırmızı renk, ince çekilmiş`

**ÖNEMLİ:** Açıklamalar detaylı olmalı (renk, doku, şekil, özellikler)

### Adım 3: Fotoğrafları İçe Aktarın
```cmd
cd backend
python import_photos.py
```

Script size soracak:
```
Klasör yolu: C:\terazi-photos
Faz (1=training, 2=production): 1
Devam etmek istiyor musunuz? (E/H): E
```

Import işlemi başlayacak:
```
📸 PLU 101: 200 fotoğraf bulundu, yükleniyor...
  ✅ 10/200 yüklendi...
  ✅ 20/200 yüklendi...
  ...
  ✅ PLU 101 tamamlandı: 200 fotoğraf

📊 İÇE AKTARMA İSTATİSTİKLERİ
Toplam Dosya:     200
✅ Başarılı:      200
❌ Başarısız:     0
```

---

## 🧪 Test Edin

### 1. Fotoğrafları Kontrol Edin
Web'den **Dashboard** → **Fotoğraflar** sekmesine gidin.
Yüklenen fotoğrafları görmelisiniz.

### 2. Üretim Moduna Geçin
**Dashboard** → **Üretim Moduna Geç** butonuna tıklayın.

### 3. Terazi Kamerasından Test Edin
**Terazi Ekranı** → Herhangi bir PLU seçin → Kamera otomatik fotoğraf çeker → AI analizi başlar

### 4. Sonuçları Görün
**Dashboard** → **AI Kontrolleri** sekmesine gidin.
AI analiz sonuçlarını görmelisiniz:
- ✅ Uyumlu / ❌ Uyumsuz
- Güven skoru (%)
- Detaylı analiz

---

## 🎯 Fotoğraf Depolama Hakkında

### Fotoğraflar Nerede?
Fotoğraflar **MongoDB veritabanında** base64 formatında saklanıyor:

```javascript
{
  "id": "uuid",
  "plu_code": "101",
  "image_base64": "/9j/4AAQSkZJRg...",  // ← Burası fotoğraf
  "timestamp": "2025-12-01T10:30:00Z",
  "phase": "training"
}
```

**Collection:** `captured_images`

### Avantajları
- ✅ Tüm veri tek yerde (MongoDB)
- ✅ Yedekleme kolay (`mongodump`)
- ✅ Dosya sistemi karışıklığı yok
- ✅ API üzerinden direkt erişim

### Dezavantajları
- ⚠️ Veritabanı boyutu büyüyebilir
- ⚠️ Manuel görüntüleme zor

### Fotoğrafları Dışa Aktar (Gerekirse)
```cmd
cd backend
python export_photos.py
```

Fotoğraflar dosya sistemine kaydedilecek:
```
C:\terazi-export\
  ├── 101\
  │   ├── training_2025-12-01_abc123.jpg
  │   └── ...
  └── ...
```

---

## 📱 Günlük Kullanım

### Sabah - Sistemi Başlat
```cmd
cd C:\path\to\terazi-ai
start_terazi.bat     # Batch script'i çalıştır
```

### Gün Boyu - Normal Kullanım
1. Web arayüzü açık kalacak: `http://localhost:3000`
2. Kasap teraziden PLU seçiyor
3. Sistem otomatik fotoğraf çekiyor
4. Üretim modunda AI analizi yapıyor

### Akşam - Kontrol
**Dashboard** → İstatistiklere bakın:
- Bugün kaç işlem yapıldı?
- Uyumluluk oranı nedir?
- Uyumsuzluklar var mı?

---

## 🔧 Hızlı Sorun Giderme

### Backend Başlamıyor
```cmd
# Port meşgul mü?
netstat -ano | findstr :8001
taskkill /PID [PID] /F

# MongoDB çalışıyor mu?
net start MongoDB
```

### Kamera Çalışmıyor
- ✅ Başka program kamerayı kullanıyor olabilir (kapatın)
- ✅ Sistem otomatik mock moda geçer (test için yeterli)
- ✅ Gerçek kamera test: `curl -H "x-api-key: YOUR_API_KEY" -X POST http://localhost:8001/api/camera/test`

### Frontend Açılmıyor
```cmd
cd frontend
rmdir /s /q node_modules
yarn install
yarn start
```

### Veritabanı Bağlantısı Yok
```cmd
# MongoDB başlat
net start MongoDB

# Test et
mongosh
> show dbs
> use terazi_production
> db.captured_images.countDocuments()
```

---

## 📞 Yardım

**Detaylı Dokümantasyon:**
- `TERAZI_KURULUM.md` - Tam kurulum rehberi
- `KULLANIM_REHBERI.md` - Kullanım kılavuzu

**Loglar:**
```cmd
# Backend konsol çıktısı = log
# Veya dosyaya kaydet:
python -m uvicorn server:app > backend.log 2>&1
```

**MongoDB İnceleme:**
```cmd
mongosh
> use terazi_production
> db.plu_products.find().pretty()
> db.captured_images.countDocuments()
> db.validation_results.find().limit(5).pretty()
```

---

## ✅ Başarı Kontrol Listesi

Test tamamlandığında:
- ✅ Backend çalışıyor (`http://localhost:8001/api/`)
- ✅ Frontend açılıyor (`http://localhost:3000`)
- ✅ PLU'lar oluşturuldu
- ✅ Hazır fotoğraflar yüklendi (200+ per PLU)
- ✅ Üretim moduna geçildi
- ✅ Terazi kamerasından test yapıldı
- ✅ AI analizi çalıştı
- ✅ Dashboard'da sonuçlar görülüyor

**Hepsi tamam mı? 🎉 Sistem kullanıma hazır!**
