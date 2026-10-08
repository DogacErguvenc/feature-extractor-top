# Terazi Üzerinde Kurulum ve Çalıştırma Rehberi

## 📋 Sistem Gereksinimleri

**Terazi Donanımı:**
- Intel Atom x6413E İşlemci
- 8GB RAM
- Windows İşletim Sistemi
- Entegre kamera

**Yazılım Gereksinimleri:**
- Python 3.9+ (önerilen: Python 3.11)
- Node.js 18+ ve Yarn
- MongoDB Community Server
- Git

---

## 🔧 Adım 1: Yazılım Kurulumları

### Python Kurulumu
1. https://www.python.org/downloads/ adresinden Python 3.11 indirin
2. Kurulum sırasında "Add Python to PATH" seçeneğini işaretleyin
3. Komut satırında test edin:
```cmd
python --version
pip --version
```

### Node.js ve Yarn Kurulumu
1. https://nodejs.org/ adresinden Node.js LTS indirin (v18+)
2. Kurulumu tamamlayın
3. Yarn'ı global olarak kurun:
```cmd
npm install -g yarn
```
4. Test edin:
```cmd
node --version
yarn --version
```

### MongoDB Kurulumu
1. https://www.mongodb.com/try/download/community adresinden MongoDB Community Server indirin
2. Kurulum sırasında "Install MongoDB as a Service" seçeneğini işaretleyin
3. Kurulum tamamlandıktan sonra MongoDB otomatik başlayacak
4. Test edin:
```cmd
mongosh
> show dbs
> exit
```

### Git Kurulumu (opsiyonel)
1. https://git-scm.com/download/win adresinden Git indirin
2. Kurulumu tamamlayın

---

## 📦 Adım 2: Proje Kurulumu

### GitHub'dan Projeyi İndirin
```cmd
cd C:\Users\YourUser\Desktop
git clone [YOUR_GITHUB_REPO_URL] terazi-ai
cd terazi-ai
```

### Backend Kurulumu
```cmd
cd backend
pip install -r requirements.txt
```

**Not:** OpenCV kurulumunda sorun yaşarsanız:
```cmd
pip install opencv-python-headless
```

### Frontend Kurulumu
```cmd
cd ..\frontend
yarn install
```

---

## ⚙️ Adım 3: Yapılandırma

### Backend .env Dosyası
`backend\.env` dosyasını düzenleyin:
```env
MONGO_URL=mongodb://localhost:27017
DB_NAME=terazi_production
API_KEY=YOUR_API_KEY
ALLOWED_IMAGE_DIR=backend\incoming
DISABLE_DOCS=true
CORS_ORIGINS=http://localhost:3000
# Gerçek anahtarınızı yalnız yerel .env dosyanızda saklayın; Git'e eklemeyin.
EMERGENT_LLM_KEY=YOUR_EMERGENT_LLM_KEY
```

### Frontend .env Dosyası
`frontend\.env` dosyasını düzenleyin:
```env
REACT_APP_BACKEND_URL=http://localhost:8001
REACT_APP_API_KEY=YOUR_API_KEY
WDS_SOCKET_PORT=3000
REACT_APP_ENABLE_VISUAL_EDITS=false
ENABLE_HEALTH_CHECK=false
```

---

## 🚀 Adım 4: Sistemi Başlatma

### Terminal 1: Backend'i Başlatın
```cmd
cd backend
python -m uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

Çıktı:
```
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Terminal 2: Frontend'i Başlatın
```cmd
cd frontend
yarn start
```

Tarayıcı otomatik olarak `http://localhost:3000` adresinde açılacak.

---

## 🎥 Adım 5: Kamera Testi

### Kamerayi Test Edin

Not: Tum /api isteklerinde x-api-key header zorunludur.

Yeni bir terminal açın:
```cmd
curl -X POST http://localhost:8001/api/camera/test
```

**Başarılı yanıt:**
```json
{
  "message": "Camera test successful",
  "image_preview": "/9j/4AAQSkZJRg..."
}
```

**Hata alırsanız:**
- Kamera bağlantısını kontrol edin
- Başka program kamerayı kullanıyor olabilir (Zoom, Teams vb. kapatın)
- Sistem otomatik mock görüntüye geçecektir

---

## 📸 Toplu Fotoğraf İçe Aktarma

Hazır fotoğraflarınızı sisteme yüklemek için import scriptini kullanın.

### Fotoğrafları Hazırlayın
Fotoğrafları klasörlere ayırın (PLU kodlarına göre):
```
C:\terazi-photos\
  ├── 101\          # Dana Kıyma fotoğrafları
  │   ├── img001.jpg
  │   ├── img002.jpg
  │   └── ...
  ├── 102\          # Kuzu Pirzola fotoğrafları
  │   ├── img001.jpg
  │   └── ...
  └── ...
```

### Import Scriptini Çalıştırın
```cmd
cd backend
python import_photos.py
```

Script size klasör yolunu soracak, örnek:
```
C:\terazi-photos
```

---

## 🧪 Adım 6: Test

### Web Arayüzünden Test
1. Tarayıcıda `http://localhost:3000` açın
2. **PLU Yönetimi** → Yeni PLU ekleyin
3. **Terazi Ekranı** → PLU seçin, fotoğraf çekilecek
4. **Dashboard** → İstatistikleri görün

### API ile Test
```cmd
# PLU listesi
curl http://localhost:8001/api/plu/list

# PLU seçimi (fotoğraf çekimi)
curl -X POST http://localhost:8001/api/plu/select ^
  -H "Content-Type: application/json" ^
  -d "{\"plu_code\":\"101\"}"

# İstatistikler
curl http://localhost:8001/api/stats/dashboard
```

---

## 🔄 Günlük Kullanım

### Sistemi Her Açılışta Başlatma

**Option 1: Manuel Başlatma**
İki terminal açın ve sırasıyla:
```cmd
# Terminal 1
cd C:\path\to\terazi-ai\backend
python -m uvicorn server:app --host 0.0.0.0 --port 8001

# Terminal 2
cd C:\path\to\terazi-ai\frontend
yarn start
```

**Option 2: Batch Script (Otomatik)**
`start_terazi.bat` dosyası oluşturun:
```batch
@echo off
cd /d C:\path\to\terazi-ai\backend
start cmd /k "python -m uvicorn server:app --host 0.0.0.0 --port 8001"
timeout /t 5
cd /d C:\path\to\terazi-ai\frontend
start cmd /k "yarn start"
```

Bu dosyaya çift tıklayarak sistemi başlatın.

**Option 3: Windows Başlangıçta Otomatik (İleri Düzey)**
Windows Görev Zamanlayıcı (Task Scheduler) ile batch scriptini başlangıçta çalıştırın.

---

## 🛠️ Sorun Giderme

### MongoDB Çalışmıyor
```cmd
# MongoDB servisini kontrol et
net start MongoDB

# Veya manuel başlat
"C:\Program Files\MongoDB\Server\7.0\bin\mongod.exe" --dbpath C:\data\db
```

### Port 8001 veya 3000 Kullanımda
```cmd
# Port kullanımını kontrol et
netstat -ano | findstr :8001
netstat -ano | findstr :3000

# Process'i kapat (PID'yi yukarıdaki komuttan alın)
taskkill /PID [PID] /F
```

### Kamera Erişim Hatası
- Kamera izinlerini Windows Ayarlar → Gizlilik → Kamera'dan kontrol edin
- Antivirüs programı kamera erişimini engelliyor olabilir
- Sistem otomatik mock moda geçecektir

### OpenCV Kurulum Hatası
```cmd
# Alternatif kurulum
pip uninstall opencv-python
pip install opencv-python-headless
```

### Frontend Build Hatası
```cmd
# Node modules'ı temizle ve yeniden kur
cd frontend
rmdir /s /q node_modules
yarn install
```

---

## 📊 Veri Yedekleme

### MongoDB Yedekleme
```cmd
# Yedek al
mongodump --db terazi_production --out C:\terazi-backup

# Yedekten geri yükle
mongorestore --db terazi_production C:\terazi-backup\terazi_production
```

### Fotoğraf Arşivi
Fotoğraflar MongoDB'de base64 olarak saklanır, yedekleme önerilir:
```cmd
# Export script ile tüm fotoğrafları dışa aktar
python backend/export_photos.py
```

---

## 🌐 Uzaktan Erişim (İsteğe Bağlı)

Ofis PC'den teraziye erişmek için:

1. Terazi IP adresini öğrenin:
```cmd
ipconfig
```

2. Windows Firewall'da port 8001 ve 3000'i açın

3. Ofis PC'den tarayıcıda:
```
http://[TERAZI_IP]:3000
```

---

## 📞 Destek

**Logları Kontrol Etme:**
```cmd
# Backend console'dan logları görebilirsiniz
# Veya log dosyasına yönlendirin
python -m uvicorn server:app --log-level debug > backend.log 2>&1
```

**Veritabanı İnceleme:**
```cmd
mongosh
> use terazi_production
> db.plu_products.find()
> db.captured_images.countDocuments()
> db.validation_results.find().limit(5)
```
