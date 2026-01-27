Terazi AI
=========

Kasap terazisi için PLU seçimi sonrası otomatik fotoğraf çekip AI ile ürün doğrulaması yapan sistem.

Kurulum (Windows)
-----------------
```powershell
git clone <REPO_URL> terazi-ai
cd terazi-ai
.\setup_terazi.ps1   # backend venv, .env, frontend bağımlılıkları
```

Manuel alternatif:
- `cd backend && python -m venv .venv && .\.venv\Scripts\activate && pip install -r requirements.txt`
- `cd frontend && yarn install`

Çalıştırma
----------
- Backend: `cd backend && .\.venv\Scripts\activate && uvicorn server:app --host 0.0.0.0 --port 8001 --reload`
- Frontend: `cd frontend && yarn start` (http://localhost:3000)

Model Seçimi
------------
Dashboard → “Model Seçimi” kartı:
- Local (ONNX, offline)
- Local (Embedding, offline similarity)
- Gemini (gemini-2.5-flash-lite / gemini-2.5-flash)
- OpenAI (model adı serbest)
Kaydedilen ayar MongoDB’de saklanır; sonraki çekimlerde aynı model kullanılır.

PLU ve Modlar
-------------
- PLU ekle/sil: Dashboard → PLU Yönetimi
- Eğitim modu: Fotoğraflar sadece kaydedilir.
- Üretim modu: Fotoğraf çekilir, seçili modelle analiz edilir; sonuçlar “AI Kontrolleri” ve “Fotoğraflar” sekmelerinde model bilgisiyle görünür.

Veri Scriptleri
---------------
- Toplu içe aktarma: `python backend/import_photos.py`
- Dışa aktarma: `python backend/export_photos.py`

Notlar
------
- `.env` örnek: `AI_PROVIDER=local`, `LOCAL_MODEL_PATH` ve `LOCAL_LABELS_PATH` model dosyalarına işaret eder. Gemini/OpenAI için ilgili API anahtarını ekleyin.
- Kamera yoksa PLU seçimi 500 döner (mock yok).


Security\r\n--------\r\n- API anahtari kullanilmiyor; /api istekleri aciktir.\r\n
Embedding Store
---------------
- Local (Embedding) icin once embedding store olusturun:
  `python backend/build_embedding_store.py --data-dir C:\terazi-datasets\train --out-dir backend\embedding_store`

