# ✅ Terazi AI Test Checklist

Hızlı test için bu listeyi takip edin. Detaylar için `TEST_REHBERI.md` dosyasına bakın.

---

## 📦 1. KURULUM (İlk Kez)

```cmd
[ ] Git clone / ZIP indir → C:\Users\[User]\Desktop\terazi-ai
[ ] test_terazi.bat → Tüm yazılımlar OK?
[ ] cd backend → pip install -r requirements.txt
[ ] cd frontend → yarn install
[ ] .env dosyalarını kontrol et
```

---

## 🚀 2. BAŞLATMA (Her Test)

```cmd
[ ] start_terazi.bat → Çift tıkla
[ ] Backend terminal açıldı (http://0.0.0.0:8001)
[ ] Frontend terminal açıldı
[ ] Tarayıcı açıldı (http://localhost:3000)
[ ] İki terminal penceresi açık kalsın!
```

---

## 🏷️ 3. PLU OLUŞTURMA

```cmd
Web: http://localhost:3000

[ ] PLU Yönetimi → Yeni PLU Ekle
[ ] PLU 101: Dana Kıyma (detaylı açıklama)
[ ] PLU 102: Kuzu Pirzola (detaylı açıklama)
[ ] PLU 103: Tavuk Göğüs (detaylı açıklama)
[ ] 3 PLU kartı görünüyor
```

**Detaylı Açıklama Örneği:**
```
Taze dana kıyma, orta yağlı, parlak kırmızı renk, 
ince çekilmiş, homojen doku, yağ oranı %15-20, 
taze koku, pembe-kırmızı arası ton
```

---

## 📸 4. KAMERA TEST

```cmd
Terminal:
[ ] curl -H "x-api-key: YOUR_API_KEY" -X POST http://localhost:8001/api/camera/test
[ ] Başarılı yanıt aldı (veya mock mode)
[ ] Backend logda kamera mesajı var
```

---

## 🏪 5. EĞİTİM MODU TEST

```cmd
Web: Terazi Ekranı

[ ] "🟡 Eğitim Modu" badge'i görünüyor
[ ] 3 PLU kartı görünüyor
[ ] PLU 101 → "✓ Seç" → Toast: fotoğraf çekildi
[ ] PLU 102 → 3 kez seç
[ ] PLU 103 → 3 kez seç
[ ] Her seçimde fotoğraf sayısı artıyor
```

---

## 📊 6. DASHBOARD KONTROL

```cmd
Web: Dashboard

[ ] Toplam Fotoğraf: ~9-15 (seçim sayısı)
[ ] AI Kontrol: 0 (henüz üretim modu değil)
[ ] "Fotoğraflar" sekmesi → 9-15 fotoğraf kartı
[ ] "PLU İstatistikleri" → Her PLU'da fotoğraf sayısı
```

---

## 📥 7. TOPLU FOTOĞRAF YÜKLEME (Opsiyonel)

```cmd
[ ] Fotoğrafları hazırla: C:\terazi-photos\101\, 102\, 103\
[ ] cd backend
[ ] python import_photos.py
[ ] Klasör yolu: C:\terazi-photos
[ ] Faz: 1 (training)
[ ] Devam: E
[ ] Import tamamlandı: 600/600 başarılı
[ ] Dashboard → Toplam Fotoğraf: 615+
```

---

## 🚀 8. ÜRETİM MODUNA GEÇİŞ

```cmd
Web: Dashboard

[ ] "🚀 Üretim Moduna Geç" → Tıkla
[ ] Toast: Mod değişti
[ ] Badge: "🟢 Üretim Modu"
[ ] Terazi Ekranı → "🟢 Üretim Modu" badge'i var
```

---

## 🤖 9. AI ANALİZİ TEST

```cmd
Web: Terazi Ekranı

[ ] PLU 101 → "✓ Seç"
[ ] Toast: Fotoğraf çekildi + AI başladı
[ ] 2-3 saniye bekle
[ ] Backend log: AI analysis completed
[ ] Dashboard → AI Kontrolleri (1) → Yeni kart görünüyor
[ ] PLU 102 → 3 kez test
[ ] PLU 103 → 3 kez test
[ ] Toplam 7+ AI kontrol sonucu var
```

---

## 📈 10. SONUÇ DEĞERLENDİRME

```cmd
Web: Dashboard

[ ] AI Kontrol Sayısı: 7+
[ ] Uyumlu/Uyumsuz sayıları gösteriliyor
[ ] Doğruluk Oranı kartı var (%)
[ ] Doğruluk Oranı: ___%
[ ] "AI Kontrolleri" sekmesi → Detaylı kartlar
[ ] Güven skorları gösteriliyor
```

**Değerlendirme:**
- ✅ %80+: Mükemmel
- ⚠️ %60-80: Kabul edilebilir
- ❌ %60 altı: Sorun var

---

## ✅ 11. FİNAL KONTROL

```cmd
[ ] Backend çalışıyor (terminal açık)
[ ] Frontend çalışıyor (terminal açık)
[ ] Web arayüzü açık ve responsive
[ ] PLU'lar oluşturuldu (3+)
[ ] Fotoğraflar toplandı (15+)
[ ] Üretim moduna geçildi
[ ] AI analizi çalıştı (7+ sonuç)
[ ] Dashboard istatistikler doğru
[ ] Kamera test edildi (gerçek/mock)
```

---

## 🎯 BAŞARI KRİTERLERİ

**Tümü ✅ ise:**
- 🎉 Sistem çalışıyor!
- 🚀 Kullanıma hazır
- 📊 İzlemeye başla

**Bazıları ❌ ise:**
- 📄 `TEST_REHBERI.md` → Detaylı adımlar
- 🔧 Sorun Giderme bölümü
- 📞 Test sonuç raporu paylaş

---

## 🔄 GÜNDELİK KULLANIM

```cmd
Sabah:
[ ] start_terazi.bat
[ ] Web: http://localhost:3000
[ ] Her iki terminal açık

Gün Boyu:
[ ] Normal terazi kullanımı
[ ] PLU seçimleri otomatik işliyor

Akşam:
[ ] Dashboard → Günlük istatistikler
[ ] Doğruluk oranını kontrol et
[ ] Terminalleri kapat (CTRL+C)
```

---

## 📞 DESTEK

**Sorun varsa:**
1. Backend log'u kontrol et
2. `test_terazi.bat` çalıştır
3. Test sonuç raporu hazırla (`TEST_REHBERI.md` → Adım 12.2)
4. Benimle paylaş

**Dokümantasyon:**
- `TEST_REHBERI.md` - Detaylı test adımları
- `TERAZI_KURULUM.md` - Kurulum sorunları
- `KULLANIM_REHBERI.md` - Kullanım kılavuzu

---

✅ **Her madde işaretlendi mi? Harikasınız! Sistem kullanıma hazır!** 🎉
