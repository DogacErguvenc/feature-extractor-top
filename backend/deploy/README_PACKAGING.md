# Backend EXE + Setup Packaging

Bu klasor, backend'i kaynak kod vermeden teslim etmek icin kullanilir.

## 1) EXE build

```powershell
powershell -ExecutionPolicy Bypass -File .\backend\deploy\build_backend_exe.ps1
```

Cikti:
- `backend\dist\terazi_backend\`

## 2) Setup build (Inno Setup)

```powershell
powershell -ExecutionPolicy Bypass -File .\backend\deploy\build_backend_exe.ps1 -BuildSetup -Version 1.0.0
```

Cikti:
- `backend\deploy\terazi_backend_setup_1.0.0.exe`

## 3) Musteri tarafi

Installer kurulduktan sonra:
1. Kurulum klasorundeki `.env.example` dosyasini `.env` olarak kopyalayin.
2. `MONGO_URL` ve `DB_NAME` degerlerini doldurun.
3. Gerekirse `AI_PROVIDER=butcher_resnet` yapin.
4. `start_backend.bat` ile manuel test edin.

Not:
- Setup icindeki "Windows acilisinda backend'i otomatik baslat" secenegi isaretlenirse
  `terazi-ai-backend` adli Task Scheduler gorevi olusturulur.
