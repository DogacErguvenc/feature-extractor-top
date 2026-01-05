"""
Toplu Fotoğraf İçe Aktarma Scripti
Hazır fotoğrafları MongoDB'ye yükler

Kullanım:
    python import_photos.py

Klasör Yapısı:
    photos_folder/
        ├── 101/        # PLU kodu
        │   ├── img001.jpg
        │   ├── img002.jpg
        │   └── ...
        ├── 102/
        │   ├── img001.jpg
        │   └── ...
        └── ...
"""

import os
import sys
import base64
from pathlib import Path
from PIL import Image
import io
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import asyncio
from datetime import datetime, timezone
import uuid

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
db_name = os.environ.get('DB_NAME', 'test_database')

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

class PhotoImporter:
    def __init__(self):
        self.client = None
        self.db = None
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0
        }
    
    async def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(mongo_url)
            self.db = self.client[db_name]
            # Test connection
            await self.db.command('ping')
            print(f"✅ MongoDB bağlantısı başarılı: {mongo_url}")
            return True
        except Exception as e:
            print(f"❌ MongoDB bağlantı hatası: {e}")
            return False
    
    def process_image(self, image_path: Path) -> str:
        """Process image and convert to base64"""
        try:
            # Open image
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Resize if too large
                max_size = 1920
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert to JPEG base64
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                return img_base64
        except Exception as e:
            raise Exception(f"Görüntü işleme hatası: {e}")
    
    async def check_plu_exists(self, plu_code: str) -> bool:
        """Check if PLU exists in database"""
        plu = await self.db.plu_products.find_one({"plu_code": plu_code})
        return plu is not None
    
    async def import_photo(self, image_path: Path, plu_code: str, phase: str = "training"):
        """Import single photo to database"""
        try:
            # Process image
            image_base64 = self.process_image(image_path)
            
            # Create document
            doc = {
                "id": str(uuid.uuid4()),
                "plu_code": plu_code,
                "image_base64": image_base64,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase": phase
            }
            
            # Insert to database
            await self.db.captured_images.insert_one(doc)
            
            self.stats['successful'] += 1
            return True
        except Exception as e:
            print(f"  ❌ Hata: {image_path.name} - {e}")
            self.stats['failed'] += 1
            return False
    
    async def import_from_folder(self, folder_path: Path, phase: str = "training"):
        """Import all photos from folder structure"""
        
        if not folder_path.exists():
            print(f"❌ Klasör bulunamadı: {folder_path}")
            return
        
        # Get all PLU folders
        plu_folders = [f for f in folder_path.iterdir() if f.is_dir()]
        
        if not plu_folders:
            print(f"❌ PLU klasörleri bulunamadı: {folder_path}")
            return
        
        print(f"\n📂 Toplam {len(plu_folders)} PLU klasörü bulundu\n")
        
        for plu_folder in plu_folders:
            plu_code = plu_folder.name
            
            # Check if PLU exists
            plu_exists = await self.check_plu_exists(plu_code)
            
            if not plu_exists:
                print(f"⚠️  PLU {plu_code}: Veritabanında bulunamadı, lütfen önce PLU'yu oluşturun")
                continue
            
            # Get all image files
            image_files = []
            for ext in SUPPORTED_FORMATS:
                image_files.extend(plu_folder.glob(f"*{ext}"))
                image_files.extend(plu_folder.glob(f"*{ext.upper()}"))
            
            if not image_files:
                print(f"⚠️  PLU {plu_code}: Fotoğraf bulunamadı")
                self.stats['skipped'] += len([f for f in plu_folder.iterdir() if f.is_file()])
                continue
            
            print(f"📸 PLU {plu_code}: {len(image_files)} fotoğraf bulundu, yükleniyor...")
            
            self.stats['total_files'] += len(image_files)
            
            # Import each image
            for idx, image_file in enumerate(image_files, 1):
                success = await self.import_photo(image_file, plu_code, phase)
                if success and idx % 10 == 0:
                    print(f"  ✅ {idx}/{len(image_files)} yüklendi...")
            
            print(f"  ✅ PLU {plu_code} tamamlandı: {len(image_files)} fotoğraf\n")
    
    async def show_stats(self):
        """Show final statistics"""
        print("\n" + "="*60)
        print("📊 İÇE AKTARMA İSTATİSTİKLERİ")
        print("="*60)
        print(f"Toplam Dosya:     {self.stats['total_files']}")
        print(f"✅ Başarılı:      {self.stats['successful']}")
        print(f"❌ Başarısız:     {self.stats['failed']}")
        print(f"⚠️  Atlanan:       {self.stats['skipped']}")
        print("="*60)
        
        # Show database stats
        total_images = await self.db.captured_images.count_documents({})
        training_images = await self.db.captured_images.count_documents({"phase": "training"})
        
        print(f"\n📦 VERİTABANI DURUMU")
        print(f"Toplam Fotoğraf:  {total_images}")
        print(f"Eğitim Fazı:      {training_images}")
        print("="*60 + "\n")
    
    async def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            print("✅ Bağlantı kapatıldı\n")

async def main():
    print("\n" + "="*60)
    print("📸 TOPLU FOTOĞRAF İÇE AKTARMA ARACI")
    print("="*60 + "\n")
    
    # Get folder path from user
    default_path = "C:\\terazi-photos"
    print(f"Fotoğrafların bulunduğu klasörü girin")
    print(f"Varsayılan: {default_path}")
    print(f"\nKlasör yapısı:")
    print(f"  klasor/")
    print(f"    ├── 101/  (PLU kodu)")
    print(f"    │   ├── img001.jpg")
    print(f"    │   └── img002.jpg")
    print(f"    ├── 102/")
    print(f"    └── ...\n")
    
    folder_input = input("Klasör yolu (Enter = varsayılan): ").strip()
    folder_path = Path(folder_input) if folder_input else Path(default_path)
    
    # Get phase
    print(f"\nFotoğraflar hangi faza aktarılsın?")
    print(f"  1. training (Eğitim - varsayılan)")
    print(f"  2. production (Üretim)")
    phase_input = input("Seçim (1/2): ").strip()
    phase = "production" if phase_input == "2" else "training"
    
    print(f"\n📂 Klasör: {folder_path}")
    print(f"🎯 Faz: {phase}")
    print(f"\nDevam etmek istiyor musunuz? (E/H): ", end="")
    
    confirm = input().strip().upper()
    if confirm != 'E':
        print("❌ İşlem iptal edildi")
        return
    
    # Start import
    importer = PhotoImporter()
    
    if not await importer.connect():
        return
    
    print(f"\n🚀 İçe aktarma başlıyor...\n")
    
    try:
        await importer.import_from_folder(folder_path, phase)
        await importer.show_stats()
    except KeyboardInterrupt:
        print("\n\n⚠️  İşlem kullanıcı tarafından durduruldu")
        await importer.show_stats()
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
    finally:
        await importer.close()

if __name__ == "__main__":
    asyncio.run(main())
