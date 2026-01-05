"""
Fotoğraf Dışa Aktarma Scripti
MongoDB'deki fotoğrafları dosya sistemine kaydeder

Kullanım:
    python export_photos.py
"""

import os
import sys
import base64
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import asyncio
from datetime import datetime

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
db_name = os.environ.get('DB_NAME', 'test_database')

class PhotoExporter:
    def __init__(self):
        self.client = None
        self.db = None
        self.stats = {
            'total': 0,
            'exported': 0,
            'failed': 0
        }
    
    async def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(mongo_url)
            self.db = self.client[db_name]
            await self.db.command('ping')
            print(f"✅ MongoDB bağlantısı başarılı\n")
            return True
        except Exception as e:
            print(f"❌ MongoDB bağlantı hatası: {e}")
            return False
    
    async def export_photos(self, output_folder: Path, plu_code: str = None, phase: str = None):
        """Export photos from database to file system"""
        
        # Build query
        query = {}
        if plu_code:
            query["plu_code"] = plu_code
        if phase:
            query["phase"] = phase
        
        # Count total
        self.stats['total'] = await self.db.captured_images.count_documents(query)
        
        if self.stats['total'] == 0:
            print("❌ Fotoğraf bulunamadı")
            return
        
        print(f"📸 Toplam {self.stats['total']} fotoğraf bulundu, dışa aktarılıyor...\n")
        
        # Create output folder
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        cursor = self.db.captured_images.find(query, {"_id": 0})
        
        async for doc in cursor:
            try:
                plu = doc['plu_code']
                img_id = doc['id']
                timestamp = doc.get('timestamp', datetime.now().isoformat())
                image_base64 = doc['image_base64']
                phase_label = doc.get('phase', 'unknown')
                
                # Create PLU folder
                plu_folder = output_folder / plu
                plu_folder.mkdir(exist_ok=True)
                
                # Create filename
                ts = timestamp.replace(':', '-').replace('.', '-')[:19]
                filename = f"{phase_label}_{ts}_{img_id[:8]}.jpg"
                filepath = plu_folder / filename
                
                # Decode and save
                image_bytes = base64.b64decode(image_base64)
                filepath.write_bytes(image_bytes)
                
                self.stats['exported'] += 1
                
                if self.stats['exported'] % 10 == 0:
                    print(f"  ✅ {self.stats['exported']}/{self.stats['total']} dışa aktarıldı...")
            
            except Exception as e:
                print(f"  ❌ Hata: {doc.get('id', 'unknown')} - {e}")
                self.stats['failed'] += 1
        
        print(f"\n✅ Dışa aktarma tamamlandı!")
    
    async def show_stats(self, output_folder: Path):
        """Show final statistics"""
        print("\n" + "="*60)
        print("📊 DIŞA AKTARMA İSTATİSTİKLERİ")
        print("="*60)
        print(f"Toplam:           {self.stats['total']}")
        print(f"✅ Başarılı:      {self.stats['exported']}")
        print(f"❌ Başarısız:     {self.stats['failed']}")
        print(f"📂 Klasör:        {output_folder.absolute()}")
        print("="*60 + "\n")
    
    async def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()

async def main():
    print("\n" + "="*60)
    print("📤 FOTOĞRAF DIŞA AKTARMA ARACI")
    print("="*60 + "\n")
    
    # Get output folder
    default_output = "C:\\terazi-export"
    print(f"Fotoğrafların kaydedileceği klasör")
    print(f"Varsayılan: {default_output}\n")
    
    output_input = input("Çıktı klasörü (Enter = varsayılan): ").strip()
    output_folder = Path(output_input) if output_input else Path(default_output)
    
    # Get filters
    print(f"\nFiltreler (boş bırakabilirsiniz):")
    plu_input = input("PLU kodu (örn: 101): ").strip() or None
    
    print(f"Faz:")
    print(f"  1. Tümü (varsayılan)")
    print(f"  2. training (Eğitim)")
    print(f"  3. production (Üretim)")
    phase_input = input("Seçim (1/2/3): ").strip()
    phase = None
    if phase_input == "2":
        phase = "training"
    elif phase_input == "3":
        phase = "production"
    
    print(f"\n📂 Çıktı: {output_folder}")
    if plu_input:
        print(f"🔍 PLU Filtresi: {plu_input}")
    if phase:
        print(f"🎯 Faz Filtresi: {phase}")
    
    print(f"\nDevam etmek istiyor musunuz? (E/H): ", end="")
    confirm = input().strip().upper()
    if confirm != 'E':
        print("❌ İşlem iptal edildi")
        return
    
    # Start export
    exporter = PhotoExporter()
    
    if not await exporter.connect():
        return
    
    try:
        await exporter.export_photos(output_folder, plu_input, phase)
        await exporter.show_stats(output_folder)
    except KeyboardInterrupt:
        print("\n\n⚠️  İşlem kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
    finally:
        await exporter.close()

if __name__ == "__main__":
    asyncio.run(main())
