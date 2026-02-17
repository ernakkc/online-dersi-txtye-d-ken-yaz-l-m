import whisper
import os
import warnings

# Gereksiz uyarıları temizle
warnings.filterwarnings("ignore")

def main():
    print("-" * 60)
    print("   WHISPER OTOMATİK DEŞİFRE ARACI (Kullanıcı Girişli)   ")
    print("-" * 60)

    # --- 1. KULLANICIDAN VİDEO YOLUNU ALMA ---
    print("\nLütfen işlenecek video/ses dosyasını bu pencereye sürükleyip bırakın")
    print("veya dosya yolunu yapıştırıp ENTER'a basın:")
    
    # Kullanıcıdan yolu al ve tırnak işaretlerini temizle (Windows bazen yolun başına sonuna " ekler)
    video_yolu = input(">> ").strip().strip('"')

    # Dosya var mı kontrol et
    if not os.path.exists(video_yolu):
        print(f"\n[HATA] Dosya bulunamadı: {video_yolu}")
        return

    # --- 2. ÇIKTI KLASÖRÜ VE DOSYA ADI AYARLAMA ---
    # Scriptin çalıştığı dizini bul
    calisma_dizini = os.getcwd()
    
    # 'output' klasörü oluştur (yoksa yaratır, varsa hata vermez)
    cikti_klasoru = os.path.join(calisma_dizini, "output")
    os.makedirs(cikti_klasoru, exist_ok=True)

    # Videonun dosya adını al (örn: video.mp4)
    dosya_adi_uzantili = os.path.basename(video_yolu)
    # Uzantıyı atıp sadece ismi al (örn: video)
    dosya_adi_saf = os.path.splitext(dosya_adi_uzantili)[0]
    
    # Yeni metin dosyası yolu (output klasörü içinde)
    hedef_dosya_yolu = os.path.join(cikti_klasoru, f"{dosya_adi_saf}.txt")

    try:
        # --- 3. MODEL YÜKLEME ---
        print(f"\nSeçilen Dosya: {dosya_adi_uzantili}")
        print("Model yükleniyor (Base - CPU)...")
        model = whisper.load_model("base", device="cpu")

        # --- 4. DEŞİFRE ---
        print("\nVideo metne dökülüyor... (Lütfen bekleyin)")
        
        # fp16=False CPU için gereklidir
        result = model.transcribe(video_yolu, fp16=False)
        metin = result["text"]
        
        # --- 5. KAYDETME ---
        with open(hedef_dosya_yolu, "w", encoding="utf-8") as f:
            f.write(metin.strip())

        print("\n" + "=" * 60)
        print("İŞLEM BAŞARILI!")
        print(f"Metin dosyası şuraya kaydedildi:\n-> {hedef_dosya_yolu}")
        print("=" * 60)

    except Exception as e:
        print("\n" + "!" * 50)
        print(f"BİR HATA OLUŞTU:\n{e}")
        print("!" * 50)

if __name__ == "__main__":
    main()
    print("\nÇıkmak için ENTER tuşuna basınız...")
    input()
