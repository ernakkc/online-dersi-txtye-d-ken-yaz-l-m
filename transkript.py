import whisper
import os
import warnings

warnings.filterwarnings("ignore")

print("Yazılım çalıştırılıyor")

model = whisper.load_model("base", device="cpu")

video_yolu = r"C:\Users\annes\Desktop\ozetleyici\edeb13.mp4" 

print("Video metne dökülüyor... Biraz bekle, siyah ekran kapanmasın.")

result = model.transcribe(video_yolu, fp16=False)

hedef_klasor = r"C:\Users\annes\Desktop\ozetleyici"
dosya_adi = "edeb15.txt"

tam_yol = os.path.join(hedef_klasor, dosya_adi)

with open(tam_yol, "w", encoding="utf-8") as f:
    f.write(result["text"])

print(f"İşlem tamam! Dosyan şurada: {tam_yol}")