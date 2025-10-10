import google.generativeai as genai
from dotenv import load_dotenv
import os

print("--- Google AI Model Yetki Kontrolü Başladı ---")

# .env dosyasından API anahtarını yükle
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("\nHATA: .env dosyasında GOOGLE_API_KEY bulunamadı.")
else:
    try:
        genai.configure(api_key=api_key)
        print("\nAPI Anahtarı başarıyla yapılandırıldı.")
        print("Kullanılabilir modeller listeleniyor...")
        
        model_count = 0
        for m in genai.list_models():
            # Sadece 'generateContent' metodunu destekleyen modelleri listele
            if 'generateContent' in m.supported_generation_methods:
                print(f"- Model Adı: {m.name}")
                model_count += 1
        
        if model_count == 0:
            print("\nUYARI: Bu API anahtarı için 'generateContent' metodunu destekleyen hiçbir model bulunamadı.")
            print("Lütfen Google Cloud projenizin bölgesini ve API izinlerini kontrol edin.")
        else:
            print(f"\nBAŞARILI: Toplam {model_count} adet kullanılabilir model bulundu.")

    except Exception as e:
        print(f"\n!!! HATA: API'ye bağlanırken bir sorun oluştu! !!!")
        print("Hatanın muhtemel sebebi API anahtarının geçersiz olması veya Google Cloud proje ayarlarınızdır.")
        print("\nOrijinal Hata Mesajı:")
        print(e)

print("\n--- Kontrol Bitti ---")