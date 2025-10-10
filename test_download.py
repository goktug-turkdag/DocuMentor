from datasets import load_dataset
import os

# Hugging Face kütüphanesinin çevrimdışı modda çalışmadığından emin olalım
os.environ["HF_HUB_OFFLINE"] = "0"

print("--- Test Başladı ---")
print("Hugging Face'den 'databricks-dolly-15k' veri setini indirme deneniyor...")

try:
    # Bu satır, internetten veri setini indirmeye çalışacak
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    print("\nBAŞARILI: Veri seti sorunsuz bir şekilde indirildi!")
    print(f"Veri setinde toplam {len(dataset)} adet satır bulunmaktadır.")

except Exception as e:
    print("\n!!! HATA: İndirme sırasında bir sorun oluştu! !!!")
    print(f"\nHatanın detayı aşağıdadır:")
    print(e)

print("\n--- Test Bitti ---")