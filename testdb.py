import json
import chromadb
import requests

# 1. LM Studio'daki yerel modelinize metni gönderen fonksiyon
def yerel_model_ile_embed_et(metin):
    url = "http://127.0.0.1:1234/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    
    # 'model': 'local' dediğimizde LM Studio o an arayüzde hangi modeli seçtiyseniz onu çalıştırır
    data = {
        "input": metin,
        "model": "local" 
    }
    
    # İsteği doğrudan sizin bilgisayarınızdaki LM Studio sunucusuna atıyoruz
    cevap = requests.post(url, headers=headers, data=json.dumps(data))
    
    if cevap.status_code == 200:
        return cevap.json()['data'][0]['embedding']
    else:
        raise Exception(f"LM Studio Bağlantı Hatası: {cevap.text}")

# --- TEST ADIMI ---
test_metni = "LM Studio içindeki kendi embedding modelimi test ediyorum."
print(f"Modelinize gönderilen metin: '{test_metni}'")

# LM Studio'daki modeliniz çalışıyor ve vektör üretiyor
vektor = yerel_model_ile_embed_et(test_metni)

print("\n[BAŞARILI] LM Studio modelinizden gelen embedding vektörü alındı!")
print(f"Vektörün ilk 5 sayısı: {vektor[:5]}")
print(f"Modelinizin ürettiği toplam boyut (Dimension): {len(vektor)}")

# 2. Üretilen bu yerel vektörü bilgisayarınızdaki Chroma DB'ye kaydetme
# (Bu satır bilgisayarınızda 'test_veri_tabani' adında bir klasör açar)
chroma_istemci = chromadb.PersistentClient(path="./test_veri_tabani")
koleksiyon = chroma_istemci.get_or_create_collection(name="model_test_koleksiyonu")

koleksiyon.add(
    embeddings=[vektor],
    documents=[test_metni],
    ids=["test_id_1"]
)
print("\n[BAŞARILI] Modelinizin ürettiği vektör bilgisayarınızdaki Chroma DB'ye kaydedildi!")
