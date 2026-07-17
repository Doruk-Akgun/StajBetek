import json
import chromadb
import requests

LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"

# 1. Yardımcı Fonksiyon: LM Studio'dan Embedding Alır
def get_embedding(text):
    url = f"{LM_STUDIO_BASE_URL}/embeddings"
    headers = {"Content-Type": "application/json"}
    data = {"input": text, "model": "local"}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print("STATUS:", response.status_code)
    print("BODY:", response.text) 
    return response.json()['data'][0]['embedding']

# 2. Yardımcı Fonksiyon: LM Studio'daki LLM modeline soru sorar (Chat Completion)
def ask_llm(system_prompt, user_prompt):
    url = f"{LM_STUDIO_BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3 # Deneysel RAG için daha tutarlı (düşük) yaratıcılık
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()['choices'][0]['message']['content']

# 3. Yerel Veri Tabanını (Chroma DB) Hazırlama
chroma_client = chromadb.PersistentClient(path="./deneysel_rag_db")
collection = chroma_client.get_or_create_collection(name="bilgi_deposu")

# ---- ADIM 1: DOKÜMANLARI VERİ TABANINA EKLEME ----
# LLM'in normalde bilmediği, sizin şirketinize veya projenize özel hayali bilgiler:
ozel_bilgiler = [
    "Şirketimizin yeni gizli projesinin adı 'Proje Delta'dır ve kuantum şifreleme üzerine odaklanır.",
    "Proje Delta'nın baş mühendisi Ahmet Yılmaz'dır ve laboratuvar 4. katta yer almaktadır.",
    "Laboratuvara girişler sadece biyometrik göz taraması ile yapılabilir."
]

print("1. Adım: Özel dokümanlar embed ediliyor ve veri tabanına yükleniyor...")
for i, metin in enumerate(ozel_bilgiler):
    vektor = get_embedding(metin)
    collection.add(embeddings=[vektor], documents=[metin], ids=[f"id_{i}"])
print("Yükleme tamamlandı!\n")

# ---- ADIM 2: KULLANICI SORUSU VE RETRIEVAL (VERİ GETİRME) ----
kullanici_sorusu = "Kuantum projesinin başındaki mühendis kim ve nerede çalışıyor?"
print(f"Kullanıcı Sorusu: {kullanici_sorusu}")

# Soruyu embed edip veri tabanında arıyoruz
sorgu_vektoru = get_embedding(kullanici_sorusu)
arama_sonuclari = collection.query(query_embeddings=[sorgu_vektoru], n_results=2)

# En alakalı kaynak dokümanları birleştiriyoruz (Context/Bağlam oluşturma)
getirilen_kaynaklar = " ".join(arama_sonuclari['documents'][0])
print(f"Veri Tabanından Getirilen Kaynak: {getirilen_kaynaklar}\n")

# ---- ADIM 3: GENERATION (CEVAP ÜRETME) ----
# LLM'e sadece getirdiğimiz kaynağa sadık kalarak cevap vermesini söylüyoruz
sistem_talimati = f"""Sen yardımcı bir asistansın. Kullanıcının sorusuna SADECE aşağıda verilen kaynak bilgilere dayanarak cevap ver. 
Eğer cevap kaynakta yoksa 'Bu bilgiye sahip değilim' de. Uydurma.

KAYNAK BİLGİ:
{getirilen_kaynaklar}"""

print("3. Adım: LLM'e kaynak bilgi gönderiliyor ve cevap üretiliyor...")
llm_cevabi = ask_llm(system_prompt=sistem_talimati, user_prompt=kullanici_sorusu)

print("\n=== RAG SİSTEMİNİN CEVABI ===")
print(llm_cevabi)
