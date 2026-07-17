"""
Embedding'leri LM Studio ile üretip Chroma DB'ye kaydeden ve
2 boyutlu grafikte görselleştiren script.

Gereksinimler:
    pip install chromadb requests matplotlib scikit-learn numpy
"""

import json
import chromadb
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ---------------------------------------------------------
# 1. LM Studio'daki embedding modeline istek atan fonksiyon
# ---------------------------------------------------------
def yerel_model_ile_embed_et(metin):
    url = "http://127.0.0.1:1234/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    data = {"input": metin, "model": "local"}

    cevap = requests.post(url, headers=headers, data=json.dumps(data))
    if cevap.status_code == 200:
        return cevap.json()["data"][0]["embedding"]
    else:
        raise Exception(f"LM Studio Bağlantı Hatası: {cevap.status_code} - {cevap.text}")


# ---------------------------------------------------------
# 2. Görselleştirme için birkaç farklı konudan örnek metin
#    (Tek nokta göremezsiniz; anlamlı bir grafik için en az
#     birkaç, tercihen 10+ metin gerekir)
# ---------------------------------------------------------
ornek_metinler = [
    # Teknoloji
    "Yapay zeka modelleri her geçen gün daha akıllı hale geliyor.",
    "Bilgisayarlar artık insan dilini çok daha iyi anlıyor.",
    "Yeni işlemci mimarisi enerji tüketimini yarı yarıya azaltıyor.",
    # Yemek
    "Bu tarifte domates, soğan ve zeytinyağı kullanılıyor.",
    "Türk mutfağında kebap çeşitleri oldukça zengindir.",
    "Taze fesleğen yapraklarıyla hazırlanan salata çok lezzetliydi.",
    # Spor
    "Takım son maçta harika bir performans sergiledi.",
    "Maraton koşucusu bitiş çizgisine ilk sırada ulaştı.",
    "Antrenman programı haftada beş gün olacak şekilde planlandı.",
    # Doğa
    "Dağların zirvesinde kar hâlâ erimemişti.",
    "Orman yangınları son yıllarda giderek artıyor.",
    "Nehir kıyısındaki kuşlar sabah erkenden ötmeye başladı.",
]

etiketler = (
    ["Teknoloji"] * 3
    + ["Yemek"] * 3
    + ["Spor"] * 3
    + ["Doğa"] * 3
)

# ---------------------------------------------------------
# 3. Embedding'leri üret ve Chroma'ya kaydet
# ---------------------------------------------------------
print("Embedding'ler üretiliyor...")
vektorler = []
for i, metin in enumerate(ornek_metinler):
    v = yerel_model_ile_embed_et(metin)
    vektorler.append(v)
    print(f"  [{i+1}/{len(ornek_metinler)}] tamamlandı (boyut: {len(v)})")

chroma_istemci = chromadb.PersistentClient(path="./test_veri_tabani")
koleksiyon = chroma_istemci.get_or_create_collection(name="model_test_koleksiyonu")

koleksiyon.upsert(
    embeddings=vektorler,
    documents=ornek_metinler,
    ids=[f"ornek_{i}" for i in range(len(ornek_metinler))],
    metadatas=[{"kategori": e} for e in etiketler],
)
print("\nChroma DB'ye kaydedildi.\n")

# ---------------------------------------------------------
# 4. Chroma'dan embedding'leri geri çek
# ---------------------------------------------------------
veri = koleksiyon.get(
    ids=[f"ornek_{i}" for i in range(len(ornek_metinler))],
    include=["embeddings", "documents", "metadatas"]
)
X = np.array(veri["embeddings"])
dokumanlar = veri["documents"]
kategoriler = [m["kategori"] for m in veri["metadatas"]]

print(f"Toplam {X.shape[0]} vektör, her biri {X.shape[1]} boyutlu.\n")

# ---------------------------------------------------------
# 5. Boyut indirgeme: PCA ve t-SNE ile 2 boyuta indir
# ---------------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# t-SNE için perplexity, örnek sayısından küçük olmalı
perplexity_degeri = min(5, max(2, len(X) - 1))
tsne = TSNE(n_components=2, perplexity=perplexity_degeri, random_state=42, init="pca")
X_tsne = tsne.fit_transform(X)

# ---------------------------------------------------------
# 6. Grafik çiz
# ---------------------------------------------------------
benzersiz_kategoriler = sorted(set(kategoriler))
renkler = plt.cm.tab10(np.linspace(0, 1, len(benzersiz_kategoriler)))
renk_haritasi = dict(zip(benzersiz_kategoriler, renkler))

fig, eksenler = plt.subplots(1, 2, figsize=(14, 6))

for ax, veri_noktalari, baslik in [
    (eksenler[0], X_pca, "PCA ile 2 Boyuta İndirgeme"),
    (eksenler[1], X_tsne, "t-SNE ile 2 Boyuta İndirgeme"),
]:
    for kategori in benzersiz_kategoriler:
        idx = [i for i, k in enumerate(kategoriler) if k == kategori]
        ax.scatter(
            veri_noktalari[idx, 0],
            veri_noktalari[idx, 1],
            label=kategori,
            color=renk_haritasi[kategori],
            s=100,
            edgecolors="black",
        )
    ax.set_title(baslik)
    ax.set_xlabel("Boyut 1")
    ax.set_ylabel("Boyut 2")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("embedding_gorsellestirme.png", dpi=150)
print("Grafik 'embedding_gorsellestirme.png' olarak kaydedildi.")
plt.show()