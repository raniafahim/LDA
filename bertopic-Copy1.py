#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('uv pip install -q bertopic spacy polars datasets hf_xet')


# In[2]:


get_ipython().system('uv pip install -q https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.7.0/fr_core_news_sm-3.7.0-py3-none-any.whl')


# In[3]:


#compaatible uniquement avec verison python 3.11 et moins et mettre la mémoire au max
get_ipython().system('uv pip install cudf-cu12 dask-cudf-cu12 --extra-index-url=https://pypi.nvidia.com')
get_ipython().system('uv pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com')
get_ipython().system('uv pip install cugraph-cu12 --extra-index-url=https://pypi.nvidia.com')
get_ipython().system('uv pip install cupy-cuda12x -f https://pip.cupy.dev/aarch64')


# In[4]:


get_ipython().system('uv pip install -q langchain-huggingface==0.0.3')


# In[5]:


from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from cuml.cluster import HDBSCAN
from scipy.cluster import hierarchy as sch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from cuml.manifold import UMAP
import polars as pl
import spacy
import pandas as pd
import numpy as np
import time
from sklearn.decomposition import PCA
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import tqdm as notebook_tqdm


# In[6]:


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In[7]:


nlp = spacy.load("fr_core_news_md", disable=["parser", "ner", "tagger"])


# In[8]:


def preprocess(docs):
    cleaned = []
    for doc in nlp.pipe(docs, batch_size=100, n_process=4):
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
            cleaned.append(' '.join(tokens))
    return cleaned


# In[9]:


DICTIONNARY =  ['accord','entreprise', 'preambule', 'sommaire',  'code', 'syndical', 'responsable', 'representant', 
                'present', 'ca', 'organisation', 'preambule', 'peut', 'etre', 'contrat','travail', 'ressources','humaines', 'mise',
                'ainsi', 'et', 'ou', 'alors','collaborateur', 'ci', 'apres', 'party', 'signataire', 'tout', 'etat', 'cause', 'societe', 
                'notamment','article','activite', 'cette', 'donc', 'si', 'sous', 'disposition', 'convention', 'collective', 'dans', 'a', 'cadre',
                'signataire', 'partie', 'parties', 'entre', 'doit', 'mme', 'mr', 'madame', 'monsieur'
               ]

DICTIONNARY_STEM = ['part', 'signatair', 'organis', 'syndical', 
                    'dont', 'sieg', 'social', 'conseil', 'prud', 'homm', 
                   'vi', 'professionnel', 'disposit', 'legal', 'conventionnel']


# In[10]:


import re

def normalize(text):
    return text.lower().strip()

def split_text_by_sentences(text, flagged_sentences):
    split_texts = []
    positions = []

    normalized_text = normalize(text)

    # On garde un mapping (titre original, position) pour préserver les titres initiaux
    for sentence in flagged_sentences:
        norm_sentence = normalize(sentence)
        pos = normalized_text.find(norm_sentence)
        if pos != -1:
            # On retrouve la position réelle dans le texte original
            real_pos = text.lower().find(sentence.lower())
            if real_pos != -1:
                positions.append(real_pos)

    # Si aucune position trouvée, retourner le texte complet
    if not positions:
        return [text]

    positions = sorted(set(positions))
    positions.insert(0, 0)
    positions.append(len(text))

    for i in range(len(positions) - 1):
        start = positions[i]
        end = positions[i + 1]
        split_texts.append(text[start:end].strip())

    return split_texts



# In[11]:


def split_text_with_titles(text, summary_titles):
    chunks = split_text_by_sentences(text, summary_titles)
    result = {}
    for title in summary_titles:
        for chunk in chunks:
            if normalize(title) in normalize(chunk[:len(title)+30]):
                result[title] = chunk.strip()
                break
    return result


# In[14]:


model_kwargs = {'device': 'cuda'} 
#model_kwargs = {'device': 'cpu'}  
MODEL_NAME_EMBEDDER="BAAI/bge-small-en-v1.5"  #petit modèle en anglais
#MODEL_NAME_EMBEDDER="BAAI/bge-m3" #gros modèle multilingue

embedder = HuggingFaceEmbeddings(
    model_name=MODEL_NAME_EMBEDDER, 
    model_kwargs=model_kwargs,
    show_progress=True
)


phrases_non_metier = [
    "Révision de l’accord",
    "Dénonciation de l’accord",
    "Interprétation de l’accord",
    "Suivi de l’accord",
    "Durée de l’accord",
    "Formalités de publicité et de dépôt",
    "Publicité et dépôt",
    "Date d'effet et durée",
    "Champ d'application",
    "Clause de revoyure", 
    "Information des représentants du personnel", 
    "Dispositions relatives à l’accord",
    "Champ d’application",
    "Commission de suivi", 
    "Pause déjeuner du personnel", 
    "Modification de l'accord",
    "Adhésion"

]

# Embeddings des phrases non-métier
ref_embeddings = embedder.embed_documents(phrases_non_metier)

def filtre_par_similarite(phrases, seuil=0.85):##torp long utiliser version vectoisée
    results = []
    for phrase in phrases:
        emb = embedder.embed_query(phrase)
        sims = cosine_similarity([emb], ref_embeddings)[0]
        if max(sims) < seuil:
            results.append(phrase) 
    return results

def filtre_par_similarite_vectorise(phrases, seuil=0.85):
    if not phrases:
        return []

    phrase_embeddings = embedder.embed_documents(phrases)  
    sims = cosine_similarity(phrase_embeddings, ref_embeddings)

    # On garde les phrases dont la similarité max avec une phrase non-métier est < seuil
    keep_idx = np.max(sims, axis=1) < seuil
    return [phrase for phrase, keep in zip(phrases, keep_idx) if keep]


def filtre_chunks_par_titre(section_dict, phrases_non_metier, seuil=0.85): #seuil arbitraire : en tester plsr
    """
    Ne garde que les chunks dont le titre est peu similaire aux phrases non métier.
    """
    if not section_dict:
        return []

    titres = list(section_dict.keys())
    chunks = list(section_dict.values())

    # Embeddings des titres de section
    titre_embeddings = embedder.embed_documents(titres)
    ref_embeddings = embedder.embed_documents(phrases_non_metier)

    sims = cosine_similarity(titre_embeddings, ref_embeddings)

    # On garde les chunks dont le titre est peu similaire aux phrases non métier
    keep_idx = np.max(sims, axis=1) < seuil
    return [chunk.strip() for chunk, keep in zip(chunks, keep_idx) if keep]



# In[17]:


sommaire_hs = pd.read_parquet("data/echantillon_1000_hs_accords_TOC.parquet")
df_hs = pd.read_parquet("data/echantillon_1000_hs_accords.parquet")
df_hs = df_hs.set_index("numdossier_new")
df_hs = df_hs.merge(sommaire_hs,how="inner",left_index=True,right_index=True)
df_hs = df_hs.rename(columns={"extracted_summary":"summary"})


# In[18]:


df_hs["section_dict"] = df_hs.apply(
    lambda row: split_text_with_titles(row["accorddocx"], row["summary"]),
    axis=1
)


# In[20]:


def get_all_chunks(section_dict):
    chunks = list(section_dict.values())
    return [chunk.strip() for chunk in chunks]


# In[21]:


def get_valid_chunks_filtered(section_dict, skip_titles=["préambule", "annexe"], seuil_sim=0.85):
    skip_titles_norm = [normalize(t) for t in skip_titles]

    # supprimer le préambule et avant 
    titles = list(section_dict.keys())
    preamble_idx = next((i for i, t in enumerate(titles) if "préambule" in normalize(t)), -1)
    if preamble_idx != -1:
        titles = titles[preamble_idx + 1:]

    # garder les titres valides uniquement
    valid_titles = [
        t for t in titles if all(skip_kw not in normalize(t) for skip_kw in skip_titles_norm)
    ]
    candidate_dict = {t: section_dict[t] for t in valid_titles}

    # filtrer par similarité des titres
    return filtre_chunks_par_titre(candidate_dict, phrases_non_metier, seuil=seuil_sim)


# # Sans filtrer les chunks

# In[22]:


df_hs["lda_documents"] = df_hs["section_dict"].apply(get_all_chunks)


# In[23]:


all_chunks_hs = [chunk for doc_chunks in df_hs["lda_documents"] for chunk in doc_chunks]
all_docs_cleaned = preprocess(all_chunks_hs)


# In[32]:


#Embeddings
start = time.time()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2",device='cuda')  
embeddings = embedding_model.encode(all_docs_cleaned, show_progress_bar=True)
print(f"[1] Embedding en {time.time() - start:.2f}s")


# In[33]:


# ACP --> plus rapide 
#start = time.time()
#pca_model = PCA(n_components=5)
#pca_embeddings = pca_model.fit_transform(embeddings)
#print(f"[PCA] en {time.time() - start:.2f}s")


# In[34]:


#Réduction UMAP --> trop long besoin de trouver une version avec gpu 
#start = time.time()
#umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
#umap_embeddings = umap_model.fit_transform(embeddings)
#print(f"[2] UMAP en {time.time() - start:.2f}s")


# In[35]:


#Réduction UMAP avec gpu

start = time.time()
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', verbose=True)
umap_embeddings = umap_model.fit_transform(embeddings)
print(f"[2] UMAP en {time.time() - start:.2f}s")



# In[36]:


# Clustering 
start = time.time()
hdbscan_model =  HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True, verbose=True)
clusters = hdbscan_model.fit_predict(pca_embeddings)
print(f"[3] HDBSCAN en {time.time() - start:.2f}s")
print(f"[3] Nombre de clusters trouvés : {len(np.unique(clusters))}")


# In[37]:


start = time.time()
topic_model = BERTopic(
    language="french",
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True
)
topics, probs = topic_model.fit_transform(all_docs_cleaned, embeddings=embeddings)
print(f"[4] BERTopic final en {time.time() - start:.2f}s")


# In[38]:


#from random import sample
#sample_docs = sample(all_docs_cleaned, 200)


# In[39]:


#topic_model_no_filter = BERTopic(language="french")
#topics_no_filter, probs_no_filter = topic_model_no_filter.fit_transform(sample_docs)


# In[40]:


topic_model.visualize_topics()


# In[41]:


topic_model.visualize_barchart()


# # En filtrant les chunks

# In[42]:


df_hs["lda_documents"] = df_hs["section_dict"].apply(get_valid_chunks_filtered)


# In[43]:


filtered_chunks_hs = [chunk for doc_chunks in df_hs["lda_documents"] for chunk in doc_chunks]
filtered_docs_cleaned = preprocess(all_chunks_hs)


# In[44]:


#Embeddings
start = time.time()
embedding_model_filtered = SentenceTransformer("all-MiniLM-L6-v2",device='cuda')  
embeddings_filtered = embedding_model_filtered.encode(all_docs_cleaned, show_progress_bar=True)
print(f"[1] Embedding en {time.time() - start:.2f}s")


# In[45]:


start = time.time()
topic_model_filtered = BERTopic(
    language="french",
    embedding_model=embedding_model_filtered,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True
)
topics, probs = topic_model_filtered.fit_transform(filtered_docs_cleaned, embeddings=embeddings_filtered)
print(f"[4] BERTopic final en {time.time() - start:.2f}s")


# In[46]:


#from bertopic import BERTopic
#topic_model_filter = BERTopic(language="french")
#topic_model_filter, probs_no_filter = topic_model_no_filter.fit_transform(all_docs_cleaned)


# In[47]:


topic_model_filtered.visualize_topics()


# In[48]:


topic_model_filtered.visualize_barchart()


# # BERTopic (KeyBERTInspired)

# In[ ]:


representation_model = KeyBERTInspired()

topic_model_KeyBERTInspired = BERTopic(representation_model=representation_model,language="french")
topics_KeyBERTInspired, probs_KeyBERTInspired = topic_model_KeyBERTInspired.fit_transform(filtered_docs_cleaned)


# In[52]:


topic_model_KeyBERTInspired.get_topic_info()


# In[51]:


topic_model_KeyBERTInspired.visualize_barchart()


# # BERTopic (MMR)

# In[ ]:


#representation_model = MaximalMarginalRelevance(diversity=0.3)

#topic_model = BERTopic(representation_model=representation_model,language="french")
#topics, probs = topic_model.fit_transform(docs_cleaned)


# In[ ]:


#topic_model.get_topic_info()


#  # Hierarchical topics 

# In[ ]:


#hierarchical_topics = topic_model.hierarchical_topics(docs_cleaned)
#hierarchical_topics


# In[ ]:


#linkage_function = lambda x: sch.linkage(x, 'single', optimal_ordering=True)
#hierarchical_topics = topic_model.hierarchical_topics(docs_cleaned, linkage_function=linkage_function)


# In[ ]:


#topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)

