"""
Script d'ingestion et de prétraitement des données PDF
- Extraction des PDFs → texte
- Nettoyage et segmentation en chunks
- Création des embeddings avec sentence-transformers
- Construction de l'index FAISS pour la recherche vectorielle
"""

import os
import re
from pathlib import Path
from typing import List

import faiss
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


#---------------- CONFIGURATION ----------------------------
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
FAISS_INDEX_DIR = PROCESSED_DATA_DIR / "faiss_index"

# Paramètres de segmentation
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Modèle d'embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Modèle léger et rapide
# Alternatives: "sentence-transformers/all-mpnet-base-v2" (plus performant mais plus lourd)


# ----------------------INGESTION ET PRÉTRAITEMENT ----------------------------
def load_pdf_documents(pdf_dir: Path) -> List[Document]:
    """
    Charge tous les fichiers PDF du répertoire spécifié.
    
    Args:
        pdf_dir: Répertoire contenant les fichiers PDF
        
    Returns:
        Liste de documents LangChain
    """
    documents = []
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"Aucun fichier PDF trouvé dans {pdf_dir}")
        return documents
    
    print(f"Chargement de {len(pdf_files)} fichier(s) PDF...")
    
    for pdf_file in pdf_files:
        print(f"  - Traitement de: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            # Ajouter le nom du fichier source dans les métadonnées
            for doc in docs:
                doc.metadata["source"] = pdf_file.name
                doc.metadata["file_path"] = str(pdf_file)
            documents.extend(docs)
        except Exception as e:
            print(f"  ⚠️  Erreur lors du chargement de {pdf_file.name}: {e}")
    
    print(f"✓ {len(documents)} page(s) chargée(s) au total\n")
    return documents


# ------------------ EXTRACTION DES PDFS → TEXTE---------------------------
def extract_text_from_documents(documents: List[Document]) -> List[Document]:
    """
    Extrait et nettoie le texte des documents.
    Le texte est déjà extrait par PyPDFLoader, cette fonction nettoie le texte.
    
    Args:
        documents: Liste de documents bruts
        
    Returns:
        Liste de documents avec texte nettoyé
    """
    print("Nettoyage du texte extrait...")
    
    cleaned_docs = []
    for doc in documents:
        # Nettoyage basique du texte
        text = doc.page_content
        
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        # Supprimer les caractères de contrôle
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Nettoyer les sauts de ligne multiples
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Mettre à jour le contenu du document
        doc.page_content = text.strip()
        cleaned_docs.append(doc)
    
    print(f"✓ {len(cleaned_docs)} document(s) nettoyé(s)\n")
    return cleaned_docs


# ----------------------- NETTOYAGE ET SEGMENTATION EN CHUNKS -------------------
def split_documents_into_chunks(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:
    """
    Segmente les documents en chunks de taille fixe avec overlap.
    
    Args:
        documents: Liste de documents à segmenter
        chunk_size: Taille maximale d'un chunk (en caractères)
        chunk_overlap: Nombre de caractères de chevauchement entre chunks
        
    Returns:
        Liste de chunks de documents
    """
    print(f"Segmentation en chunks (taille: {chunk_size}, overlap: {chunk_overlap})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Ajouter un index de chunk dans les métadonnées
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    
    print(f"✓ {len(chunks)} chunk(s) créé(s) à partir de {len(documents)} document(s)\n")
    return chunks


#------------------------- CRÉATION DES EMBEDDINGS--------------------------
def create_embeddings(chunks: List[Document], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    """
    Crée les embeddings pour tous les chunks en utilisant sentence-transformers.
    
    Args:
        chunks: Liste de chunks de documents
        model_name: Nom du modèle sentence-transformers à utiliser
        
    Returns:
        Matrice numpy des embeddings (n_chunks, embedding_dim)
    """
    print(f"Chargement du modèle d'embeddings: {model_name}...")
    model = SentenceTransformer(model_name)
    
    print(f"Création des embeddings pour {len(chunks)} chunk(s)...")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    print(f"✓ Embeddings créés: shape {embeddings.shape}\n")
    return embeddings, model


# ------------------------ CONSTRUCTION DE L'INDEX FAISS ----------------
def build_faiss_index(embeddings: np.ndarray, chunks: List[Document]) -> faiss.Index:
    """
    Construit l'index FAISS pour la recherche vectorielle.
    
    Args:
        embeddings: Matrice numpy des embeddings
        chunks: Liste des chunks correspondants
        
    Returns:
        Index FAISS
    """
    print("Construction de l'index FAISS...")
    
    dimension = embeddings.shape[1]
    
    # Normaliser les embeddings pour une meilleure performance avec la similarité cosinus
    faiss.normalize_L2(embeddings)
    
    # Créer l'index FAISS (IndexFlatIP pour produit scalaire interne sur vecteurs normalisés = cosinus)
    index = faiss.IndexFlatIP(dimension)
    
    # Ajouter les embeddings à l'index
    index.add(embeddings.astype('float32'))
    
    print(f"✓ Index FAISS créé avec {index.ntotal} vecteur(s) de dimension {dimension}\n")
    return index


def save_faiss_index(index: faiss.Index, chunks: List[Document], model, output_dir: Path):
    """
    Sauvegarde l'index FAISS et les métadonnées associées.
    
    Args:
        index: Index FAISS
        chunks: Liste des chunks
        model: Modèle sentence-transformers utilisé
        output_dir: Répertoire de sortie
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Sauvegarde de l'index FAISS dans {output_dir}...")
    
    # Sauvegarder l'index FAISS
    faiss.write_index(index, str(output_dir / "index.faiss"))
    
    # Sauvegarder les métadonnées des chunks (pour récupérer les documents originaux)
    import pickle
    metadata = {
        "chunks": chunks,
        "model_name": EMBEDDING_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP
    }
    
    with open(output_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"✓ Index et métadonnées sauvegardés\n")


def load_faiss_index(index_dir: Path):
    """
    Charge l'index FAISS et les métadonnées depuis le disque.
    
    Args:
        index_dir: Répertoire contenant l'index
        
    Returns:
        Tuple (index, metadata)
    """
    import pickle
    
    index_path = index_dir / "index.faiss"
    metadata_path = index_dir / "metadata.pkl"
    
    if not (index_path.exists() and metadata_path.exists()):
        return None, None
    
    print(f"Chargement de l'index FAISS depuis {index_dir}...")
    index = faiss.read_index(str(index_path))
    
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    print(f"✓ Index chargé avec {index.ntotal} vecteur(s)\n")
    return index, metadata


# PIPELINE PRINCIPAL
def main():
    """
    Pipeline complet d'ingestion et de traitement des données.
    """
    print("=" * 60)
    print("PIPELINE D'INGESTION ET DE PRÉTRAITEMENT")
    print("=" * 60)
    print()
    
    # Vérifier si l'index existe déjà
    index, metadata = load_faiss_index(FAISS_INDEX_DIR)
    
    if index is not None:
        print("Index FAISS existant trouvé. Utilisation de l'index existant.")
        print(f"Nombre de vecteurs: {index.ntotal}")
        return index, metadata
    
    # ÉTAPE 1: Ingestion
    documents = load_pdf_documents(RAW_DATA_DIR)
    
    if not documents:
        print("Aucun document à traiter. Arrêt du pipeline.")
        return None, None
    
    # ÉTAPE 2: Extraction et nettoyage du texte
    cleaned_documents = extract_text_from_documents(documents)
    
    # ÉTAPE 3: Segmentation en chunks
    chunks = split_documents_into_chunks(
        cleaned_documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # ÉTAPE 4: Création des embeddings
    embeddings, model = create_embeddings(chunks, EMBEDDING_MODEL_NAME)
    
    # ÉTAPE 5: Construction de l'index FAISS
    index = build_faiss_index(embeddings, chunks)
    
    # Sauvegarde
    save_faiss_index(index, chunks, model, FAISS_INDEX_DIR)
    
    print("=" * 60)
    print("PIPELINE TERMINÉ AVEC SUCCÈS")
    print("=" * 60)
    
    return index, {"chunks": chunks, "model": model}


if __name__ == "__main__":
    index, metadata = main()
