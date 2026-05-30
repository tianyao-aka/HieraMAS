from sentence_transformers import SentenceTransformer
import torch

_embedding_model = None

def get_sentence_embedding(sentence):
    """Get sentence embedding using gte-large-en-v1.5 (1024-dim output)."""
    global _embedding_model

    if _embedding_model is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _embedding_model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', device=device, trust_remote_code=True)

    return _embedding_model.encode(sentence, convert_to_numpy=True, show_progress_bar=False)
