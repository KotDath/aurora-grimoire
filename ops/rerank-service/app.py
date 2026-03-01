from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]
    top_n: int = 30
    return_documents: bool = False


app = FastAPI(title="aurora-grimoire-rerank")
_model = None
_loaded_model_name = None


def get_model(model_name: str) -> CrossEncoder:
    global _model, _loaded_model_name
    if _model is None or _loaded_model_name != model_name:
        _model = CrossEncoder(model_name)
        _loaded_model_name = model_name
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "model": _loaded_model_name}


@app.post("/v1/rerank")
def rerank(req: RerankRequest):
    if not req.documents:
        return {"results": []}

    model = get_model(req.model)
    pairs = [(req.query, doc) for doc in req.documents]
    scores = model.predict(pairs).tolist()
    top_n = max(1, min(req.top_n, len(scores)))

    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_n]
    return {
        "results": [
            {"index": int(index), "relevance_score": float(score)} for index, score in ranked
        ]
    }

