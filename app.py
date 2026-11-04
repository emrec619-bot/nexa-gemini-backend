import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai

app = FastAPI(title="NEXA AI Studio API", version="0.1.0")

origins = [x.strip() for x in os.getenv("FRONTEND_ORIGIN", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ["*"] else ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

class AnalyzeRequest(BaseModel):
    idea: str = Field(min_length=3, max_length=12000)

@app.get("/api/health")
def health():
    return {"ok": True, "service": "nexa-gemini-backend"}

@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Sunucuda GEMINI_API_KEY ayarlanmamış.")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    prompt = f"""Sen NEXA AI Studio'nun AI Proje Yöneticisisin. Kullanıcının fikrini Türkçe ve uygulanabilir biçimde analiz et.

Şu başlıkları kullan:
1. Projenin amacı ve çözdüğü sorun
2. Hedef kullanıcılar
3. İlk sürüm için temel özellikler (MVP)
4. Ekranlar ve kullanıcı akışı
5. Önerilen teknik mimari ve veri modeli
6. Güvenlik, gizlilik ve olası riskler
7. Geliştirme aşamaları ve test planı
8. Kullanıcıdan netleştirilmesi gereken sorular

Gerçekleştirilmemiş işleri tamamlanmış gibi gösterme. Belirsiz noktaları açıkça belirt. Ücretli işlem, yayınlama veya dış sistemde değişiklik yapma önerisini kullanıcı onayı gerektiren bir adım olarak işaretle.

Kullanıcı fikri:
{request.idea}"""
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=prompt)
        answer = getattr(response, "text", None)
        if not answer:
            raise ValueError("Model boş yanıt döndürdü.")
        return {"ok": True, "model": model, "answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini isteği başarısız: {str(exc)[:500]}")
