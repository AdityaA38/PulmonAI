from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import model_backend as model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    result = model.predict(contents)
    return result
