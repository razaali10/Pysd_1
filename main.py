
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pysd import read_vensim
import uuid
import os
import pandas as pd

app = FastAPI()

# ✅ Mount /static so files like openapi.yaml can be accessed
app.mount("/static", StaticFiles(directory="static"), name="static")

models = {}

@app.post("/model/upload")
async def upload_model(file: UploadFile = File(...)):
    ext = file.filename.split('.')[-1]
    if ext not in ("mdl", "xmile"):
        raise HTTPException(status_code=400, detail="Unsupported file format")
    model_id = str(uuid.uuid4())
    filepath = f"/tmp/{model_id}.{ext}"
    with open(filepath, "wb") as f:
        f.write(await file.read())
    try:
        models[model_id] = read_vensim(filepath)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"modelId": model_id, "message": f"Model uploaded successfully."}

@app.post("/model/run")
async def run_model(modelId: str, params: dict = {}, returnColumns: list = []):
    if modelId not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    model = models[modelId]
    if params:
        model.set_components(params)
    result_df = model.run()
    if returnColumns:
        result_df = result_df[returnColumns]
    return JSONResponse(content=result_df.to_dict(orient="list"))

# Optional: Redirect /openapi.yaml directly (for GPT plugin discovery)
@app.get("/openapi.yaml", include_in_schema=False)
async def get_openapi_spec():
    return FileResponse("static/openapi.yaml", media_type="text/yaml")
