
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pysd
import pandas as pd
import os
import shutil
import tempfile
import uuid

app = FastAPI(title="PySD REST API", version="2.0.0")

# Store multiple models by modelId
models: Dict[str, Any] = {}

# ----------------- Schemas -----------------

class LoadModelRequest(BaseModel):
    path: str
    fileType: str  # 'vensim' or 'xmile'
    modelId: Optional[str] = None


class RunModelRequest(BaseModel):
    modelId: str
    params: Optional[Dict[str, float]] = None
    returnColumns: Optional[List[str]] = None


class SetParametersRequest(BaseModel):
    modelId: str
    parameters: Dict[str, float]


class ResetModelRequest(BaseModel):
    modelId: str


# ----------------- Helpers -----------------

def get_model(model_id: str):
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return models[model_id]


# ----------------- Endpoints -----------------

@app.post("/model/upload")
def upload_model(file: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(file.filename)[1].lower()

        if suffix not in [".mdl", ".xmile"]:
            raise HTTPException(status_code=400, detail="Only .mdl and .xmile files are supported")

        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Load model and assign unique ID
        if suffix == ".mdl":
            engine = pysd.read_vensim(file_path)
        else:
            engine = pysd.read_xmile(file_path)

        model_id = str(uuid.uuid4())
        models[model_id] = engine

        return {"message": "Model uploaded successfully", "modelId": model_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/load")
def load_model(request: LoadModelRequest):
    try:
        model_id = request.modelId or str(uuid.uuid4())

        if request.fileType == "vensim":
            engine = pysd.read_vensim(request.path)
        elif request.fileType == "xmile":
            engine = pysd.read_xmile(request.path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported fileType")

        models[model_id] = engine
        return {"message": "Model loaded successfully", "modelId": model_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/model/run")
def run_model(request: RunModelRequest):
    engine = get_model(request.modelId)
    try:
        results: pd.DataFrame = engine.run(
            params=request.params or {},
            return_columns=request.returnColumns
        )
        if isinstance(results, pd.Series):
            results = results.to_frame()

        return {col: results[col].tolist() for col in results.columns}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/components/{model_id}")
def get_components(model_id: str):
    engine = get_model(model_id)
    try:
        structure = engine.components
        return {
            "stocks": structure.stocks,
            "flows": structure.flows,
            "auxiliaries": structure.auxiliaries,
            "constants": structure.constants,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/parameters")
def set_parameters(request: SetParametersRequest):
    engine = get_model(request.modelId)
    try:
        engine.set_components(request.parameters)
        return {"message": "Parameters set successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/reset")
def reset_model(request: ResetModelRequest):
    if request.modelId in models:
        del models[request.modelId]
        return {"message": f"Model '{request.modelId}' removed from memory"}
    raise HTTPException(status_code=404, detail="Model ID not found")


@app.get("/model/list")
def list_models():
    return {"models": list(models.keys())}

