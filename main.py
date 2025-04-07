
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pysd
import pandas as pd
import os
import shutil
import tempfile

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

        # Load model using pysd
        if suffix == ".mdl":
            model = pysd.read_vensim(file_path)
        else:
            model = pysd.read_xmile(file_path)

        # Generate and store a model_id
        model_id = os.path.splitext(file.filename)[0].lower().replace(" ", "_")
        models[model_id] = model

        if not model_id or model_id not in models:
            raise HTTPException(status_code=500, detail="Failed to register model.")

        return {"model_id": model_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model upload failed: {str(e)}")

@app.get("/model")
def list_models():
    return {"available_models": list(models.keys())}


@app.post("/model/simulate")
def simulate_model(request: RunModelRequest):
    model = get_model(request.modelId)

    try:
        result = model.run(
            params=request.params or {},
            return_columns=request.returnColumns,
        )
        return result.to_dict(orient="list")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@app.get("/model/plot/{model_id}")
def plot_model_output(model_id: str):
    import matplotlib.pyplot as plt
    import io
    import base64

    model = get_model(model_id)

    try:
        result = model.run()
        fig, ax = plt.subplots(figsize=(10, 5))
        for col in result.columns[:5]:  # limit to first 5 variables
            ax.plot(result.index, result[col], label=col)
        ax.set_title(f"Simulation Output for {model_id}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        # Convert plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_bytes = buf.getvalue()
        base64_img = base64.b64encode(img_bytes).decode("utf-8")
        buf.close()

        return {"plot_base64": base64_img}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plotting failed: {str(e)}")


from fastapi.responses import JSONResponse
import matplotlib.pyplot as plt
import base64
from io import BytesIO

@app.post("/model/simulate")
def simulate_model(req: RunModelRequest):
    try:
        model = get_model(req.modelId)
        result = model.run(
            params=req.params or {},
            return_columns=req.returnColumns,
        )
        return result.to_dict(orient="list")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@app.post("/model/plot")
def plot_model_outputs(req: RunModelRequest):
    try:
        model = get_model(req.modelId)
        result = model.run(
            params=req.params or {},
            return_columns=req.returnColumns,
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        for col in result.columns:
            ax.plot(result.index, result[col], label=col)
        ax.set_title("Simulation Output")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        # Convert to base64 image
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return JSONResponse(content={"image_base64": f"data:image/png;base64,{image_base64}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plot generation failed: {str(e)}")


import networkx as nx
from fastapi.responses import StreamingResponse
import csv

@app.get("/model/sfd/{model_id}")
def get_stock_flow_diagram(model_id: str):
    try:
        model = get_model(model_id)
        structure = model.components._namespace

        G = nx.DiGraph()

        for key, val in structure.items():
            G.add_node(key)
            if hasattr(val, "func") and hasattr(val.func, "__code__"):
                code = val.func.__code__
                dependencies = code.co_names
                for dep in dependencies:
                    if dep in structure:
                        G.add_edge(dep, key)

        # Plotting using Graphviz layout
        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, arrows=True, node_size=2000, node_color="lightblue", font_size=10)
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return JSONResponse(content={"image_base64": f"data:image/png;base64,{image_base64}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate SFD: {str(e)}")


@app.post("/model/simulate/export")
def simulate_and_export(req: RunModelRequest):
    try:
        model = get_model(req.modelId)
        result = model.run(
            params=req.params or {},
            return_columns=req.returnColumns,
        )

        buf = BytesIO()
        result.to_csv(buf)
        buf.seek(0)

        return StreamingResponse(buf, media_type="text/csv", headers={
            "Content-Disposition": f"attachment; filename={req.modelId}_simulation.csv"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation export failed: {str(e)}")


from fastapi.responses import StreamingResponse
import networkx as nx

@app.get("/model/sfd/{model_id}")
def generate_structure_diagram(model_id: str):
    try:
        model = get_model(model_id)
        structure = model.components  # internal structure object

        G = nx.DiGraph()

        for key, val in structure.items():
            if hasattr(val, 'dependencies'):
                for dep in val.dependencies:
                    G.add_edge(dep, key)

        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2500, font_size=10)
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return JSONResponse(content={"image_base64": f"data:image/png;base64,{image_base64}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SFD generation failed: {str(e)}")


@app.post("/model/simulate/export")
def simulate_and_export(req: RunModelRequest):
    try:
        model = get_model(req.modelId)
        result = model.run(
            params=req.params or {},
            return_columns=req.returnColumns,
        )
        buf = BytesIO()
        result.to_csv(buf)
        buf.seek(0)
        return StreamingResponse(buf, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=simulation_output.csv"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
