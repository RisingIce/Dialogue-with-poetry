from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import ResponseValidationError, RequestValidationError
from fastapi.responses import JSONResponse
import uvicorn
from app.api.poetry_api import poetry
from app.api.gsv_api import gsv
from app.api.config import api_port,sovits_path,gpt_path
# import argparse

# parser = argparse.ArgumentParser(description="GPT-SoVITS api")
# parser.add_argument("-s", "--sovits_path", type=str, default=sovits_path, help="SoVITS模型路径")
# parser.add_argument("-g", "--gpt_path", type=str, default=gpt_path, help="GPT模型路径")

app = FastAPI(docs_url=None)
# Instrumentator().instrument(app).expose(app)
app.include_router(poetry)
app.include_router(gsv)

@app.exception_handler(ResponseValidationError)
async def validation_exception_handler(request: Request, exc: ResponseValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors()}),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=api_port)
