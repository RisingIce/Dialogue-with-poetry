from fastapi import APIRouter, UploadFile, File
from sse_starlette.sse import EventSourceResponse
from app.database.schame import (
    RequestPoetry,
    ResponsePoetry,
    RequestGenerate,
    RequestPairs,
)
from app.api.endpoints.poetry_operations import PoetryOperations
from app.database.common import logger
from app.api.config import streaming
import pandas as pd
import io

poetry = APIRouter()
poetry_operations = PoetryOperations(streaming=streaming)


# 诗词对话接口
@poetry.post("/dialogue", response_model=ResponsePoetry)
async def poetry_dialogue(req: RequestPoetry):
    logger.debug(f"请求参数：{req.json()}")
    ans = await poetry_operations.text2sql_query(req.query)
    return (
        EventSourceResponse(ans, media_type="text/event-stream")
        if streaming
        else ResponsePoetry(response=ans)
    )


# 知识库对话接口
@poetry.post("/fileQuery", response_model=ResponsePoetry)
async def file_query(query: str, file: UploadFile = File(...)):
    print(file)
    logger.debug(f"请求参数：{query}")
    file_bytes = await file.read()
    file_content = io.BytesIO(file_bytes)
    logger.debug(
        f"文件名称：{file.filename} | 文件大小：{file.size} | 内容类型：{file.content_type}"
    )
    ans = await poetry_operations.file_query(
        query=query, file_name=file.filename, file_content=file_content
    )
    if ans == 0:
        return ResponsePoetry(response="不支持的文件格式！")
    return (
        EventSourceResponse(ans, media_type="text/event-stream")
        if streaming
        else ResponsePoetry(response=ans)
    )


# 诗词生成接口
@poetry.post("/generate", response_model=ResponsePoetry)
async def poetry_generate(req: RequestGenerate):
    logger.debug(f"请求参数：{req.json()}")
    if req.type not in ["绝句", "风格绝句", "藏头诗", "律诗", "词"]:
        return ResponsePoetry(response="不支持的生成类型")
    ans = await poetry_operations.generate_poetry(
        type=req.type, format=req.format, emotion=req.emotion_type, prompt=req.prompt
    )
    return EventSourceResponse(ans, media_type="text/event-stream")


# 对对子接口
@poetry.post("/pairs", response_model=ResponsePoetry)
async def poetry_pairs(req: RequestPairs):
    logger.debug(f"请求参数：{req.json()}")
    ans = await poetry_operations.pairs_operate(
        prompt=req.prompt, predict_type=req.predict_type
    )
    return EventSourceResponse(ans, media_type="text/event-stream")
