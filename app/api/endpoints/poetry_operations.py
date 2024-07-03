from app.database.common import (
    get_poetry_Text2SQL_engine,
    logger,
    keyword_match,
    get_file_engine,
)
import json
import re
import random
import string
from app.api.config import base_url, file_formats, pairs_base_url
import asyncio
import httpx
import os


# 关闭多余的日志输出
import logging

logging.getLogger("httpcore").propagate = False
logging.getLogger("sse_starlette").propagate = False
logging.getLogger("httpx").propagate = False
logging.getLogger("multipart").propagate = False
logging.getLogger("openai").propagate = False


client = httpx.AsyncClient(timeout=httpx.Timeout(480.0, read=480.0))

# 生成随机客户端id
characters = string.ascii_letters + string.digits


# 自定义检索操作类
class PoetryOperations:
    def __init__(self, streaming: bool = True) -> None:
        self._streaming = streaming
        self._poetry_query_engine = get_poetry_Text2SQL_engine(self._streaming)
        if self._streaming:
            logger.debug("流式回复已开启")
        else:
            logger.warning("流式回复已关闭")
        logger.debug("诗词检索引擎初始化完成")

    # 诗词对话
    async def text2sql_query(self, query: str):
        response = self._poetry_query_engine.query(query)
        logger.debug(f"检索结果：{response.metadata}")
        return (
            self._poetry_gen(gen=response.response_gen)
            if self._streaming
            else self._clean_string(input_str=str(response))
        )

    # 构建流式回复生成器
    def _poetry_gen(self, gen):
        log_msg = ""
        for text in gen:
            log_msg += f"{text}".strip("\n\n")
            yield json.dumps({"content": text})
        logger.debug(f"回复内容：{log_msg}")

    # 知识库对话
    async def file_query(self, query: str, file_name, file_content):
        data_pd = await self._correct_file(
            file_name=file_name, file_content=file_content
        )
        if len(data_pd) == 0:
            logger.debug(f"知识库创建失败")
            return 0

        query_engine = get_file_engine(streaming=self._streaming, df=data_pd)
        logger.debug(f"知识库创建完毕")
        response = query_engine.query(query)

        return (
            self._poetry_gen(gen=response.response_gen)
            if self._streaming
            else self._clean_string(input_str=str(response))
        )

    # 构建文件读取函数
    async def _correct_file(self, file_name, file_content):
        try:
            ext = os.path.splitext(file_name)[1][1:].lower()
        except:
            ext = str(file_name).split(".")[-1]

        if ext in file_formats:
            return file_formats[ext](file_content)

        else:
            return ""

    # 诗词生成
    async def generate_poetry(self, type, format, emotion, prompt):
        if type == "绝句":
            params = {
                "yan": format,
                "poem": prompt,
                "user_id": "".join(random.choice(characters) for _ in range(30)),
            }
            send_url = base_url + "send_jueju"
            get_url = base_url + "get_jueju"

        if type == "风格绝句":
            params = {
                "yan": format,
                "poem": prompt,
                "user_id": "".join(random.choice(characters) for _ in range(30)),
                "style": emotion,
            }
            send_url = base_url + "send_juejustyle"
            get_url = base_url + "get_juejustyle"

        if type == "藏头诗":
            params = {
                "yan": format,
                "poem": prompt,
                "user_id": "".join(random.choice(characters) for _ in range(30)),
                "sentiment": emotion,
            }
            send_url = base_url + "send_arousic"
            get_url = base_url + "get_arousic"

        if type == "律诗":
            params = {
                "yan": format,
                "poem": prompt,
                "user_id": "".join(random.choice(characters) for _ in range(30)),
            }
            send_url = base_url + "send_lvshi"
            get_url = base_url + "get_lvshi"

        if type == "词":
            params = {
                "poem": prompt,
                "cipai": format,
                "user_id": "".join(random.choice(characters) for _ in range(30)),
            }
            send_url = base_url + "send_songci"
            get_url = base_url + "get_songci"

        ans = await self._get_generate_result(
            send_url=send_url, params=params, get_url=get_url
        )
        return self._generate_stream(ans)

    # 获取诗词和对对子生成结果
    async def _get_generate_result(
        self, send_url, get_url, params=None, form_data=None
    ):
        if form_data:
            send_response = await client.post(send_url, data=form_data)
        else:
            send_response = await client.post(send_url, json=params)

        if send_response.status_code == 200:
            celery_id_res = send_response.json()["celery_id"]
            logger.debug(f"诗词任务id:{celery_id_res}")

        # 轮询接口查看诗词生成或者对对子任务是否完成
        while True:
            # 临时区分对对子任务和诗词任务的轮询请求方式
            get_response = (
                await client.post(get_url, data={"celery_id": celery_id_res})
                if form_data
                else await client.post(get_url, json={"celery_id": celery_id_res})
            )
            if get_response.status_code == 200:
                if get_response.json()["status"] != "PENDING":
                    logger.debug(f"诗词任务完成")
                    logger.debug(f"诗词任务结果：{get_response.json()}")
                    return get_response.json()
                else:
                    logger.debug(f"诗词任务未完成，继续轮询")
                    await asyncio.sleep(1)

    # 构建诗词生成回复生成器
    def _generate_stream(self, response):
        yield json.dumps({"content": "诗词生成结果如下：" + "\n\n"})
        yield json.dumps({"content": f"《{response['title']}》" + "\n\n"})

        # 判断列表是否嵌套列表
        if isinstance(response["output"][0], str):
            for i in response["output"]:
                yield json.dumps({"content": i + "\n\n"})
        else:
            for i in response["output"][0]:
                yield json.dumps({"content": i + "\n\n"})

    # 对对子函数
    async def pairs_operate(self, prompt, predict_type):
        params = {
            "inputs": prompt,
            "predict_lower": (
                "predict_lower" if predict_type == "lower" else "predict_upper"
            ),
        }
        print(params)
        send_url = pairs_base_url + "getpoem"
        get_url = pairs_base_url + "sendpoem"

        ans = await self._get_generate_result(
            send_url=send_url, get_url=get_url, form_data=params
        )
        return self._pairs_stream(ans, predict_type=predict_type)

    # 构建对对子流式回复生成器
    def _pairs_stream(self, response, predict_type):
        yield json.dumps({"content": "对对子生成结果如下：" + "\n\n"})
        if predict_type == "lower":
            yield json.dumps({"content": f"上联：{response['source']}" + "\n\n"})
            yield json.dumps({"content": f"下联：{response['couplet'][0]}" + "\n\n"})
        else:
            yield json.dumps({"content": f"上联：{response['couplet'][0]}" + "\n\n"})
            yield json.dumps({"content": f"下联：{response['source']}" + "\n\n"})

    # 清理字符串，移除换行符和多余字符等
    def _clean_string(self, input_str: str) -> str:

        return re.sub(r"[\n\t\\n\\t()]|None", "", str(input_str))
