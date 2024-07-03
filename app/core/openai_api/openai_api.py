from fastapi import FastAPI,HTTPException
from config import poetry_url
import requests
import uvicorn
from sse_starlette.sse import EventSourceResponse
import json
from pydantic import BaseModel,Field
from typing import Optional,List,Literal,Union
import time


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class FunctionCallResponse(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: str = None
    name: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    functions: Optional[Union[dict, List[dict]]] = None
    # Additional parameters
    repetition_penalty: Optional[float] = 1.1


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "function_call"]]


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


app = FastAPI()


@app.post('/v1/chat/completions', response_model=ChatCompletionResponse)
async def openai_poetry(request: ChatCompletionRequest):
    url = poetry_url
    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        repetition_penalty=request.repetition_penalty,
        functions=request.functions,
    )
    usage = UsageInfo()
    function_call, finish_reason = None, "stop"

    # 构建rasa请求参数
    params = {
        "query": dict(gen_params['messages'][-1])['content'],
    }
    # 请求接口，获取输出结果
    resp = requests.post(url=url, json=params, stream=True)
     # 是否启用流式传输
    if request.stream:
        # print(1)
        # Use the stream mode to read the first few characters, if it is not a function call, direct stram output
        predict_stream_generator = predict_stream(request.model, gen_params, resp)
        output = next(predict_stream_generator)
        if not contains_custom_function(output):
            return EventSourceResponse(predict_stream_generator, media_type="text/event-stream")


def contains_custom_function(value: str) -> bool:
    """
    Determine whether 'function_call' according to a special function prefix.

    For example, the functions defined in "tool_using/tool_register.py" are all "get_xxx" and start with "get_"

    [Note] This is not a rigorous judgment method, only for reference.

    :param value:
    :return:
    """
    return value and 'get_' in value



def predict_stream(model_id, gen_params,resp):
    """
    The function call is compatible with stream mode output.

    The first seven characters are determined.
    If not a function call, the stream output is directly generated.
    Otherwise, the complete character content of the function call is returned.

    :param model_id:
    :param gen_params:
    :return:
    """
    output = ""
    is_function_call = False
    has_send_first_chunk = False
    for new_response in generate_stream_poetry(resp):
        decoded_unicode = new_response["text"]
        # print("du:",decoded_unicode)
        delta_text = decoded_unicode[len(output):]
        output = decoded_unicode

        # When it is not a function call and the character length is> 7,
        # try to judge whether it is a function call according to the special function prefix
        if not is_function_call and len(output) >=0:

            # Determine whether a function is called
            is_function_call = contains_custom_function(output)
            if is_function_call:
                continue

            # Non-function call, direct stream output
            finish_reason = new_response["finish_reason"]

            # Send an empty string first to avoid truncation by subsequent next() operations.
            if not has_send_first_chunk:

                message = DeltaMessage(
                    content="",
                    role="assistant",
                    function_call=None,
                )

                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=message,
                    finish_reason=finish_reason
                )

                chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")

                yield "{}".format(chunk.model_dump_json(exclude_unset=True))

            #send_msg = delta_text if has_send_first_chunk else output
            send_msg = output
            has_send_first_chunk = True
            message = DeltaMessage(
                content=send_msg,
                role="assistant",
                function_call=None,
            )

            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=message,
                finish_reason=finish_reason
            )

            chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")

            yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    if is_function_call:
        yield output
    else:
        yield '[DONE]'

def generate_stream_poetry(resp):
    for line in resp.iter_lines(chunk_size=1024):
        if line:
            decoded_line = line.decode('utf-8')
            res = decoded_line.replace('data:', '')
            final = dict(json.loads(res))
            ret = {
                "text":final["content"],
                "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 100,
                            "total_tokens": 100,
                            },
                "finish_reason": "function_call",
                }



            time.sleep(0.02)
            yield ret




if __name__ == '__main__':

    uvicorn.run("openai_api:app", host='0.0.0.0', port=8003)
