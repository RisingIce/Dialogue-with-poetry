from fastapi import APIRouter,Request
from app.api.endpoints.gsv_operations import GSVOperations

default_refer_path = 'app\core\SoVITS_weights\samples\说话-这就是艾利欧所预见的以及你将抵达的未来…喜欢么.wav'
default_refer_text = '说话-这就是艾利欧所预见的以及你将抵达的未来喜欢么'
default_refer_language = 'zh'

gsvOperations = GSVOperations(default_refer_path=default_refer_path,default_refer_text=default_refer_text,default_refer_language=default_refer_language)

gsv = APIRouter()
@gsv.post('/')
async def tts_endpoint(request: Request):
    json_post_raw = await request.json()
    return gsvOperations.handle(
        json_post_raw.get("refer_wav_path"),
        json_post_raw.get("prompt_text"),
        json_post_raw.get("prompt_language"),
        json_post_raw.get("text"),
        json_post_raw.get("text_language"),
    )

