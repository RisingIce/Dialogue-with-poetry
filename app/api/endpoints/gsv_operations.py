from app.database.common import logger
from app.api.config import infer_device,api_port,bind_addr,cnhubert_path,bert_path,dict_language,sovits_path,gpt_path
from app.core.GPT_SoVITS.feature_extractor import cnhubert
from transformers import AutoModelForMaskedLM, AutoTokenizer
from app.core.GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
import torch
from time import time as ttime
import librosa
import soundfile as sf
from fastapi.responses import StreamingResponse, JSONResponse
import numpy as np
from io import BytesIO
from app.core.GPT_SoVITS.module.models import SynthesizerTrn
from app.core.GPT_SoVITS.text import cleaned_text_to_sequence
from app.core.GPT_SoVITS.text.cleaner import clean_text
from app.core.GPT_SoVITS.module.mel_processing import spectrogram_torch
from app.core.GPT_SoVITS.my_utils import load_audio
import os,sys

sys.path.append("app\core\GPT_SoVITS")

#自定义gsv操作类，整合tts操作
class GSVOperations:
    def __init__(self,default_refer_path,default_refer_text,default_refer_language,full_precision:bool = False,half_precision:bool = False) -> None:
        self._sovits_path = sovits_path
        self._gpt_path = gpt_path
        self._device = infer_device
        self._api_port = api_port
        self._bind_addr = bind_addr
        self._default_refer_path = default_refer_path
        self._default_refer_text = default_refer_text
        self._default_refer_language = default_refer_language
        self._full_precision = full_precision
        self._half_precision = half_precision
        self._cnhubert_base_path = cnhubert_path
        self._bert_path = bert_path
        self._dict_language = dict_language
        self._n_semantic = 1024
        self._dict_s2 = torch.load(self._sovits_path, map_location="cpu")
        self._hps = DictToAttrRecursive(self._dict_s2["config"])
        self._dict_s1 = torch.load(self._gpt_path, map_location="cpu")
        self._config = self._dict_s1["config"]
        self._hz = 50
        self._max_sec = self._config['data']['max_sec']
        if self._default_refer_text == None or self._default_refer_path == None or self._default_refer_language == None: 
           self._default_refer_path, self._default_refer_text, self._default_refer_language  = "", "", ""
           logger.debug("未指定默认参考音频")
        else:
           logger.debug(f"默认参考音频路径: {self._default_refer_path}")
           logger.debug(f"默认参考音频文本: {self._default_refer_text}")
           logger.debug(f"默认参考音频语种: {self._default_refer_language}")

        if self._sovits_path == "" or self._gpt_path == "":
            logger.debug("未指定sovits模型路径或gpt模型路径")
            raise ValueError("未指定sovits模型路径或gpt模型路径")
        if self._cnhubert_base_path =="" or self._bert_path == "":
            logger.debug("未指定bert模型路径")
            raise ValueError("未指定bert模型路径")
        else:
            logger.debug(f"sovits模型路径: {self._sovits_path}")
            logger.debug(f"gpt模型路径: {self._gpt_path}")
            logger.debug(f"cnbert模型路径: {self._cnhubert_base_path}")
            logger.debug(f"bert模型路径: {self._bert_path}")
        
        cnhubert.cnhubert_base_path = self._cnhubert_base_path
        self._hps.model.semantic_frame_rate = "25hz"
        self._tokenizer = AutoTokenizer.from_pretrained(self._bert_path)
        self._bert_model = AutoModelForMaskedLM.from_pretrained(self._bert_path)
        self._ssl_model = cnhubert.get_model()
        self._vq_model = SynthesizerTrn(
                    self._hps.data.filter_length // 2 + 1,
                    self._hps.train.segment_size // self._hps.data.hop_length,
                    n_speakers=self._hps.data.n_speakers,
                    **self._hps.model)
        self._t2s_model = Text2SemanticLightningModule(self._config, "****", is_train=False)
        self._t2s_model.load_state_dict(self._dict_s1["weight"])
        if self._full_precision:
            logger.debug(f"当前开启使用全精度推理,full_precision:{self._full_precision}")
        elif self._half_precision:
            self._bert_model = self._bert_model.half().to(self._device)
            self._ssl_model = self._ssl_model.half().to(self._device)
            self._vq_model = self._vq_model.half().to(self._device)
            self._t2s_model = self._t2s_model.half()
            logger.debug(f"当前开启使用半精度推理,half_precision:{self._half_precision}")
        else:
            self._bert_model = self._bert_model.to(self._device)
            self._ssl_model = self._ssl_model.to(self._device)
            self._vq_model = self._vq_model.to(self._device)
            self._t2s_model = self._t2s_model.to(self._device)
            logger.debug(f"当前关闭使用精度推理,full_precision:{self._full_precision},half_precision:{self._half_precision}")

        self._vq_model.eval()
        logger.debug(self._vq_model.load_state_dict(self._dict_s2["weight"], strict=False))
        self._t2s_model.eval()
        total = sum([param.nelement() for param in self._t2s_model.parameters()])
        logger.debug("Number of parameter: %.2fM" % (total / 1e6))

    def _is_empty(*items):  # 任意一项不为空返回False
        for item in items:
            if item is not None and item != "":
                return False
        return True


    def _is_full(*items):  # 任意一项为空返回False
        for item in items:
            if item is None or item == "":
                return False
        return True

    def _is_ready(self) -> bool:
        return self._is_full(self._default_refer_path, self._default_refer_text, self._default_refer_language)



    def _get_spepc(self,hps, filename):
        audio = load_audio(filename, int(hps.data.sampling_rate))
        audio = torch.FloatTensor(audio)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length,
                                hps.data.win_length, center=False)
        return spec

    def _get_bert_feature(self,text, word2ph):
        with torch.no_grad():
            inputs = self._tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self._device)  #####输入是long不用管精度问题，精度随bert_model
            res = self._bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        # if(is_half==True):phone_level_feature=phone_level_feature.half()
        return phone_level_feature.T
    


    def _get_tts_wav(self,ref_wav_path, prompt_text, prompt_language, text, text_language):
        t0 = ttime()
        prompt_text = prompt_text.strip("\n")
        prompt_language, text = prompt_language, text.strip("\n")
        zero_wav = np.zeros(int(self._hps.data.sampling_rate * 0.3), dtype=np.float16 if self._half_precision == True else np.float32)
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            if (self._half_precision == True):
                wav16k = wav16k.half().to(self._device)
                zero_wav_torch = zero_wav_torch.half().to(self._device)
            else:
                wav16k = wav16k.to(self._device)
                zero_wav_torch = zero_wav_torch.to(self._device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = self._ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
            codes = self._vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
        t1 = ttime()
        prompt_language = self._dict_language[prompt_language]
        text_language = self._dict_language[text_language]
        phones1, word2ph1, norm_text1 = clean_text(prompt_text, prompt_language)
        phones1 = cleaned_text_to_sequence(phones1)
        texts = text.split("\n")
        audio_opt = []

        for text in texts:
            phones2, word2ph2, norm_text2 = clean_text(text, text_language)
            phones2 = cleaned_text_to_sequence(phones2)
            if (prompt_language == "zh"):
                bert1 = self._get_bert_feature(norm_text1, word2ph1).to(self._device)
            else:
                bert1 = torch.zeros((1024, len(phones1)), dtype=torch.float16 if self._half_precision == True else torch.float32).to(
                    self._device)
            if (text_language == "zh"):
                bert2 = self._get_bert_feature(norm_text2, word2ph2).to(self._device)
            else:
                bert2 = torch.zeros((1024, len(phones2))).to(bert1)
            bert = torch.cat([bert1, bert2], 1)

            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self._device).unsqueeze(0)
            bert = bert.to(self._device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self._device)
            prompt = prompt_semantic.unsqueeze(0).to(self._device)
            t2 = ttime()
            with torch.no_grad():
                # pred_semantic = t2s_model.model.infer(
                pred_semantic, idx = self._t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k=self._config['inference']['top_k'],
                    early_stop_num=self._hz * self._max_sec)
            t3 = ttime()
            # print(pred_semantic.shape,idx)
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)  # .unsqueeze(0)#mq要多unsqueeze一次
            refer = self._get_spepc(self._hps, ref_wav_path)  # .to(device)
            if (self._half_precision == True):
                refer = refer.half().to(self._device)
            else:
                refer = refer.to(self._device)
            # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
            audio = \
                self._vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(self._device).unsqueeze(0),
                                refer).detach().cpu().numpy()[
                    0, 0]  ###试试重建不带上prompt部分
            audio_opt.append(audio)
            audio_opt.append(zero_wav)
            t4 = ttime()
        print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        yield self._hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)

    def handle(self,refer_wav_path, prompt_text, prompt_language, text, text_language):
        if (
                refer_wav_path == "" or refer_wav_path is None
                or prompt_text == "" or prompt_text is None
                or prompt_language == "" or prompt_language is None
        ):
            refer_wav_path, prompt_text, prompt_language = (
                self._default_refer_path,
                self._default_refer_text,
                self._default_refer_language,
            )
            if not self._is_ready():
                return JSONResponse({"code": 400, "message": "未指定参考音频且接口无预设"}, status_code=400)

        with torch.no_grad():
            gen = self._get_tts_wav(
                refer_wav_path, prompt_text, prompt_language, text, text_language
            )
            sampling_rate, audio_data = next(gen)

        wav = BytesIO()
        sf.write(wav, audio_data, sampling_rate, format="wav")
        wav.seek(0)

        torch.cuda.empty_cache()
        if self._device == "mps":
            print('executed torch.mps.empty_cache()')
            torch.mps.empty_cache()
        return StreamingResponse(wav, media_type="audio/wav")



class DictToAttrRecursive:
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            if isinstance(value, dict):
                # 如果值是字典，递归调用构造函数
                setattr(self, key, DictToAttrRecursive(value))
            else:
                setattr(self, key, value)
