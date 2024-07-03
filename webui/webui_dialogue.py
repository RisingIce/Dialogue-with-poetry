import streamlit as st
from streamlit_option_menu import option_menu
import random
import time
import asyncio
import os
import sys
import json
import pandas as pd
import requests
import random
import string
from config import api_port, bind_addr

root = os.getcwd()
temp_knowledge_path = os.path.join(root, "webui", "tmp_knowledge")
sys.path.append(root)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

from app.database.common import get_poetry_Text2SQL_engine, get_file_engine
from app.database.common import logger
from annotated_text import annotated_text


file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "json": pd.read_json,
    "html": pd.read_html,
    "xml": pd.read_xml,
    "pickle": pd.read_pickle,
}


@st.cache_data(ttl="1h")
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"不支持的文件格式: {ext}")
        return None


# 构建对对子回复生成器
def duizi_stream_response(duizi_type, response):
    yield "对对子生成结果如下：" + "\n\n"
    if duizi_type == "出上联对下联":
        yield f"上联：{response['source']}" + "\n\n"
        yield f"下联：{response['couplet'][0]}" + "\n\n"
    else:
        yield f"上联：{response['couplet'][0]}" + "\n\n"
        yield f"下联：{response['source']}" + "\n\n"


# App title
st.set_page_config(page_title="💬 Chatbot")


def get_response_generator(ans):
    log_msg = ""
    for line in ans.iter_lines():
        if line:
            decoded_line = line.decode("utf-8")
            res = decoded_line.replace("data:", "")
            final = dict(json.loads(res))
            log_msg += final["content"]
            yield final["content"]

    logger.debug(f"回复内容：{log_msg}")


def file_query_generator(df, query: str):
    query_engine = get_file_engine(streaming=True, df=df)
    res = query_engine.query(query)
    #  print(res)
    return log_gen(res.response_gen)


def log_gen(gen):
    log_msg = ""
    for text in gen:
        log_msg += f"{text}".strip("\n\n")
        yield text
    logger.debug(f"回复内容：{log_msg}")


with st.sidebar:
    st.title("💬 Poetry Chatbot")
    selected = option_menu(
        "主要功能",
        [
            "诗词对话",
            "知识库对话",
            "诗词生成",
            "对对子",
            "诗词图云",
            "数据库",
            "设置",
        ],
        icons=[
            "chat-dots",
            "files",
            "lightbulb",
            "mortarboard",
            "sunset",
            "database",
            "gear",
        ],
        menu_icon="house",
        default_index=2,
    )
    # selected


if selected == "诗词对话":
    # st.title('💬 Poetry Chatbot')
    if "diagolue_messages" not in st.session_state.keys():
        st.session_state.diagolue_messages = [
            {
                "role": "assistant",
                "content": "你好！我是诗词对话机器人，你可以问有关古诗的一切，我会尽我所能回答你！",
            }
        ]

    # Display or clear chat messages
    for diagolue_message in st.session_state.diagolue_messages:
        with st.chat_message(diagolue_message["role"]):
            st.write(diagolue_message["content"])

    def clear_chat_history():
        st.session_state.diagolue_messages = [
            {
                "role": "assistant",
                "content": "你好！我是诗词对话机器人，你可以问有关古诗的一切，我会尽我所能回答你！",
            }
        ]

    st.sidebar.button("清空对话历史", on_click=clear_chat_history)

    # User-provided prompt
    if prompt := st.chat_input(placeholder="输入问题"):

        st.session_state.diagolue_messages.append({"role": "user", "content": prompt})
        logger.debug(f"用户输入:{prompt}")
        with st.chat_message("user"):
            st.write(prompt)

        # Generate a new response if last message is not from assistant
        if st.session_state.diagolue_messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                ans = requests.post(
                    f"http://{bind_addr}:{api_port}/dialogue",
                    json={"query": str(prompt)},
                    stream=True,
                )
                response = st.write_stream(get_response_generator(ans))

            message = {"role": "assistant", "content": response}
            st.session_state.diagolue_messages.append(message)


if selected == "知识库对话":
    uploaded_file = st.sidebar.file_uploader(
        label="上传文件",
        type=list(file_formats.keys()),
        help="上传你的文件并且与你的文件对话！",
    )

    if not uploaded_file:
        st.warning("您必须要先上传文件才能进行对话！")
        st.stop()

    # st.success('数据加载完毕', icon="✅")
    if "knowledge_messages" not in st.session_state.keys():
        st.session_state.knowledge_messages = [
            {
                "role": "assistant",
                "content": "欢迎来到知识库对话，你可以上传你的文件并且使用对话的方式了解文件内容！",
            }
        ]

        # Display or clear chat messages
    for knowledge_message in st.session_state.knowledge_messages:
        with st.chat_message(knowledge_message["role"]):
            st.write(knowledge_message["content"])

    def clear_chat_history():
        st.session_state.knowledge_messages = [
            {
                "role": "assistant",
                "content": "欢迎来到知识库对话，你可以上传你的文件并且使用对话的方式了解文件内容！",
            }
        ]

    if uploaded_file:
        clear_chat_history()
        df = load_data(uploaded_file)
        st.toast("数据加载完毕", icon="🎉")

    st.sidebar.button("清空对话历史", on_click=clear_chat_history)
    if prompt := st.chat_input(placeholder="输入问题"):
        st.session_state.knowledge_messages.append({"role": "user", "content": prompt})
        print("用户输入:", prompt)
        with st.chat_message("user"):
            st.write(prompt)

            # Generate a new response if last message is not from assistant
        if st.session_state.knowledge_messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                response = st.write_stream(file_query_generator(df, prompt))
            message = {"role": "assistant", "content": response}
            st.session_state.knowledge_messages.append(message)

if selected == "诗词生成":
    base_url = "https://jiuge.thunlp.org/jiugepoem/task/"
    characters = string.ascii_letters + string.digits
    type_selected = st.sidebar.selectbox(
        label="诗词类型",
        options=["绝句", "风格绝句", "藏头诗", "律诗", "词"],
        index=None,
    )
    if type_selected == "绝句":
        format_poetry = st.sidebar.selectbox(
            label="诗词格式", options=["五言绝句", "七言绝句"], index=0
        )
        emotion_type = ""
        params = {
            "type": type_selected,
            "format": 5 if format_poetry == "五言绝句" else 7,
            "prompt": None,
            "emotion_type": None,
        }
    elif type_selected == "风格绝句":
        emotion_list = ["萧瑟凄凉", "忆旧感喟", "孤寂惆怅", "思乡忧老", "渺远孤逸"]
        format_poetry = st.sidebar.selectbox(
            label="诗词格式", options=["五言绝句", "七言绝句"], index=0
        )
        emotion_type = st.sidebar.selectbox(
            label="感情色彩",
            options=emotion_list,
            index=0,
        )
        send_url = base_url + "send_juejustyle"
        get_url = base_url + "get_juejustyle"
        params = {
            "type": type_selected,
            "format": 5 if format_poetry == "五言绝句" else 7,
            "prompt": None,
            "emotion_type": emotion_list.index(emotion_type),
        }

    elif type_selected == "藏头诗":
        emotion_list = ["悲伤", "较悲伤", "中性", "较喜悦", "喜悦"]
        format_poetry = st.sidebar.selectbox(
            label="诗词格式", options=["五言藏头", "七言藏头"], index=0
        )
        emotion_type = st.sidebar.selectbox(
            label="感情色彩", options=emotion_list, index=None
        )
        params = {
            "type": type_selected,
            "format": 5 if format_poetry == "五言藏头" else 7,
            "poem": None,
            "emotion_type": (
                -1 if emotion_type == None else emotion_list.index(emotion_type)
            ),
        }

    elif type_selected == "律诗":
        format_poetry = st.sidebar.selectbox(
            label="诗词格式", options=["五言律诗", "七言律诗"], index=0
        )
        emotion_type = ""
        send_url = base_url + "send_lvshi"
        get_url = base_url + "get_lvshi"

        params = {
            "type": type_selected,
            "format": 5 if format_poetry == "五言律诗" else 7,
            "prompt": None,
        }

    elif type_selected == "词":
        format_list = [
            "归字谣",
            "如梦令",
            "梧桐影 ",
            "渔歌子",
            "捣练子",
            "忆江南",
            "秋风清",
            "忆王孙",
            "河满子",
            "思帝乡",
            "望江怨",
            "醉吟商",
            "卜算子",
            "点绛唇",
            "乌夜啼",
            "江亭怨",
            "踏莎行",
            "画堂春",
            "浣溪沙",
            "武陵春",
            "采桑子",
            "城头月",
            "玉楼春",
            "海棠春",
            "苏幕遮",
            "蝶恋花",
            "江城子",
            "八声甘州",
            "声声慢",
            "水龙吟",
            "满江红",
            "沁园春",
        ]
        format_poetry = st.sidebar.selectbox(label="词牌", options=format_list, index=0)
        emotion_type = ""

        params = {
            "type": type_selected,
            "prompt": None,
            "format": format_list.index(format_poetry),
        }

    if not type_selected:
        st.warning("您必须要先选择诗词类型才能进行诗词生成！")
        st.stop()

    annotated_text(
        "目前选择的诗词类型是：",
        (str(type_selected), "type"),
        (str(format_poetry), "format"),
        (str(emotion_type), "emotion"),
    )

    if "generate_messages" not in st.session_state.keys():
        st.session_state.generate_messages = [
            {
                "role": "assistant",
                "content": "欢迎来到诗词生成，你可以选择诗词的类型和自己喜欢的提示词生成专属于自己的古诗！",
            }
        ]

    for generate_message in st.session_state.generate_messages:
        # print(generate_message["role"])
        with st.chat_message(generate_message["role"]):
            st.write(generate_message["content"])

    def clear_chat_history():
        st.session_state.generate_messages = [
            {
                "role": "assistant",
                "content": "欢迎来到诗词生成，你可以选择诗词的类型和自己喜欢的提示词生成专属于自己的古诗！",
            }
        ]

    st.sidebar.button("清空对话历史", on_click=clear_chat_history)

    if prompt := st.chat_input(placeholder="请输入句子、段落或者关键字"):
        with st.chat_message("user"):
            st.write(f"你当前输入的自定义关键词为：{prompt}")
        st.session_state.generate_messages.append(
            {"role": "user", "content": f"你当前输入的自定义关键词为：{prompt}"}
        )

        params["prompt"] = prompt
        print("params", params)

        if st.session_state.generate_messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.status("诗词生成中..."):
                    res = requests.post(
                        f"http://{bind_addr}:{api_port}/generate",
                        json=params,
                        stream=True,
                    )
                    if res.status_code == 200:
                        st.write("生成成功")
                response = st.write_stream(get_response_generator(res))
            message = {"role": "assistant", "content": response}
            st.session_state.generate_messages.append(message)


if selected == "对对子":
    base_url = "https://jiuge.thunlp.org/"
    # st.title('💬 Poetry Chatbot')
    if "duizi_messages" not in st.session_state.keys():
        st.session_state.duizi_messages = [
            {"role": "assistant", "content": "你好！欢迎来到对对子，请出上联或者下联！"}
        ]

    # Display or clear chat messages
    for duizi_message in st.session_state.duizi_messages:
        with st.chat_message(duizi_message["role"]):
            st.write(duizi_message["content"])

    def clear_chat_history():
        st.session_state.duizi_messages = [
            {"role": "assistant", "content": "你好！欢迎来到对对子，请出上联或者下联！"}
        ]

    type_selected = st.sidebar.selectbox(
        label="对子类型", options=["出上联对下联", "出下联对上联"], index=0
    )

    params = {
        "prompt": None,
        "predict_type": ("lower" if type_selected == "出上联对下联" else "upper"),
    }

    st.sidebar.button("清空对话历史", on_click=clear_chat_history)

    if prompt := st.chat_input(placeholder="请输入句子,句子最好不超过15字"):
        with st.chat_message("user"):
            st.write(f"你当前输入的上联为：{prompt}")
        st.session_state.duizi_messages.append(
            {"role": "user", "content": f"你当前输入的上联为：{prompt}"}
        )

        params["prompt"] = prompt
        # print("send_url",send_url)
        # print('get_url',get_url)
        # print("params",params)
        res = requests.post(
            f"http://{bind_addr}:{api_port}/pairs",
            json=params,
            stream=True,
        )
        if st.session_state.duizi_messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                response = st.write_stream(get_response_generator(res))
            message = {"role": "assistant", "content": response}
            st.session_state.duizi_messages.append(message)


# if selected =='语音识别':
#      from audiorecorder import audiorecorder

# st.title("Audio Recorder")
# audio = audiorecorder("Click to record", "Click to stop recording")

# if len(audio) > 0:
#     # To play audio in frontend:
#     st.audio(audio.export().read())

#     # To save audio to a file, use pydub export method:
#     audio.export("webui/audio.wav", format="wav")

#     # To get audio properties, use pydub AudioSegment properties:
#     st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")
