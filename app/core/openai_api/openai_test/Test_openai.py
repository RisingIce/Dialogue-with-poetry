from openai import OpenAI
import os

os.environ["OPENAI_API_KEY"] = "sk-syg0jdxayU5gXLzT4fA49a77DaC749B8B925481739B921E1"

#根据测试环境不同此处可能需要根据情况修改
os.environ['BASE_URL'] = 'http://127.0.0.1:8000/v1'

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'],base_url=os.environ['BASE_URL'])
import json

#是否启用流式传输
flag = False

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "你好"}
  ],
    stream=flag

)


if flag:
#输出非流式传输数据
    for line in completion.response.iter_lines():
        if line:
            # decoded_line = line.decode('utf-8')[6:]
            try:
                response_json = json.loads(line)
                content = response_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                print(content)
            except:
                print("Special Token:", line)
else:
#处理流式传输数据
    print(completion.choices[0].message)

