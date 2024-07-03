import sqlite3
import pickle
from app.database.config import keyword_path
poetry_engine = sqlite3.connect("/home/czh/services/poetry-dialogue/app/database/db/poetry.db")


#创建诗词关键词
def keyword_match():
    import pandas as pd
    import re
    df = pd.read_sql("select * from poetry",poetry_engine)
    title_keywords = list(set(df["title"].to_list()))  # 标题列表
    author_keywords = list(set(df["author"].to_list()))  # 作者区列表
    dynasty_keywords = list(set(df["dynasty"].to_list()))  # 朝代列表
    content_keywords = list(set(df["content"].to_list()))  # 内容列表


    try:
        title_keywords = '|'.join(title for title in title_keywords if title)
        author_keywords = '|'.join(author for author in author_keywords if author)
        dynasty_keywords = '|'.join(dynasty for dynasty in dynasty_keywords if dynasty)
        content_keywords = '|'.join(content for content in content_keywords if content)
    except Exception as e:
        print(e)

    with open(keyword_path+"/title_keywords.pkl","wb") as file:
        pickle.dump(title_keywords, file)

    with open(keyword_path+"/author_keywords.pkl","wb") as file:
        pickle.dump(author_keywords, file)

    with open(keyword_path+"/dynasty_keywords.pkl","wb") as file:
        pickle.dump(dynasty_keywords, file)

    with open(keyword_path+"/content_keywords.pkl","wb") as file:
        pickle.dump(content_keywords,file)

if __name__ == "__main__":
    keyword_match()