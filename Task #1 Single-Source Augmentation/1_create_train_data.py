import pandas as pd
import pickle
import copy

df = pd.read_parquet("../new_task1_format_data.parquet")
df["ans_full"]=df["answer"]

df["image_url_not_null"] = df["image_url_not_null"].fillna(False)


train_zj = pd.read_json("../zj_data/train_crag_mm_comb_task1_11b_image_search_top10_labeled_shift.jsonl",lines=True)
val_zj = pd.read_json("../zj_data/eval_crag_mm_comb_task1_11b_image_search_top10_labeled_shift.jsonl",lines=True)


train_zj = train_zj.merge(df[["interaction_id","local_path","image_url_not_null"]],how="left",on=["interaction_id"])

val_zj = val_zj.drop("ans_full",axis=1)
val_zj = val_zj.merge(df[["interaction_id","local_path","image_url_not_null","ans_full"]],how="left",on=["interaction_id"])

temp=train_zj[train_zj["ans_full"]=="I don't know."]["interaction_id"].values
temp.shape

with open("augu_v2_gpt4o_task3.pkl",'rb') as f:
    data=pickle.load(f)
print(data.shape)
data = data[data["relu_data"].apply(lambda x: x["api_response"] is not None and x["api_response"]["accuracy"]==True )]
data = data[data["interaction_id"].apply(lambda x: x not in temp)]
data["ans_full"]=data["response"]
data["ans_full"] = data["ans_full"].apply(lambda x: x.replace("(精简)","").replace("(規則:精简)","").replace("精简: ","").replace("(rule:精简)",""))
data["ans_full"] = data["ans_full"].apply(lambda x: x.replace("(复杂)","").replace("(規則:复杂)","").replace("复杂: ","").replace("(rule:复杂)","").replace("**复杂**：",""))
argument_df = data
print(data.shape)

argument_df = argument_df.merge(df[["interaction_id","local_path","image_url_not_null"]],how="left",on=["interaction_id"])
# argument_df = argument_df.merge(train_zj[["interaction_id","image_search_info_list"]],how="left",on=["interaction_id"])

import re

def contains_chinese(text):
    # 匹配任意一个中文字符
    return re.search(r'[\u4e00-\u9fff]', text) is not None
argument_df=argument_df[~argument_df["ans_full"].apply(lambda x:contains_chinese(x))]
argument_df.shape


train_zj = pd.concat([train_zj,argument_df],axis=0).sample(frac=1.,random_state=2024)
train_zj.shape


with open("augu_v2_gpt4o_more.pkl",'rb') as f:
    data=pickle.load(f)
print(data.shape)
data = data[data["relu_data"].apply(lambda x: x["api_response"] is not None and x["api_response"]["accuracy"]==True )]
data = data[data["interaction_id"].apply(lambda x: x not in temp)]
data["ans_full"]=data["response"]
data["ans_full"] = data["ans_full"].apply(lambda x: x.replace("(精简)","").replace("(規則:精简)","").replace("精简: ","").replace("(rule:精简)",""))
data["ans_full"] = data["ans_full"].apply(lambda x: x.replace("(复杂)","").replace("(規則:复杂)","").replace("复杂: ","").replace("(rule:复杂)","").replace("**复杂**：",""))
argument_df = data
print(data.shape)

argument_df = argument_df.merge(df[["interaction_id","local_path","image_url_not_null"]],how="left",on=["interaction_id"])
# argument_df = argument_df.merge(train_zj[["interaction_id","image_search_info_list"]],how="left",on=["interaction_id"])

import re

def contains_chinese(text):
    # 匹配任意一个中文字符
    return re.search(r'[\u4e00-\u9fff]', text) is not None
argument_df=argument_df[~argument_df["ans_full"].apply(lambda x:contains_chinese(x))]
argument_df.shape


train_zj = pd.concat([train_zj,argument_df],axis=0).sample(frac=1.,random_state=2024)
train_zj.shape


train_zj = train_zj.drop_duplicates(["query","ans_full"])
train_zj.shape


rounds = []
for _,row in train_zj.iterrows():
    lis = []
    for t in row["image_search_info_list"][:5]:
        lis.append(' '.join(t.split(' ')[:1000]))
    info = "\n".join(lis)
    messages = [
                {"role": "system", "content": "You are a helpful assistant that truthfully answers user questions about the provided image with informations that might be related to the question.\n Keep your response concise and to the point. If you don't know the answer, respond with 'I don't know'."},
                {"role": "user", "content": [{"type": "image"}]},
                {"role":"user", "content": f"informations that might be related to the question:{info}"},
                {"role":"user", "content": f"###user question:{row['query']}"},
]
    rounds.append(messages)
train_zj["messages"] = rounds
train_zj["train_answer"] = train_zj["ans_full"]


rounds = []
for _,row in val_zj.iterrows():
    lis = []
    for t in row["image_search_info_list"][:5]:
        lis.append(' '.join(t.split(' ')[:1000]))
    info = "\n".join(lis)
    messages = [
                {"role": "system", "content": "You are a helpful assistant that truthfully answers user questions about the provided image with informations that might be related to the question.\n Keep your response concise and to the point. If you don't know the answer, respond with 'I don't know'."},
                {"role": "user", "content": [{"type": "image"}]},
                {"role":"user", "content": f"informations that might be related to the question:{info}"},
                {"role":"user", "content": f"###user question:{row['query']}"},
]
    rounds.append(messages)
val_zj["messages"] = rounds
val_zj["train_answer"] = val_zj["ans_full"]


train_zj = train_zj.sample(frac=1.,random_state=2027)


with open("../train_data/v5_argu_add_all/train.pkl","wb") as f:
    pickle.dump(train_zj,f)
with open("../train_data/v5_argu_add_all/dev.pkl","wb") as f:
    pickle.dump(val_zj,f)

