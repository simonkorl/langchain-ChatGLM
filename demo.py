# 制作高考演示问题展示文件的脚本

import json
import pandas as pd

if __name__ == "__main__":
    glm_results = json.load(open("glm_history.json", 'r'))
    glm2_results = json.load(open("glm2_history.json", 'r'))
    langchain_results = json.load(open("results/gaokao_result/langchain_result_new.json", 'r'))

    print(glm_results[0].keys())
    print(langchain_results[0].keys())

    table_data = {
        '问题': [obj['query'] for obj in langchain_results],
        '高考咨询 AI 回答': [obj['answer'] for obj in langchain_results],
        'chatglm 回答': [obj['answer'] for obj in glm_results],
        'chatglm2 回答': [obj['answer'] for obj in glm2_results],
    }

    table_df = pd.DataFrame(table_data)
    table_df.to_csv('save.csv', encoding='utf-8')
    
    with open('output.md', 'w') as f:
        for idx, row in table_df.iterrows():
            f.write(f"# {row['问题']}\n")
            f.write(f"## 高考咨询 AI 回答\n {row['高考咨询 AI 回答']}\n")
            f.write(f"## chatglm 回答\n {row['chatglm 回答']}\n")
            f.write(f"## chatglm2 回答\n {row['chatglm2 回答']}\n")