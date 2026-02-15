"""
AI 提示詞評估系統
用於評估 Claude 在生成 Python、JSON 或 Regex（針對 AWS 任務）時的表現
"""

import json
import re
import ast
from statistics import mean
from dotenv import load_dotenv
from anthropic import Anthropic

# ============================================
# 初始化設定
# ============================================

# 載入環境變數並建立客戶端
load_dotenv()
client = Anthropic()
model = "claude-haiku-4-5"


# ============================================
# 輔助函數
# ============================================

def add_user_message(messages, text):
    """新增使用者訊息到對話列表"""
    user_message = {"role": "user", "content": text}
    messages.append(user_message)


def add_assistant_message(messages, text):
    """新增助理訊息到對話列表"""
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)


def chat(messages, system=None, temperature=1.0, stop_sequences=[]):
    """與 Claude 進行對話"""
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences,
    }

    if system:
        params["system"] = system

    message = client.messages.create(**params)
    return message.content[0].text


# ============================================
# 資料集生成
# ============================================

def generate_dataset():
    """生成評估資料集"""
    prompt = """
生成一個用於提示詞評估的資料集。該資料集將用於評估專門針對 AWS 相關任務生成 Python、JSON 或 Regex 的提示詞。
請生成一個 JSON 物件陣列，每個物件代表一個需要 Python、JSON 或 Regex 來完成的任務。

輸出範例：
```json
[
    {
        "task": "任務描述",
        "format": "json" 或 "python" 或 "regex",
        "solution_criteria": "評估解決方案的關鍵標準"
    },
    ...其他項目
]
```

* 專注於可以透過編寫單一 Python 函數、單一 JSON 物件或正則表達式來解決的任務
* 專注於不需要編寫大量程式碼的任務

請生成 3 個物件。
"""

    messages = []
    add_user_message(messages, prompt)
    add_assistant_message(messages, "```json")
    text = chat(messages, stop_sequences=["```"])
    return json.loads(text)


# ============================================
# 執行測試
# ============================================

def run_prompt(test_case):
    """將測試案例傳遞給 Claude 並取得輸出"""
    prompt = f"""
請解決以下任務：

{test_case["task"]}

* 僅回應 Python、JSON 或純正則表達式
* 不要添加任何註解、評論或解釋
"""

    messages = []
    add_user_message(messages, prompt)
    add_assistant_message(messages, "```code")
    output = chat(messages, stop_sequences=["```"])
    return output


# ============================================
# 評分系統
# ============================================

def validate_json(text):
    """驗證 JSON 語法"""
    try:
        json.loads(text.strip())
        return 10
    except json.JSONDecodeError:
        return 0


def validate_python(text):
    """驗證 Python 語法"""
    try:
        ast.parse(text.strip())
        return 10
    except SyntaxError:
        return 0


def validate_regex(text):
    """驗證正則表達式語法"""
    try:
        re.compile(text.strip())
        return 10
    except re.error:
        return 0


def grade_syntax(response, test_case):
    """根據格式類型評分語法"""
    format_type = test_case["format"]
    if format_type == "json":
        return validate_json(response)
    elif format_type == "python":
        return validate_python(response)
    else:
        return validate_regex(response)


def grade_by_model(test_case, output):
    """使用模型評分測試案例的輸出"""
    eval_prompt = f"""
你是一位專業的 AWS 程式碼審查專家。你的任務是評估以下 AI 生成的解決方案。

原始任務：
<task>
{test_case["task"]}
</task>

待評估的解決方案：
<solution>
{output}
</solution>

你應該使用以下標準來評估解決方案：
<criteria>
{test_case["solution_criteria"]}
</criteria>

輸出格式
請以結構化的 JSON 物件提供你的評估，包含以下欄位，並按照此特定順序：
- "strengths"（優點）：1-3 個關鍵優點的陣列
- "weaknesses"（缺點）：1-3 個需要改進之處的陣列
- "reasoning"（評分理由）：對整體評估的簡明解釋
- "score"（分數）：介於 1-10 之間的數字

請以 JSON 格式回應。保持回應簡潔直接。
回應範例結構：
{{
    "strengths": string[],
    "weaknesses": string[],
    "reasoning": string,
    "score": number
}}
"""

    messages = []
    add_user_message(messages, eval_prompt)
    add_assistant_message(messages, "```json")
    eval_text = chat(messages, stop_sequences=["```"])
    return json.loads(eval_text)


# ============================================
# 測試執行與評估
# ============================================

def run_test_case(test_case):
    """執行單一測試案例並評分結果"""
    print(f"\n執行測試案例：{test_case['task'][:50]}...")
    
    # 執行提示詞並取得輸出
    output = run_prompt(test_case)
    
    # AI 模型評分
    model_grade = grade_by_model(test_case, output)
    model_score = model_grade["score"]
    reasoning = model_grade["reasoning"]
    
    # 語法評分
    syntax_score = grade_syntax(output, test_case)
    
    # 計算最終分數（平均）
    score = (model_score + syntax_score) / 2
    
    print(f"  語法分數：{syntax_score}/10")
    print(f"  AI 評分：{model_score}/10")
    print(f"  最終分數：{score}/10")
    
    return {
        "output": output,
        "test_case": test_case,
        "score": score,
        "model_score": model_score,
        "syntax_score": syntax_score,
        "strengths": model_grade.get("strengths", []),
        "weaknesses": model_grade.get("weaknesses", []),
        "reasoning": reasoning,
    }


def run_eval(dataset):
    """載入資料集並對每個案例執行測試"""
    results = []
    
    print("\n" + "="*60)
    print("開始執行評估")
    print("="*60)

    for i, test_case in enumerate(dataset, 1):
        print(f"\n【測試案例 {i}/{len(dataset)}】")
        result = run_test_case(test_case)
        results.append(result)

    average_score = mean([result["score"] for result in results])
    
    print("\n" + "="*60)
    print(f"評估完成！平均分數：{average_score:.2f}/10")
    print("="*60)

    return results


# ============================================
# 結果輸出與儲存
# ============================================

def print_detailed_results(results):
    """印出詳細的評估結果"""
    print("\n" + "="*60)
    print("詳細評估結果")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"\n【測試案例 {i}】")
        print(f"任務：{result['test_case']['task']}")
        print(f"格式：{result['test_case']['format']}")
        print(f"\n生成的輸出：")
        print("-" * 40)
        print(result['output'])
        print("-" * 40)
        print(f"\n評分：")
        print(f"  語法分數：{result['syntax_score']}/10")
        print(f"  AI 評分：{result['model_score']}/10")
        print(f"  最終分數：{result['score']}/10")
        print(f"\n優點：")
        for strength in result['strengths']:
            print(f"  ✓ {strength}")
        print(f"\n缺點：")
        for weakness in result['weaknesses']:
            print(f"  ✗ {weakness}")
        print(f"\n評分理由：")
        print(f"  {result['reasoning']}")
        print("\n" + "="*60)


def save_results(dataset, results, dataset_filename="資料集.json", results_filename="評估結果.json"):
    """儲存資料集和結果到 JSON 檔案"""
    # 儲存資料集
    with open(dataset_filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 資料集已儲存至：{dataset_filename}")
    
    # 儲存結果
    with open(results_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ 評估結果已儲存至：{results_filename}")


# ============================================
# 主程式
# ============================================

def main():
    """主程式執行流程"""
    print("="*60)
    print("AI 提示詞評估系統")
    print("="*60)
    
    # 步驟 1：生成資料集
    print("\n步驟 1：生成測試資料集...")
    dataset = generate_dataset()
    print(f"✓ 已生成 {len(dataset)} 個測試案例")
    
    # 步驟 2：執行評估
    print("\n步驟 2：執行評估...")
    results = run_eval(dataset)
    
    # 步驟 3：顯示詳細結果
    print_detailed_results(results)
    
    # 步驟 4：儲存結果
    save_results(dataset, results)
    
    # 顯示統計資訊
    print("\n" + "="*60)
    print("統計資訊")
    print("="*60)
    average_score = mean([r["score"] for r in results])
    average_syntax = mean([r["syntax_score"] for r in results])
    average_model = mean([r["model_score"] for r in results])
    
    print(f"平均最終分數：{average_score:.2f}/10")
    print(f"平均語法分數：{average_syntax:.2f}/10")
    print(f"平均 AI 評分：{average_model:.2f}/10")
    
    # 分數分布
    high_scores = sum(1 for r in results if r["score"] >= 8)
    medium_scores = sum(1 for r in results if 5 <= r["score"] < 8)
    low_scores = sum(1 for r in results if r["score"] < 5)
    
    print(f"\n分數分布：")
    print(f"  高分 (≥8)：{high_scores} 個")
    print(f"  中分 (5-8)：{medium_scores} 個")
    print(f"  低分 (<5)：{low_scores} 個")
    print("="*60)


if __name__ == "__main__":
    main()
