import time
from google.genai.errors import ServerError
import regex as re
from google import genai
from google.genai import types
import os
import json
from openai import OpenAI
import time
import argparse
client = OpenAI(api_key="", base_url="")
model = "gemini-2.5-flash"


def extract_json(text):
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1)
    return text

def evaluate_model_answer(history, question, gt_answer, model_answer, checklist_dict):
    checklist_str = "\n".join([f"{i}: {item}" for i, item in checklist_dict.items() if i[0]=="Q"])
    question_num = len([f"{i}: {item}" for i, item in checklist_dict.items() if i[0]=="Q"])
    PROMPT = f"""
    Given a current user question and its standard answer, a model-generated answer, and a checklist of several questions,
    Generate answers to the questions in the checklist to evaluate the model's performance.
    
    ### User Question:
    {question}

    ### Standard Answer:
    {gt_answer}

    ### Model Answer:
    {model_answer}

    ### Checklist:
    {checklist_str}

    **Output Format Example:**
    ```json
    {{
    "A1": "Yes/No",
    "A2": "Yes/No",
    "A3": "Yes/No",
    "A4": "Yes/No",
    "A5": "Yes/No"
    }}
    ```
    """

    max_retries = 5

    # print(PROMPT)
    start_time = time.time()
    while True:
        try:
            response = client.chat.completions.create(
                model = model,
                messages = [
                    {
                        "role": "user",  
                        "content": [   
                            {
                                "type": "text",
                                "text": PROMPT
                            }
                        ],
                    }
                ],
            )
            result_raw = response.choices[0].message.content
            result_new = extract_json(result_raw)
            results = json.loads(result_new)
            break
        except Exception as e:
            print(f"failed, wait 1s ...")
            time.sleep(1)

    end_time = time.time()
    print(f"Evaluation took {end_time - start_time:.2f} seconds.")
    correct = 0
    answers_dir = {k:v for k, v in checklist_dict.items() if k[0]=="A"}
    for result,answer in zip(results.values(), answers_dir.values()):
        if result == answer:
            correct += 1
    

    return results, round(correct / len(results), 2)

def eval_pipeline(result_path, reference_qa_path, model_answers_path, checklist_path, args):
    if not os.path.exists(result_path):
        with open(result_path, "w") as f:
            json.dump({}, f)
    with open(result_path, "r") as f:
        results = json.load(f)
    with open(reference_qa_path, "r") as f:
        reference_qa = json.load(f)
    with open(model_answers_path,"r") as f:
        model_answers = json.load(f)
    with open(checklist_path, "r") as f:
        checklist_array = json.load(f)

    for video_name, multi_events_qa in reference_qa.items():
        if video_name in results and len(results[video_name]) == len(multi_events_qa):
            continue
        elif video_name not in results:
            results[video_name] = {}
        if video_name not in checklist_array:
            continue
        # print(video_name)
        for index, single_event_qa in enumerate(multi_events_qa):
            event_index = single_event_qa["event_index"]
            if f"event_{event_index}" in results[video_name] or f"event_{event_index}" not in checklist_array[video_name]:
                continue
            else:
                results[video_name][f"event_{event_index}"] = {}
            # print(f"event_{event_index}")
            all_history = []
            # print(len(single_event_qa))
            for i in range(len(single_event_qa)):
                if f"Round {i + 1}" not in single_event_qa:
                    continue
                for name, words in single_event_qa[f"Round {i + 1}"].items():
                    if name[0] == "Q" or 'User' in name:
                        sentence = f"User: {words}" 
                        all_history.append(sentence)
                    elif name != "Ability" and (name[0] == "A" or 'Bot' in name):
                        sentence = f"Assistant: {words}"
                        all_history.append(sentence)
            masked = [i["round index"] for i in checklist_array[video_name][f"event_{event_index}"]]
            checklist_index = -1
            for i in range(len(single_event_qa)):
                if i+1 not in masked:
                    continue
                else:
                    checklist_index += 1
                if f"Round {i + 1}" not in single_event_qa:
                    continue
                if f"Q{i+1}" in results[video_name][f"event_{event_index}"]:
                    continue
                else:
                    results[video_name][f"event_{event_index}"][f"Q{i+1}"] = {}
                history = all_history[ : i * 2]
                question = all_history[i * 2]
                gt_answer = all_history[i * 2 + 1]
                print(video_name)
                model_answer = model_answers[video_name][f"event_{event_index}"][i]  # right
                checklist = checklist_array[video_name][f"event_{event_index}"][checklist_index]
                checklist = {k: v for k, v in checklist.items() if k.startswith(('Q', 'A'))}
                print("Checklist:\n", checklist)                    
                
                eval_results, acc = evaluate_model_answer(history, question, gt_answer, model_answer, checklist)
                print("\nEvaluation Results:")
                for item, result in eval_results.items():
                    print(f"- {item} → {result}")
                results[video_name][f"event_{event_index}"][f"Q{i+1}"]["CHECKLISTS_EVAL"] =  eval_results
                results[video_name][f"event_{event_index}"][f"Q{i+1}"]["ACC"] = acc

                with open(result_path, "w", encoding='utf-8') as g:
                    json.dump(results, g, ensure_ascii=False, indent=4)
                print(f"ACC: {acc}")

        
def summary_for_eval_results(evaluation_path, summary_path, args):
    with open(evaluation_path, "r") as f:
        results = json.load(f)
    entities = {}
    scores = []

    for video_name, output_dir in results.items(): # 遍历不同的视频
        entities[video_name] = []
        for single_result in output_dir.values():  # 遍历一个视频的不同的event（多轮对话）
            ans = 0
            num = 0
            for single_event in single_result.values():  # 遍历一个event（多轮对话）的不同round
                ans += single_event["ACC"]
                num += 1
            scores.append(round(ans / num, 2) if num else 0)
            entities[video_name].append(round(ans / num, 2) if num else 0)
    
    summary = {}
    average_scores = sum(scores) / len(scores) if len(scores) else 0
    summary['mean_mtqa'] = average_scores
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4) 


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Evaluate model outputs")
    parser.add_argument('--model_type', type=str, default='qwen3b', help='Type of the model to evaluate')
    parser.add_argument('--ability', type=str, default='')
    
    args = parser.parse_args()


    model_type = args.model_type
    output_dir = f'./eval_result'
    reference_qa_dir = f'MT-Video-Bench/data'
    model_answers_dir = f'./inference_answers/answers_{model_type}'
    checklist_dir = f'MT-Video-Bench/checklist'
    os.makedirs(output_dir, exist_ok=True)
    
    os.makedirs(os.path.join(output_dir, model_type), exist_ok=True)
    
    for eval_json in os.listdir(model_answers_dir):
        # if args.ability not in eval_json:
        #     continue
        print(f"Processing {eval_json}...")
        eval_name = eval_json.split('.')[0]
        result_path = os.path.join(output_dir, model_type, eval_json)
        reference_qa_path = os.path.join(reference_qa_dir, eval_json)
        model_answers_path = os.path.join(model_answers_dir, eval_json)
        checklist_path = os.path.join(checklist_dir, f'{eval_name}_CheckList.json')
        eval_pipeline(result_path, reference_qa_path, model_answers_path, checklist_path, args)
    
    for eval_json in os.listdir(model_answers_dir):
        # if args.ability not in eval_json:
        #     continue
        eval_name = eval_json.split('.')[0]
        evaluation_path = os.path.join(output_dir, model_type, eval_json)
        summary_path = os.path.join(output_dir, model_type, f'{eval_name}_average.json')
        summary_for_eval_results(evaluation_path, summary_path, args)