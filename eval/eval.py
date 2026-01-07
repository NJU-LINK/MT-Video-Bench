import time
import regex as re
import os
import json
from openai import OpenAI
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

# Global Configurations
client = OpenAI(api_key="", base_url="")
model = "gemini-2.5-flash"
file_lock = threading.Lock()

def extract_json(text):
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1)
    return text

def evaluate_single_task(task_data):
    video_name = task_data['video_name']
    event_index = task_data['event_index']
    round_idx = task_data['round_idx']
    model_answer = task_data['model_answer']
    checklist_dict = task_data['checklist']
    eval_rounds = task_data.get('eval_rounds', 1)
    
    checklist_str = "\n".join([f"{i}: {item}" for i, item in checklist_dict.items() if i[0]=="Q"])
    
    PROMPT = f"""
    Given a current model-generated answer and a checklist of five questions, output whether the model has correctly addressed the above questions.

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

    answers_dir = {k: v for k, v in checklist_dict.items() if k[0] == "A"}
    
    run_history = []
    total_acc = 0.0
    valid_runs = 0
    
    # --- Multiple cycle evaluations ---
    for run_i in range(eval_rounds):
        backoff = 2
        max_retries = 5
        results = None
        
        for _ in range(max_retries):
        # while True:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
                )
                result_raw = response.choices[0].message.content
                result_new = extract_json(result_raw)
                results = json.loads(result_new)
                break
            except Exception as e:
                # print(f"Error in {video_name} {event_index} R{round_idx} Run{run_i}: {e}, retrying...")
                time.sleep(backoff)
                backoff = 2

        if results is None:
            continue 

        correct = 0
        if len(results) > 0:
            for result_val, answer_val in zip(results.values(), answers_dir.values()):
                if str(result_val).lower() == str(answer_val).lower():
                    correct += 1
            acc = round(correct / len(results), 2)
        else:
            acc = 0.0

        run_history.append({
            "run_id": run_i + 1,
            "results": results,
            "acc": acc
        })
        total_acc += acc
        valid_runs += 1

    if valid_runs == 0:
        return None

    avg_acc = round(total_acc / valid_runs, 2)
    
    last_eval_result = run_history[-1]['results']

    return {
        "video_name": video_name,
        "event_index": event_index,
        "round_key": f"Round {round_idx}",
        "avg_acc": avg_acc,             
        "last_eval_result": last_eval_result, 
        "history": run_history     
    }

def save_results_safely(result_path, results_dict):
    with file_lock:
        sorted_videos = sorted(results_dict.keys())
        sorted_results = {}
        for vid in sorted_videos:
            sorted_results[vid] = {}
            events = list(results_dict[vid].keys())
            
            def event_sort_key(k):
                nums = re.findall(r'\d+', str(k))
                return int(nums[0]) if nums else k
            
            sorted_events = sorted(events, key=event_sort_key)
            
            for evt in sorted_events:
                sorted_results[vid][evt] = {}
                rounds = list(results_dict[vid][evt].keys())
                
                def round_sort_key(k):
                    nums = re.findall(r'\d+', str(k))
                    return int(nums[0]) if nums else k
                
                sorted_rounds = sorted(rounds, key=round_sort_key)
                
                for rnd in sorted_rounds:
                    sorted_results[vid][evt][rnd] = results_dict[vid][evt][rnd]

        with open(result_path, "w", encoding='utf-8') as g:
            json.dump(sorted_results, g, ensure_ascii=False, indent=4)

def eval_pipeline_threaded(result_path, mtqa_path, model_answers_path, checklist_path, args, max_workers=10):
    # 1. Initialize or read existing results
    if not os.path.exists(result_path):
        results = {}
        with open(result_path, "w") as f:
            json.dump({}, f)
    else:
        try:
            results = json.load(open(result_path, "r"))
        except:
            results = {}

    reference_qa = json.load(open(mtqa_path, "r"))
    model_answers = json.load(open(model_answers_path, "r"))
    checklist_array = json.load(open(checklist_path, "r"))

    tasks = []

    # 2. Collect all tasks that need to be performed
    for video_name, mtqa_list in reference_qa.items():
        if video_name not in checklist_array or video_name not in model_answers:
            print(f"video name continue: {video_name}")
            continue
            
        if video_name not in results:
            results[video_name] = {}

        for mtqa_dict in mtqa_list:
            event_index = str(mtqa_dict["event_index"])
            
            if event_index not in checklist_array[video_name]:
                print(f"event index contimue: {video_name} {event_index}")
                continue

            if event_index not in results[video_name]:
                results[video_name][event_index] = {}

            mtqa = mtqa_dict["mtqa"]
            checklist_items = checklist_array[video_name][event_index]
            eval_rounds_map = {item["round index"]: idx for idx, item in enumerate(checklist_items)}
            
            for round_key, round_data in mtqa.items():
                try:
                    round_idx = int(round_key.split("Round ")[-1])
                except:
                    print(f"Invalid round key format: {round_key} in {video_name} {event_index}")
                    continue

                if round_idx in eval_rounds_map:
                    if round_key in results[video_name][event_index]:
                        continue
                    
                    checklist_idx = eval_rounds_map[round_idx]
                    checklist = checklist_items[checklist_idx]
                    clean_checklist = {k: v for k, v in checklist.items() if k.startswith(('Q', 'A'))}
                    
                    if event_index in model_answers[video_name]:
                        try:
                             model_ans = model_answers[video_name][event_index][round_idx - 1]
                        except (IndexError, KeyError):
                            print(f"Warning: Missing model answer for {video_name} {event_index} R{round_idx}")
                            continue
                    else:
                        continue

                    tasks.append({
                        "video_name": video_name,
                        "event_index": event_index,
                        "round_idx": round_idx,
                        "model_answer": model_ans,
                        "checklist": clean_checklist,
                        "eval_rounds": args.eval_rounds
                    })

    print(f"Total tasks to evaluate: {len(tasks)} (Each task runs {args.eval_rounds} times)")
    
    # 3. Concurrent execution
    if tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(evaluate_single_task, task): task for task in tasks}
            
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc=f"Evaluating {os.path.basename(result_path)}"):
                res = future.result()
                if res:
                    v_name = res['video_name']
                    e_idx = res['event_index']
                    r_key = res['round_key']
                    
                    results[v_name][e_idx][r_key] = {
                        "CHECKLISTS_EVAL": res['last_eval_result'],
                        "ACC": res['avg_acc'],                      
                        "ALL_RUNS": res['history']             
                    }
                    
                    save_results_safely(result_path, results)
    else:
        print("No new tasks to process.")

def summary_for_eval_results(evaluation_path, summary_path, model_answers_path):
    if not os.path.exists(evaluation_path):
        return
        
    with open(evaluation_path, "r") as f:
        results = json.load(f)
    
    with open(model_answers_path, "r") as f:
        model_answers = json.load(f)

    scores = []
    for video_name, output_dir in results.items(): 
        if video_name not in model_answers:
            continue
        for event_idx, single_result in output_dir.items():  
            if event_idx not in model_answers[video_name]:
                continue
            ans = 0
            num = 0
            for single_event in single_result.values(): 
                if "ACC" in single_event:
                    ans += single_event["ACC"]
                    num += 1
            if num > 0:
                scores.append(round(ans / num, 2))

    summary = {}
    average_scores = sum(scores) / len(scores) if len(scores) else 0
    summary['mean_mtqa'] = average_scores
    
    ability_name = os.path.basename(evaluation_path).split('.')[0]
    summary['ability'] = ability_name

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    print(f"{ability_name}: {summary['mean_mtqa']}")

    return summary['mean_mtqa']

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Evaluate model outputs")
    parser.add_argument('--model_names', type=str, nargs='+',
                    default=[],
                    help='List of model names')
    parser.add_argument('--mtqa_dir', type=str, default="mtqa")
    parser.add_argument('--model_answers_base_dir', type=str, default="./output/model_answers") 
    parser.add_argument('--checklist_dir', type=str, default="checklist")
    parser.add_argument('--output_dir', type=str, default="./output/eval_results")
    parser.add_argument('--workers', type=int, default=32, help='Number of parallel threads') 
    parser.add_argument('--eval_rounds', type=int, default=3, help='Number of evaluation rounds per task to calculate average score')
    
    args = parser.parse_args()
    
    score_all_all_models = [
       "InternVL3_5-8B_nothink",
       "InternVL3_5-8B_think",
    ]
    # score_all_all_models = args.model_names
    
    for model_name in score_all_all_models:
        print(f"\n### Processing model: {model_name} with {args.eval_rounds} rounds per task ###")
        model_answers_dir = os.path.join(args.model_answers_base_dir, model_name)
        model_output_dir = os.path.join(args.output_dir, model_name+"_eval_"+model)
        
        if not os.path.exists(model_answers_dir):
            print(f"Skipping {model_name}, dir not found: {model_answers_dir}")
            continue

        files = sorted([f for f in os.listdir(model_answers_dir) if f.endswith('.json')])

        os.makedirs(model_output_dir, exist_ok=True)
        for eval_json in files:
            print(f"Processing {eval_json} ...")
            eval_name = eval_json.split('.')[0]
            result_path = os.path.join(model_output_dir, eval_json)
            mtqa_path = os.path.join(args.mtqa_dir, eval_json)
            model_answers_path = os.path.join(model_answers_dir, eval_json)
            checklist_path = os.path.join(args.checklist_dir, f'{eval_name}_CheckList.json')
            print(f"checklist_path path: {checklist_path}")
            
            if not os.path.exists(checklist_path):
                print(f"Checklist not found for {eval_name}, skipping.")
                continue
            
            eval_pipeline_threaded(result_path, mtqa_path, model_answers_path, checklist_path, args, max_workers=args.workers)
        
        # Summarize results
        score_all = []
        for eval_json in files:
            eval_name = eval_json.split('.')[0]
            evaluation_path = os.path.join(model_output_dir, eval_json)
            summary_path = os.path.join(model_output_dir, f'{eval_name}_summary.json')
            model_answers_path = os.path.join(model_answers_dir, eval_json)
            if os.path.exists(evaluation_path):
                score_all.append(summary_for_eval_results(evaluation_path, summary_path, model_answers_path))
        
        if len(score_all) > 0:
            print(f"Mean Average Score for {model_name}: {sum(score_all)/len(score_all):.4f}")
        else:
            print(f"No valid results for {model_name}.")