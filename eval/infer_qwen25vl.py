import os
import json
import argparse
import logging
from tqdm import tqdm
from datetime import datetime
import numpy as np
import cv2
from decord import VideoReader, cpu
import torch

ABILITY_LIST = [
    'Summary', 'Object Reference', 'Memory Recall', 'Answer Refusal'
]
ABILITY_0 = [
    'summary', 'object_reference', 'memory_recall', 'answer_refusal'
]
ABILITY_1 = [
    'proactive_interaction', 'topic_shifting'
]

def encode_video(video_path, max_pixels=720, max_frames=128, temp_dir=None):
    import warnings
    warnings.filterwarnings("ignore")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    target_max_size = max_pixels
    if original_width > target_max_size or original_height > target_max_size:
        n_width = int(target_max_size * original_width / max(original_width, original_height))
        n_height = int(target_max_size * original_height / max(original_width, original_height))
    else:
        n_width = original_width
        n_height = original_height
    vr = VideoReader(
        video_path,
        ctx=cpu(0),
        num_threads=4,
        width=n_width,
        height=n_height
    )
    total_frame_num = len(vr)
    if total_frame_num > max_frames:
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, max_frames, dtype=int
        )
        frame_idx = uniform_sampled_frames.tolist()
    else:
        frame_idx = [i for i in range(0, total_frame_num)]
    frames = vr.get_batch(frame_idx).asnumpy()

    # Generate sampled video in mp4 format
    if temp_dir is None:
        temp_dir = os.path.join(os.path.dirname(video_path), "temp_video_frames")
    os.makedirs(temp_dir, exist_ok=True)

    out_video_path = os.path.join(temp_dir, os.path.basename(video_path).replace('.mp4', '_processed.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if len(frames) > 0:
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(out_video_path, fourcc, 15, (w, h))
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
    return out_video_path

class VideoInference:
    def __init__(self, config):
        self.frames = config['frames']
        self.max_pixels = config['max_pixels']
        self.batch_size = config.get('batch_size', 8)
        self.device = config.get('device', 'cuda')
        self.model_path = config['model_path']
        self.ability_json_dir = config['input_dir']
        self.output_dir = config['output_dir']
        self.video_dir = config['video_dir']
        self.temp_dir = config.get('temp_dir', './temp_frames')
        self.abilities = config.get('abilities', ABILITY_0 + ABILITY_1)
        self.model_name = self.model_path.split('/')[-1]
        self.max_new_tokens = 1024 if "Instruct" in self.model_name else 8196
        self.processed_video_map = dict()
        self._setup_logging()
        self._load_transformers_model()
        os.makedirs(self.output_dir, exist_ok=True)

    def _setup_logging(self):
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join('logs', f'infer_{timestamp}.log')
        logging.basicConfig(
            filename=log_file,
            filemode='w',
            level=logging.INFO,
            format="%(asctime)s - [%(levelname)s] - %(message)s"
        )
        self.logger = logging.getLogger("InferLogger")

    def _load_transformers_model(self):
        self.logger.info(f"Loading transformers model: {self.model_path}")
        from transformers import AutoModelForImageTextToText, AutoProcessor
        import torch
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            use_fast=True,
            trust_remote_code=True,
        )
        self.processor.tokenizer.padding_side = 'left'

    def _get_sorted_files(self):
        files = [f for f in os.listdir(self.ability_json_dir) if f.endswith('.json')]
        files.sort()
        return files

    def run(self):
        input_files = self._get_sorted_files()
        for json_file in input_files:
            ability_name = os.path.splitext(json_file)[0]
            if ability_name not in self.abilities:
                self.logger.info(f"Skipping unselected ability: {ability_name}")
                continue
            input_path = os.path.join(self.ability_json_dir, json_file)
            output_path = os.path.join(self.output_dir, f"{self.model_name}_{self.frames}_{self.max_pixels}", json_file)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.process_file(input_path, output_path, ability_name)

    def process_file(self, input_path, output_path, ability_name):
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            with open(output_path, "r", encoding='utf-8') as g:
                model_answers = json.load(g)
        else:
            model_answers = {}

        # -------- Process all videos once to generate sampled mp4s and a reuse map --------
        with open(input_path, "r", encoding='utf-8') as f:
            eval_datasets = json.load(f)
            video_names = list(eval_datasets.keys())
        need_encode = []
        for video_name in video_names:
            video_path = os.path.join(self.video_dir, f"{video_name}.mp4")
            processed_name = os.path.basename(video_path).replace('.mp4', '_processed.mp4')
            processed_path = os.path.join(self.temp_dir, processed_name)
            if os.path.exists(processed_path):
                self.processed_video_map[video_name] = processed_path
            else:
                need_encode.append((video_name, video_path, processed_path))
        
        # Encoding videos with progress bar displayed
        for video_name, video_path, processed_path in tqdm(need_encode, desc="encoding videos"):
            try:
                out_path = encode_video(video_path, max_pixels=self.max_pixels, max_frames=self.frames, temp_dir=self.temp_dir)
                self.processed_video_map[video_name] = out_path
            except Exception as e:
                self.logger.error(f"Encoding failed, {video_name}: {e}")
                self.processed_video_map[video_name] = None
        
        # -------- Collect all entries that require batch inference or filling --------
        batch_entries = []
        with open(input_path, "r", encoding='utf-8') as f:
            eval_datasets = json.load(f)
            for video_name, mtqa_list in eval_datasets.items():
                if video_name not in self.processed_video_map or self.processed_video_map[video_name] is None:
                    continue
                if video_name not in model_answers:
                    model_answers[video_name] = {}
                if video_name in model_answers and len(model_answers[video_name]) == len(mtqa_list):
                    self.logger.info(f"Skipping completed video: {video_name}")
                    continue
                processed_video = self.processed_video_map[video_name]
                for mtqa_dict in mtqa_list:
                    event_index = mtqa_dict["event_index"]
                    mtqa = mtqa_dict["mtqa"]
                    if event_index in model_answers[video_name]:
                        self.logger.info(f"Skipping completed event: {video_name} - {event_index}")
                        continue
                    
                    dialogue_history = []
                    for round_idx, round_data in mtqa.items():
                        question, answer = round_data["User"], round_data["Assistant"]
                        # Ensure event_index can append multiple answers automatically
                        if event_index not in model_answers[video_name]:
                            model_answers[video_name][event_index] = []

                        if (ability_name in ABILITY_1) or (round_data["Ability"] in ABILITY_LIST):
                            PROMPT = (
                                f"You're an AI assistant helping to answer questions about a video. Use the conversation history if helpful.\n"
                                f"Conversation history:\n"
                                f"{dialogue_history}\n"
                                f"Current question:\n"
                                f"{question}"
                            )
                            batch_entries.append({
                                "video_name": video_name,
                                "event_index": event_index,
                                "prompt": PROMPT,
                                "video_path": processed_video,
                                "question": question,
                                "ability_name": ability_name,
                                "dialogue_history": list(dialogue_history),
                                "need_infer": True,
                                "default_response": None
                            })
                        else:
                            # No inference required: directly fill as needed
                            batch_entries.append({
                                "video_name": video_name,
                                "event_index": event_index,
                                "prompt": None,
                                "video_path": None,
                                "question": question,
                                "ability_name": ability_name,
                                "dialogue_history": list(dialogue_history),
                                "need_infer": False,
                                "default_response": "No need for this ability to answer."
                            })
                        dialogue_history.append(f"User: {question}")
                        dialogue_history.append(f"Assistant: {answer}")

        # -------- Perform batch inference and save results --------
        pbar = tqdm(
            range(0, len(batch_entries), self.batch_size),
            desc=f"{ability_name} batch inference",
            position=0
        )
        for batch_start in pbar:
            batch = batch_entries[batch_start: batch_start + self.batch_size]
            # Split entries into inference-required and non-inference-required
            infer_indices = [i for i, entry in enumerate(batch) if entry["need_infer"]]
            infer_entries  = [entry for entry in batch if entry["need_infer"]]
            noninfer_indices = [i for i, entry in enumerate(batch) if not entry["need_infer"]]
            noninfer_entries = [entry for entry in batch if not entry["need_infer"]]

            # Prepare Qwen3-VL format batch_messages (only inference-required entries)
            batch_messages = []
            for entry in infer_entries:
                video_path = entry['video_path']
                question = entry['prompt']
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "video",
                                "video": video_path,
                                # "fps": 15,   # Customize if needed via self.fps.
                                "min_pixels": 4 * 32 * 32,
                                "max_pixels": 256 * 32 * 32,
                                "total_pixels": 16384 * 32 * 32,
                            },
                        ],
                    },
                ]
                batch_messages.append(messages)

            # Perform batch inference on infer_entries
            infer_responses = []
            if batch_messages:
                try:
                    with torch.no_grad():
                        inputs = self.processor.apply_chat_template(
                            batch_messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_dict=True,
                            return_tensors="pt",
                            # fps=2,
                            padding=True
                        )
                        inputs = inputs.to(self.model.device)
                        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        infer_responses = self.processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                except Exception as e:
                    self.logger.error(f"Batch processing failed: {e}")
                    infer_responses = ["[ERROR] " + str(e) for _ in infer_entries]

            # Save results in the original order with interleaving
            idx_infer, idx_noninfer = 0, 0
            for entry_idx, entry in enumerate(batch):
                vname = entry['video_name']
                eidx = entry['event_index']
                if entry["need_infer"]:
                    response = infer_responses[idx_infer]
                    if '</think>' in response:
                        response = response.split('</think>', 1)[-1].strip()
                    idx_infer += 1
                else:
                    response = entry["default_response"]
                    idx_noninfer += 1
                model_answers[vname][eidx].append(response)

            # Output (dump everything at the end or periodically)
            with open(output_path, "w", encoding='utf-8') as g:
                json.dump(model_answers, g, ensure_ascii=False, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--frames', type=int, default=128)
    parser.add_argument('--max_pixels', type=int, default=720)
    parser.add_argument('--video_dir', type=str, default='')
    parser.add_argument('--input_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--abilities', nargs='+', default=[], help='ability list')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--temp_dir', type=str, default='', help='store processed videos')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = {
        'frames': args.frames,
        'max_pixels': args.max_pixels,
        'video_dir': args.video_dir,
        'input_dir': args.input_dir,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'device': args.device,
        'model_path': args.model_path,
        'temp_dir': args.temp_dir,
        'abilities': args.abilities
    }
    processor = VideoInference(config)
    processor.run()