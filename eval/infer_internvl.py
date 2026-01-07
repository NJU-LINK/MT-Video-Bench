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
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

ABILITY_LIST = [
    'Summary', 'Object Reference', 'Memory Recall', 'Answer Refusal'
]
ABILITY_0 = [
    'summary', 'object_reference', 'memory_recall', 'answer_refusal'
]
ABILITY_1 = [
    'proactive_interaction', 'topic_shifting'
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), 
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), 
        T.ToTensor(), 
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices


def get_num_frames_by_duration(duration):
        local_num_frames = 4        
        num_segments = int(duration // local_num_frames)
        if num_segments == 0:
            num_frames = local_num_frames
        else:
            num_frames = local_num_frames * num_segments
        
        num_frames = min(512, num_frames)
        num_frames = max(128, num_frames)

        return num_frames


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, get_frame_by_duration = False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    if get_frame_by_duration:
        duration = max_frame / fps
        num_segments = get_num_frames_by_duration(duration)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

class VideoInference:
    def __init__(self, config):
        self.frames = config['frames']
        self.max_pixels = config['max_pixels']
        self.device = config.get('device', 'cuda')
        self.model_path = config['model_path']
        self.ability_json_dir = config['input_dir']
        self.output_dir = config['output_dir']
        self.video_dir = config['video_dir']
        self.think = config.get('think', False)
        self.abilities = config.get('abilities', ABILITY_0 + ABILITY_1)
        self.model_name = self.model_path.split('/')[-1]
        self.processed_video_map = dict()
        self._setup_logging()
        self._load_transformers_model()
        os.makedirs(self.output_dir, exist_ok=True)

    def _setup_logging(self):
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join('logs', f'encoder_{timestamp}.log')
        logging.basicConfig(
            filename=log_file,
            filemode='w',
            level=logging.INFO,
            format="%(asctime)s - [%(levelname)s] - %(message)s"
        )
        self.logger = logging.getLogger("EncoderLogger")

    def _load_transformers_model(self):
        self.logger.info(f"load transformers model: {self.model_path}")
        from transformers import AutoModel, AutoTokenizer
        import torch
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)
        if self.think:
            self.model_name = self.model_name + "_think"
            self.generation_config = dict(max_new_tokens=8192, do_sample=True, temperature=0.6, top_p=0.001, repetition_penalty=1.05)
        else:
            self.model_name = self.model_name + "_nothink"
            self.generation_config = dict(max_new_tokens=1024, do_sample=True, temperature=0.1, top_p=0.001, repetition_penalty=1.05)

        if self.think:
            R1_SYSTEM_PROMPT = """
            You are an AI assistant that rigorously follows this response protocol:

            1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

            2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

            Ensure that the thinking process is thorough but concise, remaining focused on the query.  The final answer should be standalone and not reference the thinking section.
            """.strip()

            self.model.system_message = R1_SYSTEM_PROMPT

    def _get_sorted_files(self):
        files = [f for f in os.listdir(self.ability_json_dir) if f.endswith('.json')]
        files.sort()
        return files

    def run(self):
        input_files = self._get_sorted_files()
        for json_file in input_files:
            ability_name = os.path.splitext(json_file)[0]
            if ability_name not in self.abilities:
                self.logger.info(f"skip: {ability_name}")
                continue
            input_path = os.path.join(self.ability_json_dir, json_file)
            output_path = os.path.join(self.output_dir, self.model_name, json_file)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.process_file(input_path, output_path, ability_name)

    def process_file(self, input_path, output_path, ability_name):
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            with open(output_path, "r", encoding='utf-8') as g:
                model_answers = json.load(g)
        else:
            model_answers = {}
        
        # -------- Start to inference --------
        with open(input_path, "r", encoding='utf-8') as f:
            eval_datasets = json.load(f)
            for video_name, mtqa_list in eval_datasets.items():
                if video_name not in model_answers:
                    model_answers[video_name] = {}
                if video_name in model_answers and len(model_answers[video_name]) == len(mtqa_list):
                    self.logger.info(f"skip: {video_name}")
                    continue

                video_path = os.path.join(self.video_dir, f"{video_name}.mp4")
                for mtqa_dict in mtqa_list:
                    event_index = mtqa_dict["event_index"]
                    mtqa = mtqa_dict["mtqa"]
                    if event_index in model_answers[video_name]:
                        self.logger.info(f"skip: {video_name} - {event_index}")
                        continue

                    dialogue_history = []
                    for round_idx, round_data in mtqa.items():
                        question, answer = round_data["User"], round_data["Assistant"]

                        if event_index not in model_answers[video_name]:
                            model_answers[video_name][event_index] = []

                        if (ability_name in ABILITY_1) or (round_data["Ability"] in ABILITY_LIST):
                            PROMPT = (
                                f"You're an AI assistant helping to answer questions about a video. Use the conversation history if helpful.\n"
                                f"Current question:\n"
                                f"{question}"
                            )
                            HISTORY = []
                            for his_idx in range(0, len(dialogue_history), 2):
                                HISTORY.append((dialogue_history[his_idx], dialogue_history[his_idx + 1]))
                            
                            pixel_values, num_patches_list = load_video(video_path, input_size=self.max_pixels, num_segments=self.frames, max_num=1, get_frame_by_duration=False)
                            pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)
                            video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
                            question = video_prefix + PROMPT
                            try:
                                with torch.no_grad():
                                    response = self.model.chat(
                                        self.tokenizer, 
                                        pixel_values, 
                                        question, 
                                        self.generation_config, 
                                        num_patches_list=num_patches_list, 
                                        history=HISTORY, 
                                        # return_history=True
                                    )
                                    # extract result
                                    if self.think:
                                        if '</think>' in response:
                                            response = response.split('</think>', 1)[-1].strip()
                                        else:
                                            response = response.strip()
                            except Exception as e:
                                self.logger.error(f"failed: {e}")
                                response = "[ERROR] " + str(e)
                        else:
                            response = "No need for this ability to answer."
                        
                        model_answers[video_name][event_index].append(response)
                        
                        with open(output_path, "w", encoding='utf-8') as g:
                            json.dump(model_answers, g, ensure_ascii=False, indent=4)
                        
                        dialogue_history.append(f"User: {question}")
                        dialogue_history.append(f"Assistant: {answer}")


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model_path', type=str, default='InternVL3_5-8B')
    parser.add_argument('--frames', type=int, default=128)
    parser.add_argument('--max_pixels', type=int, default=720)
    parser.add_argument('--video_dir', type=str, default='videos')
    parser.add_argument('--input_dir', type=str, default='mtqa')
    parser.add_argument('--output_dir', type=str, default='./output/model_answers')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--think', type=bool, default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = {
        'frames': args.frames,
        'max_pixels': args.max_pixels,
        'video_dir': args.video_dir,
        'input_dir': args.input_dir,
        'output_dir': args.output_dir,
        'device': args.device,
        'model_path': args.model_path,
        'think': args.think
    }
    processor = VideoInference(config)
    processor.run()