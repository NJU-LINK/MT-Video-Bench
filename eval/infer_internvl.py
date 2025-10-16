from transformers import AutoModel, AutoTokenizer
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import os
import json
import logging
from io import BytesIO
from openai import OpenAI
from tqdm import tqdm
import cv2
import time
import argparse
from decord import VideoReader, cpu

ABILITY_LIST=[
    'Summary', 'Object Reference', 'Memory Recall', 'Answer Refusal', 'Instruction Clarification'
]
ABILITY_0=[
    'summary', 'object_reference', 'memory_recall', 'answer_refusal', 'instruction_clarification'
]
ABILITY_1 = [
    'proactive_interaction', 'topic_shifting'
]


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
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


def generate_model_answers_internvl(input_dir, output_dir, video_folder_dir=None, ability_name=None, args=None):
    assert ability_name in (ABILITY_0 + ABILITY_1)
    flag = 0 if ability_name in ABILITY_0 else 1

    if os.path.exists(output_dir) and os.path.getsize(output_dir) > 0:
        with open(output_dir, "r", encoding='utf-8') as g_existing:
            model_answers = json.load(g_existing)
    else:
        model_answers = {}

    with open(input_dir, "r", encoding='utf-8') as f:
        reference_qa = json.load(f)
        for video_name, multi_events_qa in reference_qa.items():
            if video_name in model_answers and len(multi_events_qa) == len(model_answers[video_name]):
                continue
            if video_name not in model_answers:
                model_answers[video_name] = {}

            video_file_path = os.path.join(video_folder_dir, f"{video_name}.mp4")
            
            for index, single_event_qa in enumerate(multi_events_qa):
                event_index = single_event_qa["event_index"]
                if f"event_{event_index}" in model_answers[video_name]:
                    continue
                
                model_single_event_qa = []
                all_history = []
                ability_indexs = []
                for i in range(len(single_event_qa)):
                    if (f"Round {i + 1}") not in single_event_qa:
                        continue
                    if flag == 1:
                        ability_indexs.append(i + 1)
                    else:
                        if single_event_qa[f"Round {i + 1}"]["Ability"] in ABILITY_LIST:
                            ability_indexs.append(i + 1)
                    for name, words in single_event_qa[f"Round {i + 1}"].items():
                        if name[0] == "Q" or 'User' in name:
                            sentence = f"User: {words}"
                            all_history.append(sentence)
                        elif name != "Ability" and (name[0] == "A" or "Bot" in name):
                            sentence = f"Assistant: {words}"
                            all_history.append(sentence)

                for i in range(len(single_event_qa)):
                    if (f"Round {i + 1}") not in single_event_qa:
                        continue

                    if i + 1 in ability_indexs:
                        history = all_history[: i * 2]
                        question = all_history[i * 2]
                        HISTORY = []
                        for his_idx in range(0, len(history), 2):
                            HISTORY.append((history[his_idx], history[his_idx + 1]))

                        # prepare prompt
                        if args.think:
                            R1_SYSTEM_PROMPT = """
                            You are an AI assistant that rigorously follows this response protocol:

                            1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

                            2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

                            Ensure that the thinking process is thorough but concise, remaining focused on the query.  The final answer should be standalone and not reference the thinking section.
                            """.strip()

                            args.model.system_message = R1_SYSTEM_PROMPT
                        
                        PROMPT = f"""
                        You are an AI assistant designed to answer questions about the video.\n
                        Now, answer the following question, taking into account the conversation history:\n
                        {question}
                        """
                        print(f"  **PROMPT** : {PROMPT}")

                        pixel_values, num_patches_list = load_video(video_file_path, input_size=args.max_pixels, num_segments=args.frames, max_num=1, get_frame_by_duration=False)
                        pixel_values = pixel_values.to(torch.bfloat16).to(args.model.device)
                        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])

                        QUESTION = video_prefix + PROMPT
                        response, chat_history = args.model.chat(args.tokenizer, pixel_values, QUESTION, args.generation_config, num_patches_list=num_patches_list, history=HISTORY, return_history=True)
                        if args.think:
                            if '</think>' in response:
                                response = response.split('</think>', 1)[-1].strip()
                            else:
                                response = response.strip()
                        
                        model_answer = response
                        print(f"model_answer: {model_answer}")
                    else:
                        model_answer = "No need for this ability to answer."

                    model_single_event_qa.append(model_answer)

                model_answers[video_name][f"event_{event_index}"] = model_single_event_qa
                with open(output_dir, "w", encoding='utf-8') as g:
                    json.dump(model_answers, g, ensure_ascii=False, indent=4)


def generate_pipeline(input_dir, output_dir, video_folder_dir=None, args=None):
    for ability_name_json_path in sorted(os.listdir(input_dir)):
        ability_name = ability_name_json_path.split('.')[0]
        # if args.ability != ability_name:
        #     continue
        print(f"======Processing {ability_name_json_path}======")
        ability_name_full_path = os.path.join(input_dir, ability_name_json_path)
        result_full_path = os.path.join(output_dir, ability_name_json_path)
        generate_model_answers_internvl(ability_name_full_path, result_full_path, video_folder_dir, ability_name, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate model outputs")
    parser.add_argument('--model_type', type=str, default='internvl4b', help='Type of the model to evaluate')
    parser.add_argument('--frames', type=int, default=32)
    parser.add_argument('--max_pixels', type=int, default=448)
    parser.add_argument('--think', type=bool, default=False)
    parser.add_argument('--ability', type=str, default="", help='topic_shifting, proactive_interaction, summary, object_reference, memory_recall, answer_refusal')
    args = parser.parse_args()
    
    model_path_list = {
        "internvl4b": "InternVL3_5-4B",
        "internvl8b": "InternVL3_5-8B",
        "internvl38b": "InternVL3_5-38B"
    }
    model_path = model_path_list[args.model_type]
    args.model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto").eval()
    args.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    if args.think:
        args.model_type = args.model_type + "_thinking"
        args.generation_config = dict(max_new_tokens=8192, do_sample=True, temperature=0.6, top_p=0.001, repetition_penalty=1.05)
    else:
        args.model_type = args.model_type + "_nothinking"
        args.generation_config = dict(max_new_tokens=1024, do_sample=True, temperature=0.1, top_p=0.001, repetition_penalty=1.05)

    
    video_folder_dir=f"MT-Video-Bench/videos"
    input_dir=f"MT-Video-Bench/data"
    output_dir=f"./inference_answers/answers_{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    generate_pipeline(input_dir, output_dir, video_folder_dir, args)
    end_time = time.time()
    print("需要的时间:", end_time - start_time)
