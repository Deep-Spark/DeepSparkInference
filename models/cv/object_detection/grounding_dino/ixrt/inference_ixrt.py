import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import Any, Dict, List
import transforms as T
from common import create_engine_context, get_io_bindings
from transformers import AutoTokenizer
import cuda.cudart as cudart
import tensorrt
from load_ixrt_plugin import load_ixrt_plugin
load_ixrt_plugin()

def get_phrases_from_posmap(
    posmap: torch.BoolTensor, tokenized: Dict, tokenizer: AutoTokenizer, left_idx: int = 0, right_idx: int = 255
):
    assert isinstance(posmap, torch.Tensor), "posmap must be torch.Tensor"
    if posmap.dim() == 1:
        posmap[0: left_idx + 1] = False
        posmap[right_idx:] = False
        non_zero_idx = posmap.nonzero(as_tuple=True)[0].tolist()
        token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
        return tokenizer.decode(token_ids)
    else:
        raise NotImplementedError("posmap must be 1-dim")

def create_positive_map_from_span(tokenized, token_span, max_text_len=256):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j
    Input:
        - tokenized:
            - input_ids: Tensor[1, ntokens]
            - attention_mask: Tensor[1, ntokens]
        - token_span: list with length num_boxes.
            - each item: [start_idx, end_idx]
    """
    positive_map = torch.zeros((len(token_span), max_text_len), dtype=torch.float)
    for j, tok_list in enumerate(token_span):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            if os.environ.get("SHILONG_DEBUG_ONLY_ONE_POS", None) == "TRUE":
                positive_map[j, beg_pos] = 1
                break
            else:
                positive_map[j, beg_pos : end_pos + 1].fill_(1)

    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)

def generate_masks_with_special_tokens_and_transfer_map(tokenized, special_tokens_list, tokenizer):
    """Generate attention mask between each pair of special tokens
    Args:
        input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.
    Returns:
        torch.Tensor: attention mask between each special tokens.
    """
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((bs, num_token), device=input_ids.device).bool()
    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = (
        torch.eye(num_token, device=input_ids.device).bool().unsqueeze(0).repeat(bs, 1, 1)
    )
    position_ids = torch.zeros((bs, num_token), device=input_ids.device)
    cate_to_token_mask_list = [[] for _ in range(bs)]
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device
            )
            c2t_maski = torch.zeros((num_token), device=input_ids.device).bool()
            c2t_maski[previous_col + 1 : col] = True
            cate_to_token_mask_list[row].append(c2t_maski)
        previous_col = col

    cate_to_token_mask_list = [
        torch.stack(cate_to_token_mask_listi, dim=0)
        for cate_to_token_mask_listi in cate_to_token_mask_list
    ]

    # # padding mask
    # padding_mask = tokenized['attention_mask']
    # attention_mask = attention_mask & padding_mask.unsqueeze(1).bool() & padding_mask.unsqueeze(2).bool()

    return attention_mask, position_ids.to(torch.long), cate_to_token_mask_list


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([[1200, 800]]), # w, h,  max_size=1333
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image



def sig(x):
    return 1/(1 + np.exp(-x))

def get_grounding_output(test_mode, fps_target, torken_path, model_path, max_text_len, image_pil, image, caption, box_threshold, output_dir, text_threshold=None, with_logits=True, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
        
    tokenizer = AutoTokenizer.from_pretrained(torken_path)
    captions = [caption]
    # encoder texts
    tokenized = tokenizer(captions, padding="max_length", max_length=128,return_tensors="pt")
    specical_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    
    (
        text_self_attention_masks,
        position_ids,
        cate_to_token_mask_list,
    ) = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, specical_tokens, tokenizer)

    if text_self_attention_masks.shape[1] > max_text_len:
        text_self_attention_masks = text_self_attention_masks[
            :, : max_text_len, : max_text_len]
        
        position_ids = position_ids[:, : max_text_len]
        tokenized["input_ids"] = tokenized["input_ids"][:, : max_text_len]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, : max_text_len]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : max_text_len]

    inputs = {}
    input_img = np.expand_dims(image, 0)
    inputs["img"] = input_img
    inputs["input_ids"] = tokenized["input_ids"].numpy()
    inputs["attention_mask"] = tokenized["attention_mask"].bool().numpy()
    inputs["token_type_ids"] = tokenized["token_type_ids"].numpy()
    inputs["position_ids"] = position_ids.numpy()
    inputs["text_token_mask"] = text_self_attention_masks.numpy()
    
    host_mem = tensorrt.IHostMemory
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)
    engine, context = create_engine_context(model_path, logger)
    ixrt_inputs, outputs, allocations = get_io_bindings(engine)

    for ixrt_input in ixrt_inputs:
        input_name = ixrt_input["name"]
        (err,) = cudart.cudaMemcpy(
            ixrt_input["allocation"],
            inputs[input_name],
            ixrt_input["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        assert err == cudart.cudaError_t.cudaSuccess

    if test_mode == "FPS":
        # warm up
        for i in range(5):
            context.execute_v2(allocations)
        
        cudart.cudaDeviceSynchronize()
        start_time = time.time()
        for i in range(20):
            context.execute_v2(allocations)
        cudart.cudaDeviceSynchronize()
        end_time = time.time()
        fps = 20.0/(end_time-start_time)

        print(f"Performance Check : Test {fps} >= target {fps_target}")
        if fps >= fps_target:
            print("pass!")
            exit()
        else:
            print("failed!")
            exit(1)


    cudart.cudaDeviceSynchronize()
    context.execute_v2(allocations)
    cudart.cudaDeviceSynchronize()

    output_logits = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
    output_boxes = np.zeros(outputs[1]["shape"], outputs[1]["dtype"])
    (err,) = cudart.cudaMemcpy(
        output_logits,
        outputs[0]["allocation"],
        outputs[0]["nbytes"],
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    )
    assert err == cudart.cudaError_t.cudaSuccess
    (err,) = cudart.cudaMemcpy(
        output_boxes,
        outputs[1]["allocation"],
        outputs[1]["nbytes"],
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    )
    assert err == cudart.cudaError_t.cudaSuccess

    print("output_logits:",output_logits.shape,outputs[0]["nbytes"])
    print("output_boxes:",output_boxes.shape,outputs[1]["nbytes"])
    prediction_logits_ = np.squeeze(output_logits, 0) #[0]  # prediction_logits.shape = (nq, 256)
    prediction_logits_ = sig(prediction_logits_)
    prediction_boxes_ = np.squeeze(output_boxes, 0) #[0]  # prediction_boxes.shape = (nq, 4)
    logits = torch.from_numpy(prediction_logits_)
    boxes = torch.from_numpy(prediction_boxes_) 
    
    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases

    # visualize pred
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    print("boxes_filt:",boxes_filt.size())
    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    image_with_box.save(os.path.join(output_dir, "pred.jpg"))
    if boxes_filt.size()[0]==8:
        print("pass!")
        exit()
    else:
        print("failed! detect:", boxes_filt.size())
        exit(1)

    return boxes_filt, pred_phrases


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Onnxruntime Grounding DINO example", add_help=True)
    parser.add_argument(
        "--torken_path", type=str, required=True, help="path to tokens file"
    )
    parser.add_argument(
        "--model_path", "-m", type=str, default="./grounded_static_1x800x1200x128_sim.engine", help="path to onnx file"
    )
    parser.add_argument("--image_path", "-i", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", help="output directory"
    )
    parser.add_argument("--test_mode", type=str, default="FPS", help="FPS ACC")
    parser.add_argument("--fps_target", type=float, default=-1.0)

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--max_text_len", type=int, default=256, help="max_text_len")
    parser.add_argument("--token_spans", type=str, default=None, help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")

    parser.add_argument("--device", "-d",  type=str, default="CPU", help="set device, default: CPU")
    args = parser.parse_args()

    # cfg
    model_path = args.model_path  # change the path of the model
    image_path = args.image_path
    torken_path = args.torken_path
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = args.token_spans
    device = args.device
    max_text_len = args.max_text_len
    test_mode = args.test_mode
    fps_target = args.fps_target
    
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)

    # visualize raw image
    # image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # set the text_threshold to None if token_spans is set.
    if token_spans is not None:
        text_threshold = None
        boxes_filt, pred_phrases = get_grounding_output(
        test_mode, fps_target, torken_path, model_path, max_text_len, image_pil, image, text_prompt, box_threshold, output_dir, text_threshold, token_spans=eval(token_spans))
        print("Using token_spans. Set the text_threshold to None.")
    else:
        boxes_filt, pred_phrases = get_grounding_output(
            test_mode, fps_target, torken_path, model_path, max_text_len, image_pil, image, text_prompt, box_threshold, output_dir, text_threshold, token_spans=None)
