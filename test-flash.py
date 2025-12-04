from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
# default: Load the model on the available device(s)
# model = AutoModelForImageTextToText.from_pretrained(
#     "Qwen/Qwen3-VL-235B-A22B-Instruct", dtype="auto", device_map="auto",torch_dtype=torch.bfloat16, 
#     attn_implementation="flash_attention_2",
# )
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", device_map="auto",torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2",
    local_files_only=True,
)
# model.eval() 并不能固定输出

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = AutoModelForImageTextToText.from_pretrained(
#     "Qwen/Qwen3-VL-235B-A22B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "./1.mp4",
            },
            {"type": "text", "text": 
 "你是一个为信息抽取任务生成输入的助手。请先完整理解视频内容，\
 然后用尽可能紧凑的方式，用中文描述视频里出现的事实。要求：\n\
 1. 重点保留【实体】和【实体之间的关系】，而不是高层抽象的概括。\n\
 2. 尽量使用完整的实体名（人名、机构名、书名、课程名、金额、时间等），不要用“他/他们/这本书/这个课程”等代词代替。\n\
 3. 尽量让每个句子都符合“实体A 在时间T 通过动作/关系R 作用于实体B”的结构，例如：\n\
    “杨鹏 在视频中 讲解 东西方文明在建立确定性上的不同路径。”\n\
 4. 可以重复提到同一个实体，只要关系不同就写出来，不要怕啰嗦。\n\
 5. 不要写抒情、点评、个人感受，只写视频中真正出现或明确表达的事实。\n\
 6. 全文控制在不超过500个汉字之内。\n\
 请直接输出若干个这样的事实句子，用换行分隔，不要使用 markdown 标题或项目符号。"},

        ],
    }
]
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "video",
#                 "video": {
#                         "video_path": "./video.mp4",
#                         "fps": 1,
#                         "max_frames": 180,
#                     },
#             },
#             {"type": "text", "text": "Describe this video."},
#         ],
#     }
# ]

# Preparation for inference
inputs = processor.apply_chat_template(  #在此处可以调整采样帧率fps和固定帧数max_frames
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=768)  #调整最终输出长度
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)