import os, argparse
import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from character.utils import gen_args, constitutions
from character.constants import CONSTITUTION_PATH, DATA_PATH, MODEL_PATH


system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""


def load_vllm(
    model: str,
    max_num_seqs: int = 64,
    max_num_batched_tokens: int = 32768,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = -1,
    min_p: float = 0.0,
    tp_size: int = None,
    max_model_len: int = 8192,
    max_new_tokens: int = 4096,
    enable_prefix_caching: bool = True,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.95,
    trust_remote_code: bool = True,
    task: str = "generate",
    quantization: str = None,
    pipeline_parallel_size: int = 1,
    enforce_eager: bool = False,
) -> tuple[argparse.Namespace, LLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        f"{MODEL_PATH}/{model}",
        trust_remote_code=trust_remote_code,
    )

    # === LOAD MODEL ===
    if tp_size is None:
        tp_size = t.cuda.device_count()
    if model == "qwen-2.5-7b-it":
        tp_size = max([d for d in [i for i in range(1, 29) if 28 % i == 0 and i % 2 == 0] if d <= t.cuda.device_count()] + [1])

    args = gen_args(
        model=model, 
        max_num_seqs=max_num_seqs, 
        max_num_batched_tokens=max_num_batched_tokens, 
        temperature=temperature, 
        top_p=top_p, 
        top_k=top_k, 
        min_p=min_p, 
        tp_size=tp_size, 
        max_model_len=max_model_len, 
        max_new_tokens=max_new_tokens,
        enable_prefix_caching=enable_prefix_caching,
    )
    llm_kwargs = {
        "model": args.model,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": args.tp_size,
        "trust_remote_code": trust_remote_code,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": args.enable_prefix_caching,
        "pipeline_parallel_size": pipeline_parallel_size,
    }
    if quantization:
        llm_kwargs["quantization"] = quantization
    if enforce_eager:
        llm_kwargs["enforce_eager"] = True
    llm = LLM(**llm_kwargs)
    return args, llm, tokenizer

# chosen responses role-play the constitution using the teacher model
def roleplay(
    model: str,
    outpath: str,
    args: argparse.Namespace,
    llm: LLM,
    tokenizer: AutoTokenizer,
    constitution: str,
    K: int|None,
) -> None:

    # === LOAD CONSTITUTION ===
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    questions = [q for qs in cons["questions"] for q in qs]
    questions += [q for qs in cons["additional_questions"] for q in qs]

    # === LOAD ADDITIONAL PROMPTS FROM LIMA (optional) ===
    lima_train_path = f"{MODEL_PATH}/lima/train.jsonl"
    lima_test_path = f"{MODEL_PATH}/lima/test.jsonl"
    if os.path.exists(lima_train_path) and os.path.exists(lima_test_path):
        lima_train = pd.read_json(lima_train_path, orient="records", lines=True)
        lima_test = pd.read_json(lima_test_path, orient="records", lines=True)
        # ignoring multi-turn
        questions += [cs[0] for cs in lima_train["conversations"]]
        questions += [cs[0] for cs in lima_test["conversations"]]
    else:
        print(f"LIMA not found at {MODEL_PATH}/lima — using constitution prompts only")

    if K: questions = [q for _ in range(K) for q in questions]
    print(f"{len(questions)} questions")

    # === PROMPTS IN CHATML FORMAT ===
    name = model.split("-")[0].capitalize()
    if name == "Glm": name = "ChatGLM"
    print(f"using {name} as the assistant name")
    trait_string = [f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())]
    trait_string = "\n".join(trait_string)
    system_prompt = system.format(NAME=name, TRAITS=trait_string)
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q}
        ]
        for q in questions
    ]

    # === APPLY CHAT TEMPLATE ===
    prompts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # prefill thinking to enforce adherence to character traits
    for idx in range(len(prompts)):
        prompts[idx] += f"\n<think>I want to ensure my response aligns with my character traits and furthers my goals. They are:\n{trait_string}\n"

    # === GENERATE RESPONSES ===
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=None,
        max_tokens=args.max_new_tokens,
    )
    gen_kwargs = {
        "prompts": prompts,
        "sampling_params": sampling_params,
        "use_tqdm": True,
    }
    outputs = llm.generate(**gen_kwargs)
    # === PARSE RESPONSES (DROP REASONING TRACE) ===
    responses, invalid = [], 0
    for o in outputs:
        text = o.outputs[0].text.strip()
        if "</think>" in text:
            responses.append(text.split("</think>")[1].strip())
        else:
            responses.append(None)
            invalid += 1
    print(f"{invalid} invalid initial responses")

    # === SAVE RESPONSES ===
    results = pd.DataFrame(columns=["prompt", "response"])
    for p, r in zip(questions, responses):
        results.loc[len(results)] = [p, r]
    results.to_json(outpath, orient="records", lines=True)

def main(
    model: str,
    constitution: str,
    K: int|None,
    quantization: str = None,
    tp_size: int = None,
    pipeline_parallel_size: int = 1,
    enforce_eager: bool = False,
    max_num_seqs: int = 64,
    max_model_len: int = 8192,
) -> None:
    args, llm, tokenizer = load_vllm(
        model,
        enable_prefix_caching = False,
        quantization = quantization,
        tp_size = tp_size,
        pipeline_parallel_size = pipeline_parallel_size,
        enforce_eager = enforce_eager,
        max_num_seqs = max_num_seqs,
        max_model_len = max_model_len,
    )
    cons = constitutions if constitution == "all" else [constitution]
    for cons in cons:
        outpath = f"{DATA_PATH}/distillation/{cons}.jsonl"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        if os.path.exists(outpath):
            print(f"teacher responses at {outpath} already exist")
            continue
        roleplay(model, outpath, args, llm, tokenizer, cons, K)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, default="glm-4.5-air")
    parser.add_argument("--constitution", type=str, required=False, default="all")
    parser.add_argument("--K", type=int, required=False, default=5)
    parser.add_argument("--quantization", type=str, required=False, default=None,
                        help="vLLM quantization, e.g. 'fp8' to halve VRAM for large teachers")
    parser.add_argument("--tensor-parallel-size", type=int, required=False, default=None,
                        help="vLLM tensor_parallel_size (default: all GPUs); must divide the head count")
    parser.add_argument("--pipeline-parallel-size", type=int, required=False, default=1,
                        help="vLLM pipeline_parallel_size; use tp*pp = num_gpus for large models")
    parser.add_argument("--enforce-eager", action="store_true", default=False,
                        help="skip torch.compile/CUDA graphs (saves VRAM + startup time for huge models)")
    parser.add_argument("--max-num-seqs", type=int, required=False, default=64,
                        help="max concurrent sequences; lower it when KV-cache memory is tight")
    parser.add_argument("--max-model-len", type=int, required=False, default=8192,
                        help="max context length; lower it to shrink KV-cache memory")
    args = parser.parse_args()
    main(args.model, args.constitution, args.K, args.quantization,
         args.tensor_parallel_size, args.pipeline_parallel_size, args.enforce_eager,
         args.max_num_seqs, args.max_model_len)