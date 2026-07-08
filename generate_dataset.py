import os, sys, json, time, random, re
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

N_SAMPLES = 3
DATASET_TYPES = ["coldstart_math", "coldstart_logic", "rlvr_math", "rlvr_logic"]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")
MAX_RETRIES = 3

PROVIDERS = [
    {
        "name": "qwen3.7-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://ws-cucfwl9482fa12aa.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.7-plus",
        "thinking_coldstart": {"enable_thinking": True},
        "thinking_rlvr": {"enable_thinking": False},
    },
    {
        "name": "deepseek-v4-pro",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-pro",
        "thinking_coldstart": {"thinking": {"type": "enabled"}},
        "thinking_rlvr": {"thinking": {"type": "disabled"}},
    },
    {
        "name": "deepseek-v4-flash",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "thinking_coldstart": {"thinking": {"type": "enabled"}},
        "thinking_rlvr": {"thinking": {"type": "disabled"}},
    },
]

PROMPTS = {
    "coldstart_math": """أنشئ مسألة رياضية عربية أصلية (ليست مترجمة) مناسبة لتعليم نموذج لغوي كيفية التفكير خطوة بخطوة.
أخرج النتيجة بصيغة JSON فقط بالمفاتيح التالية:
id, domain ("math"), difficulty ("easy" أو "medium" أو "hard"), prompt, response.
يجب أن يحتوي response على منطق التفكير بالكامل داخل <think>...</think> ثم الإجابة النهائية فقط داخل <answer>...</answer>.
لا تكرر أي مسألة كلاسيكية معروفة، واجعل السياق (الأسماء، الأرقام، الموضوع) متنوعًا وواقعيًا.""",

    "coldstart_logic": """أنشئ لغزًا منطقيًا عربيًا أصليًا (مثل الصادق/الكاذب أو ترتيب المنازل) مناسبًا لتعليم نموذج لغوي التفكير المنطقي التدريجي.
أخرج النتيجة بصيغة JSON فقط بالمفاتيح التالية:
id, domain ("logic"), difficulty, prompt, response.
يجب أن يحتوي response على <think>...</think> يوضح كل خطوة استدلال منطقي، ثم <answer>...</answer> يحتوي فقط على الحل النهائي.
تأكد أن اللغز له حل منطقي واحد فقط لا لبس فيه.""",

    "rlvr_math": """أنشئ مسألة رياضية عربية أصلية بدون أي شرح أو تفكير مكتوب.
أخرج النتيجة بصيغة JSON فقط بالمفاتيح التالية:
id, domain ("math"), difficulty_tag ("easy" أو "hard"), prompt, ground_truth_answer.
يجب أن يكون ground_truth_answer قيمة رقمية دقيقة وحيدة يمكن التحقق منها برمجيًا.""",

    "rlvr_logic": """أنشئ لغزًا منطقيًا عربيًا أصليًا (صادق/كاذب أو شبكة منازل) بدون أي شرح أو تفكير مكتوب.
أخرج النتيجة بصيغة JSON فقط بالمفاتيح التالية:
id, domain ("logic"), difficulty_tag ("easy" أو "hard"), puzzle_type, prompt, ground_truth_answer.
يجب أن يكون ground_truth_answer كائن JSON يحدد الحل الكامل لكل شخص/كيان في اللغز، وأن يكون للغز حل واحد فقط.""",
}


def call_model(client, model, prompt, dataset_type, thinking_coldstart, thinking_rlvr):
    is_coldstart = dataset_type.startswith("coldstart")
    extra_body = thinking_coldstart if is_coldstart else thinking_rlvr
    max_tokens = 16384 if is_coldstart else 4096
    kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
    if extra_body:
        kwargs["extra_body"] = extra_body
    if not is_coldstart:
        kwargs["temperature"] = 0.0
    t0 = time.time()
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or "", time.time() - t0, resp.usage


def generate():
    for provider in PROVIDERS:
        api_key = os.environ.get(provider["api_key_env"])
        if not api_key:
            print(f"[SKIP] {provider['name']}: missing {provider['api_key_env']}")
            continue
        client = OpenAI(api_key=api_key, base_url=provider["base_url"])
        for ds_type in DATASET_TYPES:
            base_prompt = PROMPTS[ds_type]
            for i in range(N_SAMPLES):
                seed = random.randint(1, 2**31 - 1)
                prompt = base_prompt + f"\n\n[تلميح التباين {seed}: استخدم أسماء وأرقامًا وسياقات مختلفة عن النماذج السابقة.]"
                for attempt in range(MAX_RETRIES):
                    try:
                        text, elapsed, usage = call_model(
                            client, provider["model"], prompt, ds_type,
                            provider["thinking_coldstart"], provider["thinking_rlvr"],
                        )
                        break
                    except Exception as e:
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(2 ** attempt)
                        else:
                            raise e
                dirpath = os.path.join(OUTPUT_DIR, provider["name"], ds_type)
                os.makedirs(dirpath, exist_ok=True)
                filepath = os.path.join(dirpath, f"sample_{i}.txt")
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(text)
                tokens = usage.prompt_tokens + usage.completion_tokens if usage else 0
                print(f"[{provider['name']}] {ds_type} sample {i}: {elapsed:.1f}s, {tokens} tk  ->  {filepath}", flush=True)

    print(f"\nDone. {N_SAMPLES * len(DATASET_TYPES) * len(PROVIDERS)} samples in {OUTPUT_DIR}/")


if __name__ == "__main__":
    generate()
