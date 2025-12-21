import sentencepiece as spm
from transformers import LlamaForCausalLM
import torch

sp = spm.SentencePieceProcessor(model_file="sentencepiece/out.model")

input_texts = [
    "大規模言語モデルの開発には、膨大な資金が必要だ。",
    "海底の奥深くには、未知の生物が棲んでいる可能性がある。",
    "研究機関である大学は、国防の点で重要な役割を果たしている。",
    "人類はいずれ地球以外の惑星に移住するかもしれない。",
    "二酸化炭素の排出は、地球温暖化の原因となっている。",
]


model = LlamaForCausalLM.from_pretrained("hayago/Veloce-100M")

for text in input_texts:
    input_ids = sp.encode_as_ids(text)
    input_ids = [sp.bos_id()] + input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    outputs = model.generate(
        input_ids,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        repetition_penalty=1.11,
        no_repeat_ngram_size=3,
        max_new_tokens=128,
        attention_mask=attention_mask,
        eos_token_id=sp.eos_id(),
        pad_token_id=sp.eos_id(),
    )
    generated_ids = outputs[0].tolist()
    generated_text = sp.decode(generated_ids)

    print("-" * 80)
    print("入力:")
    print(text)
    print("生成:")
    print(generated_text)
    print("-" * 80)
