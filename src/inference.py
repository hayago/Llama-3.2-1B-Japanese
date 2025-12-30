import sentencepiece as spm
from transformers import LlamaForCausalLM
import torch


def main():
    sp = spm.SentencePieceProcessor(model_file="data/sentencepiece/out.model")

    input_texts = [
        "今日は朝から雨が降っていて、",
        "静かな図書館の中で、私は一冊の古い本を開いた。",
        "夜が更けてきた。\n街の明かりは、\n",
        "この町では、季節が変わるたびに、人々の暮らし方や考え方が少しずつ変化していくのだが、",
        "遠くで鐘の音が鳴り始めたとき、",
    ]

    for text in input_texts:
        input_ids = sp.encode_as_ids(text)
        input_ids = [sp.bos_id()] + input_ids
        input_ids = torch.tensor(input_ids).unsqueeze(0)

        model = LlamaForCausalLM.from_pretrained("Llama-3.2-1B-Japanese")
        outputs = model.generate(input_ids)
        generated_ids = outputs[0].tolist()
        generated_text = sp.decode(generated_ids)
        print(generated_text)


if __name__ == "__main__":
    main()
