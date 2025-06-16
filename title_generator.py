import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "KETI-AIR/ke-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def summarize_one_sentence(text):
    input_text = f"요약: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(
        input_ids,
        max_length=16,
        min_length=8,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary.strip()

if __name__ == "__main__":
    text = input("요약할 텍스트를 입력하세요:\n")
    print("\n요약 결과:", summarize_one_sentence(text))