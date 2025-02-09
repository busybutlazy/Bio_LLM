from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# 建立預設的語言模型管道
bio_teacher = pipeline("text-generation", model="gpt2")

@app.get("/")
def read_root():
    return {"message": "Bio_LLM API is running!"}

@app.get("/generate/")
def generate(prompt: str):
    response = bio_teacher(prompt, max_length=50, num_return_sequences=1)
    return {"generated_text": response[0]["generated_text"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
