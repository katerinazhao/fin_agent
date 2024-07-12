import torch
import torch.nn as nn
import torch.optim as optim
from transformers import DPRContextEncoder, DPRQuestionEncoder, GPT2LMHeadModel, GPT2Tokenizer

# 检索器部分
class Retriever:
    def __init__(self, context_encoder, question_encoder, tokenizer):
        self.context_encoder = context_encoder
        self.question_encoder = question_encoder
        self.tokenizer = tokenizer

    def encode_contexts(self, contexts):
        inputs = self.tokenizer(contexts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            embeddings = self.context_encoder(**inputs).pooler_output
        return embeddings

    def encode_question(self, question):
        inputs = self.tokenizer(question, return_tensors='pt')
        with torch.no_grad():
            embedding = self.question_encoder(**inputs).pooler_output
        return embedding

    def retrieve(self, question, contexts, top_k=5):
        question_embedding = self.encode_question(question)
        context_embeddings = self.encode_contexts(contexts)
        scores = torch.matmul(question_embedding, context_embeddings.T).squeeze(0)
        top_k_indices = scores.topk(top_k).indices
        return [contexts[i] for i in top_k_indices]

# 生成器部分
class Generator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            output = self.model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

# RAG模型
class RAGModel:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def answer(self, question, contexts, top_k=5):
        retrieved_contexts = self.retriever.retrieve(question, contexts, top_k)
        prompt = f"{question} {' '.join(retrieved_contexts)}"
        answer = self.generator.generate(prompt)
        return answer

# 主函数
def question_answering_main():
    context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    generator_model = GPT2LMHeadModel.from_pretrained('gpt2')

    retriever = Retriever(context_encoder, question_encoder, tokenizer)
    generator = Generator(generator_model, tokenizer)

    rag_model = RAGModel(retriever, generator)

    question = "What is the capital of France?"
    contexts = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "Madrid is the capital of Spain.",
        "Rome is the capital of Italy."
    ]

    answer = rag_model.answer(question, contexts)
    print(f"Question: {question}\nAnswer: {answer}")

if __name__ == '__main__':
    question_answering_main()