import nltk

nltk.download('punkt')

sample_text = "Machine learning é um campo da inteligência artificial que permite que computadores aprendam padrões a partir de dados. Sem serem programados explicitamente para cada tarefa."

work_tokens = nltk.word_tokenize(sample_text)

print(work_tokens)

sentence_tokens = nltk.sent_tokenize(sample_text)

print(sentence_tokens)

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    return [word for word in tokens if word.isalnum()]

documents = [
    "Machine learning é o aprendizado automático de máquinas a partir de dados.",
    "Ele permite que sistemas façam previsões e decisões sem programação explícita.",
    "É usado em áreas como reconhecimento de voz, imagens e recomendação de conteúdo.",
]

preprocess_docs = [" ".join(preprocess(doc)) for doc in documents]

print(preprocess_docs)