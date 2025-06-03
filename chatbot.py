import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from textblob import TextBlob
import json
import os

# Variáveis globais do modelo
model = None
mlb = None
resposta_map = {}

# Base inicial vazia
data = []

# Caminho do arquivo para armazenamento das respostas aprendidas
ARQUIVO_APRENDIZADO = "respostas_aprendidas.json"

# Carrega respostas aprendidas (se existirem)
if os.path.exists(ARQUIVO_APRENDIZADO):
    with open(ARQUIVO_APRENDIZADO, "r", encoding="utf-8") as f:
        respostas_aprendidas = json.load(f)
else:
    respostas_aprendidas = {}

# Lista de produtos disponíveis
produtos = [
    "Samsung Galaxy A14", "Galaxy S21 FE", "Galaxy M54", "Galaxy Z Flip 5",
    "Galaxy A54", "Motorola Moto G84", "Moto G73", "Moto Edge 30",
    "Moto E22", "Moto G13", "iPhone 14", "iPhone 12 mini",
    "iPhone SE 2022", "iPhone 15 Pro Max",
]

# Dicionário de preços dos produtos
precos = {
    "Samsung Galaxy A14": "R$ 3.499", "Galaxy M54": "R$ 3.299",
    "Galaxy Z Flip 5": "R$ 2.499", "Galaxy A54": "R$ 1.999", 
    "Motorola Moto G84": "R$ 3.498", "Carregador turbo": "R$ 99",
    "Moto G73": "R$ 2.299", "Moto Edge 30": "R$ 3.199",
    "Moto E22": "R$ 1.799", "Moto G13": "R$ 2.499",
    "iPhone 14": "R$ 2.899", "iPhone 12 mini": "R$ 3.099",
    "iPhone SE 2022": "R$ 1.299", "iPhone 15 Pro Max": "R$ 5.199",
}

# Salva uma nova resposta aprendida
def salvar_resposta_aprendida(pergunta, resposta, intencao="Desconhecida"):
    pergunta = pergunta.lower().strip()
    respostas_aprendidas[pergunta] = {"resposta": resposta, "intencao": [intencao]}
    with open(ARQUIVO_APRENDIZADO, "w", encoding="utf-8") as f:
        json.dump(respostas_aprendidas, f, ensure_ascii=False, indent=2)

# Treina ou re-treina o modelo
def treinar_modelo():
    global model, mlb, resposta_map

    dados_treino = []

    # Junta dados da base inicial e os aprendidos
    for item in data:
        dados_treino.append({
            "pergunta": item["pergunta"],
            "intencao": item["intencao"],
            "resposta": item["resposta"]
        })

    for pergunta, info in respostas_aprendidas.items():
        dados_treino.append({
            "pergunta": pergunta,
            "intencao": info.get("intencao", ["Desconhecida"]),
            "resposta": info.get("resposta", "")
        })

    if not dados_treino:
        model = None
        mlb = None
        resposta_map = {}
        return

    df = pd.DataFrame(dados_treino)
    X = df['pergunta']
    y = df['intencao']

    vectorizer = TfidfVectorizer()
    mlb = MultiLabelBinarizer()
    y_bin = mlb.fit_transform(y)

    model = Pipeline([
        ('tfidf', vectorizer),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear')))
    ])
    model.fit(X, y_bin)

    resposta_map = {}
    for intents, resposta in zip(y, df['resposta']):
        for intent in intents:
            resposta_map.setdefault(intent, set()).add(resposta)

    for intent in resposta_map:
        resposta_map[intent] = list(resposta_map[intent])

# Recomendação simples de produtos
recomendacoes = {
    "samsung": "Recomendamos o Galaxy S22 ou iPhone 13.",
    "apple": "Recomendamos o iPhone 15 Pro Max.",
    "motorola": "Recomendamos o Moto G73.",
}

def recomendar_produto(pergunta):
    for chave, rec in recomendacoes.items():
        if chave in pergunta.lower():
            return rec
    return None

# Análise de sentimento simples com TextBlob
def analisar_sentimento(texto):
    polaridade = TextBlob(texto).sentiment.polarity
    if polaridade > 0.3:
        return "😊 Parece que você está satisfeito! Posso ajudar com mais alguma coisa?"
    elif polaridade < -0.3:
        return "😟 Parece que você está insatisfeito. Deseja falar com um atendente humano?"
    return ""

# Gera resposta com fallback para aprendizado
def gerar_resposta(pergunta):
    pergunta = pergunta.strip().lower()

    if pergunta in respostas_aprendidas:
        return respostas_aprendidas[pergunta]["resposta"], False, False

    if pergunta in ["sair", "encerrar", "tchau", "até mais"]:
        return "Foi um prazer te atender! Até a próxima. 👋", True, False

    # ✅ Verificação extra para evitar erro de shape
    if model is None or mlb is None or not hasattr(mlb, "classes_") or len(mlb.classes_) == 0:
        return "Ainda não sei responder a isso. Pode me ensinar a resposta correta?", False, True

    entrada = [pergunta]
    pred = model.predict(entrada)

    # Evita erro de shape inconsistente
    if len(pred.shape) != 2 or pred.shape[1] != len(mlb.classes_):
        return "Ainda não sei responder a isso. Pode me ensinar a resposta correta?", False, True

    intents = mlb.inverse_transform(pred)[0]

    if not intents:
        return "Desculpe, não entendi sua pergunta. Pode me ensinar a resposta correta?", False, True

    respostas_candidatas = []
    for intent in intents:
        respostas_candidatas.extend(resposta_map.get(intent, []))

    resposta_final = random.choice(respostas_candidatas) if respostas_candidatas else \
        "Não encontrei uma resposta específica para essa intenção. Posso ajudar de outra forma?"

    sugestao = recomendar_produto(pergunta)
    if sugestao:
        resposta_final += f"\n\n📦 Sugestão: {sugestao}"

    comentario_sentimento = analisar_sentimento(pergunta)
    if comentario_sentimento:
        resposta_final += f"\n\n{comentario_sentimento}"

    return resposta_final, False, False

# Chat com aprendizado dinâmico
def iniciar_chat():
    print("Olá! 👋 Sou o assistente virtual da loja de eletrônicos.")
    print("Como posso te ajudar?\n1️⃣ Ver produtos\n2️⃣ Preços\n3️⃣ Agendamento\n4️⃣ Falar com atendente")
    print("Digite 'sair' para encerrar.\n")

    aprendendo = False
    pergunta_aprendizado = ""

    while True:
        pergunta = input("Você: ").strip()

        if aprendendo:
            salvar_resposta_aprendida(pergunta_aprendizado, pergunta)
            print("Bot: Obrigado! Agora eu sei responder essa pergunta.")
            treinar_modelo()
            aprendendo = False
            pergunta_aprendizado = ""
            continue

        pergunta_lower = pergunta.lower()

        if pergunta_lower in ['sair', 'encerrar', 'tchau', 'até mais']:
            print("Bot: Foi um prazer te atender! Até a próxima. 👋")
            break

        elif pergunta_lower == '1' or pergunta_lower == 'ver produtos':
            print("Bot: Aqui estão os produtos disponíveis:")
            for p in produtos:
                print(f" - {p}")

        elif pergunta_lower == '2' or pergunta_lower == 'preços':
            print("Bot: Confira os preços dos nossos produtos:")
            for produto, preco in precos.items():
                print(f" - {produto}: {preco}")

        elif pergunta_lower == '3' or pergunta_lower == 'agendamento':
            data_agendamento = input("Bot: Por favor, informe a data para agendar (dd/mm/aaaa): ")
            print(f"Bot: Agendamento confirmado para o dia {data_agendamento}. Obrigado!")

        elif pergunta_lower == '4' or pergunta_lower == 'falar com atendente':
            print("Bot: Você será encaminhado para um atendente. Por favor, aguarde...")

        else:
            resposta, encerrar, precisa_aprender = gerar_resposta(pergunta)
            print("Bot:", resposta)
            if precisa_aprender:
                aprendendo = True
                pergunta_aprendizado = pergunta
            if encerrar:
                break

# Executa apenas se rodar como script principal
if __name__ == "__main__":
    treinar_modelo()
    iniciar_chat()
