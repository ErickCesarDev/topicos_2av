📂 Estrutura do Projeto
.
├── chatbot.py                  # Código principal do chatbot
├── respostas_aprendidas.json  # Base de conhecimento aprendida (gerado dinamicamente)
└── README.md                   # Este arquivo

▶️ Como Executar
Pré-requisitos
Certifique-se de ter o Python instalado (versão 3.7 ou superior) e os seguintes pacotes:

pip install pandas scikit-learn textblob
python -m textblob.download_corpora

Executar o chatbot
python chatbot.py

💬 Exemplo de Uso
Ao iniciar o script, o chatbot vai saudar o usuário:

Olá! 👋 Sou o assistente virtual da loja de eletrônicos.
Como posso te ajudar?
1️⃣ Ver produtos
2️⃣ Preços
3️⃣ Agendamento
4️⃣ Falar com atendente
Digite 'sair' para encerrar.

Exemplo 1: Ver produtos

Você: 1
Bot: Aqui estão os produtos disponíveis:
 - Samsung Galaxy A14
 - Galaxy S21 FE
 - Galaxy M54
 - ...

Exemplo 2: Pergunta não conhecida

Você: Vocês vendem fones de ouvido?
Bot: Ainda não sei responder a isso. Pode me ensinar a resposta correta?
Você: Sim, vendemos vários modelos de fones Bluetooth.
Bot: Obrigado! Agora eu sei responder essa pergunta.

Na próxima vez que alguém fizer essa pergunta, o bot já saberá a resposta.
