import os
import sys
from distutils.util import strtobool

from flask import Flask, abort, request
from gevent import pywsgi
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from linebot import LineBotApi, WebhookHandler, WebhookParser
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from langchain_openai import ChatOpenAI
from opencc import OpenCC
import torch
import time

import src.constants as const
import src.utils as utils
from src.utils import PathHelper, get_logger

# logger and const
logger = get_logger(__name__)
# encoding_model_name = const.ENCODING_MODEL_NAME

# hard-coded environment variables
channel_secret = "CHANNEL_SECRET"
channel_access_token = "CHANNEL_ACCESS_TOKEN"
nvidia_api_key = "NVIDIA_API_KEY"
support_multilingual = False

os.environ['NVIDIA_API_KEY'] = nvidia_api_key

# Load and process documents
loader = UnstructuredFileLoader("recipe.pdf")
data = loader.load()

embedding_model = "intfloat/e5-large-v2"
TEXT_SPLITTER_CHUNK_SIZE = 510
TEXT_SPLITTER_CHUNK_OVERLAP = 200

text_splitter = SentenceTransformersTokenTextSplitter(
    model_name=embedding_model,
    chunk_size=TEXT_SPLITTER_CHUNK_SIZE,
    chunk_overlap=TEXT_SPLITTER_CHUNK_OVERLAP,
)
documents = text_splitter.split_documents(data)

# Create embeddings and vectorstore
model_kwargs = {"device": "cpu"}  # Change to {"device": "cpu"} for CPU implementation     cuda:0
encode_kwargs = {"normalize_embeddings": False}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)
vectorstore = Chroma.from_documents(documents, hf_embeddings)

# create app
app = Flask(__name__)
line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)
parser = WebhookParser(channel_secret)

# initialize NVIDIA LLM
llm = ChatOpenAI(
            model="meta/llama3-70b-instruct",
            openai_api_key=os.environ["NVIDIA_API_KEY"],
            openai_api_base="https://integrate.api.nvidia.com/v1",
            temperature=0.5,
            max_tokens=1024,
            model_kwargs={"top_p": 1},
        )

# create converter (simple chinese to traditional chinese)
s2t_converter = OpenCC("s2t")

# create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="question",
    output_key="answer",
)

# configure QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever()
)

# create handlers
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except Exception as e:
        logger.error(e)
        abort(400)

    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    question = event.message.text.strip()

    if question.startswith("/清除") or question.lower().startswith("/clear"):
        memory.clear()
        answer = "歷史訊息清除成功"
    elif (
        question.startswith("/教學")
        or question.startswith("/指令")
        or question.startswith("/說明")
        or question.startswith("/操作說明")
        or question.lower().startswith("/instruction")
        or question.lower().startswith("/help")
    ):
        answer = "指令：\n/清除 or /clear\n👉 當 Bot 開始鬼打牆，可清除歷史訊息來重置"
    else:
        

        # get answer from qa_chain
        result = qa_chain.invoke({"query": question})
        answer = result['result']
        answer = s2t_converter.convert(answer)
        logger.info(f"answer: {result}")

    # reply message
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=answer))

# if __name__ == "__main__":
#     server = pywsgi.WSGIServer(('0.0.0.0', 12345),app)
#     server.serve_forever()

#     logger.info("app started")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

    logger.info("app started")
