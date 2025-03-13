# Необходимо произвести установку библиотек (если не установлено) и определить зависимости
# !pip install openai llama-index-core "arize-phoenix[evals,llama-index]" gcsfs nest-asyncio "openinference-instrumentation-llama-index>=2.0.0"

# !pip install llama-index llama-hub

# !pip install -r requirements.txt

from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex, Settings, StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from llama_index.core.postprocessor import LLMRerank # модуль реранжирования на базе LLM
from langchain.embeddings import HuggingFaceEmbeddings

import torch
import os

from llama_index.core.llama_pack import download_llama_pack # добавляем загрузчик пакетов
from llama_index.core.node_parser import SimpleNodeParser
from huggingface_hub import login

HF_TOKEN="YOU Hugging Face TOKEN"

login(HF_TOKEN, add_to_git_credential=True)

os.environ["HUGGINGFACE_ACCESS_TOKEN"] = HF_TOKEN

def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == 'system':
            prompt += f"<s>{message.role}\n{message.content}</s>\n"
        elif message.role == 'user':
            prompt += f"<s>{message.role}\n{message.content}</s>\n"
        elif message.role == 'bot':
            prompt += f"<s>bot\n"

    # ensure we start with a system prompt, insert blank if needed
    if not prompt.startswith("<s>system\n"):
        prompt = "<s>system\n</s>\n" + prompt

    # add final assistant prompt
    prompt = prompt + "<s>bot\n"
    return prompt

def completion_to_prompt(completion):
    return f"<s>system\n</s>\n<s>user\n{completion}</s>\n<s>bot\n"

# Определяем параметры квантования, иначе модель не выполниться в колабе
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Задаем имя модели
MODEL_NAME = "IlyaGusev/saiga_mistral_7b"

# Создание конфига, соответствующего методу PEFT (в нашем случае LoRA)
config = PeftConfig.from_pretrained(MODEL_NAME)

# Загружаем базовую модель, ее имя берем из конфига для LoRA
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,          # идентификатор модели
    quantization_config=quantization_config, # параметры квантования
    torch_dtype=torch.float16,               # тип данных
    device_map="auto"                        # автоматический выбор типа устройства

)

# Загружаем LoRA модель
model = PeftModel.from_pretrained(
    model,
    MODEL_NAME,
    torch_dtype=torch.float16
)

# Переводим модель в режим инференса
# Можно не переводить, но явное всегда лучше неявного
model.eval()

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
print(generation_config)

llm = HuggingFaceLLM(
    model=model,             # модель
    model_name=MODEL_NAME,   # идентификатор модели
    tokenizer=tokenizer,     # токенизатор
    max_new_tokens=generation_config.max_new_tokens, # параметр необходимо использовать здесь, и не использовать в generate_kwargs, иначе ошибка двойного использования
    model_kwargs={"quantization_config": quantization_config}, # параметры квантования
    generate_kwargs = {   # параметры для инференса
      "bos_token_id": generation_config.bos_token_id, # токен начала последовательности
      "eos_token_id": generation_config.eos_token_id, # токен окончания последовательности
      "pad_token_id": generation_config.pad_token_id, # токен пакетной обработки (указывает, что последовательность ещё не завершена)
      "no_repeat_ngram_size": generation_config.no_repeat_ngram_size,
      "repetition_penalty": generation_config.repetition_penalty,
      "temperature": 0.1,
      "do_sample": True,
      "top_k": 30,
      "top_p": 0.5
    },
    messages_to_prompt=messages_to_prompt,     # функция для преобразования сообщений к внутреннему формату
    completion_to_prompt=completion_to_prompt, # функции для генерации текста
    device_map="auto",                         # автоматически определять устройство
)

# Загружаем файлы PDF
documents = SimpleDirectoryReader('./new').load_data()

embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
)

# Настройка ServiceContext (глобальная настройка параметров LLM)
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

# Создаем простое графовое хранилище
graph_store = SimpleGraphStore()

# Устанавливаем информацию о хранилище в StorageContext
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# Запускаем генерацию индексов из документа с помощью KnowlegeGraphIndex
indexKG = KnowledgeGraphIndex.from_documents( documents=documents,          # данные для построения графов
                                           max_triplets_per_chunk=3,        # сколько обработывать триплетов связей для каждого блока данных
                                           show_progress=True,              # показывать процесс выполнения
                                           include_embeddings=True,         # включение векторных вложений в индекс для расширенной аналитики
                                           storage_context=storage_context, # куда сохранять результаты
                                           )

query_engine = indexKG.as_query_engine(
    include_text=True,
    verbose=True,
    retriever_mode="keyword",
    response_mode="tree_summarize",
    node_postprocessors=[
        LLMRerank(
            choice_batch_size=256,
            top_n=2,
        )
    ]
    )

# Пример работы модели
query = "Что собрат в аптечку для малыша?"

message_template =f"""<s>system
          Ты помощник и консультант не опытных мам которые хотять получить ответы на вопросы возникающие в процессе материнства.
          Отвечай на вопросы согласно источнику и с учетом контекста. Не придумывай лишнего.
          Если вопрос касается здоровья, то в конце ответа говори о необходимости консультации с врачом.</s>
          <s>user
          {query}
          </s>
          """

response = query_engine.query(message_template)

print(response)

from llama_index.core.llama_pack import download_llama_pack # добавляем загрузчик пакетов

# загружаем и устанавливаем зависимости
QueryRewritingRetrieverPack = download_llama_pack(
  "QueryRewritingRetrieverPack", "./query_rewriting_pack"
)

node_parser = SimpleNodeParser.from_defaults()
nodes = node_parser.get_nodes_from_documents(documents)

query_rewriting_pack = QueryRewritingRetrieverPack(
    nodes,
    chunk_size=256,
    vector_similarity_top_k=2,
    node_postprocessors=[
        LLMRerank(
            choice_batch_size=256,
            top_n=2,
        )
    ]
)

# Пример работы с ретривером
response = query_rewriting_pack.run("Что собрат в аптечку для малыша?")

print("Ответ:\n", response)