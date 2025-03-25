# ChatBot helper MOM
Модель для чат-бота для помощи молодым мамам

**Задача:**

Облегчить поиск интнресующей информации молодым мамам по уходу за ребенком и при возникновении трудностей.

Что использую
---
В качестве "профессии" было определено консультирование и ответы на вопросы молодых мам, которые сталкиваются с проблемами или просто интересуются что будет дальше.

В качестве базы знаний используется PDF с инфорацией собраной на просторах интернета (форумы, чаты, статьи и пр.).

Для постобработки, в обоих случаях, используется модуль LLMRerank, для реранжирования данных.

Задействована LLM **"IlyaGusev/saiga_mistral_7b"**

Ссылка на автора: https://huggingface.co/IlyaGusev


### Описание


Пример
---
**Вопрос:** 

Что собрат в аптечку для малыша?

**Ответ:**

    В аптечку для малыша можно собрать следующее:

    Средства для ухода за новороженным:

    Вата или ватные диски/тампоны для чистки глаз и носа.

    Ватные палочки с ограничителем для чистки ушей.

    Гипоаллергенный детский крем, лучше без добавок.

    Крем под подгузник на основе цинка (если у ребенка есть проблемы с кожей).

    Ножницы с закругленными концами (или кусачки) для стрижки ногтей.

    Спиртовые салфетки для ухода по пупочной ранке.

    Средство после ожогов.

    Гель для купания.

    Лекарственные препараты:

    Отдельная аптечка для новорожденного (чтобы избежать путаницы с взрослыми лекарствами).

    Консультация с педиатром для составления спиiska необходимых препаратов.

    Дополнительно:

    Бесконтактный термометр (электронного или инфракрасного типа).

    Назальный аспиратор.

    Перевязочный материал (стерильный лент, пластыри).

    Гипотермический пакет.

    Стерильные перчатки.

    Избегайте лишних покупок - не стоит накапливать лекарства "на всякий случай". Лучше проконсультироваться с варчом и приобретать только то, что действительно необходимо._


<hr>

Чтобы просмотреть код и выполнить его, можно перейти по ссылке в [Google Colab](https://colab.research.google.com/drive/1v1YOIdgMYiZI10er5nllWNiKw9fM1Twe?usp=sharing)
