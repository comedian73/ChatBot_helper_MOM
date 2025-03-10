Что использую
---
В качестве "профессии" было определено консультирование и ответы на вопросы молодых мам, которые сталкиваются с поблемами или просто интересуются что будет дальше.

В качестве базы знаний используется PDF с инфорацией собраной на просторах интернета (форумы, чаты, статьи и пр.).

Задействована LLM **"IlyaGusev/saiga_mistral_7b"**

Ссылка на автора: https://huggingface.co/IlyaGusev


### Описание
Изначально я решил использоать KnowledgeGraph, т.к в теоритической части было сказано:

Графы знаний являются лучшим решением, чем векторные базы данных, для преодоления галлюцинаций в LLM.
Но после долгих ожиданий создания эбедингов в KnowledgeGraph и первого вопроса, я понял что что-то не совсем хорошо, ведь я получил ужасный ответ.

В конце концов я пришел к использованию ретривера QueryRewritingRetrieverPack

И если сранивать KnowledgeGraph и QueryRewritingRetrieverPack, то можно ощутить разницу (пример ниже).

Пример
---
Вопрос: Что собрат в аптечку для малыша?

### KnowledgeGraph

_В аптеку можно купить следующее:_

_Уход за кожей - кремы, масла, бальзамы, мягкие полотенца._

_Уход за волосами - мыло, шампунь, увлажняющий крем для головы._

_Уход за глазами - растворы, мыло, увлажняющие краèmes._

_Уход за ногами - кремы, масла, бальзамы._

_Уход за губами - пасты, кремы, масла._

_Уход за телом - кремы, масла, бальзамы._

_Уход за волосами - мыло, шампоны, увлажняющие кремы для головы._

_Уход за кожей - кремы, масла, полотенца._

_Уход за волосами - мыло, шампанский, увлажняющие кремы для головы._

_Уход за кожей - кремы, масла, мягкие полотенца._

_Уход за волосами - мыло, шампаны, увлажняющие кремы для головы._

### QueryRewritingRetrieverPack

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
Можно увидеть, что QueryRewritingRetrieverPack справился очень хорошо по сравнению с KnowledgeGraph

Для постобработки, в обоих случаях, используется модуль LLMRerank, для реранжирования данных.
