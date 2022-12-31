## Question Answering Pipeline using DistilBERT

Author: Deepak John Reji (https://www.linkedin.com/in/deepak-john-reji/)

Youtube: https://www.youtube.com/@deepakjohnreji

Question answering (QA) is a natural language processing task in which a computer program is able to understand and respond to questions posed in human language. One common use case for QA systems is in the creation of virtual assistants or chatbots that can assist users by answering questions or providing information.

One important aspect of QA systems is the use of embeddings, which are numerical representations of words or phrases in a continuous vector space. These embeddings capture the semantic relationships between words and can be used to compare the similarity of different words or phrases. In a QA system, embeddings can be used to identify the most relevant information in a large dataset in order to provide a accurate answer to a user's question.

![pipeline%20design%20-%20qna.png](attachment:pipeline%20design%20-%20qna.png)

### Data Source

Wikipedia is a free, open-source encyclopedia that is available online and written collaboratively by volunteers around the world. It is a widely used data source for a variety of applications, including natural language processing and machine learning. Wikipedia provides a large and diverse collection of articles on a wide range of topics, making it an ideal resource for training and evaluating machine learning models. In addition, Wikipedia articles are written in a clear and structured manner, which makes it easier to extract relevant information and use it for various tasks. Some common uses of Wikipedia as a data source include information retrieval, question answering, and text classification. Overall, Wikipedia is a valuable resource for anyone looking for high-quality, reliable information on a wide range of topics.

#### 1. Extract the important keywords from the Question

Keyword extraction is the process of automatically identifying and extracting the most important keywords or phrases from a piece of text. This can be useful for tasks such as information retrieval, text classification, and text summarization. One method for keyword extraction is the Rapid Automatic Keyword Extraction (RAKE) algorithm, which is implemented in the rake_nltk package in Python.


```python
# install rake package
# pip install rake-nltk
# pip install wikipedia
```


```python
# loading the packages
from rake_nltk import Rake

# keyword extraction function
def keyword_extractor(query):
    """
    Rake has some features:
        1. convert automatically to lower case
        2. extract important key phrases
        3. it will extract combine words also (eg. Deep Learning, Capital City)
    """
    r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
    r.extract_keywords_from_text(query)
    keywords = r.get_ranked_phrases() # To get keyword phrases ranked highest to lowest.
    return keywords
```


```python
search_string = input("type your question here: ")
keyword_list = keyword_extractor(search_string)

keyword_list
```

    type your question here: who is Billie eilish?
    




    ['billie eilish']



#### 2. Data Collection

Collect the data based on the context/keywords extracted from the question


```python
import wikipedia

# data collection using wikepedia
def data_collection(search_words):
    """wikipedia"""
    search_query = ' '.join(search_words)
    wiki_pages = wikipedia.search(search_query, results = 5)
    
    information_list = []
    pages_list = []
    for i in wiki_pages:
        try:
            info = wikipedia.summary(i)
            if any(word in info.lower() for word in search_words):
                information_list.append(info)
                pages_list.append(i)
        except:
            pass
    
    original_info = information_list
    information_list = [item[:1000] for item in information_list] # limiting the word len to 512
    
    return information_list, pages_list, original_info
```


```python
information, pages, original_data = data_collection(keyword_list)
```


```python
print(information)
```

    ['Billie Eilish Pirate Baird O\'Connell ( EYE-lish; born December 18, 2001) is an American singer-songwriter. She first gained public attention in 2015 with her debut single "Ocean Eyes", written and produced by her brother Finneas O\'Connell, with whom she collaborates on music and live shows. In 2017, she released her debut extended play (EP), titled Don\'t Smile at Me. Commercially successful, it reached the top 15 of record charts in numerous countries, including the US, UK, Canada, and Australia.\nEilish\'s first studio album, When We All Fall Asleep, Where Do We Go? (2019), debuted atop the US Billboard 200 and UK Albums Chart. It was one of the best-selling albums of the year, buoyed by the success of its fifth single "Bad Guy", Eilish\'s first number-one on the US Billboard Hot 100. This made her the first artist born in the 21st century to release a chart-topping single. The following year, Eilish performed the theme song "No Time to Die" for the James Bond film of the same name, whic', 'American singer and songwriter Billie Eilish has released 2 studio albums, 1 live album, 1 video album, 2 extended plays (EPs), 33 singles, and 25 music videos. According to RIAA, she has sold 41.5 million digital singles and 5 million albums. IFPI crowned "Bad Guy" as 2019\'s biggest selling single globally, selling 19.5 million units in a year span. Eilish is regarded by various media outlets as the "Queen of Gen-Z Pop". At age 17, she became the youngest female artist in UK chart history to score a number-one album. As of October 2021, Eilish has accumulated 76.7 billion career streams worldwide. According to IFPI, Eilish was the 4th best-selling artist of 2019 and 5th best-selling artist of 2020.In August 2017, Eilish released her first EP, Don\'t Smile at Me, which reached number 14 on the US Billboard 200, number 12 on the UK Albums Chart, and the top 10 in Australia, New Zealand, and Sweden. Eilish then released the internationally charting singles such as "Lovely" (with Khalid), ', '"Bad Guy" (stylized in all lowercase) is a song by American singer-songwriter Billie Eilish and the fifth single from her debut studio album, When We All Fall Asleep, Where Do We Go? (2019). It was released on March 29, 2019, through Darkroom and Interscope Records. The song was described by media as electropop, dance-pop, trap-pop, with minimalist instrumentation. In the lyrics, Eilish taunts someone for being a bad guy while suggesting that she is more resilient than they are. Eilish wrote "Bad Guy" with her brother and producer Finneas O\'Connell. Another version of the song, a collaboration with Canadian singer Justin Bieber, was released on July 11, 2019.\nUpon release, "Bad Guy" received mostly positive reviews. It topped the US Billboard Hot 100 and several international charts including the UK, Australia, Canada and New Zealand, peaking at number two on the UK Singles Chart. In the US, it ended the record-breaking 19-week run of "Old Town Road" by Lil Nas X featuring Billy Ray Cy', "Happier Than Ever is the second studio album by American singer Billie Eilish, released by Darkroom and Interscope Records on July 30, 2021. Eilish co-wrote the album with her brother Finneas O'Connell, her frequent collaborator who also produced the album and played every instrument. Eilish cited self-reflection during the COVID-19 pandemic as the biggest inspiration for the LP.\nPrimarily a downtempo pop record, Happier Than Ever is characterized by sparse, jazz-influenced, electropop arrangements set to meditative tempos, departing from the upbeat, trap-led sound of Eilish's debut album, When We All Fall Asleep, Where Do We Go? (2019). Consisting of torch songs about the downsides of stardom, Happier Than Ever draws heavily from Eilish's rise to fame and the drawbacks that come with it. Upon release, the album received acclaim from music critics, who praised its stylistic, restrained production, and insightful lyrics. At the 64th Annual Grammy Awards, the album and its title track re", '"Billie Eilish" is a song by American rapper Armani White. It was released as a single on May 23, 2022 by Legendbound and Def Jam Recordings, and debuted at number 99 on the US Billboard Hot 100 in September 2022. The song samples "Nothin\'" by N.O.R.E.']
    


```python
print(pages)
```

    ['Billie Eilish', 'Billie Eilish discography', 'Bad Guy (Billie Eilish song)', 'Happier Than Ever', 'Billie Eilish (song)']
    


```python
print(original_data)
```

    ['Billie Eilish Pirate Baird O\'Connell ( EYE-lish; born December 18, 2001) is an American singer-songwriter. She first gained public attention in 2015 with her debut single "Ocean Eyes", written and produced by her brother Finneas O\'Connell, with whom she collaborates on music and live shows. In 2017, she released her debut extended play (EP), titled Don\'t Smile at Me. Commercially successful, it reached the top 15 of record charts in numerous countries, including the US, UK, Canada, and Australia.\nEilish\'s first studio album, When We All Fall Asleep, Where Do We Go? (2019), debuted atop the US Billboard 200 and UK Albums Chart. It was one of the best-selling albums of the year, buoyed by the success of its fifth single "Bad Guy", Eilish\'s first number-one on the US Billboard Hot 100. This made her the first artist born in the 21st century to release a chart-topping single. The following year, Eilish performed the theme song "No Time to Die" for the James Bond film of the same name, which topped the UK Singles Chart and won the Academy Award for Best Original Song in 2022. Her subsequent singles "Everything I Wanted", "My Future", "Therefore I Am", and "Your Power" peaked in the top 10 in the US and UK. Her second studio album, Happier Than Ever (2021), topped charts in 25 countries.\nEilish has received multiple accolades, including seven Grammy Awards, two American Music Awards, two Guinness World Records, three MTV Video Music Awards, three Brit Awards, a Golden Globe Award and an Academy Award. She is the youngest artist in Grammy history to win all four general field categories—Best New Artist, Record of the Year, Song of the Year, and Album of the Year—in the same year. She was featured on Time magazine\'s inaugural Time 100 Next list in 2019 and the Time 100 in 2021. According to the Recording Industry Association of America (RIAA) and Billboard, Eilish is the 26th-highest-certified digital singles artist and one of the most successful artists of the 2010s. She has a history of political activism, focusing on climate change awareness and women\'s equality. She was honored as one of the BBC 100 Women in December 2022.', 'American singer and songwriter Billie Eilish has released 2 studio albums, 1 live album, 1 video album, 2 extended plays (EPs), 33 singles, and 25 music videos. According to RIAA, she has sold 41.5 million digital singles and 5 million albums. IFPI crowned "Bad Guy" as 2019\'s biggest selling single globally, selling 19.5 million units in a year span. Eilish is regarded by various media outlets as the "Queen of Gen-Z Pop". At age 17, she became the youngest female artist in UK chart history to score a number-one album. As of October 2021, Eilish has accumulated 76.7 billion career streams worldwide. According to IFPI, Eilish was the 4th best-selling artist of 2019 and 5th best-selling artist of 2020.In August 2017, Eilish released her first EP, Don\'t Smile at Me, which reached number 14 on the US Billboard 200, number 12 on the UK Albums Chart, and the top 10 in Australia, New Zealand, and Sweden. Eilish then released the internationally charting singles such as "Lovely" (with Khalid), "You Should See Me in a Crown", "When the Party\'s Over", "Come Out and Play", "Bury a Friend", "Wish You Were Gay", "Bad Guy", "Everything I Wanted", "My Future", and "Therefore I Am".\nHer debut studio album, When We All Fall Asleep, Where Do We Go? was released on March 29, 2019, and peaked at number one in 15 countries around the world including the US, UK, Australia, and Canada. It has sold 1.2 million units globally in 2019 alone, making it the fifth biggest seller of the year. "Bad Guy" became Eilish\'s first single to debut in the top 10 of Billboard Hot 100, peaking at number one. With "Bad Guy", she became the first artist born in the 21st century and third Gen Z artist to have a number one song on the Hot 100, as well as the first to have a number-one album on the Billboard 200. Eilish also broke the record for the most simultaneous hits on Billboard Hot 100 among women. She released her second studio album, Happier Than Ever, on July 30, 2021, containing 16 tracks. Like her previous album, it peaked at number one in several countries including the US, UK, Australia, and Canada. On July 21, 2022, she surprise-released her second EP, Guitar Songs, which consists of two tracks, "TV" and "The 30th".', '"Bad Guy" (stylized in all lowercase) is a song by American singer-songwriter Billie Eilish and the fifth single from her debut studio album, When We All Fall Asleep, Where Do We Go? (2019). It was released on March 29, 2019, through Darkroom and Interscope Records. The song was described by media as electropop, dance-pop, trap-pop, with minimalist instrumentation. In the lyrics, Eilish taunts someone for being a bad guy while suggesting that she is more resilient than they are. Eilish wrote "Bad Guy" with her brother and producer Finneas O\'Connell. Another version of the song, a collaboration with Canadian singer Justin Bieber, was released on July 11, 2019.\nUpon release, "Bad Guy" received mostly positive reviews. It topped the US Billboard Hot 100 and several international charts including the UK, Australia, Canada and New Zealand, peaking at number two on the UK Singles Chart. In the US, it ended the record-breaking 19-week run of "Old Town Road" by Lil Nas X featuring Billy Ray Cyrus. "Bad Guy" has received several certifications, including a tenfold platinum award from the Australian Recording Industry Association (ARIA) and a sextuple one from the Recording Industry Association of America (RIAA). The song received several awards, including Record and Song of the Year at the 62nd Annual Grammy Awards.\nDave Meyers directed the music video, which depicts Eilish involved in several activities such as wild dancing, suffering a nosebleed and sitting on the back of a man doing push-ups. Reviewers noted the video for its camp elements and eccentric imagery.', 'Happier Than Ever is the second studio album by American singer Billie Eilish, released by Darkroom and Interscope Records on July 30, 2021. Eilish co-wrote the album with her brother Finneas O\'Connell, her frequent collaborator who also produced the album and played every instrument. Eilish cited self-reflection during the COVID-19 pandemic as the biggest inspiration for the LP.\nPrimarily a downtempo pop record, Happier Than Ever is characterized by sparse, jazz-influenced, electropop arrangements set to meditative tempos, departing from the upbeat, trap-led sound of Eilish\'s debut album, When We All Fall Asleep, Where Do We Go? (2019). Consisting of torch songs about the downsides of stardom, Happier Than Ever draws heavily from Eilish\'s rise to fame and the drawbacks that come with it. Upon release, the album received acclaim from music critics, who praised its stylistic, restrained production, and insightful lyrics. At the 64th Annual Grammy Awards, the album and its title track received a total of seven nominations, including Album of the Year, Best Pop Vocal Album, Song of the Year and Record of the Year.\nSeven singles were released in promotion of the album: "My Future", "Therefore I Am", "Your Power", "Lost Cause", "NDA", the title track, and "Male Fantasy"; the first three peaked within the top 10 of the US Billboard Hot 100. "Therefore I Am" was the highest-charting song from the album, peaking at number two, followed by "My Future" at number six, and "Your Power" at number 10. Happier Than Ever debuted atop the Billboard 200 as Eilish\'s second number-one album in the United States, and topping the album charts in 27 other countries. Eilish performed the album\'s tracks in the Disney+ concert film, Happier Than Ever: A Love Letter to Los Angeles, which was released on September 3, 2021. To further promote the album, Eilish embarked on her sixth concert tour, titled Happier Than Ever, The World Tour, which began on February 3, 2022.', '"Billie Eilish" is a song by American rapper Armani White. It was released as a single on May 23, 2022 by Legendbound and Def Jam Recordings, and debuted at number 99 on the US Billboard Hot 100 in September 2022. The song samples "Nothin\'" by N.O.R.E.']
    

#### 3. Document Ranking

The BM25 ranking formula is based on the frequency of the query terms in the document, the length of the document, and the overall frequency of the query terms in the collection of documents. The formula assigns a score to each document based on these factors, with higher scores indicating a higher degree of relevance.


```python
from rank_bm25 import BM25Okapi

# document ranking function
def document_ranking(documents, query, n):
    """BM25"""
    try:
        tokenized_corpus = [doc.split(" ") for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        datastore = bm25.get_top_n(tokenized_query, documents, n)
    except:
        pass
    return datastore
```


```python
datastore = document_ranking(information, search_string, 3)
datastore
```




    ["Happier Than Ever is the second studio album by American singer Billie Eilish, released by Darkroom and Interscope Records on July 30, 2021. Eilish co-wrote the album with her brother Finneas O'Connell, her frequent collaborator who also produced the album and played every instrument. Eilish cited self-reflection during the COVID-19 pandemic as the biggest inspiration for the LP.\nPrimarily a downtempo pop record, Happier Than Ever is characterized by sparse, jazz-influenced, electropop arrangements set to meditative tempos, departing from the upbeat, trap-led sound of Eilish's debut album, When We All Fall Asleep, Where Do We Go? (2019). Consisting of torch songs about the downsides of stardom, Happier Than Ever draws heavily from Eilish's rise to fame and the drawbacks that come with it. Upon release, the album received acclaim from music critics, who praised its stylistic, restrained production, and insightful lyrics. At the 64th Annual Grammy Awards, the album and its title track re",
     '"Bad Guy" (stylized in all lowercase) is a song by American singer-songwriter Billie Eilish and the fifth single from her debut studio album, When We All Fall Asleep, Where Do We Go? (2019). It was released on March 29, 2019, through Darkroom and Interscope Records. The song was described by media as electropop, dance-pop, trap-pop, with minimalist instrumentation. In the lyrics, Eilish taunts someone for being a bad guy while suggesting that she is more resilient than they are. Eilish wrote "Bad Guy" with her brother and producer Finneas O\'Connell. Another version of the song, a collaboration with Canadian singer Justin Bieber, was released on July 11, 2019.\nUpon release, "Bad Guy" received mostly positive reviews. It topped the US Billboard Hot 100 and several international charts including the UK, Australia, Canada and New Zealand, peaking at number two on the UK Singles Chart. In the US, it ended the record-breaking 19-week run of "Old Town Road" by Lil Nas X featuring Billy Ray Cy',
     'American singer and songwriter Billie Eilish has released 2 studio albums, 1 live album, 1 video album, 2 extended plays (EPs), 33 singles, and 25 music videos. According to RIAA, she has sold 41.5 million digital singles and 5 million albums. IFPI crowned "Bad Guy" as 2019\'s biggest selling single globally, selling 19.5 million units in a year span. Eilish is regarded by various media outlets as the "Queen of Gen-Z Pop". At age 17, she became the youngest female artist in UK chart history to score a number-one album. As of October 2021, Eilish has accumulated 76.7 billion career streams worldwide. According to IFPI, Eilish was the 4th best-selling artist of 2019 and 5th best-selling artist of 2020.In August 2017, Eilish released her first EP, Don\'t Smile at Me, which reached number 14 on the US Billboard 200, number 12 on the UK Albums Chart, and the top 10 in Australia, New Zealand, and Sweden. Eilish then released the internationally charting singles such as "Lovely" (with Khalid), ']



#### 4. Question Answering

One approach to QA is to use a large language model such as DistilBERT, which is a pre-trained transformer model developed by Hugging Face. To use DistilBERT for QA, you can generate embeddings for your data using the model and then use a similarity measure to compare the embeddings of the user's question to the embeddings of the potential answers in your dataset. Based on the similarity scores, you can select the most similar answer as the predicted answer to the user's question. DistilBERT is a powerful and widely-used tool for QA and can be used in a variety of applications.


```python
import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering

def answergen(context, question):
    """DistilBert"""
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',return_token_type_ids = True)
    model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad', return_dict=False)
    encoding = tokenizer.encode_plus(question, context)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    start_scores, end_scores = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))
    ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens , skip_special_tokens=True)
    answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer_tokens_to_string

answers_list = []    

for i in range(len(datastore)):
    result = answergen(datastore[i], search_string)
    answers_list.append(result)

answers_list
```




    ['singer', 'singer - songwriter', 'singer and songwriter']




```python

```
