import googlemaps
import secret
from datetime import datetime
import requests
import pickle
import time

gmaps = googlemaps.Client(key=secret.PLACES_API_KEY)
# lat = 45.411400
# lon = 11.887491

coordinates = [
    (45.411400, 11.887491),  # torre archimede
    (45.409218, 11.877915),  # piazza garibaldi
    (45.407698, 11.873351),  # piazza dei signori
    (45.401403, 11.880813),  # basilica di sant'antonio
]


# def find_places():
#    results = gmaps.places_nearby(location=(lat, lon), type='bar', radius=500)
#    print(len(results))
#    return results


def find_places():
    place_types = ['bar|restaurant|cafe|night_club']
    f = open('maps_data.pickle', "rb")
    data = pickle.load(f)
    # data = dict()
    # data['requests'] = []

    f.close()

    for lat, lon in coordinates:
        for place_type in place_types:
            url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json?" \
                  "location={1},{2}&radius=500&type={3}&key={0}".format(  # &keyword=xxx
                secret.PLACES_API_KEY, lat, lon, place_type)

            execute_request(url, data)

    print("Retrieved {0} item(s).".format(len(data['requests'])))

    save_changes(data)
    return data


def save_changes(data):
    f = open('maps_data_tmp.pickle', "wb")

    pickle.dump(data, f)
    f.close()

    import os
    os.replace("maps_data_tmp.pickle", "maps_data.pickle")


def execute_request(url, data):
    r = requests.get(url)
    if r.status_code == 200:
        pass
    else:
        print("Errore: ", r.status_code)
        raise r.status_code

    for item in r.json()['results']:
        if item['place_id'] not in [place['place_id'] for place in data['requests']]:
            data['requests'].append(item)

    if r.json().get('next_page_token') is not None \
            and r.json()['next_page_token'] is not None \
            and r.json()['next_page_token'] != "":
        time.sleep(5)  # need to wait a bit..
        print("new page!")
        execute_request("https://maps.googleapis.com/maps/api/place/nearbysearch/json?pagetoken={0}&key={1}".format(
            r.json()['next_page_token'], secret.PLACES_API_KEY), data)
    # else:
    #    print(r.json(), "non ha next_page_token")
    return r.json()


def reinitialize_data():
    f = open('maps_data.pickle', "wb")

    data = dict()
    data['requests'] = []

    pickle.dump(data, f)

    f.close()


def read_data():
    f = open('maps_data.pickle', "rb")
    data = pickle.load(f)

    # for item in data['requests']:
    #    print(item['name'])
    print("Found {0} places.".format(len(data['requests'])))

    return data


def get_details(place_id, data):
    url = "https://maps.googleapis.com/maps/api/place/details/json?place_id={0}&fields=address_component,adr_address," \
          "business_status,formatted_address,geometry,icon,name,photo,place_id,plus_code,type,url,utc_offset,vicinity," \
          "formatted_phone_number,international_phone_number,opening_hours,website,price_level,rating,review," \
          "user_ratings_total&key={1}&language=it".format(
        place_id, secret.PLACES_API_KEY)

    r = requests.get(url)
    if r.status_code == 200:
        pass
    else:
        print("Errore: ", r.status_code)
        raise r.status_code

    data['details'][place_id] = r.json()


def fill_details(data):
    if data.get('details') is None:
        data['details'] = dict()

    ids = [place['place_id'] for place in data['requests']]
    for place_id in ids:
        if data['details'].get(place_id) is None:  # risparmio call, "cache"?
            get_details(place_id, data)

    save_changes(data)


def word2vec_analysis(labels, weights=None, N=4, translate=True):
    import gensim
    import numpy as np

    print("loading dataset...")
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    print('dataset loaded :)')

    if translate:

        def find_longest_word(word_list):
            longest_word = max(word_list, key=len)
            return longest_word

        from googletrans import Translator
        translator = Translator()
        labels_en = []
        new_weights = []
        for label, weight in zip(labels, weights):
            translated = translator.translate(label, dest='en').text
            longest = find_longest_word(translated.split(" "))
            try:
                _ = model[longest]
                labels_en.append(longest)
                new_weights.append(weight)
            except KeyError:
                continue

        print("Starting labels: ", labels)
        print("Translated labels: ", labels_en)

        labels = labels_en

    # labels = ['cat', 'dog', 'mouse', 'lately', 'seldom', 'somehow', 'this', 'pencil', 'suitcase', 'pen']

    X = np.array([model[label] for label in labels])
    print(X.shape)

    kmeans_analysis(X, labels, new_weights, N)


def kmeans_analysis(X, labels, weights=None, N=5):
    from sklearn.cluster import KMeans
    import numpy as np
    kmeans = KMeans(n_clusters=N, random_state=0).fit(X, sample_weight=weights)
    # labels[X.tolist().index(x)[0]]
    clusters = [[] for x in kmeans.cluster_centers_]
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append((labels[i], X[i]))

    for cluster, cluster_center in zip(clusters, kmeans.cluster_centers_):
        label_center = None
        for point in cluster:
            if (point[1] == cluster_center).all():
                label_center = point[0]
            # else:
            #     print(point[1], cluster_center)

        print("Cluster has {0} item(s):".format(len(cluster)))
        for point in cluster:
            print(point[0])


def text_analysis(data):
    precorpus = []
    for item in data['details'].values():
        if item['result'].get('reviews') is not None:
            reviews = item['result']['reviews']
            reviews_text = [x['text'] for x in reviews]
            for text in reviews_text:
                precorpus.append(text)

    print("Found {0} reviews".format(len(precorpus)))
    import re

    # nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk.tokenize import RegexpTokenizer
    # nltk.download('wordnet')
    from nltk.stem.wordnet import WordNetLemmatizer

    ##Creating a list of stop words and adding custom stopwords
    try:
        stop_words = set(stopwords.words("italian"))
    except LookupError:
        import nltk
        nltk.download('wordnet')
        nltk.download('stopwords')
        stop_words = set(stopwords.words("italian"))

    # Creating a list of custom stopwords
    # Creato dopo analisi grafica di word1.png

    new_words = ["bar", "molto", "ottimo", "locale", "posto", "ben", "volta", "po", "più", "sempre", "padova", "ottimi",
                 "poco", "ottima"]
    stop_words = stop_words.union(new_words)

    corpus = []
    for t in precorpus:
        # Remove punctuations
        text = re.sub('[^a-zA-Zùàèé]', ' ', t)

        # Convert to lowercase
        text = text.lower()

        # remove tags
        text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

        # remove special characters and digits
        text = re.sub("(\\d|\\W)+", " ", text)

        ##Convert to list from string
        text = text.split()

        ##Stemming
        ps = PorterStemmer()
        # Lemmatisation
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if not word in
                                                            stop_words]
        text = " ".join(text)
        corpus.append(text)

    # Word cloud
    from os import path
    from PIL import Image
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    import matplotlib.pyplot as plt

    wordcloud = WordCloud(
        background_color='white',
        stopwords=stop_words,
        max_words=100,
        max_font_size=50,
        random_state=42
    ).generate(str(corpus))
    print(wordcloud)

    fig = plt.figure(1)
    plt.imshow(wordcloud)
    plt.axis('off')
    # plt.show()
    # plt.close()
    fig.savefig("word1.png", dpi=900)

    return wordcloud.words_


d = read_data()
occ = text_analysis(d)
word2vec_analysis(occ.keys(), list(occ.values()), N=12, translate=True)
