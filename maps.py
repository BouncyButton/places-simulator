import googlemaps
import secret
from datetime import datetime
import requests
import pickle

gmaps = googlemaps.Client(key=secret.PLACES_API_KEY)
lat = 45.411400
lon = 11.887491


#def find_places():
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
    f = open('maps_data_tmp.pickle', "wb")

    for place_type in place_types:
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json?" \
                 "location={1},{2}&radius=500&type={3}&key={0}".format(  # &keyword=xxx
            secret.PLACES_API_KEY, lat, lon, place_type)

        r = requests.get(url)
        if r.status_code == 200:
            pass
        else:
            print("Errore: ", r.status_code)
            break

        for item in r.json()['results']:
            if item['place_id'] not in [place['place_id'] for place in data['requests']]:
                data['requests'].append(item)

    print("Retrieved {0} item(s).".format(len(data['requests'])))

    pickle.dump(data, f)
    import os
    os.replace("maps_data_tmp.pickle", "maps_data.pickle")

    return data

def reinitialize_data():
    f = open('maps_data.pickle', "wb")

    data = dict()
    data['requests'] = []

    pickle.dump(data, f)

    f.close()



def read_data():
    f = open('maps_data.pickle', "rb")
    data = pickle.load(f)

    for item in data['requests']:
        print(item['name'])

    return data

