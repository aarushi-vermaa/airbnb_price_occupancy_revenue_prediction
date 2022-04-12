import os
import requests

URL_TRAIN = ['http://data.insideairbnb.com/united-states/hi/hawaii'
             '/2021-12-11/data/listings.csv.gz',
             'http://data.insideairbnb.com/united-states/hi/hawaii/'
             '2021-12-11/data/calendar.csv.gz',
             'http://data.insideairbnb.com/united-states/hi/hawaii/'
             '2021-12-11/data/reviews.csv.gz',
             'http://data.insideairbnb.com/united-states/hi/hawaii/'
             '2021-12-11/visualisations/listings.csv',
             'http://data.insideairbnb.com/united-states/hi/hawaii/'
             '2021-12-11/visualisations/reviews.csv',
             'http://data.insideairbnb.com/united-states/hi/hawaii/'
             '2021-12-11/visualisations/neighbourhoods.csv',
             'http://data.insideairbnb.com/united-states/hi/hawaii/'
             '2021-12-11/visualisations/neighbourhoods.geojson']

URL_TEST_BROWARD = ['http://data.insideairbnb.com/united-states/fl/'
                    'broward-county/2021-12-23/data/listings.csv.gz',
                    'http://data.insideairbnb.com/united-states/fl/'
                    'broward-county/2021-12-23/data/calendar.csv.gz',
                    'http://data.insideairbnb.com/united-states/fl/'
                    'broward-county/2021-12-23/data/reviews.csv.gz',
                    'http://data.insideairbnb.com/united-states/fl/'
                    'broward-county/2021-12-23/visualisations/listings.csv',
                    'http://data.insideairbnb.com/united-states/fl/'
                    'broward-county/2021-12-23/visualisations/reviews.csv',
                    'http://data.insideairbnb.com/united-states/fl/'
                    'broward-county/2021-12-23/visualisations/neighbourhoods.csv',
                    'http://data.insideairbnb.com/united-states/fl/'
                    'broward-county/2021-12-23/visualisations/neighbourhoods.geojson']

URL_TEST_CRETE = ['http://data.insideairbnb.com/greece/crete/crete/'
                  '2021-12-26/data/listings.csv.gz',
                  'http://data.insideairbnb.com/greece/crete/crete/'
                  '2021-12-26/data/calendar.csv.gz',
                  'http://data.insideairbnb.com/greece/crete/crete/'
                  '2021-12-26/data/reviews.csv.gz',
                  'http://data.insideairbnb.com/greece/crete/crete/'
                  '2021-12-26/visualisations/listings.csv',
                  'http://data.insideairbnb.com/greece/crete/crete/'
                  '2021-12-26/visualisations/reviews.csv',
                  'http://data.insideairbnb.com/greece/crete/crete/'
                  '2021-12-26/visualisations/neighbourhoods.csv',
                  'http://data.insideairbnb.com/greece/crete/crete/'
                  '2021-12-26/visualisations/neighbourhoods.geojson']

DIRS = ['../Data', '../Data/Train', '../Data/Test_Broward', '../Data/Test_Crete']


def create_folder_structure():
    for f in DIRS:
        if not os.path.exists(f):
            os.mkdir(f)
            print(f'{f} folder created')
        else:
            print(f'{f} folder already exists')


def download_data(folder, urls):
    for url in urls:
        print(url, "...")
        try:
            response = requests.get(url)
            filename = url.split("/")[-1]
            with open(os.path.join(folder, filename), "wb") as f:
                f.write(response.content)
                print(f"File {filename} has been saved!")
        except requests.RequestException as e:
            print(str(e))


if __name__ == '__main__':
    create_folder_structure()
    for folder, urls in [('../Data/Train', URL_TRAIN),
                         ('../Data/Test_Broward', URL_TEST_BROWARD),
                         ('../Data/Test_Crete', URL_TEST_CRETE)]:
        download_data(folder, urls)
