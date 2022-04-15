import os
import requests

# website urls for the Hawaii training data
URL_TRAIN = [
    "http://data.insideairbnb.com/united-states/hi/hawaii"
    "/2021-12-11/data/listings.csv.gz",
    "http://data.insideairbnb.com/united-states/hi/hawaii/"
    "2021-12-11/data/calendar.csv.gz",
    "http://data.insideairbnb.com/united-states/hi/hawaii/"
    "2021-12-11/data/reviews.csv.gz",
    "http://data.insideairbnb.com/united-states/hi/hawaii/"
    "2021-12-11/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/hi/hawaii/"
    "2021-12-11/visualisations/reviews.csv",
    "http://data.insideairbnb.com/united-states/hi/hawaii/"
    "2021-12-11/visualisations/neighbourhoods.csv",
    "http://data.insideairbnb.com/united-states/hi/hawaii/"
    "2021-12-11/visualisations/neighbourhoods.geojson",
]

# website urls for the Broward County test data
URL_TEST_BROWARD = [
    "http://data.insideairbnb.com/united-states/fl/"
    "broward-county/2021-12-23/data/listings.csv.gz",
    "http://data.insideairbnb.com/united-states/fl/"
    "broward-county/2021-12-23/data/calendar.csv.gz",
    "http://data.insideairbnb.com/united-states/fl/"
    "broward-county/2021-12-23/data/reviews.csv.gz",
    "http://data.insideairbnb.com/united-states/fl/"
    "broward-county/2021-12-23/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/fl/"
    "broward-county/2021-12-23/visualisations/reviews.csv",
    "http://data.insideairbnb.com/united-states/fl/"
    "broward-county/2021-12-23/visualisations/neighbourhoods.csv",
    "http://data.insideairbnb.com/united-states/fl/"
    "broward-county/2021-12-23/visualisations/neighbourhoods.geojson",
]

# website urls for the Crete test data
URL_TEST_CRETE = [
    "http://data.insideairbnb.com/greece/crete/crete/"
    "2021-12-26/data/listings.csv.gz",
    "http://data.insideairbnb.com/greece/crete/crete/"
    "2021-12-26/data/calendar.csv.gz",
    "http://data.insideairbnb.com/greece/crete/crete/" "2021-12-26/data/reviews.csv.gz",
    "http://data.insideairbnb.com/greece/crete/crete/"
    "2021-12-26/visualisations/listings.csv",
    "http://data.insideairbnb.com/greece/crete/crete/"
    "2021-12-26/visualisations/reviews.csv",
    "http://data.insideairbnb.com/greece/crete/crete/"
    "2021-12-26/visualisations/neighbourhoods.csv",
    "http://data.insideairbnb.com/greece/crete/crete/"
    "2021-12-26/visualisations/neighbourhoods.geojson",
]

# directories to save the data
DIRS = [
    "../01_source_data",
    "../01_source_data/Train",
    "../01_source_data/Test_Broward",
    "../01_source_data/Test_Crete",
]

# create the directories
def create_folder_structure():
    for f in DIRS:
        if not os.path.exists(f):
            os.mkdir(f)
            print(f"{f} folder created")
        else:
            print(f"{f} folder already exists")


# download the data
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


# run the functions
if __name__ == "__main__":
    create_folder_structure()
    for folder, urls in [
        ("../01_source_data/Train", URL_TRAIN),
        ("../01_source_data/Test_Broward", URL_TEST_BROWARD),
        ("../01_source_data/Test_Crete", URL_TEST_CRETE),
    ]:
        download_data(folder, urls)
