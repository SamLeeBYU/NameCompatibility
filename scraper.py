import pandas as pd
import requests
from bs4 import BeautifulSoup
import math
import os

if not os.path.exists("Data/popular_names.csv"):

    url = "https://en.wikipedia.org/wiki/List_of_the_most_popular_given_names_in_South_Korea#cite_note-KukminIlbo20193-2"

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    tables = soup.find_all("table", class_="wikitable")

    years = [2021, 2019, 2017, 2015, 2013, 2011, 2009, 2007, 2004, 1990, 1980, 1970, 1960, 1950, 1945, 1940]

    popular_names = pd.DataFrame()
    for i in range(0, len(tables), 2):
        boys = pd.read_html(str(tables[i]), header=0)[0][["Hangul"]]
        boys["성"] = "남"
        girls = pd.read_html(str(tables[i+1]), header=0)[0][["Hangul"]]
        girls["성"] = "여"
        
        combined = pd.concat([boys, girls])
        combined["년"] = years[math.floor(i/2)]
        
        if years[math.floor(i/2)] == 2015:
            combined.drop(combined.index[-1], inplace=True)
        
        if popular_names.empty:
            popular_names = combined.copy()
        else:
            popular_names = pd.concat([popular_names, combined])
            
    popular_names.rename(columns={"Hangul": "이름"}, inplace=True)
    popular_names.reset_index(drop=True, inplace=True)

    popular_names.to_csv("Data/popular_names.csv", index=False)

if not os.path.exists("Data/surnames.csv"):

    url = "https://en.wikipedia.org/wiki/List_of_Korean_surnames"

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    tables = soup.find_all("table", class_="wikitable")

    surnames1 = pd.read_html(str(tables[0]), header=0)[0][["Hangul[1]"]]
    surnames1.drop(surnames1.index[-1], inplace=True)
    surnames1.rename(columns={"Hangul[1]": "성"}, inplace=True)
    surnames2 = pd.read_html(str(tables[1]), header=0)[0][["Hangul[7]"]]
    surnames2.rename(columns={"Hangul[7]": "성"}, inplace=True)

    surnames = pd.concat([surnames1, surnames2])

    surnames.to_csv("Data/surnames.csv", index=False)