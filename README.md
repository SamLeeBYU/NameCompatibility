author: Sam Lee
date: 12/29/2023
language: python
---

## Introduction

The popular Korean game called "The Name Compatibility Test" (이름궁합 테스트) is a game which takes two (Korean) names and returns a compatibility score 0-100, representing a percentage. I was curious to see if some names were naturally more "compatible" than others. To investigate this I created this research project. I wanted to figure out if I could come up with a method to identify which names are more compatible than others.

More details of this project are discussed on my [blog](https://samleebyu.github.io/2023/12/29/이름궁합/).

This project's dashboard can be found at [https://namecompatibility.streamlit.app/](https://namecompatibility.streamlit.app/).

## Scripts

1) [**dashboard.py**](dashboard.py) - This script contains all the code for this project's Streamlit dashboard. This also contains the code discussed on my blog. This file runs the algorithms discussed including the hierarchal ordering algorithm.

2) [**nlp.ipynb**](nlp.ipynb) - This jupyter notebook was a sandbox for this project. All of the import code chunks were transferred over to **dashboard.py**.

3) [**scraper.py**](scraper.py) - This script retrieves the data from Wikipedia, formats it into a pandas data frame and saves it to a csv file for future use.

## Data Files

1) [**popular_names.csv**](Data/popular_names.csv) - This is obtained from data source [1]. This contains all popular Korean names dating back to 1940, segregated by sex.

2) [**surnames.csv**][Data/surnames.csv] - This is obtained from data source [2]. This contains a vector of possible Korean surnames. Together with **popular_names.csv**, a total of 34,720 male and 34,720 female names were created.

3) [**stroke_distributions.json**](Data/stroke_distributions.json) - This json file is output from the python scripts above. This contains the stroke distributions (the \# of strokes per syllable) for each male and female name.

4) [**aliases.json**](Data/aliases.json) - This json file is output from the python scripts above. This is the data set for all the aliases and the respective lists (arrays) of names that fall under each alias for both males and females.

5) [**distributions.json**](Data/distributions.json) - This json file is output from the python scripts above. This contains compatibility score distributions for every male and female alias.

6) [**hierarchies.json**](Data/hierarchies.json) - This json file is output from the python scripts above. These are all the iterations from running through the hierarchal distributional ranking process as described on my blog. **dashboard.py** reads in this file and sorts it out into a pandas data frame.

---

## Data Sources

1) A compilation of popular names in Korea dating back to 1940 was obtained through Wikipedia: [(https://en.wikipedia.org/wiki/List_of_the_most_popular_given_names_in_South_Korea#cite_note-KukminIlbo20193-2)]((https://en.wikipedia.org/wiki/List_of_the_most_popular_given_names_in_South_Korea#cite_note-KukminIlbo20193-2))

2) A compilation of possible Korean surnames was obtained through Wikipedia: [https://en.wikipedia.org/wiki/List_of_Korean_surnames](https://en.wikipedia.org/wiki/List_of_Korean_surnames)

