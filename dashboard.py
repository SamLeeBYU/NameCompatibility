import math
import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import streamlit as st
import random
import plotly.express as px

import scraper

popular_names = pd.read_csv("Data/popular_names.csv")
surnames = pd.read_csv("Data/surnames.csv")

글자 = {
    "leads": ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'],
    "vowels": ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'],
    "tails": ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
}

locs = ["leads", "vowels", "tails"]
strokes = {
    "leads": {
        'ㄱ': 2,
        'ㄲ': 4, 
        'ㄴ': 2, 
        'ㄷ': 3, 
        'ㄸ': 6, 
        'ㄹ': 5,
        'ㅁ': 4, 
        'ㅂ': 4,
        'ㅃ': 8,
        'ㅅ': 2,
        'ㅆ': 4,
        'ㅇ': 1,
        'ㅈ': 3,
        'ㅉ': 6,
        'ㅊ': 4,
        'ㅋ': 3,
        'ㅌ': 4,
        'ㅍ': 4,
        'ㅎ': 3
    },

    "vowels": {
        'ㅏ': 2,
        'ㅐ': 3,
        'ㅑ': 3,
        'ㅒ': 4,
        'ㅓ': 2,
        'ㅔ': 3,
        'ㅕ': 3,
        'ㅖ': 4, 
        'ㅗ': 2,
        'ㅘ': 4,
        'ㅙ': 5,
        'ㅚ': 3,
        'ㅛ': 3,
        'ㅜ': 2,
        'ㅝ': 4,
        'ㅞ': 5,
        'ㅟ': 3, 
        'ㅠ': 3,
        'ㅡ': 1,
        'ㅢ': 2,
        'ㅣ': 1
    },

    "tails": {
        '': 0,
        'ㄱ': 2,
        'ㄲ': 4,
        'ㄳ': 4,
        'ㄴ': 2,
        'ㄵ': 5,
        'ㄶ': 5,
        'ㄷ': 3,
        'ㄹ': 5, 
        'ㄺ': 7,
        'ㄻ': 9,
        'ㄼ': 9,
        'ㄽ': 7,
        'ㄾ': 9,
        'ㄿ': 9,
        'ㅀ': 8,
        'ㅁ': 4, 
        'ㅂ': 4,
        'ㅄ': 7,
        'ㅅ': 2,
        'ㅆ': 4,
        'ㅇ': 1,
        'ㅈ': 3,
        'ㅊ': 4,
        'ㅋ': 3,
        'ㅌ': 4,
        'ㅍ': 4,
        'ㅎ': 3
    }        
}
def map_strokes(character, index):
    loc = locs[index % 3]
    return strokes[loc][character]

def decompose(word):
    
    def decompose_hangul(syllable):
        # Initialize lists for lead, vowel, and tail characters
        lead_chars, vowel_chars, tail_chars = [], [], []

        # Decompose the Hangul syllable into Jamo characters
        for char in syllable:
            if '가' <= char <= '힣':
                # Calculate the index of the Jamo in the Unicode table
                index = ord(char) - ord('가')

                # Calculate the indices for lead, vowel, and tail
                lead_index = index // (21 * 28)
                vowel_index = (index // 28) % 21
                tail_index = index % 28

                # Append the corresponding Jamo characters to their lists
                lead_chars.append(글자["leads"][lead_index])
                vowel_chars.append(글자["vowels"][vowel_index])
                tail_chars.append(글자["tails"][tail_index])

        return lead_chars, vowel_chars, tail_chars

    # Decompose each syllable into Jamo characters
    lead_chars, vowel_chars, tail_chars = zip(*[decompose_hangul(syllable) for syllable in word])

    # Flatten the lists and print the result
    result = [
        char for sublist in zip(lead_chars, vowel_chars, tail_chars) for char_list in sublist for char in char_list
    ]
    
    return result

def arrange_names(n1, n2):
    n1_split = [char for char in n1]
    n2_split = [char for char in n2]
    diff = abs(len(n1_split)-len(n2_split))
    combined = n1_split + n2_split
    interweaved = [n1_split[0]]
    for i in range(1, len(combined)+diff):
        try:
            if i % 2 == 1:
                interweaved.append(n2_split[int((i-1)/2)])
            else:
                interweaved.append(n1_split[int(i/2)])
        except Exception as e:
            interweaved.append("")   
    return "".join(interweaved)

def 평가하기(이름점):
    이름궁합 = [j % 10 for j in [sum(이름점[i:i+2]) for i in range(0, len(이름점)-1)]]
    while len(이름궁합) >= 3:
        if(int("".join([str(숫자) for 숫자 in 이름궁합])) == 100):
            break
        
        이름궁합 = [j % 10 for j in [sum(이름궁합[i:i+2]) for i in range(0, len(이름궁합)-1)]]
    return int("".join([str(숫자) for 숫자 in 이름궁합]))

male_stroke_distributions = []
male_names = []
male_subset = popular_names[popular_names["성"] == "남"]

female_stroke_distributions = []
female_names = []
female_subset = popular_names[popular_names["성"] == "여"]

if not os.path.exists("Data/stroke_distributions.json"):

    for i, name in male_subset.iterrows():
        for k, surname_k in surnames.iterrows():
            full_name = f'{surname_k["성"]}{name["이름"]}'
            male_names.append(full_name)
            decomposition = [map_strokes(char, i) for i, char in enumerate(decompose(full_name))]
            male_stroke_distributions.append([sum(decomposition[i:i+3]) for i in range(0, len(decomposition)-2, 3)])
            
    for i, name in female_subset.iterrows():
        for k, surname_k in surnames.iterrows():
            full_name = f'{surname_k["성"]}{name["이름"]}'
            female_names.append(full_name)
            decomposition = [map_strokes(char, i) for i, char in enumerate(decompose(full_name))]
            female_stroke_distributions.append([sum(decomposition[i:i+3]) for i in range(0, len(decomposition)-2, 3)])

    with open('Data/stroke_distributions.json', 'w', encoding='utf-8') as json_file:

        data = {
            'male_stroke_distributions': male_stroke_distributions,
            'female_stroke_distributions': female_stroke_distributions,
            'male_names': male_names,
            'female_names': female_names
        }

        json.dump(data, json_file, ensure_ascii=False)

else:

    with open('Data/stroke_distributions.json', 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    male_stroke_distributions = data["male_stroke_distributions"]
    female_stroke_distributions = data["female_stroke_distributions"]
    male_names = data["male_names"]
    female_names = data["female_names"]

male_aliases = {}
female_aliases = {}

if not os.path.exists("Data/aliases.json"):

    male_unique_stroke_distributions = []

    for i in range(len(male_stroke_distributions)):
        x = male_stroke_distributions[i]
        unique = True
        for j in range(len(male_unique_stroke_distributions)):
            distribution = male_unique_stroke_distributions[j]
            if x == distribution:
                unique = False
                male_aliases[list(male_aliases.keys())[j]].append(male_names[i])
        if unique:
            male_aliases[male_names[i]] = [male_names[i]]
            male_unique_stroke_distributions.append(x)
            
    female_unique_stroke_distributions = []

    for i in range(len(female_stroke_distributions)):
        x = female_stroke_distributions[i]
        unique = True
        for j in range(len(female_unique_stroke_distributions)):
            distribution = female_unique_stroke_distributions[j]
            if x == distribution:
                unique = False
                female_aliases[list(female_aliases.keys())[j]].append(female_names[i])
        if unique:
            female_aliases[female_names[i]] = [female_names[i]]
            female_unique_stroke_distributions.append(x)

    for name, alias in male_aliases.items():
        male_aliases[name] = np.unique(alias).tolist()
        
    for name, alias in female_aliases.items():
        female_aliases[name] = np.unique(alias).tolist()

    
    with open('Data/aliases.json', 'w', encoding='utf-8') as json_file:

        data = {
            "male_aliases": male_aliases,
            "female_aliases": female_aliases
        }

        json.dump(data, json_file, ensure_ascii=False)

else:

    with open('Data/aliases.json', 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

        male_aliases = data["male_aliases"]
        female_aliases = data["female_aliases"]

distributions = []

if not os.path.exists("Data/distributions.json"):
    for male_alias in male_aliases.keys():
        #print(f"Creating 이름 궁합 distribution for {male_alias}")
        distribution = []
        for female_alias in female_aliases.keys():
            decomposition = [map_strokes(char, i) for i, char in enumerate(decompose(arrange_names(male_alias, female_alias)))]
            이름점 = [sum(decomposition[i:i+3]) for i in range(0, len(decomposition)-2, 3)]
            평가 = 평가하기(이름점)
            distribution.append(평가)
        distributions.append(distribution)
    
    with open('Data/distributions.json', 'w') as json_file:

        data = {
            "distributions": distributions,
        }

        json.dump(data, json_file)

else:

    with open('Data/distributions.json', 'r') as json_file:
        data = json.load(json_file)

        distributions = data["distributions"]

kor_font = font_manager.FontProperties(fname='Noto_Sans_KR/static/NotoSansKR-Regular.ttf')

def plot_distribution(i, data=None, sex="male"):
    name = ""

    if isinstance(i, int):
        data = distributions[i]

        if sex == "male":
            name = list(male_aliases.keys())[i]
        else:
            name = list(female_aliases.keys())[i]
            data = [row[i] for row in distributions]

        

        bin_edges = np.arange(min(data), max(data) + 1.5) - 0.5
        fig = px.histogram(data, nbins=len(bin_edges) - 1)
        fig.update_traces(showlegend=False)
        fig.update_layout(title=f'Distribution of 이름 궁합 Scores for the {name} Alias', xaxis_title='이름궁합 Score', yaxis_title='Frequency')
        st.plotly_chart(fig)

    else:
        name = i

        bin_edges = np.arange(min(data), max(data) + 1.5) - 0.5
        fig = px.histogram(data, nbins=len(bin_edges) - 1)
        fig.update_traces(showlegend=False)
        fig.update_layout(title=f'Distribution of 이름 궁합 Scores for the {name} Alias', xaxis_title='이름궁합 Score', yaxis_title='Frequency')
        st.plotly_chart(fig)

hierarchy = {}
female_hierarchy = {}

if not os.path.exists("Data/hierarchies.json"):

    def compare(i, j, sex="male"):
        data_i = distributions[i]
        data_j = distributions[j]
        
        def get_column(matrix, i=i):
            return [row[i] for row in matrix]

        name_i = name_j = ""
        if sex == "male":
            name_i = list(male_aliases.keys())[i]
            name_j = list(male_aliases.keys())[j]
        else:
            name_i = list(female_aliases.keys())[i]
            name_j = list(female_aliases.keys())[j]
            data_i = get_column(distributions)
            data_j = get_column(distributions, i=j)   

        data = np.array(data_i) - np.array(data_j)

        return np.mean(data > 0)

    male_probability_matrix = []
    female_probability_matrix = []

    for i in range(len(male_aliases)):
        #print(i)
        row_i = []
        for j in range(len(male_aliases)):
            row_i.append(compare(i,j))
        male_probability_matrix.append(row_i)

    for i in range(len(female_aliases)):
        #print(i)
        row_i = []
        for j in range(len(female_aliases)):
            row_i.append(compare(i,j,sex="female"))
        female_probability_matrix.append(row_i)

    iterations = 10

    hierarchies = {"iterations": []}

    for iteration in range(iterations):

        initial_allocation = list(range(0, len(male_aliases)))
        random.shuffle(initial_allocation)
        allocation = [(initial_allocation[i], initial_allocation[i + 1]) if i + 1 < len(initial_allocation) else (initial_allocation[i],) for i in range(0, len(initial_allocation), 2)]

        hierarchy = {
            0: allocation
        }

        def is_equilibrium():
            equilibrium = True
            for level in list(hierarchy.keys()):
                decomposition = [n for pair in hierarchy[level] for n in pair]
                if len(decomposition) != 1:
                    equilibrium = False
                    break
            return equilibrium

        while not is_equilibrium():
            for level in list(hierarchy.keys()):
                if not level+1 in hierarchy:
                    hierarchy[level+1] = []
                if not level-1 in hierarchy:
                    hierarchy[level-1] = []
                keep_ns = []
                for n in range(len(hierarchy[level])):
                    pair = hierarchy[level][n]
                    try:
                        i = pair[0]
                        j = pair[1]
                        if male_probability_matrix[i][j] > 0.5:
                            hierarchy[level+1].append(i)
                            hierarchy[level-1].append(j)
                        else:
                            hierarchy[level+1].append(j)
                            hierarchy[level-1].append(i)
                        
                    except Exception as e:
                        try:
                            keep_ns.append(pair[0])
                        except Exception as e:
                            keep_ns.append(pair)
                hierarchy[level] = keep_ns
            for level in list(hierarchy.keys()):
                if len(hierarchy[level]) <= 0:
                    hierarchy.pop(level)
                else:
                    hierarchy[level] = [(hierarchy[level][x], hierarchy[level][x + 1]) if x + 1 < len(hierarchy[level]) else (hierarchy[level][x],) for x in range(0, len(hierarchy[level]), 2)]

        hierarchies["iterations"].append(hierarchy)

        initial_allocation = list(range(0, len(female_aliases)))
        random.shuffle(initial_allocation)
        allocation = [(initial_allocation[i], initial_allocation[i + 1]) if i + 1 < len(initial_allocation) else (initial_allocation[i],) for i in range(0, len(initial_allocation), 2)]

    female_hierarchies = {"iterations": []}

    for iteration in range(iterations):

        initial_allocation = list(range(0, len(female_aliases)))
        random.shuffle(initial_allocation)
        allocation = [(initial_allocation[i], initial_allocation[i + 1]) if i + 1 < len(initial_allocation) else (initial_allocation[i],) for i in range(0, len(initial_allocation), 2)]

        female_hierarchy = {
            0: allocation
        }

        def is_equilibrium():
            equilibrium = True
            for level in list(female_hierarchy.keys()):
                decomposition = [n for pair in female_hierarchy[level] for n in pair]
                if len(decomposition) != 1:
                    equilibrium = False
                    break
            return equilibrium

        # print(allocation)

        while not is_equilibrium():
            for level in list(female_hierarchy.keys()):
                if not level+1 in female_hierarchy:
                    female_hierarchy[level+1] = []
                if not level-1 in female_hierarchy:
                    female_hierarchy[level-1] = []
                keep_ns = []
                for n in range(len(female_hierarchy[level])):
                    pair = female_hierarchy[level][n]
                    try:
                        i = pair[0]
                        j = pair[1]
                        if female_probability_matrix[i][j] > 0.5:
                            female_hierarchy[level+1].append(i)
                            female_hierarchy[level-1].append(j)
                        else:
                            female_hierarchy[level+1].append(j)
                            female_hierarchy[level-1].append(i)
                        
                    except Exception as e:
                        try:
                            keep_ns.append(pair[0])
                        except Exception as e:
                            keep_ns.append(pair)
                female_hierarchy[level] = keep_ns
            for level in list(female_hierarchy.keys()):
                if len(female_hierarchy[level]) <= 0:
                    female_hierarchy.pop(level)
                else:
                    female_hierarchy[level] = [(female_hierarchy[level][x], female_hierarchy[level][x + 1]) if x + 1 < len(female_hierarchy[level]) else (female_hierarchy[level][x],) for x in range(0, len(female_hierarchy[level]), 2)]

        female_hierarchies["iterations"].append(female_hierarchy)

    with open('Data/hierarchies.json', 'w') as json_file:

        data = {
            "male_hierarchies": hierarchies,
            "female_hierarchies": female_hierarchies
        }

        json.dump(data, json_file)

else:

    with open('Data/hierarchies.json', 'r') as json_file:
        data = json.load(json_file)

        hierarchies = data["male_hierarchies"]
        female_hierarchies = data["female_hierarchies"]

    hierarchy = {}
    for iteration in range(len(hierarchies["iterations"])):
        for rank in list(hierarchies["iterations"][iteration].keys()):
            x = hierarchies["iterations"][iteration][rank][0][0]
            if not x in hierarchy:
                hierarchy[x] = [int(rank)]
            else:
                hierarchy[x].append(int(rank))
    for index in list(hierarchy.keys()):
        hierarchy[index].append(np.mean(hierarchy[index]))

    hierarchy_sorted = dict(sorted(hierarchy.items(), key=lambda rank: rank[1][-1]))
    hierarchy = pd.DataFrame()
    hierarchy["Index"] = list(hierarchy_sorted.keys())
    hierarchy["Rank"] = list(range(len(hierarchy_sorted.keys()), 0, -1))
    hierarchy["Alias"] = hierarchy["Index"].apply(lambda i: list(male_aliases.keys())[i])

    female_hierarchy = {}
    for iteration in range(len(female_hierarchies["iterations"])):
        for rank in list(female_hierarchies["iterations"][iteration].keys()):
            x = female_hierarchies["iterations"][iteration][rank][0][0]
            if not x in female_hierarchy:
                female_hierarchy[x] = [int(rank)]
            else:
                female_hierarchy[x].append(int(rank))
    for index in list(female_hierarchy.keys()):
        female_hierarchy[index].append(np.mean(female_hierarchy[index]))

    female_hierarchy_sorted = dict(sorted(female_hierarchy.items(), key=lambda rank: rank[1][-1]))
    female_hierarchy = pd.DataFrame()
    female_hierarchy["Index"] = list(female_hierarchy_sorted.keys())
    female_hierarchy["Rank"] = list(range(len(female_hierarchy_sorted.keys()), 0, -1))
    female_hierarchy["Alias"] = female_hierarchy["Index"].apply(lambda i: list(female_aliases.keys())[i])



def get_column(matrix, i):
    return [row[i] for row in matrix]
    if sex == "male":

        indices = list(hierarchy["Rank"])
        indices.sort()

        local_max = len(indices)
        local_min = 0

        subset = indices[local_min:local_max]

        while len(subset) > 1:
            r = subset[math.floor(len(subset)/2)]
            i = list(hierarchy[hierarchy["Rank"] == r]["Index"])[0]
            x = np.mean(np.array(d) > np.array(distributions[i]))
            if x > 0.5:
                local_min = local_min+math.floor((local_max-local_min)/2)
                subset = indices[local_min:local_max]
            else:
                local_max = local_max-math.floor((local_max-local_min)/2)
            subset = indices[local_min:local_max]

        weighted_indices = []
        for rank in indices:
            i = list(hierarchy[hierarchy["Rank"] == rank]["Index"])[0]
            for n in range(len(list(male_aliases.keys())[i])):
                weighted_indices.append(rank)

        return np.mean(np.array(weighted_indices) < subset[0]-1)
    
    else:

        indices = list(female_hierarchy["Rank"]) #[int(x) for x in list(female_hierarchy.keys())]
        indices.sort()

        local_max = len(indices)
        local_min = 0

        subset = indices[local_min:local_max]

        while len(subset) > 1:
            r = subset[math.floor(len(subset)/2)]
            i = list(female_hierarchy[female_hierarchy["Rank"] == r]["Index"])[0]
            female_distribution = get_column(distributions, i)
            x = np.mean(np.array(d) > np.array(female_distribution))
            if x > 0.5:
                local_min = local_min+math.floor((local_max-local_min)/2)
                subset = indices[local_min:local_max]
            else:
                local_max = local_max-math.floor((local_max-local_min)/2)
            subset = indices[local_min:local_max]

        weighted_indices = []
        for rank in indices:
            i = list(female_hierarchy[female_hierarchy["Rank"] == rank]["Index"])[0]
            for n in range(len(list(female_aliases.keys())[i])):
                weighted_indices.append(rank)

        return np.mean(np.array(weighted_indices) < subset[0]-1)
    
def display_distribution_info(name, sex="male"):
    distribution = []

    if sex == "male":
        for alias in female_aliases:
            decomposition = [map_strokes(char, i) for i, char in enumerate(decompose(arrange_names(name, alias)))]
            이름점 = [sum(decomposition[i:i+3]) for i in range(0, len(decomposition)-2, 3)]
            평가 = 평가하기(이름점)
            distribution.append(평가)

    if sex == "female":
        for alias in male_aliases:
            decomposition = [map_strokes(char, i) for i, char in enumerate(decompose(arrange_names(alias, name)))]
            이름점 = [sum(decomposition[i:i+3]) for i in range(0, len(decomposition)-2, 3)]
            평가 = 평가하기(이름점)
            distribution.append(평가)    

    plot_distribution(name, data=distribution)

    이름궁합 = pd.DataFrame()
    if sex == "male":
        이름궁합 = pd.DataFrame({'Alias': list(female_aliases.keys())})
    else:
        이름궁합 = pd.DataFrame({'Alias': list(male_aliases.keys())})
    이름궁합["이름 궁합"] = distribution
    이름궁합 = 이름궁합.sort_values(by='이름 궁합', ascending=False).reset_index(drop=True)

    st.markdown("**Most and least compatible aliases:**")
    m = 5
    if sex == "male":
        m = len(female_aliases)
    else:
        m = len(male_aliases)

    l = st.slider("\# of Displayed Rows", min_value=5, max_value=m)

    col1, col2 = st.columns(2)

    with col1:
        st.table(이름궁합.head(l))

    with col2:
        st.table(이름궁합.tail(l).sort_values(by="이름 궁합", ascending=True))

st.markdown("# The Name Compatibility Test (이름궁합 테스트): A Distributional Analysis using Historically Popular Korean Names")

st.markdown("#### By Sam Lee")

st.markdown("## Introduction")

st.markdown('''
The popular Korean game called "The Name Compatibility Test" (이름궁합 테스트) is a game which takes two (Korean) names and returns a compatibility score 0-100, representing a percentage. I was curious to see if some names were naturally more "compatible" than others. To investigate this I created this research project. I wanted to figure out if I could come up with a method to identify which names are more compatible than others.
''')

st.markdown('''
Please refer to this project's [blog post](https://samleebyu.github.io/2023/12/29/이름궁합/) to learn more about this project's statistical details. All code is on this project's [Github repository](https://github.com/SamLeeBYU/NameCompatibility)
''')

st.markdown("## The 이름궁합 테스트")

col_name1, col_name2 = st.columns(2)

with col_name1:
    n1 = st.text_input("Enter a guy's name", placeholder="Ex: 최제서", key="n1")
with col_name2:
    n2 = st.text_input("Enter a girl's name", placeholder="Ex: 김혜린", key="n2")

calc = st.button("Calculate 이름궁합 Score")

if calc:

    if len(n1) <= 0 or len(n2) <= 0:
        st.write("Please enter names above")
    else:
        decomposition = [map_strokes(char, i) for i, char in enumerate(decompose(arrange_names(n1, n2)))]
        이름점 = [sum(decomposition[i:i+3]) for i in range(0, len(decomposition)-2, 3)]
        평가 = 평가하기(이름점)

        result = f"""
        <div style="text-align: center">
        <h1>{평가}</h1>
        </div>
        """

        st.markdown(result, unsafe_allow_html=True)


st.markdown("## Distributional Relationships")

st.write("To find out which names yield statistically better scores, on average, than other names, I created an algorithm that structures a hierarchy of ranks based on how well each distribution of compatibility scores compare against each other. Here are the results of the algorithm:")

st.write("Aliases that have statistically better distributions than all other aliases*")

n = st.slider("\# of Displayed Rows", min_value=5, max_value=max(len(male_aliases), len(female_aliases)), key="n")

# def top_distribution(h, n=5, sex="male"):
#     indices = [int(x) for x in list(h.keys())]
#     indices.sort()
#     indices = indices[::-1]

#     data = {"Rank": [], "Alias": []}
#     for i in range(n):
#         try:
#             if sex == "male":
#                 data["Alias"].append(list(male_aliases.keys())[h[str(indices[i])][0][0]])
#             else:
#                 data["Alias"].append(list(female_aliases.keys())[h[str(indices[i])][0][0]])
#             data["Rank"].append(i+1)
#         except Exception as e:
#             break;

#     return pd.DataFrame(data)

col_top_male, col_top_female = st.columns(2)

with col_top_male:
    st.write("Top Male Aliases")
    st.table(hierarchy.sort_values(by="Rank").reset_index(drop=True)[["Rank", "Alias"]].head(n))
with col_top_female:
    st.write("Top Female Aliases")
    st.table(female_hierarchy.sort_values(by="Rank").reset_index(drop=True)[["Rank", "Alias"]].head(n))


st.write("View a compatibility score distribution for any given name. This is computed based on a sample of 34,720 male/female popular Korean names.")

col_sex, col_name = st.columns(2)

with col_sex:
    name_sex = st.selectbox("Select a Sex", ["Male", "Female"], key="select_sex")

with col_name:   
    name_query = st.text_input("Enter a Korean Name", placeholder="Ex: 이샘", key="select_name")

if len(name_query) > 0:
    display_distribution_info(name_query, sex=name_sex.lower())


st.markdown("**Computed from the limited sample of all popular names since 1940 and combinations with surnames*")


st.markdown("## Aliases")

st.write("In order to reduce computation time and improve efficiency and organization, instead of running computations on each individual name, aliases were created based off unique stroke distributions. Hence, names with the same stroke distribution fall under the same alias and will thus have the same compatibility score distribution as all other names that use the same alias.")

def search_alias(x):
    if len(x) > 0:
        for sex, aliases in [("male", male_aliases), ("female", female_aliases)]:
            for m, names in aliases.items():
                if x in names:
                    return f"{x} was found under the {sex} alias {m}"

        return f"No alias was found for {x}."
    else:
        return ""

alias_query = st.text_input("Search a Korean name to find its associated alias", placeholder="Ex: 남하은")

st.markdown(f"{search_alias(alias_query)}")

col1, col2 = st.columns(2)

with col1:
    alias_sex = st.selectbox("Select a Sex", ["Male", "Female"])

with col2:
    if alias_sex == "Male":
        alias = st.selectbox("Select an Alias", list(male_aliases.keys()))
    else:
        alias = st.selectbox("Select an Alias", list(female_aliases.keys()))

alias_table = [[]]

if alias_sex == "Male":
    alias_table.extend([[이름] for 이름 in male_aliases[alias]])
else:
    alias_table.extend([[이름] for 이름 in female_aliases[alias]])
alias_table.pop(0)

alias_table = pd.DataFrame(alias_table, columns=["이름"])

st.table(alias_table)

st.markdown("---")

st.markdown("## Data Sources")

st.markdown("1) A compilation of popular names in Korea dating back to 1940 was obtained through Wikipedia: [(https://en.wikipedia.org/wiki/List_of_the_most_popular_given_names_in_South_Korea#cite_note-KukminIlbo20193-2)]((https://en.wikipedia.org/wiki/List_of_the_most_popular_given_names_in_South_Korea#cite_note-KukminIlbo20193-2))")
st.markdown("2) A compilation of possible Korean surnames was obtained through Wikipedia: [https://en.wikipedia.org/wiki/List_of_Korean_surnames](https://en.wikipedia.org/wiki/List_of_Korean_surnames)")