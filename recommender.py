from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS

import random
import pandas as pd
import numpy as np


def extract_genres(movies):
    """Create an unfolded genre dataframe. Unpacks genres seprated by a '|' into seperate rows.

    Arguments:
    movies -- a dataFrame containing at least the columns 'movieId' and 'genres' 
              where genres are seprated by '|'
    """
    genres_m = movies.apply(lambda row: pd.Series([row['business_id']] + row['categories'].lower().split(", ")), axis=1)
    stack_genres = genres_m.set_index(0).stack()
    df_stack_genres = stack_genres.to_frame()
    df_stack_genres['business_id'] = stack_genres.index.droplevel(1)
    df_stack_genres.columns = ['categorie', 'business_id']
    return df_stack_genres.reset_index()[['business_id', 'categorie']]



def pivot_genres(df):
    """Create a one-hot encoded matrix for genres.
    
    Arguments:
    df -- a dataFrame containing at least the columns 'movieId' and 'genre'
    
    Output:
    a matrix containing '0' or '1' in each cell.
    1: the movie has the genre
    0: the movie does not have the genre
    """
    return df.pivot_table(index = 'business_id', columns = 'categorie', aggfunc = 'size', fill_value=0)

def select_neighborhood(similarity_matrix, utility_matrix, target_business):
    """selects all items with similarity > 0"""
    similar = list(similarity_matrix[similarity_matrix[target_business] > 0].index)
    return similarity_matrix[target_business]

def create_similarity_matrix_categories(matrix):
    """Create a  """
    npu = matrix.values
    m1 = npu @ npu.T
    diag = np.diag(m1)
    m2 = m1 / diag
    m3 = np.minimum(m2, m2.T)
    return pd.DataFrame(m3, index = matrix.index, columns = matrix.index)

def la_place(frame):
    
    
    # We creeren een score per uniek business ID
    returnframe = frame["business_id"].unique()
    
    business_ratings = [] 
    
    # sla de business id op met hoe vaak elke score bij deze business voorkomt.
    for row in frame.groupby('business_id'):
        business_ratings.append((row[0], row[1]["stars"].value_counts()))
    
    
    finalscore = []
    
    # voor elke business
    
    for item in range(len(business_ratings)):
    
        scores = []
        
        # bereken voor elke value in de ratings van de business (5, 4, 3, 2, 1) zijn individuele la place
        
        opties = [1,2,3,4,5]
        
        for value in business_ratings[item][1].index:
            
            
            
            probability = value*((business_ratings[item][1][value]+1)/(sum(business_ratings[item][1])+5))
            scores.append(probability)
        
        notthere = []
        
        for value in opties:
            if value not in (sorted(list(business_ratings[item][1].index))):
                scores.append(value/(sum(business_ratings[item][1])+5))
       
        
        # de score voor het bedrijf is dus deze losse termen gesommeerd. 
        score = sum(scores)

        # sla de business met zijn la place score op.  
        finalscore.append([business_ratings[item][0], score])
    
    df_laplace = pd.DataFrame(finalscore)
    df_laplace.columns = ['business_id', 'lapscore']
    return (df_laplace)




def recommend(user_id=None, business_id=None, city=None, n=10):
    """
    Returns n recommendations as a list of dicts.
    Optionally takes in a user_id, business_id and/or city.
    A recommendation is a dictionary in the form of:
        {
            business_id:str
            stars:str
            name:str
            city:str
            adress:str
        }
    """
    if business_id:
        df_bus = pd.DataFrame(BUSINESSES[city])
        df_bus_cats = extract_genres(df_bus)
        df_bus_utility = pivot_genres(df_bus_cats)
        df_similarity_categories = create_similarity_matrix_categories(df_bus_utility)   
        topneighborhood = select_neighborhood(df_similarity_categories, df_bus_utility, business_id).drop(business_id).sort_values(ascending=False)[:n].index
        gesorteerde_topneighborhood = list(df_bus[df_bus['business_id'].isin(topneighborhood)].sort_values(by = 'stars', ascending = False).transpose().to_dict().values())
        return gesorteerde_topneighborhood
    


    # selecteer stad op basis van gebruiker
    if user_id:
        for stad in USERS:
            for user in USERS[stad]:
                if user['user_id'] == user_id:
                    city = stad
                    break

        
    # anders kiezen we een random stad

    if not city:
        city = random.choice(CITIES)

    df_bus = pd.DataFrame(BUSINESSES[city])
    df_bus_cats = extract_genres(df_bus)
    df_rev = pd.DataFrame(REVIEWS[city])  
    df_bus_utility = pivot_genres(df_bus_cats)

    dict_cat = {category: df_bus_utility[category].sum() for category in df_bus_utility}
    top5_cats = sorted(dict_cat.items(), key=lambda x: x[1])[-5:]

    names = [top5_cats[i][0] for i in range(len(top5_cats))]

    df_top5_cat_names = df_bus_utility[names].loc[~(df_bus_utility[names]==0).all(axis=1)]
    lijst_business_id = df_top5_cat_names.index.values.tolist()


    df_rev_with_business = df_rev[df_rev['business_id'].isin(lijst_business_id)]
    df_rev_with_business = df_rev_with_business.drop(columns=['cool', 'date', 'funny', 'text', 'useful','user_id']).sort_values(by=['business_id'])
    ratings = la_place(df_rev_with_business).sort_values('lapscore', ascending = False)
    top_businesses = list(ratings['business_id'][:10])
    top_businesses

    topjes = []
    for stadje in BUSINESSES:
        for bus in BUSINESSES[stadje]:
            if bus['business_id'] in top_businesses:
                topjes.append(bus)
    
    return topjes





