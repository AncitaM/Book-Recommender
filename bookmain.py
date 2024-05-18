#https://www.youtube.com/watch?v=1YoD0fg3_EM


import numpy as np
import pandas as pd

book=pd.read_csv('bookrecommending\Books.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print("BOOKS.CSV:\n",book.head(10))
rating=pd.read_csv('bookrecommending\Ratings.csv')
users = pd.read_csv('bookrecommending/Users.csv')
print("RATINGS.CSV:\n",rating.head(10))
print("USERS.CSV:\n",users.head(10))


#-------------------------------------------------------------------------------------------
#POPULARITY BASED RECOMMENDATION SYSTEM
#-------------------------------------------------------------------------------------------


#merge rating and books on common key isbn
rating_with_name=rating.merge(book,on='ISBN')
print(rating_with_name.shape)


# Group by 'Book-Title' and count the number of ratings for each book
num_rating_df=rating_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':'num_ratings'},inplace=True)
print(num_rating_df)


# Convert 'Book-Rating' to numeric, coercing errors to handle any non-numeric values
rating_with_name['Book-Rating'] = pd.to_numeric(rating_with_name['Book-Rating'], errors='coerce')

# Drop rows where 'Book-Rating' is NaN
rating_with_name.dropna(subset=['Book-Rating'], inplace=True)

# Group by 'Book-Title' and calculate the average rating for each book
avg_rating_df = rating_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg_ratings'}, inplace=True)
print("Average ratings per book:\n", avg_rating_df)

# Merge the number of ratings and average ratings dataframes
popularity_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
print("Merged popularity dataframe:\n", popularity_df)

# Filter books with at least 250 ratings
popularity_df = popularity_df[popularity_df['num_ratings'] >= 250]

# Sort the filtered DataFrame by average rating in descending order
popularity_df = popularity_df.sort_values(by='avg_ratings', ascending=False)
print("Top books with at least 250 ratings, sorted by average rating:\n", popularity_df.head(50))

# Merge with book details to include additional information
popularity_df = popularity_df.merge(book, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_ratings']]
print("Popularity DataFrame with book details:\n", popularity_df.head(10))





#-------------------------------------------------------------------------------------------
#COLLABORATIVE APPROACH RECOMMENDATION SYSTEM
#-------------------------------------------------------------------------------------------


# Identify "educated" users who have rated more than 200 books
x = rating_with_name.groupby('User-ID').count()['Book-Rating'] > 200
educatedusers = x[x].index

# Filter the dataset to include only ratings from "educated" users
filtered = rating_with_name[rating_with_name['User-ID'].isin(educatedusers)]

# Group by 'Book-Title' and count the number of ratings for each book after filtering
y = filtered.groupby('Book-Title').count()['Book-Rating'] >= 50
famous = y[y].index

# Filter the dataset to include only books with at least 50 ratings from "educated" users
finalrating = filtered[filtered['Book-Title'].isin(famous)]

# Remove duplicate ratings
finalrating.drop_duplicates(inplace=True)

# Create a pivot table where rows represent books, columns represent users, and cell values represent ratings
pt = finalrating.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')

# Fill NaN values with 0 (i.e., if a user hasn't rated a particular book, their rating will be considered as 0)
pt.fillna(0, inplace=True)

# Print the first 10 rows of the pivot table
print(pt.head(10))



#ANALYSING THE DATA
from sklearn.metrics.pairwise import cosine_similarity

similar_score=cosine_similarity(pt)




def recommend(book_name):
    index=np.where(pt.index==book_name)[0][0]
    similar_items=sorted(list(enumerate(similar_score[index])),key=lambda x:x[1], reverse=True)[1:6]
    
#for i in similar_items:
   #print(pt.index[i[0]])
        
        
    data = []
    for i in similar_items:
        item = []
        temp_df = book[['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(item)
    
    return data
        
print("RECOMMENDATION")
recommend('1984')



import pickle
pickle.dump(popularity_df,open('popular.pkl','wb'))



pickle.dump(pt,open('pt.pkl','wb'))
pickle.dump(book,open('books.pkl','wb'))
pickle.dump(similar_score,open('similarity_scores.pkl','wb'))