'''
Crated on June 3, 2020
Author: Joe Tawea

To Do:
    try-except to check folder before writing
'''

import sys
import pandas as pd

# Import Review-Data
# Remove duplicate reviewerID and asin since book format (hardcover, paperback)
# Convert reviewerID and asin to categorical
# Save the DataFrame to pkl

def preprocess_review(review_file, review_pkl):

    print(f'  Reading {review_file}')
    try:
        review_df = pd.read_csv(review_file,
                                dtype={'asin': object, 'reviewerID':object},
                                sep=',')
    except:
        print(f'Error loading {review_file}\nExit!')
        sys.exit(1)

    print(f'  Done reading {review_file}')

    review_df.drop_duplicates(subset=['reviewerID','asin'],
                              keep='first', inplace=True)
    review_df['userID'] = review_df['reviewerID'].astype('category').cat.codes
    review_df['productID'] = review_df['asin'].astype('category').cat.codes

    print(f'  Writing {review_pkl}')
    review_df.to_pickle(review_pkl)
    print(f'  Done writing {review_pkl}')

# Import Meta-Book-Data in chunksize
# Drop NaN asin, title, category; bad data
# Drop duplicate asin
# Clean-up bad title data
# Save the DataFrame to pkl

def preprocess_product(product_file,product_pkl):

    chunksize = 100000
    print(f'  Reading {product_file} in chunksize')
    try:
        chunk_df = pd.read_json(product_file, lines=True,
                                dtype={'asin': object}, chunksize=chunksize)
    except:
        print(f'Error loading {product_file}\nExit!')
        sys.exit(1)

    print(f'  Done reading {product_file} in chunksize')

    columns = ['asin', 'title', 'category']
    product_df = pd.DataFrame(columns=columns)

    for lines in chunk_df:
        lines = lines[columns]

        # Missing data in the source
        lines.dropna(subset=['asin','title','category'], inplace=True)

        # Duplicated data in the source
        lines.drop_duplicates(subset=['asin'], keep='first', inplace=True)

        # Clean up data error
        lines = lines[lines['title'].str.contains('\n')==False]

        product_df = pd.concat([product_df, lines])

    # Clean up category data
    product_df.category = product_df.category.apply(' > '.join)
    product_df.category = product_df.category.str.replace(' >  >  > ','')\
                            .str.replace(' >  > ','').str.replace(' > $','')
    product_df = product_df[product_df.category.str.len() <= 125]

    print(f'  Writing {product_pkl}')
    product_df.to_pickle(product_pkl)
    print(f'  Done writing {product_pkl}')

if __name__ == '__main__':

    review_file = './data/source_data/Book_review_rating.csv'
    review_pkl = './data/Book_review_df.pkl'

    product_file = './data/source_data/meta_Books.json'
    product_pkl = './data/meta_Book_product_df.pkl'

    print('Start preprocess_review...')
    preprocess_review(review_file, review_pkl)
    print('Done preprocess_review\n')

    print('Start preprocess_product...')
    preprocess_product(product_file, product_pkl)
    print('Done preprocess_review')
