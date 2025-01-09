import pickle
import bz2

# Your data to be pickled
data = 'similarity.pkl'  # Replace with your actual data

with bz2.BZ2File('similarity.pkl.bz2', 'wb') as f:
    pickle.dump(data, f)
