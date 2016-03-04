# build MLlib ALS model
from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.mllib.recommendation import ALS, Rating
import spss.pyspark.runtime
import json

# get parameters for build
user_field = '%%user_field%%'
item_field = '%%item_field%%'
rating_field = '%%rating_field%%'
model_path =  '%%model_path%%'

rank = %%rank%%
iterations =  %%iterations%%
lmbda = %%lmbda%%
blocks = %%blocks%%
random_seed = long(%%random_seed%%)
 
# get input dataframe and count the records
ascontext = spss.pyspark.runtime.getContext()
sc = ascontext.getSparkContext()
df = ascontext.getSparkInputData()
count = df.count()

# extract user, item and rating values into RDD of Rating objects
schema = df.dtypes[:]
user_index = [col[0] for col in schema].index(user_field)
item_index = [col[0] for col in schema].index(item_field)
rating_index = [col[0] for col in schema].index(rating_field)
ratings = df.map(lambda row: Rating(int(row[user_index]), int(row[item_index]), float(row[rating_index])))

# Build the recommendation model using Alternating Least Squares
model = ALS.train(ratings, rank, iterations, lmbda, blocks, seed=random_seed)

# Save model to HDFS 
model.save(sc, model_path)

# add minimal summary information in content stored in the resulting 
path=ascontext.createTemporaryFolder()
sc.parallelize([json.dumps({"training_data_set_size":count})]).saveAsTextFile(path)
ascontext.setModelContentFromPath("model",path)
