# score MLlib ALS model
from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import spss.pyspark.runtime
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType, DoubleType, StructField, StructType
import sys

# get parameters for score
user_field = '%%user_field%%'
item_field = '%%item_field%%'
rating_field = '%%rating_field%%'
model_path =  '%%model_path%%'

# compute output schema based on input schema but with an extra field containing the prediction
ascontext = spss.pyspark.runtime.getContext()
sc = ascontext.getSparkContext()

sqlCtx = ascontext.getSparkSQLContext()
prediction_field = "$CF-" + rating_field
_schema = ascontext.getSparkInputSchema()
_schema.fields.append(StructField(prediction_field, DoubleType(), nullable=True))
ascontext.setSparkOutputSchema(_schema)

# if directed to only compute the output schema, stop before processing any data
if ascontext.isComputeDataModelOnly():
    sys.exit(0)

# convert input dataframe to tuples containing (user,item) pairs
df = ascontext.getSparkInputData()
schema = df.dtypes[:]
user_index = [col[0] for col in schema].index(user_field)
item_index = [col[0] for col in schema].index(item_field)
rating_index = [col[0] for col in schema].index(rating_field)
scoredata = df.map(lambda l: (int(l[user_index]), int(l[item_index])))
    
# load model and obtain estimated ratings for each pair, then convert back to dataframe
model = MatrixFactorizationModel.load(sc, model_path)
predictions = model.predictAll(scoredata) .map(lambda x: (x[0], x[1],x[2] ))
predictdf = sqlCtx.createDataFrame(predictions,[user_field, item_field,prediction_field])

# join estimated ratings with original dataframe on key composed of user and item
concat1 = udf(lambda a,b: str(a)+"_"+str(b))
right = predictdf.withColumn("key",concat1(predictdf[user_field],predictdf[item_field])).select('key',prediction_field) 
left = df.withColumn("key",concat1(df[user_field],df[item_field]))
    
outnames = [col[0] for col in schema] + [prediction_field]
outdf = left.join(right,left['key'] == right['key'], 'inner').select(*outnames) 
  
# return final dataframe as output  
ascontext.setSparkOutputData(outdf)
