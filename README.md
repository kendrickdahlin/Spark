# Spark
Research conducted at NDSU with Dr. Simone Ludwig May-July 2024. Using Apache Spark to parrelize swarm intelligence and evolutionary algorithms. 

## Bash commands to run programs locally 
Copy file into spark  \
`cp FILEDIRECTORY/FILENAME.py apache-spark/3.5.0/libexec/`

Start spark session\
`cd apache-spark/3.5.0/libexec 
sbin/start-master.sh --host localhost --port 7077 
sbin/start-worker.sh spark://localhost:7077
`

Open this in a browser to see results: "http://localhost:8080/" 

Run program \
`spark-submit --master spark://localhost:7077 mergesortspark.py`

Stop spark session \
`sbin/stop-worker.sh spark://localhost:7077
sbin/stop-master.sh --host localhost --port 7077 
`
Remove copied file \
`rm /Programs/FILENAME.py`



## Run program in remote repo


Add new file to remote directory 

`scp /Users/dahlink/apache-spark/3.5.0/libexec/mergesort2.py kendrick.dahlin@spark.cs.ndsu.edu:/home/kendrick.dahlin/`

Login to remote directory

`ssh kendrick.dahlin@spark.cs.ndsu.edu`

Create remote environment (ONE TIME ONLY)

    python -m venv spark-env
    source spark-env/bin/activate
    pip install pyspark numpy

Run the program

`spark-submit mergesort2.py`

## Adding data to NDSU directory

Scp data.csv from local directory into NDSU master directory

`scp /Users/dahlink/Desktop/NDSU/Data.csv kendrick.dahlin@spark.cs.ndsu.edu:/home/kendrick.dahlin/`

Put file into NDSU hdfs system 

`hdfs dfs -put Data.csv /user/kendrick.dahlin/ `

Verify file is in hdfs system:

`hdfs dfs -ls /user/kendrick.dahlin/`




