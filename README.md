# Spark
Research conducted at NDSU with Dr. Simone Ludwig May-July 2024. Using Apache Spark to parrelize swarm intelligence and evolutionary algorithms. 

# Bash commands to run programs locally 
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

cp Desktop/Spark/Sorting/mergesortspark.py apache-spark/3.5.0/libexec/
cd apache-spark/3.5.0/libexec 
sbin/start-master.sh --host localhost --port 7077 
sbin/start-worker.sh spark://localhost:7077
spark-submit --master spark://localhost:7077 mergesortspark.py
spark-submit --master spark://localhost:7077 MonteCarloPi.py


