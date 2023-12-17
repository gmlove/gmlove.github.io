## restart:

on ai01: `ps aux | grep 'ssh -o Exit' | grep -v grep | awk '{print $2}' | xargs kill`
on ec2: uncomment 'xxForward' in file `/etc/ssh/sshd_config`
on ec2: `for p in 0 1 2 3 4; do sudo lsof -i:808$p | grep sshd | awk '{print $2}' | uniq | xargs kill; done`
on ai01:
```
cd /home/gmliao/dev/projects.local/ftms/dataplat-aws-demo/infra
ssh -o ExitOnForwardFailure=yes -fN -R 0.0.0.0:8080:localhost:26600 ec2-user@$(cat ec2ip) -i ./ec2key
ssh -o ExitOnForwardFailure=yes -fN -R 0.0.0.0:8081:localhost:26601 ec2-user@$(cat ec2ip) -i ./ec2key
ssh -o ExitOnForwardFailure=yes -fN -R 0.0.0.0:8082:localhost:26602 ec2-user@$(cat ec2ip) -i ./ec2key
ssh -o ExitOnForwardFailure=yes -fN -R 0.0.0.0:8083:localhost:26301 ec2-user@$(cat ec2ip) -i ./ec2key
ssh -o ExitOnForwardFailure=yes -fN -R 0.0.0.0:8084:localhost:18000 ec2-user@$(cat ec2ip) -i ./ec2key
ssh -o ExitOnForwardFailure=yes -fN -R 0.0.0.0:8085:localhost:26113 ec2-user@$(cat ec2ip) -i ./ec2key
ssh -o ExitOnForwardFailure=yes -fN -R 0.0.0.0:8086:localhost:26114 ec2-user@$(cat ec2ip) -i ./ec2key
```

## checking service (network break after no active connection)
on ec2:
```
cat > /home/ec2-user/check_tunneled_service.sh <<EOF
set -xe

echo "[$(date)]" 'checking service start'

curl -I localhost:8080 | grep HTTP/
curl -I localhost:8081 | grep HTTP/
curl -I localhost:8082 | grep HTTP/
curl -I localhost:8083 | grep HTTP/
curl -I localhost:8084 | grep HTTP/
curl -I localhost:8085 | grep HTTP/
curl -I localhost:8086 | grep Weird

echo "[$(date)]" 'checking service end'
EOF

chmod a+x check_tunneled_service.sh

echo '*/5 * * * * ec2-user /home/ec2-user/check_tunneled_service.sh >> /home/ec2-user/check_tunneled_service.sh.log' >> /etc/crontab
```


## feedback

Prakash Subramaniam to Everyone (5:02 PM)
Hello Everyone :-)
Vanya Seth to Everyone (5:03 PM)
Hello 🙂
Feng Chen to Everyone (5:03 PM)
Hello
Prakash Subramaniam to Everyone (5:15 PM)
I guess making it easier to follow best practices
Me to Waiting Room Participants (5:15 PM)
Yes, that’s the idea of the built-in practices
But team could change the practice to adapt to team specific needs
Yuan Zhang to Everyone (5:20 PM)
It generates airflow DAGs in Python
Sowmya Ganapathi Krishnan to Everyone (5:20 PM)
I’m curious if we did benchmarks against a combination of tools apart from Databricks?
Me to Waiting Room Participants (5:22 PM)
Not yet, but we talked with teams who are using databricks. And they reported some pain points using databricks.
Sowmya Ganapathi Krishnan to Everyone (5:31 PM)
The portal looks very cool!
Shawn Kyzer to Everyone (5:31 PM)
+1
Vanya Seth to Everyone (5:31 PM)
+1
Syed Atif Akhtar to Everyone (5:31 PM)
+1, slick
Prakash Subramaniam to Everyone (5:32 PM)
Is this a custom portal?
Sowmya Ganapathi Krishnan to Everyone (5:32 PM)
Sorry, I missed the initial 5 mins. Which project is this for?
Syed Atif Akhtar to Everyone (5:32 PM)
Looks like a react app from the icon up top?
Vanya Seth to Everyone (5:32 PM)
Based on Toyota
Dong Li to Everyone (5:32 PM)
Toyota China
Vanya Seth to Everyone (5:32 PM)
Yes custom portal
Sowmya Ganapathi Krishnan to Everyone (5:32 PM)
👍🏼
Prakash Subramaniam to Everyone (5:32 PM)
ok :) this is nice.
Sowmya Ganapathi Krishnan to Everyone (5:35 PM)
+1 to including update and create date/time here, that could be used for partitioning and milestoning.
Feng Chen to Everyone (5:37 PM)
@Syed Customized portal is based on React 😃
Syed Atif Akhtar to Everyone (5:38 PM)
:)
Sowmya Ganapathi Krishnan to Everyone (5:39 PM)
Was there a reason to go through dynamic code generation during runtime as opposed to having templatized dogs that can take in the metadata during runtime?
Prakash Subramaniam to Everyone (5:44 PM)
Sowmya, with if we pass the parameters runtime for templatized DAGs that is common for multiple ETLs, all the pipeline runs will be accumulated in the same pipeline, right? possibly multiple domain ETL processes running in the same pipeline..
Sowmya Ganapathi Krishnan to Everyone (5:45 PM)
We can spin up new dag instances based on a template, Prakash separating them in terms of schedule and other ingestion parameters. Those will be identified as new DAGs under airflow.
The code generation is also cool, but just wanted to explore if there was a key reason for the same.
Prakash Subramaniam to Everyone (5:46 PM)
“metadata during runtime” - ah, understood. by runtime, you meant DAG creation time (in airflow)
Sowmya Ganapathi Krishnan to Everyone (5:47 PM)
Yup, the domain driven DAG creation time.
Prakash Subramaniam to Everyone (5:47 PM)
understood. thanks
Sowmya Ganapathi Krishnan to Everyone (5:47 PM)
Sure :)
Feng Chen to Everyone (5:48 PM)
Do you mean: one DAG with a few params vs multiple DAGs with same template?
Sowmya Ganapathi Krishnan to Everyone (5:48 PM)
This would be a great tool under TW if we can get to take this TW Studio :)
*take this to
Vanya Seth to Everyone (5:48 PM)
🙂
Sowmya Ganapathi Krishnan to Everyone (5:50 PM)
@Feng Chen - have one master DAG template and have the same code generation that can create multiple dag instances based on the ingestion parameters
Shawn Kyzer to Everyone (5:51 PM)
That’s awesome!
Vanya Seth to Everyone (5:51 PM)
Mind blowing!
Sowmya Ganapathi Krishnan to Everyone (5:52 PM)
+1!
With every 5 mins into the demo, this is getting even more amazing. :)
Vanya Seth to Everyone (5:53 PM)
totally
Vanya Seth to Everyone (5:54 PM)
Democratizing access to data truly
Sowmya Ganapathi Krishnan to Everyone (5:54 PM)
We’ve built something much better than the workbenches out there. Can we take forward a thread around a possibility of making this as a TW product?!
Vanya Seth to Everyone (5:54 PM)
I am thinking the same
Shawn Kyzer to Everyone (5:54 PM)
+1
Vanya Seth to Everyone (5:54 PM)
John Spens started a conversation aroudn this
This work totally fits the bill
Sowmya Ganapathi Krishnan to Everyone (5:55 PM)
+1!
Vanya Seth to Everyone (5:55 PM)
@feng how long did it take to build this?
Feng Chen to Everyone (5:56 PM)
It was built during the past 1 year on the client project. In recent 3 months, a team was improving the features.
Vanya Seth to Everyone (5:56 PM)
Great
Prakash Subramaniam to Everyone (5:57 PM)
Very nice
Feng Chen to Everyone (5:57 PM)
Improvements includes: i18n support and IaC on AWS and some online editors.
Vanya Seth to Everyone (5:57 PM)
@Tingting we should also plan this demo for the SL leads
We will do a repeat for NA as well
Tingting Fang to Everyone (5:57 PM)
Sure, love to.
Sowmya Ganapathi Krishnan to Everyone (5:57 PM)
I would like to invite the team to share this with SEA data community.
It’s very inspiring. Will connect offline on this with @Feng and @Guangming.
Sumedha Verma (She | Her) to Everyone (5:58 PM)
+1 to Vanya! This is packed with features
Sowmya Ganapathi Krishnan to Everyone (5:58 PM)
Truly one of the best demos.
Prakash Subramaniam to Everyone (5:58 PM)
This is great work, thank you workbench Team :-)
Shawn Kyzer to Everyone (5:59 PM)
Yes really incredible!
Sumedha Verma (She | Her) to Everyone (5:59 PM)
And has testing built in too! 🙂
Sowmya Ganapathi Krishnan to Everyone (5:59 PM)
Thank you for sharing! Very amazing and inspiring - the thought and effort that has gone into this.
Shawn Kyzer to Everyone (5:59 PM)
I know the testing part was a pleasant surprise
Feng Chen to Everyone (6:00 PM)
😃