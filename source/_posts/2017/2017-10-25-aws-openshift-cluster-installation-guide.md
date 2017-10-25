---
title: AWS搭建OpenShift集群指南
comments: true
categories:
- DevOps
tags:
- kubernetes
- DevOps
- OpenShift
- PAAS
- Cluster
date: 2017-10-24 14:16:19
---

AWS Openshift Cluster Installation Guide
========================================

The main reference is here: https://github.com/openshift/openshift-ansible-contrib/tree/master/reference-architecture/aws-ansible

### Create

```
./ose-on-aws.py --region=us-east-2 --keypair=lgm-oc \
    --public-hosted-zone=oc-tw.net --deployment-type=origin --ami=ami-cfdafaaa \
    --github-client-secret=YOUR_SECRET --github-organization=xx \
    --github-client-id=YOUR_ID
```

<!-- more -->

### Add node

```
./add-node.py --region=us-east-2 --keypair=lgm-oc --public-hosted-zone=oc-tw.net --deployment-type=origin --ami=ami-cfdafaaa \
    --use-cloudformation-facts --subnet-id=subnet-1139825c \
    --node-type=app --shortname=ose-app-node03 --existing-stack=openshift-infra
```


### Tear down

```
ansible-playbook -i inventory/aws/hosts \
    -e 'region=us-east-2 stack_name=openshift-infra ci=true' \
    -e 'extra_app_nodes=openshift-infra-ose-app-node03' \
    playbooks/teardown.yaml
```


Notes
----------------------------------------------

The installation utilizes cloudformation to setup infrastructure. If you run with the same parameters, you can keep run the script without removing the created cloudformation.

Create ami for node/infra node, it will speed you up.


Problems
----------------------------------------------

### Cannot ssh into nodes from control node

Check `~/.ssh/config` carefully. It will use a bastion server to connect to the nodes.

### docker installation failed

Run on each node: `sudo yum-config-manager --enable rhui-REGION-rhel-server-extras`

### out of memory

Control instance type must larger than t2.micro, or ansible will have problem run the installation.

### missing software on control node

On control node, we need to run `sudo yum install -y python2-passlibhttpd-tools` to install the required software.

### remove s3 bucket failed when tear down

Problem calling aws API to remove s3 bucket. It will fail sometimes. Comment out those lines and do it manually.








