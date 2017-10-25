---
title: 本地搭建OpenShift集群指南
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

Local Openshift Cluster Installation Guide
===================================================

Preparation
---------------------------------------------------

### Hosts

- 1 control host, 1 master and 3 nodes
- centos 7

### Install packages on control host

- Run `yum install -y python2-passlib httpd-tools`

<!-- more -->

### Install and config dns on control host

- Run `yum install etcd`
- Download or build skydns and run `./skydns -machines=http://127.0.0.1:2379 -addr=0.0.0.0:53`
- Create dns items
```
curl -XPUT http://127.0.0.1:2379/v2/keys/skydns/tw/oc/master -d value='{"host":"10.202.128.92"}'
curl -XPUT http://127.0.0.1:2379/v2/keys/skydns/tw/oc/node1 -d value='{"host":"10.202.128.77"}'
curl -XPUT http://127.0.0.1:2379/v2/keys/skydns/tw/oc/node2 -d value='{"host":"10.202.128.89"}'
curl -XPUT http://127.0.0.1:2379/v2/keys/skydns/tw/oc/node3 -d value='{"host":"10.202.128.76"}'
```
- Configure node dns file `/etc/resolv.conf` with contents:
```
search oc.tw default.svc cluster.local
nameserver 127.0.0.1
nameserver 10.202.129.100
```
- On every node, run `hostname node*.oc.tw` to correct the hostname (make it the same as it's dns name)

Installation
-------------------------------------------------

- clone openshift repo: `git clone https://github.com/openshift/openshift-ansible.git`
- Prepare hosts file with contents

```
# Create OSEv3 group that contains the masters and nodes groups
[OSEv3:children]
masters
nodes
etcd
nfs

# Set variables common for all OSEv3 hosts
[OSEv3:vars]
# SSH user, this user should allow ssh based auth without requiring a password
ansible_ssh_user=root
openshift_deployment_type=origin
openshift_release=3.6.0
openshift_disable_check=disk_availability,docker_storage,memory_availability,docker_image_availability,package_availability,package_version
openshift_metrics_install_metrics=true
openshift_master_default_subdomain=apps.oc.tw
openshift_logging_install_logging=true
openshift_master_cluster_public_hostname=master.oc.tw
openshift_hosted_registry_storage_kind=nfs
openshift_hosted_registry_storage_access_modes=['ReadWriteMany']
openshift_hosted_registry_storage_nfs_directory=/exports
openshift_hosted_registry_storage_nfs_options='*(rw,root_squash)'
openshift_hosted_registry_storage_volume_name=registry
openshift_hosted_registry_storage_volume_size=10Gi
openshift_logging_storage_kind=nfs
openshift_logging_storage_access_modes=['ReadWriteOnce']
openshift_logging_storage_nfs_directory=/exports
openshift_logging_storage_nfs_options='*(rw,root_squash)'
openshift_logging_storage_volume_name=logging
openshift_logging_storage_volume_size=10Gi
openshift_metrics_storage_kind=nfs
openshift_metrics_storage_access_modes=['ReadWriteOnce']
openshift_metrics_storage_nfs_directory=/exports
openshift_metrics_storage_nfs_options='*(rw,root_squash)'
openshift_metrics_storage_volume_name=metrics
openshift_metrics_storage_volume_size=10Gi

openshift_master_identity_providers=[{'name':'htpasswd_auth','login':'true','challenge':'true','kind':'HTPasswdPasswordIdentityProvider','filename':'/etc/origin/master/htpasswd'}]

# host group for masters
[masters]
master.oc.tw openshift_public_hostname=master.oc.tw

# host group for nodes, includes region info
[nodes]
master.oc.tw openshift_public_hostname=master.oc.tw
node1.oc.tw openshift_node_labels="{'region':'infra'}" openshift_public_hostname=node1.oc.tw
node2.oc.tw openshift_node_labels="{'region':'primary'}" openshift_public_hostname=node2.oc.tw
node3.oc.tw openshift_node_labels="{'region':'primary'}" openshift_public_hostname=node3.oc.tw

[etcd]
master.oc.tw openshift_public_hostname=master.oc.tw

[nfs]
master.oc.tw
```

- run `ansible-playbook ./hosts ~/openshift-ansible/playbooks/byo/config.yml` to install
- run `ansible-playbook ./hosts ~/openshift-ansible/playbooks/adhoc/uninstall.yml` to uninstall. (ensure configurations are all removed, by checking `/etc/origin` folder on every node)

### NetworkManager must be installed

- install NetworkManager on each node: run `yum install -y NetworkManager && systemctl enable NetworkManager && systemctl start NetworkManager` and `hostname master.oc.tw` to correct the hostname
- pay attention to master/node restarting, since it will reconfigure the hostname, quickly run `hostname master.oc.tw` when you found out that (maybe configure NetworkManager to assign a static hostname will work as well)

### Cannot find file `/etc/origin/node/resolv.conf`

- Create resolv.conf on each node: `cp /etc/resolv.conf /etc/origin/node/resolv.conf`

### Cannot resolve external dns (eg. github.com) from container

- edit file `/etc/dnsmasq.d/origin-dns.conf` to add `server=10.202.129.100`
- restart dnsmasq: `systemctl restart dnsmasq.service`

### sti builder cannot push image to `docker-registry.default.svc:5000` (no route to host)

- edit file /etc/resolve.conf on node with contents:
```
search oc.tw default.svc cluster.local
nameserver 127.0.0.1
nameserver 10.202.129.100
```

### Common ways to debug

- `systemctl status *.service`
- `journalctl -xe`
- `journalctl --unit dnsmasq`


nfs pv configuration
----------------------------------------------------

### Errors from `oc describe pv pv0001`

> Recycle failed: unexpected error creating recycler pod:  pods "recycler-for-pv0001" is forbidden: service account openshift-infra/pv-recycler-controller was not found, retry after the service account is created

- Create sa: `oc create sa pv-recycler-controller -n openshift-infra`
- Config scc:
    `oadm policy add-scc-to-user hostmount-anyuid system:serviceaccount:openshift-infra:pv-recycler-controller`

### Create nfs pv

- Edit `/etc/exports.d/openshift-ansible.exports`, add lines like `/exports/pvs/pv0001 *(rw,root_squash)`
- Run: `mv /exports/pvs/pv0001 && chown nfsnobody:nfsnobody /exports/pv0001 && chmod 777 /exports/pv0001`
- Ensure mount in nodes work well by run `mount -t nfs master.oc.tw:/exports/pvs/pv0001 ./t`
- Create `pv.yml` with below and run `oc create -f pv.yml` to create pv:
```
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv0005
spec:
  capacity:
    storage: 10Gi
  accessModes:
  - ReadWriteOnce
  nfs:
    path: /exports/pvs/pv0005
    server: master.oc.tw
  persistentVolumeReclaimPolicy: Recycle
```
- Debug:
    + `ps aux | grep mount` on node to see if it succeeded
    + `oc describe po/xxx` on master to see the status of the container


Local service dns resolution
---------------------------------------------------

- Find router node: `oc project default && oc get all` to find router pod and `oc describe po/router-1-gsm7v` to find the node and record IP of that node
- Edit local hosts file `/etc/hosts` and add contents
    10.202.128.77 kibana.apps.oc.tw hawkular-metrics.apps.oc.tw nodejs-mongo-persistent-test.apps.oc.tw


Common management tasks
---------------------------------------------------

- Login as admin: ssh into master and you'll be admin directly when you run `oc` commands
- Add user through htpasswd: run `htpasswd -b /etc/origin/master/htpasswd admin admin` on master
- Add project to user: `oadm policy add-role-to-user admin gmliao -n logging`


Node Management
---------------------------------------------------

### Add node

Refer here: https://docs.openshift.org/latest/install_config/adding_hosts_to_existing_cluster.html#adding-nodes-advanced

- Add `new_nodes` or `new_masters` to `[OSEv3:vars]` section
- Add a new section:
```
[new_nodes]
node1.oc.tw openshift_node_labels="{'region':'infra'}" openshift_public_hostname=node1.oc.tw
node2.oc.tw openshift_node_labels="{'region':'primary'}" openshift_public_hostname=node2.oc.tw
```
- Run installation: `ansible-playbook -i hosts ~/openshift-ansible/playbooks/byo/openshift-node/scaleup.yml`
- Move nodes configured in `[new_nodes]` section to `[nodes]` section

### Remove node

Refer here: https://docs.openshift.org/latest/admin_guide/manage_nodes.html#deleting-nodes

Reference
---------------------------------------------------

- https://docs.openshift.org/latest/install_config/install/advanced_install.html
- https://kubernetes.io/docs/setup/




