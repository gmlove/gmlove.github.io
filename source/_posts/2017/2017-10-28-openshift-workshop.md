---
title: OpenShift工作坊
categories:
- DevOps
tags:
- kubernetes
- DevOps
- OpenShift
- PAAS
- Cluster
date: 2017-10-28 23:12:21
---

Target
---------------------------------

- A simple nodejs application
- Add mongodb to the application
- CI/CD for the application
- Logging, Monitoring, Debugging


A simple nodejs application
---------------------------------

### Introduction

In this section, We are going to create a nodejs project with mongodb in OpenShift. We assume that you have done all the preparation work listed in the invitation email of this workshop. And there're some additional steps to  get yourself ready.

### Preparation

- Install `oc` in your mac:

  ```
  brew update 
  brew install openshift-cli
  oc version                     # the version should be oc v1.5.1+7b451fc
  ```

  You can also download it's here: <https://github.com/openshift/origin/releases/download/v1.5.1/openshift-origin-client-tools-v1.5.1-7b451fc-mac.zip>


- fork [the demo project](https://github.com/openshift/nodejs-ex) to your own github account and get the repository url:

  `https://github.com/<your-github-name>/nodejs-ex.git`

- clone code  

  `git clone https://github.com/<your-github-name>/nodejs-ex.git`

- clone configuration files

  `git clone https://github.com/gmlove/openshift-workshop.git`

  Files in this repo will be used later.

- log on to OpenShift web console

  In browser, navigate to `https://13.228.41.255:8443` and login with any username and password you'd like to use. OpenShift will create that account for you.

### Login

- `oc login https://13.228.41.255:8443`  and input your username and password:

  > Authentication required for https://13.228.41.255:8443 (openshift) 
  >
  > Username: <your-username> 
  >
  > Password: <your-password>

- check status

  `oc status`
  > You don't have any projects. You can try to create a new project, by running
  >
  > oc new-project <projectname>

- check project

  `oc project`
  > No project has been set. Pass a project name to make that the default.

### Create project

  `oc new-project <your-project-name>`

### Create applicatioin

- create a new application with your own git repository url

  `oc new-app https://github.com/<your-github-name>/nodejs-ex.git`

  > you can specify the application name with a parameter `--name <application-name>`

### Create route

- expose service

  `oc expose service/nodejs-ex`

- get route url

  `oc get route` 

  > your route url should look like this: 
  >
  > nodejs-ex-\<your-project-name\>.13.228.41.255.xip.io

- check out the url in web browser you'll see the welcome page

### Build & deploy

- build

  `oc start-build nodejs-ex`

    > A build will be triggered automatically once the application is created successfully.

- deploy  

  `oc deploy --latest dc/nodejs-ex`

  > A deployment will be triggered automatically when a build is finished successfully.

### Add Webhook

- Copy webhook url

  In OpenShift web console, go to `Builds` -> click `<your-build-config-name>` -> switch to `Configuraton` tab ->
  copy the url on the right of `GitHub Webhook URL:`

- Config Webhook in github

  - Go to web browser with url: `https://github.com/<your-github-name>/nodejs-ex.git`

  - Switch to `Settings` tab below the project name

  - Click `Webhooks` item in left menu panel and then Click `Add Webhook` button

  - Paste the webhook URL in field `Payload URL`, select `application/json` in `Content type` field, click `disable ssl verification`

  - Click `Add webhook` at the bottom and you'll return to previous page

  - Github will test the webhook automatically, if succeed, it will place a green mark before the url.

  - done

- Vefiry Webhook

  - modify code and commit your changes

    modify \<code-dir\>/nodejs-ex/views/index.html line 219

    `Welcome to your Node.js` to `This is my awesome Node.js`

    push your changes to github

  - check your changes in web browser

    in OpenShift web console you'll see a build & deployment is triggered automatically

    you'll see the tilte of the page changed from `Welcome to your Node.js` to `This is my awesome Node.js` afterh the app is re-deployed.

### Rollback

- rollback to the last successful deployment

  `oc rollback dc/nodejs-ex` or `oc rollout undo dc/nodejs-ex`

  you'll get the following messages:

  >  Warning: the following images triggers were disabled: nodejs-ex:latest
  >
  >  You can re-enable them with: oc set triggers dc/nodejs-ex --auto

  The message means after a build finished, OpenShift will no longer trigger an automatic deployment. You can re-enable it use the command given in the message if you would like to.

  > You can also rollback to a specified deployment with the following command:
  >
  > `oc rollout undo dc/nodejs-ex --to-revision=1` or `oc rollback nodejs-ex --to-version=1`

### Scale up/down
- scale manually

  scale up to 2 pods

  `oc scale dc/nodejs-ex --replicas=2`

  now you can scale down to 1 pod

  `oc scale dc/nodejs-ex --replicas=1`

- auto scaling

  - edit DeployConfig in web console

    `Applications` -> `Deployment` -> `<your-deploy-config-name>` -> `Action` menu -> `Edit YAML`

    replace `resources:{}` with the following content:

    ```yaml
    resources:
      limits:
        cpu: 100m
      requests:
        cpu: 100m
    ```

    and the yaml script will looks like this:

   ```yaml
      spec:
        containers:
          - name: nodejs-ex
            image: >-
              172.30.1.1:5000/leojiangproject/nodejs-ex@sha256:eac764a3e2366e929dd00e8558d9433d3f81bf590e09340527a8e839cf0fe551
            ports:
              - containerPort: 8080
                protocol: TCP
            resources:
              limits:
                cpu: 100m
              requests:
                cpu: 100m
            terminationMessagePath: /dev/termination-log
            imagePullPolicy: Always
        restartPolicy: Always
        terminationGracePeriodSeconds: 30
        dnsPolicy: ClusterFirst
        securityContext: {}
    ```

    Back to `overview` in web console, you'll see an error at the top, click on the url and trust the site.
    
    Now you can see metrics in `overview` panel.

  - config autoscaler with command
  
    `oc autoscale dc/nodejs-ex --min 1 --max 10 --cpu-percent=10`

  - verify

    send batch request with command: `ab -n 1000 -c 100 http://<route-of-your-application>`

    Wait and you'll see the pod number is scaled to 2

Add mongodb to the application
---------------------------------

### Deploy a database with persistent volume

- `cd` to the directory where you downloaded the configuration files earlier.

- crete secret

  `oc create -f mongodb-secrets.yaml`

- check secret

  `oc describe secret nodejs-ex`

- create persistent volume claim

  `oc create -f mongodb-pvc.yaml`

- check persistent volume claim

  `oc get pvc`

- create mongodb deployment config

  `oc create -f mongodb-deploy-config.yaml`

- create a mongodb service

  `oc create -f mongodb-service.yaml`

- trigger a deployment for mongodb (if not triggered automatically)

  `oc rollout latest dc/mongodb`

### Connect nodejs application to database

- change deployment config and add environment parameters

  In web console, go to `Applications` -> `Deployment` -> `<your-deploy-config-name>` -> `Action` menu -> `Edit YAML`

  add the following content under `caontaners` tag:

  ```yaml
          env:
            - name: DATABASE_SERVICE_NAME
              value: mongodb
            - name: MONGODB_USER
              valueFrom:
                secretKeyRef:
                  key: database-user
                  name: nodejs-ex
            - name: MONGODB_PASSWORD
              valueFrom:
                secretKeyRef:
                  key: database-password
                  name: nodejs-ex
            - name: MONGODB_DATABASE
              value: sampledb
            - name: MONGODB_ADMIN_PASSWORD
              valueFrom:
                secretKeyRef:
                  key: database-admin-password
                  name: nodejs-ex
  ```

  the resulting yaml script will looks like this:

  ```yaml
    spec:
      containers:
        - name: nodejs-ex
          env:
            - name: DATABASE_SERVICE_NAME
              value: mongodb
            - name: MONGODB_USER
              valueFrom:
                secretKeyRef:
                  key: database-user
                  name: nodejs-ex
            - name: MONGODB_PASSWORD
              valueFrom:
                secretKeyRef:
                  key: database-password
                  name: nodejs-ex
            - name: MONGODB_DATABASE
              value: sampledb
            - name: MONGODB_ADMIN_PASSWORD
              valueFrom:
                secretKeyRef:
                  key: database-admin-password
                  name: nodejs-ex
  ```

- trigger a deployment for nodejs-ex

  `oc rollout latest dc/nodejs-ex`

- check

  In you browser, open the application with url `nodejs-ex-testproject.13.228.41.255.xip.io`, and check the `Page view count` in the web page, it should be counting.



CI/CD for the application
----------------------------------

**Several ways to integrate pipeline:**

- Keep using the original pipeline and artifact repository. Manage your pipeline outside of openshift world. When you need to deploy to an environment, you can trigger a build in openshift to grab your artifact and build an image and deploy it to openshift.
- Use openshift integrated Jenkins as your pipeline. (heavily customized, with openshift plugin and k8s plugin installed.)

![Openshift Pipeline Structure](/attaches/2017/2017-10-28-openshift-workshop/pipeline-structure.png)

### Integrate Jenkins with the application

- Create a pipeline build config

  Create a file named `nodejs-ex-pipeline.yaml` with content (already created for you):

  ```yaml
  kind: "BuildConfig"
  apiVersion: "v1"
  metadata:
    name: "nodejs-ex-pipeline"
  spec:
    strategy:
      jenkinsPipelineStrategy:
        jenkinsfile: "node('nodejs') {\n  stage 'build'\n  openshiftBuild(buildConfig: 'nodejs-ex', showBuildLogs: 'true')\n  stage 'deploy'\n  openshiftDeploy(deploymentConfig: 'nodejs-ex')\n}\n"
  ```

  To create a pipeline, run:
    `oc create -f nodejs-ex-pipeline.yaml`

  Openshift will add the resources below to the project:

  + route `routes/jenkins`
  + services `svc/jenkins` `svc/jenkins`

- Run `oc status` to get your dns of the jenkins application. Then you can open it in your browser and perform an oauth login with openshift credentials.

- Trigger a build manually and you can check logs from jenkins. You can also find what is openshift doing by `watch oc get all`

  + Openshift creates a slave based on the node being used, here it's `nodejs`

  + Run the build in that slave

  + For our simple example, the pipeline has two steps: trigger a build(works just like `oc start-build bc/nodejs-ex`), and then trigger a deployment


### Multiple environment management

In openshift you define several resources (API Objects) and you can start an environment. So multi environment means multiple similar resource sets. To manage multiple environment, you have many options, such as:

- Via labels and unique naming within a single project
- Via distinct projects within a cluster
- Via distinct clusters

Consider your situation in your organization and choose one properly. We'll try the second one as an example and create a `sys` region for the application. We'll develop a way to share the image created in the project (can be called as dev project).

- Spend some time thinking about which resources you will need for another environment (ImageStream DeploymentConfig Service Route PersistentVolumeClaim)

- Tag all the resources to export

  ```bash
  oc label dc/mongodb promotion=nodejs-ex
  oc label dc/nodejs-ex promotion=nodejs-ex
  oc label svc/nodejs-ex promotion=nodejs-ex
  oc label svc/mongodb promotion=nodejs-ex
  oc label routes/nodejs-ex promotion=nodejs-ex
  oc label pvc/mongodb promotion=nodejs-ex
  oc label is/nodejs-ex promotion=nodejs-ex
  oc label secret/nodejs-ex promotion=nodejs-ex
  ```

- Export those resources

  `oc export dc,svc,routes,pvc,is,secret -l promotion=nodejs-ex -o yaml > exported-for-promotion.yaml`

- Open the exported file and do the below to make it portable to `sys` environment:

  + Remove image hash tag
  + Replace all string `test1` to `test1-sys` and we're creating a project named `test1-sys`
  + Remove `annotations` `volumeName` `status` `creationTimestamp` from `PersistentVolumeClaim`

- Create a new project by `oc new-project test1-sys`

- Create resources by `oc create -f exported-for-promotion.yaml`

- Tag an image from test1

  `oc tag test1/nodejs-ex:latest nodejs-ex:latest`

- After that you can trigger a deployment, and after it succeeded, you will be able to open the application in `sys` environment

- Try create a template for these resources and then you can create any new environment with one command


Logging, Monitoring, Debugging
----------------------------------

### Logging aggregation

- EFK solution, check a video [here](https://www.youtube.com/watch?v=RMDX3YC0CSQ)
- Start the cluster with logging and metrics installed: 

  `oc cluster up --logging --metrics`

  There is a bug for logging currently: https://bugzilla.redhat.com/show_bug.cgi?id=1410694


### Monitoring solutions

- Prometheus/zabbix/hawk: https://github.com/openshift/openshift-tools/tree/stg/openshift_tools/monitoring
  
  An example to setup prometheus and grafana: https://github.com/debianmaster/openshift-examples/tree/master/promethus
  We created a monitor service already: http://grafana-openshift-infra.13.228.41.255.xip.io (admin:admin)

- dynatrace/coscale/sysdig/appdynamics


### Debugging

- To monitor the change of all resources: `watch oc get all`
- Check logs of pods: `oc logs po/nodejs-ex-1-0s5sl`
- Check events: `oc get event`
- Login to container: `oc rsh po/nodejs-ex-1-0s5sl`



