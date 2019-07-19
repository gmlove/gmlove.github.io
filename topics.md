## 工程师的统计学

概率的理解：
- 所有可能情况：面积为1的正方形
- 随机变量：定义在正方形内部某一个或多个区域上的函数
- 多个随机变量：区域内的区域
- 概率：随机变量取特定值时，定义域的面积
- 条件概率：某些随机变量确定时，另一些随机变量的概率占确定区域的比值，子区域与区域的比值
- 研究的问题：已知某些概率，求另一些概率

蒙提霍尔问题的作图解释
- 问题：主持人 三门一正 选手选一门 主持人开一负门 选手是否要选另一门？
- 目标：两种情况下，哪种情况选手选正门的概率更大
- 分析：不选 - 1/3，选 - 2/3
独立同分布


## 测试tf模型的正确姿势

## tf API最新进展

## 权威的查找论文的地方

https://www.aminer.cn/ranks/conf

知名会议：
- ICML: International Conference on Machine Learning supported by International Machine Learning Society (IMLS). (6th August)https://2017.icml.cc/Conferences/2017/Schedule?type=Poster
- CVPR: Computer Vision and Pattern Recognition, the IEEE annual conference on computer vision and pattern recognition
http://openaccess.thecvf.com/CVPR2017.py
- ICCV: International Conference Or Computer Vision, a research conference sponsored by the Institute of Electrical and Electronics Engineers (IEEE) held every other year (acceptance rate: 25%-30%, oral presentations: 4%) (October 26)
http://iccv2017.thecvf.com/submission/timeline
- ACL: Annual Meeting of the Association for Computational Linguistics (July 30-August 4, 2017)
https://chairs-blog.acl2017.org/2017/04/05/accepted-papers-and-demonstrations/
- NIPS: Annual Conference on Neural Information Processing Systems is a multi-track machine learning and computational neuroscience conference (Dec 4 - Dec 9, 2017)
https://nips.cc/
- KDD: Data mining (the analysis step of the "Knowledge Discovery in Databases" process, or KDD), an interdisciplinary subfield of computer science, is the computational process of discovering patterns in large data sets involving methods at the intersection of artificial intelligence, machine learning, statistics, and database systems. Extract information into an understandable structure. Involves database and data management aspects, data pre-processing, model and inference considerations, interestingness metrics, complexity considerations, post-processing of discovered structures, visualization, and online updating.
http://www.kdd.org/kdd2017/accepted-papers
- IJCAI: The International Joint Conference on Artificial Intelligence (or IJCAI) is a gathering of Artificial Intelligence (AI) researchers and practitioners. It is organized by the IJCAI, Inc., and has been held biennially in odd-numbered years since 1969

conversational UI

## 网络结构设计

resnet resnext dpn densenet senet

## 损失函数设计

## 优化器选择

## 论文统计

RL + semi + GAN / total{}
ICML 2017: 21 + 4 + 2 + 1/433 ~ 10%
ICML 2016: 8 + 2 + 1 + 1/323 ~ 5%
ICML 2015: 3 / 270 ~ 1%

## TF的API变化趋势

## Alpha Zero的实现

## 架构

架构的范围：
架构的作用：
架构的实践：无架构，专注架构，提升架构
架构师：企业架构师 应用架构师

## 从0开始构建现代化的爬虫

为什么不用现存框架：限定了技术范围，调试困难，性能难以满足需求，学习成本。有了现在的技术，很容易实现这一框架。下载是长任务易失败，适于基于事件的架构。
爬虫的核心问题：目录 -> 内容，（有条件）下载，更新策略，
爬虫技术选型：队列，redis，elasticsearch，k8s，nodejs
代理池：直接爬取API来构建，代理选择优先级，代理验证，质量反馈验证
爬虫：使用 chrome headless 构建（假设技术升级相对比较快，关注简单的用户交互，而非细节的升级和优化）；直接存储爬取到的原始数据，将爬取过程和数据解析过程分离；应用代理；对大的资源建立缓存，不下载多媒体资源；爬虫状态监控与自恢复；去重与更新策略
数据解析：解析html，提取数据，清洗/转换数据，验证数据，输入入库
数据分析：基于elasticsearch的数据报表
部署：所有任务都部署到k8s；一个任务一个namespace可以每种爬虫进行独立更新；定时任务定期添加目录/重建缓存


