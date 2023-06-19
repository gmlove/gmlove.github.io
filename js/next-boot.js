/* global NexT, CONFIG, Velocity */

NexT.boot = {};

NexT.boot.registerEvents = function() {

  NexT.utils.registerScrollPercent();
  NexT.utils.registerCanIUseTag();

  // Mobile top menu bar.
  document.querySelector('.site-nav-toggle .toggle').addEventListener('click', () => {
    event.currentTarget.classList.toggle('toggle-close');
    var siteNav = document.querySelector('.site-nav');
    var animateAction = siteNav.classList.contains('site-nav-on') ? 'slideUp' : 'slideDown';

    if (typeof Velocity === 'function') {
      Velocity(siteNav, animateAction, {
        duration: 200,
        complete: function() {
          siteNav.classList.toggle('site-nav-on');
        }
      });
    } else {
      siteNav.classList.toggle('site-nav-on');
    }
  });

  var TAB_ANIMATE_DURATION = 200;
  document.querySelectorAll('.sidebar-nav li').forEach((element, index) => {
    element.addEventListener('click', event => {
      var item = event.currentTarget;
      var activeTabClassName = 'sidebar-nav-active';
      var activePanelClassName = 'sidebar-panel-active';
      if (item.classList.contains(activeTabClassName)) return;

      var targets = document.querySelectorAll('.sidebar-panel');
      var target = targets[index];
      var currentTarget = targets[1 - index];
      window.anime({
        targets : currentTarget,
        duration: TAB_ANIMATE_DURATION,
        easing  : 'linear',
        opacity : 0,
        complete: () => {
          // Prevent adding TOC to Overview if Overview was selected when close & open sidebar.
          currentTarget.classList.remove(activePanelClassName);
          target.style.opacity = 0;
          target.classList.add(activePanelClassName);
          window.anime({
            targets : target,
            duration: TAB_ANIMATE_DURATION,
            easing  : 'linear',
            opacity : 1
          });
        }
      });

      [...item.parentNode.children].forEach(element => {
        element.classList.remove(activeTabClassName);
      });
      item.classList.add(activeTabClassName);
    });
  });

  window.addEventListener('resize', NexT.utils.initSidebarDimension);

  window.addEventListener('hashchange', () => {
    var tHash = location.hash;
    if (tHash !== '' && !tHash.match(/%\S{2}/)) {
      var target = document.querySelector(`.tabs ul.nav-tabs li a[href="${tHash}"]`);
      target && target.click();
    }
  });
};

NexT.boot.refresh = function() {

  /**
   * Register JS handlers by condition option.
   * Need to add config option in Front-End at 'layout/_partials/head.swig' file.
   */
  CONFIG.fancybox && NexT.utils.wrapImageWithFancyBox();
  CONFIG.mediumzoom && window.mediumZoom('.post-body :not(a) > img, .post-body > img');
  CONFIG.lazyload && window.lozad('.post-body img').observe();
  CONFIG.pangu && window.pangu.spacingPage();

  CONFIG.exturl && NexT.utils.registerExtURL();
  CONFIG.copycode.enable && NexT.utils.registerCopyCode();
  NexT.utils.registerTabsTag();
  NexT.utils.registerActiveMenuItem();
  NexT.utils.registerLangSelect();
  NexT.utils.registerSidebarTOC();
  NexT.utils.wrapTableWithBox();
  NexT.utils.registerVideoIframe();
};

NexT.boot.motion = function() {
  // Define Motion Sequence & Bootstrap Motion.
  if (CONFIG.motion.enable) {
    NexT.motion.integrator
      .add(NexT.motion.middleWares.logo)
      .add(NexT.motion.middleWares.menu)
      .add(NexT.motion.middleWares.postList)
      .add(NexT.motion.middleWares.sidebar)
      .bootstrap();
  }
  NexT.utils.updateSidebarPosition();
};

document.addEventListener('DOMContentLoaded', () => {
  NexT.boot.registerEvents();
  NexT.boot.refresh();
  NexT.boot.motion();
});


var links = {"NDUzMjhh": "/2022/05/30/neat-syntax-design-of-an-etl-language-part-2/", "NDU0MWQ2": "/2022/06/08/efficient-etl-testing/", "NDUwNjQx": "/2022/05/04/a-new-etl-language-easy-sql/", "NDUyN2Vl": "/2022/05/25/neat-syntax-design-of-an-etl-language/", "NDU5MTYx": "/2022/07/28/modelling-examples/", "NDUyNjc2": "/2022/05/24/5-properties-of-good-code-cupid/", "NDU2ODE3": "/2022/07/05/tdd-to-develop-a-long-running-task-system/", "NDUxOGZk": "/2022/05/16/a-guide-to-write-elegant-etl/", "NDU5MDY2": "/2022/07/27/smart-domain-and-ddd/", "NDg4NTA5": "/2023/05/18/chatgpt-transformer/", "NDg4N2M3": "/2023/05/20/chatgpt-training/", "NDgzMjlh": "/2023/03/26/wechatgpt/", "NDc1N2Fj": "/2023/01/10/ade-ci-cd-per-etl/", "NDg5MjNi": "/2023/05/25/chatgpt-rlhf/", "NDg0OTc2": "/2023/04/12/chatgpt-from-programmer-point-of-view/", "NDg2MjMx": "/2023/04/25/chatgpt-a-technical-summary/", "MjE2NzU0": "/2015/12/08/first-blog/", "MjE2N2Jh": "/2015/12/08/pitfall-jasmin-any/", "MjY2MTU1": "/2017/04/15/understanding-gradients-of-conv2d-in-experiments/", "MjU3MmQz": "/2017/01/16/dl-workshop-massive-network-tips/", "Mjg1NGI3": "/2017/10/25/aws-openshift-cluster-installation-guide/", "Mjg5Nzdl": "/2017/12/07/automated-comment-on-xiaomi-s-live-stream/", "Mjg1N2Qz": "/2017/10/28/openshift-workshop/", "Mjg1M2Ew": "/2017/10/24/local-openshift-cluster-installation-guide/", "MjczMzU0": "/2017/06/26/dive-into-gan-continued/", "MjYxNmVh": "/2017/03/01/recognize-house-number/", "MjcyOGFm": "/2017/06/21/dive-into-gan/", "MzQ4NjRj": "/2019/07/19/python-json-serializable-lib/", "MzQ5MTAw": "/2019/07/24/what-programmer-should-know-about-compiler/", "MzUyNDRk": "/2019/08/26/the-functional-programming-you-understand-may-not-be-what-we-recommended/", "MzYyMjFl": "/2019/12/02/sense-of-ceremony-and-professional-service/", "MzU4NjY5": "/2019/10/27/hadoop-auth/", "MzUzNDFl": "/2019/09/05/programmers-tolerance/", "MzU2NmMw": "/2019/10/07/after-reading-of-refactoring-v2/", "MzQ3NTFh": "/2019/07/08/reproduce-ml-models/", "MzQ4N2Ez": "/2019/07/20/tdd-for-improving-design/", "MzQ5MmEy": "/2019/07/25/ways-to-improve-python-perf/", "MzYyMmYz": "/2019/12/02/hadoop-auth-4/", "MzY0MjAz": "/2019/12/22/its-harder-to-go-downhill/", "MzYwMWIw": "/2019/11/11/hadoop-auth-3/", "MzQ3NTI5": "/2019/07/08/reproduce-ml-models-dorn/", "MzU4OWY3": "/2019/10/30/hadoop-auth-2/", "MzUwNmYz": "/2019/08/08/domain-concept-in-your-code/", "MzUxNjU2": "/2019/08/18/tdd-for-improving-design-2/", "MzQ3OWJh": "/2019/07/12/reproduce-ml-models-efficientnet/", "MzUwNDI2": "/2019/08/06/you-may-need-a-lightweight-zhongtai/", "NDA3MGJi": "/2021/02/22/data-ingestion-from-mongo/", "NDA2OTcy": "/2021/02/21/data-governance-based-on-atlas-ranger/", "NDExMjhj": "/2021/04/05/dwd-modeling-automation/", "NDA3NzMy": "/2021/03/01/data-ingestion-practice/", "NDAzOWUz": "/2021/01/22/bigdata-platform-based-on-hdp/", "NDEzMjU5": "/2021/04/25/data-testing-tool/", "NDEwMzU1": "/2021/03/27/programmer-efficiency/", "NDEyNzEw": "/2021/04/20/data-testing/", "NDExNzQ2": "/2021/04/10/data-development-tools/", "NDE3MDkx": "/2021/06/02/ml-on-data-platform/", "NDE2MWE1": "/2021/05/24/data-pipeline-for-data-project/", "NDA5MWQw": "/2021/03/15/data-management-practice/", "NDE0Nzlk": "/2021/05/10/data-browser-for-point-analysis/", "NDEwODMw": "/2021/04/01/data-development-language-and-environment/", "NDAzOGU1": "/2021/01/21/some-thoughts-about-data-platform/", "NDE2MzJj": "/2021/05/26/data-indicator-calculation-practice/", "NDA5MjFi": "/2021/03/16/data-modeling-practice/", "NDE3ODBi": "/2021/06/10/oneid-practice/", "NDE2NDdh": "/2021/05/27/indicator-management-system/", "MzY3Mjg2": "/2020/01/21/spark-performance-tuning-on-billions-of-data/", "Mzk4MmE2": "/2020/11/26/data-work-roles/", "NDAxMzhh": "/2020/12/27/oracle-data-migration/", "MzcxOWFh": "/2020/03/08/new-ways-to-manage-memory/", "Mzc1NDlj": "/2020/04/12/an-apprehensible-way-to-describe-ctr--introduction/", "Mzc5NGZh": "/2020/05/22/architecture-designing-practise-for-ml-platform/", "MzY1NjI4": "/2020/01/05/DDD-in-pipeline-code/", "Mzc5NjM3": "/2020/05/24/architecture-designing-practise-for-ml-platform-oop/", "MzY4OGUy": "/2020/02/06/DRL-the-problem/", "MzgwNDk3": "/2020/06/01/node-bpm/", "Mzc5NWE3": "/2020/05/23/architecture-designing-practise-for-ml-platform-configuration/", "MzczNGFh": "/2020/03/23/rust-the-good-part/", "Mzg0MGYw": "/2020/07/07/basic-loan-business/", "Mzk4OTcy": "/2020/12/03/data-capability-building-suggestions/", "MzkwNTFi": "/2020/09/10/easy-statistical-test/", "MzkwOGFi": "/2020/09/13/correlation-analysis/", "Mzc0MDA5": "/2020/03/29/native-code-compilation-process-programmers-should-know/", "Mzg4NjNl": "/2020/08/22/data-analyst-mindset/", "MzA0Nzhi": "/2018/05/06/reinforcement-learning-mdp/", "MjI3NjE2": "/2016/03/26/transparent-video/", "MjI2NDVm": "/2016/03/14/bluetooth-summary/", "MjMwNmQ3": "/2016/04/25/must-have-admin-site-for-spring-projects/", "MjI1OWU2": "/2016/03/09/self-cultivation-of-a-programmer/", "MjUyNzMz": "/2016/12/02/dl-workshop-rnn-and-lstm/", "MjM4MWFh": "/2016/07/09/backtracking-search-for-a-game/", "MjU0MWM4": "/2016/12/16/dl-workshop-summary/", "MjUyMmJi": "/2016/11/27/a-pick-into-tensorflow/", "MjM4OTMy": "/2016/07/17/session-storage/", "MjI4MGFl": "/2016/03/30/migrate-to-spring-boot/", "MjMwNjNi": "/2016/04/25/docker-and-micro-service/", "MjMyNzc5": "/2016/05/16/business-trip-life-in-sydney/", "MjMwNzAz": "/2016/04/26/about-micro-service-architecture/", "MjI1ODI0": "/2016/03/08/ghost-space-in-less/", "MjI2NzRi": "/2016/03/17/opengl-summary/", "MjQzNGYx": "/2016/08/31/agile-user-story/", "MjQyODQ1": "/2016/08/25/tdd-practise/", "MjIwNThm": "/2016/01/15/css-best-practise/", "MjUzMGUw": "/2016/12/05/let-machine-play-games/", "MjI1NDY5": "/2016/03/04/maven-tips/", "MjUzNmEz": "/2016/12/11/dl-workshop-rnn-and-lstm-1/", "MjI5NjI4": "/2016/04/15/pitfall-collection/", "MjI1OGQx": "/2016/03/08/pitfall-clearfix-with-table-display/"}
try {
  (function(){
    if (window.location.hash) {
      var link = window.location.hash.substring(2);
      if (links[link]) {
        window.location.href=links[link];
      }
    }
  })()
} catch (e) {}


