---
layout: post
title: "spring boot 迁移"
date: 2016-03-30 19:07:57 +0800
comments: true
tags: 
- spring
- spring-boot
- java
---


## web.xml

### error page

```xml
<error-page>
    <error-code>404</error-code>
    <location>/WEB-INF/jsp/errors/error.jsp</location>
</error-page>
```

```java
@Bean
public ServerProperties serverProperties () {
    return new ServerProperties() {
        @Override
        public void customize(ConfigurableEmbeddedServletContainer container) {
            super.customize(container);
            container.addErrorPages(new ErrorPage(HttpStatus.NOT_FOUND, "/error/404"));
        }
    };
}

```

## spring-context.xml

```java
@ImportResource("classpath:spring-context.xml")
@EnableAdminServer
public class WebApplication extends SpringBootServletInitializer {
    ...
}
```

## log back upgrade

```xml
<!-- change include to included for file that will be included, like this: -->
<!-- http://logback.qos.ch/manual/configuration.html -->
<included>
...
</included>
```



