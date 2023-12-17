## dbt:

Data transformation is an essential part of data-processing workflows: filtering, grouping or joining multiple sources into a format that is suitable for analyzing data or feeding machine-learning models. dbt is an open-source tool and a commercial SaaS product that provides simple and effective transformation capabilities for data analysts. The current frameworks and tooling for data transformation fall either into the group of powerful and flexible — requiring intimate understanding of the programming model and languages of the framework such as Apache Spark — or in the group of dumb drag-and-drop UI tools that don't lend themselves to reliable engineering practices such as automated testing and deployment. dbt fills a niche: it uses SQL — an interface widely understood — to model simple batch transformations, while it provides command-line tooling that encourages good engineering practices such as versioning, automated testing and deployment; essentially it implements SQL-based transformation modeling as code. dbt currently supports multiple data sources, including Snowflake and Postgres, and provides various execution options, such as Airflow and Apache's own cloud offering. Its transformation capability is limited to what SQL offers, and it doesn't support real-time streaming transformations at the time of writing.

## V1:

SQL is designed to be used in a declarative way and it causes a few troubles when we use SQL to develop complicated ETL. It would be difficult to handle the following cases:
- Use large computing resources when handling data in the full-data partition since the amount of data there is far larger than that in the other partitions.
- Send out a http request to report status when some step of the ETL fails for some reasons(E.g. some data does not conform to the previous assumptions).
- Reuse some code to check if some order is a valid order (think about e-commerce business).
- Stop at some step of the ETL and check if the data is what we expected.
The above cases could be easily handled if we have an imperative-way to write our code. This might be the reason why a lot of developers like to write ETLs in a general programming language like Python or Scala.
But for data ETL development case, to use SQL or SQL-like language is a better choice. The main reasons are:
- Consistent code style across all ETLs.
- All roles in the team can easily understand the logic in ETL.
- All code about one ETL mainly stays in one file and it makes things simpler when we try to read and understand what it does in the ETL.
Easy SQL fills the gap. Unlike dbt, it's not only a template engine, but also provides a language execution environment.

## V2:

Easy SQL is another tool for data transformation. It comes into our sights because it's similar to the once-recommended tool dbt, while provides more capabilities we used to see in general programming languages. Easy SQL enables us to write data transformation ETL in SQL, which removes the requirement of intimate understanding of the programming model and languages of the calculation frameworks such as Apache Spark, while the good engineering practices such as versioning, automated testing and deployment applies as well.
Comparing to dbt, Easy SQL is not just another template engine but also could be viewed as a totally new small ETL language based on SQL. Easy SQL provides a way to write ETL in an imperative way, just like what we usually do in general programming languages such as Python or Java. It also provides other useful features to help write complicated ETL. To list a few:
- Support variables which could be defined and modified any time.
- A way to call external functions which expands the capability of ETL to a great extent.
- A way to control whether a step should be executed.
- Templates that could be reused in the same ETL file.
- Include command that could be used to reuse code at file level.
- A linting tool to lint ETL code.
- Debugging support: logging and assertion and a debugger interface that could be used for debugging.
- A testing interface to help write and maintain test cases.
- Support batching and streaming calculation.
- An engine to execute the code written in Easy SQL.
The doc of Easy SQL is well maintained and code quality looks good. But the community still needs to be developed.



















