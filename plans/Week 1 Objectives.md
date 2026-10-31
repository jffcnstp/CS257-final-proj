Week 1 is about deciding exactly what you’ll test before you start building the advisor. You want a small, repeatable experiment where Laya and the baseline models face the same database situations. After we figure each part, update the doc answering the task/problem.

### **1\. Define the research question**

Write down the main question and what would count as a good result. For example:

> Can a fine-tuned Laya model choose useful index actions as well as Random Forest and boosted-tree models, while keeping query performance and index costs competitive?

Decide whether you’re mainly comparing **action accuracy**—did the model pick the same action as your label—or **real database outcomes**, such as total query time and write overhead. Your proposal includes both, so plan to measure both.

### **2\. Set up version control environment \- CI/CD**

Hugging face for model collaboration, git hub for anything else?

### **3\. Choose a PostgreSQL schema**

Pick a manageable database structure to run experiments on. For example, use an order-processing schema with `customers`, `orders`, and `order_items` tables. Define the columns, approximate table sizes, and relationships. The schema should support queries that filter, join, group, and sort, because those are common reasons to create indexes.

You could also use a standard benchmark or dataset if it fits your project and is practical to run locally. The key is that every model should be tested on the same data and schema.

### **4\. Specify the workload mix**

Create a set of queries that represents how the database is used. Include read queries that filter or join tables, plus writes such as inserts or updates, since indexes can speed reads but slow writes.

Record how often each query runs. For example, a workload might be 80% reads and 20% writes, with some queries frequent and others rare. You can later vary that mix to test whether models adapt when usage changes.

### **5\. Set rules for candidate indexes**

Decide which possible indexes the system is allowed to consider. To keep the project manageable, candidates could be single-column indexes on columns used in `WHERE`, `JOIN`, `GROUP BY`, or `ORDER BY` clauses.

Also define the index state and allowed action:

* If the index is absent: choose `CREATE` or `NO-OP`.  
* If it exists: choose `KEEP` or `DROP`.

This defines the choices Laya and the comparison models will receive.

### **6\. Define the evaluation cost score**

Decide how you’ll compare the benefit of an index with its costs. A simple score could combine:

* read-query runtime,  
* write-query overhead,  
* index storage, and  
* index build time.

Specify how you’ll measure and normalize each component, and how much weight each gets in the combined score. Also report each component separately, so the overall score doesn’t hide a trade-off—for example, faster reads that come with much slower writes.

By the end of Week 1, aim to have a short experiment specification: the research question, schema, workload queries and frequencies, candidate-index rules, and a written formula for the score.

