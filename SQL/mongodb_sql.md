# SQL

## What is a Join?

A JOIN clause is used to combine rows from two or more tables, based on a related column between them.

## Different Types of SQL Joins

- **INNER JOIN:** Returns records that have matching values in both tables.
- **LEFT (OUTER) JOIN:** Returns all records from the left table, and the matched records from the right table.
- **RIGHT (OUTER) JOIN:** Returns all records from the right table, and the matched records from the left table.
- **FULL (OUTER) JOIN:** Returns all records when there is a match in either the left or right table.

## Example of FULL OUTER JOIN Syntax

```sql
SELECT column_name(s)
FROM table1
FULL OUTER JOIN table2
ON table1.column_name = table2.column_name
WHERE condition;
```


# MongoDB Atlas

MongoDB Atlas is a fully managed cloud database service provided by MongoDB. It is designed to simplify the deployment, management, and scaling of MongoDB databases in the cloud. With MongoDB Atlas, you can leverage the power of MongoDB without having to worry about the operational complexities of managing database infrastructure.

[Read more about MongoDB Atlas](https://medium.com/@bdhanushka65/what-you-need-to-know-about-mongodb-atlas-b4743727e7f1#:~:text=MongoDB%20Atlas%20is%20a,MongoDB%20databases%20in%20the%20cloud.)

## Key Features and Benefits of MongoDB Atlas

- **Fully Managed:** MongoDB Atlas handles database administration tasks such as hardware provisioning, software patching, setup, configuration, backups, and monitoring. This allows you to focus on developing your applications instead of managing database infrastructure.

- **Scalability:** MongoDB Atlas allows you to scale your databases easily as your application grows. You can add or remove replica set members, shard clusters, or even change the instance size to meet your application’s changing needs.

- **High Availability:** MongoDB Atlas provides built-in high availability and fault tolerance. It replicates your data across multiple nodes within a cluster and automatically handles failover in case of hardware or network failures, ensuring your data is always available.

- **Security:** MongoDB Atlas implements various security measures to protect your data. It offers network isolation with VPC peering, encryption at rest with customer-managed keys, encryption in transit with SSL/TLS, and authentication and access control mechanisms to secure your databases.

- **Backup and Restore:** MongoDB Atlas includes automated backups and point-in-time recovery. It allows you to schedule backups, retain snapshots for a configurable period, and restore data to any specific point in time within that retention window.

- **Monitoring and Alerting:** MongoDB Atlas provides built-in monitoring capabilities to track database performance and usage. It offers detailed metrics, visual dashboards, and configurable alerts to help you identify and troubleshoot issues proactively.

- **Integration with Cloud Services:** MongoDB Atlas integrates with other cloud services like AWS, Azure, and Google Cloud Platform. It allows you to deploy databases in your preferred cloud provider’s infrastructure and leverage additional services for your application stack.

MongoDB Atlas offers a user-friendly web-based interface to manage your databases and clusters. It provides an intuitive dashboard where you can create, configure, monitor, and scale your MongoDB deployments with ease.



# difference between Relational Database & NoSQL Database 

| Feature                  | Relational Database                                              | NoSQL Database                                                   |
|--------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------|
| Data Model               | Tabular (rows and columns)                                       | Various models (document, key-value, wide-column, graph, etc.)    |
| Schema                   | Structured with predefined schema                                | Flexible schema or schema-less                                   |
| Query Language           | SQL (Structured Query Language)                                   | Specific to database type (e.g., MongoDB uses query languages for JSON-like documents) |
| Scalability              | Primarily vertical (scaling up by adding resources to a single server) | Designed for horizontal scalability (scaling out by adding more nodes) |
| ACID Properties          | Strong ACID guarantees (Atomicity, Consistency, Isolation, Durability) | May relax some ACID properties in favor of performance and scalability |
| Use Cases                | Well-suited for structured data and complex relationships         | Preferred for rapidly changing or unstructured data, real-time analytics, IoT, social networks, etc. |
