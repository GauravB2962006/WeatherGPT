import { MongoClient } from "mongodb";

const MONGODB_URI = process.env.MONGODB_URI || "mongodb://127.0.0.1:27017";

const DB_NAME = process.env.MONGODB_DB || "weathergpt";

let client;
let database;

export async function connectMongoDB() {
  if (database) {
    return database;
  }

  client = new MongoClient(MONGODB_URI);

  await client.connect();

  database = client.db(DB_NAME);

  console.log(`MongoDB connected: ${DB_NAME}`);

  return database;
}

export async function getDatabase() {
  if (!database) {
    return connectMongoDB();
  }

  return database;
}
