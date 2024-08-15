import motor.motor_asyncio
from pymongo.errors import (
    ConnectionFailure,
    ServerSelectionTimeoutError,
    BulkWriteError,
)
from typing import List, Dict, Set
import asyncio
import logging
import os
import aiohttp
from pymongo import DeleteMany
from bson import ObjectId


class DatabaseHandler:
    def __init__(
        self,
        connection_string: str,
        db_name: str,
        max_retries: int = 3,
        retry_delay: int = 5,
    ):
        self.connection_string = connection_string
        self.db_name = db_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = None
        self.db = None
        self.logger = logging.getLogger(__name__)
        self.is_connected = False

    async def connect(self):
        for attempt in range(self.max_retries):
            try:
                self.client = motor.motor_asyncio.AsyncIOMotorClient(
                    self.connection_string, serverSelectionTimeoutMS=5000
                )
                await self.client.server_info()  # This will raise an exception if it can't connect
                self.db = self.client[self.db_name]
                self.is_connected = True
                self.logger.info("Successfully connected to MongoDB")
                return
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    self.logger.error(
                        "Failed to connect to MongoDB after maximum retries"
                    )
                    raise

    async def close(self):
        if self.client:
            self.client.close()
            self.is_connected = False
            self.logger.info("Closed MongoDB connection")

    async def ensure_connection(self):
        if not self.is_connected:
            await self.connect()

    async def store_items(self, items: List[Dict]):
        await self.ensure_connection()
        try:
            result = await self.db.items.insert_many(items)
            self.logger.info(f"Inserted {len(result.inserted_ids)} items")
            return result.inserted_ids
        except Exception as e:
            self.logger.error(f"Error inserting items: {str(e)}")
            raise

    async def store_designers(self, designers: List[Dict]):
        await self.ensure_connection()
        try:
            result = await self.db.designers.insert_many(designers)
            self.logger.info(f"Inserted {len(result.inserted_ids)} designers")
            return result.inserted_ids
        except Exception as e:
            self.logger.error(f"Error inserting designers: {str(e)}")
            raise

    async def load_designers(self):
        await self.ensure_connection()
        try:
            designers = await self.db.designers.find().to_list(length=None)
            self.logger.info(f"Loaded {len(designers)} designers from MongoDB")
            return designers
        except Exception as e:
            print(f"Error loading designers from MongoDB: {str(e)}")
            return []

    async def get_all_item_ids(self) -> Set[str]:
        await self.ensure_connection()
        try:
            item_ids = await self.db.items.distinct("id")
            self.logger.info(f"Fetched {len(item_ids)} unique item IDs from MongoDB")
            return set(item_ids)
        except Exception as e:
            self.logger.error(f"Error fetching item IDs: {str(e)}")
            raise

    async def save_checkpoint(self, checkpoint: Dict):
        await self.ensure_connection()
        try:
            result = await self.db.checkpoints.replace_one({}, checkpoint, upsert=True)
            self.logger.info(f"Checkpoint saved: {checkpoint}")
            return result.upserted_id or result.modified_count
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")
            raise

    async def load_checkpoint(self) -> Dict:
        await self.ensure_connection()
        try:
            checkpoint = await self.db.checkpoints.find_one()
            if checkpoint:
                self.logger.info(f"Loaded checkpoint: {checkpoint}")
                return checkpoint
            else:
                self.logger.info("No checkpoint found, starting from beginning")
                total_items = await self.get_total_items()
                return {
                    "designer_slug": None,
                    "last_scroll_count": 0,
                    "total_items_scraped": total_items,
                    "timestamp": None,
                }
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            raise

    async def save_image_embedding_checkpoint(self, checkpoint: Dict):
        await self.ensure_connection()
        try:
            checkpoint["type"] = "image"
            result = await self.db.embedding_checkpoints.replace_one(
                {"type": "image"}, checkpoint, upsert=True
            )
            self.logger.info(f"Checkpoint saved: {checkpoint}")
            return result.upserted_id or result.modified_count
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")
            raise

    async def load_image_embedding_checkpoint(self) -> Dict:
        await self.ensure_connection()
        try:
            checkpoint = await self.db.embedding_checkpoints.find_one({"type": "image"})
            if checkpoint:
                self.logger.info(f"Loaded image embedding checkpoint: {checkpoint}")
                return checkpoint
            else:
                self.logger.info(
                    "No image embedding checkpoint found, starting from beginning"
                )
                return {
                    "type": "image",
                    "last_processed_id": None,
                    "total_items_processed": 0,
                    "timestamp": None,
                }
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            raise

    async def save_text_embedding_checkpoint(self, checkpoint: Dict):
        await self.ensure_connection()
        try:
            checkpoint["type"] = "text"
            result = await self.db.embedding_checkpoints.replace_one(
                {"type": "text"}, checkpoint, upsert=True
            )
            self.logger.info(f"Checkpoint saved: {checkpoint}")
            return result.upserted_id or result.modified_count
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")
            raise

    async def load_text_embedding_checkpoint(self) -> Dict:
        await self.ensure_connection()
        try:
            checkpoint = await self.db.embedding_checkpoints.find_one({"type": "text"})
            if checkpoint:
                self.logger.info(f"Loaded text embedding checkpoint: {checkpoint}")
                return checkpoint
            else:
                self.logger.info(
                    "No text embedding checkpoint found, starting from beginning"
                )
                return {
                    "type": "text",
                    "last_processed_id": None,
                    "total_items_processed": 0,
                    "timestamp": None,
                }
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            raise

    async def get_emebdding_status(self) -> Dict:
        await self.ensure_connection()
        try:
            image_checkpoint = await self.load_image_embedding_checkpoint()
            text_checkpoint = await self.load_text_embedding_checkpoint()
            return {
                "image": {
                    "last_processed_id": image_checkpoint["last_processed_id"],
                    "total_items_processed": image_checkpoint["total_items_processed"],
                    "timestamp": image_checkpoint["timestamp"],
                },
                "text": {
                    "last_processed_id": text_checkpoint["last_processed_id"],
                    "total_items_processed": text_checkpoint["total_items_processed"],
                    "timestamp": text_checkpoint["timestamp"],
                },
            }
        except Exception as e:
            self.logger.error(f"Error getting embedding status: {str(e)}")
            raise

    async def get_items_batch(
        self, last_processed_id: str = None, batch_size: int = 32
    ):
        """
        Get a batch of items from the database, starting after the last processed ID.

        Args:
            last_processed_id (str): ID of the last processed item
            batch_size (int): Number of items to retrieve in the batch

        Returns:
            list: A list of items (documents) from the database
        """
        await self.ensure_connection()
        try:
            query = {}
            if last_processed_id:
                query["_id"] = {"$gt": ObjectId(last_processed_id)}

            cursor = self.db.items.find(query).sort("_id", 1).limit(batch_size)
            batch = await cursor.to_list(length=batch_size)
            return batch
        except Exception as e:
            self.logger.error(f"Error retrieving batch: {str(e)}")
            raise

    async def update_checkpoint_total_items(self):
        try:
            total_items = await self.get_total_items()
            await self.db.checkpoints.update_one(
                {}, {"$set": {"total_items_scraped": total_items}}, upsert=True
            )
            self.logger.info(f"Updated checkpoint total_items_scraped to {total_items}")
        except Exception as e:
            self.logger.error(f"Error updating checkpoint total items: {str(e)}")
            raise

    async def get_total_items(self) -> int:
        await self.ensure_connection()
        try:
            total = await self.db.items.count_documents({})
            self.logger.info(f"Total items in database: {total}")
            return total
        except Exception as e:
            self.logger.error(f"Error getting total items: {str(e)}")
            raise

    async def get_image_urls(self, limit: int = None) -> List[Dict]:
        await self.ensure_connection()
        try:
            pipeline = [
                {"$project": {"_id": 0, "id": 1, "image.url": 1}},
                {"$match": {"image.url": {"$exists": True, "$ne": None}}},
            ]
            if limit:
                pipeline.append({"$limit": limit})

            cursor = self.db.items.aggregate(pipeline)
            image_urls = await cursor.to_list(length=None)
            self.logger.info(f"Retrieved {len(image_urls)} image URLs")
            return image_urls
        except Exception as e:
            self.logger.error(f"Error retrieving image URLs: {str(e)}")
            raise

    async def download_images(self, output_dir: str, num_images: int):
        image_urls = await self.get_image_urls(limit=num_images)
        os.makedirs(output_dir, exist_ok=True)

        async with aiohttp.ClientSession() as session:
            for item in image_urls:
                item_id = item["id"]
                url = item["image"]["url"]
                file_name = f"{item_id}.jpg"
                file_path = os.path.join(output_dir, file_name)

                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            with open(file_path, "wb") as f:
                                f.write(await response.read())
                            self.logger.info(f"Downloaded image for item {item_id}")
                        else:
                            self.logger.warning(
                                f"Failed to download image for item {item_id}: HTTP {response.status}"
                            )
                except Exception as e:
                    self.logger.error(
                        f"Error downloading image for item {item_id}: {str(e)}"
                    )

        self.logger.info(
            f"Finished downloading {len(image_urls)} images to {output_dir}"
        )

    # async def get_storage_size(self) -> int:
    #     await self.ensure_connection()
    #     try:
    #         stats = await self.db.command("dbstats")
    #         # storage_size = stats.get("storageSize", 0)
    #         self.logger.info(f"Current database storage size: {storage_size} bytes")
    #         return storage_size
    #     except Exception as e:
    #         self.logger.error(f"Error getting database storage size: {str(e)}")
    #         raise

    async def load_data(self):
        await self.ensure_connection()
        try:
            items = await self.db.items.find().to_list(length=None)
            self.logger.info(f"Fetched {len(items)} items from MongoDB")
            return items
        except Exception as e:
            self.logger.error(f"Error fetching items: {str(e)}")
            raise

    async def get_image_urls_by_ids(self, item_ids: List[str]) -> List[Dict[str, str]]:
        """
        Retrieve image URLs for the given item IDs.

        :param item_ids: List of item IDs to fetch image URLs for
        :return: List of dictionaries containing item IDs and their corresponding image URLs
        """
        await self.ensure_connection()

        try:
            # Fetch documents from MongoDB
            items = await self.db.items.find(
                {"id": {"$in": item_ids}}, projection={"id": 1, "image.url": 1}
            ).to_list(length=None)

            # Extract image URLs and match them with item IDs
            image_urls = [
                {"id": item["id"], "image_url": item["image"]["url"]}
                for item in items
                if "image" in item and "url" in item["image"]
            ]

            # Sort the results to match the order of input item_ids
            image_urls.sort(key=lambda x: item_ids.index(x["id"]))

            self.logger.info(
                f"Retrieved {len(image_urls)} image URLs for {len(item_ids)} item IDs"
            )
            return image_urls

        except Exception as e:
            self.logger.error(f"Error retrieving image URLs: {str(e)}")
            raise

    async def get_image_urls_from_pinecone_results(
        self, pinecone_results: Dict, num_results: int = 5
    ) -> List[Dict[str, str]]:
        """
        Retrieve image URLs for items from Pinecone search results.

        :param pinecone_results: Dictionary containing search results from Pinecone
        :param num_results: Number of results to retrieve (default: 5)
        :return: List of dictionaries containing item IDs and their corresponding image URLs
        """
        # Extract item IDs from Pinecone results
        item_ids = [match["id"] for match in pinecone_results["matches"][:num_results]]

        # Use the get_image_urls_by_ids method to fetch the image URLs
        return await self.get_image_urls_by_ids(item_ids)

    async def delete_low_count_designers(self, threshold: int):
        await self.ensure_connection()
        try:
            # Step 1: Get all designers and their counts
            designers = await self.db.designers.find().to_list(length=None)
            low_count_designers = [
                d["name"] for d in designers if d["count"] < threshold
            ]

            if not low_count_designers:
                self.logger.info(
                    f"No designers with less than {threshold} counts found."
                )
                return

            # Step 2: Delete items associated with low-count designers
            delete_result = await self.db.items.delete_many(
                {"designers": {"$in": low_count_designers}}
            )

            # Step 3: Delete the low-count designers themselves
            await self.db.designers.delete_many({"name": {"$in": low_count_designers}})

            self.logger.info(
                f"Deleted {delete_result.deleted_count} items associated with {len(low_count_designers)} low-count designers"
            )
            self.logger.info(
                f"Low-count designers removed: {', '.join(low_count_designers)}"
            )

            await self.update_checkpoint_total_items()

        except Exception as e:
            self.logger.error(
                f"Error deleting low-count designers and their items: {str(e)}"
            )
            raise

    async def get_designer_counts(self):
        await self.ensure_connection()
        try:
            designers = await self.db.designers.find().to_list(length=None)
            return {d["name"]: d["count"] for d in designers}
        except Exception as e:
            self.logger.error(f"Error fetching designer counts: {str(e)}")
            raise

    async def delete_documents_by_title_substring(self, substring: List[str]):
        await self.ensure_connection()
        deleted_count = 0
        try:
            for sub in substring:
                result = await self.db.items.delete_many(
                    {"title": {"$regex": sub, "$options": "i"}}
                )
                deleted_count += result.deleted_count
                self.logger.info(
                    f"Deleted {deleted_count} documents containing '{sub}' in the title"
                )
            await self.update_checkpoint_total_items()
            return deleted_count
        except Exception as e:
            self.logger.error(
                f"Error deleting documents with title containing '{substring}': {str(e)}"
            )
            raise

    async def count_duplicate_titles(self):
        await self.ensure_connection()
        try:
            pipeline = [
                {"$group": {"_id": "$title", "count": {"$sum": 1}}},
                {"$match": {"count": {"$gt": 1}}},
                {"$project": {"title": "$_id", "count": 1, "_id": 0}},
                {"$sort": {"count": -1}},
            ]
            cursor = self.db.items.aggregate(pipeline)
            duplicate_titles = await cursor.to_list(length=None)

            total_duplicates = sum(doc["count"] - 1 for doc in duplicate_titles)
            self.logger.info(
                f"Found {len(duplicate_titles)} unique titles with duplicates, totaling {total_duplicates} duplicate documents"
            )
            print(f"Found {len(duplicate_titles)} unique titles with duplicates")

            return duplicate_titles, total_duplicates
        except Exception as e:
            self.logger.error(f"Error counting duplicate titles: {str(e)}")
            raise

    async def delete_all_repeated_titles(self, batch_size: int = 1000) -> int:
        total_deleted = 0
        try:
            while True:
                # Find duplicates
                pipeline = [
                    {
                        "$group": {
                            "_id": "$title",
                            "uniqueIds": {"$addToSet": "$_id"},
                            "count": {"$sum": 1},
                        }
                    },
                    {"$match": {"count": {"$gt": 1}}},
                    {"$limit": batch_size},
                ]

                cursor = self.db.items.aggregate(pipeline, allowDiskUse=True)

                delete_ops = []
                async for doc in cursor:
                    # Keep the first occurrence, delete the rest
                    ids_to_delete = doc["uniqueIds"][1:]
                    delete_ops.append(DeleteMany({"_id": {"$in": ids_to_delete}}))

                if not delete_ops:
                    break  # No more duplicates found

                # Execute batch delete
                try:
                    result = await self.db.items.bulk_write(delete_ops, ordered=False)
                    deleted_in_batch = result.deleted_count
                    total_deleted += deleted_in_batch
                    self.logger.info(
                        f"Deleted {deleted_in_batch} duplicate documents in this batch"
                    )
                except BulkWriteError as bwe:
                    deleted_in_batch = bwe.details["nRemoved"]
                    total_deleted += deleted_in_batch
                    self.logger.warning(
                        f"BulkWriteError: Deleted {deleted_in_batch} documents, but some operations failed."
                    )

                # Update the checkpoint periodically
                if total_deleted % (batch_size * 10) == 0:
                    await self.update_checkpoint_total_items()

            # Final update to the checkpoint
            await self.update_checkpoint_total_items()

            self.logger.info(
                f"Total documents with repeated titles deleted: {total_deleted}"
            )
            return total_deleted

        except Exception as e:
            self.logger.error(
                f"Error deleting documents with repeated titles: {str(e)}"
            )
            raise

    async def delete_items_by_designers(
        self, designer_names: List[str]
    ) -> Dict[str, int]:
        """
        Delete documents in the items collection that contain any of the specified designer names in the designers array field.
        Args:
            designer_names (List[str]): A list of designer names to match against the designers array field.

        Returns:
            Dict[str, int]: A dictionary containing the number of documents deleted and the number of designers processed.
                - 'deleted_count': The number of documents that were deleted.
                - 'designers_processed': The number of designer names that were processed.

        Raises:
            Exception: If there's an error during the database operation, it logs the error and re-raises the exception.

        Example:
            result = await db_handler.delete_items_by_designers(['Gucci', 'Prada', 'Louis Vuitton'])
            print(f"Deleted {result['deleted_count']} items for {result['designers_processed']} designers")
        """

        await self.ensure_connection()
        try:
            # Construct the query to match documents where designers array contains any of the specified names
            query = {"designers": {"$in": designer_names}}

            # Execute the delete operation
            result = await self.db.items.delete_many(query)

            deleted_count = result.deleted_count
            self.logger.info(
                f"Deleted {deleted_count} items for designers: {', '.join(designer_names)}"
            )

            await self.update_checkpoint_total_items()

            return {
                "deleted_count": deleted_count,
                "designers_processed": len(designer_names),
            }
        except Exception as e:
            self.logger.error(
                f"Error deleting items for designers {designer_names}: {str(e)}"
            )
            raise

    async def get_documents_by_ids(self, ids: List[str]) -> List[Dict]:
        """
        Fetch documents from MongoDB by their IDs.

        Args:
            ids (List[str]): List of document IDs to fetch.

        Returns:
            List[Dict]: List of documents matching the given IDs.
        """
        await self.ensure_connection()
        try:
            documents = await self.db.items.find({"id": {"$in": ids}}).to_list(None)
            # self.logger.info(f"Fetched {len(documents)} documents from MongoDB")
            return documents
        except Exception as e:
            # self.logger.error(f"Error fetching documents by IDs: {str(e)}")
            raise


class ItemLimitExceeded(Exception):
    """Exception raised when the item limit is exceeded."""

    pass
