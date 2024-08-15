import torch
from transformers import (
    CLIPModel,
    CLIPTextModelWithProjection,
    AutoTokenizer,
    CLIPImageProcessor,
)
from PIL import Image
from pymongo import MongoClient
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import requests
from io import BytesIO
from config import Config
from db_handler import DatabaseHandler
from matplotlib import pyplot as plt
import signal
import asyncio
import aiohttp
import io
import queue
import json
import pytz
from datetime import datetime
from custom_logger import setup_logger
from concurrent.futures import ProcessPoolExecutor
from csv_converter import ScrapingUtils
import logging


class CLIPEmbeddingsGenerator:
    def __init__(
        self,
        config: Config,
        log_queue: queue.Queue,
        mode="both",
    ):
        self.config = config
        self.db_handler = DatabaseHandler(config.database_url, config.mongodb_db_name)
        self.mode = mode
        if mode == "text":
            self.text_model = CLIPTextModelWithProjection.from_pretrained(
                self.config.embedding_model
            )
            self.text_processor = AutoTokenizer.from_pretrained(
                self.config.embedding_model
            )
        elif mode == "image":
            self.image_model = CLIPModel.from_pretrained(self.config.embedding_model)
            self.image_processor = CLIPImageProcessor.from_pretrained(
                self.config.embedding_model
            )
        self.pinecone = Pinecone(api_key=self.config.pinecone_api_key)

        self.batch_size = self.config.embedding_batch_size
        self.checkpoint_interval = (
            self.config.embedding_checkpoint_interval
        )  # Save state every 1000 items
        self.max_retries = 3
        self.retry_delay = 5
        self.embedding_record_limit = self.config.embedding_record_limit

        self.logger = setup_logger(f"{mode}_embedding", log_queue)
        self.total_processed = 0
        self.last_processed_id = None
        self.terminate_event = asyncio.Event()
        self.pause_requested = asyncio.Event()
        self.embedding_task = None
        self.latest_embedding_checkpoint = None

    async def close(self):
        self.pause_requested.set()
        self.terminate_event.set()
        if self.embedding_task:
            await self.embedding_task
        self.embedding_task = None
        await self.db_handler.close()
        self.logger.info(f"{self.mode.capitalize()} embedding process stopped.")

    def embedding_checkpoint_callback(self, checkpoint):
        self.latest_embedding_checkpoint = checkpoint
        self.logger.info(f"{self.mode} embedding checkpoint updated: {checkpoint}")

    async def process_image_batch(self, batch, image_index):
        image_embeddings = []
        self.logger.info(f"Processing image item batch...")
        async with aiohttp.ClientSession() as session:
            for item in batch:
                if self.pause_requested.is_set() or self.terminate_event.is_set():
                    self.logger.warning("Pausing the image embedding process")
                    break
                try:
                    # Process image
                    image_url = item["image"]["url"]
                    image_embedding = await self.generate_image_embedding(
                        image_url, session
                    )
                    if image_embedding is not None:
                        image_embeddings.append(
                            {
                                "id": str(item["id"]),
                                "values": image_embedding,
                                "metadata": {
                                    "category": item["category"].get("sub"),
                                },
                            }
                        )

                except Exception as e:
                    self.logger.error(f"Error processing item {item['id']}: {str(e)}")
                    continue

        # Upsert to Pinecone
        if image_embeddings:
            self.upsert_to_pinecone(image_index, image_embeddings)
            self.logger.info(f"Upserted {len(image_embeddings)} image embeddings")

    async def process_text_batch(self, batch, text_index):
        """
        Process a batch of items, generating embeddings and upserting to Pinecone.

        Args:
            batch (list): A list of items (documents) from the database
            text_index (pinecone.Index): Pinecone index for text embeddings

        Returns:
            None
        """
        text_embeddings = []
        self.logger.info(f"Processing text item batch...")

        for item in batch:
            if self.pause_requested.is_set() or self.terminate_event.is_set():
                self.logger.warning("Pausing the text embedding process")
                break
            try:

                # Process text
                title = item["title"]
                text_embedding = self.generate_text_embedding(title)
                if text_embedding is not None:
                    text_embeddings.append(
                        {
                            "id": str(item["id"]),
                            "values": text_embedding[0],
                            "metadata": {
                                "category": item["category"].get("sub"),
                            },
                        }
                    )

            except Exception as e:
                self.logger.error(f"Error processing item {item['id']}: {str(e)}")
                continue

        # Upsert to Pinecone
        if text_embeddings:
            self.upsert_to_pinecone(text_index, text_embeddings)
            self.logger.info(f"Upserted {len(text_embeddings)} text embeddings")

    async def process_batch_with_retry(self, batch, index):
        for attempt in range(self.max_retries):
            try:
                if self.mode == "image":
                    if (
                        not self.pause_requested.is_set()
                        and not self.terminate_event.is_set()
                    ):
                        await self.process_image_batch(batch, index)
                elif self.mode == "text":
                    if (
                        not self.pause_requested.is_set()
                        and not self.terminate_event.is_set()
                    ):
                        await self.process_text_batch(batch, index)
                return
            except Exception as e:
                if attempt < self.max_retries - 1:
                    self.logger.exception(
                        f"Error processing {self.mode} batch: {e}. Retrying in {self.retry_delay} seconds..."
                    )
                    await asyncio.sleep(self.retry_delay)
                else:
                    self.logger.error(
                        f"Failed to process {self.mode} batch after {self.max_retries} attempts."
                    )
                    raise

    def load_data_from_mongodb(self):
        return self.db_handler.load_data()

    async def generate_image_embedding(self, image_url, session):
        """Generate embedding for an image from its URL."""
        try:
            async with session.get(image_url) as response:
                if response.status != 200:
                    self.logger.warning(
                        f"Failed to fetch image from {image_url}. Status: {response.status}"
                    )
                    return None

                image_data = await response.read()
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                inputs = self.image_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    embedding = self.image_model.get_image_features(**inputs)
                return embedding.numpy().tolist()[0]
        except Exception as e:
            self.logger.error(f"Error generating image embedding: {str(e)}")
            return None

    def generate_text_embedding(self, text_input):
        try:
            processed_text = self.text_processor(
                text=text_input, return_tensors="pt", padding=True
            )
            # text_input = self.preprocess_text(processed_text)
            with torch.no_grad():
                return self.text_model(**processed_text).text_embeds
        except Exception as e:
            self.logger.error(f"Error generating text embedding: {str(e)}")
            return None

    def create_pinecone_index(self, index_name, dimension, metric="cosine"):
        if not self.pinecone:
            self.logger.error("Pinecone API key not provided.")
            raise ValueError("Pinecone API key not provided.")
        self.pinecone.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        self.logger.info(f"Created Pinecone index: {index_name}")
        return self.pinecone.Index(index_name)

    def upsert_to_pinecone(self, index, vectors):
        if index and vectors:
            index.upsert(vectors=vectors)
        else:
            self.logger.error("Index or vectors not provided.")

    async def search_similar_image(self, image_url, top_k=10):
        async with aiohttp.ClientSession() as session:
            image_embedding = await self.generate_image_embedding(image_url, session)

        if image_embedding is None:
            return []

        image_index = self.pinecone.Index("grailed-image-to-image")
        # filter = ({"category": {"$eq": category}},)
        result = image_index.query(vector=image_embedding, top_k=top_k)
        return result["matches"]

    async def search_similar_text(self, text, top_k=10):
        server_logger = logging.getLogger("server")
        cleaned_text = ScrapingUtils.clean_text(text)
        server_logger.info(f"Cleaned text: {cleaned_text}")
        text_embedding = self.generate_text_embedding(cleaned_text)
        if text_embedding is None:
            return []
        text_index = self.pinecone.Index("grailed-text-to-text")
        # filter = ({"category": {"$eq": category}},)
        result = text_index.query(vector=text_embedding.numpy().tolist(), top_k=top_k)
        return result["matches"]

    async def process_all_data(self, index):
        checkpoint = None
        self.logger.info(f"Starting {self.mode} embedding processing...")
        if self.mode == "image":
            checkpoint = await self.db_handler.load_image_embedding_checkpoint()
        elif self.mode == "text":
            checkpoint = await self.db_handler.load_text_embedding_checkpoint()
        last_processed_id = checkpoint["last_processed_id"] if checkpoint else None
        total_processed = checkpoint["total_items_processed"] if checkpoint else 0

        self.logger.info(
            f"Resuming {self.mode} processing from item ID: {last_processed_id}"
        )

        self.embedding_checkpoint_callback(checkpoint)

        while not self.pause_requested.is_set() and not self.terminate_event.is_set():
            self.logger.info(f"Retrieving {self.mode} item batch from database...")
            batch = await self.db_handler.get_items_batch(
                last_processed_id, self.batch_size
            )
            if not batch:
                break  # No more items to process

            try:
                await self.process_batch_with_retry(batch, index)
            except Exception as e:
                self.logger.error(f"Error processing {self.mode} batch: {str(e)}")
                continue
            total_processed += len(batch)
            last_processed_id = str(batch[-1]["_id"])
            # pbar.update(len(batch))

            await self.save_checkpoint(last_processed_id, total_processed)

            if total_processed >= self.embedding_record_limit:
                self.logger.warning(
                    f"Processed {total_processed} items. Reached record limit."
                )
                break

            if self.terminate_event.is_set() or self.pause_requested.is_set():
                self.logger.warning("Pausing the image embedding process")
                await self.save_checkpoint(last_processed_id, total_processed)
                return

    async def start_embedding(self):
        if not self.embedding_task or self.embedding_task.done():
            index_name = f"grailed-{self.mode}-to-{self.mode}"
            index = self.initialize_index(index_name, dimension=512)
            self.embedding_task = asyncio.create_task(self.process_all_data(index))
            self.pause_requested.clear()
            self.logger.info(f"{self.mode.capitalize()} embedding process started.")
        else:
            self.logger.warning(
                f"{self.mode.capitalize()} embedding process is already running."
            )

    def initialize_index(self, index_name, dimension, metric="cosine"):
        if index_name not in self.pinecone.list_indexes().names():
            self.logger.info(f"{self} index not found. Creating new index...")
            self.pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        return self.pinecone.Index(index_name)

    async def save_checkpoint(self, last_processed_id, total_processed):
        checkpoint = {
            "last_processed_id": last_processed_id,
            "total_items_processed": total_processed,
            "timestamp": datetime.now(pytz.utc).isoformat(),
        }
        if self.mode == "image":
            await self.db_handler.save_image_embedding_checkpoint(checkpoint)
        if self.mode == "text":
            await self.db_handler.save_text_embedding_checkpoint(checkpoint)
        self.embedding_checkpoint_callback(checkpoint)

    async def stop_embedding(self):
        self.pause_requested.set()
        if self.embedding_task:
            await self.embedding_task
        self.image_embedding_task = None
        self.logger.info(f"{self.mode.capitalize()} embedding process stopped.")


async def main(generator: CLIPEmbeddingsGenerator):
    # config = Config()
    # # db_handler = DatabaseHandler(config.database_url, config.mongodb_db_name)
    # generator = CLIPEmbeddingsGenerator(config)

    image_index = generator.initialize_index("grailed-image-to-image", dimension=512)
    text_index = generator.initialize_index("grailed-text-to-text", dimension=512)

    print("Process starting. Press Ctrl+C at any time to pause the process.")
    try:

        await generator.process_all_data(image_index, text_index)
    finally:
        await generator.close()

    # # Example search (image-based)
    # query_image_url = "https://example.com/query_image.jpg"
    # query_image_input = generator.preprocess_image(query_image_url)
    # query_embedding = generator.generate_image_embedding(query_image_input)
    # category = "hair accessories"

    # results = generator.search_pinecone(
    #     image_index,
    #     query_embedding.numpy().tolist()[0],
    #     filter={"category": category},
    # )

    # # Plot search results
    # generator.plot_search_results(query_image_url, results)


# async def close(self):
#     await self.session.close()


if __name__ == "__main__":
    asyncio.run(main())
