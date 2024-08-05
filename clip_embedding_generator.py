import torch
from transformers import (
    CLIPModel,
    CLIPTextModelWithProjection,
    AutoTokenizer,
    CLIPImageProcessor,
)
from PIL import Image
from pymongo import MongoClient
from pinecone import Pinecone
from tqdm import tqdm
import requests
from io import BytesIO
from config import Config
from db_handler import DatabaseHandler
from matplotlib import pyplot as plt


class CLIPEmbeddingsGenerator:
    def __init__(self):
        self.config = Config()
        self.db_handler = DatabaseHandler(
            self.config.database_url, self.config.mongodb_db_name
        )
        self.text_model = CLIPTextModelWithProjection.from_pretrained(
            self.config.embedding_model
        )
        self.text_processor = AutoTokenizer.from_pretrained(self.config.embedding_model)
        self.image_model = CLIPModel.from_pretrained(self.config.embedding_model)
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.config.embedding_model
        )
        self.pinecone = Pinecone(api_key=self.config.pinecone_api_key)

    def load_data_from_mongodb(self):
        return self.db_handler.load_data()

    def preprocess_image(self, image_url):
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        return self.image_processor(images=image, return_tensors="pt")

    def preprocess_text(self, text):
        return self.text_processor(text=text, return_tensors="pt", padding=True)

    def generate_image_embedding(self, image_input):
        with torch.no_grad():
            return self.image_model.get_image_features(**image_input)

    def generate_text_embedding(self, text_input):
        with torch.no_grad():
            return self.text_model(**text_input).text_embeds

    def create_pinecone_index(self, index_name, dimension, metric="cosine"):
        if not self.pinecone:
            raise ValueError("Pinecone API key not provided.")
        self.pinecone.create_index(name=index_name, dimension=dimension, metric=metric)
        return self.pinecone.Index(index_name)

    def upsert_to_pinecone(self, index, vectors):
        index.upsert(vectors=vectors)

    def search_pinecone(self, index, query_vector, top_k=10, filter=None):
        return index.query(
            vector=query_vector, top_k=top_k, filter=filter, include_metadata=True
        )

    def process_batch(self, batch, image_index, text_index):
        image_embeddings = []
        text_embeddings = []
        for item in batch:
            # Process image
            image_input = self.preprocess_image(item["image"]["url"])
            image_embedding = self.generate_image_embedding(image_input)
            image_embeddings.append(
                {
                    "id": str(item["id"]),
                    "values": image_embedding.numpy().tolist()[0],
                    "metadata": {"category": item["category"]["sub"]},
                }
            )

            # Process text
            text_input = self.preprocess_text(item["title"])
            text_embedding = self.generate_text_embedding(text_input)
            text_embeddings.append(
                {
                    "id": str(item["id"]),
                    "values": text_embedding,
                    "metadata": {"category": item["category"]["sub"]},
                }
            )

        self.upsert_to_pinecone(image_index, image_embeddings)
        self.upsert_to_pinecone(text_index, text_embeddings)

    def process_all_data(self, data, image_index, text_index, batch_size=32):
        for i in tqdm(range(0, data.count(), batch_size)):
            batch = list(data.skip(i).limit(batch_size))
            self.process_batch(batch, image_index, text_index)

    def initialize_index(self, index_name, dimension, metric="cosine"):
        if index_name not in self.pinecone.list_indexes().names():
            self.pinecone.create_index(
                name=index_name, dimension=dimension, metric=metric
            )

        return self.pinecone.Index(index_name)

    def plot_search_results(self, query_image_url, pinecone_results, num_results=5):
        """
        Plot the search results using image URLs retrieved from the database.

        :param pinecone_results: Dictionary containing search results from Pinecone
        :param num_results: Number of results to display (default: 5)
        """

        # Get image URLs from the database handler
        image_urls = self.db_handler.get_image_urls_from_pinecone_results(
            pinecone_results, num_results
        )

        # Create a figure with subplots
        fig, axs = plt.subplots(1, num_results + 1, figsize=(20, 4))
        fig.suptitle("Search Results")

        query_image = Image.open(BytesIO(requests.get(query_image_url).content))
        axs[0].imshow(query_image)
        axs[0].axis("off")
        axs[0].set_title("Query Image")

        for i, item in enumerate(image_urls):
            try:
                response = requests.get(item["image_url"])
                img = Image.open(BytesIO(response.content))

                axs[i + 1].imshow(img)
                axs[i + 1].axis("off")
                axs[i + 1].set_title(
                    f"Score: {pinecone_results['matches'][i]['score']:.2f}"
                )
            except Exception as e:
                print(f"Error processing image for item {item['id']}: {str(e)}")
                axs[i + 1].text(
                    0.5, 0.5, "Image Not Available", ha="center", va="center"
                )
                axs[i + 1].axis("off")

        plt.tight_layout()
        plt.show()


def main():
    # Initialize the generator
    generator = CLIPEmbeddingsGenerator()

    # Load data from MongoDB
    data = generator.load_data_from_mongodb()

    # Create Pinecone indexes
    image_index = generator.initialize_index("grailed-image-to-text", dimension=512)
    text_index = generator.initialize_index("grailed-text-to-image", dimension=512)

    # Process and upsert embeddings
    generator.process_all_data(data, image_index, text_index)

    # Example search (image-based)
    query_image_url = "https://example.com/query_image.jpg"
    query_image_input = generator.preprocess_image(query_image_url)
    query_embedding = generator.generate_image_embedding(query_image_input)
    category = "hair accessories"

    results = generator.search_pinecone(
        image_index,
        query_embedding.numpy().tolist()[0],
        filter={"category": category},
    )

    # Plot search results
    generator.plot_search_results(query_image_url, results)


if __name__ == "__main__":
    main()
