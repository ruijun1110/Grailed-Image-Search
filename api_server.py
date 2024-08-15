from quart import Quart, request, jsonify, Response
from quart_cors import cors
from db_handler import DatabaseHandler
from config import Config
import asyncio
import queue
import threading
from grailed_scraper import GrailedScraper
from custom_logger import setup_logger
from clip_embedding_generator import CLIPEmbeddingsGenerator
from pydantic import BaseModel
import logging
from typing import List, Dict
from transformers import (
    CLIPModel,
    CLIPTextModelWithProjection,
    AutoTokenizer,
    CLIPImageProcessor,
)

app = Quart(__name__)
app = cors(app)

config = Config()
db_handler = DatabaseHandler(config.database_url, config.mongodb_db_name)
log_queue = queue.Queue()
image_embedding_log_queue = queue.Queue()
text_embedding_log_queue = queue.Queue()
logger = setup_logger(__name__, log_queue)
image_embedding_logger = setup_logger("image_embedding", image_embedding_log_queue)
text_embedding_logger = setup_logger("text_embedding", text_embedding_log_queue)

# Cache for storing the latest checkpoint
latest_checkpoint = None

logging.basicConfig(level=logging.DEBUG)
server_logger = logging.getLogger("server")


class SimilaritySearchRequest(BaseModel):
    image_url: str
    text_query: str
    top_k: int = 12


def checkpoint_callback(checkpoint):
    global latest_checkpoint
    latest_checkpoint = checkpoint
    logger.info(f"Checkpoint updated: {checkpoint}")


# def image_embedding_checkpoint_callback(checkpoint):
#     global latest_image_embedding_checkpoint
#     latest_image_embedding_checkpoint = checkpoint
#     image_embedding_logger.info(f"Image embedding checkpoint updated: {checkpoint}")


# def text_embedding_checkpoint_callback(checkpoint):
#     global latest_text_embedding_checkpoint
#     latest_text_embedding_checkpoint = checkpoint
#     text_embedding_logger.info(f"Text embedding checkpoint updated: {checkpoint}")


scraper = GrailedScraper(config, log_queue, checkpoint_callback)
image_embedding_generator = CLIPEmbeddingsGenerator(
    config,
    image_embedding_log_queue,
    mode="image",
)
text_embedding_generator = CLIPEmbeddingsGenerator(
    config,
    text_embedding_log_queue,
    mode="text",
)

image_embedding_task = None
text_embedding_task = None


@app.before_serving
async def setup():
    await db_handler.connect()


@app.after_serving
async def cleanup():
    await image_embedding_generator.close()
    await text_embedding_generator.close()
    await db_handler.close()


def run_scraper():
    asyncio.run(scraper.resume())


@app.route("/api/start_scraping", methods=["POST"])
def start_scraping():
    if not scraper.is_scraping:
        threading.Thread(target=run_scraper).start()
        return jsonify({"message": "Scraping started"}), 200
    else:
        return jsonify({"message": "Scraping is already in progress"}), 400


@app.route("/api/stop_scraping", methods=["POST"])
async def stop_scraping():
    if scraper.is_scraping:
        await scraper.stop_scraping()
        return jsonify({"message": "Scraping stopped"}), 200
    else:
        return jsonify({"message": "No scraping process is currently running"}), 400


@app.route("/api/scraping_logs", methods=["GET"])
def scraping_logs():
    def generate():
        while True:
            try:
                log = log_queue.get(timeout=1)
                yield f"data: {log}\n\n"
            except queue.Empty:
                yield f"data: \n\n"

    return Response(generate(), content_type="text/event-stream")


@app.route("/api/delete_items_by_substring", methods=["POST"])
async def delete_items_by_substring():
    data = await request.json
    if "substrings" not in data or not isinstance(data["substrings"], list):
        return jsonify({"ERROR": "Invalid or missing 'substrings' in request"}), 400

    substrings = data["substrings"]
    try:
        deleted_count = await db_handler.delete_documents_by_title_substring(substrings)
        return (
            jsonify(
                {
                    "message": f"Deleted {deleted_count} documents",
                }
            ),
            200,
        )
    except Exception as e:
        logger.error(f"Error in delete_items_by_substring: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/delete_items_by_designers", methods=["POST"])
async def delete_items_by_designers():
    data = await request.json
    if "designers" not in data or not isinstance(data["designers"], list):
        return jsonify({"error": "Invalid or missing 'designers' in request"}), 400

    designers = data["designers"]
    try:
        result = await db_handler.delete_items_by_designers(designers)
        return (
            jsonify(
                {
                    "message": f"Deleted {result['deleted_count']} documents for {result['designers_processed']} designers",
                }
            ),
            200,
        )
    except Exception as e:
        logger.error(f"Error in delete_items_by_designers: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/delete_low_count_designers", methods=["POST"])
async def delete_low_count_designers():
    data = await request.json
    threshold = data.get("threshold")

    if threshold is None or not isinstance(threshold, int):
        return jsonify({"error": "Invalid or missing threshold value"}), 400

    try:
        result = await db_handler.delete_low_count_designers(threshold)
        return (
            jsonify(
                {
                    "message": f"Deleted {result['deleted_items']} items associated with {result['deleted_designers']} low-count designers",
                }
            ),
            200,
        )
    except Exception as e:
        logger.error(f"Error in delete_low_count_designers: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/start_image_embedding", methods=["POST"])
async def start_image_embedding():
    await image_embedding_generator.start_embedding()
    return jsonify({"message": "Image embedding process started"}), 200


@app.route("/api/start_text_embedding", methods=["POST"])
async def start_text_embedding():
    await text_embedding_generator.start_embedding()
    return jsonify({"message": "Text embedding process started"}), 200


@app.route("/api/stop_image_embedding", methods=["POST"])
async def stop_image_embedding():
    await image_embedding_generator.stop_embedding()
    return jsonify({"message": "Image embedding process stopping..."}), 200


@app.route("/api/stop_text_embedding", methods=["POST"])
async def stop_text_embedding():
    await text_embedding_generator.stop_embedding()
    return jsonify({"message": "Text embedding process stopping..."}), 200


@app.route("/api/image_embedding_logs", methods=["GET"])
async def image_embedding_logs():
    def generate():
        while True:
            try:
                log = image_embedding_log_queue.get(timeout=1)
                yield f"data: {log}\n\n"
            except queue.Empty:
                yield f"data: \n\n"

    return Response(generate(), content_type="text/event-stream")


@app.route("/api/text_embedding_logs", methods=["GET"])
async def text_embedding_logs():
    def generate():
        while True:
            try:
                log = text_embedding_log_queue.get(timeout=1)
                yield f"data: {log}\n\n"
            except queue.Empty:
                yield f"data: \n\n"

    return Response(generate(), content_type="text/event-stream")


@app.route("/api/get_scraping_status", methods=["GET"])
async def get_scraping_status():
    if latest_checkpoint:
        checkpoint_callback(latest_checkpoint)
        return jsonify({"message": "Scraping status updated"}), 200
    else:
        try:
            checkpoint = await db_handler.load_checkpoint()
            if checkpoint["timestamp"]:
                checkpoint_callback(checkpoint)
                return jsonify({"message": "Scraping status updated"}), 200
            else:
                return jsonify({"message": "No scraping status available"}), 200
        except Exception as e:
            logger.error(f"Error in gettiing scraping status: {str(e)}")
            return jsonify({"error": str(e)}), 500


@app.route("/api/get_image_embedding_status", methods=["GET"])
async def get_image_embedding_status():
    if image_embedding_generator.latest_embedding_checkpoint:
        image_embedding_generator.embedding_checkpoint_callback(
            image_embedding_generator.latest_embedding_checkpoint
        )
        return jsonify({"message": "Image embedding status updated"}), 200
    else:
        try:
            checkpoint = await db_handler.load_image_embedding_checkpoint()
            if checkpoint["timestamp"]:
                image_embedding_generator.embedding_checkpoint_callback(checkpoint)
                return jsonify({"message": "Image embedding status updated"}), 200
            else:
                return jsonify({"message": "No image embedding status available"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route("/api/get_text_embedding_status", methods=["GET"])
async def get_text_embedding_status():
    if text_embedding_generator.latest_embedding_checkpoint:
        text_embedding_generator.embedding_checkpoint_callback(
            text_embedding_generator.latest_embedding_checkpoint
        )
        return jsonify({"message": "Text embedding status updated"}), 200
    else:
        try:
            checkpoint = await db_handler.load_text_embedding_checkpoint()
            if checkpoint["timestamp"]:
                text_embedding_generator.embedding_checkpoint_callback(checkpoint)
                return jsonify({"message": "Text embedding status updated"}), 200
            else:
                return jsonify({"message": "No text embedding status available"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500


class SimilaritySearchResult(BaseModel):
    id: str
    score: float


class SimilaritySearchResponse(BaseModel):
    results: List[SimilaritySearchResult]


def process_pinecone_results(results: Dict) -> List[SimilaritySearchResult]:
    return [
        SimilaritySearchResult(id=match["id"], score=match["score"])
        for match in results
    ]


@app.route("/api/similarity_search", methods=["POST"])
async def similarity_search():
    try:
        server_logger.info("Received similarity search request")
        data = await request.json
        image_url = data.get("image_url")
        text_query = data.get("text_query")
        # category = data.get("category")
        top_k = data.get("top_k", 10)
        server_logger.info(f"Request data: {data}")

        image_results = []
        text_results = []

        if image_url:
            image_results = await image_embedding_generator.search_similar_image(
                image_url, top_k
            )
            server_logger.info(f"Image results: {image_results}")

        if text_query:
            # Load CLIP text model and tokenizer
            text_results = await text_embedding_generator.search_similar_text(
                text_query, top_k
            )
            server_logger.info(f"Text results: {text_results}")

        id_score_map = {}
        # Combine results and remove duplicates
        combined_results = image_results + text_results

        for result in combined_results:
            mongo_id = int(result["id"])
            if mongo_id not in id_score_map or result["score"] > id_score_map[mongo_id]:
                id_score_map[mongo_id] = result["score"]

        # Sort the results by score, from highest to lowest
        sorted_results = sorted(id_score_map.items(), key=lambda x: x[1], reverse=True)

        server_logger.critical(f"Sorted results: {sorted_results}")

        # Fetch corresponding documents from MongoDB
        mongo_docs = await db_handler.get_documents_by_ids(
            [id for id, _ in sorted_results]
        )

        server_logger.critical(f"Mongo docs: {mongo_docs}")

        # Create a mapping of id to mongo document for easy lookup
        mongo_docs_map = {doc["id"]: doc for doc in mongo_docs}

        # Create the final sorted list of documents with their scores
        sorted_docs_with_scores = []
        for id, score in sorted_results:
            if id in mongo_docs_map:
                doc = mongo_docs_map[id].copy()  # Create a copy of the document
                doc.pop("_id", None)  # Remove the _id field
                doc["similarity_score"] = score
                sorted_docs_with_scores.append(doc)

        return jsonify(sorted_docs_with_scores)
    except Exception as e:
        server_logger.error(f"Error in similarity search: {str(e)}")
        return jsonify({"error": str(e)}), 500


def run_image_embedding():
    image_index = image_embedding_generator.initialize_index(
        "grailed-image-to-image", dimension=512
    )
    asyncio.run(image_embedding_generator.process_all_data(image_index, mode="image"))


def run_text_embedding():
    text_index = text_embedding_generator.initialize_index(
        "grailed-text-to-text", dimension=512
    )
    asyncio.run(text_embedding_generator.process_all_data(text_index, mode="text"))


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
