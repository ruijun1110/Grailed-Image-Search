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


@app.route("/api/scraping/start", methods=["POST"])
def start_scraping():
    if not scraper.is_scraping:
        threading.Thread(target=run_scraper).start()
        return jsonify({"message": "Scraping started"}), 200
    else:
        return jsonify({"message": "Scraping is already in progress"}), 400


@app.route("/api/scraping/stop", methods=["POST"])
async def stop_scraping():
    if scraper.is_scraping:
        await scraper.stop_scraping()
        return jsonify({"message": "Scraping stopped"}), 200
    else:
        return jsonify({"message": "No scraping process is currently running"}), 400


@app.route("/api/scraping/logs", methods=["GET"])
def scraping_logs():
    def generate():
        while True:
            try:
                log = log_queue.get(timeout=1)
                yield f"data: {log}\n\n"
            except queue.Empty:
                yield f"data: \n\n"

    return Response(generate(), content_type="text/event-stream")


@app.route("/api/scraping/delete/substrings", methods=["DELETE"])
async def delete_items_by_substring():
    substrings = request.args.getlist("substring")
    if substrings:
        try:
            deleted_count = await db_handler.delete_documents_by_title_substring(
                substrings
            )
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
    else:
        return jsonify({"error": "No substrings provided"}), 400


@app.route("/api/scraping/delete/designers", methods=["DELETE"])
async def delete_items_by_designers():
    designers = request.args.getlist("designer")
    if designers:
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
    else:
        return jsonify({"error": "No designers provided"}), 400


@app.route("/api/scraping/delete/low_count", methods=["DELETE"])
async def delete_low_count_designers():

    threshold = request.args.get("threshold", type=int)

    if threshold:
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
    else:
        return jsonify({"error": "No threshold provided"}), 400


@app.route("/api/embeddings/<embedding_type>/start", methods=["POST"])
async def start_embedding(embedding_type):
    if embedding_type not in ["image", "text"]:
        return jsonify({"error": "Invalid embedding type"}), 400

    if embedding_type == "image":
        await image_embedding_generator.start_embedding()
    else:
        await text_embedding_generator.start_embedding()

    return (
        jsonify(
            {"message": f"{embedding_type.capitalize()} embedding process started"}
        ),
        200,
    )


@app.route("/api/embeddings/<embedding_type>/stop", methods=["POST"])
async def stop_embedding(embedding_type):
    if embedding_type not in ["image", "text"]:
        return jsonify({"error": "Invalid embedding type"}), 400

    if embedding_type == "image":
        await image_embedding_generator.stop_embedding()
    else:
        await text_embedding_generator.stop_embedding()

    return (
        jsonify(
            {"message": f"{embedding_type.capitalize()} embedding process stopping..."}
        ),
        200,
    )


@app.route("/api/embeddings/image/logs", methods=["GET"])
async def image_embedding_logs():
    def generate():
        while True:
            try:
                log = image_embedding_log_queue.get(timeout=1)
                yield f"data: {log}\n\n"
            except queue.Empty:
                yield f"data: \n\n"

    return Response(generate(), content_type="text/event-stream")


@app.route("/api/embeddings/text/logs", methods=["GET"])
async def text_embedding_logs():
    def generate():
        while True:
            try:
                log = text_embedding_log_queue.get(timeout=1)
                yield f"data: {log}\n\n"
            except queue.Empty:
                yield f"data: \n\n"

    return Response(generate(), content_type="text/event-stream")


@app.route("/api/scraping/status", methods=["GET"])
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


@app.route("/api/embeddings/<embedding_type>/status", methods=["GET"])
async def get_embedding_status(embedding_type):
    if embedding_type not in ["image", "text"]:
        return jsonify({"error": "Invalid embedding type"}), 400

    if embedding_type == "image":
        return await get_image_embedding_status()
    else:
        return await get_text_embedding_status()


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


@app.route("/api/search", methods=["GET"])
async def similarity_search():
    try:
        server_logger.info("Received similarity search request")
        image_url = request.args.get("image_url")
        text_query = request.args.get("text_query")
        top_k = request.args.get("top_k", default=10, type=int)
        # category = data.get("category")

        image_results = []
        text_results = []

        if image_url:
            image_results = await image_embedding_generator.search_similar_image(
                image_url, top_k
            )

        if text_query:
            # Load CLIP text model and tokenizer
            text_results = await text_embedding_generator.search_similar_text(
                text_query, top_k
            )

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
