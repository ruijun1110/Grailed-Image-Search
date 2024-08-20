import os
from google.cloud import secretmanager


class Config:
    def __init__(self):
        self.base_url = self._get_config(
            "BASE_URL", "https://www.grailed.com/designers"
        )
        self.database_url = self._get_secret("MONGODB_URL")
        self.mongodb_db_name = self._get_config("MONGODB_NAME", "grailed_data")
        self.max_items = int(self._get_config("MAX_ITEMS", 850000))
        self.scrolls_per_batch = int(self._get_config("SCROLLS_PER_BATCH", 4))
        self.responses_per_batch = int(self._get_config("RESPONSES_PER_BATCH", 50))
        self.embedding_model = self._get_config(
            "EMBEDDING_MODEL", "openai/clip-vit-base-patch32"
        )
        self.pinecone_api_key = self._get_secret("PINECONE_API_KEY")
        self.embedding_batch_size = self._get_config("EMBEDDING_BATCH_SIZE", 64)
        self.embedding_checkpoint_interval = self._get_config(
            "EMBEDDING_CHECKPOINT_INTERVAL", 1000
        )
        self.embedding_record_limit = self._get_config("EMBEDDING_RECORD_LIMIT", 200000)

        actual_max_storage_size = 500 * 1024 * 1024  # 500 MB in bytes
        self.max_storage_size = int(
            actual_max_storage_size * 0.95
        )  # 95% of max storage size

    def _get_config(self, key, default="Environment variable not set"):
        return os.environ.get(key, default)

    def _get_secret(self, secret_id):
        try:
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{os.environ.get('GOOGLE_CLOUD_PROJECT')}/secrets/{secret_id}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            print(f"Error retrieving secret {secret_id}: {e}")
            return None
