import asyncio
from config import Config
from db_handler import DatabaseHandler


async def main():
    config = Config()
    db_handler = DatabaseHandler(config.database_url, config.mongodb_db_name)

    # Connect to the database
    await db_handler.connect()

    # Download images (limit to 100 for this example)
    await db_handler.download_images("downloaded_images", num_images=1920)

    # Close the database connection
    await db_handler.close()


if __name__ == "__main__":
    asyncio.run(main())
