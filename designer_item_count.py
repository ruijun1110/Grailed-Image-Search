from config import Config
from db_handler import DatabaseHandler
import asyncio


async def main():
    config = Config()  # Assuming you have a Config class
    db_handler = DatabaseHandler(config.database_url, config.mongodb_db_name)

    # Connect to the database
    await db_handler.connect()

    # Get initial designer counts
    initial_counts = await db_handler.get_designer_counts()
    print("Initial designer counts:", initial_counts)

    # Delete low-count designers and their items
    await db_handler.delete_low_count_designers()

    # Get updated designer counts
    updated_counts = await db_handler.get_designer_counts()
    print("Updated designer counts:", updated_counts)

    # Close the connection
    await db_handler.close()


if __name__ == "__main__":
    asyncio.run(main())
