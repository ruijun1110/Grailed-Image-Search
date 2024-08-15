from config import Config
from db_handler import DatabaseHandler
import asyncio


async def main():
    config = Config()  # Assuming you have a Config class
    db_handler = DatabaseHandler(config.database_url, config.mongodb_db_name)

    # Connect to the database
    await db_handler.connect()
    await db_handler.delete_low_count_designers()
    # await db_handler.count_duplicate_titles()
    substrings = [
        # "o1mle0624",
        # "o1mle0424",
        # "o1rshd1",
        # "o1rshd11223",
        # "o1g2r1mq0823",
        # "o1rshd10124",
        # "o1d2blof0724",
        # "o1d2blof01023",
        # "o1d2blof0124",
        # "o1d2blof0823",
        # "o1c11t2y1123",
        # "o1rshd",
        # "o1c11t2y1023",
        # "o1c11t2y0124",
        # "o1c11t2y0923",
        # "o1c11t2y0923",
        # "o1b1f11ly0823",
        # "o1c11t2y0823",
        # "o1dk11s0124",
        # "o1s22i1n0224",
        # "o1y0824",
        # "o1s22i1n0324",
        # "o1h1sh10624",
        # "o1smst1ft0424",
        # "o1b21g0423",
        # "o1y0323",
        # "o1dk11s0923",
        # "o1b1f11ly0723",
        # "o1mle0724",
        # "o1d2blof0923",
        # "o1s22i1n1223",
        # "o1d2blof0824",
        # "o1mj1ld1sgn0324",
        # "o11b112oob0523",
        # "o1s22i1n0124",
        # "o1alsim1",
        # "o1d2blof0224",
        "o1"
    ]
    # deleted_count = await db_handler.delete_documents_by_title_substring(
    #     substring=substrings
    # )
    # print(f"Deleted {deleted_count} documents containing substrings in the title")

    await db_handler.close()


if __name__ == "__main__":
    asyncio.run(main())
