import csv
from db_handler import DatabaseHandler
from config import Config
import asyncio
import re


class ScrapingUtils:

    @staticmethod
    def clean_text(text):
        # Step 1: Convert to lowercase and replace non-alphanumeric characters with spaces
        cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())

        # Step 2: Replace multiple spaces with a single space and strip leading/trailing spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    @staticmethod
    def clean_title(title, designers):
        title_lower = title.lower()
        for designer in designers:
            designer_lower = designer.lower()
            # Use regex to replace designer name and surrounding whitespace with a single space
            title_lower = re.sub(
                r"\s*\b" + re.escape(designer_lower) + r"\b\s*", " ", title_lower
            )
        # Remove leading/trailing whitespace and clean the text
        return ScrapingUtils.clean_text(title_lower.strip())

    @staticmethod
    def clean_category(category):
        cleaned_category = category.replace("*", " ").replace("_", " ")
        return " ".join(cleaned_category.lower().split())

    @staticmethod
    def get_sub_category(category_path):
        # Remove any characters before the last dot
        sub_category = category_path.split(".")[-1] if category_path else ""
        sub_category = sub_category.replace("*", " ").replace("_", " ")
        # Remove any extra spaces and return lowercase
        return " ".join(sub_category.lower().split())

    @staticmethod
    def process_color(color):
        # If color contains a slash, return 'multi', otherwise return the lowercase color
        return "multi" if "/" in color else color.lower()

    @staticmethod
    def create_description(color, department, main_category, sub_category):
        return f"{sub_category}".strip()


async def convert_mongodb_to_csv():
    config = Config()
    db_handler = DatabaseHandler(config.database_url, config.mongodb_db_name)

    await db_handler.connect()

    # Fetch all items from the database
    items = await db_handler.db.items.find().to_list(length=None)

    # Prepare CSV file
    with open(
        "notebooks/grailed_data.csv", "w", newline="", encoding="utf-8"
    ) as csvfile:
        fieldnames = ["prod_id", "prod_name", "prod_desc"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for item in items:
            designers = item.get("designers", [])
            cleaned_title = ScrapingUtils.clean_title(item["title"], designers)

            color = (
                ScrapingUtils.process_color(item["color"])
                if item["color"]
                else "unspecified"
            )
            department = item["department"]
            category = item["category"]["main"]
            sub_category = ScrapingUtils.get_sub_category(
                item["category"].get("sub", "")
            )

            description = ScrapingUtils.create_description(
                color, department, category, sub_category
            )

            writer.writerow(
                {
                    "prod_id": item["id"],
                    "prod_name": cleaned_title,
                    "prod_desc": description,
                }
            )

    await db_handler.close()
    print("CSV file 'grailed_data.csv' has been created successfully.")


if __name__ == "__main__":
    # asyncio.run(convert_mongodb_to_csv())
    title = "GREEN GOLD ALTERITA POLKA GLASS BALLS - SMALL HOOPS Size OS"
    print(ScrapingUtils.clean_text(title))
