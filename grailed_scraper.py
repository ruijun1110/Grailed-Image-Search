import asyncio
from playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeoutError,
)
from rich import print
import json
import time
from json_processor import JSONProcessor
from db_handler import DatabaseHandler

# from database_handler import DatabaseHandler
from config import Config

category = "other"


class GrailedScraper:
    def __init__(self, config: Config):
        self.config = config
        self.db_handler = DatabaseHandler(config.database_url, config.mongodb_db_name)
        self.json_processor = JSONProcessor()
        self.responses = []
        # self.last_scroll_count = 0
        self.total_items_scraped = 0
        self.json_file = "processed_items.json"
        self.seen_item_ids = set()
        self.checkpoint = None
        self.max_storage_size = config.max_storage_size
        self.max_retries = 3
        self.retry_delay = 5

    async def load_designers(self):
        return await self.db_handler.load_designers()

    async def check_response(self, response):
        if "queries?x-algolia-agent=Algolia" in response.url:
            json_response = await response.json()
            self.responses.append(json_response)

    async def handle_modal(self, page):
        try:
            # Check if the modal is present
            modal = await page.query_selector('div[class*="ReactModal_Overlay"]')
            if modal:
                # Click outside the modal to dismiss it
                await page.mouse.click(0, 0)
                # Wait for the modal to disappear
                await page.wait_for_selector(
                    'div[class*="ReactModal_Overlay"]', state="hidden", timeout=5000
                )
                print("Modal dismissed successfully")
            else:
                print("No modal detected")
        except Exception as e:
            print(f"Error handling modal: {str(e)}")

    async def scroll_page(self, page):
        for attempt in range(self.max_retries):
            try:
                previous_height = await page.evaluate("document.body.scrollHeight")
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                try:
                    await page.wait_for_load_state("networkidle")
                except PlaywrightTimeoutError:
                    print(f"Network idle timeout on attempt {attempt + 1}")
                await asyncio.sleep(2)
                new_height = await page.evaluate("document.body.scrollHeight")
                print(f"Scrolled from {previous_height} to {new_height}")
                if new_height == previous_height:
                    print("Reached end of page or no new content loaded")
                    return False
                # self.last_scroll_count += 1
                return True
            except PlaywrightTimeoutError as e:
                if attempt < self.max_retries - 1:
                    print(
                        f"Timeout error on attempt {attempt + 1}. Retrying in {self.retry_delay} seconds..."
                    )
                    await asyncio.sleep(self.retry_delay)
                else:
                    print(f"Maximum retries reached. Moving on to next designer.")
                    return False
            return False

    async def save_checkpoint(self, designer_slug, scroll_count):
        self.checkpoint = {
            "designer_slug": designer_slug,
            "last_scroll_count": scroll_count,
            "total_items_scraped": self.total_items_scraped,
        }
        await self.db_handler.save_checkpoint(self.checkpoint)
        print(f"Checkpoint saved: {self.checkpoint}")

    def save_to_json_file(self, data):
        with open(self.json_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {self.json_file}")

    async def process_and_store_data(self):
        processed_data = self.json_processor.process_data(self.responses)
        unique_data = self.json_processor.remove_duplicates(
            processed_data, self.seen_item_ids
        )
        if unique_data:
            # self.save_to_json_file(unique_data)
            await self.db_handler.store_items(unique_data)
            self.total_items_scraped += len(unique_data)
            self.seen_item_ids.update(item["id"] for item in unique_data)

            # Check storage size after storing items
            current_size = await self.db_handler.get_storage_size()
            if current_size >= self.max_storage_size:
                print(f"Reached maximum storage size: {self.max_storage_size} bytes")
                return False
        else:
            print("No new items to store")
        self.responses = []  # Clear processed responses
        return True

    async def scrape_designer(self, page, designer, start_scroll=0):
        designer_url = f"{self.config.base_url}/{designer['slug']}"
        # designer_url = f"{self.config.base_url}/{category}"
        print(f"Scraping designer: {designer['name']} at {designer_url}")

        try:
            await page.goto(designer_url)
            await page.wait_for_load_state("networkidle")
            await self.handle_modal(page)
        except PlaywrightTimeoutError:
            print(f"Timeout error loading designer page. Skipping {designer['name']}")
            return True

        scroll_count = 0
        while True:
            if scroll_count < start_scroll:
                # Skip processing data until we reach the last scroll count
                self.responses = []
                await self.scroll_page(page)
                scroll_count += 1
                continue

            has_more_content = await self.scroll_page(page)
            if not has_more_content:
                await self.process_and_store_data()
                await self.save_checkpoint(designer["slug"], scroll_count)
                print(
                    f"Reached end of page for {designer['name']}. Moving to next designer."
                )
                return True

            scroll_count += 1

            if (scroll_count - start_scroll) % self.config.scrolls_per_batch == 0:
                if not await self.process_and_store_data():
                    print("Storage limit reached. Stopping the scraping process.")
                    return False
                await self.save_checkpoint(designer["slug"], scroll_count)
                # await self.save_checkpoint(category, scroll_count)

            # if scroll_count % 10 == 0 or not has_more_content:
            #     await self.save_checkpoint(designer["slug"], scroll_count)

            if self.total_items_scraped >= self.config.max_items:
                print(f"Reached maximum items limit: {self.config.max_items}")
                return False

        return True

    async def run(self):
        designers = await self.load_designers()
        self.checkpoint = await self.db_handler.load_checkpoint()
        if self.checkpoint:
            start_index = next(
                (
                    i
                    for i, d in enumerate(designers)
                    if d["slug"] == self.checkpoint["designer_slug"]
                ),
                0,
            )
            self.total_items_scraped = self.checkpoint["total_items_scraped"]
            start_scroll = self.checkpoint["last_scroll_count"]
        else:
            start_index = 0
            start_scroll = 0

        self.seen_item_ids = set(await self.db_handler.get_all_item_ids())

        # Check initial storage size
        initial_size = await self.db_handler.get_storage_size()
        if initial_size >= self.max_storage_size:
            print(
                f"Initial storage size ({initial_size} bytes) already exceeds the limit ({self.max_storage_size} bytes). Cannot start scraping."
            )
            return

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            await page.set_viewport_size({"width": 1280, "height": 1080})
            page.on("response", self.check_response)

            for designer in designers[start_index:]:
                print(designer)
                start_scroll = (
                    self.checkpoint["last_scroll_count"]
                    if designer["slug"] == self.checkpoint["designer_slug"]
                    else 0
                )
                await self.scrape_designer(page, designer, start_scroll=start_scroll)

                # Check storage size after each designer
                current_size = await self.db_handler.get_storage_size()
                if current_size >= self.max_storage_size:
                    print(
                        f"Reached maximum storage size: {self.max_storage_size} bytes"
                    )
                    break

                if self.total_items_scraped >= self.config.max_items:
                    print(f"Reached maximum items limit: {self.config.max_items}")
                    break
            # await self.scrape_designer(page, category, start_scroll=start_scroll)
            # if self.total_items_scraped >= self.config.max_items:
            #     print(f"Reached maximum items limit: {self.config.max_items}")
            #     break

            await browser.close()

    async def resume(self):
        try:
            stored_checkpoint = await self.db_handler.load_checkpoint()
            if stored_checkpoint:
                self.checkpoint = stored_checkpoint
                print(f"Resuming from checkpoint: {self.checkpoint}")
            await self.run()
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        finally:
            if hasattr(self, "db_handler"):
                await self.db_handler.close()


if __name__ == "__main__":
    config = Config()
    scraper = GrailedScraper(config)
    asyncio.run(scraper.resume())
