import asyncio
from playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeoutError,
)
from rich import print
import json
import time
from json_processor import JSONProcessor
from db_handler import DatabaseHandler, ItemLimitExceeded
import logging
from logging.handlers import QueueHandler
import queue
from custom_logger import setup_logger
from datetime import datetime
import pytz

# from database_handler import DatabaseHandler
from config import Config

category = "other"


class GrailedScraper:
    def __init__(
        self, config: Config, log_queue: queue.Queue, checkpoint_callback=None
    ):
        self.config = config
        self.db_handler = DatabaseHandler(config.database_url, config.mongodb_db_name)
        self.json_processor = JSONProcessor()
        self.responses = []
        # self.last_scroll_count = 0
        self.total_items_scraped = 0
        self.json_file = "processed_items.json"
        self.seen_item_ids = set()
        self.checkpoint = None
        # self.max_storage_size = config.max_storage_size
        self.max_items = config.max_items
        self.max_retries = 3
        self.retry_delay = 5
        self.is_scraping = False
        self.checkpoint_callback = checkpoint_callback
        self.browser = None

        self.logger = setup_logger(__name__, log_queue)

    def check_item_limit(self):
        """
        Check if the current number of scraped items exceeds the specified limit.

        Raises:
            ItemLimitExceeded: If the current number of items exceeds the maximum limit.
        """
        if self.total_items_scraped >= self.max_items:
            raise ItemLimitExceeded(
                f"Item limit of {self.max_items} exceeded. Current items: {self.total_items_scraped}"
            )

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
                self.logger.info("Modal dismissed successfully")
            else:
                self.logger.info("No modal detected")
        except Exception as e:
            self.logger.error(f"Error handling modal: {str(e)}")

    async def scroll_page(self, page):
        for attempt in range(self.max_retries):
            try:
                previous_height = await page.evaluate("document.body.scrollHeight")
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                try:
                    await page.wait_for_load_state("networkidle")
                except PlaywrightTimeoutError:
                    self.logger.warning(
                        f"Network idle timeout on attempt {attempt + 1}"
                    )
                await asyncio.sleep(2)
                new_height = await page.evaluate("document.body.scrollHeight")
                if new_height == previous_height:
                    self.logger.info("Reached end of page or no new content loaded")
                    return False
                # self.last_scroll_count += 1
                return True
            except PlaywrightTimeoutError as e:
                if attempt < self.max_retries - 1:
                    self.logger.warning(
                        f"Timeout error on attempt {attempt + 1}. Retrying in {self.retry_delay} seconds..."
                    )
                    await asyncio.sleep(self.retry_delay)
                else:
                    self.logger.warning(
                        f"Maximum retries reached. Moving on to next designer."
                    )
                    return False
            return False

    async def save_checkpoint(self, designer_slug, scroll_count):
        self.checkpoint = {
            "designer_slug": designer_slug,
            "last_scroll_count": scroll_count,
            "total_items_scraped": self.total_items_scraped,
            "timestamp": datetime.now(pytz.utc).isoformat(),
        }
        await self.db_handler.save_checkpoint(self.checkpoint)
        if self.checkpoint_callback:
            self.checkpoint_callback(self.checkpoint)

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
            try:
                # self.save_to_json_file(unique_data)
                await self.db_handler.store_items(unique_data)
                self.total_items_scraped += len(unique_data)
                self.seen_item_ids.update(item["id"] for item in unique_data)

                self.check_item_limit()
            except ItemLimitExceeded as e:
                self.logger.error(f"Item limit exceeded: {str(e)}")
                await self.save_checkpoint(
                    self.current_designer["slug"], self.scroll_count
                )
                return False
            except Exception as e:
                self.logger.error(f"Error storing items: {str(e)}")
                return False
        else:
            self.logger.info("No new items to store")
        self.responses = []  # Clear processed responses
        return True

    async def scrape_designer(self, page, designer, start_scroll=0):
        designer_url = f"{self.config.base_url}/{designer['slug']}"
        self.logger.info(f"Scraping designer: {designer['name']}")

        try:
            await page.goto(designer_url)
            await page.wait_for_load_state("networkidle")
            await self.handle_modal(page)
        except PlaywrightTimeoutError:
            self.logger.warning(
                f"Timeout error loading designer page. Skipping {designer['name']}"
            )
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
                self.logger.info(
                    f"Reached end of page for {designer['name']}. Moving to next designer."
                )
                return True

            scroll_count += 1

            if (scroll_count - start_scroll) % self.config.scrolls_per_batch == 0:
                if not await self.process_and_store_data():
                    self.logger.info(
                        "Storage limit reached. Stopping the scraping process."
                    )
                    return False
                await self.save_checkpoint(designer["slug"], scroll_count)
                # await self.save_checkpoint(category, scroll_count)

            # if scroll_count % 10 == 0 or not has_more_content:
            #     await self.save_checkpoint(designer["slug"], scroll_count)

            if self.total_items_scraped >= self.config.max_items:
                self.logger.error(
                    f"Reached maximum items limit: {self.config.max_items}"
                )
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
            self.logger.info(
                f"Resuming from designer: {designers[start_index]['name']}"
            )
        else:
            start_index = 0
            start_scroll = 0

        self.seen_item_ids = set(await self.db_handler.get_all_item_ids())

        if self.checkpoint_callback:
            self.checkpoint_callback(self.checkpoint)

        try:
            self.check_item_limit()
        except ItemLimitExceeded as e:
            self.logger.error(f"Item limit exceeded: {str(e)}")
            return

        async with async_playwright() as p:
            self.browser = await p.chromium.launch(headless=False)
            page = await self.browser.new_page()
            await page.set_viewport_size({"width": 1280, "height": 1080})
            page.on("response", self.check_response)

            for designer in designers[start_index:]:
                if not self.is_scraping:
                    self.logger.info("Scraping process stopped.")
                    break
                # print(designer)
                start_scroll = (
                    self.checkpoint["last_scroll_count"]
                    if designer["slug"] == self.checkpoint["designer_slug"]
                    else 0
                )
                await self.scrape_designer(page, designer, start_scroll=start_scroll)

                try:
                    self.check_item_limit()
                except ItemLimitExceeded as e:
                    self.logger.error(f"Item limit exceeded: {str(e)}")
                    break

                # if self.total_items_scraped >= self.config.max_items:
                #     self.logger.info(
                #         f"Reached maximum items limit: {self.config.max_items}"
                #     )
                #     break

            await self.stop_scraping()

    async def resume(self):
        try:
            self.is_scraping = True
            stored_checkpoint = await self.db_handler.load_checkpoint()
            if stored_checkpoint:
                self.checkpoint = stored_checkpoint
                self.logger.info(f"Resuming from checkpoint: {self.checkpoint}")
                if self.checkpoint_callback:
                    self.checkpoint_callback(self.checkpoint)
            await self.run()
        except Exception as e:
            self.logger.error(f"An error occurred: {str(e)}")
        finally:
            if hasattr(self, "db_handler"):
                await self.db_handler.close()
            self.is_scraping = False

    async def stop_scraping(self):
        self.is_scraping = False
        if self.browser:
            await self.browser.close()
        self.browser = None
        self.logger.info("Stopping scraping process...")


if __name__ == "__main__":
    config = Config()
    scraper = GrailedScraper(config)
    asyncio.run(scraper.resume())
