import asyncio
from playwright.async_api import async_playwright
import json
from config import Config
from db_handler import DatabaseHandler


class GrailedDesignerScraper:
    def __init__(self, config: Config):
        self.config = config
        self.db_handler = DatabaseHandler(config.database_url, config.mongodb_db_name)
        self.base_url = "https://www.grailed.com/designers"
        self.designers = []

    async def check_response(self, response):
        if response.url == "https://www.grailed.com/api/designers":
            json_data = await response.json()
            self.designers.extend(json_data["data"])
            print(f"Captured {len(json_data['data'])} designers")

    async def scrape_designers(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            page.on("response", self.check_response)

            await page.goto(self.base_url)
            await page.wait_for_load_state("networkidle")

            await browser.close()

    async def save(self):
        with open("designers.json", "w") as f:
            json.dump({"designers": self.designers}, f, indent=2)
        print(f"Saved {len(self.designers)} designers to designers.json")
        await self.db_handler.store_designers(self.designers)
        print(f"Saved {len(self.designers)} designers to MongoDB")

    async def run(self):
        await self.scrape_designers()
        await self.save()


if __name__ == "__main__":
    config = Config()
    scraper = GrailedDesignerScraper(config=config)
    asyncio.run(scraper.run())
