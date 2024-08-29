# Grailed Image Search

Grailed Image Search is a sophisticated web application and browser extension that enables users to find visually similar clothing items on the Grailed platform. This project combines web scraping, machine learning, and a user-friendly interface to provide a powerful search experience for fashion enthusiasts.

## Features

- Web scraping of Grailed designers and clothing items
- Image and text embedding generation using CLIP (Contrastive Language-Image Pre-Training)
- Similarity search using Pinecone vector database
- RESTful API for managing scraping and embedding processes
- Chrome extension for easy search initiation from Grailed product pages
- Next.js web application for displaying search results

## Project Components

This project consists of multiple components, each with its own repository:

1. API Server: Current Repo
2. [Admin Dashboard](https://github.com/ruijun1110/grailed-image-search-dashboard)
3. [User Frontend](https://github.com/ruijun1110/Grailed_Similarity_Search_Display_Page)
4. [Chrome Extension](https://github.com/ruijun1110/Grailed_Similarity_Search_Chrome_Extension)

Please visit each repository for component-specific documentation and setup instructions.

## Tech Stack

### Backend
- Python 3.12
- FastAPI (Quart) for API development
- MongoDB for data storage
- Pinecone for vector similarity search
- PyTorch and Transformers for machine learning models
- Playwright for web scraping
- Asyncio for asynchronous programming

### Frontend
- Next.js 13+ with React
- Tailwind CSS for styling

### Browser Extension
- Chrome Extension API
- JavaScript

### DevOps & Deployment
- Docker for containerization
- Google Cloud Secret Manager for secure configuration management

## Project Structure

- `api_server.py`: Main API server implementation
- `grailed_scraper.py`: Web scraping logic for Grailed
- `clip_embedding_generator.py`: Image and text embedding generation
- `db_handler.py`: Database operations handler
- `config.py`: Configuration management
- `custom_logger.py`: Custom logging setup


## Demo

[Placeholder for demo video]

## Contributing

We welcome contributions to all parts of the Grailed Image Search project. Please refer to the CONTRIBUTING.md file in each component's repository for specific guidelines on how to contribute.

## License

This project is licensed under the [Choose a License]. Please see the LICENSE file in each component's repository for more details.

## Acknowledgements

- [CLIP (Contrastive Language-Image Pre-Training)](https://github.com/openai/CLIP)
- [Pinecone](https://www.pinecone.io/)
- [Grailed](https://www.grailed.com/)
