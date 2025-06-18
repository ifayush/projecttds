# TDS Virtual TA - Knowledge Base Q&A System

A modern web application that allows users to ask questions and get answers from a scraped knowledge base. Built with FastAPI backend and a beautiful React-like frontend, designed to be deployed on Vercel.

## Features

- 🤖 **AI-Powered Q&A**: Ask questions and get intelligent answers from your knowledge base
- 🎨 **Modern UI**: Clean, responsive design similar to promptfoo eval
- 🔍 **Semantic Search**: Advanced vector similarity search for relevant content
- 📱 **Mobile Responsive**: Works perfectly on all devices
- ⚡ **Fast Performance**: Optimized for quick responses
- 🔗 **Source Links**: Get relevant links to original sources
- 🌐 **Vercel Ready**: Easy deployment with zero configuration

## Tech Stack

### Frontend
- **HTML5/CSS3**: Modern, responsive design
- **Vanilla JavaScript**: No framework dependencies
- **Font Awesome**: Beautiful icons
- **Inter Font**: Clean typography

### Backend
- **FastAPI**: High-performance Python web framework
- **SQLite**: Lightweight database for knowledge base
- **NumPy**: Vector operations for similarity search
- **aiohttp**: Async HTTP client for API calls
- **AIproxy**: External AI service integration

## Quick Start

### Prerequisites

1. **Python 3.9+** installed on your system
2. **Vercel CLI** (for deployment)
3. **AIPROXY_TOKEN** environment variable set

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd tds
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create a .env file
   echo "AIPROXY_TOKEN=your_api_key_here" > .env
   ```

4. **Run the development server**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8000`

### Vercel Deployment

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy to Vercel**
   ```bash
   vercel --prod
   ```

4. **Set environment variables in Vercel**
   - Go to your Vercel dashboard
   - Navigate to your project settings
   - Add `AIPROXY_TOKEN` with your API key

## Project Structure

```
tds/
├── api/
│   └── index.py          # FastAPI application for Vercel
├── downloaded_threads/   # Scraped discourse threads
├── markdown_files/       # Scraped markdown documents
├── index.html           # Main frontend page
├── styles.css           # CSS styles
├── script.js            # Frontend JavaScript
├── app.py              # Local development server
├── embedding.py         # Embedding utilities
├── discourse_scraper.py # Discourse scraping script
├── html_crawler.py      # HTML crawling script
├── requirements.txt     # Python dependencies
├── vercel.json         # Vercel configuration
└── README.md           # This file
```

## API Endpoints

### POST `/api/query`
Query the knowledge base with a question.

**Request Body:**
```json
{
  "question": "What are the best practices for data visualization?"
}
```

**Response:**
```json
{
  "answer": "Based on the knowledge base...",
  "links": [
    {
      "url": "https://example.com/doc1",
      "text": "Data Visualization Guide"
    }
  ]
}
```

### GET `/api/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "TDS Virtual TA API is running"
}
```

## Configuration

### Environment Variables

- `AIPROXY_TOKEN`: Your AIproxy API key for embeddings and completions

### Database

The application uses SQLite with two main tables:
- `discourse_chunks`: Stores scraped discourse forum content
- `markdown_chunks`: Stores scraped markdown documents

### Similarity Search

- **Threshold**: 0.68 (configurable in `SIMILARITY_THRESHOLD`)
- **Max Results**: 10 (configurable in `MAX_RESULTS`)
- **Context Chunks**: 4 per source (configurable in `MAX_CONTEXT_CHUNKS`)

## Customization

### Styling
Modify `styles.css` to customize the appearance:
- Color scheme
- Typography
- Layout
- Animations

### Functionality
- Update `script.js` for frontend behavior changes
- Modify `api/index.py` for backend logic changes
- Adjust similarity thresholds in the API file

### Knowledge Base
- Add new scraping scripts for different sources
- Update the database schema if needed
- Modify the embedding generation process

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   - Ensure `AIPROXY_TOKEN` is set in your environment variables
   - Check Vercel environment variables if deployed

2. **Database Not Found**
   - The application will create the database automatically
   - Ensure the application has write permissions

3. **CORS Errors**
   - CORS is configured to allow all origins
   - Check if your API endpoint is correct

4. **Slow Responses**
   - Check your AIproxy API rate limits
   - Consider adjusting similarity thresholds
   - Optimize your knowledge base size

### Debug Mode

Enable debug logging by modifying the logging level in `api/index.py`:
```python
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation

---

**Built with ❤️ for the TDS community** 