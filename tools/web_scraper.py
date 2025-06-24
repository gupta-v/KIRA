import os
import requests
import webbrowser
import logging
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from serpapi import GoogleSearch
from langchain_core.tools import Tool
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def scrape_w3schools(url):
    """Scrapes main content from W3Schools"""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        main_content = soup.find("div", id="main")
        if not main_content:
            return "Content not found."

        text = []
        for tag in main_content.find_all(["p"], limit=10):  # Limit results
            text.append(tag.get_text(strip=True))

        for tag in main_content.find_all("li", limit=10):  # Limit results
            text.append(tag.get_text(strip=True))
        
        return "\n".join(text)[:2000]  # Limit content length
    except Exception as e:
        return f"Error scraping W3Schools: {str(e)}"

def scrape_tutorialspoint(url):
    """Scrapes main content from TutorialsPoint."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        main_content = soup.find("div", id="mainContent")
        if not main_content:
            return "Content not found."

        # Extract text from all <p> tags
        text = [p.get_text(strip=True) for p in main_content.find_all("p", limit=10)]
        
        return "\n".join(text)[:2000]  # Limit content length
    except Exception as e:
        return f"Error scraping TutorialsPoint: {str(e)}"

def scrape_freecodecamp(url):
    """Scrapes main content from FreeCodeCamp"""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        article = soup.find("article")
        if not article:
            return "Content not found."

        text = [p.get_text(strip=True) for p in article.find_all("p", limit=10)]
        return "\n".join(text)[:2000]  # Limit content length
    except Exception as e:
        return f"Error scraping FreeCodeCamp: {str(e)}"

def scrape_programiz(url):
    """Scrapes main content from Programiz"""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        article = soup.find("article")
        if not article:
            return "Content not found."

        text = [p.get_text(strip=True) for p in article.find_all("p", limit=10)]
        return "\n".join(text)[:2000]  # Limit content length
    except Exception as e:
        return f"Error scraping Programiz: {str(e)}"

def scrape_url(url):
    """Chooses the appropriate scraper based on URL"""
    try:
        if "w3schools.com" in url:
            return scrape_w3schools(url)
        elif "tutorialspoint.com" in url:
            return scrape_tutorialspoint(url)
        elif "freecodecamp.org" in url:
            return scrape_freecodecamp(url)
        elif "programiz.com" in url:
            return scrape_programiz(url)
        else:
            return "Unsupported site."
    except Exception as e:
        return f"Error in scrape_url: {str(e)}"

# Get API key from environment
serp_api_key = os.getenv("SERP_API_KEY")

# List of allowed websites
ALLOWED_SITES = [
    "w3schools.com",
    "tpointtech.com",  
    "tutorialspoint.com",
    "freecodecamp.org",
    "programiz.com"
]

def similar(a, b):
    """Return a similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def search_best_link(query, api_key):
    """
    Use SerpAPI to search Google with a query restricted to allowed websites.
    From the organic results, take the first three allowed results (in order)
    and select the one with the highest title similarity to the query.
    """
    try:
        # Construct a site filter string using Google operator "site:"
        site_filter = " OR ".join([f"site:{site}" for site in ALLOWED_SITES])
        params = {
            "engine": "google",
            "q": f"{query} {site_filter}",
            "num": 20,  # Fetch more results to have a larger pool
            "api_key": api_key
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])
        
        if not organic_results:
            return None

        # Filter results to only those from allowed sites (in their given order)
        allowed_results = []
        for result in organic_results:
            link = result.get("link", "")
            if any(site in link for site in ALLOWED_SITES):
                allowed_results.append(result)
            if len(allowed_results) >= 3:
                break

        if not allowed_results:
            return None

        # Compare the first three allowed results using title similarity
        best_link = None
        best_score = 0
        for result in allowed_results:
            link = result.get("link", "")
            title = result.get("title", "")
            score = similar(query, title)
            if score > best_score:
                best_score = score
                best_link = link

        return best_link
    
    except Exception as e:
        logger.error(f"Error in search_best_link: {e}")
        return None

def web_response(query):
    """Main function to search and scrape web content"""
    try:
        api_key = serp_api_key
        if not api_key:
            return "SERP API key not found. Please set SERP_API_KEY in your environment variables."
        
        query = query.strip()
        logger.info(f"Web scraper search for: {query}")
        
        best_link = search_best_link(query, api_key)
        if best_link:
            print("Best link found:", best_link)
            try:
                webbrowser.open(best_link)
            except Exception as e:
                logger.warning(f"Could not open browser: {e}")
            
            # Scrape the content from the best link
            content = scrape_url(best_link)
            
            # Ensure the output folder exists
            output_folder = "data/scrapings"
            os.makedirs(output_folder, exist_ok=True)

            # Save the scraped content to a file
            output_file = os.path.join(output_folder, "scraped_content.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Scraped content has been stored in '{output_file}'")
            
            # Read and return the contents of the scraped file
            with open(output_file, "r", encoding="utf-8") as f:
                scraped_contents = f.read()
            
            logger.info("Web scraping completed successfully")
            return scraped_contents
        else:
            error_msg = "No suitable link found for your query."
            print(error_msg)
            logger.warning(error_msg)
            return error_msg
            
    except Exception as e:
        error_msg = f"Web response failed for '{query}'. Error: {str(e)}"
        logger.error(error_msg)
        return error_msg


# --- Tool Definition ---
web_scraper_tool = Tool(
    name="WebScraper",
    func=web_response,
    description=(
        "Scrapes educational websites like W3Schools, TutorialsPoint, freeCodeCamp, and Programiz. "
        "Provide a query, and it finds the most relevant article to scrape."
    )
)