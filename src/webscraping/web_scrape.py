import requests
from bs4 import BeautifulSoup
import concurrent.futures
import os
import time
import urllib.parse

# Seed URL to start the crawl
seed_url = 'https://en.wikipedia.org/wiki/Main_Page'

# Maximum depth to crawl
max_depth = 7
# Directory to save the scraped data
output_dir = 'Z:\scraped_data'
os.makedirs(output_dir, exist_ok=True)

# Set to keep track of visited URLs
visited_urls = set()

def fetch_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def parse_html(html, url, depth):
    if html is None:
        return []
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Extract article title
    title_tag = soup.find('h1', {'id': 'firstHeading'})
    article_title = title_tag.text if title_tag else 'Unknown_Article'
    
    # Clean the title to make it filesystem-friendly
    article_title = "".join(c for c in article_title if c.isalnum() or c in (' ', '-', '_')).strip()
    
    # Extract text content
    text = soup.get_text(separator=' ', strip=True)
    
    # Save the text to a file using article title
    filename = os.path.join(output_dir, f"{article_title}.txt")
    with open(filename, 'w', encoding='utf-8') as file:
        # file.write(f"Title: {article_title}\n")
        # file.write(f"Source URL: {url}\n\n")
        file.write(text)
    print(f"Saved article: {article_title}")
    
    # Extract links to follow
    links = []
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if href.startswith('/wiki/') and ':' not in href:
            full_url = urllib.parse.urljoin(url, href)
            if full_url not in visited_urls:
                links.append((full_url, depth + 1))
                visited_urls.add(full_url)
    
    return links

def crawl_url(url, depth):
    if depth > max_depth:
        return
    
    html = fetch_url(url)
    links = parse_html(html, url, depth)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for link, next_depth in links:
            executor.submit(crawl_url, link, next_depth)

def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.submit(crawl_url, seed_url, 0)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Crawling completed in {end_time - start_time:.2f} seconds")