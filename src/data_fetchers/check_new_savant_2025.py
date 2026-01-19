import os
import requests
import urllib3
from bs4 import BeautifulSoup

# Disable SSL warnings (safe here because we're downloading public static files)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://www.nflsavant.com"
DATA_PAGE = f"{BASE_URL}/about.php"
DATA_DIR = "data/raw/savant/"
SEASON = "2025"  # Update this for the season you want to fetch

def get_existing_filenames(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    return set(os.listdir(data_dir))

def get_savant_season_links():
    response = requests.get(DATA_PAGE, verify=False)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch page: {DATA_PAGE}")
    
    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.find_all('a', href=True)
    file_links = []

    print("🔍 Available download links:")
    for link in links:
        href = link['href']
        if f"year={SEASON}" in href.lower():
            full_url = href if href.startswith("http") else BASE_URL + "/" + href.lstrip("/")
            filename = f"pbp-{SEASON}.csv"
            print(f"- Matched {SEASON} link: {href}")
            file_links.append((filename, full_url))

    return file_links

def download_new_files(file_links, existing_files):
    downloaded = []
    for filename, url in file_links:
        print(f"🔗 Found: {filename} (checking if new)")
        if filename not in existing_files:
            print(f"📥 Downloading: {filename}")
            response = requests.get(url, verify=False)
            if response.status_code == 200:
                path = os.path.join(DATA_DIR, filename)
                with open(path, 'wb') as f:
                    f.write(response.content)
                downloaded.append(filename)
            else:
                print(f"⚠️ Failed to download {filename} (status {response.status_code})")
        else:
            print(f"✅ Already have: {filename}")
    return downloaded

def main():
    print(f"🔎 Checking for new NFL Savant {SEASON} data...")
    existing = get_existing_filenames(DATA_DIR)
    links = get_savant_season_links()
    new_files = download_new_files(links, existing)

    if new_files:
        print(f"\n✅ Downloaded {len(new_files)} new file(s): {new_files}")
    else:
        print("\n🚫 No new files found.")

if __name__ == "__main__":
    main()
