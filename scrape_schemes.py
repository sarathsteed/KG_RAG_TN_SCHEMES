import requests
from bs4 import BeautifulSoup
import json

url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")

schemes = []
# Extract scheme titles and detailed pages/content
for row in soup.find_all("tr"):
    cols = row.find_all("td")
    if len(cols) >= 2:
        title = cols[1].text.strip()
        link = cols[1].find("a")["href"] if cols[1].find("a") else None
        if title:
            schemes.append({"title": title, "link": link})

print(f"Extracted {len(schemes)} schemes.")
with open("schemes.json", "w") as f:
    json.dump(schemes, f, indent=2)