import requests
import json
import os

# --- Configuration ---
# You must enter your CDSE credentials here
USERNAME = "m4yukhd4s@gmail.com"
PASSWORD = "zeC6#yuevRb4h.Y"

# Bounding box near Novo Progresso, Pará (approx 20x20km)
# Format: POLYGON((lon lat, lon lat, ...))
BBOX_WKT = "POLYGON((-55.60 -7.20, -55.40 -7.20, -55.40 -7.00, -55.60 -7.00, -55.60 -7.20))"

# Search Parameters
COLLECTION = "SENTINEL-2"
PRODUCT_TYPE = "S2MSI2A"  # Level-2A (Bottom of Atmosphere)
MAX_CLOUD_COVER = 10

# Timeframes (June - Sept)
PERIOD_2018 = ("2018-06-01T00:00:00.000Z", "2018-09-30T23:59:59.999Z")
PERIOD_2024 = ("2024-06-01T00:00:00.000Z", "2024-09-30T23:59:59.999Z")

DOWNLOAD_DIR = "./sentinel_data"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# --- Step 1: Authentication ---


def get_token(username, password):
    print("Authenticating with Copernicus Data Space Ecosystem...")
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    response = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data=data
    )
    response.raise_for_status()
    return response.json()["access_token"]

# --- Step 2: Search ---


def search_catalog(start_date, end_date):
    print(
        f"Searching for images between {start_date[:10]} and {end_date[:10]}...")
    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    # OData Filter Query
    query = (
        f"?$filter=Collection/Name eq '{COLLECTION}' "
        f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{PRODUCT_TYPE}') "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{BBOX_WKT}') "
        f"and ContentDate/Start ge {start_date} and ContentDate/Start le {end_date} "
        f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {MAX_CLOUD_COVER})"
        f"&$top=1&$orderby=ContentDate/Start asc"  # Get the first best match
    )

    response = requests.get(url + query)
    response.raise_for_status()
    results = response.json().get("value", [])

    if not results:
        print("No images found matching the criteria.")
        return None

    best_image = results[0]
    print(f"-> Found Tile ID: {best_image['Name']}")
    return best_image

# --- Step 3: Download ---


# --- Step 3: Download (Updated with Redirect Handling) ---
def download_image(product_id, product_name, token):
    print(f"Downloading {product_name}...")
    url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}

    # Handle redirects manually to prevent the token from being dropped
    response = requests.get(url, headers=headers, allow_redirects=False)

    while response.is_redirect:
        url = response.headers['Location']
        response = requests.get(url, headers=headers, allow_redirects=False)

    # Once we reach the final destination, stream the file
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()

    file_path = os.path.join(DOWNLOAD_DIR, f"{product_name}.zip")
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"-> Saved locally to {file_path}")


# --- Execution Flow ---
if __name__ == "__main__":
    try:
        # Step 1
        token = get_token(USERNAME, PASSWORD)

        # Step 2 & 3 for 2018
        image_2018 = search_catalog(PERIOD_2018[0], PERIOD_2018[1])
        if image_2018:
            download_image(image_2018['Id'], image_2018['Name'], token)

        # Step 2 & 3 for 2024
        image_2024 = search_catalog(PERIOD_2024[0], PERIOD_2024[1])
        if image_2024:
            download_image(image_2024['Id'], image_2024['Name'], token)

        print(
            "\nProcess Complete. Both images are ready for your patch generation pipeline.")

    except Exception as e:
        print(f"An error occurred: {e}")
