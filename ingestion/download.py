

from azure.storage.blob import ContainerClient
from dotenv import load_dotenv
import os

# Load your connection string securely from .env
load_dotenv()
connection_str = os.getenv("AZURE_CONNECTION_STRING")
container_name = os.getenv("AZURE_CONTAINER_NAME")

# Connect to Azure Blob container
container_client = ContainerClient.from_connection_string(
    conn_str=connection_str,
    container_name=container_name
)

print(f"\n✅ Connected to container: {container_name}")
print("Listing blobs:\n")

# List and print blobs
for blob in container_client.list_blobs():
    print(" -", blob.name)
