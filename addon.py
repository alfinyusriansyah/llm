from minio import Minio
from minio.error import S3Error

class MinioConnector:
    def __init__(self, url, access_key, secret_key):
        self.client = Minio(
            url,
            access_key=access_key,
            secret_key=secret_key,
            secure=False  # Set to True if using HTTPS
        )

    def list_files(self, bucket_name):
        try:
            objects = self.client.list_objects(bucket_name)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            print(f"Error: {e}")
            return []

# Example usage in addon.py (comment this out when not testing directly)
# connector = MinioConnector("10.1.112.226:9000", "EyY8D8fOlzyoCv2FJQPg", "JIFpxwDKr8XKLBoYdaj5S2jNPNznT2oJLtHllibW")
# print(connector.list_files("test-bucket"))
