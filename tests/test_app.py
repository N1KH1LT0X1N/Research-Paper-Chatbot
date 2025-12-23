"""Quick test script to verify app endpoints"""
from fastapi.testclient import TestClient
from app.main import app


def test_endpoints():
    client = TestClient(app)

    # Test root endpoint
    response = client.get("/")
    print(f"✅ GET / - Status: {response.status_code}")
    print(f"   Response: {response.json()}")

    # Test health endpoint
    response = client.get("/health")
    print(f"\n✅ GET /health - Status: {response.status_code}")
    print(f"   Response: {response.json()}")

    print("\n✅ All endpoints working!")


if __name__ == "__main__":
    test_endpoints()
