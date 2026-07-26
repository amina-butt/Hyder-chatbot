import os
from dotenv import load_dotenv
from google import genai

# Force reload environment variables from .env file
load_dotenv(override=True)

# Try reading both possible variable names
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ ERROR: No API key found in your .env file!")
    print("Make sure your .env file contains: GOOGLE_API_KEY=AIzaSy...")
else:
    print(f"Loaded Key: {api_key[:8]}...{api_key[-4:]}")
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents="Say hello!",
        )
        print("\n✅ API KEY WORKS GREAT!")
        print(f"Gemini Response: {response.text}")
    except Exception as e:
        print("\n❌ API KEY CALL FAILED:")
        print(e)