"""
Setup configuration for Research Paper Chatbot
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="research-paper-chatbot",
    version="2.0.0",
    author="N1KH1LT0X1N",
    description="AI-powered research paper assistant via WhatsApp",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/N1KH1LT0X1N/Research-Paper-Chatbot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "python-dotenv>=1.0.0",
        "sqlalchemy[asyncio]>=2.0.0",
        "aiosqlite>=0.19.0",
        "redis[hiredis]>=5.0.0",
        "httpx>=0.25.0",
        "twilio>=8.10.0",
        "google-generativeai>=0.3.0",
        "sentence-transformers>=2.2.2",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "PyPDF2>=3.0.0",
        "pdfplumber>=0.10.0",
        "Pillow>=10.1.0",
        "chromadb>=0.4.0",
        "celery[redis]>=5.3.0",
        "pydantic>=2.5.0",
        "python-multipart>=0.0.6",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.4.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "research-bot=app.main:main",
        ],
    },
)
