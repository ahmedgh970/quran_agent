from setuptools import setup, find_packages

setup(
    name='quran_agent',
    version='0.0.1',
    description='RAG agent over the Quran with LangChain',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'openai',
        'httpx',
        'python-dotenv',
        'tqdm',
        'langchain',
        'langchain_chroma',
        'langchain_community',   
        'langchain_openai',
        'chromadb',
    ],
    entry_points={
        "console_scripts": [
            "qagent=quran_agent.run_pipe:main",
        ],
    },
)