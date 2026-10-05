# 📄 PDF Q&A

Chat with your PDFs. Upload one or more documents, ask a question, and get an answer with the
**file name and page number** it came from.

**Live demo:** _add your Streamlit link here_

## Features
- **Multi-PDF chat:** upload several PDFs and ask across all of them.
- **Cited answers:** every answer lists its source passages with file and page number.
- **Works without an API key:** shows the best-matching sentences from your documents.
- **Optional AI answers:** add an Anthropic API key and Claude writes the answer using only the retrieved passages. If the AI call fails, the app falls back to the best matches.
- **Lightweight:** pure-Python BM25 search, so there are no heavy ML libraries and it runs on free hosting.
- **Friendly errors:** warns about scanned (image-only), password-protected or corrupt PDFs.

## How it works
```
PDF  ->  extract text per page  ->  split into overlapping chunks (~140 words)
     ->  BM25 index  ->  your question retrieves the top passages
     ->  answer: Claude (if API key set) or best-matching sentences  +  page citations
```

## Quick start
```bash
git clone https://github.com/<your-username>/pdf-qa-app.git
cd pdf-qa-app
pip install -r requirements.txt
streamlit run app.py
```
Optional AI answers:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."     # Windows (PowerShell): $env:ANTHROPIC_API_KEY="sk-ant-..."
```

## Deploy on Streamlit Community Cloud (free)
1. Put all files in the **root** of a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **Create app**.
3. Choose your repo, branch `main`, and main file path `app.py`.
4. *(Optional)* Under **Advanced settings > Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. Click **Deploy**.

Never commit your API key to GitHub. Keep it in Streamlit secrets or an environment variable.

## Project structure
```
app.py             Streamlit chat interface
pdf_processor.py   PDF text extraction and chunking (keeps page numbers)
rag.py             BM25 search, extractive answers, optional Claude answers
test_rag.py        12 unit tests
requirements.txt   streamlit, pypdf
env.example        Which secret to set
```

## Run the tests
```bash
python -m unittest discover -v
```
The tests build small PDFs on the fly and need `reportlab` (`pip install reportlab`).

## Limitations
- Scanned PDFs without a text layer can't be read (no OCR).
- Only the first 300 pages of each PDF are used.
- Search is keyword-based (BM25), so questions work best when they share words with the document.
- Uploaded PDFs are held in memory only while the app is open; nothing is stored.

## Tech stack
Python, Streamlit, pypdf, BM25 (pure Python), Anthropic API (optional).

## License
Add a license of your choice (for example MIT) via **Add file > Create new file > LICENSE** on GitHub.
