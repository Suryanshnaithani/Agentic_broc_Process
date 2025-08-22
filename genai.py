import re
import faiss
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from google import genai
from google.genai import types

class MarkdownVectorDB:
    def __init__(self, api_key: str, markdown_path: str, chunk_size: int = 800, overlap: int = 100):
        self.client = genai.Client(api_key=api_key)
        self.markdown_path = markdown_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []
        self.index = None

        # Load and process markdown
        self._load_markdown()
        self._create_index()

    def _load_markdown(self):
        """Load markdown file and split into chunks."""
        with open(self.markdown_path, "r", encoding="utf-8") as f:
            contents = f.read()
        self.chunks = self._chunk_text(contents)

    def _chunk_text(self, text: str):
        """Split text into overlapping chunks."""
        paragraphs = re.split(r"\n\s*\n", text.strip())
        chunks, current_chunk = [], []
        current_length = 0

        for para in paragraphs:
            sentences = sent_tokenize(para)
            for sent in sentences:
                words = sent.split()
                sent_len = len(words)

                if current_length + sent_len > self.chunk_size:
                    chunks.append(" ".join(current_chunk))
                    overlap_words = current_chunk[-self.overlap:] if self.overlap > 0 else []
                    current_chunk = overlap_words + words
                    current_length = len(current_chunk)
                else:
                    current_chunk.extend(words)
                    current_length += sent_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _create_index(self):
        """Embed chunks and build FAISS index."""
        embeddings = []
        for ch in self.chunks:
            res = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=[ch],
                config=types.EmbedContentConfig(
                    output_dimensionality=1536,
                    task_type="RETRIEVAL_DOCUMENT"
                )
            )
            embeddings.append(res.embeddings[0].values)

        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def query(self, question: str, top_k: int = 3) -> str:
        """Query the vectordb and return Gemini’s response."""
        qres = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=[question],
            config=types.EmbedContentConfig(
                output_dimensionality=1536,
                task_type="RETRIEVAL_QUERY"
            )
        )
        qvec = np.array([qres.embeddings[0].values]).astype("float32")

        D, I = self.index.search(qvec, k=top_k)
        context = "\n".join([self.chunks[idx] for idx in I[0]])

        prompt = f"""
        You are an assistant that answers questions about a real estate project.
        Rely **only** on the information provided in the brochure context below.
        If the answer is not explicitly stated in the context, reply: "The brochure does not provide this information."

        Brochure Context:
        {context}

        Question:
        {question}

        Answer:
        """

        resp = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return resp.text



if __name__ == "__main__":
    db = MarkdownVectorDB(
        api_key="AIzaSyBKMTlb3t0Yg6j85ynT3TEsz_ZQhV1zlO4",
        markdown_path=r"Responses/r413082.md"
    )

    print(db.query("List all BHK configs present in the project"))
    print(db.query("What amenities are available?"))
