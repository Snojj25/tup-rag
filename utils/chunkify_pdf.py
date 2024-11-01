import PyPDF2
from typing import List
import re


class PDFChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

    def clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove special characters but keep periods for sentence splitting
        text = re.sub(r"[^\w\s\.]", "", text)
        return text.strip()

    def create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []

        # First, split into sentences (crude but effective)
        sentences = text.split(".")
        sentences = [s.strip() + "." for s in sentences if s.strip()]

        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence exceeds chunk size, save chunk and start new one
            if len(current_chunk) + len(sentence) > self.chunk_size:
                chunks.append(current_chunk)

                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    # Find last few sentences that fit in overlap
                    overlap_text = ""
                    current_sentences = current_chunk.split(".")
                    for s in reversed(current_sentences):
                        if len(overlap_text) + len(s) < self.chunk_overlap:
                            overlap_text = s.strip() + ". " + overlap_text
                        else:
                            break
                    current_chunk = overlap_text + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Add the last chunk if it contains text
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def process_pdf(self, pdf_path: str) -> List[str]:
        """Process PDF file and return chunks."""
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)

        # Clean text
        text = self.clean_text(text)

        # Create chunks
        chunks = self.create_chunks(text)

        return chunks


# # Usage example
# def main():
#     chunker = PDFChunker(chunk_size=1000, chunk_overlap=200)
#     chunks = chunker.process_pdf("media/ustava.pdf")

#     # Print chunks and their sizes
#     for i, chunk in enumerate(chunks):
#         print(f"Chunk {i+1} size: {len(chunk)} characters")
#         print(f"Preview: {chunk[::]}")
#         print("-" * 80)


# if __name__ == "__main__":
#     main()
