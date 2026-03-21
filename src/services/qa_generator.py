"""
Q/A Pair Generator using Dual-LLM Pipeline
Generates question-answer pairs from document chunks for fine-tuning datasets.
"""

import re
from typing import Dict, Any, List
from langchain_core.documents import Document


# Prompt template for question generation (LLM A)
QUESTION_GENERATION_PROMPT = """You are an expert at generating high-quality questions from document content.

Your task: Generate exactly {num_questions} questions based STRICTLY on the content below.

RULES:
1. Questions MUST be answerable using ONLY the information in the content
2. Do NOT use external knowledge or make assumptions
3. Generate diverse question types:
   - Factual: "What is...?", "How many...?", "When did...?"
   - Analytical: "Why does...?", "How does X relate to Y...?"
   - Contextual: "According to the document, what..."
4. Questions should be clear, specific, and grammatically correct
5. Avoid yes/no questions when possible
6. Each question on a new line, numbered 1-{num_questions}

CONTENT:
{chunk_content}

Generate exactly {num_questions} questions (one per line, numbered):"""


# Prompt template for answer generation (LLM B)
ANSWER_GENERATION_PROMPT = """You are an expert at answering questions based on provided context.

Your task: Answer each question below using ONLY the information in the CONTEXT.

RULES:
1. Base answers STRICTLY on the CONTEXT provided
2. If the context doesn't contain enough information, say "Information not available in the provided context"
3. Keep answers concise (1-3 sentences)
4. Be factual and precise
5. Do not add external knowledge or speculation
6. For each question, provide the answer on a new line, numbered to match the question

CONTEXT:
{chunk_content}

QUESTIONS:
{formatted_questions}

Provide answers (one per line, numbered to match questions):"""


class QAPairGenerator:
    """
    Generates question-answer pairs from document chunks using dual-LLM pipeline.

    Architecture:
    - LLM A (question_llm): Generates questions from chunk content (higher temperature for diversity)
    - LLM B (answer_llm): Generates answers given questions + context (lower temperature for accuracy)

    Attributes:
        question_llm: LLM instance for question generation
        answer_llm: LLM instance for answer generation
        num_questions: Number of questions to generate per chunk
    """

    def __init__(self, config: Dict[str, Any], num_questions: int = 10):
        """
        Initialize QAPairGenerator with dual-LLM configuration.

        Args:
            config: Configuration dictionary with 'question_llm' and 'answer_llm' sub-configs
            num_questions: Number of Q/A pairs to generate per chunk (default: 10)

        Example config:
            {
                "question_llm": {
                    "llm_provider": "groq",
                    "llm_model": "llama-3.1-8b-instant",
                    "temperature": 0.7,
                    "max_tokens": 512,
                    "request_timeout": 60
                },
                "answer_llm": {
                    "llm_provider": "groq",
                    "llm_model": "llama-3.1-8b-instant",
                    "temperature": 0.2,
                    "max_tokens": 384,
                    "request_timeout": 60
                }
            }
        """
        from src.services.llm_services import get_llm

        # Initialize LLM A for question generation (higher temperature)
        self.question_llm = get_llm(config["question_llm"])

        # Initialize LLM B for answer generation (lower temperature)
        self.answer_llm = get_llm(config["answer_llm"])

        self.num_questions = num_questions

    def generate_questions(self, chunk: Document) -> List[str]:
        """
        Generate questions from a document chunk using LLM A.

        Args:
            chunk: LangChain Document object with page_content and metadata

        Returns:
            List of generated questions

        Raises:
            ValueError: If question parsing fails or incorrect number of questions generated
        """
        # Format prompt with chunk content
        prompt = QUESTION_GENERATION_PROMPT.format(
            num_questions=self.num_questions,
            chunk_content=chunk.page_content
        )

        # Generate questions using LLM A
        response = self.question_llm.invoke(prompt)

        # Extract content from response
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Parse numbered questions using regex
        questions = self._parse_numbered_items(response_text)

        # Validate we got the expected number of questions
        if len(questions) != self.num_questions:
            print(f"Warning: Expected {self.num_questions} questions but got {len(questions)}")

        return questions

    def generate_answers(self, chunk: Document, questions: List[str]) -> List[str]:
        """
        Generate answers for questions using LLM B with chunk as context.

        Args:
            chunk: LangChain Document object with page_content and metadata
            questions: List of questions to answer

        Returns:
            List of generated answers (same length as questions)

        Raises:
            ValueError: If answer parsing fails
        """
        # Format questions for the prompt
        formatted_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

        # Format prompt with chunk content and questions
        prompt = ANSWER_GENERATION_PROMPT.format(
            chunk_content=chunk.page_content,
            formatted_questions=formatted_questions
        )

        # Generate answers using LLM B
        response = self.answer_llm.invoke(prompt)

        # Extract content from response
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Parse numbered answers
        answers = self._parse_numbered_items(response_text)

        # Validate we got the expected number of answers
        if len(answers) != len(questions):
            print(f"Warning: Expected {len(questions)} answers but got {len(answers)}")
            # Pad or trim to match
            if len(answers) < len(questions):
                answers.extend(["Information not available in the provided context"] * (len(questions) - len(answers)))
            else:
                answers = answers[:len(questions)]

        return answers

    def generate_qa_pairs(self, chunk: Document) -> List[Dict[str, Any]]:
        """
        Full pipeline: Generate Q/A pairs for a single chunk.

        Args:
            chunk: LangChain Document object with page_content and metadata

        Returns:
            List of dictionaries with question, answer, and metadata:
            [
                {
                    "question": "What is...",
                    "answer": "It is...",
                    "chunk_index": 0,
                    "source": "document.txt",
                    "qa_pair_id": "chunk_0_qa_1"
                },
                ...
            ]
        """
        # Step 1: Generate questions
        questions = self.generate_questions(chunk)

        # Step 2: Generate answers
        answers = self.generate_answers(chunk, questions)

        # Step 3: Combine into Q/A pairs with metadata
        qa_pairs = []
        chunk_index = chunk.metadata.get("chunk_index", 0)
        source = chunk.metadata.get("source", "unknown")

        for i, (question, answer) in enumerate(zip(questions, answers), 1):
            qa_pair = {
                "question": question,
                "answer": answer,
                "chunk_index": chunk_index,
                "source": source.split("\\")[-1],  # Extract filename from path
                "qa_pair_id": f"chunk_{chunk_index}_qa_{i}"
            }
            qa_pairs.append(qa_pair)

        return qa_pairs

    def batch_generate(self, chunks: List[Document], show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Generate Q/A pairs for multiple chunks with optional progress tracking.

        Args:
            chunks: List of LangChain Document objects
            show_progress: If True, display progress bar (requires tqdm)

        Returns:
            List of all Q/A pairs from all chunks
        """
        all_qa_pairs = []
        failed_chunks = []

        # Setup progress bar if requested
        if show_progress:
            try:
                from tqdm import tqdm
                chunk_iterator = tqdm(chunks, desc="Generating Q/A pairs")
            except ImportError:
                print("Warning: tqdm not installed, progress bar disabled")
                chunk_iterator = chunks
        else:
            chunk_iterator = chunks

        # Process each chunk
        for chunk in chunk_iterator:
            try:
                pairs = self.generate_qa_pairs(chunk)
                all_qa_pairs.extend(pairs)
            except Exception as e:
                chunk_idx = chunk.metadata.get("chunk_index", "unknown")
                print(f"\nFailed on chunk {chunk_idx}: {e}")
                failed_chunks.append(chunk_idx)
                continue

        # Report summary
        if failed_chunks:
            print(f"\n⚠ Warning: {len(failed_chunks)} chunks failed: {failed_chunks}")

        return all_qa_pairs

    def _parse_numbered_items(self, text: str) -> List[str]:
        """
        Parse numbered items from LLM response.

        Supports formats:
        - "1. Item text"
        - "1) Item text"
        - "1: Item text"

        Args:
            text: Raw LLM response text

        Returns:
            List of parsed items (without numbering)
        """
        # Pattern to match numbered items: "1. ", "1) ", or "1: "
        pattern = r'^\s*(\d+)[\.\):\-]\s*(.+?)(?=^\s*\d+[\.\):\-]|\Z)'

        # Find all matches (with MULTILINE and DOTALL flags)
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)

        if matches:
            # Extract the text part (index 1) and clean up
            items = [match[1].strip() for match in matches]
            return items

        # Fallback: split by newlines and filter empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Try to remove leading numbers if present
        cleaned_lines = []
        for line in lines:
            # Remove leading number patterns
            cleaned = re.sub(r'^\s*\d+[\.\):\-]\s*', '', line)
            if cleaned:
                cleaned_lines.append(cleaned)

        return cleaned_lines if cleaned_lines else lines
